"""Compare native-free geometric control laws with oracle distance predictions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose_control_reachability as base
import torch


def solve(state, target, e3, scales, ld, t, dt, mode):
    ligand, poi, _ = state
    coords = (e3[0], poi[0], ligand[0])
    desired = (e3[0], target[1][0], target[0][0])
    centers = torch.stack([x.mean(0) for x in coords])
    width = 12 + int(ld['edge_mask'].sum())
    torsion = base.horizontal_torsion_jacobian(ligand[0], ld['edge_index'], ld['edge_mask'], ld['mask_rotate'])
    matrices, errors, weights = [], [], []
    for receiver, anchor in ((2, 0), (2, 1), (1, 0)):
        ids = torch.linspace(0, len(coords[anchor])-1, min(16, len(coords[anchor]))).long()
        left, right = coords[receiver], coords[anchor][ids]
        ja = base.point_jacobian(left, receiver, centers, scales, width)
        if receiver == 2: ja[:, :, 12:] = torsion
        jb = base.point_jacobian(right, anchor, centers, scales, width)
        delta = right[None] - left[:, None]
        distance = (delta.square().sum(-1) + .01).sqrt()/10
        goal = ((desired[anchor][ids][None]-desired[receiver][:,None]).square().sum(-1)+.01).sqrt()/10
        direction = delta/(delta.square().sum(-1,keepdim=True)+.01).sqrt()
        jac = torch.einsum('ijc,ijcd->ijd',direction,jb[None]-ja[:,None])/10
        matrices.append(jac.reshape(-1,width))
        errors.append((distance-goal).flatten())
        weights.append(torch.ones_like(distance).flatten()/distance.numel()/3)
    matrix, error, weight = torch.cat(matrices),torch.cat(errors),torch.cat(weights)
    if mode.startswith('gn'):
        clock = dt
    else:
        clock = t.clamp_min(dt)
    ridge = .01
    if 'anneal' in mode: ridge = .01*float(t)**2+1e-6
    if 'small' in mode: ridge = 1e-4
    if 'relative' in mode:
        ridge = float((matrix.square()*weight[:,None]).sum()/width)*1e-3+1e-8
    vector,_ = base.solve_velocity(matrix,error/clock,weight,ridge=ridge)
    fraction = 1.
    if 'trust' in mode:
        angular = vector[[3,4,5,9,10,11]].reshape(2,3).norm(dim=-1).max()*torch.pi*dt
        torsional = vector[12:].abs().max()*torch.pi*dt if width>12 else vector.new_zeros(())
        fraction = float((.5/torch.maximum(angular,torsional).clamp_min(1e-12)).clamp_max(1))
        vector = vector*fraction
    return vector, {'ridge':ridge,'trust_fraction':fraction,'weighted_error':float((weight*error.square()).sum().sqrt())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='results/model45_root_cause/control_law')
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    saved=torch.load('results/model45_root_cause/fixed_fit/cuda_seed73_state.pt',map_location='cpu',weights_only=False)
    ed,pd,ld=saved['e3_data'],saved['poi_data'],saved['ligand_data']
    _,_,cases=base._load_index('results/model45_partition/ternary_train.json')
    metric=base.validation_metric_context(cases[saved['case']],'/mnt/disk/wl/MoRE-tri/CoopDiff/DeepTernary/data/TernaryDB/pdbs')
    flow=base.SDPIFlow(inference_steps=20)
    scales=flow.scales(ed['coords'],pd['coords']).double()
    modes=['horizon','horizon_anneal','horizon_small','horizon_relative','horizon_anneal_trust','gn_anneal_trust','gn_small_trust','gn_relative_trust']
    report={'runs':[],'modes':modes,'oracle':True,'steps':20,'total_clock':1,'native_free_control_laws':True}
    for seed,device in ((73,'cuda'),(74,'cpu')):
        initial=saved['state'][:3] if device=='cuda' else base.TernaryInitializer(no_torsion=False).initialize(ed['coords'][0],pd['coords'][0],ld,seed=seed)[:3]
        projected,_=base.reachable_ligand_target(ld,initial[0],initial[1],pd['coords'])
        initial=tuple(x.double() for x in initial)
        target=(torch.as_tensor(projected).double().reshape_as(initial[0]),pd['coords'].double())
        for mode in modes:
            state=initial;trace=[]
            with torch.no_grad():
                for t,dt in flow.reverse_times(initial[0]):
                    vector,diagnostics=solve(state,target,ed['coords'].double(),scales,ld,t[0],dt[0],mode)
                    state=base.advance(state,vector,dt,scales,ld)
                    trace.append(diagnostics)
            result={'seed':seed,'initial_device':device,'mode':mode,'poi_rmsd':float((state[1]-target[1]).square().sum(-1).mean().sqrt()),'ligand_rmsd':float((state[0]-target[0]).square().sum(-1).mean().sqrt()),'trace':trace,
                **base.compute_fullatom_dockq(state[0][0].numpy(),state[1][0].numpy(),ed['coords'][0].numpy(),**metric)}
            report['runs'].append(result)
            (out/'summary.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)
    report['status']='complete'
    (out/'summary.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()

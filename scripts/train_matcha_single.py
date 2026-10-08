"""Fixed-init single-case Matcha assembly overfit, with exact protein DockQ.

This is an explicitly isolated research checkpoint format, not a production
training checkpoint. Only the ternary module is optimized; Matcha stays frozen.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tricoopdiff.diffdock_compat
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.initialization.endpoints import reachable_ligand_target
from tricoopdiff.metrics.dockq import compute_fullatom_dockq
from tricoopdiff.models.assembly import TernaryAssembly, TernaryAssemblyModel, production_model_config
from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.train import AssemblyTrainer, clip_grad_norm_finite, NonFiniteTrainingError
from tricoopdiff.runtime import (atomic_torch_save, sha256_file, evaluation_context,
                                capture_rng_state, restore_rng_state)

CASE = '4G8A_B_C_LP5'
FORMAT = 'matcha_fixed_case_research_resume_v2'


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move(item, device) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--weights-from', help='Explicit weights-only transfer; resets optimizer and counters')
    parser.add_argument('--updates', type=int, default=300, help='Total optimizer updates, including resumed updates')
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--lr', type=float, help='Fresh-run default 1e-4; required explicitly for weights-only transfer')
    parser.add_argument('--node-dim', type=int, default=128)
    parser.add_argument('--steric-weight', type=float, default=.05)
    parser.add_argument('--goal-dockq', type=float, default=.90)
    parser.add_argument('--goal-loss', type=float, default=.01)
    parser.add_argument('--source-dir', default='results/matcha_qualification_source')
    parser.add_argument('--weights-dir', default='results/matcha_qualification_weights')
    args = parser.parse_args()
    if args.resume and args.weights_from:
        parser.error('--resume and --weights-from are mutually exclusive')
    if args.weights_from and args.lr is None:
        parser.error('weights-only transfer resets Adam; choose --lr explicitly, do not treat it as resume')
    args.lr = 1e-4 if args.lr is None else args.lr
    if (args.updates < 0 or args.eval_every < 1 or args.save_every < 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not 0 <= args.goal_dockq <= 1 or not math.isfinite(args.goal_loss) or args.goal_loss < 0):
        parser.error('invalid update budget, interval, learning rate, or goal threshold')
    out = Path(args.out).resolve()
    if not args.resume and (out/'config.json').exists():
        parser.error('output already contains a run; use --resume or a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)
    device = 'cuda'
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    transfer = torch.load(args.weights_from, map_location='cpu', weights_only=False) if args.weights_from else None
    if transfer is not None and (transfer.get('case') != CASE or transfer.get('seed') != 73):
        raise ValueError('weights source must belong to the same fixed-case experiment')
    if checkpoint is not None:
        if checkpoint.get('format') != FORMAT or checkpoint['case'] != CASE or checkpoint['seed'] != 73:
            raise ValueError('checkpoint does not match the fixed research experiment')
    case_dir = ROOT / f'data/ternary_deepternary/train/{CASE}'
    p1, p2 = case_dir/'protein1.pdb', case_dir/'protein2.pdb'
    cache_path = ROOT/f'results/featurize_cache/{CASE}.pt'
    input_manifest = {str(path.relative_to(ROOT)): sha256_file(path)
                      for path in (p1, p2, case_dir/'ligand.pdb', cache_path)}
    if checkpoint is not None and checkpoint['input_manifest'] != input_manifest:
        raise ValueError('resume indexed structures or feature cache changed')
    if transfer is not None and transfer.get('input_manifest', input_manifest) != input_manifest:
        raise ValueError('fixed-case transfer indexed structures or feature cache changed')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    ed, pd, ld = [move(copy.deepcopy(cache[key]), device) for key in ('e3_data', 'poi_data', 'ligand_data')]
    for data in (ed, pd, ld):
        for key in ('coords', 'features'):
            if data[key].ndim == 2:
                data[key] = data[key].unsqueeze(0)
    if isinstance(ld['mol'], bytes):
        ld['mol'] = Chem.Mol(ld['mol'])
    fixed = checkpoint if checkpoint is not None else transfer
    if fixed is None:
        ligand, poi, body, provenance = TernaryInitializer(no_torsion=False).initialize(
            ed['coords'][0], pd['coords'][0], ld, seed=73)
        target, projection = reachable_ligand_target(ld, ligand, poi, pd['coords'], no_torsion=False)
        coupling = dict(initial_state=dict(ligand=ligand.detach().cpu(), poi=poi.detach().cpu(),
                                           body_frame=body.detach().cpu()),
                        target_ligand=torch.as_tensor(target).detach().cpu())
    else:
        coupling = fixed['fixed_coupling']
        provenance = fixed['initialization_provenance']
        projection = fixed['target_projection']
    # Native structure is used only by loss / evaluation, not as an expert input.
    native_path = out/'native_proteins.pdb'
    records = []
    for path, chain in ((p1, 'A'), (p2, 'B')):
        records.extend(line[:21]+chain+line[22:] for line in path.read_text().splitlines()
                       if line.startswith('ATOM'))
    native_path.write_text('\n'.join(records)+'\n')
    metric_context = dict(protein1_pdb=str(p1), protein2_pdb=str(p2),
        ligand_pdb=str(case_dir/'ligand.pdb'), native_complex_pdb=str(native_path),
        deepternary_root=str(ROOT/'DeepTernary'))

    from tricoopdiff.models.expert import FrozenMatcha, expert_runtime_identity
    expert = FrozenMatcha(source_dir=str(ROOT/args.source_dir), weights_dir=str(ROOT/args.weights_dir))
    expert = expert.to(device)
    ed, pd, ld = expert.prepare_case(ed, pd, ld, (str(p1), str(p2)))
    config = production_model_config(args.node_dim, int(ed['features'].shape[-1]),
        expert_scalar_dim=expert.node_scalar_dim, expert_vector_dim=expert.node_vector_dim,
        expert_backend='matcha')
    model = TernaryAssemblyModel(expert, TernaryAssembly(**config), expert_input_grad=True,
                                checkpoint_ternary=True).to(device)
    trainer = AssemblyTrainer(model, SDPIFlow(inference_steps=20), lr=args.lr,
        residual_weight=0., interface_loss_weight=1., steric_loss_weight=args.steric_weight,
        training_source_mix='sdpi_only', training_endpoint_protocol='none', device=device)
    params = [p for p in model.trifuse.parameters() if p.requires_grad]
    manifest = expert_runtime_identity(expert)
    runtime = dict(lr=args.lr, steric_weight=args.steric_weight, steps=20,
                   source='sdpi', endpoint_weight=.1, interface_weight=1., near_target_aux=False,
                   ligand_geometry_scale_angstrom=1.,
                   gradient_clip=10., expert_input_grad=True,
                   code_identity={str(path.relative_to(ROOT)): sha256_file(path) for path in (
                       Path(__file__).resolve(), ROOT/'tricoopdiff/models/assembly.py',
                       ROOT/'tricoopdiff/train.py', ROOT/'tricoopdiff/sdpi_flow.py')})
    step = 0
    if transfer is not None:
        saved_config = dict(transfer['model_config'])
        legacy_conditioning = 'conditioning_contract' not in saved_config
        if legacy_conditioning:
            saved_config['conditioning_contract'] = config['conditioning_contract']
        if (saved_config != config or transfer['backend_manifest']['stage_weights_sha256']
                != manifest['stage_weights_sha256']):
            raise ValueError('weights-only transfer requires matching architecture and pretrained stages')
        weights = dict(transfer['trainable_weights'])
        if legacy_conditioning:
            # Explicit, function-preserving migration, not a partial state load.
            weights['condition_geometry.weight'] = torch.zeros_like(model.trifuse.condition_geometry.weight)
        model.trifuse.load_state_dict(weights, strict=True)
        trainer.ema = {name: value.detach().clone() for name, value in model.trifuse.state_dict().items()}
        print(json.dumps(dict(kind='weights_only_transfer', source=args.weights_from,
                              source_step=transfer['step'], optimizer_reset=True,
                              zero_initialized_geometry=legacy_conditioning)), flush=True)
    if checkpoint is not None:
        if checkpoint['model_config'] != config or checkpoint['runtime'] != runtime:
            raise ValueError('resume architecture/training contract mismatch')
        if checkpoint['backend_manifest'] != manifest:
            raise ValueError('resume pretrained backend identity changed')
        model.trifuse.load_state_dict(checkpoint['trainable_weights'], strict=True)
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError('resume update counter must be a nonnegative integer')
        restore_rng_state(checkpoint['rng'])
    batch = dict(e3_data=ed, poi_data=pd, ligand_data=ld, case=CASE,
                 coupling_override=coupling, diagnostics=True, source=0)
    (out/'config.json').write_text(json.dumps(dict(case=CASE, seed=73, model_config=config,
        runtime=runtime, backend_manifest=manifest, input_manifest=input_manifest, checkpoint_format=FORMAT,
        source_note='LP5 phosphorylated lipid pressure case, not PROTAC'), indent=2))

    def save(filename='progress.pt'):
        payload = dict(format=FORMAT, case=CASE, seed=73, step=step,
            model_config=config, runtime=runtime, backend_manifest=manifest, input_manifest=input_manifest,
            weights_source=args.weights_from if checkpoint is None else checkpoint.get('weights_source'),
            trainable_weights=model.trifuse.state_dict(), optimizer=trainer.optimizer.state_dict(),
            rng=capture_rng_state(), fixed_coupling=coupling, initialization_provenance=provenance,
            target_projection=projection)
        atomic_torch_save(payload, out/filename)

    def evaluate():
        with evaluation_context(model), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            result = trainer.compute_loss(**batch)
        metric = compute_fullatom_dockq(result['predicted_ligand'][0].cpu().numpy(),
            result['predicted_poi'][0].cpu().numpy(), ed['coords'][0].cpu().numpy(), **metric_context)
        values = dict(kind='evaluation', step=step, loss=float(result['loss_total']), **metric,
            ligand_target_rmsd_angstrom=float(result['ligand_target_rmsd_angstrom']),
            **{name: float(value) for name, value in result['components'].items()})
        print(json.dumps(values), flush=True)
        with (out/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(values)+'\n')
        torch.save(dict(ligand=result['predicted_ligand'].cpu(), poi=result['predicted_poi'].cpu()),
                   out/'latest_prediction.pt')
        return values

    before = evaluate()
    save()
    reached_goal = before['dockq'] >= args.goal_dockq and before['loss'] <= args.goal_loss
    if reached_goal:
        save('goal.pt')
    start = time.time()
    while step < args.updates and not reached_goal:
        trainer.optimizer.zero_grad(set_to_none=True)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            result = trainer.compute_loss(**batch)
            result['loss_total'].backward()
        try:
            gradient = float(clip_grad_norm_finite(params, 10.))
        except NonFiniteTrainingError:
            save('nonfinite_before_update.pt')
            raise
        trainer.optimizer.step()
        step += 1
        row = dict(kind='train', step=step, loss=float(result['loss_total'].detach()),
                   grad_norm=gradient, elapsed_seconds=time.time()-start)
        print(json.dumps(row), flush=True)
        with (out/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        if step % args.save_every == 0:
            save()
        if step % args.eval_every == 0 or step == args.updates:
            current = evaluate()
            if current['dockq'] >= args.goal_dockq and current['loss'] <= args.goal_loss:
                reached_goal = True
                save('goal.pt')
                break
    save()
    print(json.dumps(dict(kind='finished', completed_updates=step,
                         goal_reached=reached_goal,
                         resumed_from=checkpoint['step'] if checkpoint else None)), flush=True)


if __name__ == '__main__':
    main()

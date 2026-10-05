"""Fixed-start, fixed-target memorization diagnostic; not a holdout benchmark."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch
import train_entry as entry
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.metrics.dockq import compute_fullatom_dockq
from tricoopdiff.train import clip_grad_norm_finite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--objectives', nargs='+', choices=('production_main', 'coordinate_only'),
                        default=['production_main', 'coordinate_only'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out', default='results/model45_root_cause/fixed_fit')
    opts = parser.parse_args()
    out = ROOT / opts.out
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / 'results/model45_single_case/manifest.json').read_text())
    argv = manifest['runs'][0]['command'][3:]
    args = entry.parse_args(argv)
    args.lr = opts.lr
    device = torch.device('cuda')
    entry.seed_training(73, device)
    case = '1AFQ_B_C_0FG'
    item = {x['case']: x for x in entry._load_entries(args.train_data)}[case]
    score_args = entry._load_score_model_args(args.expert_model_dir)
    ed, pd, ld, dim = entry.featurize_complex(case, item['protein1'], item['protein2'], item['ligand'],
        score_args, args.device, cache_dir=args.cache_dir, cache_read_only=True, cache_only=True)
    model, flow, trainer = entry.build_trainer(args, device, input_dim=dim)
    shared = torch.load(ROOT / 'results/model45_single_case/shared_initial_state.pt', map_location=device, weights_only=True)
    model.trifuse.load_state_dict(shared, strict=True)
    initial = TernaryInitializer(no_torsion=model.expert.model.no_torsion).initialize(ed['coords'][0], pd['coords'][0], ld, seed=73)
    coupling = {'initial_state': dict(zip(('ligand', 'poi', 'body_frame'), initial[:3]))}
    torch.save(dict(e3_data=ed, poi_data=pd, ligand_data=ld, state=initial, seed=73,
                    device=str(device), case=case), out / 'cuda_seed73_state.pt')
    batch = dict(case=case, e3_data=ed, poi_data=pd, ligand_data=ld, source=0,
                 fullatom_targets=entry.fullatom_targets(item, ed, pd), coupling_override=coupling, diagnostics=True)
    context = entry.validation_metric_context(item, args.native_root)
    started = time.monotonic()
    controls = []
    def hook(module, inputs, output):
        controls.append({k: float(v.detach().mean()) for k, v in output[0].items()})
    model.trifuse.operator_controller.register_forward_hook(hook)
    protocol = dict(case=case, seed=73, updates=opts.updates, lr=opts.lr,
        weights='shared_initial_state.pt, reset per arm', objectives=opts.objectives,
        endpoint_loss_weight=trainer.endpoint_loss_weight, dropout=False, near_target=False, recovery=False, ema=False,
        fixed_initial_state=True, same_structure_train_validation=True, generalization_claim=False,
        coordinate_objective='normalized ligand and protein coordinate MSE only; no auxiliary target geometry or contact losses',
        target='production reachable ligand projection, deterministic same initial state')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    def emit(record):
        record['elapsed_seconds'] = time.monotonic() - started
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    for arm in opts.objectives:
        model.trifuse.load_state_dict(shared, strict=True)
        model.eval()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=opts.lr, weight_decay=args.weight_decay)
        first_target = None
        for step in range(opts.updates + 1):
            controls.clear()
            optimizer.zero_grad(set_to_none=True)
            result = trainer.compute_loss(**batch)
            coordinate = result['components']['loss_ligand_geometry'] + result['components']['loss_protein_geometry']
            objective = result['loss_total'] if arm == 'production_main' else trainer.endpoint_loss_weight * coordinate
            if first_target is None:
                first_target = result['target_ligand'].detach().clone()
            assert torch.equal(first_target, result['target_ligand']), 'fixed-start reachable target changed'
            record = dict(arm=arm, step=step, objective=float(objective.detach()),
                production_loss=float(result['loss_total'].detach()), coordinate_loss=float(coordinate.detach()),
                components={k: float(v.detach()) for k,v in result['components'].items()},
                controls_mean={k: sum(c[k] for c in controls) / len(controls) for k in controls[0]})
            record.update(compute_fullatom_dockq(result['predicted_ligand'][0].detach().cpu().numpy(),
                result['predicted_poi'][0].detach().cpu().numpy(), ed['coords'][0].detach().cpu().numpy(), **context))
            if step < opts.updates:
                objective.backward()
                grads = {}
                for prefix in ('operator_controller', 'constraint_head', 'condition_'):
                    values = [p.grad.double().square().sum() for n,p in model.trifuse.named_parameters()
                              if prefix in n and p.grad is not None]
                    grads[prefix] = float(torch.stack(values).sum().sqrt()) if values else None
                record['gradient_norms_before_clip'] = grads
                record['gradient_norm_before_clip'] = float(clip_grad_norm_finite(parameters, 1.))
                assert all(p.grad is None for p in model.expert.parameters())
                optimizer.step()
            emit(record)
            if step < opts.updates and (step + 1) % 5 == 0:
                torch.save({'model': model.trifuse.state_dict(), 'optimizer': optimizer.state_dict(),
                            'completed_updates': step + 1}, out / f'{arm}_latest.pt')
            del result, objective, coordinate
        torch.save(model.trifuse.state_dict(), out / f'{arm}_weights.pt')
    (out / 'status.json').write_text(json.dumps({'status':'complete','updates_per_arm':opts.updates}))


if __name__ == '__main__':
    main()

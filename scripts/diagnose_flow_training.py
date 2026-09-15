"""Fit one real training input/target pair before attempting larger flow runs.

Run from the repository root on a compute GPU. This deliberately disables
dropout, retains the production objective/optimizer/clipping, and never saves
a checkpoint intended for evaluation. Use a training index, not a benchmark.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import tricoopdiff.diffdock_compat  # noqa: F401,E402
from train_entry import (TernaryDataLoader, _load_score_model_args,
                         build_trainer, parse_args, seed_training)


def module_group(name):
    if 'lora_' in name:
        return 'lora_sub2' if 'sub2' in name else 'lora_sub1'
    for component in ('film', 'bridge', 'torsion_coordinator'):
        if component in name:
            return component
    return 'trifuse_other'


def grouped_norm(named_values):
    sums = {}
    for name, value in named_values:
        group = module_group(name)
        sums.setdefault(group, 0.0)
        if value is not None:
            sums[group] += value.detach().double().square().sum().item()
    return {name: value ** 0.5 for name, value in sums.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-index', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=200,
                        help='Optimizer steps; 0 records an untrained gradient/scale probe')
    parser.add_argument('--time', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-every', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--expert-model-dir', default='checkpoints/score_model')
    parser.add_argument('--cache-dir', default='results/featurize_cache')
    parser.add_argument('--random-paths', action='store_true',
                        help='Cycle over the supplied small index with fresh times/paths')
    parser.add_argument('--condition-ablation', action='store_true')
    args = parser.parse_args()
    if args.condition_ablation and not args.random_paths:
        parser.error('condition-ablation requires random-paths held-out probes')
    if args.steps < 0:
        parser.error('steps must be nonnegative')
    if not 0 <= args.time <= 1 or args.gradient_every < 1:
        parser.error('time must be in [0,1] and gradient-every positive')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    os.environ['NAN_CLAMP'] = '100'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    device = torch.device('cuda')
    seed_training(args.seed, device)
    config = parse_args([
        '--train_data', args.train_index, '--val_data', args.train_index,
        '--output_dir', str(output), '--lr', str(args.lr),
        '--expert_model_dir', args.expert_model_dir, '--device', 'cuda',
    ])
    loader = TernaryDataLoader(
        args.train_index, _load_score_model_args(args.expert_model_dir),
        shuffle=False, device='cuda', cache_dir=args.cache_dir,
        random_flip_proteins=False,
    )
    if args.case not in loader.entries_by_case:
        raise ValueError('diagnostic case must belong to the supplied training index')
    batch = loader._load_one(loader.entries_by_case[args.case])
    batches = ([loader._load_one(entry) for entry in loader.entries]
               if args.random_paths else [batch])
    seed_training(args.seed, device)
    model, flow, trainer = build_trainer(
        config, device, input_dim=batch['e3_data']['features'].shape[-1])
    trainer.set_phase(1)
    model.eval()  # Autograd remains enabled; remove dropout noise from this gate.
    model._debug_step0_probe = True
    coupling = flow.sample_coupling(
        batch['e3_data']['coords'], batch['poi_data']['coords'],
        batch['ligand_data'], no_torsion=bool(model.expert.model.no_torsion))
    torch.save({'batch': batch, 'coupling': coupling, 'time': args.time},
               output / 'fixed_pair.pt')
    validation = []
    if args.random_paths:
        seed_training(args.seed + 10000, device)
        for sample in batches:
            heldout = flow.sample_coupling(
                sample['e3_data']['coords'], sample['poi_data']['coords'],
                sample['ligand_data'], no_torsion=bool(model.expert.model.no_torsion))
            validation.extend((sample, heldout, t) for t in (.1, .5, .9))
        seed_training(args.seed + 1, device)

    def validate_probes():
        records = []
        model.eval()
        with torch.no_grad():
            for sample, heldout, time in validation:
                result = trainer.compute_loss(
                    **copy.deepcopy(sample), coupling_override=copy.deepcopy(heldout),
                    t_override=torch.tensor([time]), diagnostics=True)
                records.append({'case': sample['case'], 'time': time,
                                'loss': result['loss_total'].item(),
                                'components': {k: v.item() for k, v in result['components'].items()},
                                'rms': result['rms']})
        return records

    initial_validation = validate_probes()
    expert = Path(args.expert_model_dir) / 'best_ema_inference_epoch_model.pt'
    metadata = {
        **vars(args), 'commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_diff': subprocess.check_output(
            ['git', 'diff', '--', 'tricoopdiff/train.py', 'scripts/train_entry.py'],
            text=True),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'expert_sha256': hashlib.sha256(expert.read_bytes()).hexdigest(),
        'optimizer': trainer.optimizer_config, 'dropout_enabled': args.random_paths,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), Path('tricoopdiff/train.py'),
                                       Path('scripts/train_entry.py'))},
        'command': sys.argv,
        'training_index': json.loads(Path(args.train_index).read_text()),
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    parameters = {id(p) for group in trainer.optimizer.param_groups
                  for p in group['params']}
    named = [(n, p) for n, p in model.named_parameters() if id(p) in parameters]
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and id(p) not in parameters]
    if missing:
        raise RuntimeError(f'trainable parameters absent from optimizer: {missing}')
    history = []
    with (output / 'steps.jsonl').open('w') as log:
        for step in range(args.steps + 1):
            if args.random_paths:
                model.train()
                batch = batches[step % len(batches)]
                coupling = flow.sample_coupling(
                    batch['e3_data']['coords'], batch['poi_data']['coords'],
                    batch['ligand_data'], no_torsion=bool(model.expert.model.no_torsion))
                time = float(torch.rand(()).item())
            else:
                time = args.time
            trainer.optimizer.zero_grad(set_to_none=True)
            # Copy defensively: graph preparation must not mutate the frozen pair.
            losses = trainer.compute_loss(
                **copy.deepcopy(batch), t_override=torch.tensor([time]),
                coupling_override=copy.deepcopy(coupling), diagnostics=True)
            row = {'step': step, 'case': batch['case'], 'time': time,
                   'n_torsions': len(coupling['angles']), 'loss': losses['loss_total'].item(),
                   'components': {k: v.item() for k, v in losses['components'].items()},
                   'rms': losses['rms'],
                   'expert_probe': model._last_step0_probe}
            active = [(n, p) for n, p in named if p.requires_grad]
            if step % args.gradient_every == 0:
                contributions = {}
                for key, loss in losses['components'].items():
                    if loss.requires_grad:
                        grads = torch.autograd.grad(
                            loss, [p for _, p in active], retain_graph=True,
                            allow_unused=True)
                        contributions[key] = grouped_norm(
                            (n, g) for (n, _), g in zip(active, grads))
                row['component_gradient_norms_unweighted'] = contributions
            if step < args.steps:
                losses['loss_total'].backward()
                row['gradient_norms'] = grouped_norm((n, p.grad) for n, p in active)
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in active], 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError('non-finite diagnostic gradients')
                row['preclip_norm'] = norm.item()
                row['clip_factor'] = min(1.0, 1.0 / max(norm.item(), 1e-12))
                before = {n: p.detach().clone() for n, p in named}
                trainer.optimizer.step()
                row['update_norms'] = grouped_norm((n, p - before[n]) for n, p in named)
                sizes = grouped_norm(before.items())
                row['relative_update_norms'] = {
                    n: v / max(sizes[n], 1e-12) for n, v in row['update_norms'].items()}
                trainer.global_step += 1
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            history.append(row['loss'])
            print(f"step={step} loss={row['loss']:.6g}", flush=True)
    summary = {'initial_loss': history[0], 'final_loss': history[-1],
               'final_initial_ratio': history[-1] / max(history[0], 1e-12),
               'last_20_mean': float(np.mean(history[-20:])),
               'scope': 'one fixed training pair; not a generalization metric'}
    if args.random_paths:
        summary.pop('final_initial_ratio')  # Different targets cannot form a fit ratio.
        summary.update(scope='small training subset; held-out paths, not held-out structures',
                       first_20_mean=float(np.mean(history[:20])),
                       initial_validation=initial_validation,
                       final_validation=validate_probes())
        trainer.save_checkpoint(str(output / 'diagnostic_model.pt'))
        if args.condition_ablation:
            from flow_condition_audit import audit_conditioning
            seed_training(args.seed + 20000, device)
            summary['condition_ablation'] = audit_conditioning(trainer, validation)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

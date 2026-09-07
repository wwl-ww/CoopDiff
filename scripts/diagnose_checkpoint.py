"""Run checkpoint-specific path replay and controlled optimization diagnostics.

All treatments reload the same checkpoint, including optimizer state. Outputs
are diagnostic artifacts only; this does not resume production training.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from diagnose_flow_training import grouped_norm
from flow_condition_audit import audit_conditioning
from train_entry import (TernaryDataLoader, _load_score_model_args,
                         build_trainer, parse_args, seed_training)


def objective(losses, stage):
    if stage == 'all':
        return losses['loss_total']
    names = ['loss_e3l_tr', 'loss_pol_tr']
    if stage == 'rigid':
        names += ['loss_e3l_rot', 'loss_pol_rot']
    elif stage != 'translation':
        raise ValueError(f'unknown objective: {stage}')
    return sum(losses['components'][name] for name in names)


def measure(trainer, pair, stage='all'):
    batch, coupling, value = pair
    result = trainer.compute_loss(
        **copy.deepcopy(batch), coupling_override=copy.deepcopy(coupling),
        t_override=torch.tensor([value]), diagnostics=True)
    record = {'case': batch['case'], 'time': value,
              'loss': result['loss_total'].item(),
              'objective': objective(result, stage).item(),
              'components': {k: v.item() for k, v in result['components'].items()},
              'rms': result['rms']}
    return result, record


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-index', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reference-checkpoint')
    parser.add_argument('--reference-summary')
    parser.add_argument('--path-seed', type=int, default=10043)
    parser.add_argument('--case', default='5Z86_A_M_PGV')
    parser.add_argument('--time', type=float, default=.1)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--extra-paths', type=int, default=2)
    parser.add_argument('--probe-only', action='store_true',
                        help='Record checkpoint gradient/scale probes without optimizer updates')
    parser.add_argument('--cache-dir', default='results/featurize_cache')
    parser.add_argument('--expert-model-dir', default='checkpoints/score_model')
    args = parser.parse_args()
    if args.steps < 1 or args.extra_paths < 0 or not 0 <= args.time <= 1:
        parser.error('steps positive, extra-paths nonnegative, time in [0,1] required')
    if bool(args.reference_checkpoint) != bool(args.reference_summary):
        parser.error('supply both reference-checkpoint and reference-summary')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    os.environ['NAN_CLAMP'] = '100'
    seed_training(42, device)
    config = parse_args([
        '--train_data', args.train_index, '--val_data', args.train_index,
        '--output_dir', str(out), '--lr', '1e-4',
        '--expert_model_dir', args.expert_model_dir, '--device', 'cuda',
    ])
    loader = TernaryDataLoader(
        args.train_index, _load_score_model_args(args.expert_model_dir),
        shuffle=False, device='cuda', cache_dir=args.cache_dir)
    batches = [loader._load_one(entry) for entry in loader.entries]
    selected = next(b for b in batches if b['case'] == args.case)
    model, flow, trainer = build_trainer(
        config, device, input_dim=selected['e3_data']['features'].shape[-1])
    trainer.set_phase(1)
    model._debug_step0_probe = True
    seed_training(args.path_seed, device)
    validation = []
    for batch in batches:
        coupling = flow.sample_coupling(
            batch['e3_data']['coords'], batch['poi_data']['coords'],
            batch['ligand_data'], no_torsion=bool(model.expert.model.no_torsion))
        validation.extend((batch, coupling, t) for t in (.1, .5, .9))
        if batch['case'] == args.case:
            pair = (batch, coupling, args.time)
    torch.save({'batch': pair[0], 'coupling': pair[1], 'time': pair[2]}, out / 'fixed_pair.pt')
    torch.save(validation, out / 'replay_probes.pt')

    def evaluate(pairs):
        model.eval()
        with torch.no_grad():
            return [measure(trainer, p)[1] for p in pairs]

    report = {'checkpoint': args.checkpoint,
              'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'args': vars(args), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'scope': 'three training structures; independent paths, not independent structures'}
    if args.reference_checkpoint:
        trainer.load_checkpoint(args.reference_checkpoint)
        replay = evaluate(validation)
        expected = json.loads(Path(args.reference_summary).read_text())['final_validation']
        if len(replay) != len(expected):
            raise AssertionError('reference probe count mismatch')
        errors = []
        replay_matches = True
        for actual, old in zip(replay, expected):
            if (actual['case'], actual['time']) != (old['case'], old['time']):
                raise AssertionError('reference probe identity mismatch')
            errors.append(abs(actual['loss'] - old['loss']))
            # CUDA scatter reductions on these large graphs are not bitwise
            # deterministic. Check relative loss tolerance and target scales.
            replay_matches &= bool(np.isclose(actual['loss'], old['loss'], rtol=1e-4, atol=1e-4))
            replay_matches &= all(np.isclose(value, old['rms'][key], rtol=1e-6, atol=1e-6)
                                  for key, value in actual['rms'].items() if '_target_rms' in key)
        report['reference_replay_max_error'] = max(errors)
        report['reference_replay'] = replay
        report['reference_replay_loss_tolerance'] = {'rtol': 1e-4, 'atol': 1e-4}
        if not replay_matches:
            dump(out / 'summary.json', report)
            raise AssertionError(f'failed exact historical path replay: {max(errors)}')
        print(f'Reference replay max error: {max(errors):.6g}', flush=True)

    trainer.load_checkpoint(args.checkpoint)
    report['epoch'] = trainer.resume_epoch
    report['global_step'] = trainer.global_step
    report['checkpoint_validation'] = evaluate(validation)
    print('Checkpoint validation mean:', np.mean([r['loss'] for r in report['checkpoint_validation']]), flush=True)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainer.optimizer.zero_grad(set_to_none=True)
    result, _ = measure(trainer, pair)
    report['component_gradient_norms_unweighted'] = {}
    for name, loss in result['components'].items():
        if loss.requires_grad:
            gradients = torch.autograd.grad(loss, [p for _, p in named],
                                            retain_graph=True, allow_unused=True)
            report['component_gradient_norms_unweighted'][name] = grouped_norm(
                (n, g) for (n, _), g in zip(named, gradients))
    del result
    if args.probe_only:
        dump(out / 'summary.json', report)
        print('Completed read-only checkpoint gradient probe', flush=True)
        return
    expanded = list(validation)
    for path_index in range(args.extra_paths):
        seed_training(args.path_seed + 100 + path_index, device)
        for batch in batches:
            coupling = flow.sample_coupling(
                batch['e3_data']['coords'], batch['poi_data']['coords'],
                batch['ligand_data'], no_torsion=bool(model.expert.model.no_torsion))
            expanded.extend((batch, coupling, t) for t in (.1, .5, .9))
    report['expanded_validation'] = evaluate(expanded)
    seed_training(20043, device)
    report['condition_e3l'] = audit_conditioning(trainer, expanded)
    dump(out / 'summary.json', report)
    print('Checkpoint conditioning:', report['condition_e3l']['correct_mean'],
          report['condition_e3l']['shuffled_mean'], flush=True)

    treatments = [
        ('translation', 'translation', 1., 1., 100.),
        ('rigid', 'rigid', 1., 1., 100.),
        ('all', 'all', 1., 1., 100.),
        ('lr_half', 'all', .5, 1., 100.),
        ('clip_10', 'all', 1., 10., 100.),
        ('clamp_200', 'all', 1., 1., 200.),
    ]
    report['treatments'] = {}
    for name, stage, lr_factor, clip, clamp in treatments:
        # Restore weights, optimizer moments, learning rates, and warmup position.
        os.environ['NAN_CLAMP'] = '100'
        trainer.load_checkpoint(args.checkpoint)
        seed_training(42, device)
        model.eval()
        os.environ['NAN_CLAMP'] = str(clamp)
        for group in trainer.optimizer.param_groups:
            group['lr'] *= lr_factor
        all_named = dict(model.named_parameters())
        named = [(n, p) for n, p in all_named.items() if p.requires_grad]
        optimizer_ids = {id(p) for g in trainer.optimizer.param_groups for p in g['params']}
        if any(id(p) not in optimizer_ids for _, p in named):
            raise AssertionError('trainable parameter missing from optimizer')
        records = []
        with (out / f'{name}.jsonl').open('w') as log:
            for step in range(args.steps + 1):
                trainer.optimizer.zero_grad(set_to_none=True)
                result, row = measure(trainer, pair, stage)
                row['step'] = step
                row['global_step'] = trainer.global_step
                row['expert_probe'] = model._last_step0_probe
                if step < args.steps:
                    objective(result, stage).backward()
                    row['gradients'] = grouped_norm((n, p.grad) for n, p in named)
                    norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], clip)
                    if not torch.isfinite(norm):
                        raise FloatingPointError(f'{name}: non-finite gradients')
                    row['preclip_norm'] = norm.item()
                    row['clip_factor'] = min(1., clip / max(norm.item(), 1e-12))
                    before = {n: p.detach().clone() for n, p in named}
                    trainer.optimizer.step()
                    row['updates'] = grouped_norm((n, p - before[n]) for n, p in named)
                    trainer.global_step += 1
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                records.append(row)
                if step % 10 == 0 or step == args.steps:
                    print(f'{name} step={step} total={row["loss"]:.6g} objective={row["objective"]:.6g}', flush=True)
        report['treatments'][name] = {
            'stage': stage, 'lr_factor': lr_factor, 'clip': clip, 'clamp': clamp,
            'initial': records[0]['objective'], 'final': records[-1]['objective'],
            'initial_total': records[0]['loss'], 'final_total': records[-1]['loss'],
            'validation_after': evaluate(validation),
        }
        dump(out / 'summary.json', report)
    os.environ['NAN_CLAMP'] = '100'
    print(f'Completed diagnostics: {out}', flush=True)


if __name__ == '__main__':
    main()

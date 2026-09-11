#!/usr/bin/env python
"""Test whether one model can jointly memorize several off-path translations."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import tricoopdiff.diffdock_compat
from local_fixed_translation_overfit import build_trainer, fixed_state, measure
from local_paired_pilot import load_batches, sha256
from train_entry import _load_score_model_args
from tricoopdiff.checkpoint_contract import validate_expert_base_identity
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.reproducibility import configure_inference_reproducibility


def summarize(trainer, examples):
    """Measure every fixed state without updating parameters."""
    rows = []
    with torch.no_grad():
        for example in examples:
            _, metrics = measure(
                trainer, example['batch'], example['state'], example['time'])
            rows.append({
                'case': example['case'], 'time': float(example['time'].item()),
                **metrics,
            })
    return {
        'mean_loss': float(np.mean([row['loss'] for row in rows])),
        'minimum_cosine': min(
            row[branch]['cosine'] for row in rows
            for branch in ('e3_ligand', 'poi_ligand')),
        'rows': rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts', type=Path, required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cases', nargs='+', default=(
        '3CXH_C_E_6PH', '4I0P_D_H_PGE', '4IL6_C_I_LMT'))
    ap.add_argument('--times', nargs='+', type=float, default=(.8, .5, .3))
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--seed', type=int, default=27101)
    ap.add_argument('--holdout-seed-offset', type=int, default=10000)
    ap.add_argument('--optimizer-mode', choices=('sequential', 'mean_gradient'),
                    default='sequential')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.epochs < 1 or any(not .3 <= value <= 1 for value in args.times):
        raise ValueError('epochs must be positive and times must lie in [0.3, 1]')
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    policy = configure_inference_reproducibility(True)
    device = 'cuda'
    score_args = _load_score_model_args(str(args.artifacts / 'checkpoints/score_model'))
    batches = load_batches(args.artifacts, 'val', 8, device, score_args)
    by_case = {batch['case']: batch for batch in batches}
    missing = set(args.cases) - set(by_case)
    if missing:
        raise ValueError(f'cases not in validation index: {sorted(missing)}')
    base = torch.load(args.base, map_location='cpu', weights_only=False)
    expert = build_diffdock_expert(str(args.artifacts / 'checkpoints/score_model'), device)
    validate_expert_base_identity(base, expert)
    trainer = build_trainer(base, expert, device)
    def make_examples(seed_offset):
        examples = []
        for case_index, case in enumerate(args.cases):
            batch = copy.deepcopy(by_case[case])
            for time_value in args.times:
                run_seed = (args.seed + seed_offset + case_index * 100
                            + round(time_value * 10))
                np.random.seed(run_seed)
                torch.manual_seed(run_seed)
                torch.cuda.manual_seed_all(run_seed)
                coupling = trainer.diffusion.sample_coupling(
                    batch['e3_data']['coords'], batch['poi_data']['coords'],
                    batch['ligand_data'])
                time = torch.tensor([time_value], device=device)
                examples.append({
                    'case': case, 'time': time, 'batch': batch,
                    'state': fixed_state(trainer, batch, coupling, time),
                })
        return examples

    examples = make_examples(0)
    holdout_examples = make_examples(args.holdout_seed_offset)
    rng = np.random.default_rng(args.seed)
    report = {
        'status': 'running',
        'contract': 'fixed_translation_joint_overfit_generalization_v2',
        'base': str(args.base), 'base_sha256': sha256(args.base),
        'seed': args.seed, 'epochs': args.epochs,
        'cases': args.cases, 'times': args.times,
        'holdout_seed_offset': args.holdout_seed_offset,
        'optimizer_mode': args.optimizer_mode,
        'reproducibility': policy, 'trace': [],
    }
    milestones = {0, 1, 2, 5, 10, 25, 50, 100, 200, args.epochs}
    for epoch in range(args.epochs + 1):
        if epoch in milestones:
            summary = {
                'epoch': epoch,
                'train': summarize(trainer, examples),
                'holdout': summarize(trainer, holdout_examples),
            }
            report['trace'].append(summary)
            args.out.joinpath('report.json').write_text(
                json.dumps(report, indent=2) + '\n')
            print(json.dumps({
                'epoch': epoch,
                'train_mean_loss': summary['train']['mean_loss'],
                'train_minimum_cosine': summary['train']['minimum_cosine'],
                'holdout_mean_loss': summary['holdout']['mean_loss'],
                'holdout_minimum_cosine': summary['holdout']['minimum_cosine'],
            }), flush=True)
        if epoch == args.epochs:
            break
        order = rng.permutation(len(examples))
        if args.optimizer_mode == 'mean_gradient':
            trainer.optimizer.zero_grad(set_to_none=True)
        for index in order:
            example = examples[int(index)]
            if args.optimizer_mode == 'sequential':
                trainer.optimizer.zero_grad(set_to_none=True)
            loss, _ = measure(
                trainer, example['batch'], example['state'], example['time'])
            divisor = len(examples) if args.optimizer_mode == 'mean_gradient' else 1
            (loss / divisor).backward()
            if args.optimizer_mode == 'sequential':
                parameters = [p for group in trainer.optimizer.param_groups
                              for p in group['params'] if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError('non-finite joint fixed-state gradient')
                trainer.optimizer.step()
        if args.optimizer_mode == 'mean_gradient':
            parameters = [p for group in trainer.optimizer.param_groups
                          for p in group['params'] if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError('non-finite mean fixed-state gradient')
            trainer.optimizer.step()
    report['status'] = 'complete'
    args.out.joinpath('report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

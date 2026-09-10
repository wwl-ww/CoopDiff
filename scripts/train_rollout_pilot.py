#!/usr/bin/env python
"""Fine-tune the best factorial arm with short rollout augmentation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
import tricoopdiff.diffdock_compat  # noqa: F401
from factorial_arm import PairedTorsionFlow, restrict_non_single
from train_entry import TernaryDataLoader, _load_score_model_args
from tricoopdiff.consensus import ConsensusFuse
from tricoopdiff.consensus_experiment import (
    EndpointTranslationRolloutTrainer, ExperimentalConsensus,
    RolloutAugmentedTrainer,
)
from tricoopdiff.expert_factory import build_diffdock_expert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-cases', type=int, default=32)
    parser.add_argument('--val-cases', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=11021)
    parser.add_argument('--rollout-probability', type=float, default=0.5)
    parser.add_argument('--rollout-steps', type=int, default=1)
    parser.add_argument(
        '--offpath-target', choices=('coupling', 'e3_endpoint_translation'),
        default='coupling')
    parser.add_argument('--rollout-min-time', type=float, default=0.25)
    parser.add_argument('--endpoint-blend', type=float, default=0.25)
    parser.add_argument('--endpoint-correction-ratio', type=float, default=1.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    expert_dir = '/tmp/coopdiff/checkpoints/score_model'
    index_dir = Path('/tmp/coopdiff/results/overnight_root_cause_v3_plan/indexes')
    score_args = _load_score_model_args(expert_dir)
    train = TernaryDataLoader(
        index_dir / 'ternary_train_ccd.json', score_args, shuffle=True,
        device='cuda', cache_dir='/tmp/coopdiff/results/overnight_root_cause_v3_plan/cache_ccd')
    val = TernaryDataLoader(
        index_dir / 'ternary_val_ccd.json', score_args, shuffle=False,
        device='cuda', cache_dir='/tmp/coopdiff/results/overnight_root_cause_v3_plan/cache_ccd')
    if train.clusters:
        train.clusters = train.clusters[:args.train_cases]
    else:
        train.entries = train.entries[:args.train_cases]
    if val.clusters:
        val.clusters = val.clusters[:args.val_cases]
    else:
        val.entries = val.entries[:args.val_cases]
    ccd = json.loads((index_dir / 'ternary_val_ccd.json').read_text())
    ccd_paths = {entry['case']: entry['ligand'] for entry in ccd['entries']}
    ccd_train = json.loads((index_dir / 'ternary_train_ccd.json').read_text())
    ccd_paths.update({entry['case']: entry['ligand'] for entry in ccd_train['entries']})

    checkpoint = torch.load(args.base_checkpoint, map_location='cuda', weights_only=False)
    expert = build_diffdock_expert(expert_dir, 'cuda')
    fuse = ConsensusFuse(**checkpoint['model_config'])
    fuse.load_state_dict(checkpoint['ema_trifuse_state_dict'])
    model = ExperimentalConsensus(expert, fuse, checkpoint['expert_lora_config'])
    contract = {
        'chemistry': 'ccd', 'torsion': 'single_bond', 'adaptation': 'frozen',
        'seed': args.seed, 'training': 'rollout_augmentation',
        'rollout_probability': args.rollout_probability,
        'rollout_steps': args.rollout_steps,
        'offpath_target': args.offpath_target,
        'rollout_min_time': args.rollout_min_time,
        'endpoint_blend': args.endpoint_blend,
        'endpoint_correction_ratio': args.endpoint_correction_ratio,
        'base_checkpoint': str(args.base_checkpoint.resolve()),
    }
    trainer_class = (EndpointTranslationRolloutTrainer
                     if args.offpath_target == 'e3_endpoint_translation'
                     else RolloutAugmentedTrainer)
    endpoint_args = ({
        'rollout_min_time': args.rollout_min_time,
        'endpoint_blend': args.endpoint_blend,
        'endpoint_correction_ratio': args.endpoint_correction_ratio,
    } if trainer_class is EndpointTranslationRolloutTrainer else {})
    trainer = trainer_class(
        model, PairedTorsionFlow(), device='cuda', experiment_contract=contract,
        lr=5e-5, lora_lr=1e-5, lora_early_lr=5e-6,
        lora_warmup_steps=0, activation_clamp_mode='irrep_norm',
        rollout_probability=args.rollout_probability,
        rollout_steps=args.rollout_steps, **endpoint_args)

    val_batches = [restrict_non_single(batch, ccd_paths[batch['case']]) for batch in val]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    history = []
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in train:
            batch = restrict_non_single(batch, ccd_paths[batch['case']])
            losses.append(trainer.train_step(**batch)['loss_total'])
        values = []
        with trainer.ema_parameters(), torch.no_grad():
            model.eval()
            old_probability = trainer.rollout_probability
            trainer.rollout_probability = 0.0
            try:
                for batch in val_batches:
                    values.append(float(trainer.compute_loss(**batch)['loss_total']))
            finally:
                trainer.rollout_probability = old_probability
        row = {
            'epoch': epoch, 'train_loss': float(np.mean(losses)),
            'teacher_val_loss': float(np.mean(values)),
            'rollout_augmented_states': trainer.rollout_augmented_states,
        }
        history.append(row)
        (args.out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        trainer.save_checkpoint(args.out / 'latest.pt', epoch, best)
        if row['teacher_val_loss'] < best:
            best = row['teacher_val_loss']
            trainer.save_checkpoint(args.out / 'best.pt', epoch, best)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()

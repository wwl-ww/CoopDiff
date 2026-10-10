#!/usr/bin/env python
"""Small model-v3 loss diagnostics on a fixed general validation subset."""

import argparse
import json
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from train_entry import (
    TernaryDataLoader,
    _load_score_model_args,
    build_trainer,
    seed_training,
)
from tricoopdiff.experiments.trajectory_experiment import TrajectoryConsistencyTrainer


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('local', 'trajectory'), required=True)
    p.add_argument('--train-data', required=True)
    p.add_argument('--val-data', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--expert-model-dir', default='checkpoints/score_model')
    p.add_argument('--cache-dir', default='results/featurize_cache')
    p.add_argument('--max-train-clusters', type=int, default=64)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--seed', type=int, default=90124)
    p.add_argument('--device', default='cuda')
    p.add_argument('--trajectory-steps', type=int, default=3)
    p.add_argument('--trajectory-weight', type=float, default=0.1)
    p.add_argument('--trajectory-min-time', type=float, default=0.25)
    return p.parse_args()


def main():
    args = args_parser()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seed_training(args.seed, device)
    # TernaryDataLoader isolates featurization RNG with torch.random.fork_rng;
    # initialize CUDA first so the selected device has a registered generator
    # even when the process starts from a cold cache.
    if device.type == 'cuda':
        torch.cuda.init()
    score_args = _load_score_model_args(args.expert_model_dir)

    train_loader = TernaryDataLoader(
        args.train_data, score_args, shuffle=True, device=str(device),
        cache_dir=args.cache_dir, max_clusters=args.max_train_clusters,
        random_flip_proteins=False,
    )
    val_loader = TernaryDataLoader(
        args.val_data, score_args, shuffle=False, device=str(device),
        cache_dir=args.cache_dir, random_flip_proteins=False,
    )
    first = next(iter(train_loader))
    input_dim = int(first['e3_data']['features'].shape[-1])
    model_args = argparse.Namespace(
        expert_model_dir=args.expert_model_dir, node_dim=128, ns=32, nv=4,
        lr=1e-3, weight_decay=1e-4,
        nan_clamp=100., activation_clamp_mode='irrep_norm',
        torsion_expert_aux_weight=.25, residual_weight=1e-3, ema_decay=.999,
        training_arm='A1', training_endpoint_protocol='none', phase=1,
    )
    model, diffusion, production_trainer = build_trainer(
        model_args, device, input_dim=input_dim)
    if args.mode == 'local':
        trainer = production_trainer
    else:
        trainer = TrajectoryConsistencyTrainer(
            model=model, diffusion=diffusion, device=device, lr=1e-3,
            weight_decay=1e-4, activation_clamp=100.,
            torsion_expert_aux_weight=0.0, residual_weight=1e-3,
            ema_decay=.999, trajectory_steps=args.trajectory_steps,
            trajectory_weight=args.trajectory_weight,
            trajectory_min_time=args.trajectory_min_time,
            experiment_contract={
                'diagnostic_mode': 'trajectory',
                'validation_index': str(Path(args.val_data).resolve()),
            },
        )
    trainer.set_phase(1)
    config = vars(args) | {'input_dim': input_dim, 'generative_process': 'sdpi_flow_v1'}
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_values = []
        start = time.time()
        for batch in train_loader:
            result = trainer.train_step(**batch, source=0)
            train_values.append(float(result['loss_total']))
        val_values = []
        val_components = []
        for batch in val_loader:
            result = trainer.validate_step(**batch, source=0)
            val_values.append(float(result['loss_total']))
            val_components.append(result)
        val_loss = float(np.mean(val_values))
        trainer.scheduler_step(val_loss)
        record = {
            'epoch': epoch, 'train_loss': float(np.mean(train_values)),
            'val_loss': val_loss,
            'val_cases': len(val_values), 'seconds': time.time() - start,
            'val_loss_trajectory': float(np.mean([
                x.get('loss_trajectory', 0.) for x in val_components]))
            if args.mode == 'trajectory' else 0.,
        }
        with (output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if val_loss < best:
            best = val_loss
            trainer.save_checkpoint(output / 'best_model.pt', epoch=epoch,
                                    best_val_loss=best)
    trainer.save_checkpoint(output / 'final_model.pt', epoch=args.epochs,
                            best_val_loss=best)


if __name__ == '__main__':
    main()

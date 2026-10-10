#!/usr/bin/env python
"""Short production-trainer benchmark, with no DockQ or validation rollouts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import tricoopdiff.diffdock_compat
from train_entry import (parse_args, build_trainer, TernaryDataLoader,
                         _load_score_model_args, seed_training,
                         iter_training_batches, _guarded_train_batch)


def gpu_load():
    """Snapshot whole-card occupancy, including other users' processes."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return result.stdout.strip().splitlines() if result.returncode == 0 else result.stderr.strip()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-mode', choices=['on', 'off'], required=True)
    parser.add_argument('--warmup-updates', type=int, default=1)
    parser.add_argument('--timed-updates', type=int, default=2)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--memory-fraction', type=float, default=.25)
    opt = parser.parse_args()
    if opt.warmup_updates < 0 or opt.timed_updates < 1 or not 0 < opt.memory_fraction <= 1:
        parser.error('invalid update counts or memory fraction')
    os.chdir(ROOT)
    output = Path(opt.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise RuntimeError('benchmark output already exists; use a fresh directory')
    args = parse_args(['--train_data', 'results/small_sample_seed73/train.json',
        '--val_data', 'results/small_sample_seed73/same_complex_val.json',
        '--output_dir', str(output), '--expert-backend', 'matcha', '--training-arm', 'A1',
        '--seed', '73', '--batch_size', '4', '--lr', '.0001', '--no-random-flip-proteins',
        '--cache_dir', 'results/featurize_cache',
        *(['--no-checkpoint-ternary'] if opt.checkpoint_mode == 'off' else [])])
    args.training_endpoint_protocol = 'none'
    args.generative_process = 'sdpi_flow_v1'
    device = torch.device('cuda', torch.cuda.current_device())
    torch.cuda.set_per_process_memory_fraction(opt.memory_fraction, device)
    score_args = _load_score_model_args(args.expert_model_dir)
    seed_training(73, device)
    probe = TernaryDataLoader(args.train_data, score_args, shuffle=False,
                             device=str(device), cache_dir=args.cache_dir)
    first = next(iter(probe))
    input_dim = int(first['e3_data']['features'].shape[-1])
    del first, probe
    seed_training(73, device)
    model, _, trainer = build_trainer(args, device, input_dim=input_dim)
    model.to(device)
    trainer.set_phase(args.phase)
    loader = TernaryDataLoader(args.train_data, score_args, shuffle=True,
        device=str(device), cache_dir=args.cache_dir, expert=model.expert,
        random_flip_proteins=False, cached_only=args.cached_only)
    if len(loader.entries) != 8:
        raise RuntimeError('benchmark requires all eight production cases')
    manifest = {'options': vars(opt), 'model_config': args.resolved_model_config,
        'arguments': vars(args), 'initial_trainable_sha256': parameter_hash(model),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_name': torch.cuda.get_device_name(device), 'gpu_before': gpu_load(),
        'torch_version': torch.__version__, 'expert_provenance': model.expert.provenance,
        'cases': [entry['case'] for entry in loader.entries]}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    total = opt.warmup_updates + opt.timed_updates
    rows = []
    while trainer.global_step < total:
        for batches in iter_training_batches(loader, args.batch_size):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            before = gpu_load()
            started = time.perf_counter()
            metrics = _guarded_train_batch(trainer, [{**batch, 'source': 0} for batch in batches],
                failure_path=output/'projection_failure.pt')
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            row = {'global_step': trainer.global_step,
                'warmup': trainer.global_step <= opt.warmup_updates,
                'elapsed_seconds': elapsed, 'samples_per_second': len(batches)/elapsed,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                'cases': [batch['case'] for batch in batches], 'metrics': metrics,
                'gpu_before': before, 'gpu_after': gpu_load(),
                'trainable_sha256': parameter_hash(model)}
            rows.append(row)
            with (output/'updates.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            print(json.dumps({key: value for key, value in row.items() if key != 'metrics'}), flush=True)
            if trainer.global_step >= total:
                break
    timed = [row for row in rows if not row['warmup']]
    summary = {'status': 'complete', 'checkpoint_mode': opt.checkpoint_mode,
        'warmup_updates': opt.warmup_updates, 'timed_updates': len(timed),
        'seconds_per_update': sum(row['elapsed_seconds'] for row in timed)/len(timed),
        'samples_per_second': 4*len(timed)/sum(row['elapsed_seconds'] for row in timed),
        'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in timed),
        'peak_reserved_bytes': max(row['peak_reserved_bytes'] for row in timed),
        'final_trainable_sha256': parameter_hash(model), 'gpu_after': gpu_load()}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Synchronous multi-GPU A1 training; default total target is one update.

Launch with torchrun --standalone --nproc-per-node=2 scripts/train_distributed.py
plus production train_entry arguments. --batch_size is the GLOBAL batch size.
This is explicit gradient all-reduction, not a DDP-wrapped forward. Resume
requires the same world size, global batch, dataset and sampling policy.
"""
import argparse
import fcntl
import hashlib
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist

import tricoopdiff.diffdock_compat
from tricoopdiff.distributed import (GradientSynchronizer, shard_global_batch,
                                    validate_distributed_checkpoint)
from tricoopdiff.runtime import atomic_json, atomic_torch_save, capture_rng_state, preserve_rng, restore_rng_state, sha256_file
from train_entry import (parse_args, build_trainer, TernaryDataLoader,
                         _load_score_model_args, seed_training)
from featurize_common import featurize_complex


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-fraction', type=float, default=.2)
    parser.add_argument('--max-updates', type=int, default=1, help='Total optimizer update target, including resumed updates')
    parser.add_argument('--checkpoint-every', type=int, default=1)
    parser.add_argument('--collective-timeout-seconds', type=int, default=900)
    opt, extra = parser.parse_known_args()
    args = parse_args(extra)
    if not 0 < opt.memory_fraction <= 1:
        parser.error('memory fraction must be in (0,1]')
    if opt.max_updates < 1 or opt.collective_timeout_seconds < 1 or opt.checkpoint_every < 1:
        parser.error('update budget and collective timeout must be positive')
    if args.training_arm != 'A1':
        parser.error('initial distributed support is A1 SDPI only')
    if args.cached_only or args.max_train_clusters:
        parser.error('use an explicit preprocessed index without cached-only filtering or cluster truncation')
    return opt, args


@preserve_rng
def epoch_entries(loader, epoch, seed, device):
    # Identical schedule on every rank, independent from rank-specific rollout
    # streams. Cluster representative selection keeps the production policy.
    seed_training(seed + epoch, device)
    return loader._epoch_entries()


def save_state(trainer, args, rank, world_size, output, cursor):
    digest = hashlib.sha256()
    for name, tensor in trainer.model.trifuse.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digests = [None] * world_size
    dist.all_gather_object(digests, digest.hexdigest())
    if len(set(digests)) != 1:
        raise RuntimeError('trained parameters differ across ranks')
    states = [None] * world_size if rank == 0 else None
    dist.gather_object(capture_rng_state(), states, dst=0)
    if rank == 0:
        payload = trainer.checkpoint_payload(epoch=cursor['epoch'])
        # Prevent generic single-GPU resume from consuming only rank 0's RNG.
        payload['checkpoint_kind'] = 'distributed_training'
        payload['distributed_state'] = {
            'contract': 'explicit_sample_weighted_allreduce_v1',
            'world_size': world_size, 'global_batch_size': args.batch_size,
            'rank_rng_states': states, 'next_data_cursor': cursor,
            'rank_parameter_sha256': digests,
            'resume_supported': True,
            'resume_contract': 'rank_rng_cursor_dataset_v1',
            'train_index_sha256': sha256_file(args.train_data),
            'seed': args.seed, 'phase': args.phase,
            'random_flip_proteins': args.random_flip_proteins,
            'epoch_size': args.distributed_epoch_size,
            'checkpoint_ternary': args.checkpoint_ternary,
        }
        atomic_torch_save(payload, output / f'checkpoint_update{trainer.global_step}.pt')
    dist.barrier()


def main():
    opt, args = options()
    if 'LOCAL_RANK' not in os.environ:
        raise SystemExit('launch with torchrun; direct Python invocation is unsupported')
    rank, local_rank, world_size = (int(os.environ[name]) for name in
                                  ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if not torch.cuda.is_available():
        raise SystemExit('production distributed entry requires CUDA; helper tests use CPU/Gloo')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.cuda.set_per_process_memory_fraction(opt.memory_fraction, device)
    args.device = str(device)
    if args.batch_size % world_size or args.batch_size < world_size:
        raise SystemExit('--batch_size is GLOBAL and must be divisible by WORLD_SIZE')
    os.chdir(ROOT)
    dist.init_process_group('nccl', device_id=device,
                            timeout=timedelta(seconds=opt.collective_timeout_seconds))
    try:
        output = Path(args.output_dir)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            writer_lock = (output / 'writer.lock').open('a')
            fcntl.flock(writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (output / 'distributed_manifest.json').exists():
                if not args.resume or Path(args.resume).resolve().parent != output.resolve():
                    raise RuntimeError('existing output can only resume its own checkpoint')
                metrics_path = output / 'distributed_metrics.jsonl'
                if metrics_path.exists():
                    last = json.loads(metrics_path.read_text().splitlines()[-1])
                    resumed_step = torch.load(args.resume, map_location='cpu', weights_only=False)['global_step']
                    if last['update'] > resumed_step:
                        raise RuntimeError('resume checkpoint precedes existing metrics; use a fresh output directory')
        dist.barrier()
        args.training_endpoint_protocol = 'none'
        args.generative_process = 'sdpi_flow_v1'
        score_args = _load_score_model_args(args.expert_model_dir)
        loader = TernaryDataLoader(args.train_data, score_args, shuffle=True,
            device=str(device), cache_dir=args.cache_dir,
            random_flip_proteins=args.random_flip_proteins)
        if not args.cache_dir:
            raise ValueError('--cache_dir with preprocessed features is required')
        if not len(loader) or len(loader) % args.batch_size:
            raise ValueError('epoch case count must be a positive multiple of GLOBAL batch size; no dropping/padding')
        # Require existing caches: parallel cache creation is not the purpose of
        # this entry, and all ranks must see the same complete population.
        missing = [entry['case'] for entry in loader.entries
                   if not (Path(args.cache_dir) / (entry['case'] + '.pt')).is_file()]
        if missing:
            raise FileNotFoundError(f'preprocess features before torchrun; missing {missing[:8]}')
        args.distributed_epoch_size = len(loader)
        resumed_payload, resumed_state = None, None
        if args.resume:
            resumed_payload = torch.load(args.resume, map_location='cpu', weights_only=False)
            resumed_state = validate_distributed_checkpoint(resumed_payload, world_size, args.batch_size,
                train_index_sha256=sha256_file(args.train_data), seed=args.seed, phase=args.phase,
                random_flip_proteins=args.random_flip_proteins, epoch_size=len(loader),
                checkpoint_ternary=args.checkpoint_ternary)
            if resumed_state.get('resume_contract') is None:
                # The original one-update smoke saved every optimizer/RNG field,
                # but only its adjacent manifest pins the sampling policy.
                old_manifest = json.loads((Path(args.resume).parent / 'distributed_manifest.json').read_text())
                old_args = old_manifest['config']
                if Path(old_args['train_data']).resolve() != Path(args.train_data).resolve():
                    raise ValueError('legacy distributed resume requires the same index path')
                for key in ('seed', 'phase', 'random_flip_proteins', 'checkpoint_ternary'):
                    if old_args.get(key) != getattr(args, key):
                        raise ValueError(f'legacy distributed checkpoint {key} differs')
            preserve_rng(restore_rng_state)(resumed_state['rank_rng_states'][rank])
            if opt.max_updates < resumed_payload['global_step']:
                raise ValueError('--max-updates is a total target and cannot precede the saved update')
        first = loader.entries[0]
        _, _, _, input_dim = featurize_complex(first['case'], first['protein1'],
            first['protein2'], first['ligand'], score_args, str(device), cache_dir=args.cache_dir)
        seed_training(args.seed, device)
        model, _, trainer = build_trainer(args, device, input_dim=input_dim)
        trainer.set_phase(args.phase)
        loader.expert = model.expert
        # Broadcast parameters AND buffers; do not rely solely on identical
        # constructor seeds. Frozen expert replicas are also checked by this.
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=0)
        if args.resume:
            trainer.load_checkpoint(args.resume, distributed_context={
                'world_size': world_size, 'global_batch_size': args.batch_size})
        else:
            for name, tensor in trainer.ema.items():
                tensor.copy_(model.trifuse.state_dict()[name])
        sync = GradientSynchronizer(device)
        trainer.gradient_sync = sync
        trainer.distributed_error_check = sync.check_error
        local_batch_size = args.batch_size // world_size
        trainer.distributed_sample_offset = rank * local_batch_size
        if args.resume:
            restore_rng_state(resumed_state['rank_rng_states'][rank])
        else:
            seed_training(args.seed + 1000003 * rank, device)
        if rank == 0:
            atomic_json({'synchronization': 'explicit_sample_weighted_allreduce_v1',
                'world_size': world_size, 'global_batch_size': args.batch_size,
                'local_batch_size': local_batch_size, 'max_updates': opt.max_updates,
                'resume_supported': True,
                'resume_from': args.resume, 'initial_global_step': trainer.global_step,
                'legacy_dataset_identity': ('path_only' if resumed_state is not None and
                    resumed_state.get('resume_contract') is None else None), 'validation': 'not run by this training-only entry',
                'config': vars(args)}, output / 'distributed_manifest.json')
        epoch, offset = ((resumed_state['next_data_cursor']['epoch'],
                          resumed_state['next_data_cursor']['offset']) if resumed_state else (0, 0))
        schedule = epoch_entries(loader, epoch, args.seed, device)
        for _ in range(trainer.global_step, opt.max_updates):
            global_entries = schedule[offset:offset + args.batch_size]
            local_entries = shard_global_batch(global_entries, args.batch_size, rank, world_size)
            # Loading failures must be coordinated before any rank starts a
            # backward collective. CUDA faults still abort via torchrun/timeout.
            batches, error = [], None
            try:
                batches = [{**loader._load_one(entry), 'source': 0} for entry in local_entries]
            except Exception as caught:
                error = caught
            sync.check_error(error)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            metrics = trainer.train_batch(batches)
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.perf_counter() - started, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            records = [None] * world_size if rank == 0 else None
            dist.gather_object({'rank': rank, 'cases': [e['case'] for e in local_entries],
                'metrics': metrics, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)},
                records, dst=0)
            offset += args.batch_size
            if offset == len(schedule):
                epoch, offset = epoch + 1, 0
                schedule = epoch_entries(loader, epoch, args.seed, device)
            if rank == 0:
                with (output / 'distributed_metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps({'update': trainer.global_step,
                        'samples_seen': trainer.samples_seen, 'elapsed_seconds': elapsed.item(),
                        'ranks': records}) + '\n')
                print(f'update={trainer.global_step} global_samples={trainer.samples_seen} '
                      f'seconds={elapsed.item():.2f}', flush=True)
            if trainer.global_step % opt.checkpoint_every == 0 or trainer.global_step == opt.max_updates:
                save_state(trainer, args, rank, world_size, output, {'epoch': epoch, 'offset': offset})
        if resumed_payload is not None and trainer.global_step == resumed_payload['global_step']:
            save_state(trainer, args, rank, world_size, output, {'epoch': epoch, 'offset': offset})
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

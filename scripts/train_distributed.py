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
import math
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
                                    validate_distributed_checkpoint, balanced_sample_assignment,
                                    move_batch_tensors, presample_endpoint_randomness, endpoint_near_policy,
                                    all_reduce_and_wait)
from tricoopdiff.runtime import atomic_json, atomic_torch_save, capture_rng_state, preserve_rng, restore_rng_state, sha256_file
from train_entry import (parse_args, build_trainer, TernaryDataLoader,
                         _load_score_model_args, seed_training, validate)
from featurize_common import featurize_complex, esm_encoder_identity


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-checkpoint', help='Initialize trainable weights only; start a new optimizer/RNG/data run')
    parser.add_argument('--init-weights', choices=('raw', 'ema'), default='raw')
    parser.add_argument('--coupled-center-bank', help='Frozen predicted center bank; opt in to E initialization')
    parser.add_argument('--coupled-center-groups', default='train,independent_center,unseen_complex',
        help='Explicit required center groups; all present groups are audited')
    parser.add_argument('--coupled-validation-bank', help='Fixed validation bank identity recorded for separate evaluation')
    parser.add_argument('--torsion-std', type=float, default=.15)
    parser.add_argument('--translation-std', type=float, default=.5)
    parser.add_argument('--rotation-std', type=float, default=.05)
    parser.add_argument('--memory-fraction', type=float, default=.2)
    parser.add_argument('--cost-balanced-sharding', action='store_true',
        help='Presample original logical-rank RNG then rebalance zero-dropout endpoint work')
    parser.add_argument('--torch-interop-threads', type=int, default=None,
        help='Runtime CPU interop pool size; omitted preserves PyTorch default')
    parser.add_argument('--ternary-compile-messages', action='store_true',
        help='Compile pure tensor interaction messages using Inductor; errors fail closed.')
    parser.add_argument('--ternary-query-chunk-size', type=int, default=64,
        help='Interaction query rows per chunk below 600 nodes; large graphs retain 64.')
    parser.add_argument('--ternary-checkpoint-min-nodes', type=int, default=0,
        help='Runtime recomputation threshold: below this total node count retain ternary activations; 0 preserves historical policy')
    parser.add_argument('--ternary-checkpoint-reserve-gib', type=float, default=0.,
        help='Force recomputation below this CUDA headroom plus estimated activation budget; 0 disables guard')
    parser.add_argument('--ternary-checkpoint-activation-bytes-per-node-squared', type=int, default=114688,
        help='Conservative small-case activation headroom heuristic, not a guaranteed upper bound')
    parser.add_argument('--run-epochs', type=int, help='Total complete epochs with EMA validation, scheduler and best checkpoints')
    parser.add_argument('--allow-feature-cache-miss', action='store_true',
        help='Keep the complete population and wait for an external cache writer; never featurize on training ranks')
    parser.add_argument('--feature-cache-wait-seconds', type=float, default=3600.)
    parser.add_argument('--max-updates', type=int, default=None, help='Total optimizer update target, including resumed updates')
    parser.add_argument('--checkpoint-every', type=int, default=1)
    parser.add_argument('--checkpoint-updates', default='', help='Additional explicit checkpoint update numbers')
    parser.add_argument('--collective-timeout-seconds', type=int, default=900)
    opt, extra = parser.parse_known_args()
    if not math.isfinite(opt.ternary_checkpoint_reserve_gib) or opt.ternary_checkpoint_reserve_gib < 0 or opt.ternary_checkpoint_activation_bytes_per_node_squared < 0:
        parser.error('checkpoint reserve and activation budget must be finite and nonnegative')
    if opt.ternary_query_chunk_size < 1:
        parser.error('--ternary-query-chunk-size must be positive')
    if opt.ternary_checkpoint_min_nodes < 0:
        parser.error('--ternary-checkpoint-min-nodes must be nonnegative')
    if opt.run_epochs is not None and (opt.run_epochs < 1 or opt.max_updates is not None):
        parser.error('--run-epochs must be positive and cannot combine with --max-updates')
    if opt.max_updates is None and opt.run_epochs is None:
        opt.max_updates = 1
    args = parse_args(extra)
    if opt.init_checkpoint and args.resume:
        parser.error('--init-checkpoint and --resume are mutually exclusive')
    opt.coupled_center_groups = tuple(opt.coupled_center_groups.split(','))
    if (not opt.coupled_center_groups or len(set(opt.coupled_center_groups)) != len(opt.coupled_center_groups)
            or not set(opt.coupled_center_groups) <= {'train', 'independent_center', 'unseen_complex'}
            or 'train' not in opt.coupled_center_groups):
        parser.error('coupled-center-groups must include train and contain unique supported groups')
    try:
        opt.checkpoint_updates = {int(value) for value in opt.checkpoint_updates.split(',') if value}
    except ValueError:
        parser.error('checkpoint-updates must be comma-separated integers')
    if any(value < 0 for value in opt.checkpoint_updates):
        parser.error('checkpoint updates must be nonnegative')
    if not 0 < opt.memory_fraction <= 1:
        parser.error('memory fraction must be in (0,1]')
    if (opt.max_updates is not None and opt.max_updates < 1) or opt.collective_timeout_seconds < 1 or opt.checkpoint_every < 1:
        parser.error('update budget and collective timeout must be positive')
    if opt.coupled_center_bank and args.random_flip_proteins:
        parser.error('coupled frozen centers require --no-random-flip-proteins')
    if any(value < 0 for value in (opt.torsion_std, opt.translation_std, opt.rotation_std)):
        parser.error('coupled perturbation scales must be nonnegative')
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



def audit_coupled_centers(centers, cache_dir,
                          required_groups=('train', 'independent_center', 'unseen_complex')):
    """Recompute every frozen center's native-free geometry before training."""
    from run_coupled_small_sample import validate_center_geometry
    from tricoopdiff.initialization.candidates import _proposal_geometry

    records, cached_e3 = [], {}
    for group in ('train', 'independent_center', 'unseen_complex'):
        if group not in centers:
            if group in required_groups:
                raise ValueError(f'center bank is missing required group {group}')
            continue
        if not isinstance(centers[group], dict):
            raise ValueError(f'center bank has malformed group {group}')
        if not centers[group]:
            if group in required_groups:
                raise ValueError(f'center bank has empty required group {group}')
            # Explicitly unrequested groups may be serialized as empty maps.
            continue
        for case, values in centers[group].items():
            if case not in cached_e3:
                cached = torch.load(Path(cache_dir) / f'{case}.pt',
                                    map_location='cpu', weights_only=False)
                cached_e3[case] = cached['e3_data']['coords'].reshape(-1, 3)
            for record in values if isinstance(values, list) else [values]:
                center = record['center']
                geometry, acceptable, _ = _proposal_geometry(cached_e3[case],
                    center['poi'].reshape(-1, 3), center['ligand'].reshape(-1, 3))
                recalculated = {**center, 'geometry': geometry,
                    'acceptable': center.get('acceptable') is True and acceptable}
                error = None
                try:
                    validate_center_geometry(recalculated, case=case, seed=record['seed'])
                except ValueError as caught:
                    error = str(caught)
                records.append({'group': group, 'case': case, 'seed': record['seed'],
                    'sha256': record['sha256'], 'stored_acceptable': center.get('acceptable'),
                    'recomputed_acceptable': acceptable, 'geometry': geometry,
                    'valid': error is None, 'error': error})
    return {'contract': 'recomputed_native_free_two_anchor_geometry_v1',
            'required_groups': list(required_groups),
            'audited_groups': sorted({record['group'] for record in records}),
            'count': len(records), 'valid_count': sum(r['valid'] for r in records),
            'invalid_count': sum(not r['valid'] for r in records), 'records': records}



class TimedGradientSynchronizer(GradientSynchronizer):
    """Host wall times; NCCL enqueue time is not pure GPU/network latency."""

    def reset_timing(self):
        self.error_check_host_seconds = 0.
        self.gradient_sync_host_seconds = 0.

    def check_error(self, error):
        started = time.perf_counter()
        try:
            return super().check_error(error)
        finally:
            self.error_check_host_seconds += time.perf_counter() - started

    def __call__(self, parameters, local_samples):
        started = time.perf_counter()
        try:
            return super().__call__(parameters, local_samples)
        finally:
            self.gradient_sync_host_seconds += time.perf_counter() - started

    def __init__(self, device):
        super().__init__(device)
        self.reset_timing()

def merge_epoch_validation(reports):
    """Merge disjoint A1 validation shards with sample-weighted means."""
    rows = [row for report in reports for row in report['per_case']]
    cases = [row['case'] for row in rows]
    if not rows or len(set(cases)) != len(cases):
        raise ValueError('distributed validation must cover distinct nonempty cases')
    counts = [report['source_counts'].get('sdpi', 0) for report in reports]
    if sum(counts) != len(rows) or any(report['source_counts'].get('bitseed', 0) for report in reports):
        raise ValueError('formal distributed validation supports complete A1 SDPI only')
    total = sum(counts)
    result = dict(reports[0])
    result.update(selection_loss=sum(report['selection_loss'] * count
                                     for report, count in zip(reports, counts)) / total,
        selection_dockq=sum(report['selection_dockq'] * count
                            for report, count in zip(reports, counts)) / total,
        source_counts={'sdpi':total,'bitseed':0}, per_case=rows)
    result['by_source'] = {'sdpi':result['selection_loss']}
    result['dockq_by_source'] = {'sdpi':result['selection_dockq']}
    return result


@preserve_rng
def distributed_validate(trainer, loader, device, rank, world_size, sync, *, native_root):
    """Evaluate disjoint deterministic representatives, preserving rank RNG."""
    entries = epoch_entries(loader, 0, 73, device)
    local = entries[rank::world_size]
    if not local:
        raise ValueError('validation requires at least one representative per rank')

    class ValidationShard:
        entries_by_case = loader.entries_by_case

        def __iter__(self):
            for entry in local:
                yield loader._load_one(entry)

    report, error = None, None
    try:
        report = validate(trainer, ValidationShard(), device, native_root=native_root, weights='ema')
    except Exception as caught:
        error = caught
    sync.check_error(error)
    reports = [None] * world_size
    dist.all_gather_object(reports, report)
    merged = merge_epoch_validation(reports)
    if set(row['case'] for row in merged['per_case']) != set(entry['case'] for entry in entries):
        raise ValueError('distributed validation changed the official representative population')
    return merged

def save_state(trainer, args, rank, world_size, output, cursor, *, names=None, selection=None,
               best_val_loss=float("inf"), best_val_dockq=float("-inf")):
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
        payload = trainer.checkpoint_payload(epoch=cursor['epoch'],
            best_val_loss=best_val_loss, best_val_dockq=best_val_dockq)
        if selection is not None:
            payload['checkpoint_selection'] = selection
        # Prevent generic single-GPU resume from consuming only rank 0's RNG.
        payload['checkpoint_kind'] = 'distributed_training'
        payload['distributed_state'] = {
            'contract': 'explicit_sample_weighted_allreduce_v1',
            'world_size': world_size, 'global_batch_size': args.batch_size,
            'rank_rng_states': states, 'next_data_cursor': cursor,
            'rank_parameter_sha256': digests,
            'resume_supported': True,
            'initialization_protocol': args.initialization_protocol,
            'weight_initialization': getattr(args, 'weight_initialization', None),
            'resume_contract': 'rank_rng_cursor_dataset_v1',
            'train_index_sha256': sha256_file(args.train_data),
            'seed': args.seed, 'phase': args.phase,
            'random_flip_proteins': args.random_flip_proteins,
            'epoch_size': args.distributed_epoch_size,
            'checkpoint_ternary': args.checkpoint_ternary,
            'sharding_protocol': getattr(args, 'distributed_sharding_protocol', 'contiguous_logical_rank_v1'),
            'epoch_validation_enabled': getattr(args, 'distributed_epoch_validation', False),
            'val_index_sha256': (sha256_file(args.val_data) if getattr(args, 'distributed_epoch_validation', False) else None),
        }
        for name in names or [f'checkpoint_update{trainer.global_step}.pt']:
            atomic_torch_save(payload, output / name)
    dist.barrier()


def main():
    opt, args = options()
    if opt.torch_interop_threads is not None:
        if opt.torch_interop_threads < 1:
            raise SystemExit('--torch-interop-threads must be positive')
        torch.set_num_interop_threads(opt.torch_interop_threads)
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
        args.distributed_sharding_protocol = ('logical_rank_presampled_zero_dropout_lpt_v1'
            if opt.cost_balanced_sharding else 'contiguous_logical_rank_v1')
        args.training_endpoint_protocol = 'none'
        args.generative_process = 'sdpi_flow_v1'
        score_args = _load_score_model_args(args.expert_model_dir)
        # These ranks only consume caches: pin full-content encoder identity once
        # for this process lifetime, never infer it from mutable stat metadata.
        encoder_pin = [None]
        if rank == 0:
            try:
                encoder_pin[0] = {'identity': esm_encoder_identity(), 'error': None}
            except Exception as exc:
                encoder_pin[0] = {'identity': None, 'error': repr(exc)}
        dist.broadcast_object_list(encoder_pin, src=0)
        if encoder_pin[0]['error'] is not None:
            raise RuntimeError('ESM startup identity failed: ' + encoder_pin[0]['error'])
        expected_esm_encoder_identity = encoder_pin[0]['identity']
        if rank == 0:
            print(json.dumps({'event': 'esm_identity_pinned',
                'scope': 'startup_full_content_rank0_broadcast_read_only_cache_followers',
                'identity': expected_esm_encoder_identity}), flush=True)
        loader = TernaryDataLoader(args.train_data, score_args, shuffle=True,
            device=str(device), cache_dir=args.cache_dir,
            random_flip_proteins=args.random_flip_proteins, cache_read_only=True,
            cache_wait_seconds=opt.feature_cache_wait_seconds,
            matcha_metadata_read_only=opt.allow_feature_cache_miss,
            expected_esm_encoder_identity=expected_esm_encoder_identity)
        if not args.cache_dir:
            raise ValueError('--cache_dir with preprocessed features is required')
        remainder = len(loader) % args.batch_size
        if not len(loader) or (remainder and remainder < world_size):
            raise ValueError('each global batch, including the unpadded epoch tail, needs at least one real sample per rank')
        args.distributed_epoch_validation = opt.run_epochs is not None
        if opt.run_epochs is not None:
            args.epochs = opt.run_epochs
            opt.max_updates = opt.run_epochs * ((len(loader) + args.batch_size - 1) // args.batch_size)
        # Require existing caches: parallel cache creation is not the purpose of
        # this entry, and all ranks must see the same complete population.
        missing = [entry['case'] for entry in loader.entries
                   if not (Path(args.cache_dir) / (entry['case'] + '.pt')).is_file()]
        if missing and not opt.allow_feature_cache_miss:
            raise FileNotFoundError(f'preprocess features before torchrun; missing {missing[:8]}')
        if rank == 0:
            print(json.dumps({'event':'population_preflight','entries':len(loader.entries),
                'epoch_clusters':len(loader),'feature_cache_missing':len(missing),
                'cache_policy':'external_writer_follower' if opt.allow_feature_cache_miss else 'require_complete'}),flush=True)
        args.distributed_epoch_size = len(loader)
        centers = None
        args.initialization_protocol = {'name': 'online_sdpi_v1'}
        if opt.cost_balanced_sharding and opt.coupled_center_bank:
            raise ValueError('cost balancing currently preserves online-SDPI streams only')
        if opt.coupled_center_bank:
            from run_coupled_small_sample import perturb, center_hash
            centers = torch.load(opt.coupled_center_bank, map_location='cpu', weights_only=False)
            bank_audit = audit_coupled_centers(centers, args.cache_dir, opt.coupled_center_groups)
            if rank == 0:
                atomic_json(bank_audit, output / 'center_bank_audit.json')
            if bank_audit['invalid_count']:
                raise ValueError(f"coupled center bank rejected: {bank_audit['invalid_count']}/{bank_audit['count']} centers fail recomputed native-free geometry")
            if set(centers.get('train', {})) != set(loader.entries_by_case):
                raise ValueError('coupled center bank must exactly cover training cases')
            if len(loader) != len(loader.entries):
                raise ValueError('coupled epoch occurrence requires one materialized case per cluster')
            for case, records in centers['train'].items():
                if len(records) != 2:
                    raise ValueError(f'{case}: coupled training requires exactly two frozen centers')
                for record in records:
                    if record.get('native_information_used') is not False:
                        raise ValueError(f'{case}: center is not certified native-free')
                    if record.get('sha256') != center_hash(record['center']):
                        raise ValueError(f'{case}: predicted center hash differs')
            args.initialization_protocol = {
                'name': 'coupled_frozen_centers_epoch_v1',
                'geometry_contract': 'recomputed_native_free_two_anchor_geometry_v1',
                'required_center_groups': list(opt.coupled_center_groups),
                'audited_center_groups': bank_audit['audited_groups'],
                'center_bank_sha256': sha256_file(opt.coupled_center_bank),
                'validation_bank_sha256': (sha256_file(opt.coupled_validation_bank)
                    if opt.coupled_validation_bank else None),
                'torsion_std': opt.torsion_std, 'translation_std': opt.translation_std,
                'rotation_std': opt.rotation_std, 'perturbation_seed_base': 7301,
                'center_index': 'epoch % 2', 'occurrence': 'epoch',
            }

        resumed_payload, resumed_state = None, None
        if args.resume:
            resumed_payload = torch.load(args.resume, map_location='cpu', weights_only=False)
            resumed_state = validate_distributed_checkpoint(resumed_payload, world_size, args.batch_size,
                train_index_sha256=sha256_file(args.train_data), seed=args.seed, phase=args.phase,
                random_flip_proteins=args.random_flip_proteins, epoch_size=len(loader),
                checkpoint_ternary=args.checkpoint_ternary)
            args.weight_initialization = resumed_state.get('weight_initialization')
            if resumed_state.get('sharding_protocol', 'contiguous_logical_rank_v1') not in (
                    'contiguous_logical_rank_v1', args.distributed_sharding_protocol):
                raise ValueError('resume cannot discard logical-rank presampling protocol')
            if resumed_state.get('epoch_validation_enabled', False) != args.distributed_epoch_validation:
                raise ValueError('resume cannot change epoch validation/scheduler policy')
            if args.distributed_epoch_validation and resumed_state.get('val_index_sha256') != sha256_file(args.val_data):
                raise ValueError('resume validation population differs')
            saved_initialization = resumed_state.get('initialization_protocol', {'name': 'online_sdpi_v1'})
            if saved_initialization != args.initialization_protocol:
                raise ValueError('distributed checkpoint initialization protocol differs; SDPI and E cannot be interchanged')
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
            first['protein2'], first['ligand'], score_args, str(device), cache_dir=args.cache_dir,
            cache_read_only=True, cache_wait_seconds=opt.feature_cache_wait_seconds,
            expected_esm_encoder_identity=expected_esm_encoder_identity)
        seed_training(args.seed, device)
        # Reconstruct the parent's exact architecture/expert frame, while keeping
        # the new run's optimizer, scheduler, random streams and counters fresh.
        resume_path = args.resume
        if opt.init_checkpoint:
            args.resume = opt.init_checkpoint
        try:
            model, _, trainer = build_trainer(args, device, input_dim=input_dim)
        finally:
            args.resume = resume_path
        if opt.init_checkpoint:
            from tricoopdiff.models.expert import validate_expert_base_identity
            parent = torch.load(opt.init_checkpoint, map_location='cpu', weights_only=False)
            validate_expert_base_identity(parent, model.expert)
            field = 'trifuse_state_dict' if opt.init_weights == 'raw' else 'ema_trifuse_state_dict'
            model.trifuse.load_state_dict(parent[field], strict=True)
            if trainer.optimizer.state or trainer.global_step != 0 or trainer.samples_seen != 0:
                raise RuntimeError('weight-only initialization must retain fresh optimizer and counters')
            args.weight_initialization = {
                'checkpoint': str(Path(opt.init_checkpoint).resolve()),
                'sha256': sha256_file(opt.init_checkpoint), 'weights': opt.init_weights,
                'parent_global_step': parent['global_step'],
                'optimizer_loaded': False, 'rng_loaded': False, 'counters_reset': True,
            }
            del parent
        model.set_ternary_compile_messages(opt.ternary_compile_messages)
        trainer.require_deterministic_forward = opt.cost_balanced_sharding
        if opt.cost_balanced_sharding and any(
                (isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0)
                or (isinstance(module, torch.nn.MultiheadAttention) and module.dropout > 0)
                for module in model.modules()):
            raise ValueError('cost-balanced legacy-RNG compatibility requires zero-dropout endpoint forward')
        model.set_ternary_query_chunk_size(opt.ternary_query_chunk_size)
        model.checkpoint_min_nodes = opt.ternary_checkpoint_min_nodes
        model.checkpoint_memory_reserve_bytes = int(opt.ternary_checkpoint_reserve_gib * 2**30)
        model.checkpoint_activation_bytes_per_node_squared = opt.ternary_checkpoint_activation_bytes_per_node_squared
        model.checkpoint_memory_fraction = opt.memory_fraction
        args.ternary_checkpoint_min_nodes = opt.ternary_checkpoint_min_nodes
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
        sync = TimedGradientSynchronizer(device)
        trainer.gradient_sync = sync
        trainer.distributed_error_check = sync.check_error
        local_batch_size = args.batch_size // world_size
        trainer.distributed_sample_offset = rank * local_batch_size
        if args.resume:
            restore_rng_state(resumed_state['rank_rng_states'][rank])
        else:
            seed_training(args.seed + 1000003 * rank, device)
        if rank == 0:
            atomic_json(vars(args), output / 'config.json')
            atomic_json({'synchronization': 'explicit_sample_weighted_allreduce_v1',
                'world_size': world_size, 'global_batch_size': args.batch_size,
                'sharding_protocol': args.distributed_sharding_protocol,
                'sharding_policy': {'rng': 'original_contiguous_logical_rank_streams',
                    'near_clock': 'original_absolute_global_sample_index',
                    'cost': 'branch_and_controller_nodes_squared_plus_torsions_times_main_near_steps',
                    'forward': 'zero_dropout_rng_unchanged_checked_before_optimizer',
                    'gradient': 'sample_weighted_mean_fp_reassociation_allowed'} if opt.cost_balanced_sharding else None,
                'local_batch_size': local_batch_size, 'max_updates': opt.max_updates,
                'checkpoint_every': opt.checkpoint_every,
                'cpu_runtime_policy': {'torch_intraop_threads': torch.get_num_threads(),
                    'torch_interop_threads': torch.get_num_interop_threads(),
                    'environment': {key: os.environ.get(key) for key in
                        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'OMP_WAIT_POLICY', 'KMP_BLOCKTIME', 'TORCH_NCCL_BLOCKING_WAIT',
                         'TORCHINDUCTOR_COMPILE_THREADS', 'MAX_JOBS', 'TORCHINDUCTOR_CACHE_DIR')}},
                'static_cache': 'tensor_storage_version_bounded_v1',
                'memory_runtime_policy': {'checkpoint_ternary': args.checkpoint_ternary,
                    'ternary_query_chunk_size': opt.ternary_query_chunk_size,
                    'ternary_compile_messages': opt.ternary_compile_messages,
                    'ternary_compile_backend': 'inductor' if opt.ternary_compile_messages else None,
                    'ternary_compile_policy': 'two_operand_dynamic_nodes_geometry_contract_fifo_v5' if opt.ternary_compile_messages else None,
                    'ternary_compile_recompile_limit': 32 if opt.ternary_compile_messages else None,
                    'ternary_compile_isolate_recompiles': bool(opt.ternary_compile_messages),
                    'dynamo_process_cache_order': 'compilation_order_lru_disabled' if opt.ternary_compile_messages else 'torch_default',
                    'ternary_compile_geometry_contract': 'gradient_preserving_clone_contiguous_no_view_offset_zero' if opt.ternary_compile_messages else None,
                    'collective_host_wait_policy': 'explicit_async_work_wait_v1',
                    'ternary_query_chunk_large_nodes': 600,
                    'ternary_query_chunk_large_size': 64,
                    'ternary_checkpoint_min_nodes': opt.ternary_checkpoint_min_nodes,
                    'memory_fraction': opt.memory_fraction,
                    'reserve_gib': opt.ternary_checkpoint_reserve_gib,
                    'activation_bytes_per_node_squared': opt.ternary_checkpoint_activation_bytes_per_node_squared,
                    'oom_policy': 'fail_closed_collectively_no_gradient_replay'},
                'checkpoint_updates': sorted(opt.checkpoint_updates),
                'esm_encoder_identity_pin': {'identity': expected_esm_encoder_identity,
                    'scope': 'startup_full_content_rank0_broadcast_read_only_cache_followers'},
                'resume_supported': True,
                'initialization_protocol': args.initialization_protocol,
                'entry_sha256': sha256_file(__file__),
                'resume_from': args.resume, 'initial_global_step': trainer.global_step,
                'legacy_dataset_identity': ('path_only' if resumed_state is not None and
                    resumed_state.get('resume_contract') is None else None), 'validation': 'distributed EMA per epoch' if opt.run_epochs else 'not run by this training-only entry',
                'config': vars(args)}, output / 'distributed_manifest.json')
        epoch, offset = ((resumed_state['next_data_cursor']['epoch'],
                          resumed_state['next_data_cursor']['offset']) if resumed_state else (0, 0))
        schedule = epoch_entries(loader, epoch, args.seed, device)
        best_val_loss, best_val_dockq = trainer.resume_best_val_loss, trainer.resume_best_val_dockq
        val_loader = (TernaryDataLoader(args.val_data, score_args, shuffle=False, device=str(device),
            cache_dir=args.cache_dir, cache_read_only=True, cache_wait_seconds=opt.feature_cache_wait_seconds,
            expert=model.expert, matcha_metadata_read_only=opt.allow_feature_cache_miss,
            expected_esm_encoder_identity=expected_esm_encoder_identity) if opt.run_epochs is not None else None)
        if trainer.global_step == 0:
            save_state(trainer, args, rank, world_size, output, {'epoch': epoch, 'offset': offset},
                    best_val_loss=best_val_loss, best_val_dockq=best_val_dockq)
        for _ in range(trainer.global_step, opt.max_updates):
            global_entries = schedule[offset:offset + args.batch_size]
            local_entries = shard_global_batch(global_entries, args.batch_size, rank, world_size)
            trainer.distributed_sample_indices = None
            trainer.distributed_sample_offset = sum(len(shard_global_batch(global_entries,
                args.batch_size, previous, world_size)) for previous in range(rank))
            # Loading failures must be coordinated before any rank starts a
            # backward collective. CUDA faults still abort via torchrun/timeout.
            batches, error, initialization = [], None, []
            sync.reset_timing()
            data_started = time.perf_counter()
            try:
                batches = [{**loader._load_one(entry), 'source': 0} for entry in local_entries]
                if centers is not None:
                    from train_entry import case_seed
                    for batch in batches:
                        case = batch['case']
                        record = centers['train'][case][epoch % 2]
                        seed = case_seed(f'{case}:training_perturbation:{epoch}', 7301)
                        coupling, provenance = perturb(batch, record, seed, opt)
                        batch['coupling_override'] = coupling
                        initialization.append({'case': case, 'occurrence': epoch,
                            'center_sha256': record['sha256'], 'perturbation_seed': seed,
                            **provenance})
            except Exception as caught:
                error = caught
            data_loading_seconds = time.perf_counter() - data_started
            sync.check_error(error)
            redistribution_seconds = 0.
            if opt.cost_balanced_sharding:
                redistribution_started = time.perf_counter()
                prepared, error = [], None
                try:
                    for index, batch in enumerate(batches):
                        sample_index = trainer.samples_seen + trainer.distributed_sample_offset + index
                        prepared.append(move_batch_tensors(presample_endpoint_randomness(
                            trainer, batch, sample_index), 'cpu'))
                except Exception as caught:
                    error = caught
                sync.check_error(error)
                gathered = [None] * world_size
                dist.all_gather_object(gathered, prepared)
                global_batches = [batch for logical_shard in gathered for batch in logical_shard]
                costs = []
                for index, batch in enumerate(global_batches):
                    e3, poi, lig = (int(batch[key + '_data']['coords'].shape[-2])
                                    for key in ('e3', 'poi', 'ligand'))
                    near_t = endpoint_near_policy(trainer, trainer.samples_seen + index)[1] if trainer.near_target_weight else 0.
                    steps = trainer.diffusion.inference_steps * (1. + near_t)
                    torsions = int(batch['ligand_data']['edge_mask'].sum())
                    costs.append(steps * ((e3 + lig)**2 + (poi + lig)**2 + (e3 + poi + lig)**2 + torsions * lig))
                assigned = balanced_sample_assignment(costs, world_size)[rank]
                batches = [move_batch_tensors(global_batches[index], device) for index in assigned]
                local_entries = [global_entries[index] for index in assigned]
                trainer.distributed_sample_indices = [trainer.samples_seen + index for index in assigned]
                redistribution_seconds = time.perf_counter() - redistribution_started
                del gathered, global_batches, prepared
            data_error_check_host_seconds = sync.error_check_host_seconds
            if trainer.global_step == (resumed_payload['global_step'] if resumed_payload else 0):
                print(json.dumps({'event': 'first_batch_data_ready', 'rank': rank,
                    'next_update': trainer.global_step + 1,
                    'cases': [entry['case'] for entry in local_entries],
                    'data_loading_seconds': data_loading_seconds,
                    'redistribution_seconds': redistribution_seconds,
                    'global_sample_indices': trainer.distributed_sample_indices,
                    'esm_identity_policy': 'startup_pinned_full_content_read_only'}), flush=True)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            metrics = trainer.train_batch(batches)
            torch.cuda.synchronize(device)
            local_elapsed_seconds = time.perf_counter() - started
            elapsed = torch.tensor(local_elapsed_seconds, device=device)
            reduce_started = time.perf_counter()
            all_reduce_and_wait(elapsed, op=dist.ReduceOp.MAX)
            max_elapsed_seconds = elapsed.item()
            report_reduce_host_seconds = time.perf_counter() - reduce_started
            case_work = []
            for batch, metric in zip(batches, metrics):
                sizes = {key: int(batch[key + '_data']['coords'].shape[-2])
                         for key in ('e3', 'poi', 'ligand')}
                case_work.append({'case':batch['case'], 'nodes':sizes,
                    'total_nodes':sum(sizes.values()),
                    'projection_nfev':metric.get('target_projection_nfev')})
            gather_started = time.perf_counter()
            records = [None] * world_size if rank == 0 else None
            dist.gather_object({'rank': rank, 'cases': [e['case'] for e in local_entries],
                'local_elapsed_seconds':local_elapsed_seconds, 'data_loading_seconds':data_loading_seconds,
                'redistribution_seconds':redistribution_seconds,
                'global_sample_indices':trainer.distributed_sample_indices,
                'data_error_check_host_seconds':data_error_check_host_seconds,
                'backward_error_check_host_seconds':sync.error_check_host_seconds-data_error_check_host_seconds,
                'gradient_sync_host_seconds':sync.gradient_sync_host_seconds,
                'report_reduce_host_seconds':report_reduce_host_seconds, 'case_work':case_work,
                'metrics': metrics, 'initialization': initialization, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)},
                records, dst=0)
            rank0_gather_host_seconds = time.perf_counter() - gather_started
            offset += len(global_entries)
            completed_epoch = offset == len(schedule)
            if completed_epoch:
                epoch, offset = epoch + 1, 0
                schedule = epoch_entries(loader, epoch, args.seed, device)
            if rank == 0:
                with (output / 'distributed_metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps({'update': trainer.global_step,
                        'samples_seen': trainer.samples_seen, 'elapsed_seconds': max_elapsed_seconds,
                        'rank0_gather_host_seconds':rank0_gather_host_seconds,
                        'timing_contract':'host_wall_train_includes_collective_wait_no_extra_cuda_synchronize_v1',
                        'ranks': records}) + '\n')
                print(f'update={trainer.global_step} global_samples={trainer.samples_seen} '
                      f'seconds={max_elapsed_seconds:.2f}', flush=True)
            if completed_epoch and val_loader is not None:
                validation = distributed_validate(trainer, val_loader, device, rank, world_size,
                                                  sync, native_root=args.native_root)
                val_loss, val_dockq = validation['selection_loss'], validation['selection_dockq']
                trainer.scheduler_step(val_loss)
                improved_loss, improved_pp = val_loss < best_val_loss, val_dockq > best_val_dockq
                best_val_loss, best_val_dockq = min(best_val_loss, val_loss), max(best_val_dockq, val_dockq)
                if rank == 0:
                    print(f'epoch={epoch} validation_loss={val_loss:.6f} PP_DockQ={val_dockq:.6f} '
                          f'lr={trainer.optimizer.param_groups[0]["lr"]:.8g}',flush=True)
                    atomic_json({'epoch':epoch, 'update':trainer.global_step, **validation},
                                output / f'validation_epoch{epoch}.json')
                for improved, filename, metric, scope, value in (
                    (improved_loss, 'best_loss_model.pt', 'endpoint_selection_loss', 'training_objective', val_loss),
                    (improved_pp, 'best_model.pt', 'protein_protein_dockq', 'PP_only', val_dockq)):
                    if improved:
                        save_state(trainer, args, rank, world_size, output, {'epoch':epoch,'offset':offset},
                            names=[filename], best_val_loss=best_val_loss, best_val_dockq=best_val_dockq,
                            selection={'metric':metric,'scope':scope,'value':value,'weights':'ema'})
                save_state(trainer, args, rank, world_size, output, {'epoch':epoch,'offset':offset},
                    names=[f'checkpoint_epoch{epoch}.pt','progress_model.pt'],
                    best_val_loss=best_val_loss, best_val_dockq=best_val_dockq)
            if (trainer.global_step % opt.checkpoint_every == 0 or trainer.global_step == opt.max_updates
                    or trainer.global_step in opt.checkpoint_updates):
                save_state(trainer, args, rank, world_size, output, {'epoch': epoch, 'offset': offset},
                    best_val_loss=best_val_loss, best_val_dockq=best_val_dockq)
        if val_loader is not None and trainer.global_step == opt.max_updates:
            save_state(trainer, args, rank, world_size, output, {'epoch':epoch,'offset':offset},
                names=['final_model.pt'], best_val_loss=best_val_loss, best_val_dockq=best_val_dockq)
        if resumed_payload is not None and trainer.global_step == resumed_payload['global_step']:
            save_state(trainer, args, rank, world_size, output, {'epoch': epoch, 'offset': offset},
                    best_val_loss=best_val_loss, best_val_dockq=best_val_dockq)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

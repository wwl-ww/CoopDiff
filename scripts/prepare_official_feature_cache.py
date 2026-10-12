#!/usr/bin/env python
"""Offline persistent ESM worker: official cluster-first priority, then all members."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT)]
import torch
import featurize_common as features
from tricoopdiff.inputs.esm_embedder import ESMEmbedder
from tricoopdiff.runtime import atomic_torch_save, sha256_file
from train_entry import _load_score_model_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--memory-fraction', type=float, default=.35)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--priority-case')
    parser.add_argument('--priority-index', type=Path)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error('rank must be within world size')
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / f'worker{args.rank}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output / f'worker{args.rank}.pid').write_text(str(os.getpid()))
    device = torch.device('cuda', 0)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    identity = features.esm_encoder_identity()
    if not features._esm_identity_verified(identity):
        raise ValueError('encoder local assets incomplete; preparation cannot download')
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    sequence_dir = output / 'sequence_embeddings' / identity_hash
    sequence_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output / 'feature_cache'
    cache_dir.mkdir(exist_ok=True)
    # The resident encoder is frozen for this worker; its actual identity is
    # checked before and after loading, then retained for all feature records.
    embedder = ESMEmbedder(model_name=identity['model'], device=str(device))
    if identity != features.esm_encoder_identity():
        raise ValueError('encoder assets changed while loading the resident model')
    features.esm_encoder_identity = lambda: identity
    sequence_hits = 0
    sequence_generated = 0
    def persistent_embeddings(protein_paths, unused_device):
        nonlocal sequence_hits, sequence_generated
        sequences = features.get_sequences(protein_paths, [None] * len(protein_paths))
        unique = list(dict.fromkeys(chain for seq in sequences for chain in seq.split(':')))
        results = {}
        pending = []
        for seq in unique:
            path = sequence_dir / (hashlib.sha256(seq.encode()).hexdigest() + '.pt')
            if path.is_file():
                record = torch.load(path, map_location='cpu', weights_only=False)
                if record['sequence'] != seq or record['encoder_identity_sha256'] != identity_hash:
                    raise ValueError('sequence cache identity mismatch')
                embedding = record['embedding']
                if embedding.shape != (len(seq), 1280) or not torch.isfinite(embedding).all():
                    raise ValueError('invalid sequence embedding cache')
                results[seq] = embedding
                sequence_hits += 1
            else:
                pending.append((seq, path))
        # Sort within each protein pair only; batch padding never truncates a chain.
        pending.sort(key=lambda item: len(item[0]))
        cursor = 0
        while cursor < len(pending):
            batch_size = 1 if len(pending[cursor][0]) > 1000 else 4
            batch = pending[cursor:cursor + batch_size]
            try:
                embeddings = embedder.embed_from_fasta([item[0] for item in batch], batch_size=len(batch))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                embeddings = []
                for seq, _ in batch:
                    embeddings.extend(embedder.embed_from_fasta([seq], batch_size=1))
            for (seq, path), embedding in zip(batch, embeddings):
                if embedding.shape != (len(seq), 1280) or not torch.isfinite(embedding).all():
                    raise ValueError('resident encoder produced invalid sequence embedding')
                atomic_torch_save({'sequence': seq, 'encoder_identity_sha256': identity_hash,
                                   'embedding': embedding}, path)
                results[seq] = embedding
                sequence_generated += 1
            cursor += len(batch)
        return {path: [results[chain] for chain in seq.split(':')]
                for path, seq in zip(protein_paths, sequences)}
    features.generate_esm_embeddings = persistent_embeddings
    priority = []
    full = []
    for split in ('val', 'train'):
        priority.extend(json.loads((args.data_root / f'{split}_cluster_first.json').read_text())['entries'])
        full.extend(json.loads((args.data_root / f'{split}_full.json').read_text())['entries'])
    priority_cases = {entry['case'] for entry in priority}
    queue = [(entry, 'cluster_first') for entry in priority]
    queue.extend((entry, 'remaining_members') for entry in full if entry['case'] not in priority_cases)
    if args.priority_case:
        queue.sort(key=lambda pair: pair[0]['case'] != args.priority_case)
    if args.priority_index:
        requested = json.loads(args.priority_index.read_text())
        requested = requested.get('entries', []) if isinstance(requested, dict) else requested
        order = {entry['case']: index for index, entry in enumerate(requested)}
        available = {entry['case'] for entry, _ in queue}
        if not set(order) <= available:
            raise ValueError('priority index contains cases outside official full population')
        queue.sort(key=lambda pair: order.get(pair[0]['case'], len(order)))
    if len(queue) != len({entry['case'] for entry, _ in queue}):
        raise ValueError('official preparation queue contains duplicate cases')
    queue = queue[args.rank::args.world_size]
    if args.limit is not None:
        queue = queue[:args.limit]
    model_args = _load_score_model_args(ROOT / 'checkpoints/score_model')
    started = time.monotonic()
    manifest = {'pid': os.getpid(), 'rank': args.rank, 'world_size': args.world_size,
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'encoder_identity': identity, 'encoder_identity_sha256': identity_hash,
                'script_sha256': sha256_file(__file__), 'queued_cases': len(queue),
                'feature_cache': str(cache_dir),
                'priority_index': str(args.priority_index) if args.priority_index else None,
                'priority_index_sha256': sha256_file(args.priority_index) if args.priority_index else None,
                'queue_cases_sha256': hashlib.sha256(json.dumps([entry['case'] for entry, _ in queue]).encode()).hexdigest(),
                'purpose': 'full official population; cluster-first is preparation priority only'}
    (output / f'worker{args.rank}_manifest.json').write_text(json.dumps(manifest, indent=2))
    log = (output / f'worker{args.rank}_cases.jsonl').open('a', buffering=1)
    success = failed = 0
    for number, (entry, phase) in enumerate(queue, 1):
        before = time.monotonic()
        record = {'case': entry['case'], 'phase': phase, 'number': number}
        try:
            features.featurize_complex(entry['case'], entry['protein1'], entry['protein2'], entry['ligand'],
                                       model_args, str(device), cache_dir=str(cache_dir))
            record['status'] = 'ok'
            success += 1
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
            failed += 1
            torch.cuda.empty_cache()
        record['seconds'] = time.monotonic() - before
        record['sequence_hits'] = sequence_hits
        record['sequence_generated'] = sequence_generated
        log.write(json.dumps(record) + '\n')
        progress = {**manifest, 'status': 'running', 'processed': number, 'success': success, 'failed': failed,
                    'last_case': entry['case'], 'phase': phase, 'elapsed_seconds': time.monotonic() - started,
                    'sequence_hits': sequence_hits, 'sequence_generated': sequence_generated}
        (output / f'worker{args.rank}_progress.json').write_text(json.dumps(progress, indent=2))
        print(json.dumps(record), flush=True)
    progress['status'] = 'complete_with_failures' if failed else 'complete'
    (output / f'worker{args.rank}_progress.json').write_text(json.dumps(progress, indent=2))


if __name__ == '__main__':
    main()

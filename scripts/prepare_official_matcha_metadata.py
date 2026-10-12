#!/usr/bin/env python
"""Prepare official Matcha metadata with one resident frozen sequence encoder."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT)]
import torch
import tricoopdiff.diffdock_compat
import featurize_common as features
from train_entry import TernaryDataLoader, _load_score_model_args
from tricoopdiff.models.expert import FrozenMatcha, expert_base_identity
from tricoopdiff.runtime import atomic_json, atomic_torch_save, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--memory-fraction', type=float, default=.2)
    parser.add_argument('--device', default='cuda:0', help='cuda:N or cpu for offline metadata preparation')
    parser.add_argument('--sequence-cache-dir', type=Path,
                        help='reuse an existing content-keyed sequence embedding directory')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'writer.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output / 'pid').write_text(str(os.getpid()))
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    expert = FrozenMatcha(str(ROOT / 'results/matcha_qualification_source'),
                         str(ROOT / 'results/matcha_qualification_weights'), device=str(device))
    identity = expert_base_identity(expert)
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    encoder_path = str(expert.weights_dir / 'esm2_t12_35M')
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
    encoder = AutoModelForMaskedLM.from_pretrained(encoder_path, local_files_only=True).to(device).eval()
    encoder.requires_grad_(False)
    sequence_root = (args.sequence_cache_dir if args.sequence_cache_dir else output / 'sequence_embeddings') / identity['sequence_encoder_sha256']
    sequence_root.mkdir(parents=True, exist_ok=True)

    class ResidentSequenceEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder

        def forward(self, **tokens):
            digest = hashlib.sha256()
            for key in sorted(tokens):
                if torch.is_tensor(tokens[key]):
                    value = tokens[key].detach().cpu().contiguous()
                    digest.update(key.encode())
                    digest.update(str((tuple(value.shape), value.dtype)).encode())
                    digest.update(value.numpy().tobytes())
            path = sequence_root / (digest.hexdigest() + '.pt')
            if path.is_file():
                hidden = torch.load(path, map_location=device, weights_only=True)
            else:
                with torch.no_grad():
                    hidden = self.encoder(**tokens).hidden_states[-1].detach()
                if not torch.isfinite(hidden).all():
                    raise ValueError('nonfinite Matcha sequence embedding')
                atomic_torch_save(hidden.cpu(), path)
            return SimpleNamespace(hidden_states=(hidden,))

    resident = ResidentSequenceEncoder().to(device).eval()
    # Only this isolated writer replaces the factories. Parent prepare_case
    # retains its original algorithm, metadata schema and frozen identity.
    AutoModelForMaskedLM.from_pretrained = staticmethod(lambda *a, **k: resident)
    AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: tokenizer)
    score_args = _load_score_model_args(ROOT / 'checkpoints/score_model')
    entries, helpers = {}, {}
    for split in ('train', 'val'):
        index = args.data_root / f'{split}_full.json'
        loader = TernaryDataLoader(str(index), score_args, shuffle=False,
                                  device=str(device), cache_dir=str(args.cache_dir), expert=expert)
        for entry in loader.entries:
            case = entry['case']
            if case in entries and entries[case] != entry:
                raise ValueError(f'ambiguous official case identity: {case}')
            entries[case], helpers[case] = entry, loader
    manifest = {'pid': os.getpid(), 'expert_identity': identity,
                'script_sha256': sha256_file(__file__), 'total_cases': len(entries),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'cache_dir': str(args.cache_dir), 'resident_sequence_encoder': True,
                'device': str(device), 'sequence_cache_dir': str(sequence_root)}
    atomic_json(manifest, output / 'manifest.json')
    completed, failed = set(), {}
    log = (output / 'cases.jsonl').open('a', buffering=1)
    started = time.monotonic()
    while len(completed) + len(failed) < len(entries):
        progress_made = False
        for path in sorted(args.cache_dir.glob('*.pt')):
            case = path.stem
            if case not in entries or case in completed or case in failed:
                continue
            before = time.monotonic()
            try:
                cached = torch.load(path, map_location='cpu', weights_only=False)
                if cached.get('cache_version') != features._FEATURIZE_CACHE_VERSION:
                    raise ValueError('official writer requires current feature schema')
                if not features._esm_identity_verified(cached.get('esm_encoder_identity', {})):
                    raise ValueError('official writer requires verified feature encoder identity')
                data = tuple(features._to_device_and_batch(cached[key], str(device))
                             for key in ('e3_data', 'poi_data', 'ligand_data'))
                helpers[case]._prepare_matcha(entries[case], data)
                completed.add(case)
                record = {'case': case, 'status': 'ok'}
            except Exception as error:
                failed[case] = f'{type(error).__name__}: {error}'
                record = {'case': case, 'status': 'failed', 'error': failed[case]}
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            record['seconds'] = time.monotonic() - before
            log.write(json.dumps(record) + '\n')
            atomic_json({**manifest, 'status': 'running', 'completed': len(completed),
                         'failed': failed, 'elapsed_seconds': time.monotonic() - started},
                        output / 'progress.json')
            print(json.dumps(record), flush=True)
            progress_made = True
        if not progress_made:
            time.sleep(2)
    atomic_json({**manifest, 'status': 'complete_with_failures' if failed else 'complete',
                 'completed': len(completed), 'failed': failed}, output / 'progress.json')


if __name__ == '__main__':
    main()

"""Evaluate one frozen structural ranker on a sealed labeled archive dataset."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import tricoopdiff.diffdock_compat  # noqa: E402,F401
import torch

from train_structural_ranker import evaluate  # noqa: E402
from tricoopdiff.structural_ranker import StructuralQualityRanker  # noqa: E402
from tricoopdiff.structural_ranking_data import (  # noqa: E402
    StructuralArchiveDataset, file_digest,
)
from tricoopdiff.structural_ranking_runtime import checkpoint_capabilities  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--available-only', action='store_true',
                        help='Evaluate the sealed archive/label intersection and report partial coverage')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('use a new output path')

    checkpoint_digest = file_digest(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != 'structural_quality_ranker':
        raise ValueError('unsupported ranker checkpoint architecture')
    capabilities = checkpoint_capabilities(checkpoint)
    dataset = StructuralArchiveDataset(args.archives, args.labels, args.split)
    manifest_cases = list(dataset.cases)
    if args.available_only:
        dataset.cases = [case for case in manifest_cases
                         if (args.archives / f'{case}.pt').is_file()
                         and (args.archives / f'{case}.complete.json').is_file()
                         and (args.labels / f'{case}.json').is_file()]
        if not dataset.cases:
            raise ValueError('no sealed archive/label pairs are currently available')
    model = StructuralQualityRanker(**checkpoint['config']).to(args.device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    started = time.monotonic()
    metrics = evaluate(model, dataset, args.device,
                       future_enabled=capabilities['future'])
    if file_digest(args.checkpoint) != checkpoint_digest:
        raise ValueError('checkpoint changed during evaluation')
    report = {
        'scope': 'fixed frozen ranker evaluation; no checkpoint selection',
        'split': args.split,
        'checkpoint_sha256': checkpoint_digest,
        'archive_manifest_sha256': dataset.manifest_digest,
        'label_manifest_sha256': dataset.label_manifest_digest,
        'cases': len(dataset),
        'manifest_cases': len(manifest_cases),
        'partial_coverage': len(dataset) != len(manifest_cases),
        'seconds': time.monotonic() - started,
        'metrics': metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.out)
    print(json.dumps({
        'cases': report['cases'],
        'seconds': report['seconds'],
        'initial': metrics['initial'],
        'final': metrics['final'],
        'union': metrics['union'],
    }, indent=2))


if __name__ == '__main__':
    main()

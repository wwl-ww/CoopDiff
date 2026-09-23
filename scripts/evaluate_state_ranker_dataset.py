"""Evaluate one frozen legacy state-feature ranker on a labeled archive."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import torch

from evaluation_common import sha256_file
from train_state_ranker import evaluate, load_split
from tricoopdiff.pose_ranking import StateQualityRanker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('use a new output path')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != 'state_quality_ranker_v1':
        raise ValueError('unsupported ranker checkpoint architecture')
    manifest, cases = load_split(
        args.archives, args.labels, args.split, checkpoint['geometry_only'])
    model = StateQualityRanker(**checkpoint['model_config']).to(args.device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    started = time.monotonic()
    metrics = evaluate(model, cases, checkpoint['mean'], checkpoint['scale'], args.device)
    report = {
        'scope': 'fixed frozen legacy state-ranker evaluation; no checkpoint selection',
        'split': args.split,
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'archive_manifest_sha256': sha256_file(args.archives / 'manifest.json'),
        'cases': len(cases),
        'seconds': time.monotonic() - started,
        'runtime_matches_training': checkpoint['generator'].get('runtime_sha256') == manifest.get('runtime_sha256'),
        'metrics': metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('cases', 'seconds', 'runtime_matches_training')}, indent=2))


if __name__ == '__main__':
    main()

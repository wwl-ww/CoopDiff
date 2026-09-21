"""Fixed-checkpoint train/validation diagnostic, never selects a new model."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import tricoopdiff.diffdock_compat  # noqa: E402,F401
import torch
from train_structural_ranker import evaluate
from tricoopdiff.structural_ranker import StructuralQualityRanker
from tricoopdiff.structural_ranking_data import StructuralArchiveDataset, audit_datasets, file_digest
from tricoopdiff.structural_ranking_runtime import checkpoint_capabilities


def evaluation_options(checkpoint):
    """Never evaluate an untrained future head as a usable selector."""
    capabilities = checkpoint_capabilities(checkpoint)
    config = checkpoint['metadata']['arguments']
    return {'future_enabled': capabilities['future'],
            'future_target': config.get('future_target', 'final')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('use a new diagnostic report')
    digest = file_digest(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['metadata']['arguments']
    options = evaluation_options(checkpoint)
    if config['candidate_scope'] != 'archived-pairs' or len(config['train_archives']) != 1:
        raise ValueError('this diagnostic requires one full-pool training shard')
    train = StructuralArchiveDataset(config['train_archives'][0], config['train_labels'][0], 'train')
    val = StructuralArchiveDataset(config['val_archives'], config['val_labels'], 'val')
    audit_datasets(train, val)
    for name, dataset in (('train', train), ('val', val)):
        if dict(dataset.seals) != checkpoint['metadata'][name + '_case_seals']:
            raise ValueError('diagnostic data differs from training snapshot')
    model = StructuralQualityRanker(**checkpoint['config']).to(args.device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    started = time.monotonic()
    report = {'checkpoint_sha256': digest, 'epoch': checkpoint['epoch'],
              'scope': 'fixed epoch train-versus-reused-validation diagnostic, not a new held-out test',
              'quality_walker_scope': config['quality_walker_scope']}
    for name, dataset in (('train', train), ('val', val)):
        print(f'EVALUATE {name} cases={len(dataset)}', flush=True)
        report[name] = evaluate(model, dataset, args.device, **options)
    report['seconds'] = time.monotonic() - started
    if file_digest(args.checkpoint) != digest:
        raise ValueError('checkpoint changed during diagnostic')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.out)
    print(f'DONE {args.out}', flush=True)


if __name__ == '__main__':
    main()

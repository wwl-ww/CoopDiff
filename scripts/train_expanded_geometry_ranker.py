"""Train a post-ranker on native-free C-alpha plus heavy-atom geometry."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import torch

from evaluation_common import sha256_file
from extract_expanded_ranking_features import PROTOCOL
from train_state_ranker import combine_training_parts, evaluate, quality_class_support
from tricoopdiff.pose_ranking import StateQualityRanker, state_quality_loss


def load_split(archives, feature_dir, labels, split):
    archive_manifest_path = archives / 'manifest.json'
    archive_manifest = json.loads(archive_manifest_path.read_text())
    feature_manifest = json.loads((feature_dir / 'manifest.json').read_text())
    label_manifest = json.loads((labels / 'manifest.json').read_text())
    if (feature_manifest['protocol'] != PROTOCOL
            or feature_manifest['archive_manifest_sha256'] != sha256_file(archive_manifest_path)
            or label_manifest['archive_manifest_sha256'] != sha256_file(archive_manifest_path)):
        raise ValueError('expanded features/labels do not match archive manifest')
    cases = []
    for case in archive_manifest['cases']:
        archive_path = archives / f'{case}.pt'
        archive = torch.load(archive_path, map_location='cpu', weights_only=False)
        feature = torch.load(feature_dir / f'{case}.pt', map_location='cpu', weights_only=False)
        label = json.loads((labels / f'{case}.json').read_text())
        digest = sha256_file(archive_path)
        if (archive['split'] != split or feature['archive_sha256'] != digest
                or label['archive_sha256'] != digest or feature['case'] != case
                or label['case'] != case):
            raise ValueError('expanded ranker split/archive identity mismatch')
        truth = {}
        for row in label['labels']:
            truth[row['pool_index'], 'initial'] = float(row['initial']['dockq'])
            if row['valid']:
                truth[row['pool_index'], 'final'] = float(row['final']['dockq'])
        rows = feature['rows']
        keys = [(row['pool_index'], row['stage']) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) != set(truth):
            raise ValueError('expanded feature rows do not exactly cover labels')
        matrix = torch.from_numpy(np.stack([row['features'] for row in rows])).float()
        matrix = torch.log1p(matrix)
        dockq = torch.tensor([truth[key] for key in keys]).float()
        if (matrix.shape != (len(rows), len(feature_manifest['feature_names']))
                or not torch.isfinite(matrix).all() or (matrix < 0).any()):
            raise ValueError('invalid expanded feature matrix')
        cases.append({'case': case, 'features': matrix, 'dockq': dockq,
                      'stages': [row['stage'] for row in rows]})
    return archive_manifest, cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('train-archives', 'train-features', 'train-labels', 'val-archives',
                'val-features', 'val-labels', 'split-audit', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--additional-train-archives', type=Path, nargs='*', default=[])
    parser.add_argument('--additional-train-features', type=Path, nargs='*', default=[])
    parser.add_argument('--additional-train-labels', type=Path, nargs='*', default=[])
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    counts = (len(args.additional_train_archives), len(args.additional_train_features),
              len(args.additional_train_labels))
    if len(set(counts)) != 1 or args.epochs < 1 or args.out.exists():
        parser.error('additional shards must match; use positive epochs and a new output')
    archive_dirs = [args.train_archives, *args.additional_train_archives]
    feature_dirs = [args.train_features, *args.additional_train_features]
    label_dirs = [args.train_labels, *args.additional_train_labels]
    loaded = [load_split(*part, 'train') for part in zip(
        archive_dirs, feature_dirs, label_dirs)]
    train_manifest, train = combine_training_parts(loaded)
    val_manifest, val = load_split(
        args.val_archives, args.val_features, args.val_labels, 'val')
    audit = json.loads(args.split_audit.read_text())
    if [case['case'] for case in train] != audit['train_cases']:
        raise ValueError('expanded ranker training order differs from split audit')
    if {case['case'] for case in val} != set(audit['val_cases']):
        raise ValueError('expanded ranker validation differs from split audit')
    if train_manifest['checkpoint_sha256'] != val_manifest['checkpoint_sha256']:
        raise ValueError('expanded ranker generator checkpoint mismatch')
    if args.train_limit is not None:
        if not 1 <= args.train_limit <= len(train):
            raise ValueError('invalid expanded ranker training prefix')
        train = train[:args.train_limit]
    random.seed(args.seed); torch.manual_seed(args.seed)
    values = torch.cat([case['features'] for case in train])
    mean, scale = values.mean(0), values.std(0).clamp_min(.01)
    model = StateQualityRanker(values.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        for index in order:
            case = train[index]
            prediction = model(((case['features'] - mean) / scale).to(args.device))
            loss = state_quality_loss(prediction, case['dockq'].to(args.device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        metrics = evaluate(model, val, mean, scale, args.device)
        history.append({'epoch': epoch, 'validation_loss': metrics['loss']})
        if best is None or metrics['loss'] < best['metrics']['loss']:
            best = {'epoch': epoch, 'metrics': metrics,
                    'state_dict': {name: value.detach().cpu().clone()
                                   for name, value in model.state_dict().items()}}
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        'architecture': 'expanded_geometry_ranker_v1', 'model_config': model.config,
        'state_dict': best['state_dict'], 'mean': mean, 'scale': scale,
        'best_epoch': best['epoch'], 'generator_checkpoint_sha256':
            train_manifest['checkpoint_sha256'],
        'split_audit_sha256': sha256_file(args.split_audit),
        'training_case_ids': [case['case'] for case in train],
    }
    torch.save(checkpoint, args.out / 'best.pt')
    (args.out / 'report.json').write_text(json.dumps({
        'scope': 'internal heavy-atom post-ranker; validation reused for epoch selection',
        'features': '14 C-alpha plus 37 native-free heavy-atom summaries',
        'train_cases': len(train), 'val_cases': len(val), 'seed': args.seed,
        'epochs': args.epochs, 'best_epoch': best['epoch'], 'metrics': best['metrics'],
        'label_support': {'train': quality_class_support(train),
                          'val': quality_class_support(val)}, 'history': history,
    }, indent=2) + '\n')
    print(json.dumps(best['metrics']['leaderboards'], indent=2))


if __name__ == '__main__':
    main()

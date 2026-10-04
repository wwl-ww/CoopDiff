"""Fit a heavy-atom post-ranker with a ranking-only objective ablation."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch

from evaluation_common import sha256_file
from train_expanded_geometry_ranker import load_split
from train_state_ranker import combine_training_parts, evaluate, quality_class_support
from tricoopdiff.ranking.ranking_inference import StateQualityRanker, state_ranking_only_loss


@torch.no_grad()
def evaluate_objective(model, cases, mean, scale, device, temperature):
    metrics = evaluate(model, cases, mean, scale, device)
    losses = []
    for case in cases:
        output = model(((case['features'] - mean) / scale).to(device))
        losses.append(float(state_ranking_only_loss(
            output, case['dockq'].to(device), temperature)))
    metrics['loss'] = sum(losses) / len(losses)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('train-archives', 'train-features', 'train-labels', 'val-archives',
                'val-features', 'val-labels', 'split-audit', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--additional-train-archives', type=Path, nargs='*', default=[])
    parser.add_argument('--additional-train-features', type=Path, nargs='*', default=[])
    parser.add_argument('--additional-train-labels', type=Path, nargs='*', default=[])
    parser.add_argument('--train-limit', type=int, default=160)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--temperature', type=float, choices=(.01, .05), required=True)
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
        raise ValueError('objective-ablation training order differs from split audit')
    if {case['case'] for case in val} != set(audit['val_cases']):
        raise ValueError('objective-ablation validation differs from split audit')
    if train_manifest['checkpoint_sha256'] != val_manifest['checkpoint_sha256']:
        raise ValueError('objective-ablation generator checkpoint mismatch')
    if not 1 <= args.train_limit <= len(train):
        raise ValueError('invalid objective-ablation training prefix')
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
            output = model(((case['features'] - mean) / scale).to(args.device))
            loss = state_ranking_only_loss(
                output, case['dockq'].to(args.device), args.temperature)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        metrics = evaluate_objective(
            model, val, mean, scale, args.device, args.temperature)
        history.append({'epoch': epoch, 'validation_objective': metrics['loss']})
        if best is None or metrics['loss'] < best['metrics']['loss']:
            best = {'epoch': epoch, 'metrics': metrics,
                    'state_dict': {name: value.detach().cpu().clone()
                                   for name, value in model.state_dict().items()}}
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save({
        'architecture': 'expanded_geometry_ranker_ranking_only_v1',
        'model_config': model.config, 'state_dict': best['state_dict'],
        'mean': mean, 'scale': scale, 'temperature': args.temperature,
        'best_epoch': best['epoch'],
        'generator_checkpoint_sha256': train_manifest['checkpoint_sha256'],
        'split_audit_sha256': sha256_file(args.split_audit),
        'training_case_ids': [case['case'] for case in train],
    }, args.out / 'best.pt')
    (args.out / 'report.json').write_text(json.dumps({
        'scope': 'internal loss ablation; validation reused for epoch selection',
        'features': 'same sealed 51D expanded geometry',
        'objective': 'within-case listwise ranking only',
        'temperature': args.temperature, 'train_cases': len(train),
        'val_cases': len(val), 'seed': args.seed, 'epochs': args.epochs,
        'best_epoch': best['epoch'], 'metrics': best['metrics'],
        'label_support': {'train': quality_class_support(train),
                          'val': quality_class_support(val)}, 'history': history,
    }, indent=2) + '\n')
    print(json.dumps(best['metrics']['leaderboards'], indent=2))


if __name__ == '__main__':
    main()

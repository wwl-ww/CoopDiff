#!/usr/bin/env python
"""Train a set-aware 256-to-64 selector against future refined value."""

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
from tricoopdiff.initial_selector import (future_selector_retention,
                                          initial_pool_features)
from tricoopdiff.setwise_ranking import (SetwiseCandidateRanker,
                                         setwise_quality_loss,
                                         source_indices)


def load_cases(archive_dirs, label_dirs, split, require_full_pool=False):
    """Expose all 256 initial states while supervising only refined pairs."""
    if len(archive_dirs) != len(label_dirs) or not archive_dirs:
        raise ValueError('archive and label shard lists must match')
    cases, seen, manifests = [], set(), []
    for archives, labels in zip(archive_dirs, label_dirs):
        archive_manifest_path = archives / 'manifest.json'
        archive_manifest = json.loads(archive_manifest_path.read_text())
        label_manifest_path = labels / 'manifest.json'
        label_manifest = json.loads(label_manifest_path.read_text())
        if label_manifest['archive_manifest_sha256'] != sha256_file(archive_manifest_path):
            raise ValueError('setwise label/archive manifest mismatch')
        manifests.append({'archive': sha256_file(archive_manifest_path),
                          'labels': sha256_file(label_manifest_path)})
        for case in archive_manifest['cases']:
            pdb = case.split('_')[0].upper()
            if pdb in seen:
                raise ValueError('duplicate PDB across setwise-selector shards')
            seen.add(pdb)
            archive_path = archives / f'{case}.pt'
            archive = torch.load(archive_path, map_location='cpu', weights_only=False)
            label = json.loads((labels / f'{case}.json').read_text())
            if (archive['case'] != case or archive['split'] != split
                    or label['case'] != case
                    or label['archive_sha256'] != sha256_file(archive_path)):
                raise ValueError('setwise selector split/archive identity mismatch')
            pool = initial_pool_features(archive)
            pairs = {pair['pool_index']: pair for pair in archive['pairs']}
            truth = {row['pool_index']: row for row in label['labels']}
            if len(pairs) != len(archive['pairs']) or set(pairs) != set(truth):
                raise ValueError('future labels must exactly cover refined pairs')
            if require_full_pool and set(pairs) != set(range(256)):
                raise ValueError('setwise validation requires all 256 refined candidates')
            targets = {name: torch.zeros(256) for name in ('initial', 'final', 'union')}
            supervised = torch.zeros(256, dtype=torch.bool)
            for index, pair in pairs.items():
                row = truth[index]
                if pair['source'] != pool[index]['source'] or pair['valid'] != row['valid']:
                    raise ValueError('setwise source/validity mismatch')
                initial = float(row['initial']['dockq'])
                final = float(row['final']['dockq']) if row['valid'] else 0.
                union = float(row['union_dockq'])
                if abs(union - (max(initial, final) if row['valid'] else initial)) > 1e-7:
                    raise ValueError('invalid setwise union DockQ label')
                targets['initial'][index] = initial
                targets['final'][index] = final
                targets['union'][index] = union
                supervised[index] = True
            matrix = torch.log1p(torch.from_numpy(
                np.asarray([row['geometry'] for row in pool])).float())
            if matrix.shape != (256, 14) or not torch.isfinite(matrix).all():
                raise ValueError('invalid complete setwise initial geometry')
            cases.append({
                'case': case, 'features': matrix,
                'sources': [row['source'] for row in pool],
                'supervision_mask': supervised, **targets,
            })
    return cases, manifests


def model_inputs(case, mean, scale, device):
    features = ((case['features'] - mean) / scale).to(device)
    sources = source_indices(case['sources'], device=device)
    return features, sources


@torch.no_grad()
def evaluate(model, cases, mean, scale, target, device):
    model.eval()
    rows, losses = [], []
    for case in cases:
        features, sources = model_inputs(case, mean, scale, device)
        prediction = model(features, sources)
        labels = case[target].to(device)
        losses.append(float(setwise_quality_loss(
            prediction, labels, supervision_mask=case['supervision_mask'].to(device))))
        rows.append({'case': case['case'], **future_selector_retention(
            prediction['score'].cpu(), case['initial'], case['final'],
            case['sources'])})
    result = {'loss': sum(losses) / len(losses), 'cases': len(rows), 'rows': rows}
    for stage in ('initial', 'final', 'union'):
        defined = [row[stage]['oracle_retention'] for row in rows
                   if row[stage]['oracle_retention'] is not None]
        result[stage] = {
            'mean_selected_best': sum(row[stage]['selected_best'] for row in rows) / len(rows),
            'mean_full_pool_oracle': sum(row[stage]['full_pool_oracle'] for row in rows) / len(rows),
            'mean_case_oracle_retention': sum(defined) / len(defined) if defined else None,
            'selected_hits_0.23': sum(row[stage]['selected_hit'] for row in rows),
            'full_pool_hits_0.23': sum(row[stage]['full_pool_hit'] for row in rows),
            'oracle_index_retained': sum(row[stage]['oracle_index_retained'] for row in rows),
        }
    result['rescued_cases_available'] = sum(row['rescued_candidates'] > 0 for row in rows)
    result['rescued_cases_retained'] = sum(row['rescued_hit_retained'] for row in rows)
    result['mean_selected_bitseed'] = sum(
        row['selected_source_counts']['bitseed'] for row in rows) / len(rows)
    result['mean_selected_sdpi'] = sum(
        row['selected_source_counts']['sdpi'] for row in rows) / len(rows)
    return result


def selection_key(metrics, target):
    """Checkpoint order aligned with allocation quality, not calibration loss."""
    stage = metrics[target]
    return (stage['mean_selected_best'], stage['selected_hits_0.23'], -metrics['loss'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-archives', type=Path, nargs='+', required=True)
    parser.add_argument('--train-labels', type=Path, nargs='+', required=True)
    parser.add_argument('--val-archives', type=Path, nargs='+', required=True)
    parser.add_argument('--val-labels', type=Path, nargs='+', required=True)
    parser.add_argument('--split-audit', type=Path, required=True)
    parser.add_argument('--target', choices=('final', 'union'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model-dim', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.lr <= 0 or args.out.exists():
        parser.error('positive epochs/lr and a new output directory are required')

    train, train_manifests = load_cases(
        args.train_archives, args.train_labels, 'train')
    val, val_manifests = load_cases(
        args.val_archives, args.val_labels, 'val', require_full_pool=True)
    audit = json.loads(args.split_audit.read_text())
    if [case['case'] for case in train] != audit['train_cases']:
        raise ValueError('setwise-selector training order differs from approved split')
    if {case['case'] for case in val} != set(audit['val_cases']):
        raise ValueError('setwise-selector validation differs from approved split')
    if args.train_limit is not None:
        if not 1 <= args.train_limit <= len(train):
            raise ValueError('invalid setwise-selector training prefix')
        train = train[:args.train_limit]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    values = torch.cat([case['features'] for case in train])
    mean, scale = values.mean(0), values.std(0).clamp_min(.01)
    model = SetwiseCandidateRanker(
        values.shape[1], model_dim=args.model_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, feedforward_dim=2 * args.model_dim,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        train_loss = 0.
        for index in order:
            case = train[index]
            features, sources = model_inputs(case, mean, scale, args.device)
            prediction = model(features, sources)
            loss = setwise_quality_loss(
                prediction, case[args.target].to(args.device),
                supervision_mask=case['supervision_mask'].to(args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            train_loss += float(loss.detach())
        metrics = evaluate(model, val, mean, scale, args.target, args.device)
        history.append({
            'epoch': epoch, 'train_loss': train_loss / len(train),
            'validation_loss': metrics['loss'],
            'selection_key': list(selection_key(metrics, args.target)),
            'target_metrics': metrics[args.target],
        })
        if best is None or selection_key(metrics, args.target) > best['selection_key']:
            best = {
                'epoch': epoch, 'metrics': metrics,
                'selection_key': selection_key(metrics, args.target),
                'state_dict': {name: value.detach().cpu().clone()
                               for name, value in model.state_dict().items()},
            }

    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        'architecture': 'setwise_future_selector', 'target': args.target,
        'model_config': model.config, 'state_dict': best['state_dict'],
        'mean': mean, 'scale': scale, 'pool_size': 256, 'selection_budget': 64,
        'best_epoch': best['epoch'], 'selection_key': list(best['selection_key']),
        'generator_checkpoint_sha256': json.loads(
            (args.train_archives[0] / 'manifest.json').read_text())['checkpoint_sha256'],
        'train_case_ids': [case['case'] for case in train],
        'split_audit_sha256': sha256_file(args.split_audit),
        'input_manifests': {'train': train_manifests, 'val': val_manifests},
    }
    torch.save(checkpoint, args.out / 'best.pt')
    report = {
        'scope': 'internal set-aware future-value selector; not benchmark top-1',
        'architecture': ('permutation-equivariant self-attention over each case; '
                         'explicit source embeddings; no pool positions'),
        'training_context': ('train and validation both expose all 256 initial states; '
                             'training loss is masked to the 64 refined labels'),
        'checkpoint_selection': ('maximize validation target selected-best, then >=0.23 '
                                 'hits, then minimize composite loss'),
        'target': args.target, 'seed': args.seed, 'epochs': args.epochs,
        'train_cases': len(train), 'val_cases': len(val),
        'best_epoch': best['epoch'], 'metrics': best['metrics'], 'history': history,
    }
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: best['metrics'][key] for key in
                      ('initial', 'final', 'union', 'rescued_cases_available',
                       'rescued_cases_retained')}, indent=2))


if __name__ == '__main__':
    main()

"""Train an initial-geometry selector against C40 final or initial/final-union value."""

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
from tricoopdiff.ranking.initial_selector import future_selector_retention, initial_pool_features
from tricoopdiff.ranking.ranking_inference import StateQualityRanker, state_quality_loss


def load_cases(archive_dirs, label_dirs, split, require_full_pool=False):
    """Join sealed C40 pairs to labels without exposing final-state features."""
    if len(archive_dirs) != len(label_dirs) or not archive_dirs:
        raise ValueError('archive and label shard lists must match')
    cases, seen, manifests = [], set(), []
    for archives, labels in zip(archive_dirs, label_dirs):
        archive_manifest_path = archives / 'manifest.json'
        archive_manifest = json.loads(archive_manifest_path.read_text())
        label_manifest_path = labels / 'manifest.json'
        label_manifest = json.loads(label_manifest_path.read_text())
        if label_manifest['archive_manifest_sha256'] != sha256_file(archive_manifest_path):
            raise ValueError('future label/archive manifest mismatch')
        manifests.append({'archive': sha256_file(archive_manifest_path),
                          'labels': sha256_file(label_manifest_path)})
        for case in archive_manifest['cases']:
            pdb = case.split('_')[0].upper()
            if pdb in seen:
                raise ValueError('duplicate PDB across future-selector shards')
            seen.add(pdb)
            archive_path = archives / f'{case}.pt'
            archive = torch.load(archive_path, map_location='cpu', weights_only=False)
            label = json.loads((labels / f'{case}.json').read_text())
            if (archive['case'] != case or archive['split'] != split
                    or label['case'] != case or label['archive_sha256'] != sha256_file(archive_path)):
                raise ValueError('future selector split/archive identity mismatch')
            pool_rows = initial_pool_features(archive)
            pool_by_index = {row['pool_index']: row for row in pool_rows}
            pairs = {pair['pool_index']: pair for pair in archive['pairs']}
            labels_by_index = {row['pool_index']: row for row in label['labels']}
            if len(pairs) != len(archive['pairs']) or set(pairs) != set(labels_by_index):
                raise ValueError('future labels must exactly cover archived refinement pairs')
            if require_full_pool and set(pairs) != set(range(256)):
                raise ValueError('future-selector validation requires all 256 refined candidates')
            features, initial, final, union, sources = [], [], [], [], []
            for index in sorted(pairs):
                pair, truth, pool = pairs[index], labels_by_index[index], pool_by_index[index]
                if pair['source'] != pool['source'] or truth['valid'] != pair['valid']:
                    raise ValueError('future selector source/validity mismatch')
                initial_q = float(truth['initial']['dockq'])
                final_q = float(truth['final']['dockq']) if truth['valid'] else 0.
                union_q = float(truth['union_dockq'])
                expected_union = max(initial_q, final_q) if truth['valid'] else initial_q
                if abs(union_q - expected_union) > 1e-7:
                    raise ValueError('invalid union DockQ label')
                features.append(pool['geometry'])
                initial.append(initial_q); final.append(final_q); union.append(union_q)
                sources.append(pool['source'])
            matrix = torch.log1p(torch.from_numpy(np.asarray(features)).float())
            values = {name: torch.tensor(rows).float()
                      for name, rows in [('initial', initial), ('final', final), ('union', union)]}
            if matrix.shape != (len(pairs), 14) or not torch.isfinite(matrix).all():
                raise ValueError('invalid future-selector initial geometry')
            cases.append({'case': case, 'features': matrix, **values, 'sources': sources})
    return cases, manifests


@torch.no_grad()
def evaluate(model, cases, mean, scale, target, device):
    model.eval(); rows, losses = [], []
    for case in cases:
        prediction = model(((case['features'] - mean) / scale).to(device))
        losses.append(float(state_quality_loss(prediction, case[target].to(device))))
        rows.append({'case': case['case'], **future_selector_retention(
            prediction['q_hat'].cpu(), case['initial'], case['final'], case['sources'])})
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-archives', type=Path, nargs='+', required=True)
    parser.add_argument('--train-labels', type=Path, nargs='+', required=True)
    parser.add_argument('--val-archives', type=Path, nargs='+', required=True)
    parser.add_argument('--val-labels', type=Path, nargs='+', required=True)
    parser.add_argument('--split-audit', type=Path, required=True)
    parser.add_argument('--target', choices=('final', 'union'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.out.exists():
        parser.error('positive epochs and a new output directory are required')
    train, train_manifests = load_cases(args.train_archives, args.train_labels, 'train')
    val, val_manifests = load_cases(args.val_archives, args.val_labels, 'val', require_full_pool=True)
    audit = json.loads(args.split_audit.read_text())
    if [case['case'] for case in train] != audit['train_cases']:
        raise ValueError('future-selector training order differs from approved split')
    if {case['case'] for case in val} != set(audit['val_cases']):
        raise ValueError('future-selector validation differs from approved split')
    if args.train_limit is not None:
        if not 1 <= args.train_limit <= len(train):
            raise ValueError('invalid future-selector training prefix')
        train = train[:args.train_limit]
    random.seed(args.seed); torch.manual_seed(args.seed)
    values = torch.cat([case['features'] for case in train])
    mean, scale = values.mean(0), values.std(0).clamp_min(.01)
    model = StateQualityRanker(14).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        for index in order:
            case = train[index]
            output = model(((case['features'] - mean) / scale).to(args.device))
            loss = state_quality_loss(output, case[args.target].to(args.device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        metrics = evaluate(model, val, mean, scale, args.target, args.device)
        history.append({'epoch': epoch, 'validation_loss': metrics['loss']})
        if best is None or metrics['loss'] < best['metrics']['loss']:
            best = {'epoch': epoch, 'metrics': metrics,
                    'state_dict': {name: value.detach().cpu().clone()
                                   for name, value in model.state_dict().items()}}
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save({
        'architecture': 'future_geometry_selector_v1', 'target': args.target,
        'generator_checkpoint_sha256': json.loads(
            (args.train_archives[0] / 'manifest.json').read_text())['checkpoint_sha256'],
        'model_config': model.config, 'state_dict': best['state_dict'], 'mean': mean,
        'scale': scale, 'pool_size': 256, 'selection_budget': 64,
        'best_epoch': best['epoch'], 'train_case_ids': [case['case'] for case in train],
        'split_audit_sha256': sha256_file(args.split_audit),
        'input_manifests': {'train': train_manifests, 'val': val_manifests},
    }, args.out / 'best.pt')
    (args.out / 'report.json').write_text(json.dumps({
        'scope': 'internal C40 future-value selector; full-pool internal validation',
        'warning': 'training pairs are source-balanced random subsets; validation covers all 256',
        'target': args.target, 'seed': args.seed, 'epochs': args.epochs,
        'train_cases': len(train), 'val_cases': len(val), 'best_epoch': best['epoch'],
        'metrics': best['metrics'], 'history': history,
    }, indent=2) + '\n')
    print(json.dumps({key: best['metrics'][key] for key in
                      ('initial', 'final', 'union', 'rescued_cases_available',
                       'rescued_cases_retained')}, indent=2))


if __name__ == '__main__':
    main()

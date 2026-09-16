"""Fit the frozen-archive R1 quality baseline; no benchmark labels are accepted."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch

from evaluation_common import sha256_file
from tricoopdiff.pose_ranking import StateQualityRanker, state_quality_loss
from tricoopdiff.ranking_audit import join_labels
from tricoopdiff.ranking_inference import GENERATOR_KEYS, archive_rows, export_quality_checkpoint, feature_vector


def load_split(archives, labels, split, geometry_only=False):
    manifest = json.loads((archives / 'manifest.json').read_text())
    label_manifest = json.loads((labels / 'manifest.json').read_text())
    if label_manifest['archive_manifest_sha256'] != sha256_file(archives / 'manifest.json'):
        raise ValueError('label/archive manifest identity mismatch')
    names = manifest['cases']
    if (not names or len(set(names)) != len(names)
            or any(c != Path(c).name or c in ('.', '..') for c in names)):
        raise ValueError('empty, unsafe or duplicate case list')
    cases = []
    for case in names:
        archive_path, label_path = archives / f'{case}.pt', labels / f'{case}.json'
        label = json.loads(label_path.read_text())
        receipt = json.loads((archives / f'{case}.complete.json').read_text())
        digest = sha256_file(archive_path)
        if label['archive_sha256'] != digest or receipt['archive_sha256'] != digest:
            raise ValueError('label/archive identity mismatch')
        archive = torch.load(archive_path, map_location='cpu', weights_only=False)
        if archive['case'] != case or archive['manifest'] != manifest:
            raise ValueError('embedded archive identity mismatch')
        if archive['split'] != split or split not in ('train', 'val'):
            raise ValueError('training accepts only explicitly split internal archives')
        rows = archive_rows(archive)
        joined = join_labels({'case': case, 'archive_sha256': digest, 'rows': [
            {**{k: row[k] for k in ('pool_index', 'stage', 'source')}, 'score': 0.} for row in rows]}, label)
        truth = {(row['pool_index'], row['stage']): row['dockq'] for row in joined}
        feature_rows, dockq, stages = [], [], []
        frozen_dim = archive.get('probe_feature_dim') or next((len(f['frozen']) for p in archive['pairs']
                           for f in (p['initial_features'], p['final_features'])
                           if f is not None and f['probe_valid']), None)
        if not geometry_only and frozen_dim is None:
            raise ValueError(f'{case}: no usable frozen features; use geometry-only diagnostic')
        for pair in archive['pairs']:
            for stage in ('initial', 'final'):
                if stage == 'final' and not pair['valid']:
                    continue
                features = pair[f'{stage}_features']
                vector = feature_vector(features, frozen_dim, geometry_only)
                feature_rows.append(vector)
                dockq.append(truth[pair['pool_index'], stage])
                stages.append(stage)
        matrix, target = torch.stack(feature_rows), torch.tensor(dockq).float()
        if not torch.isfinite(matrix).all() or not torch.isfinite(target).all():
            raise ValueError(f'{case}: non-finite data')
        cases.append({'case': case, 'features': matrix, 'dockq': target, 'stages': stages})
    return manifest, cases


def quality_class_support(cases):
    values = torch.cat([case['dockq'] for case in cases])
    counts = [int((values < .23).sum()), int(((values >= .23) & (values < .49)).sum()),
              int((values >= .49).sum())]
    return {'cases': len(cases), 'states': len(values), 'class_counts_lt23_23to49_ge49': counts,
            'classes_with_no_labels': [i for i, count in enumerate(counts) if not count],
            'cases_with_ge23': sum(bool((case['dockq'] >= .23).any()) for case in cases),
            'cases_with_ge49': sum(bool((case['dockq'] >= .49).any()) for case in cases),
            'warning': 'states from one case are correlated; low loss alone does not prove top1 quality'}


def combine_training_parts(parts):
    """Join separately sealed case shards, never rewrite their source manifests."""
    if not parts:
        raise ValueError('at least one training shard is required')
    manifest = parts[0][0]
    cases, seen = [], set()
    for other, rows in parts:
        for key in (*GENERATOR_KEYS, 'walkers', 'selection', 'proposal_seed', 'selection_seed', 'diffusion_seed'):
            if other[key] != manifest[key]:
                raise ValueError(f'training shard generator/sampling mismatch: {key}')
        for row in rows:
            pdb = row['case'].split('_')[0].upper()
            if pdb in seen:
                raise ValueError('duplicate training PDB across archive shards')
            seen.add(pdb)
            cases.append(row)
    return manifest, cases


@torch.no_grad()
def evaluate(model, cases, mean, scale, device):
    rows, losses = [], []
    model.eval()
    for case in cases:
        prediction = model(((case['features'] - mean) / scale).to(device))
        scores = prediction['q_hat'].cpu()
        target = case['dockq']
        losses.append(float(state_quality_loss(prediction, target.to(device))))
        stages = {}
        for stage in ('initial', 'final', 'union'):
            ids = [i for i, value in enumerate(case['stages']) if stage == 'union' or value == stage]
            if not ids:
                stages[stage] = {'top1': None, 'oracle': None, 'top5_best': None, 'count': 0}
                continue
            ids = torch.tensor(ids)
            order = ids[torch.argsort(scores[ids], descending=True, stable=True)]
            stages[stage] = {'top1': float(target[order[0]]), 'oracle': float(target[ids].max()),
                             'top5_best': float(target[order[:5]].max()), 'count': len(ids)}
        rows.append({'case': case['case'], **stages})
    summary = {}
    for stage in ('initial', 'final', 'union'):
        available = [row[stage] for row in rows if row[stage]['top1'] is not None]
        # Missing output is a failed case; retain all cases in aggregate denominators.
        summary[stage] = {'mean_top1': sum(r['top1'] for r in available) / len(rows),
                          'mean_oracle': sum(r['oracle'] for r in available) / len(rows),
                          'hits_0.23': sum(r['top1'] >= .23 for r in available),
                          'hits_0.49': sum(r['top1'] >= .49 for r in available),
                          'top5_hits_0.23': sum(r['top5_best'] >= .23 for r in available),
                          'cases': len(rows), 'missing': len(rows) - len(available)}
    return {'loss': sum(losses) / len(losses), 'leaderboards': summary, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('train-archives', 'train-labels', 'val-archives', 'val-labels', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Record train curves and last-epoch metrics without changing selection')
    parser.add_argument('--split-audit', type=Path, required=True)
    parser.add_argument('--additional-train-archives', type=Path, nargs='*', default=[])
    parser.add_argument('--additional-train-labels', type=Path, nargs='*', default=[])
    parser.add_argument('--train-limit', type=int,
                        help='Use a predeclared prefix of the complete audited training case order')
    args = parser.parse_args()
    if args.epochs < 1 or args.out.exists():
        parser.error('positive epochs and a new output directory are required')
    if len(args.additional_train_archives) != len(args.additional_train_labels):
        parser.error('each additional archive shard requires its own labels directory')
    archive_parts = [args.train_archives, *args.additional_train_archives]
    label_parts = [args.train_labels, *args.additional_train_labels]
    loaded = [load_split(a, b, 'train', args.geometry_only) for a, b in zip(archive_parts, label_parts)]
    train_manifest, train = combine_training_parts(loaded)
    val_manifest, val = load_split(args.val_archives, args.val_labels, 'val', args.geometry_only)
    audit = json.loads(args.split_audit.read_text())
    for split, cases in (('train', train), ('val', val)):
        if {c['case'] for c in cases} != set(audit[f'{split}_cases']):
            raise ValueError(f'{split}: data do not match the approved split audit')
    benchmark = set(audit['benchmark_pdbs_excluded'])
    if any(c['case'].split('_')[0].upper() in benchmark for c in train + val):
        raise ValueError('benchmark PDB cannot be used to fit or select this ranker')
    for key in GENERATOR_KEYS:
        if train_manifest[key] != val_manifest[key]:
            raise ValueError(f'train/validation generator mismatch: {key}')
    if {c['case'].split('_')[0].upper() for c in train} & {c['case'].split('_')[0].upper() for c in val}:
        raise ValueError('same-PDB training/validation leakage')
    if args.train_limit is not None:
        if not 1 <= args.train_limit <= len(train) or [c['case'] for c in train] != audit['train_cases']:
            raise ValueError('training prefix must follow the complete predeclared audit order')
        train = train[:args.train_limit]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    values = torch.cat([c['features'] for c in train])
    mean, scale = values.mean(0), values.std(0).clamp_min(.01)
    model = StateQualityRanker(values.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        for i in order:
            case = train[i]
            output = model(((case['features'] - mean) / scale).to(args.device))
            loss = state_quality_loss(output, case['dockq'].to(args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val, mean, scale, args.device)
        history.append({'epoch': epoch, 'validation_loss': metrics['loss']})
        if args.diagnostics:
            train_metrics = evaluate(model, train, mean, scale, args.device)
            history[-1].update(train_loss=train_metrics['loss'],
                               train_leaderboards=train_metrics['leaderboards'],
                               validation_leaderboards=metrics['leaderboards'])
        if best is None or metrics['loss'] < best['metrics']['loss']:
            best = {'epoch': epoch, 'metrics': metrics,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    diagnostics = None
    if args.diagnostics:
        diagnostics = {'scope': 'training fit diagnostic; last epoch is not a newly selected model',
                       'last_epoch': {'epoch': args.epochs,
                                      'train': train_metrics, 'val': metrics}}
        model.load_state_dict(best['state_dict'], strict=True)
        diagnostics['selected_epoch_train'] = evaluate(model, train, mean, scale, args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = {'architecture': 'state_quality_ranker_v1', 'model_config': model.config,
                  **best, 'mean': mean, 'scale': scale, 'history': history,
                  'geometry_only': args.geometry_only, 'generator': train_manifest,
                  'split_audit_sha256': sha256_file(args.split_audit),
                  'training_manifest_sha256': sha256_file(args.train_archives / 'manifest.json'),
                  'training_shards_sha256': [{
                      'archives': sha256_file(a / 'manifest.json'), 'labels': sha256_file(b / 'manifest.json')}
                      for a, b in zip(archive_parts, label_parts)],
                  'training_case_ids': [c['case'] for c in train],
                  'validation_manifest_sha256': sha256_file(args.val_archives / 'manifest.json')}
    torch.save(checkpoint, args.out / 'best.pt')
    torch.save(export_quality_checkpoint(checkpoint), args.out / 'inference.pt')
    (args.out / 'report.json').write_text(json.dumps({
        'scope': 'internal pipeline smoke, not untouched benchmark or generator holdout',
        'best_epoch': best['epoch'], 'features': 'geometry_only' if args.geometry_only else 'frozen_plus_geometry',
        'selection': 'minimum internal validation composite loss', 'metrics': best['metrics'],
        'train_cases': len(train), 'val_cases': len(val),
        'label_support': {'train': quality_class_support(train), 'val': quality_class_support(val)},
        'fit_diagnostics': diagnostics,
        'seed': args.seed, 'epochs': args.epochs,
        'optimizer_updates': args.epochs * len(train),
        'training_case_ids': [c['case'] for c in train],
    }, indent=2) + '\n')
    print(json.dumps(best['metrics']['leaderboards'], indent=2))


if __name__ == '__main__':
    main()

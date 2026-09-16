"""Train a geometry-only 256-to-64 initial candidate pre-selector."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch

from evaluation_common import sha256_file
from tricoopdiff.initial_selector import POOL_LABEL_VERSION, selector_retention
from tricoopdiff.pose_ranking import StateQualityRanker, state_quality_loss


def load_cases(archive_dirs, label_dirs, split):
    if len(archive_dirs) != len(label_dirs) or not archive_dirs:
        raise ValueError('archive and label shard lists must match')
    cases, seen, manifests = [], set(), []
    for archives, labels in zip(archive_dirs, label_dirs):
        archive_manifest_path = archives / 'manifest.json'
        archive_manifest = json.loads(archive_manifest_path.read_text())
        label_manifest = json.loads((labels / 'manifest.json').read_text())
        if (label_manifest.get('protocol') != POOL_LABEL_VERSION
                or label_manifest['archive_manifest_sha256'] != sha256_file(archive_manifest_path)):
            raise ValueError('initial label/archive manifest mismatch')
        manifests.append({'archive': sha256_file(archive_manifest_path),
                          'labels': sha256_file(labels / 'manifest.json')})
        for case in archive_manifest['cases']:
            pdb = case.split('_')[0].upper()
            if pdb in seen:
                raise ValueError('duplicate PDB across selector shards')
            seen.add(pdb)
            archive_path = archives / f'{case}.pt'
            digest = sha256_file(archive_path)
            label = json.loads((labels / f'{case}.json').read_text())
            archive = torch.load(archive_path, map_location='cpu', weights_only=False)
            if archive['split'] != split or label['archive_sha256'] != digest:
                raise ValueError('selector split/archive identity mismatch')
            rows = label['rows']
            if (len(rows) != 256 or [row['pool_index'] for row in rows] != list(range(256))
                    or any(row['source'] != archive['pool'][i]['source'] for i, row in enumerate(rows))):
                raise ValueError('selector rows do not exactly cover the pool')
            features = torch.tensor([row['geometry'] for row in rows]).float()
            dockq = torch.tensor([row['dockq']['dockq'] for row in rows]).float()
            if (features.shape != (256, 14) or not torch.isfinite(features).all()
                    or (features < 0).any() or not torch.isfinite(dockq).all()):
                raise ValueError('invalid selector features or labels')
            cases.append({'case': case, 'features': torch.log1p(features), 'dockq': dockq,
                          'sources': [row['source'] for row in rows]})
    return cases, manifests


@torch.no_grad()
def evaluate(model, cases, mean, scale, device):
    model.eval()
    rows, losses = [], []
    for case in cases:
        prediction = model(((case['features'] - mean) / scale).to(device))
        losses.append(float(state_quality_loss(prediction, case['dockq'].to(device))))
        rows.append({'case': case['case'], **selector_retention(
            prediction['q_hat'].cpu(), case['dockq'], case['sources'], keep=64)})
    n = len(rows)
    defined = [row['oracle_retention'] for row in rows if row['oracle_retention'] is not None]
    return {
        'loss': sum(losses) / n,
        'summary': {
            'cases': n,
            'mean_selected_best': sum(row['selected_best'] for row in rows) / n,
            'mean_full_pool_oracle': sum(row['full_pool_oracle'] for row in rows) / n,
            'mean_case_oracle_retention': sum(defined) / len(defined) if defined else None,
            'selected_hits_0.23': sum(row['selected_hit'] for row in rows),
            'full_pool_hits_0.23': sum(row['full_pool_hit'] for row in rows),
            'oracle_index_retained': sum(row['oracle_index_retained'] for row in rows),
            'mean_selected_bitseed': sum(row['selected_source_counts']['bitseed'] for row in rows) / n,
            'mean_selected_sdpi': sum(row['selected_source_counts']['sdpi'] for row in rows) / n,
        },
        'rows': rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-archives', type=Path, nargs='+', required=True)
    parser.add_argument('--train-labels', type=Path, nargs='+', required=True)
    parser.add_argument('--val-archives', type=Path, nargs='+', required=True)
    parser.add_argument('--val-labels', type=Path, nargs='+', required=True)
    parser.add_argument('--split-audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.out.exists():
        parser.error('positive epochs and a new output directory are required')
    train, train_manifests = load_cases(args.train_archives, args.train_labels, 'train')
    val, val_manifests = load_cases(args.val_archives, args.val_labels, 'val')
    audit = json.loads(args.split_audit.read_text())
    if [case['case'] for case in train] != audit['train_cases']:
        raise ValueError('selector training order differs from the approved split')
    if {case['case'] for case in val} != set(audit['val_cases']):
        raise ValueError('selector validation cases differ from the approved split')
    if args.train_limit is not None:
        if not 1 <= args.train_limit <= len(train):
            raise ValueError('invalid selector training prefix')
        train = train[:args.train_limit]
    random.seed(args.seed); torch.manual_seed(args.seed)
    values = torch.cat([case['features'] for case in train])
    mean, scale = values.mean(0), values.std(0).clamp_min(.01)
    model = StateQualityRanker(14).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train))); random.Random(args.seed + epoch).shuffle(order)
        for index in order:
            case = train[index]
            output = model(((case['features'] - mean) / scale).to(args.device))
            loss = state_quality_loss(output, case['dockq'].to(args.device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        metrics = evaluate(model, val, mean, scale, args.device)
        history.append({'epoch': epoch, 'validation_loss': metrics['loss']})
        if best is None or metrics['loss'] < best['metrics']['loss']:
            best = {'epoch': epoch, 'metrics': metrics,
                    'state_dict': {name: value.detach().cpu().clone()
                                   for name, value in model.state_dict().items()}}
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        'architecture': 'initial_geometry_selector_v1', 'model_config': model.config,
        'state_dict': best['state_dict'], 'mean': mean, 'scale': scale,
        'selection_budget': 64, 'pool_size': 256, 'best_epoch': best['epoch'],
        'selection_rule': 'minimum internal validation composite loss',
        'train_case_ids': [case['case'] for case in train],
        'split_audit_sha256': sha256_file(args.split_audit),
        'input_manifests': {'train': train_manifests, 'val': val_manifests},
    }
    torch.save(checkpoint, args.out / 'best.pt')
    (args.out / 'report.json').write_text(json.dumps({
        'scope': 'internal initial-pool pre-selector; no refined states and no benchmark fitting',
        'seed': args.seed, 'epochs': args.epochs, 'train_cases': len(train),
        'val_cases': len(val), 'best_epoch': best['epoch'],
        'selection': checkpoint['selection_rule'], 'metrics': best['metrics'],
        'history': history,
    }, indent=2) + '\n')
    print(json.dumps(best['metrics']['summary'], indent=2))


if __name__ == '__main__':
    main()

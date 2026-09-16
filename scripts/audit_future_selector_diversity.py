"""Audit future-value selectors with native-free redundancy penalties."""

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch

from train_future_selector import load_cases
from tricoopdiff.initial_selector import (diversity_aware_indices,
                                          future_selector_retention_at_indices)
from tricoopdiff.pose_ranking import StateQualityRanker


PENALTIES = (0., .025, .05, .1, .2)


@torch.no_grad()
def evaluate(checkpoint_path, cases, penalties=PENALTIES):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if (checkpoint.get('architecture') != 'future_geometry_selector_v1'
            or checkpoint.get('target') not in ('final', 'union')
            or checkpoint.get('pool_size') != 256
            or checkpoint.get('selection_budget') != 64):
        raise ValueError('unsupported future-selector checkpoint')
    model = StateQualityRanker(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True); model.eval()
    rows = []
    for case in cases:
        descriptors = (case['features'] - checkpoint['mean']) / checkpoint['scale']
        scores = model(descriptors)['q_hat']
        for penalty in penalties:
            indices = diversity_aware_indices(scores, descriptors, 64, penalty)
            rows.append({'case': case['case'], 'penalty': penalty,
                         **future_selector_retention_at_indices(
                             indices, case['initial'], case['final'], case['sources'])})
    return checkpoint['target'], rows


def summarize(rows, penalties=PENALTIES):
    summaries = []
    for penalty in penalties:
        selected = [row for row in rows if row['penalty'] == penalty]
        row = {'penalty': penalty, 'cases': len(selected)}
        for stage in ('final', 'union'):
            values = [item[stage]['oracle_retention'] for item in selected
                      if item[stage]['oracle_retention'] is not None]
            row[stage] = {
                'mean_selected_best': statistics.mean(
                    item[stage]['selected_best'] for item in selected),
                'mean_case_oracle_retention': statistics.mean(values) if values else None,
                'selected_hits_0.23': sum(
                    item[stage]['selected_hit'] for item in selected),
                'oracle_index_retained': sum(
                    item[stage]['oracle_index_retained'] for item in selected),
            }
        row['rescued_cases_retained'] = sum(
            item['rescued_hit_retained'] for item in selected)
        row['rescued_cases_available'] = sum(
            item['rescued_candidates'] > 0 for item in selected)
        summaries.append(row)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rankers', type=Path, required=True)
    parser.add_argument('--val-archives', type=Path, nargs='+', required=True)
    parser.add_argument('--val-labels', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--penalties', type=float, nargs='+', default=PENALTIES)
    args = parser.parse_args(argv)
    penalties = tuple(float(value) for value in args.penalties)
    if (not penalties or penalties[0] != 0 or len(set(penalties)) != len(penalties)
            or any(value < 0 for value in penalties) or args.out.exists()):
        parser.error('use a new output, unique nonnegative penalties, and include 0 first')
    cases, manifests = load_cases(args.val_archives, args.val_labels, 'val',
                                  require_full_pool=True)
    checkpoints = sorted(args.rankers.glob('*_n160_s*/best.pt'))
    if not checkpoints:
        raise ValueError('no future-selector checkpoints found')
    runs = []
    for checkpoint in checkpoints:
        target, rows = evaluate(checkpoint, cases, penalties)
        runs.append({'run': checkpoint.parent.name, 'target': target,
                     'summary': summarize(rows, penalties), 'rows': rows})
    payload = {
        'scope': 'internal C40 full-pool validation; no penalty frozen for deployment',
        'selection': ('predicted future value minus redundancy penalty; similarity uses '
                      'only standardized initial native-free geometry'),
        'control': 'penalty=0 exactly reproduces stable learned-score top-64',
        'penalties': list(penalties), 'validation_cases': len(cases),
        'input_manifests': manifests, 'runs': runs,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'report.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# Future-value diversity audit', '', payload['scope'], '',
             '| Run | Penalty | Final best/retention/hits | Union best/retention/hits | Rescued cases |',
             '| --- | ---: | --- | --- | ---: |']
    for run in runs:
        for row in run['summary']:
            final, union = row['final'], row['union']
            lines.append(f"| {run['run']} | {row['penalty']:.3f} | "
                         f"{final['mean_selected_best']:.4f} / "
                         f"{final['mean_case_oracle_retention']:.4f} / "
                         f"{final['selected_hits_0.23']}/{row['cases']} | "
                         f"{union['mean_selected_best']:.4f} / "
                         f"{union['mean_case_oracle_retention']:.4f} / "
                         f"{union['selected_hits_0.23']}/{row['cases']} | "
                         f"{row['rescued_cases_retained']}/{row['rescued_cases_available']} |")
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'runs': len(runs), 'cases': len(cases),
                      'penalties': list(penalties)}, indent=2))


if __name__ == '__main__':
    main()

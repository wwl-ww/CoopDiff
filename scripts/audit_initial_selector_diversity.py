"""Audit learned initial selectors with a predeclared redundancy penalty grid."""

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch

from train_initial_selector import load_cases
from tricoopdiff.initial_selector import (diversity_aware_indices,
                                          selector_retention_at_indices)
from tricoopdiff.pose_ranking import StateQualityRanker


DEFAULT_PENALTIES = (0., .025, .05, .1, .2)


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, cases, penalties):
    """Evaluate one frozen selector; labels only enter after indices are chosen."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if (checkpoint.get('architecture') != 'initial_geometry_selector_v1'
            or checkpoint.get('pool_size') != 256
            or checkpoint.get('selection_budget') != 64):
        raise ValueError('unsupported selector checkpoint')
    model = StateQualityRanker(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    mean, scale = checkpoint['mean'], checkpoint['scale']
    rows = []
    for case in cases:
        descriptors = (case['features'] - mean) / scale
        scores = model(descriptors)['q_hat']
        for penalty in penalties:
            indices = diversity_aware_indices(scores, descriptors, keep=64,
                                               penalty=penalty)
            rows.append({'case': case['case'], 'penalty': penalty,
                         **selector_retention_at_indices(
                             indices, case['dockq'], case['sources'])})
    return rows


def summarize(rows, penalties):
    result = []
    for penalty in penalties:
        selected = [row for row in rows if row['penalty'] == penalty]
        defined = [row['oracle_retention'] for row in selected
                   if row['oracle_retention'] is not None]
        result.append({
            'penalty': penalty,
            'cases': len(selected),
            'mean_selected_best': statistics.mean(
                row['selected_best'] for row in selected),
            'mean_case_oracle_retention': statistics.mean(defined) if defined else None,
            'selected_hits_0.23': sum(row['selected_hit'] for row in selected),
            'oracle_index_retained': sum(row['oracle_index_retained'] for row in selected),
            'mean_selected_bitseed': statistics.mean(
                row['selected_source_counts']['bitseed'] for row in selected),
            'mean_selected_sdpi': statistics.mean(
                row['selected_source_counts']['sdpi'] for row in selected),
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rankers', type=Path, required=True,
                        help='Directory containing n<size>_s<seed>/best.pt')
    parser.add_argument('--val-archives', type=Path, nargs='+', required=True)
    parser.add_argument('--val-labels', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--penalties', type=float, nargs='+', default=DEFAULT_PENALTIES)
    args = parser.parse_args(argv)
    penalties = tuple(float(value) for value in args.penalties)
    if (not penalties or penalties[0] != 0 or len(set(penalties)) != len(penalties)
            or any(value < 0 for value in penalties) or args.out.exists()):
        parser.error('use a new output, unique nonnegative penalties, and include 0 first')
    cases, manifests = load_cases(args.val_archives, args.val_labels, 'val')
    checkpoints = sorted(args.rankers.glob('n*_s*/best.pt'))
    if not checkpoints:
        raise ValueError('no selector checkpoints found')
    runs = []
    for checkpoint in checkpoints:
        relative = checkpoint.parent.name
        rows = evaluate_checkpoint(checkpoint, cases, penalties)
        runs.append({'run': relative, 'checkpoint': str(checkpoint),
                     'summary': summarize(rows, penalties), 'rows': rows})
    payload = {
        'scope': 'internal validation diagnostic; penalty is not frozen for deployment',
        'selection': ('q_hat minus penalty times maximum RBF similarity to an already '
                      'selected candidate; descriptors are standardized native-free geometry'),
        'control': 'penalty=0 must exactly reproduce stable learned-score top-64',
        'penalties': list(penalties), 'validation_cases': len(cases),
        'input_manifests': manifests, 'runs': runs,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'report.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# Learned-score diversity audit', '', payload['scope'], '',
             '| Run | Penalty | Selected-best | Retention | ≥0.23 | Oracle ID | BiTSeed/SDPI |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for run in runs:
        for row in run['summary']:
            lines.append(f"| {run['run']} | {row['penalty']:.3f} | "
                         f"{row['mean_selected_best']:.4f} | "
                         f"{row['mean_case_oracle_retention']:.4f} | "
                         f"{row['selected_hits_0.23']}/{row['cases']} | "
                         f"{row['oracle_index_retained']}/{row['cases']} | "
                         f"{row['mean_selected_bitseed']:.1f}/{row['mean_selected_sdpi']:.1f} |")
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'runs': len(runs), 'cases': len(cases),
                      'penalties': list(penalties)}, indent=2))


if __name__ == '__main__':
    main()

"""Refine DeepTernary's model-ranked top-1 PROTAC candidate."""

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from evaluate_conditioned_entry import main as evaluate_main


def model_top1_prediction(prior_root, case, expected_candidates=None):
    """Resolve model top-1 (minimum predicted RMSD), never native-score oracle."""
    case_dir = (Path(prior_root) / case).resolve()
    summary = case_dir / f'summary_{case}.csv'
    rows = list(csv.DictReader(summary.open(newline='')))
    if not rows or len({int(row['seed']) for row in rows}) != len(rows):
        raise ValueError(f'{case}: empty summary or duplicate seeds')
    if expected_candidates is not None and len(rows) != expected_candidates:
        raise ValueError(
            f'{case}: expected {expected_candidates} candidates, found {len(rows)}'
        )
    values = [float(row['pred_p2_rmsd']) for row in rows]
    if any(value < 0 for value in values):
        raise ValueError(f'{case}: invalid official surrogate value')
    best = min(zip(values, rows), key=lambda item: (item[0], int(item[1]['seed'])))[1]
    prediction = Path(best['complex_pred']).resolve()
    if prediction.parent != case_dir or not prediction.is_file():
        raise ValueError(f'{case}: official prediction escapes or is absent from case directory')
    return prediction, int(best['seed'])


# Compatibility for existing imports. This has always selected on model output.
official_prediction = model_top1_prediction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--prior-root', required=True)
    parser.add_argument('--condition-manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model-dir', default='checkpoints/score_model')
    parser.add_argument('--cache-dir', default='results/featurize_cache')
    parser.add_argument('--t-anchor', type=float, default=.30)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--condition-guidance-scale', type=float, default=1.)
    parser.add_argument(
        '--expected-candidates', type=int, default=None,
        help='Require this many candidates (40 matches DeepTernary PROTAC default).',
    )
    args = parser.parse_args(argv)
    prediction, deepternary_seed = model_top1_prediction(
        args.prior_root, args.case, args.expected_candidates,
    )
    checkpoint = Path(args.checkpoint).resolve()
    condition_manifest = Path(args.condition_manifest).resolve()
    output = Path(args.out).resolve()
    evaluate_main([
        '--checkpoint', str(checkpoint), '--project-root', args.project_root,
        '--index', args.index, '--case', args.case,
        '--condition-manifest', str(condition_manifest), '--out', str(output),
        '--device', args.device, '--model-dir', args.model_dir,
        '--cache-dir', args.cache_dir, '--walkers', '1', '--steps', str(args.steps),
        '--seed', str(7521 + deepternary_seed), '--init-mode', 'deepternary_pose',
        '--t-anchor', str(args.t_anchor), '--deepternary-pose', str(prediction),
        '--condition-guidance', 'residual',
        '--condition-guidance-scale', str(args.condition_guidance_scale),
    ])


if __name__ == '__main__':
    main()

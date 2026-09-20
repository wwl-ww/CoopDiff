"""Build a checksummed, label-side dataset for DeepTernary candidate ranking."""

import argparse
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from evaluation_common import sha256_file
from tricoopdiff.deepternary_ranking import (
    FEATURE_NAMES, FEATURE_VERSION, deepternary_candidate_features,
)


DATASET_VERSION = 'deepternary_refined_candidate_labels_v2'
RESULT_NAME = re.compile(r'(.+)_seed([0-9]+)\.json$')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-root', type=Path, nargs='+', required=True)
    parser.add_argument('--refined-results', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('choose a new output path')
    rows = []
    result_paths = sorted(path for root in args.refined_results
                          for path in root.glob('*_seed*.json'))
    for result_path in result_paths:
        match = RESULT_NAME.fullmatch(result_path.name)
        if match is None:
            continue
        case, seed = match.group(1), int(match.group(2))
        summary_paths = [root / case / f'summary_{case}.csv' for root in args.prior_root]
        summary_paths = [path for path in summary_paths if path.is_file()]
        if len(summary_paths) != 1:
            raise ValueError(f'{case}: expected one official summary, found {summary_paths}')
        summary_path = summary_paths[0]
        candidates = {int(row['seed']): row
                      for row in csv.DictReader(summary_path.open(newline=''))}
        if seed not in candidates:
            raise ValueError(f'{case}: refined seed absent from official summary')
        candidate = candidates[seed]
        prediction = Path(candidate['complex_pred'])
        report = json.loads(result_path.read_text())
        if Path(report['run_config']['deepternary_pose']).resolve() != prediction.resolve():
            raise ValueError(f'{case} seed {seed}: prediction/report identity mismatch')
        rows.append({
            'case': case, 'seed': seed,
            'features': deepternary_candidate_features(
                prediction, float(candidate['pred_p2_rmsd']), float(candidate['clash_ratio'])).tolist(),
            'refined_dockq': float(report['structure']['dockq_best']),
            'prediction_sha256': sha256_file(prediction),
            'result_sha256': sha256_file(result_path),
        })
    cases = sorted({row['case'] for row in rows})
    counts = {case: sum(row['case'] == case for row in rows) for case in cases}
    if not rows or len(set((row['case'], row['seed']) for row in rows)) != len(rows):
        raise ValueError('selector dataset is empty or has duplicate candidate IDs')
    payload = {
        'protocol': DATASET_VERSION,
        'scope': 'label-side training data; refined DockQ is never an inference feature',
        'feature_version': FEATURE_VERSION, 'feature_names': list(FEATURE_NAMES),
        'cases': cases, 'candidate_counts': counts,
        'prior_roots': [str(path.resolve()) for path in args.prior_root],
        'refined_result_roots': [str(path.resolve()) for path in args.refined_results],
        'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'cases': len(cases), 'candidates': len(rows),
                      'features': len(FEATURE_NAMES)}, indent=2))


if __name__ == '__main__':
    main()

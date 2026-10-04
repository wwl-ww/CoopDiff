"""Audit already-written native-free ranks against separately generated labels."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from evaluation_common import sha256_file
from tricoopdiff.ranking.ranking_audit import join_labels, summarize_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('use a new audit output')
    manifest = json.loads((args.predictions / 'manifest.json').read_text())
    complete = json.loads((args.predictions / 'complete.json').read_text())
    label_manifest = json.loads((args.labels / 'manifest.json').read_text())
    if (complete['cases'] != manifest['cases'] or not complete['cases']
            or len(set(complete['cases'])) != len(complete['cases'])
            or label_manifest['archive_manifest_sha256'] != manifest['archive_manifest_sha256']):
        raise ValueError('incomplete predictions or mismatched label archive')
    cases, signatures = {}, {}
    for case in complete['cases']:
        if case != Path(case).name or case in ('.', '..'):
            raise ValueError('unsafe case identifier')
        prediction_path, label_path = args.predictions / f'{case}.json', args.labels / f'{case}.json'
        prediction, label = json.loads(prediction_path.read_text()), json.loads(label_path.read_text())
        if prediction['case'] != case:
            raise ValueError('case file mismatch')
        cases[case] = join_labels(prediction, label)
        signatures[case] = {'prediction': sha256_file(prediction_path), 'label': sha256_file(label_path)}
    result = {'protocol': 'saved_state_ranking_audit_v1', 'method': manifest['method'],
              'scope': 'saved paired states only; source strata keep all cases in denominators',
              'random_reference': 'exact uniform-state expectation, not a fitted model or extra trajectory',
              'threshold_units': 'case counts; random counts are expectations',
              'retention_policy': 'mean of per-case ratios; zero oracle undefined and counted separately',
              'predictions_manifest_sha256': sha256_file(args.predictions / 'manifest.json'),
              'labels_manifest_sha256': sha256_file(args.labels / 'manifest.json'),
              'files_sha256': signatures, **summarize_cases(cases)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result['summary']['all/union'], indent=2))


if __name__ == '__main__':
    main()

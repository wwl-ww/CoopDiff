"""Audit complete ranking labels for class and case support; never fit a selector."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch

from evaluation_common import sha256_file
from tricoopdiff.ranking.ranking_audit import join_labels, label_support
from tricoopdiff.ranking.ranking_inference import archive_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('use a new class-support audit output')
    manifest = json.loads((args.archives / 'manifest.json').read_text())
    labels_manifest = json.loads((args.labels / 'manifest.json').read_text())
    if labels_manifest['archive_manifest_sha256'] != sha256_file(args.archives / 'manifest.json'):
        raise ValueError('label/archive manifest mismatch')
    cases, signatures = {}, {}
    for case in manifest['cases']:
        if case != Path(case).name or case in ('.', '..') or case in cases:
            raise ValueError('unsafe or duplicate case')
        path, label_path = args.archives / f'{case}.pt', args.labels / f'{case}.json'
        receipt = json.loads((args.archives / f'{case}.complete.json').read_text())
        digest = sha256_file(path)
        if receipt['archive_sha256'] != digest:
            raise ValueError('archive receipt mismatch')
        archive = torch.load(path, map_location='cpu', weights_only=False)
        if archive['case'] != case or archive['manifest'] != manifest:
            raise ValueError('archive identity mismatch')
        metadata = [{**{key: row[key] for key in ('pool_index', 'stage', 'source')}, 'score': 0.}
                    for row in archive_rows(archive)]
        # Reuse exact state coverage and identity checks; the dummy score is never ranked here.
        cases[case] = join_labels({'case': case, 'archive_sha256': digest, 'rows': metadata},
                                 json.loads(label_path.read_text()))
        signatures[case] = {'archive': digest, 'labels': sha256_file(label_path)}
    payload = {'protocol': 'ranking_label_support_v1', 'scope': 'class coverage only; no fitting or case selection',
               'warning': 'initial/final poses within a case are correlated, not independent examples',
               'manifest_sha256': sha256_file(args.archives / 'manifest.json'),
               'files_sha256': signatures, 'support': label_support(cases)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(payload['support']['all/union'], indent=2))


if __name__ == '__main__':
    main()

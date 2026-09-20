"""Seal source and absolute-path indexes for a foreground conditioned run."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def prepare(train_path, val_path, out):
    train, val = [json.loads(path.read_text()) for path in (train_path, val_path)]
    train_pdbs = {row['case'].split('_')[0].upper() for row in train['entries']}
    excluded = set(train['metadata']['excluded_benchmark_pdb_ids'])
    if train_pdbs & excluded:
        raise ValueError('benchmark PDB in training data')
    blocked = train_pdbs | excluded
    clusters = [cluster for cluster in val['clusters'] if not any(
        case.split('_')[0].upper() in blocked for case in cluster['items'])]
    retained = {case for cluster in clusters for case in cluster['items']}
    val = {**val, 'clusters': clusters,
           'entries': [row for row in val['entries'] if row['case'] in retained]}
    if not clusters or not val['entries']:
        raise ValueError('no disjoint validation clusters remain')
    for payload, source in ((train, train_path), (val, val_path)):
        for row in payload['entries']:
            for key in ('protein1', 'protein2', 'ligand'):
                path = Path(row[key])
                if not path.is_absolute():
                    path = source.resolve().parent.parent / path
                if not path.is_file():
                    raise FileNotFoundError(path)
                row[key] = str(path.resolve())
    out.mkdir(parents=True, exist_ok=False)
    fingerprints = {}
    runtime = out / 'runtime'
    for directory in ('scripts', 'tricoopdiff'):
        for source in sorted((ROOT / directory).rglob('*.py')):
            relative = source.relative_to(ROOT)
            destination = runtime / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            fingerprints[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    for name, payload in (('train', train), ('val', val)):
        with (out / f'{name}_index.json').open('x') as handle:
            json.dump(payload, handle, indent=2)
    audit = {'train_entries': len(train['entries']), 'train_clusters': len(train['clusters']),
             'val_entries': len(val['entries']), 'val_clusters': len(clusters),
             'train_val_pdb_overlap': [], 'benchmark_train_overlap': [],
             'validation_scope': 'internal development; parent used historical validation',
             'source_index_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (train_path, val_path)},
             'runtime_sha256': fingerprints}
    with (out / 'preparation.json').open('x') as handle:
        json.dump(audit, handle, indent=2)
    return {key: value for key, value in audit.items() if key != 'runtime_sha256'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.train, args.val, args.out), indent=2))

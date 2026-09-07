#!/usr/bin/env python
"""Fail closed on incomplete official training indexes; not a model-quality gate."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inspect(train_path, val_path):
    errors, splits, fingerprints = [], [], {}
    rejected_total = 0
    for split, path, expected_clusters in (
            ('train', train_path, 588), ('val', val_path, 65)):
        raw = Path(path).read_bytes()
        fingerprints[split] = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        entries = payload.get('entries', [])
        cases = [entry['case'] for entry in entries]
        clusters = payload.get('clusters', [])
        metadata = payload.get('metadata', {})
        rejected = payload.get('rejected', [])
        rejected_total += len(rejected)
        if not entries or len(cases) != len(set(cases)):
            errors.append(f'{split}: empty or duplicate cases')
        if metadata.get('clusters_requested') != expected_clusters or not clusters:
            errors.append(f'{split}: expected {expected_clusters} original clusters and nonempty usable clusters')
        members = [case for cluster in clusters for case in cluster['items']]
        if set(members) != set(cases) or len(members) != len(set(members)):
            errors.append(f'{split}: cluster membership differs from entries or overlaps')
        if metadata.get('protocol') != 'joint_train_intersection_v1' or metadata.get('split') != split:
            errors.append(f'{split}: incorrect protocol/split')
        if metadata.get('joint_pool_cases') != 7198 or metadata.get('joint_pool_clusters') != 653:
            errors.append(f'{split}: incorrect original pool size')
        if any(item.get('code') != 'unsupported_ligand_graph' for item in rejected):
            errors.append(f'{split}: unexpected structure materialization failures')
        for entry in entries:
            for key in ('protein1', 'protein2', 'ligand'):
                file = Path(entry[key])
                file = file if file.is_absolute() else ROOT / file
                if not file.is_file() or file.stat().st_size == 0:
                    errors.append(f'{entry["case"]}: missing/empty {key}')
        splits.append(set(cases))
    if splits[0] & splits[1]:
        errors.append('train/validation cases overlap')
    if sum(map(len, splits)) != 7087 or rejected_total != 111:
        errors.append('official pool requires 7087 accepted and 111 explicitly excluded cases')
    benchmark = set()
    for path in ('DeepTernary/data/PROTAC/protac22.txt',
                 'DeepTernary/data/MolecularGlue/test_all.txt'):
        benchmark.update(line.strip()[:4].upper() for line in (ROOT / path).read_text().splitlines() if line.strip())
    leaked = sorted(case for cases in splits for case in cases if case[:4].upper() in benchmark)
    if leaked:
        errors.append(f'benchmark PDB leakage: {leaked}')
    return {'data_pass': not errors, 'model_training_ready': False,
            'note': 'Data-only check; full-time gradient, warmup and training-set overfit gates still required.',
            'counts': list(map(len, splits)), 'index_sha256': fingerprints, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default='data/dual_v1/ternary_train.json')
    parser.add_argument('--val', default='data/dual_v1/ternary_val.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = inspect(args.train, args.val)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['data_pass'] else 1)


if __name__ == '__main__':
    main()

"""Extract a small official train/validation subset from a local TernaryDB archive."""

import argparse
import json
from pathlib import Path
import tarfile

from make_deepternary_index import (
    case_pdb_id, export_case, merge_cluster_intersection, read_case_list,
    split_clusters, write_index,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='data/TernaryDB.tar.gz')
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--split', choices=('train', 'val'), default='train')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count must be positive')
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = Path('DeepTernary/data')
    pools = [json.loads((base / name).read_text()) for name in (
        'PROTAC/train_clusters_poc6.json',
        'MolecularGlue/train_clusters_poc3_exUNL.json')]
    excluded = {case_pdb_id(case) for name in (
        'PROTAC/protac22.txt', 'MolecularGlue/test_all.txt')
        for case in read_case_list(base / name)}
    clusters = merge_cluster_intersection(pools, excluded)
    if len(clusters) != 653 or sum(len(c['items']) for c in clusters) != 7198:
        raise RuntimeError('official joint pool changed')
    train, val = split_clusters(clusters, .1, 0)
    selected = train if args.split == 'train' else val
    members = {case for cluster in selected for case in cluster['items']}
    entries = []
    with tarfile.open(args.archive, 'r|gz') as archive:
        for member in archive:
            path = Path(member.name)
            if not member.isfile() or path.name != 'gt_complex.pdb':
                continue
            case = path.parent.name
            if case not in members or member.size > 50_000_000:
                continue
            destination = root / 'pdbs' / case / 'gt_complex.pdb'
            destination.parent.mkdir(parents=True)
            with archive.extractfile(member) as source:
                destination.write_bytes(source.read())
            entry = export_case(case, root / 'pdbs', root / 'structures')
            entries.append(entry)
            destination_index = root / f'{args.split}.json'
            write_index(destination_index, entries, [], {
                'protocol': 'joint_train_intersection_v1', 'diagnostic_subset': True,
                'split_seed': 0, 'validation_fraction': .1,
                'selection': f'first archived members of the official {args.split} split',
                'split': args.split,
            })
            print(f"case={case}\nindex={destination_index}", flush=True)
            if len(entries) == args.count:
                return
    raise RuntimeError(f'archive contains fewer {args.split} members than requested')


if __name__ == '__main__':
    main()

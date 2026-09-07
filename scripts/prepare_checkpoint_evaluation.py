"""Materialize official joint-pool and benchmark inputs in a scratch directory."""

import argparse
import json
from pathlib import Path
import tarfile

from make_deepternary_index import (
    build_index, case_pdb_id, merge_cluster_intersection, read_case_list,
    write_index,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--protac-root', required=True)
    args = parser.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = Path(__file__).resolve().parents[1] / 'DeepTernary/data'
    protac = read_case_list(base / 'PROTAC/protac22.txt')
    mgd = read_case_list(base / 'MolecularGlue/test_all.txt')
    assert len(protac) == len(set(protac)) == 22
    assert len(mgd) == len(set(mgd)) == 94
    pools = [json.loads((base / name).read_text()) for name in (
        'PROTAC/train_clusters_poc6.json',
        'MolecularGlue/train_clusters_poc3_exUNL.json')]
    clusters = merge_cluster_intersection(
        pools, {case_pdb_id(case) for case in protac + mgd})
    joint = {case for cluster in clusters for case in cluster['items']}
    assert len(joint) == 7198 and len(clusters) == 653
    wanted = joint | set(mgd)
    found = set()
    with tarfile.open(args.archive, 'r|gz') as archive:
        for member in archive:
            path = Path(member.name)
            case = path.parent.name
            if (not member.isfile() or path.name != 'gt_complex.pdb'
                    or case not in wanted):
                continue
            if case in found or member.size > 50_000_000:
                raise RuntimeError(f'duplicate or oversized archive member: {member.name}')
            destination = root / 'release/pdbs' / case / 'gt_complex.pdb'
            destination.parent.mkdir(parents=True)
            with archive.extractfile(member) as source:
                destination.write_bytes(source.read())
            found.add(case)
            if len(found) % 500 == 0:
                print(f'extracted {len(found)}/{len(wanted)}', flush=True)
    missing = sorted(wanted - found)
    (root / 'extraction.json').write_text(json.dumps({
        'requested': len(wanted), 'extracted': len(found), 'missing': missing,
    }, indent=2))
    if missing:
        raise RuntimeError(f'archive missing {len(missing)} cases: {missing[:5]}')
    entries, rejected = build_index(mgd, root / 'release/pdbs', root / 'mgd')
    write_index(root / 'mgd94.json', entries, rejected, {
        'protocol': 'mgd94_bound', 'requested': 94, 'seeds_per_case': 1,
        'split': 'test',
    })
    if rejected:
        raise RuntimeError(f'MGD materialization rejected {len(rejected)} cases')
    entries = []
    for case in protac:
        directory = Path(args.protac_root).resolve() / case
        paths = dict(protein1=directory / 'unbound_protein1.pdb',
                     protein2=directory / 'unbound_protein2.pdb',
                     ligand=directory / 'ligand.sdf',
                     complex=directory / 'gt_complex.pdb')
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        entries.append({'case': case, **{k: str(v) for k, v in paths.items()}})
    write_index(root / 'protac22_unbound.json', entries, [], {
        'protocol': 'protac22_unbound', 'requested': 22, 'seeds_per_case': 40,
        'split': 'test', 'ranking': 'oracle_only_no_supervised_ranking_head',
    })
    print(f'benchmark indexes ready: {root}', flush=True)


if __name__ == '__main__':
    main()

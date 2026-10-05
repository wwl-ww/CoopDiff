#!/usr/bin/env python
"""Build the leak-controlled joint DeepTernary training indexes.

The release stores each complex as ``pdbs/<case>/gt_complex.pdb`` with the two
protein partners normalized to chains A and B and the ligand as HETATM records.
The training pool is the exact member intersection of DeepTernary's published
PROTAC and Molecular Glue training cluster files. Cases sharing a PDB ID with
either official benchmark are excluded. Cluster relationships from both tasks
are merged, then a deterministic cluster-level train/validation split is made.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class UnsupportedLigandGraphError(RuntimeError):
    """The deposited ligand cannot satisfy CoopDiff's single-graph contract."""


class UnionFind:
    """Minimal disjoint-set implementation used to merge cluster relations."""

    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def case_pdb_id(case):
    """Return the case-insensitive four-character PDB identifier."""
    return case.split('_', 1)[0].upper()


def cluster_members(clusters):
    """Return the unique item members represented by a cluster file."""
    return {case for cluster in clusters for case in cluster.get('items', [])}


def merge_cluster_intersection(cluster_sets, excluded_pdb_ids=()):
    """Intersect member pools and union all within-source cluster relations."""
    if not cluster_sets:
        return []
    shared = set.intersection(*(cluster_members(clusters) for clusters in cluster_sets))
    excluded_pdb_ids = {value.upper() for value in excluded_pdb_ids}
    shared = {case for case in shared if case_pdb_id(case) not in excluded_pdb_ids}
    union_find = UnionFind(shared)
    representative_cases = set()
    for clusters in cluster_sets:
        for cluster in clusters:
            members = [case for case in cluster.get('items', []) if case in shared]
            if members:
                anchor = members[0]
                for member in members[1:]:
                    union_find.union(anchor, member)
            representative_cases.update(
                case for case in cluster.get('representative', []) if case in shared
            )
    components = {}
    for case in shared:
        components.setdefault(union_find.find(case), []).append(case)
    merged = []
    for members in components.values():
        members = sorted(members)
        representatives = sorted(set(members) & representative_cases)
        merged.append({
            'representative': representatives or [members[0]],
            'items': members,
        })
    return sorted(merged, key=lambda cluster: cluster['items'][0])


def split_clusters(clusters, validation_fraction=0.1, seed=0):
    """Split connected published-cluster/PDB groups, never individual cases."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    clusters = merge_pdb_groups(clusters)
    if len(clusters) < 2:
        raise ValueError('at least two independent groups are required for a split')
    order = list(range(len(clusters)))
    random.Random(seed).shuffle(order)
    n_validation = min(len(clusters) - 1, max(1, round(len(clusters) * validation_fraction)))
    validation_ids = set(order[:n_validation])
    train = [cluster for index, cluster in enumerate(clusters)
             if index not in validation_ids]
    validation = [cluster for index, cluster in enumerate(clusters)
                  if index in validation_ids]
    return train, validation


def merge_pdb_groups(clusters):
    """Transitively preserve source clusters and unite cases from the same PDB."""
    members = sorted(cluster_members(clusters))
    groups = UnionFind(members)
    by_pdb, representatives = {}, set()
    for cluster in clusters:
        items = cluster['items']
        representatives.update(cluster.get('representative', []))
        for case in items:
            groups.union(items[0], case)
            previous = by_pdb.setdefault(case_pdb_id(case), case)
            groups.union(previous, case)
    components = {}
    for case in members:
        components.setdefault(groups.find(case), []).append(case)
    return sorted([{'items': items,
                    'representative': sorted(set(items) & representatives) or items[:1]}
                   for items in components.values()], key=lambda c: c['items'][0])


def audit_partition(train, validation, source_clusters=()):
    """Fail closed on duplicate cases, PDB overlap or split source clusters."""
    case_sets = []
    for split, payload in zip(('train', 'val'), (train, validation)):
        if payload.get('metadata', {}).get('split', split) != split:
            raise ValueError('index split label disagrees with its training/validation role')
        entries = payload['entries']
        cases = {entry['case'] for entry in entries}
        if not cases or len(cases) != len(entries):
            raise ValueError('index is empty or contains duplicate case IDs')
        excluded = set(payload.get('metadata', {}).get('excluded_benchmark_pdb_ids', []))
        if {case_pdb_id(case) for case in cases} & excluded:
            raise ValueError('benchmark PDB found in training/validation pool')
        case_sets.append(cases)
    left, right = case_sets
    overlap = {case_pdb_id(c) for c in left} & {case_pdb_id(c) for c in right}
    if overlap:
        raise ValueError(f'train/validation PDB overlap: {len(overlap)} IDs; {sorted(overlap)[:8]}')
    clusters = list(source_clusters) + train.get('clusters', []) + validation.get('clusters', [])
    for cluster in clusters:
        members = set(cluster['items'])
        if members & left and members & right:
            raise ValueError('a published cluster crosses train/validation')
    return {'train_cases': len(left), 'validation_cases': len(right),
            'pdb_overlap': 0, 'source_clusters_checked': len(source_clusters)}


def reuse_entries(paths):
    """Read existing materialized structures without changing shared data files."""
    entries, rejected, hashes = {}, [], {}
    for path in paths:
        raw = Path(path).read_bytes()
        hashes[str(Path(path).resolve())] = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        rejected.extend(payload.get('rejected', []))
        for entry in payload['entries']:
            if entry['case'] in entries:
                raise ValueError('duplicate case in reused indexes')
            entry = dict(entry)
            for key in ('protein1', 'protein2', 'ligand'):
                resolved = Path(entry[key]).resolve()
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
                entry[key] = str(resolved)
            entries[entry['case']] = entry
    return entries, rejected, hashes


def audit_index_files(train_path, validation_path, source_cluster_paths=None):
    """Audit published cluster isolation and fingerprint the actual run inputs."""
    if source_cluster_paths is None:
        source_cluster_paths = [
            _PROJECT_ROOT / 'DeepTernary/data/PROTAC/train_clusters_poc6.json',
            _PROJECT_ROOT / 'DeepTernary/data/MolecularGlue/train_clusters_poc3_exUNL.json']
    payloads, hashes = [], []
    for path in (train_path, validation_path, *source_cluster_paths):
        raw = Path(path).read_bytes()
        payloads.append(json.loads(raw))
        hashes.append(hashlib.sha256(raw).hexdigest())
    sources = [cluster for clusters in payloads[2:] for cluster in clusters]
    report = audit_partition(*payloads[:2], source_clusters=sources)
    return {**report, 'train_sha256': hashes[0], 'validation_sha256': hashes[1],
            'source_cluster_sha256': hashes[2:]}


def select_cluster_cases(clusters, mode='representative', excluded=None):
    """Select one representative per cluster, or every clustered member."""
    if mode not in {'representative', 'all'}:
        raise ValueError(f'unknown cluster selection mode: {mode}')
    excluded = set(excluded or ())
    selected = []
    for cluster in clusters:
        items = [case for case in cluster.get('items', []) if case not in excluded]
        representatives = [
            case for case in cluster.get('representative', []) if case not in excluded
        ]
        if mode == 'all':
            selected.extend(items)
        elif representatives:
            selected.append(representatives[0])
        elif items:
            selected.append(items[0])
    return list(dict.fromkeys(selected))


def read_case_list(path):
    """Read a DeepTernary txt/csv split, using its first column as case ID."""
    return [
        line.strip().split(',')[0]
        for line in Path(path).read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def _portable(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def count_pdb_heavy_atoms(path):
    """Count non-hydrogen atoms using the PDB element column."""
    count = 0
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.startswith(('ATOM  ', 'HETATM')):
            continue
        element = line[76:78].strip().upper()
        if element not in {'H', 'D', 'T'}:
            count += 1
    return count


def validate_ligand_graph(path):
    """Repair plausible missing PDB bonds and enforce one connected ligand.

    PDB files do not require ``CONECT`` records, and RDKit occasionally misses
    a long but chemically plausible covalent bond when inferring connectivity
    from coordinates (for example the 1.82-A C--C bond in 2NU7_B_E_COA).
    Add only the shortest inter-fragment bond that fits a conservative
    covalent-radius cutoff. True salts or other distant components remain an
    error instead of being silently joined or discarded.
    """
    from rdkit import Chem

    molecule = Chem.MolFromPDBFile(str(path), sanitize=True, removeHs=True)
    if molecule is None:
        raise RuntimeError('RDKit failed to parse or sanitize ligand')
    if molecule.GetNumConformers() != 1:
        raise RuntimeError(
            f'ligand has {molecule.GetNumConformers()} conformers; expected one'
        )
    if molecule.GetNumBonds() == 0:
        raise RuntimeError('ligand graph contains no inferred bonds')
    bridges = []
    editable = Chem.RWMol(molecule)
    conformer = molecule.GetConformer()
    periodic_table = Chem.GetPeriodicTable()
    while True:
        fragments = Chem.GetMolFrags(editable)
        if len(fragments) == 1:
            break
        candidates = []
        for left_index, left in enumerate(fragments):
            for right in fragments[left_index + 1:]:
                for atom_i in left:
                    for atom_j in right:
                        position_i = conformer.GetAtomPosition(atom_i)
                        position_j = conformer.GetAtomPosition(atom_j)
                        distance = position_i.Distance(position_j)
                        radius = (
                            periodic_table.GetRcovalent(
                                molecule.GetAtomWithIdx(atom_i).GetAtomicNum()
                            )
                            + periodic_table.GetRcovalent(
                                molecule.GetAtomWithIdx(atom_j).GetAtomicNum()
                            )
                        )
                        # RDKit's covalent radii are deliberately small
                        # (C=0.68 A in the deployed version). Allow coordinate
                        # uncertainty up to 1.4x the summed radii; this admits
                        # the known 1.82-A C--C miss while remaining well below
                        # nonbonded C--C contact distances.
                        if 0.6 * radius <= distance <= 1.4 * radius:
                            candidates.append((distance, atom_i, atom_j))
        if not candidates:
            sizes = sorted((len(fragment) for fragment in fragments), reverse=True)
            raise UnsupportedLigandGraphError(
                f'ligand has disconnected fragments {sizes} with no plausible '
                'missing covalent bond'
            )
        _, atom_i, atom_j = min(candidates)
        editable.AddBond(atom_i, atom_j, Chem.BondType.SINGLE)
        bridges.append((atom_i + 1, atom_j + 1))

    if bridges:
        lines = Path(path).read_text(encoding='utf-8').splitlines()
        insertion = next((i for i, line in enumerate(lines)
                          if line.startswith('END')), len(lines))
        conect = [f'CONECT{atom_i:5d}{atom_j:5d}' for atom_i, atom_j in bridges]
        lines[insertion:insertion] = conect
        temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
        temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        temporary.replace(path)
        molecule = Chem.MolFromPDBFile(str(path), sanitize=True, removeHs=True)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise RuntimeError('ligand connectivity repair did not round-trip')


def export_case(case, pdb_root, output_root):
    """Split one normalized ``gt_complex.pdb`` into two proteins and a ligand."""
    from Bio.PDB import PDBIO, PDBParser, Select

    ligand_name = case.rsplit('_', 1)[-1].upper()
    source = Path(pdb_root) / case / 'gt_complex.pdb'
    if not source.is_file():
        raise FileNotFoundError(source)
    structure = PDBParser(QUIET=True).get_structure(case, str(source))

    class ProteinSelect(Select):
        def __init__(self, chain_id):
            self.chain_id = chain_id

        def accept_chain(self, chain):
            return int(chain.get_id() == self.chain_id)

        def accept_residue(self, residue):
            return int(residue.get_id()[0] == ' ')

    class LigandSelect(Select):
        def accept_residue(self, residue):
            return int(
                residue.get_id()[0] != ' '
                and residue.resname.strip().upper() == ligand_name
            )

    case_dir = Path(output_root) / case
    case_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        'protein1': case_dir / 'protein1.pdb',
        'protein2': case_dir / 'protein2.pdb',
        'ligand': case_dir / 'ligand.pdb',
    }
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(paths['protein1']), ProteinSelect('A'))
    io.save(str(paths['protein2']), ProteinSelect('B'))
    io.save(str(paths['ligand']), LigandSelect())

    atom_counts = {}
    for key, path in paths.items():
        atom_counts[key] = sum(
            line.startswith(('ATOM  ', 'HETATM'))
            for line in path.read_text(encoding='utf-8').splitlines()
        )
        if atom_counts[key] == 0:
            raise RuntimeError(f'{case}: exported {key} contains no atoms')
    atom_counts['ligand_heavy'] = count_pdb_heavy_atoms(paths['ligand'])
    validate_ligand_graph(paths['ligand'])
    return {
        'case': case,
        'protein1': _portable(paths['protein1']),
        'protein2': _portable(paths['protein2']),
        'ligand': _portable(paths['ligand']),
        'source': 'DeepTernary/TernaryDB-v1.0.1',
        'atom_counts': atom_counts,
    }


def build_index(cases, pdb_root, output_root, min_ligand_heavy_atoms=1):
    """Export available cases and retain explicit rejection reasons."""
    entries, rejected = [], []
    for index, case in enumerate(cases, start=1):
        try:
            entry = export_case(case, pdb_root, output_root)
            ligand_atoms = entry['atom_counts']['ligand_heavy']
            if ligand_atoms < min_ligand_heavy_atoms:
                raise RuntimeError(
                    f'ligand has {ligand_atoms} heavy atoms; '
                    f'minimum is {min_ligand_heavy_atoms}'
                )
            entries.append(entry)
        except Exception as error:
            rejected.append({
                'case': case,
                'code': (
                    'unsupported_ligand_graph'
                    if isinstance(error, UnsupportedLigandGraphError)
                    else 'materialization_error'
                ),
                'reason': str(error),
            })
        if index % 100 == 0:
            print(f'Processed {index}/{len(cases)} cases', flush=True)
    return entries, rejected


def write_index(path, entries, rejected, metadata, clusters=None):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'entries': entries,
        'rejected': rejected,
        'metadata': metadata,
    }
    if clusters is not None:
        entry_cases = {entry['case'] for entry in entries}
        payload['clusters'] = [{
            'representative': [
                case for case in cluster['representative'] if case in entry_cases
            ],
            'items': [case for case in cluster['items'] if case in entry_cases],
        } for cluster in clusters]
        payload['clusters'] = [cluster for cluster in payload['clusters']
                               if cluster['items']]
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', default='data/TernaryDB_release')
    parser.add_argument('--protac-clusters',
                        default='DeepTernary/data/PROTAC/train_clusters_poc6.json')
    parser.add_argument('--mgd-clusters', default=(
        'DeepTernary/data/MolecularGlue/train_clusters_poc3_exUNL.json'))
    parser.add_argument('--protac-test', default='DeepTernary/data/PROTAC/protac22.txt')
    parser.add_argument('--mgd-test', default='DeepTernary/data/MolecularGlue/test_all.txt')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--expect-pool-cases', type=int, default=7198)
    parser.add_argument('--expect-pool-clusters', type=int, default=653)
    parser.add_argument('--expect-unsupported-ligands', type=int, default=111,
                        help='Versioned count of disconnected multi-component '
                             'ligands excluded by the CoopDiff graph contract')
    parser.add_argument('--output-root', default='data/ternary_deepternary')
    parser.add_argument('--index-dir', default='data')
    parser.add_argument('--reuse-index', nargs=2, metavar=('TRAIN', 'VAL'),
                        help='Reuse materialized entries; write new indexes without exporting structures')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--min-ligand-heavy-atoms', '--min-ligand-atoms',
                        dest='min_ligand_heavy_atoms', type=int, default=1,
                        help='Structural sanity threshold; official cluster '
                             'membership is otherwise preserved')
    args = parser.parse_args()
    reused, rejected_pool, input_hashes = reuse_entries(args.reuse_index) if args.reuse_index else (None, [], {})
    if args.reuse_index and any(
            (Path(args.index_dir) / f'ternary_{split}.json').resolve() == Path(path).resolve()
            for split in ('train', 'val') for path in args.reuse_index):
        raise ValueError('repartition must write to a new index directory')

    release = Path(args.release_root)
    protac_clusters = json.loads(Path(args.protac_clusters).read_text())
    mgd_clusters = json.loads(Path(args.mgd_clusters).read_text())
    benchmark_cases = read_case_list(args.protac_test) + read_case_list(args.mgd_test)
    benchmark_pdb_ids = {case_pdb_id(case) for case in benchmark_cases}
    clusters = merge_cluster_intersection(
        [protac_clusters, mgd_clusters], benchmark_pdb_ids,
    )
    pool_cases = sum(len(cluster['items']) for cluster in clusters)
    if pool_cases != args.expect_pool_cases or len(clusters) != args.expect_pool_clusters:
        raise RuntimeError(
            'joint-pool contract changed: expected '
            f'{args.expect_pool_cases} cases/{args.expect_pool_clusters} clusters, '
            f'found {pool_cases} cases/{len(clusters)} clusters'
        )
    train_clusters, validation_clusters = split_clusters(
        clusters, args.validation_fraction, args.split_seed,
    )
    if reused is not None and set(reused) - cluster_members(clusters):
        raise ValueError('reused entries contain cases outside the official pool')
    splits = {'train': train_clusters, 'val': validation_clusters}
    all_rejected = []
    for split, split_cluster_list in splits.items():
        cases = sorted({case for cluster in split_cluster_list
                        for case in cluster['items']})
        if args.limit is not None:
            cases = cases[:args.limit]
        if reused is None:
            entries, rejected = build_index(
                cases, release / 'pdbs', Path(args.output_root) / split,
                min_ligand_heavy_atoms=args.min_ligand_heavy_atoms)
        else:
            case_set = set(cases)
            entries = [reused[case] for case in cases if case in reused]
            rejected = [item for item in rejected_pool if item['case'] in case_set]
            if case_set != {e['case'] for e in entries} | {e['case'] for e in rejected}:
                raise ValueError('reused indexes do not account for every requested case')
        all_rejected.extend(rejected)
        destination = Path(args.index_dir) / f'ternary_{split}.json'
        write_index(destination, entries, rejected, {
            'protocol': 'joint_train_intersection_pdb_groups_v2',
            'input_index_sha256': input_hashes,
            'pdb_connected_groups': len(train_clusters) + len(validation_clusters),
            'split': split,
            'sampling': ('cluster_80pct_any_20pct_representative'
                         if split == 'train' else 'fixed_representative'),
            'requested': len(cases),
            'clusters_requested': len(split_cluster_list),
            'joint_pool_cases': sum(len(cluster['items']) for cluster in clusters),
            'joint_pool_clusters': len(clusters),
            'excluded_benchmark_pdb_ids': sorted(benchmark_pdb_ids),
            'validation_fraction': args.validation_fraction,
            'split_seed': args.split_seed,
            'min_ligand_heavy_atoms': args.min_ligand_heavy_atoms,
            'accepted': len(entries),
            'rejected': len(rejected),
            'exclusion_policy': {
                'allowed_code': 'unsupported_ligand_graph',
                'single_connected_ligand_required': True,
                'never_strip_or_force_join_fragments': True,
            },
        }, clusters=split_cluster_list)
        print(f'{split}: {len(entries)} accepted, {len(rejected)} rejected -> {destination}')
    audit = audit_index_files(Path(args.index_dir) / 'ternary_train.json',
                              Path(args.index_dir) / 'ternary_val.json',
                              [args.protac_clusters, args.mgd_clusters])
    (Path(args.index_dir) / 'partition_audit.json').write_text(json.dumps(audit, indent=2))
    unexpected = [item for item in all_rejected
                  if item['code'] != 'unsupported_ligand_graph']
    unsupported = [item for item in all_rejected
                   if item['code'] == 'unsupported_ligand_graph']
    if unexpected:
        raise RuntimeError(
            f'{len(unexpected)} joint-pool cases failed materialization for '
            'unexpected reasons; refusing to start training'
        )
    if len(unsupported) != args.expect_unsupported_ligands:
        raise RuntimeError(
            'unsupported-ligand contract changed: expected '
            f'{args.expect_unsupported_ligands}, found {len(unsupported)}'
        )
    print(
        f'Explicitly excluded {len(unsupported)} disconnected ligand graphs; '
        f'{pool_cases - len(unsupported)} usable cases remain',
        flush=True,
    )


if __name__ == '__main__':
    main()

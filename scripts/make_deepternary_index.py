#!/usr/bin/env python
"""Convert the official DeepTernary TernaryDB release for TriCoopDiff.

The release stores each complex as ``pdbs/<case>/gt_complex.pdb`` with the two
protein partners normalized to chains A and B and the ligand as HETATM records.
This script exports the three files consumed by ``TernaryDataLoader`` and uses
DeepTernary's published sequence clusters rather than making a random split.
"""

import argparse
import json
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    return {
        'case': case,
        'protein1': _portable(paths['protein1']),
        'protein2': _portable(paths['protein2']),
        'ligand': _portable(paths['ligand']),
        'source': 'DeepTernary/TernaryDB-v1.0.1',
        'atom_counts': atom_counts,
    }


def build_index(cases, pdb_root, output_root, min_ligand_heavy_atoms=20):
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
            rejected.append({'case': case, 'reason': str(error)})
        if index % 100 == 0:
            print(f'Processed {index}/{len(cases)} cases', flush=True)
    return entries, rejected


def write_index(path, entries, rejected, metadata):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'entries': entries,
        'rejected': rejected,
        'metadata': metadata,
    }, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', default='data/TernaryDB_release')
    parser.add_argument('--task', choices=['MGD', 'PROTAC'], default='MGD')
    parser.add_argument('--cluster-mode', choices=['representative', 'all'],
                        default='representative')
    parser.add_argument('--output-root', default='data/ternary_deepternary')
    parser.add_argument('--index-dir', default='data')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--min-ligand-heavy-atoms', '--min-ligand-atoms',
                        dest='min_ligand_heavy_atoms', type=int, default=20)
    args = parser.parse_args()

    release = Path(args.release_root)
    task_dir = release / args.task
    clusters = json.loads((task_dir / 'train_clusters.json').read_text())
    val_cases = read_case_list(
        task_dir / ('val.csv' if args.task == 'PROTAC' else 'val.txt')
    )
    test_cases = read_case_list(
        task_dir / ('test.csv' if args.task == 'PROTAC' else 'test.txt')
    )
    splits = {
        'train': select_cluster_cases(
            clusters, args.cluster_mode, excluded=set(val_cases) | set(test_cases),
        ),
        'val': val_cases,
        'test': test_cases,
    }
    for split, cases in splits.items():
        if args.limit is not None:
            cases = cases[:args.limit]
        entries, rejected = build_index(
            cases, release / 'pdbs', Path(args.output_root) / split,
            min_ligand_heavy_atoms=args.min_ligand_heavy_atoms,
        )
        destination = Path(args.index_dir) / f'ternary_{split}.json'
        write_index(destination, entries, rejected, {
            'task': args.task,
            'cluster_mode': args.cluster_mode if split == 'train' else None,
            'requested': len(cases),
            'min_ligand_heavy_atoms': args.min_ligand_heavy_atoms,
            'accepted': len(entries),
            'rejected': len(rejected),
        })
        print(f'{split}: {len(entries)} accepted, {len(rejected)} rejected -> {destination}')


if __name__ == '__main__':
    main()

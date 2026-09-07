#!/usr/bin/env python
"""Audit ternary-index structures without loading neural-network models."""

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else _PROJECT_ROOT / value


def _pdb_coordinates(path):
    coordinates = []
    for line in _resolve(path).read_text(encoding='utf-8').splitlines():
        if line.startswith(('ATOM  ', 'HETATM')):
            coordinates.append([float(line[30:38]), float(line[38:46]),
                                float(line[46:54])])
    result = np.asarray(coordinates, dtype=np.float32)
    if result.ndim != 2 or result.shape[1:] != (3,):
        raise ValueError(f'no valid coordinates in {path}')
    if not np.isfinite(result).all():
        raise FloatingPointError(f'non-finite coordinates in {path}')
    return result


def _contact_count(protein, ligand, cutoff=10.0):
    squared_distances = np.sum(
        (protein[:, None, :] - ligand[None, :, :]) ** 2, axis=-1,
    )
    return int(np.count_nonzero(squared_distances <= cutoff ** 2))


def audit_entry(entry, min_ligand_heavy_atoms=20):
    """Validate that one ligand is sanitized, bonded, 3D, and finite."""
    ligand_path = _resolve(entry['ligand'])
    molecule = Chem.MolFromPDBFile(str(ligand_path), sanitize=True, removeHs=True)
    if molecule is None:
        raise ValueError('RDKit failed to parse or sanitize ligand')
    if molecule.GetNumConformers() != 1:
        raise ValueError(f'expected one conformer, got {molecule.GetNumConformers()}')
    coordinates = molecule.GetConformer().GetPositions()
    if coordinates.shape != (molecule.GetNumAtoms(), 3):
        raise ValueError(f'invalid coordinate shape: {coordinates.shape}')
    if not np.isfinite(coordinates).all():
        raise FloatingPointError('ligand coordinates are non-finite')
    if molecule.GetNumAtoms() < min_ligand_heavy_atoms:
        raise ValueError(f'ligand has only {molecule.GetNumAtoms()} heavy atoms')
    if molecule.GetNumBonds() == 0:
        raise ValueError('ligand has no inferred bonds')
    protein1 = _pdb_coordinates(entry['protein1'])
    protein2 = _pdb_coordinates(entry['protein2'])
    contacts1 = _contact_count(protein1, coordinates)
    contacts2 = _contact_count(protein2, coordinates)
    if contacts1 == 0 or contacts2 == 0:
        raise ValueError(f'no dual interface contacts: {contacts1}, {contacts2}')
    return {
        'case': entry['case'],
        'ok': True,
        'heavy_atoms': molecule.GetNumAtoms(),
        'bonds': molecule.GetNumBonds(),
        'protein1_atoms': len(protein1),
        'protein2_atoms': len(protein2),
        'protein1_ligand_contacts_10a': contacts1,
        'protein2_ligand_contacts_10a': contacts2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--min-ligand-heavy-atoms', type=int, default=20,
                        help='Use 1 for the joint-pool contract; 20 is the legacy diagnostic filter')
    args = parser.parse_args()
    if args.min_ligand_heavy_atoms < 1:
        parser.error('min-ligand-heavy-atoms must be positive')

    payload = json.loads(_resolve(args.index).read_text(encoding='utf-8'))
    entries = payload.get('entries', payload)
    if args.limit is not None:
        entries = entries[:args.limit]
    records = []
    for entry in entries:
        try:
            records.append(audit_entry(entry, args.min_ligand_heavy_atoms))
        except Exception as error:
            records.append({
                'case': entry['case'],
                'ok': False,
                'error': f'{type(error).__name__}: {error}',
            })
    report = {
        'index': args.index,
        'min_ligand_heavy_atoms': args.min_ligand_heavy_atoms,
        'requested': len(entries),
        'passed': sum(record['ok'] for record in records),
        'failed': sum(not record['ok'] for record in records),
        'records': records,
    }
    destination = _resolve(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('requested', 'passed', 'failed')}))
    raise SystemExit(0 if report['failed'] == 0 else 1)


if __name__ == '__main__':
    main()

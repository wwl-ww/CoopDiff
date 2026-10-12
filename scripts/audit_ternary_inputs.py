#!/usr/bin/env python
"""Audit ternary-index structures without loading neural-network models."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


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
    parser.add_argument('--prepare-cache', action='store_true',
                        help='Materialize and validate production features, fullatom targets and symmetry maps')
    parser.add_argument('--cache-dir', default='results/featurize_cache_proposal_oracle')
    parser.add_argument('--reuse-cache-dir', action='append', default=[],
                        help='Reuse only source-signature-verified caches before rebuilding ESM')
    parser.add_argument('--expert-model-dir', default='checkpoints/score_model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--min-ligand-heavy-atoms', type=int, default=20,
                        help='Use 1 for the joint-pool contract; 20 is the legacy diagnostic filter')
    args = parser.parse_args()
    if args.min_ligand_heavy_atoms < 1:
        parser.error('min-ligand-heavy-atoms must be positive')

    from train_entry import _load_entries
    entries = _load_entries(str(_resolve(args.index)))
    if args.prepare_cache:
        import torch
        from train_entry import _load_score_model_args, fullatom_targets
        from featurize_common import featurize_complex, _featurize_signature, _FEATURIZE_CACHE_VERSION
        from tricoopdiff.initialization.endpoints import ligand_symmetries
        from tricoopdiff.runtime import atomic_torch_save
        score_args = _load_score_model_args(_resolve(args.expert_model_dir))
    if args.limit is not None:
        entries = entries[:args.limit]
    records = []
    for entry in entries:
        try:
            record = audit_entry(entry, args.min_ligand_heavy_atoms)
            if args.prepare_cache:
                filename = entry['case'].replace('/', '_') + '.pt'
                destination = _resolve(args.cache_dir) / filename
                if not destination.exists():
                    for directory in args.reuse_cache_dir:
                        source = _resolve(directory) / filename
                        if not source.is_file():
                            continue
                        cached = torch.load(source, map_location='cpu', weights_only=False)
                        version = cached.get('cache_version')
                        if version not in (8, _FEATURIZE_CACHE_VERSION):
                            continue
                        signature = _featurize_signature(entry['protein1'], entry['protein2'], entry['ligand'],
                                                        score_args, cache_version=None if version == 8 else version)
                        if cached.get('source_signature') == signature:
                            atomic_torch_save(cached, destination)
                            break
                e3, poi, ligand, _ = featurize_complex(entry['case'], entry['protein1'], entry['protein2'],
                    entry['ligand'], score_args, args.device, cache_dir=str(_resolve(args.cache_dir)))
                fullatom_targets(entry, e3, poi)
                record.update(protein_nodes=[e3['coords'].shape[1], poi['coords'].shape[1]],
                              symmetry_count=len(ligand_symmetries(ligand)), feature_cache_version=_FEATURIZE_CACHE_VERSION)
            records.append(record)
        except Exception as error:
            records.append({
                'case': entry['case'],
                'ok': False,
                'error': f'{type(error).__name__}: {error}',
            })
        if len(records) % 25 == 0 or not records[-1]['ok']:
            print(json.dumps({'completed': len(records), 'total': len(entries),
                              'failed': sum(not row['ok'] for row in records),
                              'last': records[-1]}), flush=True)
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

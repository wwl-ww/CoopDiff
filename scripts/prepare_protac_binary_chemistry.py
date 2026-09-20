"""Restore binary fragment chemistry from explicit offline CCDs, without moving atoms."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from evaluation_common import sha256_file
from restore_ligand_chemistry import restore_ligand
from tricoopdiff.protac_conditions import chemical_role_matches, mapping_diagnostics


def restore_side(pdb, ccd_dir, destination, full_molecule):
    # Names and coordinates are from the binary fragment, never the ternary reference.
    deposited = Chem.MolFromPDBFile(str(pdb), sanitize=False, removeHs=False, proximityBonding=False)
    if deposited is None:
        raise ValueError('unreadable binary fragment')
    atoms = [a for a in deposited.GetAtoms() if a.GetAtomicNum() > 1]
    components = {a.GetPDBResidueInfo().GetResidueName().strip() for a in atoms}
    if len(components) != 1:
        raise ValueError('binary fragment must contain one chemical component')
    component = next(iter(components))
    if not component.isalnum():
        raise ValueError('unsafe CCD component identifier')
    ccd = ccd_dir / f'{component}.cif'
    if not ccd.is_file():
        raise FileNotFoundError(f'explicit offline CCD missing: {ccd}')
    if destination.exists():
        raise FileExistsError('binary chemistry output already exists')
    report = restore_ligand(pdb, ccd, destination, allow_observed_subset=True)
    restored = Chem.MolFromMolFile(str(destination), removeHs=True)
    original_coords = np.asarray([deposited.GetConformer().GetAtomPosition(a.GetIdx()) for a in atoms])
    if restored is None or restored.GetNumAtoms() != len(atoms):
        raise ValueError('restoration changed atom count')
    if not np.allclose(original_coords, restored.GetConformer().GetPositions(), atol=1e-4, rtol=0):
        raise ValueError('restoration changed deposited coordinates')
    return {'restoration': report, 'restored_sdf': str(destination.resolve()),
            'restored_sha256': sha256_file(destination), 'coordinates_preserved': True,
            'mapping': chemical_role_matches(full_molecule, restored),
            'diagnostics': mapping_diagnostics(full_molecule, restored),
            'binary_pose_provenance_verified': False, 'ready_for_conditioned_inference': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--ccd-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='Root used to resolve relative paths in the index')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    entries = json.loads(args.index.read_text())['entries']
    cases, ccd_signatures = [], {}
    for entry in entries:
        case = entry['case']
        if case != Path(case).name or case in ('.', '..'):
            raise ValueError('unsafe case identifier')
        ligand_path = args.root.resolve() / entry['ligand']
        full = Chem.MolFromMolFile(str(ligand_path), removeHs=True)
        if full is None:
            raise ValueError(f'{case}: invalid full ligand chemistry')
        sides = {}
        for side in (1, 2):
            pdb = (args.root.resolve() / entry[f'protein{side}']).parent / f'unbound_lig{side}.pdb'
            destination = args.out / case / f'fragment{side}.sdf'
            try:
                components = {line[17:20].strip() for line in pdb.read_text().splitlines()
                              if line.startswith(('ATOM', 'HETATM'))}
                for component in components:
                    if component.isalnum():
                        ccd = args.ccd_dir / f'{component}.cif'
                        if ccd.is_file():
                            ccd_signatures[str(ccd.resolve())] = sha256_file(ccd)
                sides[str(side)] = {'status': 'restored', **restore_side(pdb, args.ccd_dir, destination, full)}
            except Exception as error:
                sides[str(side)] = {'status': 'unresolved', 'error': f'{type(error).__name__}: {error}',
                                    'source_pdb': str(pdb), 'source_sha256': sha256_file(pdb) if pdb.is_file() else None,
                                    'ready_for_conditioned_inference': False}
        cases.append({'case': case, 'full_ligand_sha256': sha256_file(ligand_path), 'sides': sides})
    rows = [side for case in cases for side in case['sides'].values()]
    summary = {'cases': len(cases), 'sides': len(rows), 'restored': sum(r['status'] == 'restored' for r in rows),
               'unresolved': sum(r['status'] != 'restored' for r in rows),
               'unique_maps': sum(r.get('mapping', {}).get('mapping_status') == 'unique' for r in rows),
               'ambiguous_maps': sum(r.get('mapping', {}).get('mapping_status') == 'ambiguous' for r in rows),
               'unmatched_maps': sum(r.get('mapping', {}).get('mapping_status') == 'unmatched' for r in rows),
               'unambiguous_fragment_roles': sum(r.get('mapping', {}).get('unambiguous_role_atoms') is not None for r in rows),
               'charge_normalized_roles': sum(r.get('mapping', {}).get('role_mapping_method') ==
                                              'charge_normalized_exact_stereo' for r in rows),
               'conservative_mcs_roles': sum(r.get('mapping', {}).get('role_mapping_method') ==
                                             'conservative_charge_normalized_mcs' for r in rows),
               'ready_for_conditioned_inference': 0}
    result = {'protocol': 'offline_binary_ccd_role_mapping_v2', 'summary': summary, 'cases': cases,
              'index_sha256': sha256_file(args.index), 'native_geometry_read': False,
              'ccd_sha256': ccd_signatures,
              'scope': 'chemical restoration and mapping only; binary provenance remains unverified',
              'code_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                             (Path(__file__), ROOT / 'scripts/restore_ligand_chemistry.py',
                              ROOT / 'tricoopdiff/protac_conditions.py')}}
    (args.out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

"""Audit available PROTAC binary condition files; never read ternary native geometry."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from evaluation_common import sha256_file
from tricoopdiff.conditioning.condition_features import (CONDITION_SCHEMA_VERSION, attachment_bonds,
                                         fragment_matches, local_pocket, mapping_diagnostics)


def protein_records(path):
    coords, residues = [], []
    for line in Path(path).read_text().splitlines():
        if not line.startswith('ATOM') or line[76:78].strip() in ('H', 'D'):
            continue
        if line[16:17] not in (' ', 'A'):
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        residues.append(f'{line[21]}:{line[22:26].strip()}:{line[26].strip()}')
    return np.asarray(coords), residues


def audit_case(entry, root):
    # Explicit allowlist: entry may reference scoring assets, but those are never opened.
    inputs = {key: (root / entry[key]).resolve() for key in ('protein1', 'protein2', 'ligand')}
    molecule = Chem.MolFromMolFile(str(inputs['ligand']), removeHs=True)
    if molecule is None:
        raise ValueError('full ligand chemistry could not be read')
    result = {'case': entry['case'], 'ligand_atoms': molecule.GetNumAtoms(), 'sides': {},
              'native_geometry_read': False, 'ready_for_conditioned_inference': False,
              'provenance': 'unverified_local_DeepTernary_assets',
              'required_review': 'confirm original binary structures and capping/atom-map chemistry'}
    maps = []
    for side in (1, 2):
        protein = inputs[f'protein{side}']
        path = protein.parent / f'unbound_lig{side}.pdb'
        if not path.is_file():
            result['sides'][str(side)] = {'status': 'missing_fragment'}
            continue
        fragment = Chem.MolFromPDBFile(str(path), removeHs=True)
        if fragment is None or not fragment.GetNumConformers():
            result['sides'][str(side)] = {'status': 'invalid_fragment_chemistry'}
            continue
        mapping = fragment_matches(molecule, fragment)
        coords, residues = protein_records(protein)
        pocket = local_pocket(coords, residues, fragment.GetConformer().GetPositions())
        result['sides'][str(side)] = {
            'status': mapping['mapping_status'], 'fragment_atoms': fragment.GetNumAtoms(),
            'files': {str(p): sha256_file(p) for p in (protein, path, inputs['ligand'])},
            **mapping, 'pocket': pocket, 'diagnostics': mapping_diagnostics(molecule, fragment),
        }
        if mapping['mapping_status'] == 'unique':
            atom_map = mapping['atom_maps'][0]
            result['sides'][str(side)]['attachment_bonds'] = attachment_bonds(molecule, atom_map)
            maps.append(set(atom_map))
    if len(maps) == 2:
        result['fragment_overlap'] = sorted(maps[0] & maps[1])
        result['linker_atoms'] = sorted(set(range(molecule.GetNumAtoms())) - maps[0] - maps[1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('condition audits require a new output file')
    entries = json.loads(args.index.read_text())['entries']
    rows = []
    for entry in entries:
        try:
            rows.append(audit_case(entry, ROOT))
        except Exception as error:
            rows.append({'case': entry['case'], 'status': 'audit_failed',
                         'error': f'{type(error).__name__}: {error}',
                         'ready_for_conditioned_inference': False})
    summary = {'cases': len(rows), 'audit_failures': sum(r.get('status') == 'audit_failed' for r in rows),
               'unique_side_mappings': sum(s.get('status') == 'unique' for r in rows for s in r.get('sides', {}).values()),
               'ready_cases': sum(r['ready_for_conditioned_inference'] for r in rows),
               'sides_with_element_connectivity_match': sum(
                   s.get('diagnostics', {}).get('element_connectivity_match', False)
                   for r in rows for s in r.get('sides', {}).values()),
               'sides_matching_chemistry_without_stereo': sum(
                   s.get('diagnostics', {}).get('chemistry_match_ignoring_stereo', False)
                   for r in rows for s in r.get('sides', {}).values())}
    payload = {'protocol': CONDITION_SCHEMA_VERSION, 'index_sha256': sha256_file(args.index),
               'scope': 'asset/mapping diagnostic only, provenance review required before B-mode use',
               'summary': summary, 'cases': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

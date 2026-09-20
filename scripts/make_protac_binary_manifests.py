"""Build deployable high-confidence PROTAC manifests from audited binary poses."""

import argparse
import ast
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from evaluation_common import sha256_file
from make_protac_pocket_manifests import receptor_node_coordinates
from tricoopdiff.condition_manifest import (MANIFEST_VERSION, canonical_isomeric_smiles,
                                            load_condition_manifest)


TRUSTED_METHODS = {'exact_stereo': 1., 'charge_normalized_exact_stereo': .9}
AUDIT_METHODS = {**TRUSTED_METHODS, 'conservative_charge_normalized_mcs': .5}


def molecule(path):
    value = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)[0]
    if value is None or value.GetNumConformers() != 1:
        raise ValueError(f'unreadable conditioned molecule: {path}')
    return value


def mapped_fragment_payload(fragment, mapping):
    """Canonicalize the mapped induced fragment and keep maps/coords aligned."""
    fragment_maps = mapping.get('fragment_atom_maps')
    if fragment_maps is None:
        fragment_maps = [list(range(fragment.GetNumAtoms()))
                         for _ in mapping['atom_maps']]
    if len(fragment_maps) != len(mapping['atom_maps']):
        raise ValueError('fragment/full symmetry map counts differ')
    selected = sorted(set(fragment_maps[0]))
    if not selected or any(sorted(values) != selected for values in fragment_maps):
        raise ValueError('symmetry maps do not cover one consistent fragment subgraph')
    source = fragment
    if len(selected) != fragment.GetNumAtoms():
        source = Chem.Mol(fragment)
        Chem.Kekulize(source, clearAromaticFlags=True)
    editable = Chem.RWMol()
    old_to_new = {}
    for old_index in selected:
        old_to_new[old_index] = editable.AddAtom(Chem.Atom(source.GetAtomWithIdx(old_index)))
    for bond in source.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in old_to_new and end in old_to_new:
            editable.AddBond(old_to_new[begin], old_to_new[end], bond.GetBondType())
    subgraph = editable.GetMol()
    Chem.SanitizeMol(subgraph)
    fragment_smiles = Chem.MolToSmiles(subgraph, canonical=True, isomericSmiles=True)
    smiles_order = ast.literal_eval(subgraph.GetProp('_smilesAtomOutputOrder'))
    if sorted(smiles_order) != list(range(subgraph.GetNumAtoms())):
        raise ValueError('RDKit did not report a complete mapped-fragment SMILES order')
    atom_maps = []
    for fragment_indices, full_indices in zip(fragment_maps, mapping['atom_maps']):
        by_fragment = dict(zip(fragment_indices, full_indices))
        atom_maps.append([by_fragment[selected[subgraph_index]]
                          for subgraph_index in smiles_order])
    coordinates = np.asarray(fragment.GetConformer().GetPositions(), dtype=np.float32)
    coordinates = coordinates[selected][smiles_order]
    return fragment_smiles, atom_maps, coordinates


def build_side(case_side, protein_path, allowed_methods=TRUSTED_METHODS):
    mapping = case_side.get('mapping', {})
    method = mapping.get('role_mapping_method')
    role_atoms = mapping.get('unambiguous_role_atoms')
    if method not in allowed_methods or role_atoms is None:
        return None, {'status': 'withheld', 'reason': 'mapping_not_high_confidence',
                      'mapping_method': method}
    fragment_path = Path(case_side['restored_sdf'])
    fragment = molecule(fragment_path)
    protein_coords = receptor_node_coordinates(protein_path)
    all_fragment_coords = np.asarray(
        fragment.GetConformer().GetPositions(), dtype=np.float32)
    distances = np.linalg.norm(protein_coords[:, None] - all_fragment_coords[None], axis=-1)
    pocket = (distances.min(1) <= 6.).astype(np.float32)
    fragment_smiles, atom_maps, fragment_coords = mapped_fragment_payload(fragment, mapping)
    payload = {
        'provenance': 'template',
        'evidence_id': f'binary-role-v3:{sha256_file(fragment_path)}:{sha256_file(protein_path)}',
        'confidence': allowed_methods[method],
        'fragment_smiles': fragment_smiles,
        'atom_maps': atom_maps, 'mapping_method': method,
        'audit_only': method == 'conservative_charge_normalized_mcs',
        'pocket_probabilities': pocket.tolist(),
        'binary_fragment_coords': fragment_coords.tolist(),
        'binary_protein_coords': protein_coords.tolist(),
    }
    return payload, {
        'status': 'included', 'mapping_method': method, 'role_atoms': len(role_atoms),
        'symmetry_maps': len(atom_maps), 'fragment_coverage': mapping['fragment_coverage'],
        'pocket_nodes': int(pocket.sum()), 'fragment_sha256': sha256_file(fragment_path),
        'protein_sha256': sha256_file(protein_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--chemistry-report', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--absent-out', type=Path,
                        help='Optional paired directory with all conditions absent')
    parser.add_argument('--allow-conservative-mcs', action='store_true',
                        help='Build a lower-confidence audit set; never use as deployment evidence')
    args = parser.parse_args(argv)
    allowed_methods = AUDIT_METHODS if args.allow_conservative_mcs else TRUSTED_METHODS
    args.out.mkdir(parents=True, exist_ok=False)
    if args.absent_out is not None:
        args.absent_out.mkdir(parents=True, exist_ok=False)
    entries = json.loads(args.index.read_text())['entries']
    report = json.loads(args.chemistry_report.read_text())
    report_cases = {row['case']: row for row in report['cases']}
    records = []
    for entry in entries:
        case = entry['case']
        full_path = args.root / entry['ligand']
        full = molecule(full_path)
        sides, audit, role_sets, protein_sizes = [], [], [], []
        for number in (1, 2):
            protein_path = args.root / entry[f'protein{number}']
            side, side_audit = build_side(
                report_cases[case]['sides'][str(number)], protein_path, allowed_methods)
            sides.append(side)
            audit.append(side_audit)
            protein_sizes.append(len(receptor_node_coordinates(protein_path)))
            role_sets.append(set(side_audit.get('role_atoms') and
                                 report_cases[case]['sides'][str(number)]['mapping']
                                 ['unambiguous_role_atoms'] or ()))
        if role_sets[0] & role_sets[1]:
            sides[1] = None
            audit[1] = {'status': 'withheld', 'reason': 'endpoint_role_overlap'}
            role_sets[1] = set()
        linker = sorted(set(range(full.GetNumAtoms())) - role_sets[0] - role_sets[1]) \
            if all(sides) else []
        payload = {
            'schema': MANIFEST_VERSION, 'case': case,
            'ligand_atom_count': full.GetNumAtoms(),
            'ligand_canonical_isomeric_smiles': canonical_isomeric_smiles(full),
            'sides': sides, 'linker_indices': linker, 'protac': True,
        }
        destination = args.out / f'{case}.json'
        destination.write_text(json.dumps(payload, indent=2) + '\n')
        load_condition_manifest(destination, case=case, molecule=full,
                                protein_sizes=protein_sizes)
        if args.absent_out is not None:
            absent = {**payload, 'sides': [None, None], 'linker_indices': []}
            absent_path = args.absent_out / f'{case}.json'
            absent_path.write_text(json.dumps(absent, indent=2) + '\n')
            load_condition_manifest(absent_path, case=case, molecule=full,
                                    protein_sizes=protein_sizes)
        records.append({'case': case, 'sides': audit, 'bilateral': all(sides),
                        'linker_atoms': len(linker)})
    summary = {
        'protocol': ('protac_conservative_mcs_audit_manifests_v1'
                     if args.allow_conservative_mcs else
                     'protac_high_confidence_binary_manifests_v3'),
        'cases': len(records), 'bilateral_cases': sum(row['bilateral'] for row in records),
        'included_sides': sum(side['status'] == 'included' for row in records
                              for side in row['sides']),
        'native_ternary_geometry_read': False,
        'mapping_methods_allowed': sorted(allowed_methods),
        'deployment_eligible': not args.allow_conservative_mcs,
        'index_sha256': sha256_file(args.index),
        'chemistry_report_sha256': sha256_file(args.chemistry_report),
        'records': records,
    }
    (args.out / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in
                      ('cases', 'bilateral_cases', 'included_sides',
                       'native_ternary_geometry_read')}, indent=2))


if __name__ == '__main__':
    main()

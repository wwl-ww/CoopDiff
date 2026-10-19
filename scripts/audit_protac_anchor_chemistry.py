"""Audit binary PROTAC anchors using CCD chemistry, without ternary coordinates.

The full PROTAC SDF contributes its molecular graph only. Binary ligand PDBs
contribute observed atom names and local binary-pose coordinates; their bonds
and bond orders come from CCD. This command never reads a native complex.
"""

import argparse
import hashlib
import json
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem
from rdkit.Chem import rdFMCS


BOND_TYPES = {'SING': Chem.BondType.SINGLE, 'DOUB': Chem.BondType.DOUBLE,
              'TRIP': Chem.BondType.TRIPLE, 'AROM': Chem.BondType.AROMATIC}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ccd_fragment(pdb_path, ccd_path, *, component_id=None):
    """Rebuild only observed binary-ligand atoms with CCD-defined chemistry."""
    lines = [line for line in Path(pdb_path).read_text().splitlines()
             if line.startswith('HETATM') and line[76:78].strip().upper() not in ('H', 'D')
             and (component_id is None or line[17:20].strip() == component_id)]
    if not lines:
        raise ValueError('no observed heavy ligand atoms')
    component = lines[0][17:20].strip()
    if {line[17:20].strip() for line in lines} != {component}:
        raise ValueError('multiple component names in one binary ligand')
    ccd = MMCIF2Dict(str(ccd_path))
    if ccd['_chem_comp.id'][0] != component:
        raise ValueError('CCD component ID differs from PDB residue')
    definition = {name: (element, int(charge)) for name, element, charge in zip(
        ccd['_chem_comp_atom.atom_id'], ccd['_chem_comp_atom.type_symbol'],
        ccd['_chem_comp_atom.charge']) if element not in ('H', 'D')}
    aliases = {}
    for canonical, alternate in zip(ccd['_chem_comp_atom.atom_id'],
                                    ccd.get('_chem_comp_atom.alt_atom_id', [])):
        if canonical in definition and alternate not in ('.', '?'):
            aliases.setdefault(alternate, set()).add(canonical)
    observed = []
    for line in lines:
        raw_name = line[12:16].strip()
        name = raw_name
        if name not in definition and raw_name in aliases and len(aliases[raw_name]) == 1:
            name = next(iter(aliases[raw_name]))
        if name not in definition:
            raise ValueError(f'atom {raw_name} is absent from CCD')
        element = line[76:78].strip().capitalize()
        if Chem.Atom(element).GetSymbol() != Chem.Atom(definition[name][0].capitalize()).GetSymbol():
            raise ValueError(f'element mismatch at {raw_name}')
        observed.append((name, line))
    names = [name for name, _ in observed]
    if len(set(names)) != len(names):
        raise ValueError('duplicate or aliased atom names')
    molecule = Chem.RWMol()
    conformer = Chem.Conformer(len(observed))
    lookup = {}
    for index, (name, line) in enumerate(observed):
        element, charge = definition[name]
        atom = Chem.Atom(element.capitalize())
        atom.SetFormalCharge(charge)
        lookup[name] = molecule.AddAtom(atom)
        conformer.SetAtomPosition(index, [float(line[start:start+8])
                                          for start in (30, 38, 46)])
    for left, right, order in zip(ccd.get('_chem_comp_bond.atom_id_1', []),
                                  ccd.get('_chem_comp_bond.atom_id_2', []),
                                  ccd.get('_chem_comp_bond.value_order', [])):
        if left in lookup and right in lookup:
            if order not in BOND_TYPES:
                raise ValueError(f'unknown CCD bond order {order}')
            molecule.AddBond(lookup[left], lookup[right], BOND_TYPES[order])
    molecule = molecule.GetMol()
    molecule.AddConformer(conformer)
    Chem.SanitizeMol(molecule)
    # Incomplete CCD subsets can lose terminal bonds; report, do not invent.
    return molecule, {'component': component, 'observed_heavy_atoms': len(names),
        'ccd_heavy_atoms': len(definition),
        'missing_ccd_atoms': sorted(set(definition)-set(names)),
        'observed_connected_components': len(Chem.GetMolFrags(molecule)),
        'source_pdb_sha256': sha256(pdb_path), 'ccd_sha256': sha256(ccd_path)}


def match_graph(full, fragment, timeout):
    """Enumerate chemically ordered common-core matches for joint selection."""
    result = rdFMCS.FindMCS([full, fragment], timeout=timeout,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True, completeRingsOnly=True)
    query = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
    matches = (full.GetSubstructMatches(query, uniquify=False, maxMatches=1000)
               if query is not None else ())
    return matches, {'common_core_atoms': result.numAtoms,
        'fragment_atom_count': fragment.GetNumAtoms(),
        'core_coverage': round(result.numAtoms/fragment.GetNumAtoms(), 4),
        'search_timed_out': bool(result.canceled),
        'full_graph_match_count': len(matches)}


def audit_case(entry, ccd_dir, timeout):
    full = Chem.MolFromMolFile(entry['ligand'], removeHs=True)
    if full is None:
        raise ValueError('full PROTAC SDF cannot be parsed')
    full.RemoveAllConformers()  # Forbid any native conformer information.
    directory = Path(entry['protein1']).parent
    reports, matches = [], []
    for side in (1, 2):
        pdb = directory/f'unbound_lig{side}.pdb'
        lines = [line for line in pdb.read_text().splitlines() if line.startswith('HETATM')]
        if not lines:
            raise ValueError(f'binary ligand {side} is empty')
        component = lines[0][17:20].strip()
        ccd_path = ccd_dir/f'{component}.cif'
        fragment, chemistry = ccd_fragment(pdb, ccd_path)
        selections, mapping = match_graph(full, fragment, timeout)
        reports.append({'side': side, **chemistry, **mapping})
        matches.append(selections)
    disjoint = sum(not set(left).intersection(right)
                   for left in matches[0] for right in matches[1])
    # This is an audit state, not permission to use an arbitrary MCS as a pose
    # constraint. Endpoint chemistry and mapping ambiguity still need review.
    if any(side['search_timed_out'] for side in reports):
        status = 'unresolved_timeout'
    elif not disjoint:
        status = 'no_disjoint_exact_bond_core'
    elif any(side['core_coverage'] < .75 for side in reports):
        status = 'reference_or_chemistry_mismatch'
    elif any(side['missing_ccd_atoms'] for side in reports):
        status = 'partial_binary_observation_review'
    else:
        status = 'dual_core_mapping_review'
    return {'case': entry['case'], 'status': status,
            'full_protac_heavy_atoms': full.GetNumAtoms(),
            'full_protac_graph_sha256': sha256(entry['ligand']),
            'joint_disjoint_match_pairs': disjoint,
            'sides': reports,
            'native_coordinates_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--ccd-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=10)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error('timeout must be positive')
    entries = json.loads(args.index.read_text())['entries']
    cases = []
    for entry in entries:
        try:
            row = audit_case(entry, args.ccd_dir, args.timeout)
        except Exception as error:
            row = {'case': entry['case'], 'status': 'audit_error',
                   'error': f'{type(error).__name__}: {error}',
                   'native_coordinates_used': False}
        cases.append(row)
        print(f"{row['case']} {row['status']}", flush=True)
    summary = {'protocol': 'ccd_binary_graph_joint_mcs_audit_v1',
               'requested': len(entries), 'completed': len(cases),
               'source_index_sha256': sha256(args.index),
               'native_coordinates_used': False,
               'cases': cases}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

"""Restore a complete deposited ligand from an explicit, offline CCD definition.

Coordinates and heavy-atom order come from the PDB. Bond orders and charges
come from the CCD. Missing atoms or modified components require a separate
protocol: this command never invents coordinates or guesses their chemistry.
"""

import argparse
import hashlib
import json
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem


def restore_ligand(pdb_path, ccd_path, destination, *, allow_observed_subset=False):
    pdb_path, ccd_path, destination = map(Path, (pdb_path, ccd_path, destination))
    deposited = Chem.MolFromPDBFile(str(pdb_path), sanitize=False,
                                   removeHs=False, proximityBonding=False)
    if deposited is None or deposited.GetNumConformers() != 1:
        raise ValueError('expected one readable deposited conformer')
    atoms = [a for a in deposited.GetAtoms() if a.GetAtomicNum() > 1]
    if not atoms or any(a.GetPDBResidueInfo() is None for a in atoms):
        raise ValueError('heavy atoms must have deposited atom names')
    names = [a.GetPDBResidueInfo().GetName().strip() for a in atoms]
    if len(set(names)) != len(names):
        raise ValueError('ambiguous repeated atom names or multiple ligand copies')
    ccd = MMCIF2Dict(str(ccd_path))
    component = ccd['_chem_comp.id'][0]
    if {a.GetPDBResidueInfo().GetResidueName().strip() for a in atoms} != {component}:
        raise ValueError('deposited residue name does not match CCD component')
    definition = {
        name: (element, int(charge))
        for name, element, charge in zip(ccd['_chem_comp_atom.atom_id'],
                                         ccd['_chem_comp_atom.type_symbol'],
                                         ccd['_chem_comp_atom.charge'])
        if element not in ('H', 'D')
    }
    deposited_names = list(names)
    aliases = {}
    for canonical, alternate in zip(ccd['_chem_comp_atom.atom_id'],
                                     ccd.get('_chem_comp_atom.alt_atom_id', [])):
        if canonical in definition and alternate not in ('.', '?'):
            aliases.setdefault(alternate, set()).add(canonical)
    for index, name in enumerate(names):
        if name not in definition and name in aliases:
            if len(aliases[name]) != 1:
                raise ValueError(f'ambiguous CCD atom-name alias: {name}')
            names[index] = next(iter(aliases[name]))
    if len(set(names)) != len(names):
        raise ValueError('CCD aliases map multiple deposited atoms to one atom')
    missing = sorted(set(definition) - set(names))
    extra = sorted(set(names) - set(definition))
    if extra or (missing and not allow_observed_subset):
        raise ValueError(f'incomplete or modified component: missing '
                         f'{missing}; extra {extra}')
    molecule = Chem.RWMol()
    positions = Chem.Conformer(len(atoms))
    positions.Set3D(True)
    for index, (name, source) in enumerate(zip(names, atoms)):
        element, charge = definition[name]
        # CCD uses uppercase symbols (CL/BR); RDKit expects Cl/Br.
        atom = Chem.Atom(element.capitalize())
        if atom.GetAtomicNum() != source.GetAtomicNum():
            raise ValueError(f'element mismatch for {name}')
        atom.SetFormalCharge(charge)
        atom.SetProp('ccd_atom_name', name)
        molecule.AddAtom(atom)
        positions.SetAtomPosition(index, deposited.GetConformer().GetAtomPosition(source.GetIdx()))
    lookup = {name: index for index, name in enumerate(names)}
    types = {'SING': Chem.BondType.SINGLE, 'DOUB': Chem.BondType.DOUBLE,
             'TRIP': Chem.BondType.TRIPLE, 'AROM': Chem.BondType.AROMATIC}
    hydrogen_names = {name for name, element in zip(
        ccd['_chem_comp_atom.atom_id'], ccd['_chem_comp_atom.type_symbol'])
        if element in ('H', 'D')}
    for left, right, order in zip(ccd.get('_chem_comp_bond.atom_id_1', []),
                                  ccd.get('_chem_comp_bond.atom_id_2', []),
                                  ccd.get('_chem_comp_bond.value_order', [])):
        if any(name not in lookup and name not in hydrogen_names
               and name not in definition for name in (left, right)):
            raise ValueError(f'CCD bond references an undefined atom: {left}, {right}')
        if left in hydrogen_names or right in hydrogen_names:
            continue  # Explicit CCD hydrogens are omitted from the heavy graph.
        if left not in lookup or right not in lookup:
            continue  # Explicit subset mode retains only observed heavy atoms and bonds.
        if order not in types:
            raise ValueError(f'unsupported CCD bond order: {order}')
        molecule.AddBond(lookup[left], lookup[right], types[order])
    molecule = molecule.GetMol()
    molecule.AddConformer(positions)
    Chem.SanitizeMol(molecule)
    Chem.AssignStereochemistryFrom3D(molecule)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError('CCD component is disconnected; single-ligand contract requires review')
    report = {
        'component': component, 'source_pdb': str(pdb_path.resolve()),
        'source_pdb_sha256': hashlib.sha256(pdb_path.read_bytes()).hexdigest(),
        'ccd_path': str(ccd_path.resolve()),
        'ccd_sha256': hashlib.sha256(ccd_path.read_bytes()).hexdigest(),
        'atom_names_in_order': deposited_names,
        'canonical_atom_names_in_order': names,
        'atom_name_aliases': {raw: canonical for raw, canonical in zip(deposited_names, names)
                              if raw != canonical},
        'heavy_atoms': len(atoms),
        'formal_charge': Chem.GetFormalCharge(molecule),
        'aromatic_atoms': sum(a.GetIsAromatic() for a in molecule.GetAtoms()),
        'smiles': Chem.MolToSmiles(molecule),
        'observed_subset': bool(missing), 'missing_heavy_atom_names': missing,
    }
    molecule.SetProp('chemistry_provenance', json.dumps(report, sort_keys=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with Chem.SDWriter(str(temporary)) as writer:
        writer.write(molecule)
    temporary.replace(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdb', required=True)
    parser.add_argument('--ccd', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(restore_ligand(args.pdb, args.ccd, args.out), indent=2))


if __name__ == '__main__':
    main()

"""Restore a complete deposited ligand from an explicit, offline CCD definition.

Coordinates and heavy-atom order come from the PDB. Bond orders and charges
come from the CCD. Missing atoms or modified components require a separate
protocol: this command never invents coordinates or guesses their chemistry.
"""

import hashlib
import json
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem


def restore_ligand(pdb_path, ccd_path, destination):
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
    if set(names) != set(definition):
        raise ValueError(f'incomplete or modified component: missing '
                         f'{sorted(set(definition) - set(names))}; extra '
                         f'{sorted(set(names) - set(definition))}')
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
        if any(name not in lookup and name not in hydrogen_names for name in (left, right)):
            raise ValueError(f'CCD bond references an undefined atom: {left}, {right}')
        if left in hydrogen_names or right in hydrogen_names:
            continue  # Explicit CCD hydrogens are omitted from the heavy graph.
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
    }
    molecule.SetProp('chemistry_provenance', json.dumps(report, sort_keys=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with Chem.SDWriter(str(temporary)) as writer:
        writer.write(molecule)
    temporary.replace(destination)
    return report



CHEMISTRY_CONTRACT = 'bond_order_legal_torsions_v1'
LEGACY_CHEMISTRY_CONTRACT = 'legacy_topology_v9'


def is_legal_torsion(mol, u, v):
    """Single, non-ring, non-terminal heavy-atom bond, excluding amides.

    Carbonyl/thio-carbonyl C--N bonds are constrained by conjugation; they are
    not independent uniform torsions. This is deliberately a chemistry rule,
    independent of a conformer's coordinates or a model's predicted motion.
    """
    bond = mol.GetBondBetweenAtoms(int(u), int(v))
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    if bond.GetIsAromatic() or bond.IsInRing():
        return False
    atoms = [mol.GetAtomWithIdx(int(i)) for i in (u, v)]
    if any(a.GetAtomicNum() <= 1 or
           sum(n.GetAtomicNum() > 1 for n in a.GetNeighbors()) <= 1 for a in atoms):
        return False
    for carbon, nitrogen in (atoms, atoms[::-1]):
        if carbon.GetAtomicNum() == 6 and nitrogen.GetAtomicNum() == 7:
            if any(b.GetBondType() == Chem.BondType.DOUBLE and
                   b.GetOtherAtom(carbon).GetAtomicNum() in (8, 16)
                   for b in carbon.GetBonds()):
                return False
    return True


def legal_torsion_masks(mol, edge_index):
    """Build executor and expert masks together in directed edge order."""
    import numpy as np
    edges = edge_index.detach().cpu().T.tolist()
    selected = np.zeros(len(edges), dtype=bool)
    masks = []
    seen = set()
    for index, (u, v) in enumerate(edges):
        key = tuple(sorted((u, v)))
        if key in seen or not is_legal_torsion(mol, u, v):
            continue
        seen.add(key)
        side, pending = {v}, [v]
        while pending:
            atom = pending.pop()
            for neighbor in mol.GetAtomWithIdx(atom).GetNeighbors():
                j = neighbor.GetIdx()
                if {atom, j} == {u, v}:
                    continue
                if j not in side:
                    side.add(j)
                    pending.append(j)
        if u in side:
            raise ValueError('non-ring torsion axis did not split molecule')
        # Preserve the existing smaller-fragment convention, deterministically.
        if len(side) > mol.GetNumAtoms() - len(side):
            reverse = edges.index([v, u])
            axis_index = reverse
            selected[reverse] = True
            side = set(range(mol.GetNumAtoms())) - side
        else:
            axis_index = index
            selected[index] = True
        masks.append((axis_index, side))
    # Reverse edges need not be adjacent: align rows with selected edge order.
    masks.sort(key=lambda item: item[0])
    result = np.zeros((len(masks), mol.GetNumAtoms()), dtype=bool)
    for row, (_, side) in enumerate(masks):
        result[row, list(side)] = True
    return selected, result

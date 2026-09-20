"""Offline CCD restoration must preserve geometry and reject incomplete inputs."""

import sys
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from restore_ligand_chemistry import restore_ligand


def fixture(root):
    molecule = Chem.MolFromSmiles('CC(=O)[O-]')
    conformer = Chem.Conformer(4)
    coordinates = np.array([[1., 2., 3.], [2.4, 2., 3.],
                            [3., 3., 3.], [3., 1., 3.]])
    for index, (atom, name) in enumerate(zip(molecule.GetAtoms(), ('C2', 'C1', 'O1', 'O2'))):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(name.rjust(4))
        info.SetResidueName('TST')
        info.SetResidueNumber(1)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
        conformer.SetAtomPosition(index, coordinates[index])
    molecule.AddConformer(conformer)
    pdb = root / 'ligand.pdb'
    # Match the atom-only export that caused loss of bond-order information.
    pdb.write_text('\n'.join(line for line in Chem.MolToPDBBlock(molecule).splitlines()
                              if line.startswith('HETATM')) + '\nEND\n')
    ccd = root / 'TST.cif'
    ccd.write_text('''data_TST
_chem_comp.id TST
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
O2 O -1
C1 C 0
C2 C 0
O1 O 0
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
C1 O1 DOUB
C1 C2 SING
C1 O2 SING
''')
    return pdb, ccd, coordinates


def test_restores_bond_order_and_charge_without_reordering(tmp_path):
    pdb, ccd, positions = fixture(tmp_path)
    output = tmp_path / 'restored.sdf'
    report = restore_ligand(pdb, ccd, output)
    molecule = Chem.SDMolSupplier(str(output), removeHs=True)[0]
    assert molecule is not None
    assert Chem.GetFormalCharge(molecule) == -1
    assert molecule.GetBondBetweenAtoms(1, 2).GetBondType() == Chem.BondType.DOUBLE
    assert report['atom_names_in_order'] == ['C2', 'C1', 'O1', 'O2']
    assert np.allclose(molecule.GetConformer().GetPositions(), positions, atol=1e-5)
    assert molecule.HasProp('chemistry_provenance')


def test_missing_atoms_are_not_invented(tmp_path):
    pdb, ccd, _ = fixture(tmp_path)
    pdb.write_text('\n'.join(line for line in pdb.read_text().splitlines()
                              if line[12:16].strip() != 'O2') + '\n')
    output = tmp_path / 'must_not_exist.sdf'
    try:
        restore_ligand(pdb, ccd, output)
    except ValueError as error:
        assert 'missing' in str(error) and 'O2' in str(error)
    else:
        raise AssertionError('incomplete component was silently reconstructed')
    assert not output.exists()


def test_explicit_subset_mode_restores_only_observed_ccd_atoms(tmp_path):
    pdb, ccd, _ = fixture(tmp_path)
    pdb.write_text('\n'.join(line for line in pdb.read_text().splitlines()
                              if line[12:16].strip() != 'O2') + '\n')
    output = tmp_path / 'observed.sdf'
    report = restore_ligand(pdb, ccd, output, allow_observed_subset=True)
    molecule = Chem.SDMolSupplier(str(output), removeHs=True)[0]
    assert molecule.GetNumAtoms() == 3
    assert report['observed_subset']
    assert report['missing_heavy_atom_names'] == ['O2']


def test_uppercase_ccd_halogen_symbols(tmp_path):
    pdb, ccd, _ = fixture(tmp_path)
    ccd.write_text(ccd.read_text().replace('O2 O -1', 'O2 CL 0'))
    lines = pdb.read_text().splitlines()
    pdb.write_text('\n'.join(line[:76] + 'CL' + line[78:]
                              if line[12:16].strip() == 'O2' else line
                              for line in lines) + '\n')
    output = tmp_path / 'halogen.sdf'
    restore_ligand(pdb, ccd, output)
    molecule = Chem.SDMolSupplier(str(output))[0]
    assert molecule.GetAtomWithIdx(3).GetSymbol() == 'Cl'
    assert Chem.GetFormalCharge(molecule) == 0


def test_uses_explicit_ccd_aliases_without_guessing(tmp_path):
    pdb, ccd, coordinates = fixture(tmp_path)
    content = ccd.read_text().replace('_chem_comp_atom.type_symbol',
                                    '_chem_comp_atom.alt_atom_id\n_chem_comp_atom.type_symbol')
    for before, after in [('O2 O -1', 'O2 O2 O -1'), ('C1 C 0', 'CX C1 C 0'),
                          ('C2 C 0', 'C2 C2 C 0'), ('O1 O 0', 'O1 O1 O 0'),
                          ('C1 O1', 'CX O1'), ('C1 C2', 'CX C2'), ('C1 O2', 'CX O2')]:
        content = content.replace(before, after)
    ccd.write_text(content)
    output = tmp_path / 'aliased.sdf'
    report = restore_ligand(pdb, ccd, output)
    assert report['atom_name_aliases'] == {'C1': 'CX'}
    assert report['atom_names_in_order'] == ['C2', 'C1', 'O1', 'O2']
    molecule = Chem.SDMolSupplier(str(output))[0]
    assert np.allclose(molecule.GetConformer().GetPositions(), coordinates)


def test_undefined_bond_endpoint_is_not_treated_as_hydrogen(tmp_path):
    pdb, ccd, _ = fixture(tmp_path)
    ccd.write_text(ccd.read_text() + 'C1 UNKNOWN SING\n')
    output = tmp_path / 'must_not_exist.sdf'
    try:
        restore_ligand(pdb, ccd, output)
    except ValueError as error:
        assert 'undefined atom' in str(error)
    else:
        raise AssertionError('malformed CCD bond was silently ignored')
    assert not output.exists()


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            with tempfile.TemporaryDirectory() as directory:
                test(Path(directory))
            print(name, 'passed')

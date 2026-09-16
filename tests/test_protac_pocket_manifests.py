import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from make_protac_pocket_manifests import build_case, pdb_coordinates, pocket_probabilities


def pdb_line(record, serial, name, residue, x, y, z, element):
    return (f'{record:<6}{serial:5d} {name:^4} {residue:>3} A{1:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{20.:6.2f}          {element:>2}\n')


def test_binary_template_pockets_are_local_and_node_aligned(tmp_path):
    case_dir = tmp_path / 'case'
    case_dir.mkdir()
    protein = ''.join([
        pdb_line('ATOM', 1, 'CA', 'ALA', 0, 0, 0, 'C'),
        pdb_line('ATOM', 2, 'CA', 'GLY', 10, 0, 0, 'C'),
        'END\n',
    ])
    fragment = pdb_line('HETATM', 1, 'C1', 'LIG', 1, 0, 0, 'C') + 'END\n'
    for side in (1, 2):
        (case_dir / f'protein{side}.pdb').write_text(protein)
        (case_dir / f'unbound_lig{side}.pdb').write_text(fragment)
    molecule = Chem.AddHs(Chem.MolFromSmiles('CC'))
    AllChem.EmbedMolecule(molecule, randomSeed=7)
    Chem.MolToMolFile(molecule, str(case_dir / 'ligand.sdf'))
    entry = {
        'case': 'fixture', 'protein1': 'case/protein1.pdb',
        'protein2': 'case/protein2.pdb', 'ligand': 'case/ligand.sdf',
    }
    pocket, absent, report = build_case(entry, tmp_path, 6.)
    assert pocket['sides'][0]['pocket_probabilities'] == [1., 0.]
    assert pocket['sides'][1]['pocket_probabilities'] == [1., 0.]
    assert absent['sides'] == [None, None]
    assert report['sides'][0]['pocket_ca_nodes'] == 1
    assert pdb_coordinates(case_dir / 'protein1.pdb', alpha_carbons=True).shape == (2, 3)
    values, _, _ = pocket_probabilities(
        case_dir / 'protein1.pdb', case_dir / 'unbound_lig1.pdb', 6.)
    assert np.array_equal(values, [1., 0.])


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory(prefix='protac-pocket-test-') as directory:
        test_binary_template_pockets_are_local_and_node_aligned(Path(directory))

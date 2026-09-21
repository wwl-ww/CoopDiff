"""Partial-core features preserve known sides and do not fabricate linker roles."""

from pathlib import Path
import hashlib
import json
import sys

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.condition_features import condition_feature_shapes
from tricoopdiff.conserved_core_conditions import merge_side_features, augment_conserved_core_conditions


def test_merge_keeps_known_side_and_unknown_complement():
    base = {key: torch.zeros(shape) for key, shape in condition_feature_shapes([4, 5], 8).items()}
    base['sides'][0, :3] = 1
    base['ligand'][:3, 0] = 1
    base['binary1'].fill_(3)
    addition = {key: torch.zeros_like(value) for key, value in base.items()}
    addition['sides'][1, :3] = 1
    addition['ligand'][5:, 1] = 1
    addition['ligand'][5:, 9:] = 2
    merged = merge_side_features(base, addition, 1)
    assert torch.equal(merged['binary1'], base['binary1'])
    assert torch.equal(merged['sides'][0], base['sides'][0])
    assert merged['ligand'][3:5, 3].eq(1).all()
    assert merged['ligand'][:, 2].eq(0).all(), 'unknown atoms were mislabeled as linker'
    assert base['ligand'][:, 1].eq(0).all(), 'input features were mutated'


def test_merge_rejects_overlapping_ends():
    base = {key: torch.zeros(shape) for key, shape in condition_feature_shapes([4, 5], 8).items()}
    addition = {key: torch.zeros_like(value) for key, value in base.items()}
    base['ligand'][1, 0] = 1
    addition['ligand'][1, 1] = 1
    try:
        merge_side_features(base, addition, 1)
    except ValueError as error:
        assert 'overlaps' in str(error)
    else:
        raise AssertionError('overlapping core hypotheses were accepted')


def test_core_augmentation_uses_graph_not_full_ligand_coordinates(tmp_path):
    full = Chem.MolFromSmiles('CC(C)(C)c1ccc2ncccc2c1')
    binary = Chem.MolFromSmiles('Clc1ccc2ncccc2c1')
    irrelevant = Chem.MolFromSmiles('CCCO')
    for name, molecule in [('ligand', full), ('binary1', binary), ('binary2', irrelevant)]:
        AllChem.Compute2DCoords(molecule)
        with Chem.SDWriter(str(tmp_path / f'{name}.sdf')) as writer:
            writer.write(molecule)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for index in (1, 2):
        (tmp_path / f'unbound_protein{index}.pdb').write_text('synthetic protein identity')
    report = {'cases': [{'case': 'synthetic', 'full_ligand_sha256': digest(tmp_path / 'ligand.sdf'),
        'sides': {str(index): {'restored_sdf': str(tmp_path / f'binary{index}.sdf'),
                             'restored_sha256': digest(tmp_path / f'binary{index}.sdf')}
                  for index in (1, 2)}}]}
    path = tmp_path / 'chemistry.json'
    path.write_text(json.dumps(report))
    base = {key: torch.zeros(shape) for key, shape in condition_feature_shapes([3, 3], full.GetNumAtoms()).items()}
    base['ligand'][:, 3] = 1
    kwargs = dict(case='synthetic', case_dir=tmp_path, chemistry_report=path,
                  read_protein_coordinates=lambda _: [[0., 0., 3.], [1., 0., 3.], [0., 1., 3.]])
    first, sides, audit = augment_conserved_core_conditions(base, (None, None), full, **kwargs)
    modified = Chem.Mol(full)
    modified.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    for index in range(modified.GetNumAtoms()):
        modified.GetConformer().SetAtomPosition(index, (index * 20., index ** 2, -index))
    second, _, second_audit = augment_conserved_core_conditions(base, (None, None), modified, **kwargs)
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert audit['added_sides'] == second_audit['added_sides']
    assert second_audit['nonstereogenic_tag_cleanup_before_identity_labels']['feature_cache'] == [1]
    assert sides[0] is not None and sides[1] is None
    assert first['ligand'][:, 2].eq(0).all()
    assert audit['experimental'] and not audit['calibrated_template_uncertainty']

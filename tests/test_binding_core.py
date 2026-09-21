"""Core motion frees substituents without moving the protected binary frames."""

from pathlib import Path
import sys

import torch
from rdkit import Chem

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import tricoopdiff.diffdock_compat  # noqa: F401
from test_initialization import ligand_input
from tricoopdiff.binding_core import select_binding_core
from tricoopdiff.core_training import prepare_core_training_sample
from tricoopdiff.anchored_training import prepare_anchored_training_sample
from tricoopdiff.anchored_tasks import validate_task_sample


def fixture():
    ligand = ligand_input('c1ccccc1CCCCCCc1ccccc1')
    mol = Chem.Mol(ligand['mol'])
    cuts = [mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in ((7, 8), (9, 10))]
    maps = []
    fragments = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, cuts, addDummies=False),
        asMols=True, fragsMolAtomMapping=maps)
    ends = [(fragment, mapping) for fragment, mapping in zip(fragments, maps) if len(mapping) == 8]
    assert len(ends) == 2
    xyz = ligand['coords'][0]
    e3 = {'coords': (xyz[:6] + torch.tensor([0., 0., 2.]))[None]}
    poi = {'coords': (xyz[-6:] + torch.tensor([0., 0., 2.]))[None]}
    return e3, poi, ligand, tuple(x[0] for x in ends), tuple(x[1] for x in ends)


def test_core_training_releases_periphery_and_keeps_target_reachable():
    e3, poi, ligand, fragments, maps = fixture()
    kwargs = dict(fragments=fragments, atom_maps=maps, evidence_id='fixture')
    old = prepare_anchored_training_sample(e3, poi, ligand,
        generator=torch.Generator().manual_seed(7315), **kwargs)
    bounds = {'translation': [.5, .5], 'rotation': [.2, .2]}
    sample = prepare_core_training_sample(e3, poi, ligand,
        generator=torch.Generator().manual_seed(7315), binary_bounds=bounds, **kwargs)
    validate_task_sample(sample, 'dual_core', bounds, None)
    assert len(sample.flow.plan.bonds) > len(old.flow.plan.bonds), 'periphery remains locked'
    assert [len(x) for x in sample.audit['selected_atom_maps']] == [6, 6]
    decoded = sample.flow.decode(sample.data_angles)
    assert torch.allclose(decoded[0], ligand['coords'][0], atol=1e-4)
    assert torch.allclose(decoded[1], poi['coords'][0], atol=1e-4)
    strict = sample.flow.strict
    moved = strict.decode(torch.ones(len(strict.plan.bonds)) * .4)
    left, right = sample.audit['selected_atom_maps']
    assert torch.allclose(moved[0][left], strict.ligand[left], atol=1e-5)
    assert torch.allclose(torch.cdist(moved[0][right], moved[1]),
                          torch.cdist(strict.ligand[right], strict.poi), atol=1e-4)
    assert (moved[0][6:8] - strict.ligand[6:8]).abs().max() > .01
    assert sample.conditions['ligand'][6:8, 0].eq(1).all(), 'soft binary evidence must not be discarded'
    assert sample.conditions['ligand'][:, 2].eq(0).all(), 'must not invent linker identity'


def test_core_selection_is_frame_and_atom_order_invariant():
    e3, _, ligand, fragments, maps = fixture()
    coords = ligand['coords'][0, list(maps[0])]
    a = select_binding_core(fragments[0], maps[0], coords, e3['coords'][0])
    order = list(reversed(range(len(maps[0]))))
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    b = select_binding_core(Chem.RenumberAtoms(fragments[0], order),
        tuple(maps[0][i] for i in order), coords[order] @ rotation.T + 9,
        e3['coords'][0] @ rotation.T + 9)
    assert set(a['atom_map']) == set(b['atom_map'])


def test_unobserved_or_degenerate_core_is_not_invented():
    mol = Chem.MolFromSmiles('c1ccccc1')
    for xyz, protein in ((torch.zeros(6, 3), torch.zeros(2, 3)),
                         (torch.randn(6, 3), torch.ones(2, 3) * 100)):
        try:
            select_binding_core(mol, tuple(range(6)), xyz, protein)
        except ValueError as error:
            assert 'core' in str(error)
        else:
            raise AssertionError('unsupported core accepted')


def test_multiple_contact_blocks_are_distinct_from_one_dominant_block():
    data = ligand_input('c1ccccc1CCc1ccc2ccccc2c1')
    mol = Chem.Mol(data['mol'])
    xyz = data['coords'][0]
    protein = xyz + torch.tensor([0., 0., 1.])
    mapping = tuple(range(len(xyz)))
    one = select_binding_core(mol, mapping, xyz, protein)
    multiple = select_binding_core(mol, mapping, xyz, protein, all_contact_blocks=True)
    assert set(one['atom_map']) < set(multiple['atom_map'])
    assert multiple['audit']['protected_blocks'] >= 2
    from tricoopdiff.condition_features import SideCondition, validate_atom_maps
    validate_atom_maps(mol, SideCondition('experimental_binary', 'fixture',
        fragment=multiple['fragment'], atom_map=multiple['atom_map']))


def test_core_checkpoint_cannot_resume_as_whole_fragment_task():
    from test_anchored_trainer import trainer
    bounds = {'translation': [.5, .5], 'rotation': [.2, .2]}
    value = trainer(binary_bounds=bounds, conditioning_task='dual_core', coupled=True)
    payload = value.checkpoint(epoch=0, next_case=0, seed=7315, scope='implementation_smoke')
    restored = trainer(binary_bounds=bounds, conditioning_task='dual_core', coupled=True)
    restored.restore(payload)
    assert payload['fragment_policy'] == 'binary_local_contact_rigid_block_v1'
    try:
        trainer(binary_bounds=bounds, coupled=True).restore(payload)
    except ValueError as error:
        assert 'task' in str(error)
    else:
        raise AssertionError('core checkpoint silently resumed as whole-fragment training')


def test_core_manifest_uses_same_selection_and_no_full_ligand_coordinates(tmp_path):
    import json
    from tricoopdiff.condition_manifest import MANIFEST_VERSION, canonical_isomeric_smiles
    from tricoopdiff.core_conditions import load_core_conditions
    e3, poi, ligand, fragments, maps = fixture()
    mol = Chem.Mol(ligand['mol'])
    sides = []
    for fragment, mapping, protein in zip(fragments, maps, (e3, poi)):
        # Canonical SMILES atom order is recorded explicitly, not assumed.
        smiles = Chem.MolToSmiles(fragment)
        import ast
        order = list(ast.literal_eval(fragment.GetProp('_smilesAtomOutputOrder')))
        sides.append({'provenance': 'experimental_binary', 'evidence_id': 'fixture',
            'fragment_smiles': smiles, 'atom_maps': [[mapping[i] for i in order]],
            'binary_fragment_coords': ligand['coords'][0, [mapping[i] for i in order]].tolist(),
            'binary_protein_coords': protein['coords'][0].tolist()})
    path = tmp_path / 'condition.json'
    path.write_text(json.dumps({'schema': MANIFEST_VERSION, 'case': 'fixture',
        'ligand_atom_count': mol.GetNumAtoms(),
        'ligand_canonical_isomeric_smiles': canonical_isomeric_smiles(mol), 'sides': sides}))
    kwargs = dict(case='fixture', protein_sizes=[6, 6])
    first, poses, audit = load_core_conditions(path, molecule=mol, **kwargs)
    mol.RemoveAllConformers()
    second, _, other = load_core_conditions(path, molecule=mol, **kwargs)
    assert audit == other and audit['native_ternary_geometry_read'] is False
    for key in first:
        assert torch.equal(first[key], second[key])
    assert [len(side.atom_maps[0]) for side in poses] == [6, 6]
    assert first['ligand'][6:8, 0].eq(1).all()
    assert first['ligand'][8:10, 3].eq(1).all()

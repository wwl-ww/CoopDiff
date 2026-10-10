"""CPU geometric gates for the only supported broad initialization prior."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from rdkit import Chem
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tricoopdiff.initialization import (
    INITIAL_NOISE_STD_PROPORTION, TernaryInitializer, receptor_prior,
    validate_ligand_graph,
)
from tricoopdiff.geometry.rigid import apply_rigid, assemble_interface_pair, invert_rigid


def ligand_input(smiles="CCCOCC"):
    from torch_geometric.data import HeteroData
    from tricoopdiff.diffdock.datasets.process_mols import get_lig_graph_with_matching
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=3) == 0
    graph = HeteroData()
    get_lig_graph_with_matching(mol, graph, popsize=None, maxiter=None,
                               matching=False, keep_original=False,
                               num_conformers=1, remove_hs=True)
    return {
        "mol": Chem.RemoveHs(mol).ToBinary(),
        "coords": graph['ligand'].pos.unsqueeze(0),
        "edge_index": graph['ligand', 'ligand'].edge_index,
        "edge_mask": graph['ligand'].edge_mask,
        "mask_rotate": graph['ligand'].mask_rotate,
    }


def test_initializer_preserves_all_caller_random_streams():
    import random
    from tricoopdiff.runtime import capture_rng_state, restore_rng_state

    data = ligand_input()
    e3, poi = torch.randn(8, 3), torch.randn(9, 3)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng_state(state)
    first = TernaryInitializer().initialize(e3, poi, data, seed=73)
    second = TernaryInitializer().initialize(e3, poi, data, seed=73)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert expected[:2] == actual[:2] and torch.equal(expected[2], actual[2])
    assert all(torch.equal(a, b) for a, b in zip(first[:3], second[:3]))


def test_exact_inverse_and_pair_distances():
    torch.manual_seed(7)
    for seed in range(8):
        ra, rb = torch.tensor(Rotation.random(2, random_state=seed).as_matrix())
        ta, tb = torch.randn(2, 3, dtype=torch.float64) * 30
        ligand = torch.randn(14, 3, dtype=torch.float64)
        a, b = torch.randn(2, 23, 3, dtype=torch.float64) * 20
        la, lb = apply_rigid(ligand, ra, ta), apply_rigid(ligand, rb, tb)
        placed_b, rotation = assemble_interface_pair(b, ra, ta, rb, tb)
        inv, offset = invert_rigid(rb, tb)
        assert torch.allclose(apply_rigid(lb, inv, offset), ligand, atol=1e-10)
        assert torch.allclose(rotation.T @ rotation, torch.eye(3).double(), atol=1e-10)
        assert torch.det(rotation) > 0
        assert (torch.cdist(placed_b, la) - torch.cdist(b, lb)).abs().max() < 1e-5
        # The A gauge stays unchanged; all A-side distances are preserved too.
        central_a = apply_rigid(a, *invert_rigid(ra, ta))
        assert (torch.cdist(central_a, ligand) - torch.cdist(a, la)).abs().max() < 1e-5


def test_receptor_scale_and_rejected_legacy_parameters():
    receptor = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 2., 0.]])
    center, radius, sigma = receptor_prior(receptor)
    expected = ((receptor - center).square().sum(-1).mean()).sqrt()
    assert torch.allclose(sigma, expected * INITIAL_NOISE_STD_PROPORTION / 1.73)
    assert torch.allclose(receptor_prior(receptor * 3 + 25)[2], sigma * 3)
    for kwargs in ({"initial_noise_std_proportion": -1},
                   {"initial_noise_std_proportion": 1},
                   {"tr_sigma_max": 19}, {"poi_shell_min": 30},
                   {"coupled_radius": 5}):
        try:
            TernaryInitializer(**kwargs)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"legacy distribution accepted: {kwargs}")


def test_joint_initialization_is_shared_and_reproducible():
    data = ligand_input()
    torch.manual_seed(4)
    a = torch.randn(37, 3) * 8 + 12
    b = torch.randn(19, 3) * 3 - 8
    init = TernaryInitializer()
    l, p, rotation, meta = init.initialize(a, b, data, seed=12)
    again = init.initialize(a, b, data, seed=12)
    assert torch.equal(l, again[0]) and torch.equal(p, again[1])
    assert meta == again[3]
    assert meta['init_mode'] == 'sdpi'
    assert meta['version'] == 'sdpi_v1'
    # Native coordinates must not enter inference internal geometry.
    changed = {**data, "coords": torch.randn_like(data["coords"]) * 1000}
    assert torch.equal(l, init.initialize(a, b, changed, seed=12)[0])
    assert torch.allclose((p - p.mean(1, keepdim=True)) @ rotation,
                          (b - b.mean(0)).unsqueeze(0), atol=1e-5)
    sides = meta['sides']
    ra, rb = (torch.tensor(sides[s]['rotation']) for s in ('e3', 'poi'))
    ta, tb = (torch.tensor(sides[s]['translation']) for s in ('e3', 'poi'))
    shared = apply_rigid(l[0], *invert_rigid(ra, ta))
    lb = apply_rigid(shared, rb, tb)
    assert (torch.cdist(p[0], l[0]) - torch.cdist(b, lb)).abs().max() < 1e-5
    for name, receptor in (("e3", a), ("poi", b)):
        assert abs(sides[name]['sigma_init'] - receptor_prior(receptor)[2].item()) < 1e-6
    assert len(meta['torsion_updates']) == int(data['edge_mask'].sum())


def test_diffdock_conformer_policy_is_explicit_and_reproducible():
    data = ligand_input()
    receptor = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    init = TernaryInitializer(conformer_policy="diffdock_v1_1")
    first = init.initialize(receptor, receptor * 2, data, seed=17)
    second = init.initialize(receptor, receptor * 2, data, seed=17)
    assert torch.equal(first[0], second[0])
    assert first[3]["conformer"]["policy"] == "diffdock_v1_1"
    assert first[3]["conformer"]["method"] == "ETKDGv2"
    try:
        TernaryInitializer(conformer_policy="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown conformer policy accepted")


def test_bad_atom_and_torsion_maps_fail():
    data = ligand_input()
    mol = Chem.Mol(data['mol'])
    bonds = data['edge_index'].clone()
    bonds[0, 0] = 999
    for edges, masks in ((bonds, data['mask_rotate']),
                         (data['edge_index'], data['mask_rotate'].astype(float)),
                         (data['edge_index'].flip(0), data['mask_rotate']),
                         (data['edge_index'], ~data['mask_rotate'])):
        try:
            validate_ligand_graph(mol, edges, data['edge_mask'], masks)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid atom/directed torsion mapping accepted")


def test_torsion_is_applied_once_and_no_torsion_is_explicit():
    from tricoopdiff.diffdock.utils.torsion import modify_conformer_torsion_angles

    data = ligand_input()
    receptor = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    target = 'tricoopdiff.diffdock.utils.torsion.modify_conformer_torsion_angles'
    for disabled in (False, True):
        with patch(target, wraps=modify_conformer_torsion_angles) as torsion:
            *_, metadata = TernaryInitializer(no_torsion=disabled).initialize(
                receptor, receptor * 2, data, seed=5,
            )
        assert torsion.call_count == (0 if disabled else 1)
        assert metadata['no_torsion'] == disabled
        if disabled:
            assert not any(metadata['torsion_updates'])


def test_float64_initial_torsions_preserve_bond_geometry_precision():
    data = ligand_input()
    data['coords'] = data['coords'].double()
    receptor = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]],
                            dtype=torch.float64)
    rigid = TernaryInitializer(no_torsion=True).initialize(
        receptor, receptor * 2, data, seed=73)[0][0]
    flexible = TernaryInitializer().initialize(
        receptor, receptor * 2, data, seed=73)[0][0]
    u, v = data['edge_index']
    expected = torch.linalg.vector_norm(rigid[u] - rigid[v], dim=-1)
    actual = torch.linalg.vector_norm(flexible[u] - flexible[v], dim=-1)
    assert flexible.dtype == torch.float64
    assert torch.allclose(actual, expected, atol=1e-10, rtol=0.), (
        'float64 torsion initialization silently rounded the conformer to float32')


def test_sampler_rejects_removed_modes_and_mismatched_clock():
    from tricoopdiff.legacy.diffusion import InterfaceDiffusion
    from tricoopdiff.sampler import sample_single

    model = torch.nn.Linear(1, 1)
    data = {'coords': torch.ones(1, 4, 3)}
    for kwargs in ({'init_mode': 'dual_native'}, {'init_mode': 'native'}, {'init_mode': 'coupled5a'},
                   {'init_mode': 'anchor_local'}, {'t_anchor': 0.7},
                   {'steps': 0}, {'steps': -1}, {'steps': 1.5}):
        try:
            sample_single(model, InterfaceDiffusion(), TernaryInitializer(),
                          data, data, data, device='cpu', **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"removed or mismatched prior accepted: {kwargs}")


def test_sampler_rejects_empty_ensemble():
    from tricoopdiff.sampler import sample_ensemble

    for kwargs in ({'n_walkers': 0}, {'seeds': []}):
        try:
            sample_ensemble(None, None, None, {}, {}, {}, **kwargs)
        except ValueError as error:
            assert 'walker' in str(error)
        else:
            raise AssertionError('empty ensemble must not be reported as completed')


def test_coupled_anchor_perturbation_preserves_local_geometry():
    from tricoopdiff.initialization.candidates import coupled_anchor_perturbation

    data = ligand_input('CCCCCCCC')
    ligand = data['coords']
    poi = ligand + torch.tensor([0., 0., 2.])
    center = {'ligand': ligand, 'poi': poi, 'rotation': torch.eye(3)[None]}
    e3 = ligand[0] - torch.tensor([0., 0., 2.])
    before = ligand.clone()
    result, meta = coupled_anchor_perturbation(
        center, e3, data, seed=73, translation_std=0., rotation_std=0.)
    assert meta['mode'] == 'two_anchor_coupled_torsion', 'separable anchors must use coupled motion'
    a, b = meta['e3_anchor_atoms'], meta['poi_anchor_atoms']
    assert torch.allclose(result['ligand'][0, a], ligand[0, a], atol=1e-4)
    assert torch.allclose(torch.cdist(result['ligand'][0, b], result['poi'][0]),
                          torch.cdist(ligand[0, b], poi[0]), atol=1e-4)
    edges = data['edge_index'].T
    def lengths(coords):
        return (coords[0, edges[:, 0]] - coords[0, edges[:, 1]]).norm(dim=-1)
    assert torch.allclose(lengths(result['ligand']), lengths(ligand), atol=1e-4)
    again, _ = coupled_anchor_perturbation(
        center, e3, data, seed=73, translation_std=0., rotation_std=0.)
    assert torch.equal(result['ligand'], again['ligand'])
    assert torch.equal(ligand, before), 'initialization must not mutate its center'
    rigid_data = ligand_input('c1ccccc1')
    rigid_center = {'ligand': rigid_data['coords'], 'poi': poi, 'rotation': torch.eye(3)[None]}
    _, rigid_meta = coupled_anchor_perturbation(rigid_center, e3, rigid_data, seed=73)
    assert rigid_meta['mode'] == 'rigid_only_no_separable_anchors'


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'{name}: PASS', flush=True)

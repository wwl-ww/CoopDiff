"""Model build regression checks. Run with pytest or execute this file directly."""

if __name__ == "__main__":
    from helpers import run_tests

    run_tests(__file__)

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tricoopdiff.geometry.rigid import rotation_exp as so3_exp
from tricoopdiff.models.assembly import BRANCHES, TernaryAssembly


# Model build

def test_trifuse_build():
    """TriFuse builds with correct output shapes."""
    from tricoopdiff.legacy.trifuse import TriFuse

    trifuse = TriFuse(
        node_dim=128, ns=32, nv=4,
        num_conv_layers_per_expert=6,
    )

    # Dummy ternary graph: 100 E3 residues + 150 POI residues + 80 PROTAC atoms
    # TriFuse.forward expects UNBATCHED [N, ...] inputs (batch index derived
    # internally); the leading dim here was a stray batch of 1 that broke
    # per-batch softmax (result[mask] of shape [1] vs value [N]).
    N = 330
    node_features = torch.randn(N, 128)
    node_coords = torch.randn(N, 3)
    node_types = torch.cat([
        torch.zeros(100, dtype=torch.long),
        torch.ones(150, dtype=torch.long),
        2 * torch.ones(80, dtype=torch.long),
    ])

    out = trifuse(node_features, node_coords, node_types)

    assert out['confidence'].shape == (1, 1)
    assert out['state'].shape == (1, 2)
    assert out['score_gains'].shape == (1, 5)
    assert torch.allclose(out['score_gains'], torch.ones_like(out['score_gains']))
    assert out['h_e3l'].shape == (1, trifuse.node_dim)
    assert out['h_pol'].shape == (1, trifuse.node_dim)
    assert not torch.allclose(out['h_e3l'], out['h_pol'])
    assert len(out['film_params']) == 12  # 6 layers × 2 subgraphs
    for gamma, beta in out['film_params']:
        # FiLM is specific to each ligand atom and expert layer.
        assert gamma.shape == (80, trifuse.film_dim)
        assert beta.shape == (80, trifuse.film_dim)

    # Count parameters
    n_params = sum(p.numel() for p in trifuse.parameters())
    print(f"  TriFuse params: {n_params:,}")
    # The atom-contact and bond encoders intentionally increase capacity while
    # remaining well below the 2.5M P0 budget.
    assert 500_000 < n_params < 2_500_000, f"Unexpected TriFuse params: {n_params:,}"

    print("  [OK] TriFuse build")


def test_trifuse_reads_geometry_and_preserves_rigid_invariance():
    """Context changes with interfaces, not with a common rigid transform."""
    from tricoopdiff.geometry.rigid import rotation_exp as so3_exp
    from tricoopdiff.legacy.trifuse import TriFuse

    torch.manual_seed(3)
    trifuse = TriFuse(
        node_dim=32, input_dim=16, ns=8, nv=2,
        num_conv_layers_per_expert=2,
        film_dim=4,
    ).eval()
    features = torch.randn(15, 16)
    coords = torch.randn(15, 3)
    node_types = torch.tensor([0] * 5 + [1] * 6 + [2] * 4)
    time = torch.tensor([0.4])

    base = trifuse(
        features, coords, node_types, t_e3l=time, t_pol=time,
    )['h_cond']
    changed_coords = coords.clone()
    changed_coords[node_types == 1] += torch.tensor([12.0, 0.0, 0.0])
    changed = trifuse(
        features, changed_coords, node_types, t_e3l=time, t_pol=time,
    )['h_cond']
    assert not torch.allclose(base, changed, atol=1e-5)

    rotation = so3_exp(torch.tensor([[0.4, -0.2, 0.3]]))[0]
    transformed = coords @ rotation.T + torch.tensor([8.0, -4.0, 2.0])
    rigid = trifuse(
        features, transformed, node_types, t_e3l=time, t_pol=time,
    )['h_cond']
    assert torch.allclose(base, rigid, atol=2e-5), (base - rigid).abs().max()

    swapped_types = node_types.clone()
    swapped_types[node_types == 0] = 1
    swapped_types[node_types == 1] = 0
    swapped = trifuse(
        features, coords, swapped_types, t_e3l=time, t_pol=time,
    )['h_cond']
    assert not torch.allclose(base, swapped, atol=1e-5)
    print("  [OK] TriFuse geometry sensitivity and rigid invariance")


def test_trifuse_links_chemical_features_to_local_contacts():
    """Moving chemical identities between near/far atoms changes context."""
    from tricoopdiff.legacy.trifuse import TriFuse

    torch.manual_seed(7)
    trifuse = TriFuse(
        node_dim=32, input_dim=16, ns=8, nv=2,
        num_conv_layers_per_expert=2,
    ).eval()
    features = torch.randn(9, 16)
    node_types = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    coords = torch.tensor([
        [0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [16.0, 0.0, 0.0],
        [0.0, 3.0, 0.0], [8.0, 3.0, 0.0], [16.0, 3.0, 0.0],
        [0.0, 1.0, 0.0], [8.0, 7.0, 0.0], [16.0, 12.0, 0.0],
    ])
    time = torch.tensor([0.3])
    base = trifuse(features, coords, node_types, t_e3l=time, t_pol=time)[
        'h_cond'
    ]

    # Permuting features within the Ligand preserves component-level pooling
    # and all geometry-only descriptors, but changes which identity is at the
    # closest E3/POI contacts.
    permuted = features.clone()
    permuted[6:9] = permuted[torch.tensor([8, 7, 6])]
    changed = trifuse(
        permuted, coords, node_types, t_e3l=time, t_pol=time,
    )['h_cond']
    assert not torch.allclose(base, changed, atol=1e-5)
    print("  [OK] TriFuse binds chemical identity to local contacts")


def test_trifuse_reads_sigma_and_solver_state():
    """Equal geometry at different sigma/dt must produce different context."""
    from tricoopdiff.legacy.trifuse import TriFuse

    torch.manual_seed(13)
    trifuse = TriFuse(
        node_dim=32, input_dim=16, ns=8, nv=2,
        num_conv_layers_per_expert=2,
    ).eval()
    features = torch.randn(12, 16)
    coords = torch.randn(12, 3)
    node_types = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    time = torch.tensor([0.5])
    low = torch.tensor([[-2.0, -2.0, -2.0, -2.0, -2.0,
                         0.05, 0.05, 0.05, 0.5]])
    high = torch.tensor([[2.0, 1.0, 2.0, 2.0, 1.0,
                          0.10, 0.10, 0.10, 0.5]])
    out_low = trifuse(
        features, coords, node_types, t_e3l=time, t_pol=time,
        diffusion_state=low,
    )
    out_high = trifuse(
        features, coords, node_types, t_e3l=time, t_pol=time,
        diffusion_state=high,
    )
    assert not torch.allclose(out_low['h_state'], out_high['h_state'])
    assert not torch.allclose(out_low['h_e3l'], out_high['h_e3l'])
    print("  [OK] TriFuse reads sigma and solver state")


def test_trifuse_supports_odd_hidden_dimensions():
    from tricoopdiff.legacy.trifuse import TriFuse

    model = TriFuse(node_dim=31, input_dim=16, film_dim=4,
                    num_conv_layers_per_expert=1).eval()
    features = torch.randn(9, 16)
    coords = torch.randn(9, 3)
    node_types = torch.tensor([0] * 3 + [1] * 3 + [2] * 3)
    output = model(features, coords, node_types, t_e3l=torch.tensor([.4]))
    assert output['h_e3l'].shape == (1, 31)
    assert all(gamma.shape[-1] == 4 for gamma, _ in output['film_params'])


def test_trifuse_distinguishes_mirrored_interface_handedness():
    """Signed local geometry distinguishes mirror images with equal distances."""
    from tricoopdiff.legacy.trifuse import TriFuse

    torch.manual_seed(23)
    trifuse = TriFuse(
        node_dim=32, input_dim=16, num_conv_layers_per_expert=2,
    ).eval()
    features = torch.randn(9, 16)
    node_types = torch.tensor([0] * 3 + [1] * 3 + [2] * 3)
    coords = torch.tensor([
        [-2.0, 0.0, 1.0], [-2.0, 1.0, 0.0], [-2.0, -1.0, 0.0],
        [2.0, 0.0, 1.0], [2.0, 1.0, 0.0], [2.0, -1.0, 0.0],
        [0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 2.0, 0.5],
    ])
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    base = trifuse(
        features, coords, node_types, lig_edge_index=edges,
    )['h_cond']
    mirrored = coords.clone()
    mirrored[:, 2] *= -1
    reflected = trifuse(
        features, mirrored, node_types, lig_edge_index=edges,
    )['h_cond']
    assert not torch.allclose(base, reflected, atol=1e-5)
    print("  [OK] TriFuse distinguishes mirrored interface handedness")


def test_film_zero_init():
    """FiLM is transparent at initialization (adaLN-Zero)."""
    from tricoopdiff.legacy.trifuse import FiLMHead

    head = FiLMHead(hidden_dim=64)
    h = torch.randn(2, 64)
    gamma, beta = head(h)

    # Zero-init: gamma and beta should be all zeros
    assert torch.allclose(gamma, torch.zeros_like(gamma), atol=1e-6), \
        f"Gamma not zero: max={gamma.abs().max():.2e}"
    assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-6), \
        f"Beta not zero: max={beta.abs().max():.2e}"

    print("  [OK] FiLM zero-init transparent")


def test_trifuse_atom_film_can_select_ligand_regions():
    """A trained FiLM head can assign different scalar gates per atom."""
    from tricoopdiff.legacy.trifuse import TriFuse

    torch.manual_seed(19)
    trifuse = TriFuse(
        node_dim=32, input_dim=16, num_conv_layers_per_expert=2, film_dim=8,
    ).eval()
    torch.nn.init.normal_(trifuse.film_heads[0].net.weight, std=0.02)
    features = torch.randn(12, 16)
    coords = torch.randn(12, 3)
    node_types = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    gamma, _ = trifuse(features, coords, node_types)['film_params'][0]
    assert gamma.shape == (4, 8)
    assert not torch.allclose(gamma[0], gamma[1])
    print("  [OK] FiLM can select individual ligand atoms")


def test_score_control_is_identity_bounded_and_trainable():
    """Score calibration starts exact identity and cannot reverse direction."""
    from tricoopdiff.legacy.trifuse import ScoreControlHead

    head = ScoreControlHead(hidden_dim=16, max_delta=0.25)
    context = torch.randn(3, 16)
    assert torch.equal(head(context), torch.ones(3, 5))
    torch.nn.init.normal_(head.net[-1].weight)
    gains = head(context)
    assert bool(((gains >= 0.75) & (gains <= 1.25)).all())
    print("  [OK] Score controls are identity-initialized and bounded")


def test_layerwise_bridge_starts_exactly_transparent():
    """The dual-stream bridge must not perturb native zero-init experts."""
    from tricoopdiff.legacy.trifuse import LayerwiseExpertBridge

    bridge = LayerwiseExpertBridge(
        scalar_dim=8, hidden_dim=16, num_layers=3,
    )
    e3l = torch.randn(5, 8)
    pol = torch.randn(5, 8)
    delta_e3l, delta_pol = bridge(e3l, pol, 1)
    for delta in (*delta_e3l, *delta_pol):
        assert torch.equal(delta, torch.zeros_like(delta))
    torch.nn.init.normal_(bridge.output.weight, std=0.02)
    delta_e3l, delta_pol = bridge(e3l, pol, 1)
    for delta in (*delta_e3l, *delta_pol):
        assert bool((delta.abs() <= bridge.max_delta).all())
    print("  [OK] Layerwise expert bridge is transparent and bounded")


def test_torsion_coordinator_zero_init_and_per_bond_control():
    """Shared torsion starts neutral but can make independent bond decisions."""
    from tricoopdiff.legacy.trifuse import TorsionCoordinator

    coordinator = TorsionCoordinator(context_dim=8, hidden_dim=4)
    context = torch.randn(2, 8)
    score_e3l = torch.tensor([[1.0, 4.0], [-2.0, 6.0]])
    score_pol = torch.tensor([[3.0, -2.0], [4.0, 2.0]])
    score, weight, residual = coordinator(context, score_e3l, score_pol)

    assert torch.allclose(weight, torch.full_like(weight, 0.5))
    assert torch.allclose(residual, torch.zeros_like(residual))
    assert torch.allclose(score, 0.5 * (score_e3l + score_pol))
    assert score.shape == score_e3l.shape
    print("  [OK] Torsion coordinator neutral initialization")


def test_torsion_coordinator_rejects_flat_native_shape():
    """The coordinator boundary requires an explicit batch dimension."""
    from tricoopdiff.legacy.trifuse import TorsionCoordinator

    coordinator = TorsionCoordinator(context_dim=8, hidden_dim=4)
    try:
        coordinator(torch.randn(1, 8), torch.randn(3), torch.randn(3))
    except ValueError as error:
        assert "[B,K]" in str(error)
    else:
        raise AssertionError("flat native torsion score must be normalized first")


def test_torsion_coordinator_does_not_overflow_finite_evidence():
    """Finite expert scores must not overflow inside feature construction."""
    from tricoopdiff.legacy.trifuse import TorsionCoordinator

    coordinator = TorsionCoordinator(context_dim=8, hidden_dim=4).eval()
    context = torch.zeros(1, 8)
    score = torch.full((1, 3), 1e30)
    coordinated, weight, residual = coordinator(context, score, score)
    assert torch.isfinite(coordinated).all()
    assert torch.isfinite(weight).all()
    assert torch.isfinite(residual).all()


def test_bond_context_is_local_to_each_rotatable_bond():
    """Distinct endpoints/fragments must produce distinct bond contexts."""
    from tricoopdiff.legacy.trifuse import BondContextEncoder

    torch.manual_seed(11)
    encoder = BondContextEncoder(hidden_dim=16)
    ligand_tokens = torch.randn(5, 16)
    ligand_coords = torch.tensor([
        [0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.3, 0.0],
        [4.2, 0.3, 0.0], [5.6, 1.0, 0.0],
    ])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_mask = torch.tensor([False, True, True, False])
    mask_rotate = torch.tensor([
        [False, False, True, True, True],
        [False, False, False, True, True],
    ])
    context = encoder(
        ligand_tokens, ligand_coords, edge_index, edge_mask,
        mask_rotate, torch.randn(1, 16),
    )
    assert context.shape == (1, 2, 16)
    assert not torch.allclose(context[:, 0], context[:, 1])
    print("  [OK] Rotatable bonds receive local contexts")


def test_expert_telemetry_conditions_bond_context():
    """Branch activation evidence must reach the shared torsion decision."""
    from tricoopdiff.legacy.trifuse import (
        TELEMETRY_PER_LAYER, TELEMETRY_TAIL, TriFuse,
    )

    torch.manual_seed(17)
    trifuse = TriFuse(
        node_dim=32, input_dim=16, num_conv_layers_per_expert=2,
    ).eval()
    width = 2 * TELEMETRY_PER_LAYER + TELEMETRY_TAIL
    bonds = torch.randn(1, 3, 32)
    quiet = torch.zeros(1, width)
    stressed = torch.ones(1, width)
    base = trifuse.condition_bonds_on_telemetry(bonds, quiet, quiet)
    changed = trifuse.condition_bonds_on_telemetry(bonds, stressed, quiet)
    assert base.shape == bonds.shape
    assert not torch.allclose(base, changed)
    print("  [OK] Expert telemetry conditions bond contexts")


def test_diffusion_schedules():
    """Diffusion schedules produce valid noise levels."""
    from tricoopdiff.legacy.diffusion import InterfaceDiffusion
    from tricoopdiff.models.expert import so3_score_norm

    diffusion = InterfaceDiffusion(inference_steps=20)

    # Test sigma schedule
    t = torch.tensor([0.0, 0.5, 1.0])
    tr, rot, tor = diffusion.t_to_sigma(t, t, t)
    assert tr[0] < tr[1] < tr[2], "Sigma should increase with t"
    assert abs(tr[0].item() - 0.1) < 0.05, f"Sigma at t=0 should be ~0.1, got {tr[0].item()}"

    # Test SO(3) score norm
    sigma = torch.tensor([0.5, 1.0])
    norm = so3_score_norm(sigma)
    assert norm.shape == (2,)
    assert (norm > 0).all()

    print("  [OK] Diffusion schedules")


def test_graph_construction():
    """Subgraph construction produces valid HeteroData."""
    from tricoopdiff.inputs.graphs import (
        build_subgraph_e3l, build_subgraph_pol,
        build_trifuse_graph, transform_ligand_to_poi_frame,
    )

    N_e3, N_poi, N_lig = 100, 150, 80
    e3_coords = torch.randn(N_e3, 3)
    e3_feat = torch.randn(N_e3, 5)
    poi_coords = torch.randn(N_poi, 3)
    poi_feat = torch.randn(N_poi, 5)
    lig_coords = torch.randn(N_lig, 3)
    lig_feat = torch.randn(N_lig, 7)

    # E3-L subgraph
    sub1 = build_subgraph_e3l(e3_coords, e3_feat, lig_coords, lig_feat)
    assert sub1['receptor'].pos.shape == (N_e3, 3)
    assert sub1['ligand'].pos.shape == (N_lig, 3)

    # POI-L subgraph
    lig_poi = transform_ligand_to_poi_frame(
        lig_coords,
        torch.zeros(3),
        torch.eye(3),
    )
    sub2 = build_subgraph_pol(poi_coords, poi_feat, lig_poi, lig_feat)
    assert sub2['receptor'].pos.shape == (N_poi, 3)
    assert sub2['ligand'].pos.shape == (N_lig, 3)

    # POI and ligand must undergo the same frame change before entering the
    # POI-L expert. Their relative vectors are preserved by the shared rigid
    # transform; transforming only the ligand recreates the historical sub2 NaN.
    translation = torch.tensor([4.0, -2.0, 7.0])
    rotation = torch.tensor([[0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]])
    poi_local = transform_ligand_to_poi_frame(poi_coords, translation, rotation)
    lig_local = transform_ligand_to_poi_frame(lig_coords, translation, rotation)
    before = lig_coords[0] - poi_coords[0]
    after = lig_local[0] - poi_local[0]
    assert torch.allclose(after, before @ rotation, atol=1e-6)

    # Ternary graph
    tri = build_trifuse_graph(e3_coords, e3_feat, poi_coords, poi_feat, lig_coords, lig_feat)
    assert tri['node_coords'].shape == (N_e3 + N_poi + N_lig, 3)
    assert tri['n_e3'] == N_e3
    assert tri['n_poi'] == N_poi
    assert tri['n_lig'] == N_lig

    print("  [OK] Graph construction")


def test_initializer_body_frame():
    """The rotation returned by POI initialization is the inverse frame map."""
    from scipy.spatial.transform import Rotation
    from tricoopdiff.geometry.rigid import assemble_interface_pair
    from tricoopdiff.inputs.graphs import transform_ligand_to_poi_frame

    np.random.seed(7)
    torch.manual_seed(7)
    template = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 3.0],
    ])
    e3_center = torch.zeros(1, 3)
    rotations = torch.tensor(Rotation.random(2).as_matrix(), dtype=template.dtype)
    initialized, body_rotation = assemble_interface_pair(
        template.unsqueeze(0), rotations[0:1], torch.randn(1, 3),
        rotations[1:2], torch.randn(1, 3),
    )
    translation = initialized[0].mean(0) - e3_center[0]
    local = transform_ligand_to_poi_frame(
        initialized[0], translation, body_rotation[0],
    )
    expected = template - template.mean(0)
    assert torch.allclose(local, expected, atol=1e-5), (local, expected)

    print("  [OK] Initializer body-frame rotation")


def test_frozen_expert_stays_eval():
    """Parent train mode must not reactivate a frozen pretrained expert."""
    import torch.nn as nn
    from tricoopdiff.models.expert import FrozenDiffDock

    class DummyExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.5)
            self.batchnorm = nn.BatchNorm1d(4)
            self.ns, self.nv = 4, 1
            self.sh_irreps = None
            self.device = "cpu"

        def forward(self, features):
            return self.dropout(self.batchnorm(features))

    frozen = FrozenDiffDock(DummyExpert(), num_conv_layers=1)
    assert not frozen.training and not frozen.model.training, 'freezing must disable dropout/BN immediately'
    features = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    running_mean = frozen.model.batchnorm.running_mean.clone()
    first, second = frozen(features), frozen(features)
    assert torch.equal(first, second), 'a frozen expert cannot produce training-mode dropout samples'
    assert torch.equal(frozen.model.batchnorm.running_mean, running_mean)
    assert frozen.model.batchnorm.num_batches_tracked == 0
    parent = nn.Module()
    parent.add_module("expert", frozen)
    parent.train()
    assert not frozen.training
    assert not frozen.model.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())

    print("  [OK] Frozen expert remains in eval mode")


def test_frozen_expert_features_preserve_scores_gradients_and_hook_cleanup():
    import torch.nn as nn
    from tricoopdiff.models.expert import FrozenDiffDock, configure_expert_policy
    from tricoopdiff.geometry.equivariant_clamp import clamp_irrep_features

    class FeatureLayer(nn.Module):
        out_irreps = '2x0e + 1x1o + 1x0o'

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.))

        def forward(self, coords):
            return torch.cat((coords[:, :2] * self.weight, coords,
                              coords.sum(-1, keepdim=True)), dim=-1)

    class NativeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.ModuleList([FeatureLayer()])
            self.ns, self.nv, self.sh_irreps, self.device = 2, 1, None, 'cpu'
            self.fail = False

        def forward(self, coords):
            nodes = self.conv_layers[0](coords)
            if self.fail:
                raise RuntimeError('score head failure')
            if self.activation_clamp_mode == 'irrep_norm':
                nodes = clamp_irrep_features(nodes, self.conv_layers[0].out_irreps,
                                            self.activation_clamp)
            else:
                nodes = nodes.clamp(-self.activation_clamp, self.activation_clamp)
            # Native production CGModel also returns an auxiliary confidence.
            return nodes[:, :3], nodes[:, 3:], nodes[:, -1], nodes.new_zeros(1)

    expert = FrozenDiffDock(NativeExpert(), num_conv_layers=1)
    assert expert.node_scalar_dim == 3 and expert.node_vector_dim == 1
    original_keys = set(expert.state_dict())
    for mode in ('componentwise', 'irrep_norm'):
        configure_expert_policy(expert, .75, mode)
        coords = torch.tensor([[.1, .2, .3], [2., 3., 4.]], requires_grad=True)
        ordinary = expert(coords)
        tr, rot, tor, scalars = expert.forward_features(coords)
        assert all(torch.equal(actual, expected)
                   for actual, expected in zip((tr, rot, tor), ordinary[:3]))
        expected = torch.cat((ordinary[0][:, :2], ordinary[2][:, None]), dim=-1)
        assert torch.equal(scalars, expected), '0e/0o extraction must follow irrep sections'
        assert scalars.shape == (2, expert.node_scalar_dim)
        if mode == 'componentwise':
            try:
                expert.forward_features(coords, include_vectors=True)
            except ValueError as error:
                assert 'require irrep_norm' in str(error)
            else:
                raise AssertionError('componentwise clipping must not enter an equivariant vector bridge')
            continue
        _, _, _, _, vectors = expert.forward_features(coords, include_vectors=True)
        native_nodes = expert.model.conv_layers[0](coords)
        expected_vectors = clamp_irrep_features(native_nodes[:, 2:5], '1x1o', .75).reshape(-1, 1, 3)
        assert vectors.shape == (2, 1, 3)
        assert torch.equal(vectors, expected_vectors), 'equivariant node bridge must retain and clamp l=1 channels'
        assert torch.isfinite(vectors).all()
        native_gradient = torch.autograd.grad(expected.sum(), coords)[0]
        scalars.sum().backward()
        assert torch.equal(coords.grad, native_gradient), 'features must preserve native post-clamp Jacobians'
        assert torch.isfinite(coords.grad).all() and coords.grad.abs().sum() > 0
        assert all(parameter.grad is None and not parameter.requires_grad
                   for parameter in expert.parameters())
        assert not expert.model.conv_layers[0]._forward_hooks
        assert set(expert.state_dict()) == original_keys
    # At the production threshold, replacing scalar norm clipping with clamp
    # differs by an ULP. Features must be the actual native post-clamp scalars.
    configure_expert_policy(expert, 10000., 'irrep_norm')
    coords = torch.linspace(10000. * 1.01, 10000. * 20., 1000).reshape(-1, 1).expand(-1, 3).clone()
    ordinary = expert(coords)
    expected = torch.cat((ordinary[0][:, :2], ordinary[2][:, None]), dim=-1)
    assert torch.equal(expert.forward_features(coords)[3], expected)
    assert not torch.equal(expected, expected.clamp(-10000., 10000.)), \
        'fixture must expose native scalar norm-clamp rounding, not only approximate equality'
    bad_coords = torch.tensor([[float('inf'), float('nan'), 1.]])
    ordinary = expert(bad_coords)
    expected = torch.cat((ordinary[0][:, :2], ordinary[2][:, None]), dim=-1)
    assert torch.equal(torch.isnan(expert.forward_features(bad_coords)[3]), torch.isnan(expected)), \
        'feature clipping must not conceal native invalid scalar activations'

    # Non-reentrant checkpoint calls the native read again during backward.
    # Hooks must remain local to each read and preserve the coordinate Jacobian.
    from torch.utils.checkpoint import checkpoint
    configure_expert_policy(expert, .75, 'irrep_norm')
    coords = torch.tensor([[.1, .2, .3], [.2, .1, .2]], requires_grad=True)
    direct = expert.forward_features(coords)[3]
    direct_grad = torch.autograd.grad(direct.square().sum(), coords)[0]
    recomputed = checkpoint(lambda value: expert.forward_features(value)[3], coords, use_reentrant=False)
    recomputed_grad = torch.autograd.grad(recomputed.square().sum(), coords)[0]
    assert torch.equal(recomputed, direct) and torch.equal(recomputed_grad, direct_grad)
    assert not expert.model.conv_layers[0]._forward_hooks
    expert.model.fail = True
    try:
        expert.forward_features(coords)
    except RuntimeError as error:
        assert 'score head failure' in str(error)
    else:
        raise AssertionError('feature read must not swallow native score failures')
    assert not expert.model.conv_layers[0]._forward_hooks, 'failed reads must remove their hook'
    expert.model.fail = False
    assert expert.forward_features(coords)[3].shape == (2, 3)
    print('  [OK] Frozen feature scores, coordinate gradients and hook cleanup')


def test_failed_expert_policy_configuration_preserves_the_live_policy():
    from types import SimpleNamespace
    from tricoopdiff.models.expert import configure_expert_policy

    expert = SimpleNamespace(model=SimpleNamespace(activation_clamp=10000.,
                                                   activation_clamp_mode='irrep_norm'))
    for limit, mode in ((1., 'invalid'), (float('nan'), 'componentwise'), (-1., 'componentwise')):
        try:
            configure_expert_policy(expert, limit, mode)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid numerical policies must fail')
        assert expert.model.activation_clamp == 10000.
        assert expert.model.activation_clamp_mode == 'irrep_norm'


def test_expert_loader_cannot_mislabel_the_frozen_layer_count():
    import tempfile
    from unittest.mock import patch
    from tricoopdiff.models.expert import build_diffdock_expert

    source = Path(__file__).resolve().parents[1] / 'checkpoints/score_model/model_parameters.yml'
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'model_parameters.yml').write_text(source.read_text())
        with patch('tricoopdiff.diffdock.factory.build_cg_model') as build:
            for invalid in (6, 0, True, 11.0):
                try:
                    build_diffdock_expert(root, device='cpu', num_conv_layers=invalid)
                except ValueError as error:
                    assert 'frozen checkpoint architecture' in str(error)
                else:
                    raise AssertionError('wrapper layers must not disagree with the checkpoint')
            assert not build.called, 'invalid overrides must fail before constructing/loading a model'


def test_score_guard_handles_infinity():
    """Inference must detect non-finite scores without hiding their values."""
    from tricoopdiff.sampler import _nonfinite_components

    score = torch.tensor([[1.0, float("nan"), float("inf"), -float("inf")]])
    velocities = {'e3_ligand': {'tr': score, 'rot': torch.zeros(1, 3)},
                  'poi_ligand': {'tr': torch.zeros(1, 3), 'rot': torch.zeros(1, 3)},
                  'torsion': torch.empty(1, 0)}
    assert _nonfinite_components(velocities) == ['e3_ligand.tr']
    assert torch.equal(torch.isfinite(velocities['e3_ligand']['tr']), torch.isfinite(score))

    print("  [OK] Score guard rejects rather than sanitizes infinities")


def test_reflect_pipeline():
    """TriReflect detects clashes and runs minimization."""
    from tricoopdiff.geometry.reflect import detect_clashes, tricoop_reflect

    # Create two non-bonded atoms well inside VDW contact → should clash.
    # (0.5Å is skipped by the covalent-radius 1-2 heuristic, which treats very
    # close pairs as a possible bond when no bond graph is supplied.)
    coords = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    atomic_numbers = np.array([6, 6])  # both carbon, VDW=1.7 → contact 3.4Å

    clashes = detect_clashes(coords, atomic_numbers, vdw_scale=1.0)
    # 2.5 < 3.4 (VDW contact) and 2.5 > 1.2×(0.76+0.76)=1.82 (not bonded) → clash
    assert clashes.sum() > 0, "Should detect clash for non-bonded C atoms 2.5Å apart"

    # Non-clashing pair
    coords_far = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    clashes_far = detect_clashes(coords_far, atomic_numbers, vdw_scale=1.0)
    assert clashes_far.sum() == 0, "Should NOT detect clash for atoms 10Å apart"

    # Full pipeline
    result = tricoop_reflect(coords, atomic_numbers, do_minimization=True)
    assert 'is_valid' in result
    assert 'n_clashes' in result

    print("  [OK] TriReflect pipeline")


def test_deepternary_dockq_alignment():
    """Primary DockQ matches DeepTernary's ternary-coordinate convention."""
    from tricoopdiff.evaluate import compute_deepternary_dockq

    e3 = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 6.0, 0.0]])
    ligand = np.array([[2.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    poi = np.array([[4.0, 0.0, 0.0], [4.0, 3.0, 0.0]])
    perfect = compute_deepternary_dockq(ligand, e3, poi, ligand, e3, poi)
    assert perfect is not None
    assert abs(perfect["dockq"] - 1.0) < 1e-7
    assert abs(perfect["fnat"] - 1.0) < 1e-7
    assert perfect["lrms"] < 1e-7
    assert perfect["irms"] < 1e-7

    displaced = compute_deepternary_dockq(
        ligand, e3, poi, ligand + 20.0, e3, poi + 20.0
    )
    assert displaced is not None
    assert displaced["dockq"] < perfect["dockq"]

    print("  [OK] DeepTernary DockQ alignment")


def test_cross_edge_taper_preserves_internal_weights():
    from tricoopdiff.diffdock.models.cg_model import CGModel

    expert_policy = SimpleNamespace(smooth_edges=False)
    edge_vec = torch.tensor([[1., 0., 0.], [2., 0., 0.]])
    internal = CGModel.get_edge_weight(expert_policy, edge_vec, 4.)
    cross = CGModel.get_edge_weight(expert_policy, edge_vec, 4., smooth=True)
    expected = .5 * (torch.cos(edge_vec.norm(dim=-1) * torch.pi / 4.) + 1.)
    assert internal == 1.0
    assert torch.allclose(cross[:, 0], expected)
    print("  [OK] Cross-edge taper preserves internal pretrained weights")


# Assembly

def test_candidate_quality_head_reads_three_components_and_backpropagates():
    from tricoopdiff.models.assembly import (
        production_model_config, candidate_source_tensor, CANDIDATE_SOURCE_NAMES,
    )
    _, graph, proposals, state = assembly_fixture()
    graph['expert_node_features'] = torch.randn(15, 6)
    graph['expert_node_vectors'] = torch.randn(15, 4, 3)
    graph['fullatom_scalar_features'] = torch.randn(15, 9)
    graph['fullatom_vector_features'] = torch.randn(15, 3, 3)
    model = TernaryAssembly(**production_model_config(16, 16, True,
        expert_backend='matcha', expert_scalar_dim=2, expert_vector_dim=2,
        protein_local_geometry_contract='none', candidate_driven=True))
    output = model(graph, proposals, state, source='hdock_matcha')
    assert output['quality_errors'].shape == (1, 5)
    assert output['candidate_score'].shape == (1,)
    assert model.controller.quality[0].in_features == 144, 'scoring must preserve component distributions'
    output['quality_errors'].sum().backward()
    assert model.controller.quality[-1].weight.grad.abs().sum() > 0
    assert model.controller.source.weight.grad is None, 'quality supervision must not rewrite motion features'
    assert model.controller.protein.weight.grad is None, 'quality gradients must stop before the motion trunk'
    assert candidate_source_tensor('hdock_matcha', state, CANDIDATE_SOURCE_NAMES).item() == 2
    try:
        candidate_source_tensor('hdock_matcha', state)
    except ValueError:
        pass
    else:
        raise AssertionError('legacy checkpoints must not silently interpret new candidate source')


def test_candidate_refinement_scores_final_state_and_keeps_input_pool():
    from tricoopdiff.models.assembly import TernaryAssemblyModel, production_model_config

    expert = torch.nn.Linear(1, 1)
    expert.backend = 'matcha'
    expert.frame_policy = 'backbone_rigid_transport_v1'
    expert.node_scalar_dim = 2
    expert.node_vector_dim = 2
    expert.read_pair = lambda *args: None
    expert.model = SimpleNamespace(dynamic_max_cross=False, no_torsion=True)
    fuse = TernaryAssembly(**production_model_config(16, 16, True,
        expert_backend='matcha', expert_scalar_dim=2, expert_vector_dim=2,
        protein_local_geometry_contract='none', candidate_driven=True))
    model = TernaryAssemblyModel(expert, fuse)
    calls = []

    def forward(e3, poi, ligand, t_e3l, **kwargs):
        calls.append((t_e3l.clone(), poi['coords'].clone()))
        zero = e3['coords'].new_zeros(1, 3)
        motion = {'rigid_output_mode': 'independent_centroids_v1',
                  'e3_ligand': {'tr': zero, 'rot': zero},
                  'poi_ligand': {'tr': zero+0.01, 'rot': zero},
                  'torsion': zero[:, :0]}
        return {'normalized_velocity': motion, 'quality_errors': zero.new_ones(1, 5),
                'candidate_score': -poi['coords'].mean().reshape(1)}

    model.forward = forward
    e3 = {'coords': torch.randn(1, 5, 3)}
    poi = {'coords': torch.randn(1, 6, 3)}
    ligand = {'coords': torch.randn(1, 4, 3)}
    candidate = {'ligand': ligand['coords'].clone(), 'poi': poi['coords'].clone(),
        'rotation': torch.eye(3)[None], 'candidate_id': ('generated', 1),
        'provenance': {'candidate_source': 'hdock_matcha'}}
    original = candidate['poi'].clone()
    model.train()
    result = model.refine_candidates(e3, poi, ligand, [candidate], steps=2)[0]
    assert len(calls) == 3 and calls[-1][0].item() == 0, 'score must read actual terminal state'
    assert torch.equal(calls[-1][1], result['poi'])
    assert torch.equal(candidate['poi'], original), 'refinement mutated the reusable bank'
    assert model.training, 'inference must restore the caller mode'
    assert not torch.equal(original, result['poi']), 'protein refinement action was not executed'


def test_candidate_fullatom_evidence_has_rigid_invariance_and_sidechain_sensitivity():
    from rdkit import Chem
    from tricoopdiff.models.assembly import candidate_fullatom_evidence

    coords = torch.tensor([[0., 0., 0.], [5., 1., 1.], [1., 1., 1.]])
    types = torch.tensor([0, 1, 2])
    e3 = {'steric_atoms': torch.tensor([[[0., 0., 0.], [.9, 1., 1.]]]),
          'steric_radii': torch.tensor([1.7, 1.7])}
    poi = {'coords': coords[1:2][None], 'steric_ca_center': coords[1],
           'steric_atoms': coords[1:2][None], 'steric_radii': torch.tensor([1.7])}
    ligand = {'coords': coords[2:][None], 'mol': Chem.MolFromSmiles('C')}
    scalar, vector = candidate_fullatom_evidence(coords, types, e3, poi, ligand, torch.eye(3))
    rotation = so3_exp(torch.tensor([.2, -.3, .1]))
    translation = torch.tensor([3., 4., 5.])
    def moved(value):
        return value @ rotation.T+translation
    transformed = candidate_fullatom_evidence(moved(coords), types,
        {**e3, 'steric_atoms': moved(e3['steric_atoms'])},
        {**poi, 'coords': moved(poi['coords']), 'steric_atoms': moved(poi['steric_atoms']),
         'steric_ca_center': moved(poi['steric_ca_center'])},
        {**ligand, 'coords': moved(ligand['coords'])}, torch.eye(3))
    assert torch.allclose(scalar, transformed[0], atol=1e-5)
    assert torch.allclose(vector @ rotation.T, transformed[1], atol=1e-5)
    altered = {**e3, 'steric_atoms': e3['steric_atoms']+torch.tensor([5., 0., 0.])}
    changed = candidate_fullatom_evidence(coords, types, altered, poi, ligand, torch.eye(3))[0]
    assert not torch.allclose(scalar[2], changed[2]), 'ligand node cannot see sidechain motion'

def test_rigid_body_output_heads_are_parameter_independent():
    from tricoopdiff.models.assembly import LegalPoseDecoder

    torch.manual_seed(9)
    decoder = LegalPoseDecoder(8, 4).double()
    h = torch.randn(6, 8, dtype=torch.double)
    coords, vectors = torch.randn(6, 3, dtype=torch.double), torch.randn(6, 4, 3, dtype=torch.double)
    types = torch.tensor([0, 0, 1, 1, 2, 2])
    state = torch.zeros(1, 9, dtype=torch.double)
    state[:, -1] = 1.
    output = decoder(h, vectors, coords, types, torch.empty(2, 0, dtype=torch.long),
                     torch.zeros(0, 2, dtype=torch.bool), state, [torch.empty(0), torch.empty(0)])
    for own, other in (BRANCHES, BRANCHES[::-1]):
        own_parameters = list(decoder.rigid[own].parameters())
        other_parameters = list(decoder.rigid[other].parameters())
        loss = sum(output[own][field].square().sum() for field in ('tr', 'rot'))
        gradients = torch.autograd.grad(loss, own_parameters + other_parameters,
                                        retain_graph=True, allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in gradients[:len(own_parameters)])
        assert all(g is None for g in gradients[len(own_parameters):]), (
            'one-step rigid heads must not share parameters across physical bodies')


def test_shared_rigid_migration_duplicates_values_not_parameter_storage():
    model = TernaryAssembly(node_dim=32, input_dim=16)
    legacy = dict(model.state_dict())
    for layer in ('0', '2'):
        for suffix in ('weight', 'bias'):
            prefix = 'controller.assembly.rigid'
            value = legacy.pop(f'{prefix}.{BRANCHES[0]}.{layer}.{suffix}')
            legacy.pop(f'{prefix}.{BRANCHES[1]}.{layer}.{suffix}')
            legacy[f'{prefix}.{layer}.{suffix}'] = value
    migrated = model.migrate_shared_rigid_weights(legacy)
    model.load_state_dict(migrated, strict=True)
    for left, right in zip(model.controller.assembly.rigid[BRANCHES[0]].parameters(),
                           model.controller.assembly.rigid[BRANCHES[1]].parameters()):
        assert torch.equal(left, right)
        assert left.data_ptr() != right.data_ptr(), 'migration must not tie the new body heads'


def test_matcha_evidence_preserves_complete_frame_and_equivariance():
    from tricoopdiff.models.expert import FrozenMatcha

    torch.manual_seed(17)
    receptor = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    ligand = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    frame = torch.eye(3, dtype=torch.double, requires_grad=True)
    vectors = FrozenMatcha.node_vectors(receptor, ligand, frame)
    assert vectors.shape == (11, FrozenMatcha.node_vector_dim, 3)
    assert torch.equal(vectors[:, 2:], frame.T[None].expand(11, -1, -1)), (
        'canonical-frame tokens need all three basis axes, not just the first')
    rotation = so3_exp(torch.tensor([[.2, -.5, .7]], dtype=torch.double))[0]
    moved = FrozenMatcha.node_vectors(receptor @ rotation.T + 17.,
                                     ligand @ rotation.T + 17., rotation @ frame)
    assert torch.allclose(moved, vectors @ rotation.T, atol=1e-12)
    weights = torch.randn_like(vectors)
    gradients = torch.autograd.grad((vectors * weights).sum(), (receptor, ligand, frame))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert (gradients[2].abs().sum(0) > 0).all(), 'each local frame axis must remain differentiable'


def test_logit_only_weight_migration_preserves_logits_and_zeros_new_values():
    model = TernaryAssembly(node_dim=32, input_dim=16)
    current = model.state_dict()
    legacy = {key: value.clone() for key, value in current.items()}
    heads = [('controller.structure.geometry.2', 1),
             ('controller.assembly.distance.2', 1)]
    heads.extend((f'controller.layers.{i}.geometry.2', layer.heads)
                 for i, layer in enumerate(model.controller.layers))
    for prefix, count in heads:
        for suffix in ('weight', 'bias'):
            key = f'{prefix}.{suffix}'
            legacy[key] = legacy[key][-count:].clone()
    migrated = model.migrate_logit_only_geometry(legacy)
    model.load_state_dict(migrated, strict=True)
    for prefix, count in heads:
        for suffix in ('weight', 'bias'):
            key = f'{prefix}.{suffix}'
            assert torch.equal(migrated[key][-count:], legacy[key])
            assert torch.count_nonzero(migrated[key][:-count]) == 0
    malformed = dict(legacy)
    malformed['controller.structure.geometry.2.weight'] = torch.empty(2, 32)
    try:
        model.migrate_logit_only_geometry(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError('migration must reject unrelated shape mismatches')


def test_rigid_decoder_reads_distance_with_identical_node_identities():
    from tricoopdiff.models.assembly import LegalPoseDecoder

    torch.manual_seed(3)
    decoder = LegalPoseDecoder(8, 4).double()
    h = torch.randn(1, 8, dtype=torch.double).expand(6, -1)
    coords = torch.randn(6, 3, dtype=torch.double)
    vectors = torch.randn(6, 4, 3, dtype=torch.double)
    types = torch.tensor([0, 0, 1, 1, 2, 2])
    state = torch.zeros(1, 9, dtype=torch.double)
    state[:, -1] = 1.
    output = decoder(h, vectors, coords, types, torch.empty(2, 0, dtype=torch.long),
                     torch.zeros(0, 2, dtype=torch.bool), state,
                     [torch.empty(0), torch.empty(0)])
    loss = sum(output[branch][field].square().sum()
               for branch in BRANCHES for field in ('tr', 'rot'))
    gradient = torch.autograd.grad(loss, decoder.distance[-1].weight)[0]
    assert gradient.norm() > 1e-8, (
        'rigid control must read geometric values even when atom identities coincide')


def test_interaction_geometry_reaches_scalar_values_with_repeated_identities():
    from tricoopdiff.models.assembly import EquivariantInteraction

    torch.manual_seed(2)
    layer = EquivariantInteraction(8, 4).double()
    h = torch.randn(1, 8, dtype=torch.double).expand(6, -1)
    coords = torch.randn(6, 3, dtype=torch.double)
    vectors = torch.zeros(6, 4, 3, dtype=torch.double)
    types = torch.tensor([0, 0, 1, 1, 2, 2])
    scalar, _ = layer._messages(layer.query(h), vectors, coords, layer.key(h),
                               layer.value(h), vectors, coords, types,
                               torch.zeros(6, 6, 4, dtype=torch.double))
    gradient = torch.autograd.grad(scalar.square().sum(), layer.geometry[-1].weight)[0]
    assert gradient.norm() > 1e-6, (
        'scalar geometry must not vanish merely because source identity values coincide')


def test_protein_structure_reads_distances_with_identical_node_identities():
    from tricoopdiff.models.assembly import ComponentStructureEncoder

    torch.manual_seed(1)
    encoder = ComponentStructureEncoder(8).double()
    identity = torch.randn(1, 8, dtype=torch.double).expand(6, -1)
    coords = torch.randn(6, 3, dtype=torch.double)
    types = torch.tensor([0, 0, 1, 1, 2, 2])
    edges = torch.tensor([[4, 5], [5, 4]])
    attributes = torch.zeros(2, 4, dtype=torch.double)
    original = encoder(identity, coords, types, edges, attributes)
    changed_coords = coords.clone()
    changed_coords[0] *= 4
    changed = encoder(identity, changed_coords, types, edges, attributes)
    assert not torch.allclose(original[:2], changed[:2], atol=1e-8), (
        'protein geometry must survive even when attention values have identical identities')


def assembly_fixture():
    torch.manual_seed(71)
    graph = {'node_features': torch.randn(15, 16), 'node_coords': torch.randn(15, 3),
             'node_types': torch.tensor([0] * 5 + [1] * 6 + [2] * 4),
             'lig_edge_index': torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
             'lig_edge_mask': torch.tensor([False, False, True, False, False, False]),
             'lig_mask_rotate': torch.tensor([[False, False, True, True]]),
             'lig_edge_attr': torch.nn.functional.one_hot(torch.zeros(6, dtype=torch.long), 4).float(),
             'expert_node_features': torch.cat((torch.randn(15, 116), torch.ones(15, 2)), -1),
             'expert_node_vectors': torch.randn(15, 40, 3)}
    proposals = {b: {c: torch.randn(1, 1 if c == 'tor' else 3)
                      for c in ('tr', 'rot', 'tor')} for b in BRANCHES}
    fuse = TernaryAssembly(node_dim=32, input_dim=16).eval()
    state = torch.zeros(1, 9)
    state[:, 5:8], state[:, -1] = .05, 1.
    return fuse, graph, proposals, state


def test_v8_unified_controller_equivariance_and_gradient_paths():
    from tricoopdiff.models.assembly import VECTOR_EXPERT_MODEL_PROFILE, production_model_config

    _, graph, proposals, state = assembly_fixture()
    fuse = TernaryAssembly(**production_model_config(
        32, 16, model_profile=VECTOR_EXPERT_MODEL_PROFILE, expert_vector_dim=2,
    )).train()
    assert not hasattr(fuse, 'pooling'), 'new controller must not retain summary control modules'
    graph['expert_node_vectors'] = torch.randn(15, 4, 3, requires_grad=True)
    graph['node_coords'].requires_grad_()
    base = fuse(graph, proposals, state)
    rotation = so3_exp(torch.tensor([[.2, -.5, .7]]))[0]
    moved_graph = {**graph,
                   'node_coords': graph['node_coords'] @ rotation.T + 17.,
                   'expert_node_vectors': graph['expert_node_vectors'] @ rotation.T}
    moved_proposals = {branch: {key: (value @ rotation.T if key != 'tor' else value)
                                for key, value in proposal.items()}
                       for branch, proposal in proposals.items()}
    moved = fuse(moved_graph, moved_proposals, state)
    for branch in BRANCHES:
        for field in ('tr', 'rot'):
            assert torch.allclose(moved['normalized_velocity'][branch][field],
                                  base['normalized_velocity'][branch][field] @ rotation.T,
                                  atol=4e-5), (branch, field)
    assert torch.allclose(base['normalized_velocity']['torsion'],
                          moved['normalized_velocity']['torsion'], atol=4e-5)
    assert torch.allclose(base['rank_features'], moved['rank_features'], atol=4e-5)
    output = base['normalized_velocity']
    loss = sum(output[b][c].square().sum() for b in BRANCHES for c in ('tr', 'rot'))
    loss = loss + output['torsion'].square().sum()
    loss.backward()
    for key in ('expert_node_vectors', 'node_coords'):
        gradient = graph[key].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0, key
    for name, parameter in fuse.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_conditional_node_controller_does_not_consume_binary_rigid_advice():
    from tricoopdiff.models.assembly import VECTOR_EXPERT_MODEL_PROFILE, production_model_config

    _, graph, proposals, state = assembly_fixture()
    graph['expert_node_vectors'] = torch.randn(15, 4, 3, requires_grad=True)
    graph['expert_node_features'].requires_grad_()
    fuse = TernaryAssembly(**production_model_config(
        32, 16, model_profile=VECTOR_EXPERT_MODEL_PROFILE, expert_vector_dim=2)).eval()
    base = fuse(graph, proposals, state)['normalized_velocity']
    changed = {branch: {**proposal, 'tr': proposal['tr'] * 100 + 17,
                       'rot': proposal['rot'] * -100 - 23}
               for branch, proposal in proposals.items()}
    alternate = fuse(graph, changed, state)['normalized_velocity']
    for branch in BRANCHES:
        for field in ('tr', 'rot'):
            assert torch.equal(base[branch][field], alternate[branch][field]), (
                'binary rigid advice must not bypass conditional node evidence')
    assert torch.equal(base['torsion'], alternate['torsion'])
    loss = sum(base[branch][field].square().sum()
               for branch in BRANCHES for field in ('tr', 'rot'))
    gradients = torch.autograd.grad(loss, (graph['expert_node_features'], graph['expert_node_vectors']))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients), (
        'removing rigid advice must preserve differentiable conditional binary evidence')


def test_v8_distant_component_and_atom_identity_reach_actions():
    from tricoopdiff.models.assembly import VECTOR_EXPERT_MODEL_PROFILE, production_model_config

    _, graph, proposals, state = assembly_fixture()
    graph['node_coords'][:5] += 200.
    graph['expert_node_vectors'] = torch.randn(15, 4, 3, requires_grad=True)
    graph['expert_node_features'].requires_grad_()
    fuse = TernaryAssembly(**production_model_config(
        32, 16, model_profile=VECTOR_EXPERT_MODEL_PROFILE, expert_vector_dim=2,
    )).eval()
    output = fuse(graph, proposals, state)['normalized_velocity']
    for action in (output['poi_ligand']['tr'], output['torsion']):
        gradients = torch.autograd.grad(action.square().sum(),
                                         (graph['expert_node_vectors'], graph['expert_node_features']),
                                         retain_graph=True)
        for gradient in gradients:
            assert gradient[:5].abs().sum() > 0, 'distant E3 information must reach POI and torsion'
    swapped = graph['expert_node_features'].detach().clone()
    swapped[[-1, -2]] = swapped[[-2, -1]]
    changed = fuse({**graph, 'expert_node_features': swapped}, proposals, state)['normalized_velocity']
    assert not torch.allclose(output['torsion'], changed['torsion'], atol=1e-9), (
        'bond control must retain the association between atom identity and its geometric lever')


def test_v8_node_permutation_and_empty_torsions():
    from tricoopdiff.models.assembly import VECTOR_EXPERT_MODEL_PROFILE, production_model_config

    _, graph, proposals, state = assembly_fixture()
    graph['expert_node_vectors'] = torch.randn(15, 4, 3)
    fuse = TernaryAssembly(**production_model_config(
        32, 16, model_profile=VECTOR_EXPERT_MODEL_PROFILE, expert_vector_dim=2,
    )).eval()
    base = fuse(graph, proposals, state)['normalized_velocity']
    permutation = torch.tensor([4, 2, 0, 3, 1, 10, 8, 6, 9, 7, 5, 14, 12, 11, 13])
    changed = {**graph}
    for key in ('node_features', 'node_coords', 'node_types',
                'expert_node_features', 'expert_node_vectors'):
        changed[key] = graph[key][permutation]
    ligand_permutation = permutation[11:] - 11
    inverse = torch.argsort(ligand_permutation)
    changed['lig_edge_index'] = inverse[graph['lig_edge_index']]
    changed['lig_mask_rotate'] = graph['lig_mask_rotate'][:, ligand_permutation]
    result = fuse(changed, proposals, state)['normalized_velocity']
    for branch in BRANCHES:
        for field in ('tr', 'rot'):
            assert torch.allclose(base[branch][field], result[branch][field], atol=1e-6)
    assert torch.allclose(base['torsion'], result['torsion'], atol=1e-6)
    empty_proposals = {b: {**p, 'tor': torch.zeros(1, 0)} for b, p in proposals.items()}
    assert fuse(graph, empty_proposals, state)['normalized_velocity']['torsion'].shape == (1, 0), (
        'no_torsion experts must disable torsions even on a flexible molecular graph')
    graph['lig_edge_mask'] = torch.zeros_like(graph['lig_edge_mask'])
    graph['lig_mask_rotate'] = torch.zeros(0, 4, dtype=torch.bool)
    proposals = {b: {**p, 'tor': torch.zeros(1, 0)} for b, p in proposals.items()}
    assert fuse(graph, proposals, state)['normalized_velocity']['torsion'].shape == (1, 0)


def test_v8_chunked_attention_matches_direct_backward():
    from tricoopdiff.models.assembly import EquivariantInteraction

    torch.manual_seed(128)
    layer = EquivariantInteraction(16, 4)
    h = torch.randn(69, 16, requires_grad=True)
    vectors = torch.randn(69, 4, 3, requires_grad=True)
    coords = torch.randn(69, 3, requires_grad=True)
    types = torch.arange(69) % 3
    edges = torch.zeros(2, 0, dtype=torch.long)
    attributes = torch.zeros(0, 4)
    layer.train()
    chunked = layer(h, vectors, coords, types, edges, attributes)
    chunked_grad = torch.autograd.grad(sum(x.square().sum() for x in chunked),
                                       (h, vectors, coords), retain_graph=True)
    full = layer._messages(layer.query(h), vectors, coords, layer.key(h), layer.value(h),
                           vectors, coords, types, h.new_zeros(69, 69, 4))
    expected_vectors = vectors + layer.vector_mix(full[1].transpose(1, 2)).transpose(1, 2)
    expected_vectors = expected_vectors / (1. + expected_vectors.square().mean((1, 2), keepdim=True)).sqrt()
    norms = expected_vectors.square().sum(-1).clamp_min(1e-12).sqrt()
    expected_h = layer.norm(h + layer.scalar_mix(torch.cat((h, full[0], norms), -1)))
    expected_grad = torch.autograd.grad(expected_h.square().sum() + expected_vectors.square().sum(),
                                        (h, vectors, coords))
    for actual, expected in zip(chunked, (expected_h, expected_vectors)):
        assert torch.allclose(actual, expected, atol=2e-6)
    for actual, expected in zip(chunked_grad, expected_grad):
        assert torch.allclose(actual, expected, atol=2e-5), 'chunking changed action gradients'


def test_v8_expert_node_identity_frame_and_scatter_gradients():
    from types import SimpleNamespace
    from tricoopdiff.models.assembly import TernaryAssemblyModel, production_model_config

    class FeatureExpert(torch.nn.Module):
        node_scalar_dim = 2
        node_vector_dim = 2
        num_conv_layers = 1

        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(dynamic_max_cross=True)

        def forward_features(self, graph, *, include_vectors=False):
            coords = torch.cat((graph['ligand'].pos, graph['receptor'].pos))
            identities = torch.cat((graph['ligand'].x[:, 0], graph['receptor'].x[:, 0]))
            scalars = torch.stack((coords.square().sum(-1), identities), -1)
            zero = coords.new_zeros(1, 3)
            result = (zero, zero, coords.new_zeros(1), scalars)
            if include_vectors:
                result += (coords[:, None, :].expand(-1, self.node_vector_dim, -1),)
            return result

    positions = torch.zeros(12, 3)
    positions[:, 0] = torch.tensor(list(range(9)) + [100, 101, 102])
    features = torch.zeros(1, 12, 16)
    features[0, :, 0] = torch.arange(12)
    edges = torch.stack((torch.arange(11), torch.arange(1, 12)))
    edges = torch.cat((edges, edges.flip(0)), -1)
    e3 = {'coords': positions[None].clone().requires_grad_(),
          'features': features, 'edge_index': edges}
    poi = {'coords': (positions[None] + torch.tensor([0., 2., 0.])).requires_grad_(),
           'features': features + 100, 'edge_index': edges}
    e3['local_geometry'] = torch.arange(12 * 32).reshape(1, 12, 32).float() / 100
    poi['local_geometry'] = e3['local_geometry'] + 1
    _, graph, _, state = assembly_fixture()
    state[:, -1] = .5
    ligand = {'coords': torch.tensor([[[0., 0., 0.], [1., 1., 0.],
                                       [2., 0., 1.], [3., 1., 1.]]], requires_grad=True),
              'features': torch.zeros(1, 4, 16), 'edge_index': graph['lig_edge_index'],
              'edge_attr': graph['lig_edge_attr'], 'edge_mask': graph['lig_edge_mask'],
              'mask_rotate': graph['lig_mask_rotate']}
    from tricoopdiff.models.assembly import VECTOR_EXPERT_MODEL_PROFILE
    vector_fuse = TernaryAssembly(**production_model_config(
        32, 16, model_profile=VECTOR_EXPERT_MODEL_PROFILE,
        expert_scalar_dim=2, expert_vector_dim=2)).eval()
    vector_expert = FeatureExpert()
    vector_model = TernaryAssemblyModel(vector_expert, vector_fuse)
    assert vector_expert.model.smooth_cross_edges, 'v8 requires smooth DiffDock cross-edge support'
    assert not getattr(vector_expert.model, 'smooth_edges', False), 'v8 must preserve DiffDock internal edge weights'
    angle = torch.tensor([[.3, -.2, .7]])
    frame = so3_exp(angle)
    vector_capture = []
    hook = vector_fuse.register_forward_pre_hook(lambda module, args: vector_capture.append(args[0]))
    try:
        vector_model(e3, poi, ligand, torch.tensor([.5]), r_e3_to_poi=frame,
                     receptor_crop_cutoff=torch.tensor([4.]), diffusion_state=state)
    finally:
        hook.remove()
    structural_reads = vector_capture[0]['protein_local_geometry']
    assert torch.equal(structural_reads[:12, :32], e3['local_geometry'][0])
    assert torch.equal(structural_reads[12:24, :32], poi['local_geometry'][0])
    assert structural_reads[:24, -1].eq(1).all()
    assert structural_reads[24:].eq(0).all(), 'ligand nodes must not receive protein structure slots'
    vector_reads = vector_capture[0]['expert_node_vectors']
    e3_vectors, poi_vectors = vector_reads[:12], vector_reads[12:24]
    expert_scalar_reads = vector_capture[0]['expert_node_features']
    e3_visible = expert_scalar_reads[:12, 4].bool()
    poi_visible = expert_scalar_reads[12:24, 5].bool()
    e3_expected = (e3['coords'][0] - e3['coords'][0].mean(0))[:, None].expand(-1, 2, -1)
    poi_expected = (poi['coords'][0] - poi['coords'][0].mean(0))[:, None].expand(-1, 2, -1)
    assert e3_visible.all() and poi_visible.all(), 'v8 keeps a fixed full-receptor expert node set'
    assert torch.allclose(e3_vectors[:, :2], e3_expected)
    assert e3_vectors[:, 2:].count_nonzero() == 0
    assert poi_vectors[:, :2].count_nonzero() == 0
    assert torch.allclose(poi_vectors[:, 2:], poi_expected, atol=1e-5)


    observed = vector_capture[0]['expert_node_features']
    for read, marker, visibility, offset in ((observed[:12], 1, 4, 0),
                                             (observed[12:24], 3, 5, 100)):
        assert read[:, visibility].bool().all()
        assert torch.equal(read[:, marker], torch.arange(12).float() + offset)
    (observed[:, :4].sum() + vector_reads.square().sum()).backward()
    for coords in (e3['coords'], poi['coords'], ligand['coords']):
        assert coords.grad is not None and torch.isfinite(coords.grad).all()
        assert coords.grad.abs().sum() > 0, 'binary reads must retain coordinate Jacobians'
    # Exercise the actual wrapper boundary, not only a node-count predicate.
    from unittest.mock import patch
    from tricoopdiff.models import assembly as assembly_module
    calls = []
    original_checkpoint = assembly_module.checkpoint

    def traced_checkpoint(function, *args, **kwargs):
        if function is vector_fuse:
            calls.append(function)
        return original_checkpoint(function, *args, **kwargs)

    vector_model.train()
    vector_model.checkpoint_ternary = True
    node_count = 12 + 12 + 4
    with patch.object(assembly_module, 'checkpoint', traced_checkpoint):
        vector_model.checkpoint_min_nodes = node_count + 1
        vector_model(e3, poi, ligand, torch.tensor([.5]), diffusion_state=state)
        assert not calls, 'small cases must retain ternary activations'
        vector_model.checkpoint_min_nodes = node_count
        vector_model(e3, poi, ligand, torch.tensor([.5]), diffusion_state=state)
        assert len(calls) == 1, 'threshold-sized cases must still recompute'
        vector_model.checkpoint_min_nodes = 0
        vector_model(e3, poi, ligand, torch.tensor([.5]), diffusion_state=state)
        assert len(calls) == 2, 'zero threshold must preserve historical recomputation'



def test_bounded_vector_retains_large_finite_directions_and_coordinate_gradients():
    from tricoopdiff.models.assembly import bounded_vector

    for dtype, magnitude in ((torch.float32, 1e20), (torch.float64, 1e200)):
        values = torch.tensor([[magnitude, 0., 0.]], dtype=dtype, requires_grad=True)
        compressed = bounded_vector(values)
        assert torch.equal(compressed, torch.tensor([[1., 0., 0.]], dtype=dtype)), (
            'finite expert directions disappeared because their squared norm overflowed')
        compressed.sum().backward()
        assert torch.isfinite(values.grad).all() and values.grad[0, 1] > 0
        assert torch.allclose(values.grad[0, 1] * magnitude, values.new_tensor(1.))
    ordinary = torch.tensor([[0., 0., 0.], [.4, -.8, 2.]], requires_grad=True)
    expected = ordinary / (1 + torch.linalg.vector_norm(ordinary, dim=-1, keepdim=True))
    actual = bounded_vector(ordinary)
    assert torch.equal(actual, expected), 'ordinary old-checkpoint arithmetic changed'
    actual.sum().backward()
    assert torch.isfinite(ordinary.grad).all()
    values = torch.tensor([[1e20, -2e20, 3e20]])
    rotation = so3_exp(torch.tensor([[.2, -.5, .7]]))[0]
    assert torch.allclose(bounded_vector(values @ rotation.T),
                          bounded_vector(values) @ rotation.T, atol=2e-7, rtol=0.)


def test_v8_has_no_legacy_controller_parameters_or_profile_switches():
    from tricoopdiff.models.assembly import production_model_config, VECTOR_EXPERT_MODEL_PROFILE

    config = production_model_config(32, 16, expert_vector_dim=2)
    assert config['model_profile'] == VECTOR_EXPERT_MODEL_PROFILE
    model = TernaryAssembly(**config)
    assert all(name.startswith('controller.') for name in model.state_dict())
    for obsolete in ('pooling', 'evidence_encoder', 'motion_head', 'direct_torsion_head',
                     'gate_head', 'residual_head', 'bond_context_encoder'):
        assert not hasattr(model, obsolete), obsolete
    for profile in ('ternary_endpoint_v1', 'ternary_endpoint_v4', 'ternary_endpoint_v5',
                    'ternary_endpoint_v6', 'ternary_endpoint_v7'):
        try:
            TernaryAssembly(32, 16, model_profile=profile)
        except ValueError as error:
            assert 'retired architecture' in str(error)
        else:
            raise AssertionError('retired motion architecture remained constructible')
    try:
        TernaryAssembly(32, 16, global_node_context=True)
    except TypeError:
        pass
    else:
        raise AssertionError('old architecture switches must not remain active')


def test_v8_source_conditioning_preserves_common_initialization():
    from tricoopdiff.models.assembly import production_model_config

    torch.manual_seed(129)
    plain = TernaryAssembly(**production_model_config(32, 16, expert_vector_dim=2))
    torch.manual_seed(129)
    conditioned = TernaryAssembly(**production_model_config(32, 16, True, expert_vector_dim=2))
    for name, value in plain.state_dict().items():
        assert torch.equal(value, conditioned.state_dict()[name]), name


def test_legal_pose_identity_execution_and_bond_expert_gradients():
    from tricoopdiff.models.assembly import LegalPoseDecoder
    from tricoopdiff.sdpi_flow import SDPIFlow

    _, graph, _, state = assembly_fixture()
    decoder = LegalPoseDecoder(32, 4)
    coords, types = graph['node_coords'], graph['node_types']
    h, vectors = torch.randn(15, 32), torch.randn(15, 4, 3)
    bonds = graph['lig_edge_index'][:, graph['lig_edge_mask']]
    masks = graph['lig_mask_rotate']
    opinions = [torch.randn(1, requires_grad=True) for _ in BRANCHES]
    motion = decoder(h, vectors, coords, types, bonds, masks, state, opinions)
    gradients = torch.autograd.grad(motion['torsion'].sum(), opinions)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), (
        'both experts must influence the corresponding torsion, not only its count')
    data = {'edge_index': graph['lig_edge_index'], 'edge_mask': graph['lig_edge_mask'],
            'mask_rotate': masks}
    def execute(output):
        velocity = dict(output)
        for branch, column in zip(BRANCHES, (0, 3)):
            velocity[branch] = {'tr': output[branch]['tr'] * state[:, column:column + 1].exp(),
                                'rot': output[branch]['rot'] * torch.pi}
        velocity['torsion'] = output['torsion'] * torch.pi
        return SDPIFlow.differentiable_step(coords[types == 2][None], coords[types == 1][None],
                                            torch.eye(3)[None], velocity, torch.ones(1), data)
    ligand, poi, _ = execute(motion)
    assert torch.allclose(torch.cdist(poi, poi), torch.cdist(coords[types == 1][None],
                                                          coords[types == 1][None]), atol=1e-6)
    edges = graph['lig_edge_index']
    old_lengths = (coords[types == 2][edges[0]] - coords[types == 2][edges[1]]).norm(dim=-1)
    new_lengths = (ligand[0, edges[0]] - ligand[0, edges[1]]).norm(dim=-1)
    assert torch.allclose(old_lengths, new_lengths, atol=1e-6), 'legal torsions must preserve bond lengths'
    for head in (*decoder.rigid.values(), decoder.bond):
        torch.nn.init.zeros_(head[-1].weight)
        torch.nn.init.zeros_(head[-1].bias)
    identity = decoder(h, vectors, coords, types, bonds, masks, state, opinions)
    ligand, poi, _ = execute(identity)
    assert torch.allclose(ligand[0], coords[types == 2], atol=1e-6)
    assert torch.allclose(poi[0], coords[types == 1], atol=1e-6)


def test_joint_dof_readout_couples_bonds_and_rigid_without_bond_order_dependence():
    from tricoopdiff.models.assembly import LegalPoseDecoder

    _, graph, _, state = assembly_fixture()
    decoder = LegalPoseDecoder(32, 4).eval()
    h, vectors = torch.randn(15, 32), torch.randn(15, 4, 3)
    coords, types = graph['node_coords'], graph['node_types']
    bonds = torch.tensor([[0, 1], [1, 2]])
    masks = torch.tensor([[False, True, True, True], [False, False, True, True]])
    opinions = [torch.randn(2, requires_grad=True) for _ in BRANCHES]
    output = decoder(h, vectors, coords, types, bonds, masks, state, opinions)
    for value in (output['torsion'][0, 0], output[BRANCHES[1]]['tr'].sum()):
        gradients = torch.autograd.grad(value, opinions, retain_graph=True)
        assert all(torch.isfinite(g).all() and g[1].abs() > 0 for g in gradients), (
            'other bond evidence must reach this bond and the rigid decision')
    order = torch.tensor([1, 0])
    moved = decoder(h, vectors, coords, types, bonds[:, order], masks[order], state,
                    [opinion[order] for opinion in opinions])
    assert torch.allclose(moved['torsion'], output['torsion'][:, order], atol=1e-6)
    for branch in BRANCHES:
        for field in ('tr', 'rot'):
            assert torch.allclose(moved[branch][field], output[branch][field], atol=1e-6), (
                'physical bond ordering must not alter rigid decisions')


def test_intrinsic_memory_is_independent_of_component_placement_and_ligand_torsion():
    from tricoopdiff.models.assembly import ComponentStructureEncoder
    from tricoopdiff.sdpi_flow import SDPIFlow

    _, graph, _, _ = assembly_fixture()
    encoder = ComponentStructureEncoder(32).double()
    identity = torch.randn(15, 32, dtype=torch.float64, requires_grad=True)
    coords = graph['node_coords'].double().requires_grad_()
    types = graph['node_types']
    ids = (types == 2).nonzero().flatten()
    edges = ids[graph['lig_edge_index']]
    attrs = graph['lig_edge_attr'].double()
    original = encoder(identity, coords, types, edges, attrs)
    transformed = coords.clone()
    for component in range(3):
        rotation = so3_exp(torch.randn(1, 3, dtype=torch.float64))[0]
        selected = types == component
        transformed[selected] = coords[selected] @ rotation.T + torch.randn(3).double() * 40.
    moved = encoder(identity, transformed, types, edges, attrs)
    assert torch.allclose(original, moved, atol=1e-10), 'intrinsic memory must ignore relative poses'
    velocity = {b: {'tr': coords.new_zeros(1, 3), 'rot': coords.new_zeros(1, 3)} for b in BRANCHES}
    velocity.update(torsion=coords.new_tensor([[1.3]]), rigid_output_mode='independent_centroids_v1',
                    torsion_update_mode='continuous_horizontal_v1')
    data = {'edge_index': graph['lig_edge_index'], 'edge_mask': graph['lig_edge_mask'],
            'mask_rotate': graph['lig_mask_rotate']}
    ligand, _, _ = SDPIFlow.differentiable_step(coords[types == 2][None], coords[types == 1][None],
                                               torch.eye(3).double()[None], velocity,
                                               torch.ones(1).double(), data)
    folded = coords.clone()
    folded[types == 2] = ligand[0]
    assert not torch.allclose(folded[types == 2], coords[types == 2])
    assert torch.allclose(original, encoder(identity, folded, types, edges, attrs), atol=1e-10), (
        'legal torsions must not rewrite ligand chemical identity memory')
    gradient = torch.autograd.grad(original[5, 0], coords, retain_graph=True)[0]
    assert torch.equal(gradient[types != 1], torch.zeros_like(gradient[types != 1]))
    original[:, 0].sum().backward()
    assert torch.isfinite(identity.grad).all() and identity.grad.norm() > 0


def test_chemical_multihop_and_rotor_geometry_match_physical_action():
    from tricoopdiff.models.assembly import ComponentStructureEncoder, LegalPoseDecoder

    _, graph, _, _ = assembly_fixture()
    coords, types = graph['node_coords'].double(), graph['node_types']
    ligand = types == 2
    ids = ligand.nonzero().flatten()
    identity = torch.randn(15, 32, dtype=torch.float64, requires_grad=True)
    encoder = ComponentStructureEncoder(32).double()
    encoded = encoder(identity, coords, types, ids[graph['lig_edge_index']], graph['lig_edge_attr'].double())
    gradient = torch.autograd.grad(encoded[ids[0], 0], identity)[0]
    assert gradient[ids[-1]].abs().sum() > 0, 'chemical memory must propagate beyond one bond'
    vectors = torch.randn(15, 4, 3, dtype=torch.float64)
    bonds = graph['lig_edge_index'][:, graph['lig_edge_mask']]
    masks = graph['lig_mask_rotate']
    geometry = LegalPoseDecoder.action_geometry(coords, vectors, ligand, bonds, masks)
    xyz = coords[ligand]
    step = 1e-6
    from tricoopdiff.sdpi_flow import SDPIFlow
    ligand_data = {'edge_index': bonds, 'edge_mask': torch.ones(bonds.shape[1], dtype=torch.bool),
                   'mask_rotate': masks}
    velocities = {branch: {'tr': torch.zeros(1, 3, dtype=xyz.dtype),
                          'rot': torch.zeros(1, 3, dtype=xyz.dtype)} for branch in BRANCHES}
    velocities.update(torsion=torch.tensor([[-1.]], dtype=xyz.dtype),
                      rigid_output_mode='independent_centroids_v1',
                      torsion_update_mode='continuous_horizontal_v1')
    moved, _, _ = SDPIFlow.differentiable_step(
        xyz[None], coords[types == 1][None], torch.eye(3, dtype=xyz.dtype)[None],
        velocities, torch.tensor([step], dtype=xyz.dtype), ligand_data)
    tangent = (moved[0] - xyz) / (10. * step)
    projection = (vectors[ligand] * tangent[:, None]).sum(-1)
    assert torch.allclose(geometry[0, ligand, -4:], projection, atol=1e-6), (
        'action features must encode the executed horizontal rotor derivative')
    rotation = so3_exp(torch.tensor([[.7, -.4, .9]], dtype=torch.float64))[0]
    transformed = LegalPoseDecoder.action_geometry(coords @ rotation.T + 30.,
                                                   vectors @ rotation.T, ligand, bonds, masks)
    assert torch.allclose(geometry, transformed, atol=1e-10)


def test_global_chemical_read_and_explicit_dof_overlap():
    from tricoopdiff.models.assembly import ComponentStructureEncoder, LegalPoseDecoder

    count = 12
    coords = torch.randn(count + 2, 3, dtype=torch.float64)
    types = torch.tensor([0, 1] + [2] * count)
    left = torch.arange(2, count + 1)
    right = left + 1
    edges = torch.stack((torch.cat((left, right)), torch.cat((right, left))))
    attributes = torch.zeros(len(edges[0]), 4, dtype=torch.float64)
    attributes[:, 0] = 1.
    identity = torch.randn(count + 2, 32, dtype=torch.float64, requires_grad=True)
    encoder = ComponentStructureEncoder(32).double()
    encoded = encoder(identity, coords, types, edges, attributes)
    gradient = torch.autograd.grad(encoded[2, 0], identity)[0]
    assert gradient[-1].abs().sum() > 0, 'chemical structure read must reach beyond four hops'
    chemistry, connected = encoder.chemical_relations(coords[2:], edges - 2, attributes)
    assert connected.all() and torch.allclose(chemistry[0, -1, 0], torch.tensor(12.).double().log())
    positions = torch.zeros(3, 3)
    affected = torch.tensor([[1., 1., 1., 1.], [0., 0., 1., 1.], [1., 1., 0., 0.]])
    relation = LegalPoseDecoder.joint_relations(positions, affected)
    assert relation[1, 2, 1] == 0 and relation[0, 1, 1] == .5
    assert relation[0, 1, 2] == .5 and relation[1, 0, 2] == 1., (
        'subset/superset action overlap must retain its direction')
    xyz = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    rotors = torch.tensor([[0, 1], [1, 0]])
    masks = torch.tensor([[False, True, True, False, False]]).expand(2, -1)
    axes, tangents = LegalPoseDecoder.rotor_tangents(xyz, rotors, masks)
    assert torch.allclose(axes[0], -axes[1])
    assert torch.allclose(tangents[0], -tangents[1]), (
        'action coupling must distinguish reinforcing from cancelling rotor directions')
    assert torch.allclose(tangents.mean(1), torch.zeros(2, 3, dtype=xyz.dtype), atol=1e-12)
    rotation = so3_exp(torch.tensor([[.2, -.4, .7]], dtype=xyz.dtype))[0]
    moved_axes, moved_tangents = LegalPoseDecoder.rotor_tangents(xyz @ rotation.T + 30., rotors, masks)
    assert torch.allclose(moved_axes, axes @ rotation.T, atol=1e-12)
    assert torch.allclose(moved_tangents, tangents @ rotation.T, atol=1e-12)
    gradient = torch.autograd.grad(tangents.square().sum(), xyz)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_physical_velocity_has_no_inverse_remaining_time_gain():
    from tricoopdiff.models.assembly import LegalPoseDecoder

    _, graph, _, state = assembly_fixture()
    decoder = LegalPoseDecoder(32, 4).eval()
    h = torch.randn(15, 32, requires_grad=True)
    vectors = torch.randn(15, 4, 3)
    bonds = graph['lig_edge_index'][:, graph['lig_edge_mask']]
    opinions = [torch.randn(1) for _ in BRANCHES]
    args = (h, vectors, graph['node_coords'], graph['node_types'], bonds, graph['lig_mask_rotate'])
    full = decoder(*args, state, opinions)
    short = state.clone()
    short[:, -1] = .01
    short[:, 5:8] = .01
    scaled = decoder(*args, short, opinions)
    for branch in BRANCHES:
        for field in ('tr', 'rot'):
            assert torch.equal(scaled[branch][field], full[branch][field]), (
                'decoder must not multiply the same evidence by inverse remaining time')
    assert torch.equal(scaled['torsion'], full['torsion'])
    objective = lambda output: sum(output[b][f].square().sum()
                                   for b in BRANCHES for f in ('tr', 'rot')) + output['torsion'].square().sum()
    gradients = [torch.autograd.grad(objective(output), h, retain_graph=True)[0]
                 for output in (full, scaled)]
    assert torch.equal(*gradients), 'late-time decoder must not amplify identical input derivatives'
    with torch.no_grad():
        decoder.bond[-1].bias.fill_(2.)
        for head in decoder.rigid.values():
            head[-1].weight.zero_()
            head[-1].bias.fill_(10.)
    large = decoder(*args, state, opinions)
    assert (large['torsion'] * torch.pi).abs().max() > torch.pi
    assert any((large[branch]['rot'] * torch.pi).norm() > torch.pi for branch in BRANCHES), (
        'rigid angular velocity must not impose a pi total-path budget')
    assert torch.allclose(.01 * scaled['torsion'], .2 * (.05 * full['torsion'])), (
        'executed increment must shrink with dt, not stay constant as dt decreases')
    short[:, -1], short[:, 5:8] = 0., 0.
    terminal = decoder(*args, short, opinions)
    assert all(torch.isfinite(terminal[b][f]).all() for b in BRANCHES for f in ('tr', 'rot', 'tor'))


def test_matcha_film_ablation_preserves_active_stages_and_random_stream():
    model = TernaryAssembly(node_dim=32, input_dim=16, expert_backend='matcha',
                            expert_scalar_dim=320, expert_vector_dim=3).double()
    receptor = {'features': torch.randn(1, 5, 16, dtype=torch.float64),
                'coords': torch.randn(1, 5, 3, dtype=torch.float64)}
    opposite = {'features': torch.randn(1, 7, 16, dtype=torch.float64),
                'coords': torch.randn(1, 7, 3, dtype=torch.float64)}
    _, graph, _, _ = assembly_fixture()
    ligand = {'features': graph['node_features'][graph['node_types'] == 2][None],
              'edge_index': graph['lig_edge_index'], 'edge_attr': graph['lig_edge_attr'].double(),
              'coords': torch.randn(1, 4, 3, dtype=torch.float64)}
    with torch.no_grad():
        for head in model.binary_film:
            head.bias.fill_(.25)
    rng = torch.get_rng_state().clone()
    for stage in range(3):
        baseline = model.binary_condition(receptor, opposite, ligand, stage_index=stage)
        assert all(value.count_nonzero() for value in baseline), 'fixture must exercise learned nonzero FiLM'
        for mode, stages in (('all', {0, 1, 2}), ('off', set()), ('early', {0}), ('late', {1, 2})):
            actual = model.binary_condition(receptor, opposite, ligand,
                                             stage_index=stage, ablation_mode=mode)
            assert all(torch.equal(value, expected if stage in stages else torch.zeros_like(expected))
                       for value, expected in zip(actual, baseline)), 'FiLM ablation changed an enabled stage or leaked a disabled stage'
    assert torch.equal(rng, torch.get_rng_state()), 'paired FiLM ablations must not consume random numbers'


def test_matcha_mutual_condition_preserves_prior_and_live_geometry():
    torch.manual_seed(19)
    model = TernaryAssembly(node_dim=32, input_dim=16, expert_backend='matcha',
                            expert_scalar_dim=320, expert_vector_dim=3).double()
    receptor = {'features': torch.randn(1, 5, 16, dtype=torch.float64),
                'coords': torch.randn(1, 5, 3, dtype=torch.float64, requires_grad=True)}
    opposite = {'features': torch.randn(1, 7, 16, dtype=torch.float64, requires_grad=True),
                'coords': torch.randn(1, 7, 3, dtype=torch.float64, requires_grad=True)}
    _, graph, _, _ = assembly_fixture()
    ligand = {'features': graph['node_features'][graph['node_types'] == 2][None],
              'edge_index': graph['lig_edge_index'], 'edge_attr': graph['lig_edge_attr'].double(),
              'coords': torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)}
    gamma, beta = model.binary_condition(receptor, opposite, ligand)
    assert gamma.shape == beta.shape == (1, 5, 320)
    assert gamma.count_nonzero() == beta.count_nonzero() == 0, (
        'zero-initialized FiLM must preserve the frozen pretrained prior')
    with torch.no_grad():
        model.binary_film[0].weight.normal_(std=.03)
        model.condition_geometry.weight.normal_(std=.03)
    gamma, beta = model.binary_condition(receptor, opposite, ligand)
    gradients = torch.autograd.grad((gamma + beta).square().sum(),
                                    (receptor['coords'], opposite['coords'], opposite['features'], ligand['coords']))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), (
        'mutual conditioning must retain both live coordinate paths and opposite-protein identity')
    displaced = model.binary_condition(receptor, opposite,
                                      {**ligand, 'coords': ligand['coords'] + 50.})
    assert any(not torch.allclose(a, b) for a, b in zip((gamma, beta), displaced)), (
        'opposite protein-ligand assembly changes must reach the conditional expert')
    altered = model.binary_condition(receptor, opposite,
                                    {**ligand, 'edge_attr': torch.zeros_like(ligand['edge_attr'])})
    assert any(not torch.allclose(a, b) for a, b in zip((gamma, beta), altered)), (
        'conditional interface evidence must retain ligand bond chemistry')
    rotation = so3_exp(torch.tensor([[.2, -.4, .7]], dtype=torch.float64))[0]
    moved = lambda data: {**data, 'coords': data['coords'] @ rotation.T + 30.}
    transformed = model.binary_condition(moved(receptor), moved(opposite), moved(ligand))
    assert all(torch.allclose(a, b, atol=1e-12) for a, b in zip((gamma, beta), transformed)), (
        'scalar FiLM conditions must be invariant to joint rigid transforms')
    identical = {**opposite, 'features': torch.zeros_like(opposite['features'])}
    distant = {**identical, 'coords': identical['coords'] + 100.}
    near, far = model.binary_condition(receptor, identical, ligand), model.binary_condition(receptor, distant, ligand)
    assert any(not torch.allclose(a, b) for a, b in zip(near, far)), (
        'identical opposite-protein identities must not erase absolute relative geometry')
    many = {key: value.repeat(1, 5, 1) for key, value in identical.items()}
    centered = model.binary_condition(receptor, many, ligand)
    translated = model.binary_condition({**receptor, 'coords': receptor['coords'] + 1e6},
                                        {**many, 'coords': many['coords'] + 1e6},
                                        {**ligand, 'coords': ligand['coords'] + 1e6})
    assert all(torch.allclose(a, b, atol=1e-9, rtol=0) for a, b in zip(centered, translated)), (
        'large joint translations must not corrupt pair distances through squared-norm cancellation')
    for stage in (1, 2):
        other = model.binary_condition(receptor, opposite, ligand, stage_index=stage)
        assert all(x.count_nonzero() == 0 for x in other), (
            'training one stage condition must not alter another pretrained stage')
    from tricoopdiff.models.expert import matcha_stage_clock
    for time, expected in ((1., 0), (.8, 0), (.5, 1), (.2, 2), (0., 2)):
        stage, local = matcha_stage_clock(torch.tensor([time], dtype=torch.float64))
        assert stage == expected and bool(((local >= 0) & (local <= 1)).all())


def test_matcha_coordinate_cut_keeps_film_trainable_and_rejects_clock_mismatch():
    from types import SimpleNamespace
    from tricoopdiff.models.assembly import TernaryAssemblyModel

    class PairExpert(torch.nn.Module):
        backend, node_scalar_dim, node_vector_dim = 'matcha', 2, 2
        model = SimpleNamespace(no_torsion=False)

        def read_pair(self, receptor, ligand, time, film):
            xyz = torch.cat((ligand['coords'][0], receptor['coords'][0]))
            condition = sum(x.sum() for x in film)
            tr = xyz.mean(0, keepdim=True) + condition
            return (tr, tr, tr.new_zeros(1, 1), xyz[:, :2] + condition,
                    xyz[:, None].expand(-1, 2, -1))

    _, graph, _, state = assembly_fixture()
    inputs = [{'coords': graph['node_coords'][graph['node_types'] == i][None].clone().requires_grad_(),
               'features': torch.zeros(1, int((graph['node_types'] == i).sum()), 16)} for i in range(3)]
    inputs[2].update(edge_index=graph['lig_edge_index'], edge_attr=graph['lig_edge_attr'],
                     edge_mask=graph['lig_edge_mask'], mask_rotate=graph['lig_mask_rotate'])
    for coordinate_grad in (False, True):
        fuse = TernaryAssembly(node_dim=32, input_dim=16, expert_scalar_dim=2,
                               expert_vector_dim=2, expert_backend='matcha')
        model = TernaryAssemblyModel(PairExpert(), fuse, expert_input_grad=coordinate_grad,
                                     checkpoint_ternary=True).train()
        output = model(*inputs, torch.ones(1), diffusion_state=state)
        objective = sum(output['binary_proposals'][branch]['tr'].sum() for branch in BRANCHES)
        gradients = torch.autograd.grad(objective,
            (fuse.binary_film[0].weight, *(data['coords'] for data in inputs)), allow_unused=True)
        assert gradients[0] is not None and gradients[0].abs().sum() > 0, (
            'turning off expert coordinate VJPs must not turn off trainable mutual FiLM')
        assert all((g is not None and torch.isfinite(g).all() and g.abs().sum() > 0)
                   if coordinate_grad else g is None for g in gradients[1:])
        for bad_time, bad_state in ((torch.tensor([.5]), state),
                                   (torch.ones(1), state.clone().index_fill(1, torch.tensor([5]), .1))):
            try:
                model(*inputs, bad_time, diffusion_state=bad_state)
            except ValueError:
                pass
            else:
                raise AssertionError('inconsistent expert clock or per-DOF dt must fail before reading experts')
        doubled = [{**data, 'coords': data['coords'].repeat(2, 1, 1),
                    'features': data['features'].repeat(2, 1, 1)} for data in inputs]
        try:
            model(*doubled, torch.ones(1), diffusion_state=state)
        except ValueError as error:
            assert 'one node-aligned example' in str(error)
        else:
            raise AssertionError('B>1 must not silently discard every example after the first')
        malformed = (torch.full((1, 3), float('inf')), torch.zeros(1, 3), torch.zeros(1, 1),
                     torch.zeros(9, 2), torch.zeros(9, 2, 3))
        try:
            model._checked_read(malformed, inputs[2], inputs[0], BRANCHES[0])
        except FloatingPointError:
            pass
        else:
            raise AssertionError('Matcha non-finite proposals must not be hidden by bounded vector features')


def test_matcha_reads_exact_executor_rotor_masks_including_empty_set():
    from rdkit import Chem
    from tricoopdiff.models.expert import FrozenMatcha

    mol = Chem.MolFromSmiles('CCCC')
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    selected = torch.tensor([False, False, True, False, False, False])
    mask = torch.tensor([[False, False, True, True]])
    data = dict(edge_index=edges, edge_mask=selected, mask_rotate=mask)
    bonds, masks = FrozenMatcha.canonical_torsions(mol, data, 'cpu')
    assert torch.equal(bonds, torch.tensor([[1, 2]])) and torch.equal(masks, mask)
    reverse = {**data, 'edge_mask': selected.roll(1), 'mask_rotate': ~mask}
    reversed_bonds, reversed_masks = FrozenMatcha.canonical_torsions(mol, reverse, 'cpu')
    assert torch.equal(reversed_bonds, bonds.flip(1)) and torch.equal(reversed_masks, ~masks), (
        'axis and rotating fragment must be read together, not converted by axis sign alone')
    empty = {**data, 'edge_mask': torch.zeros(6, dtype=torch.bool),
             'mask_rotate': torch.zeros(0, 4, dtype=torch.bool)}
    bonds, masks = FrozenMatcha.canonical_torsions(mol, empty, 'cpu')
    assert bonds.shape == (0, 2) and masks.shape == (0, 4)


# Module boundaries

ROOT = Path(__file__).resolve().parents[1]


def test_package_root_contains_only_production_orchestration():
    expected = {
        '__init__.py', 'train.py', 'sampler.py', 'sdpi_flow.py',
        'evaluate.py', 'runtime.py', 'diffdock_compat.py', 'distributed.py',
    }
    actual = {path.name for path in (ROOT / 'tricoopdiff').glob('*.py')}
    assert actual == expected, f'Unexpected root modules: {actual ^ expected}'
    for directory in ('inputs', 'geometry', 'initialization', 'metrics'):
        assert (ROOT / 'tricoopdiff' / directory / '__init__.py').is_file()


def test_shared_layer_imports_remain_compatible():
    from tricoopdiff.models import assembly
    from tricoopdiff.legacy import trifuse
    from tricoopdiff.models.expert import _set_time_embeddings
    from tricoopdiff.legacy.model import _set_time_embeddings as legacy_time

    assert trifuse.LigandFeatureEncoder is assembly.LigandFeatureEncoder
    for name in ('BondContextEncoder', 'TernaryGeometryEncoder',
                 'JudgeHead', 'StateHead', 'TorsionCoordinator', 'LegacyResidualMotion'):
        assert not hasattr(assembly, name), f'{name} must not remain in production'
    for name in ('BondContextEncoder', 'TernaryGeometryEncoder',
                 'JudgeHead', 'StateHead', 'TorsionCoordinator'):
        assert getattr(trifuse, name).__module__ == 'tricoopdiff.legacy.trifuse'
    assert not hasattr(assembly, 'TokenNegotiator')
    assert not hasattr(trifuse.TriFuse, 'transformer')
    assert legacy_time is _set_time_embeddings


def test_consensus_import_does_not_load_legacy_model():
    code = (
        'import sys; '
        f'sys.path.insert(0, {str(ROOT)!r}); '
        'import tricoopdiff.models.assembly; '
        'assert "tricoopdiff.legacy.model" not in sys.modules; '
        'assert "tricoopdiff.legacy.diffusion" not in sys.modules; '
        'assert "tricoopdiff.legacy.trifuse" not in sys.modules; '
        'assert not any(name.startswith(("tricoopdiff.experiments", '
        '"tricoopdiff.ranking", "tricoopdiff.conditioning")) for name in sys.modules)'
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def test_flow_import_is_independent_of_model_and_legacy_diffusion():
    code = (
        'import sys; '
        f'sys.path.insert(0, {str(ROOT)!r}); '
        'from tricoopdiff import SDPIFlow; '
        'assert SDPIFlow().inference_steps == 20; '
        'assert "tricoopdiff.models.assembly" not in sys.modules; '
        'assert "tricoopdiff.legacy.diffusion" not in sys.modules'
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def test_horizontal_rotor_decoder_is_invariant_to_independent_axis_reversal():
    from tricoopdiff.models.assembly import LegalPoseDecoder, TORSION_REPRESENTATION_CONTRACT

    torch.manual_seed(7)
    decoder = LegalPoseDecoder(16, 4,
        torsion_representation_contract=TORSION_REPRESENTATION_CONTRACT).double()
    h = torch.randn(12, 16, dtype=torch.float64, requires_grad=True)
    vectors = torch.randn(12, 4, 3, dtype=torch.float64, requires_grad=True)
    coords = torch.randn(12, 3, dtype=torch.float64)
    coords[6:] = torch.tensor([[0, 1, .3], [-1, 0, 0], [0, 0, 0],
                              [1, 0, 0], [2, 1, 1], [3, 1, 0]])
    coords.requires_grad_()
    types = torch.tensor([0] * 3 + [1] * 3 + [2] * 6)
    bonds = torch.tensor([[2, 3], [3, 4]])
    masks = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]], dtype=torch.bool)
    state = torch.zeros(1, 9, dtype=torch.float64)
    state[:, -1], state[:, 5:8] = .5, .05
    opinions = [torch.tensor([.2, -.1], dtype=torch.float64), torch.zeros(2, dtype=torch.float64)]
    original = decoder(h, vectors, coords, types, bonds, masks, state, opinions)
    reversed_bonds, reversed_masks = bonds.clone(), masks.clone()
    reversed_bonds[:, 0], reversed_masks[0] = bonds[:, 0].flip(0), ~masks[0]
    alternate = decoder(h, vectors, coords, types, reversed_bonds, reversed_masks, state, opinions)
    fields = lambda output: torch.cat([output[branch][field].flatten()
        for branch in BRANCHES for field in ('tr', 'rot')] + [output['torsion'].flatten()])
    assert torch.allclose(fields(original), fields(alternate), atol=2e-12, rtol=2e-10)
    parameters = [h, vectors, coords, *decoder.parameters()]
    first = torch.autograd.grad(fields(original).square().sum(), parameters, allow_unused=True)
    second = torch.autograd.grad(fields(alternate).square().sum(), parameters, allow_unused=True)
    for a, b in zip(first, second):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            assert torch.allclose(a, b, atol=2e-11, rtol=2e-8)
    assert first[2].abs().sum() > 0, 'equivalent actions must retain coordinate gradients'


def test_horizontal_rotor_model_contract_preserves_legacy_checkpoint_behavior():
    from tricoopdiff.models.assembly import production_model_config, TORSION_REPRESENTATION_CONTRACT

    _, graph, proposals, state = assembly_fixture()
    graph['expert_node_vectors'] = torch.randn(15, 4, 3)
    config = production_model_config(32, 16, expert_vector_dim=2)
    assert config['torsion_representation_contract'] == TORSION_REPRESENTATION_CONTRACT
    model = TernaryAssembly(**config).eval()
    alternate = dict(graph)
    alternate['lig_edge_mask'] = torch.tensor([False, False, False, True, False, False])
    alternate['lig_mask_rotate'] = ~graph['lig_mask_rotate']
    before = model(graph, proposals, state)['normalized_velocity']
    after = model(alternate, proposals, state)['normalized_velocity']
    for branch in BRANCHES:
        for field in ('tr', 'rot'):
            assert torch.allclose(before[branch][field], after[branch][field], atol=2e-6)
    assert torch.allclose(before['torsion'], after['torsion'], atol=2e-6)
    legacy_config = {key: value for key, value in config.items()
                     if key != 'torsion_representation_contract'}
    legacy = TernaryAssembly(**legacy_config)
    assert not legacy.controller.assembly.equivalent_rotors
    assert 'torsion_representation_contract' not in legacy.config
    legacy.load_state_dict(model.state_dict(), strict=True)
    assert set(legacy.state_dict()) == set(model.state_dict()), 'parameter schema must remain loadable'


def test_protein_intrinsic_geometry_availability_and_controller_gradients():
    from tricoopdiff.models.assembly import production_model_config, protein_local_geometry_features

    _, graph, proposals, state = assembly_fixture()
    graph['expert_node_vectors'] = torch.randn(15, 4, 3)
    model = TernaryAssembly(**production_model_config(32, 16, expert_vector_dim=2)).eval()
    geometry = torch.randn(15, 33)
    geometry[:, -1] = (graph['node_types'] != 2).float()
    geometry[graph['node_types'] == 2] = 0
    geometry.requires_grad_()
    output = model({**graph, 'protein_local_geometry': geometry}, proposals, state)['normalized_velocity']
    loss = output['torsion'].square().sum() + output['poi_ligand']['tr'].square().sum()
    gradient, projection_gradient = torch.autograd.grad(loss,
        (geometry, model.controller.protein_geometry.weight))
    assert gradient[:11, :32].abs().sum() > 0 and torch.isfinite(gradient).all()
    assert projection_gradient.abs().sum() > 0 and torch.isfinite(projection_gradient).all()
    assert gradient[11:].eq(0).all(), 'protein geometry may not bleed into ligand slots'
    permutation = torch.tensor([4, 2, 0, 3, 1, 10, 8, 6, 9, 7, 5, 14, 12, 11, 13])
    permuted = {**graph, 'protein_local_geometry': geometry[permutation]}
    for key in ('node_features', 'node_coords', 'node_types',
                'expert_node_features', 'expert_node_vectors'):
        permuted[key] = graph[key][permutation]
    ligand_order = permutation[11:] - 11
    permuted['lig_edge_index'] = torch.argsort(ligand_order)[graph['lig_edge_index']]
    permuted['lig_mask_rotate'] = graph['lig_mask_rotate'][:, ligand_order]
    reordered = model(permuted, proposals, state)['normalized_velocity']
    assert torch.allclose(output['torsion'], reordered['torsion'], atol=2e-6)
    for branch in BRANCHES:
        for field in ('tr', 'rot'):
            assert torch.allclose(output[branch][field], reordered[branch][field], atol=2e-6)

    receptor = {'coords': torch.zeros(1, 3, 3)}
    missing = protein_local_geometry_features(receptor)
    measured = protein_local_geometry_features({**receptor, 'local_geometry': torch.zeros(1, 3, 32)})
    assert torch.equal(missing[:, :32], measured[:, :32])
    assert missing[:, -1].eq(0).all() and measured[:, -1].eq(1).all()
    assert not torch.equal(model.controller.protein_geometry(missing),
                           model.controller.protein_geometry(measured))
    try:
        protein_local_geometry_features(receptor, required=True)
    except ValueError:
        pass
    else:
        raise AssertionError('required geometry contract must reject absent structure')
    malformed = {**receptor, 'local_geometry': torch.zeros(1, 4, 32)}
    try:
        protein_local_geometry_features(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError('structure slots must match exact receptor node order and count')
    old_config = {key: value for key, value in model.config.items()
                  if key not in ('protein_local_geometry_contract', 'torsion_representation_contract')}
    legacy = TernaryAssembly(**old_config)
    assert legacy.controller.protein_geometry is None
    assert legacy.condition_protein_geometry is None
    legacy.load_state_dict(legacy.state_dict(), strict=True)


def test_matcha_condition_reads_intrinsic_geometry_without_altering_sequence_inputs():
    from tricoopdiff.models.assembly import production_model_config

    torch.manual_seed(38)
    model = TernaryAssembly(**production_model_config(32, 16, expert_backend='matcha',
        expert_scalar_dim=8, expert_vector_dim=2)).double()
    receptor = {'features': torch.randn(1, 3, 16, dtype=torch.float64),
                'coords': torch.randn(1, 3, 3, dtype=torch.float64),
                'local_geometry': torch.randn(1, 3, 32, dtype=torch.float64, requires_grad=True)}
    opposite = {'features': torch.randn(1, 4, 16, dtype=torch.float64),
                'coords': torch.randn(1, 4, 3, dtype=torch.float64),
                'local_geometry': torch.randn(1, 4, 32, dtype=torch.float64, requires_grad=True)}
    _, graph, _, _ = assembly_fixture()
    ligand = {'coords': torch.randn(1, 4, 3, dtype=torch.float64),
              'features': graph['node_features'][11:][None].double(),
              'edge_index': graph['lig_edge_index'], 'edge_attr': graph['lig_edge_attr'].double()}
    original_sequence = receptor['features'].clone()
    gamma, beta = model.binary_condition(receptor, opposite, ligand)
    assert gamma.eq(0).all() and beta.eq(0).all(), 'fresh FiLM must preserve frozen prior'
    with torch.no_grad():
        model.binary_film[0].weight.normal_(std=.04)
    gamma, beta = model.binary_condition(receptor, opposite, ligand)
    grads = torch.autograd.grad((gamma + beta).square().sum(),
        (receptor['local_geometry'], opposite['local_geometry'], model.condition_protein_geometry.weight))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    moved = {**opposite, 'local_geometry': opposite['local_geometry'] + .3}
    changed = model.binary_condition(receptor, moved, ligand)
    assert not torch.allclose(beta, changed[1])
    assert torch.equal(original_sequence, receptor['features']), 'frozen sequence inputs must stay unchanged'


def test_chemical_path_cache_keeps_dynamic_geometry_and_graph_identity():
    import tricoopdiff.models.assembly as assembly

    relation = assembly.ComponentStructureEncoder.chemical_relations
    cache = assembly._CHEMICAL_PATH_CACHE
    cache.clear()
    coords = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], requires_grad=True)
    edges = torch.tensor([[0, 1], [1, 2]])
    attributes = torch.zeros(2, 4, requires_grad=True)
    first, connected = relation(coords, edges, attributes)
    assert len(cache) == 1
    moved = coords.detach().clone()
    moved[1, 1] = 1.
    moved.requires_grad_()
    second, second_connected = relation(moved, edges, attributes)
    assert len(cache) == 1, 'coordinate changes should reuse only static topology'
    assert torch.equal(first[..., 0], second[..., 0])
    assert not torch.equal(first[..., 5], second[..., 5]), 'bond lengths must be recomputed'
    second[..., 5].sum().backward()
    assert moved.grad is not None and moved.grad.abs().sum() > 0
    first[..., 1:5].sum().backward()
    assert torch.equal(attributes.grad, torch.ones_like(attributes))
    connected.zero_()
    assert second_connected.all(), 'callers must not mutate cached masks'
    changed = edges.clone()
    changed[:, 1] = torch.tensor([0, 2])
    third, _ = relation(coords, changed, attributes)
    assert len(cache) == 2 and first[0, 2, 0] != third[0, 2, 0]
    changed_attributes = attributes.detach().clone()
    changed_attributes[0, 0] = 1.
    fourth, _ = relation(coords, edges, changed_attributes)
    assert len(cache) == 3 and fourth[0, 1, 1] == 1.
    relation(coords.double(), edges, attributes.double())
    assert len(cache) == 4, 'dtype changes must not reuse cached tensor storage'
    permutation = torch.tensor([2, 1, 0])
    relation(coords[permutation], permutation[edges], attributes)
    assert len(cache) == 5, 'atom reindexing must key the reindexed graph'
    for value in range(assembly._CHEMICAL_PATH_CACHE_LIMIT + 2):
        relation(coords, edges, torch.full((2, 4), float(value)))
    assert len(cache) <= assembly._CHEMICAL_PATH_CACHE_LIMIT
    cache.clear()


def test_adaptive_checkpoint_headroom_guard_reuses_reserved_memory_and_respects_cap():
    from unittest.mock import patch
    from tricoopdiff.models.assembly import TernaryAssemblyModel

    policy = SimpleNamespace(checkpoint_ternary=True, checkpoint_min_nodes=600,
        checkpoint_memory_reserve_bytes=8 * 2**30,
        checkpoint_activation_bytes_per_node_squared=114688,
        checkpoint_memory_fraction=.75)
    decide = lambda count, device: TernaryAssemblyModel._should_checkpoint_ternary(policy, count, device)
    cuda, cpu = torch.device('cuda', 0), torch.device('cpu')
    with patch('torch.cuda.mem_get_info', return_value=(40 * 2**30, 80 * 2**30)) as info, \
         patch('torch.cuda.memory_allocated', return_value=10 * 2**30), \
         patch('torch.cuda.memory_reserved', return_value=12 * 2**30):
        assert not decide(255, cuda), 'ample memory should allow small-case activation retention'
        assert decide(600, cuda), 'large-case size protection must remain active'
        assert not decide(255, cpu)
        assert info.call_count == 1, 'large cases and CPU must not query CUDA memory'
    with patch('torch.cuda.mem_get_info', return_value=(2 * 2**30, 80 * 2**30)), \
         patch('torch.cuda.memory_allocated', return_value=10 * 2**30), \
         patch('torch.cuda.memory_reserved', return_value=12 * 2**30):
        assert decide(255, cuda), 'low physical free memory must force recomputation'
    with patch('torch.cuda.mem_get_info', return_value=(2 * 2**30, 80 * 2**30)), \
         patch('torch.cuda.memory_allocated', return_value=10 * 2**30), \
         patch('torch.cuda.memory_reserved', return_value=30 * 2**30):
        assert not decide(255, cuda), 'allocator cache can safely supply reusable bytes'
    with patch('torch.cuda.mem_get_info', return_value=(40 * 2**30, 80 * 2**30)), \
         patch('torch.cuda.memory_allocated', return_value=58 * 2**30), \
         patch('torch.cuda.memory_reserved', return_value=60 * 2**30):
        assert decide(255, cuda), 'the per-process cap must limit usable headroom'


def test_interaction_query_chunk_preserves_outputs_gradients_and_large_graph_guard():
    from copy import deepcopy
    from unittest.mock import patch
    from tricoopdiff.models.assembly import EquivariantInteraction

    torch.manual_seed(73)
    reference = EquivariantInteraction(8, 4).double().train()
    alternate = deepcopy(reference)
    alternate.query_chunk_size = 256
    count = 67
    types = torch.arange(count) % 3
    edges = torch.tensor([[2, 5], [5, 2]])
    attrs = torch.randn(2, 4, dtype=torch.double)
    inputs = [torch.randn(count, 8, dtype=torch.double),
              torch.randn(count, 4, 3, dtype=torch.double),
              torch.randn(count, 3, dtype=torch.double)]
    outcomes, gradients, calls = [], [], []
    for block in (reference, alternate):
        local = [x.clone().requires_grad_() for x in inputs]
        with patch.object(block, '_messages', wraps=block._messages) as messages:
            output = block(*local, types, edges, attrs)
            sum(value.square().mean() for value in output).backward()
            calls.append(messages.call_count)
        outcomes.append(output)
        gradients.append([x.grad for x in local] + [p.grad for p in block.parameters()])
    for left, right in zip(outcomes[0], outcomes[1]):
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-11)
    for left, right in zip(gradients[0], gradients[1]):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-10)
    assert calls == [4, 2], 'larger chunks must reduce both forward and recomputation dispatches'
    # Check row boundaries without allocating a large attention graph.
    alternate.eval()
    def messages(q, query_vectors, query_coords, keys, values, vectors, coords, types, bonds):
        return q.new_zeros(len(q), 3 * alternate.width + 3), query_vectors.new_zeros(len(q), 12, 3)
    with patch.object(alternate, '_messages', side_effect=messages) as mocked:
        alternate(torch.zeros(600, 8, dtype=torch.double), torch.zeros(600, 4, 3, dtype=torch.double),
                  torch.zeros(600, 3, dtype=torch.double), torch.arange(600) % 3, edges, attrs)
        assert [len(call.args[0]) for call in mocked.call_args_list] == [64] * 9 + [24]


def test_message_compile_is_runtime_only_and_requests_inductor_fullgraph():
    from unittest.mock import patch
    from tricoopdiff.models.assembly import EquivariantInteraction

    layer = EquivariantInteraction(8, 4)
    keys = tuple(layer.state_dict())
    compiled = lambda *args: layer._messages(*args)
    with patch('torch.compile', return_value=compiled) as compile_core, \
            patch('torch._C._dynamo.eval_frame._set_lru_cache') as cache_order:
        layer.set_compile_messages(True)
        cache_order.assert_called_once_with(False)
        assert layer._compiled_messages is compiled
        assert compile_core.call_args.kwargs == dict(backend='inductor', fullgraph=True, dynamic=True,
            recompile_limit=32, isolate_recompiles=True)
        assert tuple(layer.state_dict()) == keys, 'runtime fusion must not rename checkpoint weights'
    layer.set_compile_messages(False)
    assert layer._compiled_messages is None
    assert tuple(layer.state_dict()) == keys


def test_rollout_static_cache_reuses_identity_but_keeps_coordinates_and_gradients_live():
    from unittest.mock import patch
    import tricoopdiff.inputs.graphs as graphs
    import tricoopdiff.models.assembly as assembly

    graphs._TRIFUSE_STATIC_CACHE.clear()
    xyz = [torch.randn(n, 3, requires_grad=True) for n in (3, 2, 3)]
    features = [torch.randn(n, 4) for n in (3, 2, 3)]
    edges = torch.tensor([[0, 1], [1, 2]])
    attrs = torch.zeros(2, 4)
    def build(coords):
        return graphs.build_trifuse_graph(coords[0], features[0], coords[1], features[1],
                                          coords[2], features[2], lig_edge_index=edges)
    first = build(xyz)
    moved = [x + 1. for x in xyz]
    second = build(moved)
    assert first['node_features'] is second['node_features'], 'rollout should reuse frozen identity features'
    assert first['component_indices'] is second['component_indices']
    assert torch.equal(second['node_coords'], first['node_coords'] + 1.)
    second['node_coords'].sum().backward()
    assert all(torch.equal(x.grad, torch.ones_like(x)) for x in xyz)
    features[0].add_(1.)
    third = build(xyz)
    assert third['node_features'] is not second['node_features'], 'source mutation must invalidate static identity'
    third['node_types'].zero_()
    assert build(xyz)['node_types'].tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    features[0].requires_grad_()
    with_grad = build(xyz)
    with_grad['node_features'].sum().backward()
    assert torch.equal(features[0].grad, torch.ones_like(features[0])), 'learnable identity cannot be cached detached'
    assembly._CHEMICAL_SIGNATURE_CACHE.clear()
    with patch.object(assembly.hashlib, 'sha256', wraps=assembly.hashlib.sha256) as hashing:
        assembly.ComponentStructureEncoder.chemical_relations(xyz[2], edges, attrs)
        first_calls = hashing.call_count
        assembly.ComponentStructureEncoder.chemical_relations(moved[2], edges, attrs)
        assert first_calls == 2 and hashing.call_count == first_calls, 'stable topology must not be rehashed per step'
        edges[1, 1] = 0
        assembly.ComponentStructureEncoder.chemical_relations(xyz[2], edges, attrs)
        assert hashing.call_count > first_calls, 'in-place topology changes must be rehashed'
    graphs._TRIFUSE_STATIC_CACHE.clear()


def test_compiled_message_hints_fix_features_and_leave_lengths_dynamic():
    from unittest.mock import patch
    from tricoopdiff.models.assembly import EquivariantInteraction

    layer = EquivariantInteraction(8, 4).train()
    h = torch.randn(6, 8, requires_grad=True)
    vectors = torch.randn(6, 4, 3, requires_grad=True)
    coords = torch.randn(6, 3, requires_grad=True)
    types = torch.arange(6) % 3
    edges, attrs = torch.empty(2, 0, dtype=torch.long), torch.empty(0, 4)
    with patch('torch._dynamo.mark_static', wraps=torch._dynamo.mark_static) as hints:
        layer(h, vectors, coords, types, edges, attrs)
        assert not hints.called, 'eager execution must not receive compiler hints'
    # Exercise the production wrapper and real checkpoint recomputation without
    # compiling CPU kernels or downloading anything.
    layer._compiled_messages = layer._messages
    with patch('torch._dynamo.mark_static', wraps=torch._dynamo.mark_static) as hints:
        output = layer(h, vectors, coords, types, edges, attrs)
        sum(value.square().mean() for value in output).backward()
        assert hints.call_count == 16, 'all fixed dimensions must be re-marked during recomputation'
        for call in hints.call_args_list:
            tensor, dimensions = call.args
            assert tensor.dtype.is_floating_point
            assert 0 not in dimensions, 'node/query lengths must stay dynamic'
            if tensor.shape[-1] == 3:
                assert tensor._base is None and tensor.storage_offset() == 0
                assert tensor.is_contiguous(), 'compiled geometry must have a stable recompute layout'
            if tensor.ndim == 3 and tensor.shape[-1] == 4:
                assert dimensions == (2,), 'bond query/key lengths must stay dynamic'


def test_radial_two_operand_contraction_preserves_values_and_gradients():
    torch.manual_seed(73)
    inputs = [torch.randn(5, 7, 4, dtype=torch.double),
              torch.randn(7, 4, 3, dtype=torch.double),
              torch.randn(5, 7, 3, dtype=torch.double)]
    reference = [x.clone().requires_grad_() for x in inputs]
    candidate = [x.clone().requires_grad_() for x in inputs]
    old = torch.einsum('ijh,jhv,ijc->ihvc', *reference)
    weighted = candidate[0][..., None] * candidate[2][:, :, None, :]
    new = torch.einsum('ijhc,jhv->ihvc', weighted, candidate[1])
    torch.testing.assert_close(old, new, rtol=1e-12, atol=1e-12)
    old_grads = torch.autograd.grad(old.square().mean(), reference)
    new_grads = torch.autograd.grad(new.square().mean(), candidate)
    for left, right in zip(old_grads, new_grads):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)

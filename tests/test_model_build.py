"""
Smoke test: verify TriCoopDiff model can be built and run forward.

Run with: python tests/test_model_build.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def test_trifuse_build():
    """TriFuse builds with correct output shapes."""
    from tricoopdiff.trifuse import TriFuse

    trifuse = TriFuse(
        node_dim=128, ns=32, nv=4,
        num_transformer_layers=3,
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
    from tricoopdiff.diffusion import so3_exp
    from tricoopdiff.trifuse import TriFuse

    torch.manual_seed(3)
    trifuse = TriFuse(
        node_dim=32, input_dim=16, ns=8, nv=2,
        num_transformer_layers=1,
        num_conv_layers_per_expert=2,
        film_dim=4,
        dropout=0.0,
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
    from tricoopdiff.trifuse import TriFuse

    torch.manual_seed(7)
    trifuse = TriFuse(
        node_dim=32, input_dim=16, ns=8, nv=2,
        num_transformer_layers=1,
        num_conv_layers_per_expert=2,
        dropout=0.0,
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
    from tricoopdiff.trifuse import TriFuse

    torch.manual_seed(13)
    trifuse = TriFuse(
        node_dim=32, input_dim=16, ns=8, nv=2,
        num_transformer_layers=1,
        num_conv_layers_per_expert=2,
        dropout=0.0,
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


def test_film_zero_init():
    """FiLM is transparent at initialization (adaLN-Zero)."""
    from tricoopdiff.trifuse import FiLMHead

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
    from tricoopdiff.trifuse import TriFuse

    torch.manual_seed(19)
    trifuse = TriFuse(
        node_dim=32, input_dim=16, num_transformer_layers=1,
        num_conv_layers_per_expert=2, film_dim=8, dropout=0.0,
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
    from tricoopdiff.trifuse import ScoreControlHead

    head = ScoreControlHead(hidden_dim=16, max_delta=0.25)
    context = torch.randn(3, 16)
    assert torch.equal(head(context), torch.ones(3, 5))
    torch.nn.init.normal_(head.net[-1].weight)
    gains = head(context)
    assert bool(((gains >= 0.75) & (gains <= 1.25)).all())
    print("  [OK] Score controls are identity-initialized and bounded")


def test_torsion_coordinator_zero_init_and_per_bond_control():
    """Shared torsion starts neutral but can make independent bond decisions."""
    from tricoopdiff.trifuse import TorsionCoordinator

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
    from tricoopdiff.trifuse import TorsionCoordinator

    coordinator = TorsionCoordinator(context_dim=8, hidden_dim=4)
    try:
        coordinator(torch.randn(1, 8), torch.randn(3), torch.randn(3))
    except ValueError as error:
        assert "[B,K]" in str(error)
    else:
        raise AssertionError("flat native torsion score must be normalized first")


def test_torsion_coordinator_does_not_overflow_finite_evidence():
    """Finite expert scores must not overflow inside feature construction."""
    from tricoopdiff.trifuse import TorsionCoordinator

    coordinator = TorsionCoordinator(context_dim=8, hidden_dim=4).eval()
    context = torch.zeros(1, 8)
    score = torch.full((1, 3), 1e30)
    coordinated, weight, residual = coordinator(context, score, score)
    assert torch.isfinite(coordinated).all()
    assert torch.isfinite(weight).all()
    assert torch.isfinite(residual).all()


def test_bond_context_is_local_to_each_rotatable_bond():
    """Distinct endpoints/fragments must produce distinct bond contexts."""
    from tricoopdiff.trifuse import BondContextEncoder

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
    from tricoopdiff.trifuse import (
        TELEMETRY_PER_LAYER, TELEMETRY_TAIL, TriFuse,
    )

    torch.manual_seed(17)
    trifuse = TriFuse(
        node_dim=32, input_dim=16, num_transformer_layers=1,
        num_conv_layers_per_expert=2, dropout=0.0,
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
    from tricoopdiff.diffusion import InterfaceDiffusion, so3_score_norm

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
    from tricoopdiff.graphs import (
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
    from tricoopdiff.diffusion import TernaryInitializer
    from tricoopdiff.graphs import transform_ligand_to_poi_frame

    np.random.seed(7)
    torch.manual_seed(7)
    template = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 3.0],
    ])
    e3_center = torch.zeros(1, 3)
    initialized, body_rotation = TernaryInitializer().initialize_poi(
        e3_center, template, return_rotation=True, init_mode="coupled5a",
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
    from tricoopdiff.expert import FrozenDiffDock

    class DummyExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.5)
            self.ns, self.nv = 4, 1
            self.sh_irreps = None
            self.device = "cpu"

    frozen = FrozenDiffDock(DummyExpert(), num_conv_layers=1)
    parent = nn.Module()
    parent.add_module("expert", frozen)
    parent.train()
    assert not frozen.training
    assert not frozen.model.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())

    print("  [OK] Frozen expert remains in eval mode")


def test_score_sanitizer_handles_infinity():
    """Inference fallback must replace NaN and both infinities with finite values."""
    from tricoopdiff.sampler import _sanitize_score

    score = torch.tensor([[1.0, float("nan"), float("inf"), -float("inf")]])
    sanitized, changed = _sanitize_score(score)
    assert changed
    assert torch.isfinite(sanitized).all()

    print("  [OK] Score sanitizer handles infinities")


def test_reflect_pipeline():
    """TriReflect detects clashes and runs minimization."""
    from tricoopdiff.reflect import detect_clashes, tricoop_reflect

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


if __name__ == "__main__":
    print("=== TriCoopDiff Smoke Tests ===\n")

    test_trifuse_build()
    test_trifuse_reads_geometry_and_preserves_rigid_invariance()
    test_trifuse_links_chemical_features_to_local_contacts()
    test_trifuse_reads_sigma_and_solver_state()
    test_film_zero_init()
    test_trifuse_atom_film_can_select_ligand_regions()
    test_score_control_is_identity_bounded_and_trainable()
    test_torsion_coordinator_zero_init_and_per_bond_control()
    test_torsion_coordinator_rejects_flat_native_shape()
    test_torsion_coordinator_does_not_overflow_finite_evidence()
    test_bond_context_is_local_to_each_rotatable_bond()
    test_expert_telemetry_conditions_bond_context()
    test_diffusion_schedules()
    test_graph_construction()
    test_initializer_body_frame()
    test_frozen_expert_stays_eval()
    test_score_sanitizer_handles_infinity()
    test_reflect_pipeline()
    test_deepternary_dockq_alignment()

    print("\n=== All tests passed ===")

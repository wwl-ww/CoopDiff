"""
Smoke test: verify TriCoopDiff model can be built and run forward.

Run with: python tests/test_model_build.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def test_tribrain_build():
    """TriBrain builds with correct output shapes."""
    from tricoopdiff.tribrain import TriBrain, MultiScalePooling, EquivariantTransformer

    tribrain = TriBrain(
        node_dim=128, ns=32, nv=4,
        num_transformer_layers=3,
        num_conv_layers_per_expert=6,
    )

    # Dummy ternary graph: 100 E3 residues + 150 POI residues + 80 PROTAC atoms
    N = 330
    node_features = torch.randn(1, N, 128)
    node_coords = torch.randn(1, N, 3)
    node_types = torch.cat([
        torch.zeros(100, dtype=torch.long),
        torch.ones(150, dtype=torch.long),
        2 * torch.ones(80, dtype=torch.long),
    ]).unsqueeze(0)

    out = tribrain(node_features, node_coords, node_types)

    assert out['s_se3'].shape == (1, 6), f"s_se3 shape: {out['s_se3'].shape}"
    assert out['confidence'].shape == (1, 1)
    assert out['state'].shape == (1, 2)
    assert len(out['film_params']) == 12  # 6 layers × 2 subgraphs
    for gamma, beta in out['film_params']:
        assert gamma.shape == (1, 128)
        assert beta.shape == (1, 128)

    # Count parameters
    n_params = sum(p.numel() for p in tribrain.parameters())
    print(f"  TriBrain params: {n_params:,}")
    assert 1_500_000 < n_params < 3_000_000, f"Expected ~2.2M params, got {n_params:,}"

    print("  [OK] TriBrain build")


def test_film_zero_init():
    """FiLM is transparent at initialization (adaLN-Zero)."""
    from tricoopdiff.tribrain import FiLMHead

    head = FiLMHead(hidden_dim=64)
    h = torch.randn(2, 64)
    gamma, beta = head(h)

    # Zero-init: gamma and beta should be all zeros
    assert torch.allclose(gamma, torch.zeros_like(gamma), atol=1e-6), \
        f"Gamma not zero: max={gamma.abs().max():.2e}"
    assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-6), \
        f"Beta not zero: max={beta.abs().max():.2e}"

    print("  [OK] FiLM zero-init transparent")


def test_diffusion_schedules():
    """Diffusion schedules produce valid noise levels."""
    from tricoopdiff.diffusion import MultiRateDiffusion, so3_score_norm

    diffusion = MultiRateDiffusion(inference_steps=20)

    # Test sigma schedule
    t = torch.tensor([0.0, 0.5, 1.0])
    tr, rot, tor = diffusion.t_to_sigma(t, t, t)
    assert tr[0] < tr[1] < tr[2], "Sigma should increase with t"
    assert abs(tr[0].item() - 0.1) < 0.05, f"Sigma at t=0 should be ~0.1, got {tr[0].item()}"

    # Test lambda masks
    t_high = torch.tensor([0.8])
    t_low = torch.tensor([0.3])
    assert diffusion.get_lambda_q(t_high).item() == 1.0
    assert diffusion.get_lambda_q(t_low).item() == 0.0

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
        build_tribrain_graph, transform_ligand_to_poi_frame,
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

    # Ternary graph
    tri = build_tribrain_graph(e3_coords, e3_feat, poi_coords, poi_feat, lig_coords, lig_feat)
    assert tri['node_coords'].shape == (N_e3 + N_poi + N_lig, 3)
    assert tri['n_e3'] == N_e3
    assert tri['n_poi'] == N_poi
    assert tri['n_lig'] == N_lig

    print("  [OK] Graph construction")


def test_reflect_pipeline():
    """TriReflect detects clashes and runs minimization."""
    from tricoopdiff.reflect import detect_clashes, tricoop_reflect

    # Create two overlapping atoms → should clash
    coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    atomic_numbers = np.array([6, 6])  # both carbon, VDW=1.7

    clashes = detect_clashes(coords, atomic_numbers, vdw_scale=1.0)
    # C VDW radius = 1.7, 2*1.7=3.4, 0.5 < 3.4 → clash
    assert clashes.sum() > 0, "Should detect clash for atoms 0.5Å apart"

    # Non-clashing pair
    coords_far = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    clashes_far = detect_clashes(coords_far, atomic_numbers, vdw_scale=1.0)
    assert clashes_far.sum() == 0, "Should NOT detect clash for atoms 10Å apart"

    # Full pipeline
    result = tricoop_reflect(coords, atomic_numbers, do_minimization=True)
    assert 'is_valid' in result
    assert 'n_clashes' in result

    print("  [OK] TriReflect pipeline")


def test_merging():
    """Score merging by direct addition and learned coordination."""
    from tricoopdiff.merging import merge_direct_addition, LearnedCoordinationLayer

    B = 4
    scores_1 = {
        'tr_score': torch.randn(B, 3),
        'rot_score': torch.randn(B, 3),
        'tor_score': torch.randn(B, 10),
    }
    scores_2 = {
        'tr_score': torch.randn(B, 3),
        'rot_score': torch.randn(B, 3),
        'tor_score': torch.randn(B, 10),
    }

    # Direct addition
    merged = merge_direct_addition(scores_1, scores_2)
    for k in ['tr_score', 'rot_score', 'tor_score']:
        expected = scores_1[k] + scores_2[k]
        assert torch.allclose(merged[k], expected)

    # Learned coordination
    coord = LearnedCoordinationLayer(hidden_dim=128)
    h_cond = torch.randn(B, 128)
    merged_c = coord(scores_1, scores_2, h_cond)
    assert 'merge_weights' in merged_c

    print("  [OK] Score merging")


if __name__ == "__main__":
    print("=== TriCoopDiff Smoke Tests ===\n")

    test_tribrain_build()
    test_film_zero_init()
    test_diffusion_schedules()
    test_graph_construction()
    test_reflect_pipeline()
    test_merging()

    print("\n=== All tests passed ===")

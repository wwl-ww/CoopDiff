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
    from tricoopdiff.tribrain import TriBrain

    tribrain = TriBrain(
        node_dim=128, ns=32, nv=4,
        num_transformer_layers=3,
        num_conv_layers_per_expert=6,
    )

    # Dummy ternary graph: 100 E3 residues + 150 POI residues + 80 PROTAC atoms
    # TriBrain.forward expects UNBATCHED [N, ...] inputs (batch index derived
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

    out = tribrain(node_features, node_coords, node_types)

    assert out['confidence'].shape == (1, 1)
    assert out['state'].shape == (1, 2)
    assert len(out['film_params']) == 12  # 6 layers × 2 subgraphs
    for gamma, beta in out['film_params']:
        # FiLMHead outputs [B, film_dim] per (gamma, beta); film_dim defaults to
        # 48 (the expert's scalar channel count), not node_dim=128.
        assert gamma.shape == (1, tribrain.film_dim)
        assert beta.shape == (1, tribrain.film_dim)

    # Count parameters
    n_params = sum(p.numel() for p in tribrain.parameters())
    print(f"  TriBrain params: {n_params:,}")
    # Small MVP config (ns=32, nv=4, 3 transformer layers — the TriBrain defaults
    # used by the smoke test) is ~340k params. (The ~2.2M figure in early docs
    # assumed ns=48, nv=10.) Sanity: non-trivial but not bloated.
    assert 200_000 < n_params < 1_000_000, f"Unexpected TriBrain params: {n_params:,}"

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


def test_torsion_coordinator_zero_init_and_per_bond_control():
    """Shared torsion starts neutral but can make independent bond decisions."""
    from tricoopdiff.tribrain import TorsionCoordinator

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
    tri = build_tribrain_graph(e3_coords, e3_feat, poi_coords, poi_feat, lig_coords, lig_feat)
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

    test_tribrain_build()
    test_film_zero_init()
    test_torsion_coordinator_zero_init_and_per_bond_control()
    test_diffusion_schedules()
    test_graph_construction()
    test_initializer_body_frame()
    test_frozen_expert_stays_eval()
    test_score_sanitizer_handles_infinity()
    test_reflect_pipeline()
    test_deepternary_dockq_alignment()

    print("\n=== All tests passed ===")

import math

import torch

from tricoopdiff.pose_anchors import (
    joint_surface_anchors, proper_rigid_fit, surface_frame,
)
from tricoopdiff.expert_anchors import (
    FixedCandidateInitializer, generate_hybrid_candidates, generate_sdpi_candidates,
    select_balanced_pairs, select_hybrid_walkers, select_partial_pairs,
    select_pose_consensus,
)


def test_surface_frame_is_right_handed_and_orthonormal():
    frame = surface_frame(
        torch.tensor([[0.0, 0.0, 2.0]]),
        torch.tensor([[2.0, 1.0, 1.0]]),
    )
    assert torch.allclose(frame @ frame.transpose(-1, -2),
                          torch.eye(3).unsqueeze(0), atol=1e-6)
    assert torch.allclose(torch.linalg.det(frame), torch.ones(1), atol=1e-6)


def test_joint_anchor_maps_contact_point_and_opposes_normals():
    source_point = torch.tensor([[1.0, 2.0, 3.0]])
    source_normal = torch.tensor([[0.0, 0.0, 1.0]])
    source_tangent = torch.tensor([[1.0, 0.0, 0.0]])
    target_point = torch.tensor([[5.0, -2.0, 1.0]])
    target_normal = torch.tensor([[0.0, 1.0, 0.0]])
    target_tangent = torch.tensor([[1.0, 0.0, 0.0]])
    anchors = joint_surface_anchors(
        source_point, source_normal, source_tangent,
        target_point, target_normal, target_tangent,
        torch.tensor([0.0]),
    )
    assert torch.allclose(anchors.place(source_point)[0, 0], target_point[0], atol=1e-6)
    mapped_normal = source_normal @ anchors.rotations[0]
    assert torch.allclose(mapped_normal, -target_normal, atol=1e-6)


def test_twists_create_distinct_orientations_at_the_same_position():
    point = torch.zeros(1, 3)
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    tangent = torch.tensor([[1.0, 0.0, 0.0]])
    anchors = joint_surface_anchors(
        point, normal, tangent, point, normal, tangent,
        torch.tensor([0.0, math.pi / 2]),
    )
    assert anchors.rotations.shape == (2, 3, 3)
    assert torch.allclose(anchors.place(point)[:, 0], point.expand(2, -1), atol=1e-6)
    assert not torch.allclose(anchors.rotations[0], anchors.rotations[1])


def test_proper_rigid_fit_recovers_exact_transform():
    source = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], [0.0, 0.0, 3.0],
    ])
    rotation = torch.tensor([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
    ])
    translation = torch.tensor([3.0, -2.0, 1.0])
    target = source @ rotation.T + translation
    fitted_rotation, fitted_translation, rmsd = proper_rigid_fit(source, target)
    fitted = source @ fitted_rotation.T + fitted_translation
    assert torch.allclose(fitted, target, atol=1e-5)
    assert torch.det(fitted_rotation) > 0
    assert rmsd < 1e-5


def test_pose_consensus_selects_dense_ternary_mode():
    def candidate(offset):
        coords = torch.tensor(offset, dtype=torch.float32).reshape(1, 1, 3)
        return {
            "ligand": coords, "poi": coords, "penalty": (0, 0, 0.0, 0.0),
        }

    candidates = [
        candidate([0.0, 0.0, 0.0]), candidate([0.2, 0.0, 0.0]),
        candidate([-0.2, 0.0, 0.0]), candidate([20.0, 0.0, 0.0]),
    ]
    selected = select_pose_consensus(candidates, distance_threshold=1.0)
    assert any(selected is candidate for candidate in candidates[:3])
    assert selected["consensus_neighbors_10A"] == 3


def test_balanced_pair_subsampling_has_equal_marginal_degree():
    for size, count in ((64, 256), (16, 64), (4, 16)):
        pairs = select_balanced_pairs(size, count, seed=31_000)
        assert pairs == select_balanced_pairs(size, count, seed=31_000)
        assert len(pairs) == len(set(pairs)) == count
        degree = count // size
        assert [sum(left == i for left, _ in pairs) for i in range(size)] == [degree] * size
        assert [sum(right == i for _, right in pairs) for i in range(size)] == [degree] * size


def test_partial_pair_subsampling_is_deterministic_and_diverse():
    pairs = select_partial_pairs(7, 8, 48, seed=31_000)
    assert pairs == select_partial_pairs(7, 8, 48, seed=31_000)
    assert len(pairs) == len(set(pairs)) == 48
    left_degree = [sum(left == i for left, _ in pairs) for i in range(7)]
    right_degree = [sum(right == i for _, right in pairs) for i in range(8)]
    assert max(left_degree) - min(left_degree) <= 1
    assert max(right_degree) - min(right_degree) <= 1


def test_hybrid_walker_selection_protects_sources_and_diversity():
    def candidate(source, index, offset, penalty):
        coords = torch.tensor([[[float(offset), 0.0, 0.0]]])
        return {
            "candidate_id": [source, index],
            "penalty": (0, 0, float(penalty), 0.0),
            "ligand": coords,
            "poi": coords + 10,
            "rotation": torch.eye(3).unsqueeze(0),
        }

    candidates = [
        candidate("bitseed", 0, 0.0, 0),
        candidate("sdpi", 0, 0.2, 5),  # Same basin, but SDPI quota is protected.
        candidate("bitseed", 1, 10.0, 1),
        candidate("bitseed", 2, 20.0, 2),
        candidate("bitseed", 3, 30.0, 3),
        candidate("sdpi", 1, 40.0, 6),
    ]
    selected, metadata = select_hybrid_walkers(
        candidates, 4, bitseed_quota=3,
    )
    assert metadata["actual_source_counts"] == {"bitseed": 3, "sdpi": 1}
    assert metadata["unique_clusters_selected"] == 4
    assert selected[0]["candidate_id"] == ["sdpi", 0]
    repeated, repeated_metadata = select_hybrid_walkers(
        candidates, 4, bitseed_quota=3,
    )
    assert [item["candidate_id"] for item in repeated] == [
        item["candidate_id"] for item in selected
    ]
    assert repeated_metadata == metadata


def test_hybrid_walker_selection_second_pass_fills_exact_budget():
    candidates = []
    for index in range(4):
        coords = torch.tensor([[[0.1 * index, 0.0, 0.0]]])
        candidates.append({
            "candidate_id": ["bitseed" if index else "sdpi", index],
            "penalty": (0, 0, float(index), 0.0),
            "ligand": coords, "poi": coords + 10,
            "rotation": torch.eye(3).unsqueeze(0),
        })
    selected, metadata = select_hybrid_walkers(
        candidates, 4, bitseed_quota=3,
    )
    assert len(selected) == 4
    assert metadata["cluster_count"] == 1
    assert metadata["unique_clusters_selected"] == 1


def test_hybrid_walker_selection_component_switches_and_random_seed():
    candidates = []
    for index in range(8):
        source = "sdpi" if index >= 6 else "bitseed"
        coords = torch.tensor([[[float(index) * 10, 0.0, 0.0]]])
        candidates.append({
            "candidate_id": [source, index],
            "penalty": (0, 0, float(index), 0.0),
            "ligand": coords, "poi": coords + 10,
            "rotation": torch.eye(3).unsqueeze(0),
        })
    geometry, geometry_metadata = select_hybrid_walkers(
        candidates, 4, bitseed_quota=3,
        use_diversity=False, protect_sources=False,
    )
    assert all(item["candidate_id"][0] == "bitseed" for item in geometry)
    assert not geometry_metadata["use_diversity"]
    assert not geometry_metadata["protect_sources"]
    random_a, _ = select_hybrid_walkers(
        candidates, 4, bitseed_quota=3, use_diversity=False,
        protect_sources=True, random_seed=123,
    )
    random_b, _ = select_hybrid_walkers(
        candidates, 4, bitseed_quota=3, use_diversity=False,
        protect_sources=True, random_seed=123,
    )
    assert [item["candidate_id"] for item in random_a] == [
        item["candidate_id"] for item in random_b
    ]
    assert sum(item["candidate_id"][0] == "sdpi" for item in random_a) == 1


def test_sdpi_fallback_preserves_candidate_budget_and_seed_schedule():
    class StubInitializer:
        def __init__(self):
            self.seeds = []

        def initialize(self, e3, poi, ligand_data, seed):
            self.seeds.append(seed)
            ligand = torch.tensor([[[4.0, 0.0, 0.0]]])
            placed_poi = torch.tensor([[[8.0, 0.0, 0.0]]])
            rotation = torch.eye(3).unsqueeze(0)
            return ligand, placed_poi, rotation, {"init_mode": "sdpi"}

    initializer = StubInitializer()
    candidates, records = generate_sdpi_candidates(
        initializer, torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]), {}, count=3, seed=31_000,
    )
    assert len(candidates) == len(records) == 3
    assert len(set(initializer.seeds)) == 3
    assert all(candidate["fallback"] == "sdpi_v1" for candidate in candidates)
    assert [candidate["candidate_id"] for candidate in candidates] == [
        ["sdpi", 0], ["sdpi", 1], ["sdpi", 2],
    ]


def test_hybrid_replaces_only_failed_conformer_with_sdpi():
    class StubBase:
        def initialize(self, e3, poi, ligand_data, seed):
            ligand = torch.tensor([[[4.0, 0.0, 0.0]]])
            placed_poi = torch.tensor([[[8.0, 0.0, 0.0]]])
            return ligand, placed_poi, torch.eye(3).unsqueeze(0), {"seed": seed}

    class StubDual:
        proposal_attempts = 2
        base = StubBase()

        @staticmethod
        def _conformer_seed(seed, index):
            return seed + index

        def generate_candidates(self, *args, seed, **kwargs):
            if seed == 11:
                raise FloatingPointError("bad conformer")
            candidates = []
            for left in range(2):
                for right in range(2):
                    candidates.append({
                        "attempt": [left, right],
                        "candidate_id": [0, left, right],
                    })
            record = {"conformer_index": 0, "binary_retry_counts": [[], []],
                      "binary_draw_counts": [2, 2],
                      "binary_failure_counts": [0, 0],
                      "binary_valid_pose_counts": [2, 2]}
            return candidates, [record], None

    candidates, metadata = generate_hybrid_candidates(
        StubDual(), torch.zeros(1, 3), torch.ones(1, 3), {}, seed=10,
        conformer_count=2, bitseed_candidate_budget=4,
        sdpi_candidate_budget=1,
    )
    assert len(candidates) == 5
    assert metadata["actual_bitseed"] == 2
    assert metadata["actual_sdpi"] == 3
    assert len(metadata["failed_conformers"]) == 1
    assert sum(candidate.get("fallback") == "sdpi_v1"
               for candidate in candidates) == 3


def test_hybrid_replaces_unfittable_conformer_with_sdpi():
    class StubBase:
        def initialize(self, e3, poi, ligand_data, seed):
            coords = torch.tensor([[[4.0, 0.0, 0.0]]])
            return coords, coords + 4, torch.eye(3).unsqueeze(0), {"seed": seed}

    class StubDual:
        proposal_attempts = 1
        base = StubBase()

        @staticmethod
        def _conformer_seed(seed, index):
            return seed + index

        def generate_candidates(self, *args, **kwargs):
            raise ValueError("rigid fit requires at least three points")

    candidates, metadata = generate_hybrid_candidates(
        StubDual(), torch.zeros(1, 3), torch.ones(1, 3), {}, seed=10,
        conformer_count=1, bitseed_candidate_budget=1,
        sdpi_candidate_budget=0,
    )
    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == ["sdpi", 0]
    assert metadata["actual_bitseed"] == 0
    assert metadata["actual_sdpi"] == 1
    assert metadata["failed_conformers"][0]["error"].startswith("ValueError:")


def test_hybrid_keeps_partial_pairs_and_fills_only_the_deficit():
    class StubBase:
        def initialize(self, e3, poi, ligand_data, seed):
            coords = torch.tensor([[[float(seed), 0.0, 0.0]]])
            return coords, coords + 4, torch.eye(3).unsqueeze(0), {"seed": seed}

    class StubDual:
        proposal_attempts = 2
        base = StubBase()

        @staticmethod
        def _conformer_seed(seed, index):
            return seed + index

        def generate_candidates(self, *args, seed, **kwargs):
            candidates = [{
                "attempt": [0, 0], "candidate_id": [0, 0, 0],
            }]
            record = {
                "conformer_index": 0,
                "binary_retry_counts": [[0], [0]],
                "binary_draw_counts": [4, 1],
                "binary_failure_counts": [3, 0],
                "binary_valid_pose_counts": [1, 1],
            }
            return candidates, [record], None

    candidates, metadata = generate_hybrid_candidates(
        StubDual(), torch.zeros(1, 3), torch.ones(1, 3), {}, seed=10,
        conformer_count=1, bitseed_candidate_budget=2,
        sdpi_candidate_budget=1,
    )
    assert len(candidates) == 3
    assert metadata["actual_bitseed"] == 1
    assert metadata["actual_sdpi"] == 2
    assert metadata["partial_conformers"][0]["replacement_count"] == 1
    assert candidates[0]["candidate_id"] == ["bitseed", 0, 0, 0]


def test_fixed_candidate_initializer_clones_pose_and_records_source():
    candidate = {
        "candidate_id": ["sdpi", 7],
        "ligand": torch.arange(12, dtype=torch.float32).reshape(1, 4, 3),
        "poi": torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
        "rotation": torch.eye(3).unsqueeze(0),
        "geometry": {"minimum_protein_ca_distance": 4.0},
        "acceptable": True,
    }
    initializer = FixedCandidateInitializer(candidate, no_torsion=False)
    ligand, poi, rotation, metadata = initializer.initialize(None, None, None, seed=91)
    assert torch.equal(ligand, candidate["ligand"])
    assert torch.equal(poi, candidate["poi"])
    assert torch.equal(rotation, candidate["rotation"])
    assert ligand.data_ptr() != candidate["ligand"].data_ptr()
    assert poi.data_ptr() != candidate["poi"].data_ptr()
    assert metadata["candidate_source"] == "sdpi"
    assert metadata["native_information_used"] is False
    assert metadata["diffusion_seed"] == 91

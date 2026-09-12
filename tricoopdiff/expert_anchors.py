"""Native-free ternary anchors composed from two frozen DiffDock proposals."""

import random

import numpy as np
import torch

from .diffusion import InterfaceDiffusion
from .expert_time import _set_time_embeddings
from .graphs import build_subgraph_e3l, crop_receptor_to_ligand
from .initialization import TernaryInitializer, sample_binary_pose
from .pose_anchors import proper_rigid_fit
from .rigid import apply_rigid


EXPERT_ANCHOR_VERSION = "dual_diffdock_rigid_anchor_v1"
EXPERT_ANCHOR_CONSENSUS_VERSION = "dual_diffdock_rigid_anchor_consensus_v1"


def select_balanced_pairs(size, count, seed):
    """Select unique Cartesian pairs with equal use of every marginal pose."""
    if size < 1 or count < 1 or count > size ** 2 or count % size:
        raise ValueError("balanced pair count must be a positive multiple of size")
    degree = count // size
    rng = np.random.RandomState(int(seed) % (2**32))
    permutation = rng.permutation(size)
    shifts = rng.choice(size, degree, replace=False)
    pairs = [
        (left, int(permutation[(left + shift) % size]))
        for shift in shifts for left in range(size)
    ]
    if len(set(pairs)) != count:
        raise AssertionError("balanced pair selector produced duplicates")
    return pairs


def select_pose_consensus(candidates, distance_threshold=10.0):
    """Select the medoid of the densest native-free ternary-pose cluster."""
    if not candidates:
        raise ValueError("pose consensus requires at least one candidate")
    ligand = torch.stack([item["ligand"][0].reshape(-1) for item in candidates])
    poi = torch.stack([item["poi"][0].reshape(-1) for item in candidates])
    ligand_rmsd = torch.cdist(ligand, ligand) / np.sqrt(ligand.shape[1] / 3)
    poi_rmsd = torch.cdist(poi, poi) / np.sqrt(poi.shape[1] / 3)
    distances = torch.sqrt(ligand_rmsd.square() + poi_rmsd.square())
    neighbor_counts = (distances <= distance_threshold).sum(dim=1)
    nearest_count = min(5, len(candidates))
    nearest_means = distances.topk(
        nearest_count, dim=1, largest=False,
    ).values.mean(dim=1)
    best_index = min(
        range(len(candidates)),
        key=lambda index: (
            -int(neighbor_counts[index]), float(nearest_means[index]),
            candidates[index]["penalty"],
        ),
    )
    selected = candidates[best_index]
    selected["consensus_candidates"] = len(candidates)
    selected["consensus_neighbors_10A"] = int(neighbor_counts[best_index])
    selected["consensus_mean_5nn_rmsd"] = float(nearest_means[best_index])
    return selected


class DualExpertAnchorInitializer:
    """Compose two rigid-only binary docking endpoints around a shared conformer."""

    def __init__(self, expert, e3_data, poi_data, *, binary_steps=5,
                 proposal_attempts=4, conformer_policy="etkdgv3_random",
                 selection="geometry"):
        if binary_steps < 2:
            raise ValueError("binary_steps must be at least 2")
        if proposal_attempts < 1:
            raise ValueError("proposal_attempts must be positive")
        if selection not in {"geometry", "consensus"}:
            raise ValueError("anchor selection must be geometry or consensus")
        self.expert = expert
        self.e3_data = e3_data
        self.poi_data = poi_data
        self.binary_steps = int(binary_steps)
        self.proposal_attempts = int(proposal_attempts)
        self.selection = selection
        self.no_torsion = bool(getattr(expert.model, "no_torsion", False))
        self.conformer_policy = conformer_policy
        self.base = TernaryInitializer(
            no_torsion=self.no_torsion, conformer_policy=conformer_policy,
        )
        self.process = InterfaceDiffusion(inference_steps=self.binary_steps)

    @staticmethod
    def _conformer_seed(seed, conformer_index):
        """Derive stable independent conformer seeds without Python hashing."""
        return (int(seed) + int(conformer_index) * 100_000_007) % (2**31 - 1)

    @torch.no_grad()
    def _dock_rigid(self, receptor, ligand_data, position, seed):
        """Run native DiffDock rigid updates while preserving shared ligand torsions."""
        torch.manual_seed(int(seed))
        if position.is_cuda:
            torch.cuda.manual_seed_all(int(seed))
        schedule = self.process.get_inference_schedule(self.binary_steps)
        times = schedule["tr_schedule"]
        actual_steps = self.binary_steps - 1
        pos = position
        for index in range(actual_steps):
            time = torch.as_tensor(
                times[index], device=pos.device, dtype=pos.dtype,
            ).reshape(1)
            dt = (time - float(times[index + 1])
                  if index < actual_steps - 1 else time.clone())
            tr_sigma, _, _ = self.process.t_to_sigma(time, time, time)
            cutoff = (tr_sigma[0] * 3 + 20
                      if self.expert.model.dynamic_max_cross else None)
            crop = crop_receptor_to_ligand(
                receptor["coords"][0], receptor["features"][0], pos[0],
                rec_edge_index=receptor.get("edge_index"), cutoff=cutoff,
                min_residues=1, min_residues_with_edges=8,
                require_internal_edges=True,
            )
            graph = build_subgraph_e3l(
                crop["coords"], crop["features"], pos[0],
                ligand_data["features"][0], rec_edge_index=crop["edge_index"],
                lig_edge_index=ligand_data["edge_index"],
                lig_edge_attr=ligand_data["edge_attr"],
                lig_edge_mask=ligand_data["edge_mask"],
                lig_mask_rotate=ligand_data["mask_rotate"],
            )
            _set_time_embeddings(graph, time, self.expert.model)
            translation, rotation, torsion = self.expert(graph)[:3]
            if not all(torch.isfinite(value).all()
                       for value in (translation, rotation, torsion)):
                raise FloatingPointError("non-finite binary expert anchor proposal")
            zero_torsion = torch.zeros_like(torsion)
            pos, _ = self.process.reverse_step_protac(
                pos, translation, rotation, zero_torsion,
                time, time, time, dt, dt, dt,
                add_noise=index < actual_steps - 1,
            )
            if not torch.isfinite(pos).all():
                raise FloatingPointError("non-finite binary expert anchor coordinates")
        return pos

    @torch.no_grad()
    def generate_candidates(self, e3_coords, poi_coords, ligand_data, seed=None,
                            conformer_count=1, candidate_budget=None,
                            binary_retry_limit=4):
        """Generate one geometry-relaxed ternary proposal per binary-pose pair.

        Each conformer has independent ETKDG/torsion randomness. Both binary
        experts start from that same conformer, and pairs never cross conformers.
        Native coordinates and quality labels are not read here.
        """
        if seed is None:
            raise ValueError("expert-anchor generation requires an explicit seed")
        if conformer_count < 1:
            raise ValueError("conformer_count must be positive")
        if binary_retry_limit < 1:
            raise ValueError("binary_retry_limit must be positive")
        potential_count = int(conformer_count) * self.proposal_attempts ** 2
        if candidate_budget is None:
            candidate_budget = potential_count
        if (candidate_budget < 1 or candidate_budget > potential_count
                or candidate_budget % int(conformer_count)):
            raise ValueError(
                "candidate_budget must divide evenly across conformers and not "
                "exceed their Cartesian product"
            )
        pairs_per_conformer = candidate_budget // int(conformer_count)
        pair_indexes = select_balanced_pairs(
            self.proposal_attempts, pairs_per_conformer, seed,
        )
        candidates = []
        conformers = []
        first_broad = None
        for conformer_index in range(int(conformer_count)):
            conformer_seed = self._conformer_seed(seed, conformer_index)
            broad_ligand, broad_poi, broad_rotation, base_metadata = self.base.initialize(
                e3_coords, poi_coords, ligand_data, seed=conformer_seed,
            )
            if first_broad is None:
                first_broad = (broad_ligand, broad_poi, broad_rotation, base_metadata)
            dtype, device = broad_ligand.dtype, broad_ligand.device
            sides = base_metadata["sides"]
            rotation_e3 = torch.as_tensor(
                sides["e3"]["rotation"], dtype=dtype, device=device,
            )
            translation_e3 = torch.as_tensor(
                sides["e3"]["translation"], dtype=dtype, device=device,
            )
            rotation_poi = torch.as_tensor(
                sides["poi"]["rotation"], dtype=dtype, device=device,
            )
            translation_poi = torch.as_tensor(
                sides["poi"]["translation"], dtype=dtype, device=device,
            )
            shared = (broad_ligand[0] - translation_e3) @ rotation_e3
            initial_poi = apply_rigid(
                shared, rotation_poi, translation_poi,
            ).unsqueeze(0)
            banks = []
            retry_counts = []
            draw_counts = []
            for branch_index, (receptor_data, receptor_coords, initial_pose) in enumerate((
                    (self.e3_data, e3_coords, broad_ligand),
                    (self.poi_data, poi_coords, initial_poi))):
                bank = []
                branch_retries = []
                failures_since_success = 0
                max_draws = self.proposal_attempts * int(binary_retry_limit)
                draws = 0
                while len(bank) < self.proposal_attempts and draws < max_draws:
                    draw_index = draws
                    draws += 1
                    pose_seed = (conformer_seed + draw_index * 10_007
                                 + branch_index * 1_000_003)
                    if draw_index == 0:
                        pose = initial_pose
                    else:
                        random.seed(pose_seed)
                        np.random.seed(pose_seed % (2**32))
                        torch.manual_seed(pose_seed)
                        rotation0, translation0, _ = sample_binary_pose(
                            shared, receptor_coords,
                        )
                        pose = apply_rigid(
                            shared, rotation0, translation0,
                        ).unsqueeze(0)
                    try:
                        docked = self._dock_rigid(
                            receptor_data, ligand_data, pose,
                            pose_seed + 1_000_003,
                        )
                    except FloatingPointError:
                        failures_since_success += 1
                        continue
                    bank.append(docked)
                    branch_retries.append(failures_since_success)
                    failures_since_success = 0
                if len(bank) != self.proposal_attempts:
                    raise FloatingPointError(
                        f"collected {len(bank)}/{self.proposal_attempts} valid "
                        f"binary poses after {draws} deterministic draws"
                    )
                banks.append(bank)
                retry_counts.append(branch_retries)
                draw_counts.append(draws)

            conformers.append({
                "conformer_index": conformer_index,
                "seed": conformer_seed,
                "metadata": base_metadata["conformer"],
                "binary_retry_counts": retry_counts,
                "binary_draw_counts": draw_counts,
            })
            for e3_index, poi_index in pair_indexes:
                anchored_e3 = banks[0][e3_index]
                anchored_poi = banks[1][poi_index]
                rotation, translation, fit_rmsd = proper_rigid_fit(
                    anchored_poi[0], anchored_e3[0],
                )
                raw_poi = apply_rigid(poi_coords, rotation, translation)
                outward = raw_poi.mean(0) - e3_coords.mean(0)
                outward = outward / torch.linalg.vector_norm(outward).clamp_min(1e-8)
                e3_ligand_distances = torch.cdist(e3_coords, anchored_e3[0])
                variants = []
                for separation_step in range(13):
                    separation = 0.5 * separation_step
                    placed_poi = raw_poi + separation * outward
                    protein_distances = torch.cdist(e3_coords, placed_poi)
                    poi_ligand_distances = torch.cdist(placed_poi, anchored_e3[0])
                    geometry = {
                        "protein_ca_pairs_below_3A": int((protein_distances < 3.0).sum()),
                        "minimum_protein_ca_distance": float(protein_distances.min()),
                        "minimum_e3_ca_ligand_distance": float(e3_ligand_distances.min()),
                        "minimum_poi_ca_ligand_distance": float(poi_ligand_distances.min()),
                        "poi_outward_separation": separation,
                    }
                    acceptable = (
                        geometry["protein_ca_pairs_below_3A"] == 0
                        and 1.5 <= geometry["minimum_e3_ca_ligand_distance"] <= 8.0
                        and 1.5 <= geometry["minimum_poi_ca_ligand_distance"] <= 8.0
                        and 3.0 <= geometry["minimum_protein_ca_distance"] <= 15.0
                    )
                    penalty = (
                        0 if acceptable else 1,
                        geometry["protein_ca_pairs_below_3A"],
                        abs(geometry["minimum_e3_ca_ligand_distance"] - 3.5)
                        + abs(geometry["minimum_poi_ca_ligand_distance"] - 3.5)
                        + abs(geometry["minimum_protein_ca_distance"] - 5.0),
                        separation,
                    )
                    variants.append({
                        "penalty": penalty, "ligand": anchored_e3,
                        "poi": placed_poi.unsqueeze(0),
                        "rotation": rotation.unsqueeze(0),
                        "fit_rmsd": float(fit_rmsd), "geometry": geometry,
                        "attempt": [e3_index, poi_index],
                        "conformer_index": conformer_index,
                        "candidate_id": [conformer_index, e3_index, poi_index],
                        "acceptable": acceptable,
                    })
                candidates.append(min(variants, key=lambda item: item["penalty"]))
        expected = int(candidate_budget)
        if len(candidates) != expected:
            raise AssertionError(f"generated {len(candidates)} candidates, expected {expected}")
        return candidates, conformers, first_broad

    @torch.no_grad()
    def initialize(self, e3_coords, poi_coords, ligand_data, seed=None):
        if seed is None:
            raise ValueError("expert-anchor initialization requires an explicit seed")
        candidates, _, first_broad = self.generate_candidates(
            e3_coords, poi_coords, ligand_data, seed=seed, conformer_count=1,
        )
        broad_ligand, broad_poi, broad_rotation, base_metadata = first_broad
        if self.selection == "consensus":
            pair_best = {}
            for candidate in candidates:
                if not candidate["acceptable"]:
                    continue
                pair = tuple(candidate["attempt"])
                if (pair not in pair_best
                        or candidate["penalty"] < pair_best[pair]["penalty"]):
                    pair_best[pair] = candidate
            selected = (select_pose_consensus(list(pair_best.values()))
                        if pair_best else min(candidates, key=lambda item: item["penalty"]))
        else:
            selected = min(candidates, key=lambda item: item["penalty"])
        accepted = selected["acceptable"]
        if accepted:
            ligand, placed_poi, body_rotation = (
                selected["ligand"], selected["poi"], selected["rotation"],
            )
        else:
            ligand, placed_poi, body_rotation = broad_ligand, broad_poi, broad_rotation
        metadata = {
            **base_metadata,
            "init_mode": "expert_anchor",
            "version": EXPERT_ANCHOR_VERSION,
            "native_information_used": False,
            "binary_steps": self.binary_steps,
            "proposal_attempts": self.proposal_attempts,
            "proposal_pairs_considered": self.proposal_attempts ** 2,
            "anchor_selection": self.selection,
            "binary_torsion_updates": False,
            "selected_attempt": selected["attempt"],
            "shared_ligand_fit_rmsd": selected["fit_rmsd"],
            "anchor_accepted": accepted,
            "fallback": None if accepted else "broad_sdpi_clash_filter",
            "anchor_geometry": selected["geometry"],
        }
        for key in ("consensus_candidates", "consensus_neighbors_10A",
                    "consensus_mean_5nn_rmsd"):
            if key in selected:
                metadata[key] = selected[key]
        return ligand, placed_poi, body_rotation, metadata

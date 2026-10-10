"""Native-free ternary anchors composed from two frozen DiffDock proposals."""

import random

import numpy as np
import torch

from tricoopdiff.legacy.diffusion import InterfaceDiffusion
from tricoopdiff.models.expert import _set_time_embeddings
from tricoopdiff.inputs.graphs import build_subgraph_e3l, crop_receptor_to_ligand
from tricoopdiff.initialization import TernaryInitializer, sample_binary_pose
from tricoopdiff.geometry.rigid import apply_rigid, is_proper_rotation, proper_rigid_fit


EXPERT_ANCHOR_VERSION = "dual_diffdock_rigid_anchor_v1"
EXPERT_ANCHOR_CONSENSUS_VERSION = "dual_diffdock_rigid_anchor_consensus_v1"
FIXED_CANDIDATE_VERSION = "fixed_native_free_candidate_v1"


@torch.no_grad()
def coupled_anchor_perturbation(center, e3_coords, ligand_data, *, seed,
                                torsion_std=.15, translation_std=.5,
                                rotation_std=.05, contact_cutoff=8.):
    """Perturb a predicted assembly, preserving two disjoint local anchors.

    Contact atoms are selected solely from the supplied predicted center.
    One legal rotor separates the anchors: E3's anchor is restored and POI
    follows the other anchor. Without two identifiable anchors, explicitly
    use rigid-only error augmentation, not a claimed linker perturbation.
    Coordinates returned match FixedCandidateInitializer's batched contract.
    """
    from tricoopdiff.initialization import validate_ligand_graph
    from tricoopdiff.geometry.rigid import rotation_exp
    from tricoopdiff.sdpi_flow import _raw_torsions
    from rdkit import Chem

    if any(not np.isfinite(v) or v < 0 for v in
           (torsion_std, translation_std, rotation_std)) or not np.isfinite(contact_cutoff) or contact_cutoff <= 0:
        raise ValueError('perturbation scales must be finite nonnegative')
    ligand = center['ligand'].reshape(-1, 3).clone()
    poi = center['poi'].reshape(-1, 3).clone()
    body = center['rotation'].reshape(3, 3).clone()
    e3 = e3_coords.to(ligand).reshape(-1, 3)
    if not all(torch.isfinite(v).all() for v in (ligand, poi, e3, body)):
        raise ValueError('predicted anchor center must be finite')
    mol = ligand_data['mol']
    if isinstance(mol, bytes):
        mol = Chem.Mol(mol)
    bonds, masks = validate_ligand_graph(mol, ligand_data['edge_index'],
                                        ligand_data['edge_mask'], ligand_data['mask_rotate'])
    d_e3 = torch.cdist(ligand, e3).amin(1)
    d_poi = torch.cdist(ligand, poi).amin(1)
    options = []
    for index, mask in enumerate(masks):
        side = torch.as_tensor(mask, device=ligand.device)
        for e3_side in (side, ~side):
            a = torch.where(e3_side & (d_e3 < contact_cutoff))[0]
            b = torch.where(~e3_side & (d_poi < contact_cutoff))[0]
            if min(len(a), len(b)) < 3:
                continue
            # Reject collinear anchor sets rather than inventing an orientation.
            try:
                proper_rigid_fit(ligand[a], ligand[a])
                proper_rigid_fit(ligand[b], ligand[b])
            except ValueError:
                continue
            score = float(d_e3[a].mean() + d_poi[b].mean())
            options.append((score, index, a, b))
    rng = np.random.default_rng(seed)
    metadata = {'version': 'coupled_predicted_anchor_v1', 'seed': int(seed),
                'native_information_used': False, 'contact_cutoff_angstrom': contact_cutoff,
                'torsion_std_radians': torsion_std,
                'translation_std_angstrom': translation_std,
                'rotation_std_radians': rotation_std}
    if options:
        _, index, a, b = min(options, key=lambda item: item[0])
        angles = np.zeros(len(bonds))
        angles[index] = rng.normal(0., torsion_std)
        moved = ligand.new_tensor(_raw_torsions(ligand.cpu().numpy(),
                                  bonds.cpu().numpy(), masks, angles))
        ra, ta, error_a = proper_rigid_fit(moved[a], ligand[a])
        moved = apply_rigid(moved, ra, ta)
        rb, tb, error_b = proper_rigid_fit(ligand[b], moved[b])
        poi = apply_rigid(poi, rb, tb)
        body = rb @ body
        ligand = moved
        metadata.update(mode='two_anchor_coupled_torsion', rotor_index=index,
                        torsion_delta_radians=float(angles[index]),
                        e3_anchor_atoms=a.tolist(), poi_anchor_atoms=b.tolist(),
                        e3_anchor_fit_rmsd=float(error_a), poi_anchor_fit_rmsd=float(error_b))
    else:
        metadata.update(mode='rigid_only_no_separable_anchors')
    # Intentional small binding error; no promise that contacts remain exact.
    rotation = rotation_exp(ligand.new_tensor(rng.normal(0., rotation_std, 3)))
    pivot = poi.mean(0)
    offset = pivot - rotation @ pivot + ligand.new_tensor(rng.normal(0., translation_std, 3))
    poi = apply_rigid(poi, rotation, offset)
    body = rotation @ body
    if not is_proper_rotation(body):
        raise ValueError('coupled perturbation produced an invalid body frame')
    return {'ligand': ligand[None], 'poi': poi[None], 'rotation': body[None]}, metadata


class FixedCandidateInitializer:
    """Replay one already generated native-free proposal for refinement."""

    def __init__(self, candidate, *, no_torsion):
        if not _valid_candidate(candidate):
            raise ValueError("fixed initializer requires one finite proposal candidate")
        self.candidate = candidate
        self.no_torsion = bool(no_torsion)

    def initialize(self, e3_coords, poi_coords, ligand_data, seed=None):
        """Return cloned coordinates so reverse sampling cannot mutate the pool."""
        candidate = self.candidate
        source = _candidate_source(candidate)
        metadata = {
            "init_mode": "fixed_candidate",
            "version": FIXED_CANDIDATE_VERSION,
            "native_information_used": False,
            "candidate_id": candidate["candidate_id"],
            "candidate_source": source,
            "candidate_geometry": candidate.get("geometry"),
            "candidate_acceptable": bool(candidate.get("acceptable", False)),
            "diffusion_seed": seed,
        }
        return (
            candidate["ligand"].detach().clone(),
            candidate["poi"].detach().clone(),
            candidate["rotation"].detach().clone(),
            metadata,
        )


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


def select_partial_pairs(left_size, right_size, count, seed):
    """Select deterministic, marginally diverse pairs from a rectangular pool."""
    left_size, right_size, count = int(left_size), int(right_size), int(count)
    if (left_size < 1 or right_size < 1 or count < 1
            or count > left_size * right_size):
        raise ValueError("partial pair count must fit a non-empty Cartesian pool")
    if left_size == right_size and count % left_size == 0:
        return select_balanced_pairs(left_size, count, seed)
    rng = np.random.RandomState(int(seed) % (2**32))
    pairs = [(left, right) for left in range(left_size) for right in range(right_size)]
    random_rank = {
        pairs[index]: rank
        for rank, index in enumerate(rng.permutation(len(pairs)))
    }
    left_degree = [0] * left_size
    right_degree = [0] * right_size
    selected = []
    remaining = set(pairs)
    while len(selected) < count:
        pair = min(
            remaining,
            key=lambda item: (
                left_degree[item[0]] + right_degree[item[1]],
                max(left_degree[item[0]], right_degree[item[1]]),
                random_rank[item],
            ),
        )
        selected.append(pair)
        remaining.remove(pair)
        left_degree[pair[0]] += 1
        right_degree[pair[1]] += 1
    return selected


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


def _selection_key(candidate):
    return tuple(candidate["penalty"]), repr(candidate["candidate_id"])


def _candidate_source(candidate):
    candidate_id = candidate.get("candidate_id", ())
    if candidate_id and candidate_id[0] == "sdpi":
        return "sdpi"
    return "bitseed"


def _valid_candidate(candidate):
    tensors = (candidate.get("ligand"), candidate.get("poi"), candidate.get("rotation"))
    expected = ((1, None, 3), (1, None, 3), (1, 3, 3))
    for tensor, shape in zip(tensors, expected):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            return False
        if tensor.shape[0] != shape[0] or tensor.shape[-1] != shape[-1]:
            return False
        if shape[1] is not None and tensor.shape[1] != shape[1]:
            return False
        if not bool(torch.isfinite(tensor).all()):
            return False
    return (tensors[0].shape[1] > 0 and tensors[1].shape[1] > 0
            and is_proper_rotation(tensors[2]))


def _fixed_frame_rmsd(left, right):
    if left.shape != right.shape:
        return float("inf")
    return float(torch.sqrt(torch.mean(torch.sum((left - right) ** 2, dim=-1))))


@torch.no_grad()
def select_hybrid_walkers(candidates, walker_count, *, bitseed_quota=None,
                          ligand_rmsd_cutoff=2.0, poi_rmsd_cutoff=5.0,
                          use_diversity=True, protect_sources=True,
                          random_seed=None):
    """Select a deterministic, source-protected set of diverse initial walkers.

    Clusters are formed greedily in native-free geometry-penalty order in the
    fixed E3 frame. SDPI's smaller quota is allocated first so structured
    candidates cannot consume every cross-source basin. If either source or
    the independent-cluster count is insufficient, globally ranked unused
    clusters and finally duplicate-cluster candidates fill the exact budget.
    """
    walker_count = int(walker_count)
    if walker_count < 1:
        raise ValueError("walker_count must be positive")
    if bitseed_quota is None:
        if walker_count not in (32, 64):
            raise ValueError("provide bitseed_quota outside the 32/64 protocol")
        bitseed_quota = 3 * walker_count // 4
    quotas = {
        "bitseed": int(bitseed_quota),
        "sdpi": walker_count - int(bitseed_quota),
    }
    if min(quotas.values()) < 0:
        raise ValueError("source quota must fit walker_count")
    valid = [candidate for candidate in candidates if _valid_candidate(candidate)]
    if len(valid) < walker_count:
        raise ValueError(
            f"only {len(valid)} hard-valid candidates for {walker_count} walkers"
        )
    ranked = sorted(valid, key=_selection_key)
    if random_seed is not None:
        rng = np.random.RandomState(int(random_seed) % (2**32))
        ranked = [ranked[index] for index in rng.permutation(len(ranked))]
    clusters = []
    if use_diversity:
        ligand_flat = torch.stack([
            candidate["ligand"].reshape(-1) for candidate in ranked
        ])
        poi_flat = torch.stack([
            candidate["poi"].reshape(-1) for candidate in ranked
        ])
        ligand_rmsd = (
            torch.cdist(ligand_flat, ligand_flat)
            / np.sqrt(ligand_flat.shape[1] / 3)
        ).detach().cpu().numpy()
        poi_rmsd = (
            torch.cdist(poi_flat, poi_flat)
            / np.sqrt(poi_flat.shape[1] / 3)
        ).detach().cpu().numpy()
        cluster_indices = []
        for candidate_index, candidate in enumerate(ranked):
            assigned = False
            for cluster in cluster_indices:
                representative_index = cluster[0]
                if (ligand_rmsd[candidate_index, representative_index]
                        < ligand_rmsd_cutoff
                        and poi_rmsd[candidate_index, representative_index]
                        < poi_rmsd_cutoff):
                    cluster.append(candidate_index)
                    assigned = True
                    break
            if not assigned:
                cluster_indices.append([candidate_index])
        clusters = [
            [ranked[index] for index in cluster] for cluster in cluster_indices
        ]
    else:
        for candidate in ranked:
            clusters.append([candidate])

    selected = []
    selected_ids = set()
    used_clusters = set()

    def take(candidate, cluster_index):
        key = repr(candidate["candidate_id"])
        if key in selected_ids:
            return False
        selected.append(candidate)
        selected_ids.add(key)
        used_clusters.add(cluster_index)
        return True

    if protect_sources:
        # Protect the smaller broad branch before allocating structured walkers.
        for source in sorted(quotas, key=lambda item: (quotas[item], item)):
            options = []
            for cluster_index, cluster in enumerate(clusters):
                source_members = [
                    candidate for candidate in cluster
                    if _candidate_source(candidate) == source
                ]
                if source_members:
                    options.append((source_members[0], cluster_index))
            if random_seed is None:
                options.sort(key=lambda item: _selection_key(item[0]))
            needed = quotas[source]
            for candidate, cluster_index in options:
                if needed == 0:
                    break
                if cluster_index not in used_clusters and take(candidate, cluster_index):
                    needed -= 1

    # Prefer a new basin over rigidly enforcing a quota when one source lacks it.
    for cluster_index, cluster in enumerate(clusters):
        if len(selected) == walker_count:
            break
        if cluster_index not in used_clusters:
            take(cluster[0], cluster_index)
    # The second pass fills the exact walker budget with ranked basin duplicates.
    if len(selected) < walker_count:
        cluster_lookup = {
            repr(candidate["candidate_id"]): cluster_index
            for cluster_index, cluster in enumerate(clusters)
            for candidate in cluster
        }
        for candidate in ranked:
            if len(selected) == walker_count:
                break
            take(candidate, cluster_lookup[repr(candidate["candidate_id"])])
    if len(selected) != walker_count:
        raise AssertionError(f"selected {len(selected)} walkers, expected {walker_count}")
    source_counts = {
        source: sum(_candidate_source(candidate) == source for candidate in selected)
        for source in quotas
    }
    return selected, {
        "walker_count": walker_count,
        "requested_source_quotas": quotas,
        "actual_source_counts": source_counts,
        "hard_invalid_count": len(candidates) - len(valid),
        "cluster_count": len(clusters),
        "unique_clusters_selected": len(used_clusters),
        "ligand_rmsd_cutoff": float(ligand_rmsd_cutoff),
        "poi_rmsd_cutoff": float(poi_rmsd_cutoff),
        "use_diversity": bool(use_diversity),
        "protect_sources": bool(protect_sources),
        "random_seed": random_seed,
        "selected_candidate_ids": [candidate["candidate_id"] for candidate in selected],
    }


def _proposal_geometry(e3_coords, poi_coords, ligand, separation=0.0):
    """Return the native-free geometry gate used by anchor and SDPI proposals."""
    protein_distances = torch.cdist(e3_coords, poi_coords)
    e3_ligand_distances = torch.cdist(e3_coords, ligand)
    poi_ligand_distances = torch.cdist(poi_coords, ligand)
    geometry = {
        "protein_ca_pairs_below_3A": int((protein_distances < 3.0).sum()),
        "minimum_protein_ca_distance": float(protein_distances.min()),
        "minimum_e3_ca_ligand_distance": float(e3_ligand_distances.min()),
        "minimum_poi_ca_ligand_distance": float(poi_ligand_distances.min()),
        "poi_outward_separation": float(separation),
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
        float(separation),
    )
    return geometry, acceptable, penalty


@torch.no_grad()
def generate_sdpi_candidates(initializer, e3_coords, poi_coords, ligand_data,
                             count, seed, seed_stride=100_000_007):
    """Generate deterministic independent proposals from the default SDPI prior."""
    candidates, records = [], []
    for index in range(int(count)):
        proposal_seed = (int(seed) + index * int(seed_stride)) % (2**31 - 1)
        ligand, poi, body_rotation, metadata = initializer.initialize(
            e3_coords, poi_coords, ligand_data, seed=proposal_seed,
        )
        geometry, acceptable, penalty = _proposal_geometry(
            e3_coords, poi[0], ligand[0],
        )
        candidates.append({
            "penalty": penalty,
            "ligand": ligand,
            "poi": poi,
            "rotation": body_rotation,
            "fit_rmsd": None,
            "geometry": geometry,
            "attempt": ["sdpi", index],
            "conformer_index": None,
            "candidate_id": ["sdpi", index],
            "acceptable": acceptable,
            "fallback": "sdpi_v1",
        })
        records.append({
            "candidate_index": index,
            "seed": proposal_seed,
            "metadata": metadata,
        })
    return candidates, records


@torch.no_grad()
def generate_hybrid_candidates(initializer, e3_coords, poi_coords, ligand_data,
                               seed, conformer_count=4,
                               bitseed_candidate_budget=192,
                               sdpi_candidate_budget=64,
                               binary_retry_limit=2,
                               binary_consecutive_failure_limit=3):
    """Generate a fixed-budget BiTSeed/SDPI mixture with conformer fallback."""
    conformer_count = int(conformer_count)
    bitseed_candidate_budget = int(bitseed_candidate_budget)
    sdpi_candidate_budget = int(sdpi_candidate_budget)
    if conformer_count < 1 or bitseed_candidate_budget < 0 or sdpi_candidate_budget < 0:
        raise ValueError("hybrid counts must be non-negative and conformers positive")
    if bitseed_candidate_budget % conformer_count:
        raise ValueError("BiTSeed candidate budget must divide evenly across conformers")
    pairs_per_conformer = bitseed_candidate_budget // conformer_count
    size = int(initializer.proposal_attempts)
    if pairs_per_conformer > size ** 2:
        raise ValueError("per-conformer BiTSeed budget exceeds its Cartesian product")
    candidates, conformers, failures, partial = [], [], [], []
    for conformer_index in range(conformer_count):
        conformer_seed = initializer._conformer_seed(seed, conformer_index)
        try:
            generated, records, _ = initializer.generate_candidates(
                e3_coords, poi_coords, ligand_data, seed=conformer_seed,
                conformer_count=1, candidate_budget=size ** 2,
                binary_retry_limit=binary_retry_limit,
                binary_consecutive_failure_limit=binary_consecutive_failure_limit,
                allow_partial=True,
            )
        except (FloatingPointError, ValueError) as error:
            failures.append({
                "conformer_index": conformer_index,
                "seed": conformer_seed,
                "error": f"{type(error).__name__}: {error}",
                "replacement": "sdpi_v1",
                "replacement_count": pairs_per_conformer,
            })
            continue
        record = dict(records[0])
        record["conformer_index"] = conformer_index
        conformers.append(record)
        valid_left, valid_right = record["binary_valid_pose_counts"]
        retained_count = min(pairs_per_conformer, valid_left * valid_right)
        if retained_count:
            selected_pairs = set(select_partial_pairs(
                valid_left, valid_right, retained_count, seed,
            ))
        else:
            selected_pairs = set()
        for candidate in generated:
            if tuple(candidate["attempt"]) not in selected_pairs:
                continue
            candidate = dict(candidate)
            candidate["conformer_index"] = conformer_index
            candidate["candidate_id"] = [
                "bitseed", conformer_index, *candidate["attempt"],
            ]
            candidates.append(candidate)
        if retained_count < pairs_per_conformer:
            event = {
                "conformer_index": conformer_index,
                "seed": conformer_seed,
                "valid_pose_counts": [valid_left, valid_right],
                "available_pair_count": valid_left * valid_right,
                "retained_pair_count": retained_count,
                "replacement": "sdpi_v1",
                "replacement_count": pairs_per_conformer - retained_count,
            }
            partial.append(event)
            if retained_count == 0:
                failures.append(event)
    missing_bitseed = bitseed_candidate_budget - len(candidates)
    fallback_count = sdpi_candidate_budget + missing_bitseed
    sdpi, sdpi_records = generate_sdpi_candidates(
        initializer.base, e3_coords, poi_coords, ligand_data,
        fallback_count, int(seed) + 1_000_000, seed_stride=1,
    )
    candidates.extend(sdpi)
    expected = bitseed_candidate_budget + sdpi_candidate_budget
    if len(candidates) != expected:
        raise AssertionError(f"generated {len(candidates)} hybrid candidates, expected {expected}")
    metadata = {
        "requested_bitseed": bitseed_candidate_budget,
        "actual_bitseed": bitseed_candidate_budget - missing_bitseed,
        "requested_sdpi": sdpi_candidate_budget,
        "actual_sdpi": fallback_count,
        "successful_conformers": conformers,
        "failed_conformers": failures,
        "partial_conformers": partial,
        "fallback_events": partial + [
            event for event in failures if event not in partial
        ],
        "sdpi_candidates": sdpi_records,
    }
    return candidates, metadata


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
                            binary_retry_limit=2,
                            binary_consecutive_failure_limit=3,
                            allow_partial=False):
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
        if binary_consecutive_failure_limit < 1:
            raise ValueError("binary_consecutive_failure_limit must be positive")
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
                while (len(bank) < self.proposal_attempts and draws < max_draws
                       and failures_since_success < binary_consecutive_failure_limit):
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
                if len(bank) != self.proposal_attempts and not allow_partial:
                    raise FloatingPointError(
                        f"collected {len(bank)}/{self.proposal_attempts} valid "
                        f"binary poses after {draws} deterministic draws"
                    )
                banks.append(bank)
                retry_counts.append(branch_retries)
                draw_counts.append(draws)

            valid_pose_counts = [len(bank) for bank in banks]
            failure_counts = [
                draws - valid for draws, valid in zip(draw_counts, valid_pose_counts)
            ]
            stop_reasons = []
            for valid, draws, retries in zip(
                    valid_pose_counts, draw_counts, retry_counts):
                trailing_failures = draws - valid - sum(retries)
                if valid == self.proposal_attempts:
                    stop_reasons.append("complete")
                elif trailing_failures >= binary_consecutive_failure_limit:
                    stop_reasons.append("consecutive_nan_limit")
                else:
                    stop_reasons.append("draw_limit")
            conformers.append({
                "conformer_index": conformer_index,
                "seed": conformer_seed,
                "metadata": base_metadata["conformer"],
                "binary_retry_counts": retry_counts,
                "binary_draw_counts": draw_counts,
                "binary_failure_counts": failure_counts,
                "binary_valid_pose_counts": valid_pose_counts,
                "binary_stop_reasons": stop_reasons,
            })
            if allow_partial:
                pair_indexes = [
                    (left, right)
                    for left in range(valid_pose_counts[0])
                    for right in range(valid_pose_counts[1])
                ]
            else:
                pair_indexes = select_balanced_pairs(
                    self.proposal_attempts, pairs_per_conformer, seed,
                )
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
        expected = (len(candidates) if allow_partial else int(candidate_budget))
        if len(candidates) != expected:
            raise AssertionError(f"generated {len(candidates)} candidates, expected {expected}")
        return candidates, conformers, first_broad

    @torch.no_grad()
    def initialize(self, e3_coords, poi_coords, ligand_data, seed=None):
        if seed is None:
            raise ValueError("expert-anchor initialization requires an explicit seed")
        try:
            candidates, _, first_broad = self.generate_candidates(
                e3_coords, poi_coords, ligand_data, seed=seed, conformer_count=1,
            )
        except FloatingPointError as error:
            ligand, placed_poi, body_rotation, base_metadata = self.base.initialize(
                e3_coords, poi_coords, ligand_data, seed=seed,
            )
            return ligand, placed_poi, body_rotation, {
                **base_metadata,
                "init_mode": "expert_anchor",
                "version": EXPERT_ANCHOR_VERSION,
                "native_information_used": False,
                "binary_steps": self.binary_steps,
                "proposal_attempts": self.proposal_attempts,
                "anchor_accepted": False,
                "fallback": "sdpi_v1_binary_expert_failure",
                "fallback_reason": str(error),
            }
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

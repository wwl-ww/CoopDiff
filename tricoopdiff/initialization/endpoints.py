"""SDPI/BiTSeed training endpoints: projection, cache contracts and online generation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from tricoopdiff.initialization.candidates import (
    DualExpertAnchorInitializer, generate_hybrid_candidates,
    generate_sdpi_candidates,
)
from tricoopdiff.models.expert import expert_base_identity, expert_runtime_identity
from tricoopdiff.sdpi_flow import SDPIFlow, _raw_torsions
from tricoopdiff.runtime import atomic_torch_save, preserve_rng


def _coordinates(value, name):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if (array.ndim != 2 or array.shape[1] != 3 or not len(array)
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be finite [N, 3] or [1, N, 3]")
    return array


def _undirected_bonds(edge_index, atom_count):
    edges = np.asarray(edge_index)
    if edges.ndim != 2:
        raise ValueError("edge_index must be two-dimensional")
    if edges.shape[0] == 2:
        edges = edges.T
    if edges.shape[1:] != (2,):
        raise ValueError("edge_index must have shape [2, E] or [E, 2]")
    bonds = sorted({tuple(sorted(map(int, edge))) for edge in edges})
    if ((not bonds and atom_count > 1)
            or any(u == v or u < 0 or v >= atom_count for u, v in bonds)):
        raise ValueError("edge_index contains an invalid or empty bond map")
    return bonds


def endpoint_internal_geometry(reference, endpoint, edge_index, atol=1e-4):
    """Compare invariants preserved by the current torsion-plus-rigid path."""
    reference = _coordinates(reference, "reference")
    endpoint = _coordinates(endpoint, "endpoint")
    if reference.shape != endpoint.shape:
        raise ValueError("reference and endpoint atom coordinates must match")
    if not isinstance(atol, (int, float)) or not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")
    bonds = _undirected_bonds(edge_index, len(reference))
    neighbors = [set() for _ in range(len(reference))]
    for left, right in bonds:
        neighbors[left].add(right)
        neighbors[right].add(left)
    one_three = sorted({
        (min(left, right), center, max(left, right))
        for center, adjacent in enumerate(neighbors)
        for left in adjacent for right in adjacent if left < right
    })

    def distance_deltas(pairs):
        if not pairs:
            return np.zeros(0, dtype=np.float64)
        indexes = np.asarray([(item[0], item[-1]) for item in pairs])
        ref = np.linalg.norm(reference[indexes[:, 0]] - reference[indexes[:, 1]], axis=1)
        end = np.linalg.norm(endpoint[indexes[:, 0]] - endpoint[indexes[:, 1]], axis=1)
        return end - ref

    bond_delta = distance_deltas(bonds)
    one_three_delta = distance_deltas(one_three)

    def summary(delta):
        return {
            "count": int(len(delta)),
            "rms_delta_angstrom": float(np.sqrt(np.mean(delta ** 2))) if len(delta) else 0.0,
            "max_abs_delta_angstrom": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        }

    bond_summary = summary(bond_delta)
    angle_summary = summary(one_three_delta)
    maximum = max(bond_summary["max_abs_delta_angstrom"],
                  angle_summary["max_abs_delta_angstrom"])
    return {
        "contract": "sdpi_torsion_rigid_necessary_invariants_v1",
        "bond_lengths": bond_summary,
        "bonded_one_three_distances": angle_summary,
        "atol_angstrom": float(atol),
        "necessary_condition_passed": bool(maximum <= atol),
        "interpretation": (
            "failure proves non-representability; pass does not prove representability"
        ),
    }


ENDPOINT_PROJECTION_VERSION = "torsion_rigid_endpoint_projection_v1"
PROJECTION_SOLVER_VERSION = "explicit_rotation_torsion_least_squares_v2"


def _mask_array(mask_rotate, bond_count, atom_count):
    if isinstance(mask_rotate, (list, tuple)):
        if len(mask_rotate) != 1:
            raise ValueError("endpoint projection requires one ligand graph")
        mask_rotate = mask_rotate[0]
    if torch.is_tensor(mask_rotate):
        mask_rotate = mask_rotate.detach().cpu().numpy()
    masks = np.asarray(mask_rotate, dtype=bool)
    if masks.shape != (bond_count, atom_count):
        raise ValueError(
            f"mask_rotate must have shape [{bond_count},{atom_count}], got {masks.shape}"
        )
    return masks


def _fit_numpy_rigid(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3
            or len(source) == 0 or not np.isfinite(source).all()
            or not np.isfinite(target).all()):
        raise ValueError("rigid-fit coordinates must have matching [N,3] shapes")
    centered_source = source - source.mean(0)
    centered_target = target - target.mean(0)
    underdetermined = (len(source) < 3
                       or np.linalg.matrix_rank(centered_source, tol=1e-10) < 2)
    # Kabsch preserves atom correspondence even when the rotation is non-unique.
    # Independently selecting each cloud's farthest atom can reverse a linear
    # molecule merely because floating-point noise changes an argmax tie.
    left, _, right = np.linalg.svd(centered_source.T @ centered_target)
    correction = np.array([1., 1., np.linalg.det(left @ right)])
    rotation = ((left * correction) @ right).T
    translation = target.mean(0) - rotation @ source.mean(0)
    fitted = source @ rotation.T + translation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    return rotation, translation, rmsd, underdetermined


def project_endpoint_coupling(ligand_data, native_poi, endpoint_ligand,
                              endpoint_poi, *, no_torsion=False,
                              max_nfev=400, exact_atol=1e-3):
    """Project an external endpoint into the SDPI torsion-plus-SE(3) chart.

    The returned coupling can be passed directly to :meth:`SDPIFlow.path`.
    The projected coordinates and residuals are explicit: external conformer
    strain is never silently described as an exact endpoint round-trip.
    """
    native_ligand = _coordinates(ligand_data["coords"], "native ligand")
    native_poi = _coordinates(native_poi, "native POI")
    endpoint_ligand = _coordinates(endpoint_ligand, "endpoint ligand")
    endpoint_poi = _coordinates(endpoint_poi, "endpoint POI")
    if native_ligand.shape != endpoint_ligand.shape:
        raise ValueError("native and endpoint ligand atom coordinates must match")
    if native_poi.shape != endpoint_poi.shape:
        raise ValueError("native and endpoint POI coordinates must match")
    edge_index = np.asarray(
        ligand_data["edge_index"].detach().cpu()
        if torch.is_tensor(ligand_data["edge_index"])
        else ligand_data["edge_index"]
    )
    edge_mask = np.asarray(
        ligand_data["edge_mask"].detach().cpu()
        if torch.is_tensor(ligand_data["edge_mask"])
        else ligand_data["edge_mask"], dtype=bool,
    )
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("ligand edge_index must have shape [2,E]")
    if edge_mask.shape != (edge_index.shape[1],):
        raise ValueError("ligand edge_mask must have shape [E]")
    bonds = edge_index.T[edge_mask].astype(np.int64, copy=False)
    masks = _mask_array(ligand_data["mask_rotate"], len(bonds), len(native_ligand))

    target_centered = endpoint_ligand - endpoint_ligand.mean(0)

    def residual(parameters):
        # Optimize rotation explicitly. Re-fitting Kabsch inside every residual
        # makes finite-difference torsion derivatives depend on an SVD chart
        # that is non-unique near collinear/symmetric intermediate conformers.
        shape = _raw_torsions(native_ligand, bonds, masks, parameters[:-3])
        rotation = Rotation.from_rotvec(parameters[-3:]).as_matrix()
        return ((shape - shape.mean(0)) @ rotation.T - target_centered).reshape(-1)

    if no_torsion or not len(bonds):
        angles = np.zeros(len(bonds), dtype=np.float64)
        optimization = {"success": True, "status": 0, "nfev": 0}
    else:
        # Rotations live on circles, not a [-pi, pi] box. Start from native
        # atom-corresponded local phases so an exactly reachable conformer is
        # not replaced by a different local minimum of the rigid-fit residual.
        neighbors = [set() for _ in native_ligand]
        for u, v in _undirected_bonds(edge_index, len(native_ligand)):
            neighbors[u].add(v)
            neighbors[v].add(u)
        initial_angles = np.zeros(len(bonds), dtype=np.float64)
        for index, (u, v) in enumerate(bonds):
            left, right = sorted(neighbors[u] - {v}), sorted(neighbors[v] - {u})
            if not left or not right:
                continue
            phases, qualities = [], []
            for coordinates in (native_ligand, endpoint_ligand):
                axis = coordinates[v] - coordinates[u]
                length = np.linalg.norm(axis)
                if length <= 1e-10:
                    break
                axis /= length
                a, b = coordinates[left] - coordinates[u], coordinates[right] - coordinates[v]
                a -= (a @ axis)[:, None] * axis
                b -= (b @ axis)[:, None] * axis
                norms_a, norms_b = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
                quality = norms_a[:, None] * norms_b[None, :]
                a /= np.maximum(norms_a[:, None], 1e-10)
                b /= np.maximum(norms_b[:, None], 1e-10)
                phases.append(np.arctan2(np.cross(a[:, None], b[None]) @ axis, a @ b.T))
                qualities.append(quality)
            if len(phases) == 2:
                quality = qualities[0] * qualities[1]
                selected = np.unravel_index(quality.argmax(), quality.shape)
                if quality[selected] > 1e-20:
                    initial_angles[index] = (phases[0][selected] - phases[1][selected] + np.pi) % (2 * np.pi) - np.pi
        def start(angles):
            shape = _raw_torsions(native_ligand, bonds, masks, angles)
            rotation, _, _, _ = _fit_numpy_rigid(shape, endpoint_ligand)
            return np.concatenate((angles, Rotation.from_matrix(rotation).as_rotvec()))

        attempts = []
        fitted = None
        for initial in (initial_angles, np.zeros_like(initial_angles)):
            candidate = least_squares(
                residual, start(initial), max_nfev=int(max_nfev), jac='3-point',
                ftol=1e-8, xtol=1e-10, gtol=1e-8)
            attempts.append(candidate)
            # Budget exhaustion is not accepted. Continue from the actual best
            # state, rather than dropping the sample or inventing a target.
            if not candidate.success and candidate.status == 0:
                candidate = least_squares(
                    residual, candidate.x, max_nfev=3 * int(max_nfev), jac='3-point',
                    ftol=1e-8, xtol=1e-10, gtol=1e-8)
                attempts.append(candidate)
            if candidate.success and np.isfinite(candidate.fun).all():
                if fitted is None or candidate.cost < fitted.cost:
                    fitted = candidate
                if np.sqrt(np.mean(candidate.fun ** 2)) < 1e-8:
                    break
        if fitted is None:
            fitted = min(attempts, key=lambda result: result.cost)
        angles = (fitted.x[:-3] + np.pi) % (2 * np.pi) - np.pi
        optimization = {
            "success": bool(fitted.success), "status": int(fitted.status),
            "nfev": sum(int(result.nfev) for result in attempts),
            "solver": PROJECTION_SOLVER_VERSION,
            "optimality": float(fitted.optimality), "message": str(fitted.message),
            "attempts": [{"success": bool(result.success), "status": int(result.status),
                          "nfev": int(result.nfev), "cost": float(result.cost),
                          "optimality": float(result.optimality)} for result in attempts],
        }
    shape = _raw_torsions(native_ligand, bonds, masks, angles)
    shape += native_ligand.mean(0) - shape.mean(0)
    rotation_a, offset_a, ligand_rmsd, ligand_underdetermined = _fit_numpy_rigid(
        shape, endpoint_ligand,
    )
    projected_ligand = shape @ rotation_a.T + offset_a
    rotation_poi, offset_poi, poi_rmsd, poi_underdetermined = _fit_numpy_rigid(
        native_poi, endpoint_poi,
    )

    # assemble_interface_pair applies D_a o inverse(D_b). Solve D_b from the
    # fitted native-POI -> endpoint-POI transform and the fitted ligand D_a.
    rotation_b = rotation_poi.T @ rotation_a
    offset_b = rotation_poi.T @ (offset_a - offset_poi)
    center0 = native_ligand.mean(0)
    centers = np.stack((
        rotation_a @ center0 + offset_a,
        rotation_b @ center0 + offset_b,
    ))
    omegas = Rotation.from_matrix(np.stack((rotation_a, rotation_b))).as_rotvec()
    coupling = {
        "angles": angles, "omegas": omegas, "centers": centers,
        "bonds": bonds, "masks": masks,
    }
    geometry = endpoint_internal_geometry(
        native_ligand, endpoint_ligand, edge_index, atol=exact_atol,
    )
    report = {
        "contract": ENDPOINT_PROJECTION_VERSION,
        "ligand_projection_rmsd_angstrom": ligand_rmsd,
        "poi_rigid_fit_rmsd_angstrom": poi_rmsd,
        "ligand_orientation_underdetermined": ligand_underdetermined,
        "poi_orientation_underdetermined": poi_underdetermined,
        "exact_atol_angstrom": float(exact_atol),
        "endpoint_exactly_represented": bool(
            ligand_rmsd <= exact_atol and poi_rmsd <= exact_atol
        ),
        "internal_geometry": geometry,
        "torsion_optimization": optimization,
    }
    return coupling, {
        "ligand": projected_ligand,
        "poi": native_poi @ rotation_poi.T + offset_poi,
    }, report


DUAL_INIT_CACHE_VERSION = "bitseed_training_endpoints_v1"


TRAINING_ARMS = ("A1", "B", "C")


def _finite_array(value, name, shape_tail=None):
    array = np.asarray(value)
    if shape_tail is not None and array.shape[-len(shape_tail):] != shape_tail:
        raise ValueError(f"{name} has invalid shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_endpoint_record(record):
    """Validate one native-free-selected, training-target-projected endpoint."""
    required = {
        "case", "actual_source", "candidate_id", "conformer_seed",
        "binary_pose_ids", "endpoint", "training_endpoint", "coupling", "projection",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"endpoint record missing {sorted(missing)}")
    if record["actual_source"] not in ("sdpi", "bitseed"):
        raise ValueError("endpoint actual_source must be sdpi or bitseed")
    if record["actual_source"] == "bitseed" and record.get("fallback"):
        raise ValueError("an SDPI fallback cannot be labelled BiTSeed")
    if record["projection"].get("contract") != ENDPOINT_PROJECTION_VERSION:
        raise ValueError("endpoint has an unknown projection contract")
    endpoint = record["endpoint"]
    _finite_array(endpoint["ligand"], "endpoint ligand", (3,))
    _finite_array(endpoint["poi"], "endpoint POI", (3,))
    _finite_array(endpoint["body_frame"], "endpoint body frame", (3, 3))
    training_endpoint = record["training_endpoint"]
    _finite_array(training_endpoint["ligand"], "training endpoint ligand", (3,))
    _finite_array(training_endpoint["poi"], "training endpoint POI", (3,))
    _finite_array(training_endpoint["body_frame"], "training endpoint body frame", (3, 3))
    coupling = record["coupling"]
    for name in ("angles", "omegas", "centers", "bonds", "masks"):
        if name not in coupling:
            raise ValueError(f"endpoint coupling missing {name}")
    _finite_array(coupling["angles"], "coupling angles")
    _finite_array(coupling["omegas"], "coupling omegas", (2, 3))
    _finite_array(coupling["centers"], "coupling centers", (2, 3))
    if record.get("native_used_in_candidate_generation") is not False:
        raise ValueError("cache must attest native-free candidate generation")
    if record.get("native_used_in_candidate_selection") is not False:
        raise ValueError("cache must attest native-free candidate selection")


def validate_endpoint_cache(payload):
    if payload.get("cache_version") != DUAL_INIT_CACHE_VERSION:
        raise ValueError("unknown BiTSeed training endpoint cache version")
    protocol = payload.get("generation_protocol", {})
    if not protocol.get("frozen_expert_checkpoint_sha256"):
        raise ValueError("cache lacks frozen expert checkpoint hash")
    if not payload.get("entries"):
        raise ValueError("endpoint cache is empty")
    for record in payload["entries"]:
        validate_endpoint_record(record)


class CachedEndpointStore:
    """Load deterministic per-case BiTSeed couplings for arms B and C."""

    def __init__(self, path):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        validate_endpoint_cache(payload)
        self.payload = payload
        self.by_case = {}
        for record in payload["entries"]:
            self.by_case.setdefault(record["case"], []).append(record)
        if not any(record["actual_source"] == "bitseed"
                   for records in self.by_case.values() for record in records):
            raise ValueError("cache contains no actual BiTSeed endpoints")

    def coupling(self, case, occurrence=0):
        records = self.by_case.get(case, ())
        if not records:
            raise KeyError(f"case {case!r} has no cached BiTSeed endpoint")
        record = records[int(occurrence) % len(records)]
        coupling = {
            name: np.asarray(value).copy()
            for name, value in record["coupling"].items()
        }
        return coupling, deepcopy(record)


def mixed_source_for_step(global_step):
    """Fixed 1:1 batch sequence shared exactly by arms B and C."""
    return int(global_step) % 2


ONLINE_TRAINING_ENDPOINT_PROTOCOL = "online_bitseed_training_c1_m1_v1"
RESAMPLED_TRAINING_ENDPOINT_PROTOCOL = "online_bitseed_raw_resampled_v2"


class EndpointProjectionError(ValueError):
    """A sampled endpoint is unreachable under the configured motion contract."""

    def __init__(self, report, ligand_data, initial_ligand, initial_poi, native_poi):
        super().__init__(f'reachable target torsion projection did not converge: {report}')
        self.diagnostics = {
            'report': report,
            'initial_ligand': _coordinates(initial_ligand, 'initial ligand'),
            'initial_poi': _coordinates(initial_poi, 'initial POI'),
            'target_ligand': _coordinates(ligand_data['coords'], 'target ligand'),
            'target_poi': _coordinates(native_poi, 'target POI'),
            **{key: value.detach().cpu() if torch.is_tensor(value) else value
               for key, value in ligand_data.items() if key in ('edge_index', 'edge_mask', 'mask_rotate')},
        }


def reachable_ligand_target(ligand_data, initial_ligand, initial_poi, native_poi,
                            *, no_torsion=False, return_coupling=False):
    """Project only the supervision target, never the inference-like input.

    Bond lengths/rigid fragments stay those of the sampled conformer. Protein
    supervision remains the unmodified native pose. This is an attainable
    coordinate surrogate, not an assertion of exact native ligand geometry.
    """
    reference = {**ligand_data, 'coords': initial_ligand}
    coupling, projected, report = project_endpoint_coupling(
        reference, initial_poi, ligand_data['coords'], native_poi,
        no_torsion=no_torsion)
    if not report['torsion_optimization']['success']:
        raise EndpointProjectionError(report, ligand_data, initial_ligand, initial_poi, native_poi)
    if return_coupling:
        return projected['ligand'], report, coupling
    return projected['ligand'], report


def validate_raw_endpoint_record(record):
    """Raw inputs and reachable labels must not be confused with legacy couplings."""
    if record.get('actual_source') not in ('sdpi', 'bitseed'):
        raise ValueError('raw endpoint has an invalid source')
    if (record.get('native_used_in_candidate_generation') is not False
            or record.get('native_used_in_candidate_selection') is not False):
        raise ValueError('raw endpoint must attest native-free initialization')
    state = record['endpoint']
    for key in ('ligand', 'poi'):
        value = np.asarray(state[key])
        if value.ndim != 2 or value.shape[1] != 3 or not len(value):
            raise ValueError('raw endpoint coordinates must be nonempty [N,3]')
        _finite_array(value, key)
    body = np.asarray(state['body_frame'])
    if (body.shape != (3, 3) or not np.isfinite(body).all()
            or not np.allclose(body.T @ body, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(body), 1., atol=1e-4)):
        raise ValueError('raw endpoint body frame must be a proper rotation')
    target = np.asarray(record['target_ligand'])
    if target.shape != np.asarray(state['ligand']).shape or not np.isfinite(target).all():
        raise ValueError('reachable target must match the finite raw ligand shape')
    if record['projection'].get('contract') != ENDPOINT_PROJECTION_VERSION:
        raise ValueError('raw endpoint has an unknown reachable-target projection')


ONLINE_TRAINING_BASE_SEED = 41_000


ENDPOINT_INPUT_SIGNATURE_VERSION = "featurized_endpoint_inputs_v1"


def endpoint_input_signature(batch):
    """Hash ordered geometry, chemistry and graph features, independent of device."""
    digest = hashlib.sha256(ENDPOINT_INPUT_SIGNATURE_VERSION.encode())

    def update(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous().numpy()
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("endpoint inputs cannot contain object arrays")
            update((value.dtype.str, value.shape))
            update(value.tobytes(order="C"))
        elif isinstance(value, dict):
            digest.update(b"dict[")
            for key in sorted(value):
                update(key)
                update(value[key])
            digest.update(b"]")
        elif isinstance(value, (list, tuple)):
            digest.update(b"sequence[")
            for item in value:
                update(item)
            digest.update(b"]")
        else:
            data = value if isinstance(value, bytes) else json.dumps(value).encode()
            digest.update(b"bytes:" if isinstance(value, bytes) else b"json:")
            digest.update(str(len(data)).encode() + b":" + data)

    update({key: batch[key] for key in ("e3_data", "poi_data", "ligand_data")})
    return digest.hexdigest()


def _cpu_array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def case_seed(case, base_seed):
    digest = hashlib.sha256(case.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


class LazyTrainingEndpointStore:
    """Cache actual inputs and reachable labels per occurrence.

    Validation requests occurrence
    zero. Only the latest occurrence per case stays resident in host memory.
    """

    lazy = True

    def __init__(self, cache_dir, expert, *, binary_steps=20,
                 base_seed=ONLINE_TRAINING_BASE_SEED, read_only=False,
                 wait_seconds=3600.):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expert = expert
        self.binary_steps = int(binary_steps)
        self.base_seed = int(base_seed)
        self.read_only = bool(read_only)
        self.wait_seconds = float(wait_seconds)
        if self.wait_seconds < 0:
            raise ValueError('endpoint cache wait_seconds must be nonnegative')
        self.protocol = {
            "version": RESAMPLED_TRAINING_ENDPOINT_PROTOCOL,
            "conformers": 1,
            "poses_per_side": 1,
            "binary_steps": self.binary_steps,
            "binary_retry_limit": 1,
            "selection": "single_pair_no_ranking",
            "fallback": "sdpi_v1_on_any_invalid_expert_side",
            "base_seed": self.base_seed,
            "frozen_expert": expert_base_identity(expert),
            "expert_runtime": expert_runtime_identity(expert),
            "generation_version": 'raw_target_projection_v4',
        }
        self._records = {}
        self._input_signatures = {}
        self._occurrences = {}

    def _path(self, case, occurrence=0):
        safe = "".join(char if char.isalnum() or char in "-_." else "_"
                       for char in case)
        suffix = hashlib.sha256(case.encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"{safe}.{suffix}.occ{occurrence}.pt"

    def _load(self, case, input_signature, occurrence=0):
        path = self._path(case, occurrence)
        if not path.is_file():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol") != self.protocol or payload.get("case") != case:
            raise ValueError(f"{case}: lazy endpoint cache contract mismatch")
        if payload.get("input_signature") != input_signature:
            raise ValueError(
                f"{case}: endpoint input signature missing or mismatched; "
                "use a new endpoint cache directory (old cache left unchanged)"
            )
        record = payload.get("record")
        if (payload.get('occurrence') != occurrence or record.get('occurrence') != occurrence
                or record.get('case') != case):
            raise ValueError('resampled endpoint occurrence mismatch')
        validate_raw_endpoint_record(record)
        return record

    @preserve_rng
    def _generate(self, case, batch, input_signature, occurrence=0):
        e3, poi, ligand = batch["e3_data"], batch["poi_data"], batch["ligand_data"]
        initializer = DualExpertAnchorInitializer(
            self.expert, e3, poi, binary_steps=self.binary_steps,
            proposal_attempts=1, conformer_policy="etkdgv3_random",
        )
        seed = case_seed(f'{case}:{occurrence}', self.base_seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.random.default_generator.manual_seed(seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(seed)
        try:
            candidates, generation = generate_hybrid_candidates(
                initializer, e3["coords"][0], poi["coords"][0], ligand,
                seed=seed, conformer_count=1, bitseed_candidate_budget=1,
                sdpi_candidate_budget=0, binary_retry_limit=1,
                binary_consecutive_failure_limit=1,
            )
        except Exception as error:
            # Frozen experts are allowed to be undefined on OOD inputs.
            # Training must retain the case and use an SDPI endpoint no
            # matter which ordinary exception the expert path raises.
            candidates, sdpi_records = generate_sdpi_candidates(
                initializer.base, e3["coords"][0], poi["coords"][0],
                ligand, count=1, seed=seed + 1_000_000,
                seed_stride=1,
            )
            event = {
                "conformer_index": 0,
                "seed": initializer._conformer_seed(seed, 0),
                "error": f"{type(error).__name__}: {error}",
                "replacement": "sdpi_v1",
                "replacement_count": 1,
            }
            generation = {
                "requested_bitseed": 1,
                "actual_bitseed": 0,
                "requested_sdpi": 0,
                "actual_sdpi": 1,
                "successful_conformers": [],
                "failed_conformers": [event],
                "partial_conformers": [],
                "fallback_events": [event],
                "sdpi_candidates": sdpi_records,
            }
        if len(candidates) != 1:
            raise AssertionError(f"{case}: C=1/M=1 must produce exactly one endpoint")
        candidate = candidates[0]
        actual_source = candidate["candidate_id"][0]
        target, projection = reachable_ligand_target(
            ligand, candidate['ligand'], candidate['poi'], poi['coords'],
            no_torsion=initializer.no_torsion)
        record = {
            'case': case, 'actual_source': actual_source,
            'candidate_id': candidate['candidate_id'], 'occurrence': occurrence,
            'native_used_in_candidate_generation': False,
            'native_used_in_candidate_selection': False,
            'endpoint': {'ligand': _cpu_array(candidate['ligand'][0]),
                         'poi': _cpu_array(candidate['poi'][0]),
                         'body_frame': _cpu_array(candidate['rotation'][0])},
            'target_ligand': target, 'projection': projection,
            'generation': generation,
        }
        validate_raw_endpoint_record(record)
        atomic_torch_save({'case': case, 'occurrence': occurrence,
                          'protocol': self.protocol, 'record': record,
                          'input_signature': input_signature}, self._path(case, occurrence))
        return record

    def coupling(self, case, occurrence=0, *, batch=None):
        if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 0:
            raise ValueError('endpoint occurrence must be a nonnegative integer')
        if batch is None:
            raise ValueError(f"{case}: batch is required to validate endpoint inputs")
        if expert_runtime_identity(self.expert) != self.protocol['expert_runtime']:
            raise ValueError('endpoint expert runtime changed after cache construction')
        input_signature = endpoint_input_signature(batch)
        record = self._records.get(case)
        if case in self._input_signatures and self._input_signatures[case] != input_signature:
            raise ValueError(f'{case}: in-memory endpoint input signature mismatch')
        if self._occurrences.get(case) != occurrence:
            record = None
        if record is None:
            record = self._load(case, input_signature, occurrence)
        if record is None:
            if self.read_only:
                deadline = time.monotonic() + self.wait_seconds
                while record is None and time.monotonic() < deadline:
                    time.sleep(min(.1, max(0., deadline - time.monotonic())))
                    record = self._load(case, input_signature, occurrence)
                if record is None:
                    raise TimeoutError(
                        f'{case}: read-only endpoint follower timed out waiting for writer')
            else:
                record = self._generate(case, batch, input_signature, occurrence)
        self._records[case] = record
        self._input_signatures[case] = input_signature
        self._occurrences[case] = occurrence
        return {'initial_state': {name: np.asarray(value).copy()
                                  for name, value in record['endpoint'].items()},
                'target_ligand': np.asarray(record['target_ligand']).copy()}, deepcopy(record)

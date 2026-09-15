"""Necessary internal-geometry checks for SDPI flow endpoints.

The production SDPI path changes ligand torsions and a global rigid pose. Both
operations preserve bond lengths and bonded 1--3 distances. Violating either
invariant proves that an external conformer cannot be represented exactly by
the current coupling from the indexed native ligand geometry. Passing this
check is necessary, but not sufficient, for exact representability.
"""

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .pose_anchors import proper_rigid_fit
from .sdpi_flow import _raw_torsions


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
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("rigid-fit coordinates must have matching [N,3] shapes")
    centered_source = source - source.mean(0)
    centered_target = target - target.mean(0)
    underdetermined = (len(source) < 3
                       or np.linalg.matrix_rank(centered_source, tol=1e-10) < 2)
    if underdetermined:
        source_norms = np.linalg.norm(centered_source, axis=1)
        target_norms = np.linalg.norm(centered_target, axis=1)
        if source_norms.max() < 1e-10 or target_norms.max() < 1e-10:
            rotation = np.eye(3)
        else:
            source_vector = centered_source[int(np.argmax(source_norms))]
            target_vector = centered_target[int(np.argmax(target_norms))]
            rotation, _ = Rotation.align_vectors(
                target_vector[None], source_vector[None],
            )
            rotation = rotation.as_matrix()
        translation = target.mean(0) - rotation @ source.mean(0)
        fitted = source @ rotation.T + translation
        rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
        return rotation, translation, rmsd, True
    dtype = torch.float64
    rotation, translation, rmsd = proper_rigid_fit(
        torch.as_tensor(source, dtype=dtype), torch.as_tensor(target, dtype=dtype),
    )
    return (rotation.numpy(), translation.numpy(), float(rmsd), False)


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

    def residual(angles):
        shape = _raw_torsions(native_ligand, bonds, masks, angles)
        rotation, translation, _, _ = _fit_numpy_rigid(shape, endpoint_ligand)
        fitted = shape @ rotation.T + translation
        return (fitted - endpoint_ligand).reshape(-1)

    if no_torsion or not len(bonds):
        angles = np.zeros(len(bonds), dtype=np.float64)
        optimization = {"success": True, "status": 0, "nfev": 0}
    else:
        fitted = least_squares(
            residual, np.zeros(len(bonds), dtype=np.float64),
            bounds=(-np.pi, np.pi), max_nfev=int(max_nfev),
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        angles = fitted.x
        optimization = {
            "success": bool(fitted.success), "status": int(fitted.status),
            "nfev": int(fitted.nfev),
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

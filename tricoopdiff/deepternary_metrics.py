"""Exact DeepTernary full-atom DockQ bridge for TriCoopDiff predictions."""

import sys
import tempfile
from pathlib import Path

import numpy as np


def _records(path, kind):
    lines = Path(path).read_text().splitlines()
    indices = [i for i, line in enumerate(lines) if line.startswith(kind)]
    coords = np.asarray(
        [[float(lines[i][30:38]), float(lines[i][38:46]), float(lines[i][46:54])] for i in indices]
    )
    return lines, indices, coords


def _ca_coords(lines):
    return np.asarray([
        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        for line in lines
        if line.startswith("ATOM") and line[12:16].strip() == "CA"
    ])


def _fit_rigid(source, target):
    """Return row-vector transform ``source @ R + t == target``."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"rigid-fit shape mismatch: {source.shape} vs {target.shape}")
    source_center = source.mean(0)
    target_center = target.mean(0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    translation = target_center - source_center @ rotation
    return rotation, translation


def _replace_coords(lines, indices, coords, chain=None):
    if len(indices) != len(coords):
        raise ValueError(f"PDB atom-count mismatch: {len(indices)} vs {len(coords)}")
    output = list(lines)
    for index, xyz in zip(indices, coords):
        line = output[index]
        line = f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}"
        if chain is not None:
            line = f"{line[:21]}{chain}{line[22:]}"
        output[index] = line
    return output


def compute_fullatom_dockq(
    pred_ligand,
    pred_poi_ca,
    featurized_e3_ca,
    protein1_pdb,
    protein2_pdb,
    ligand_pdb,
    native_complex_pdb,
    deepternary_root,
):
    """Rebuild a predicted ternary PDB and call DeepTernary's ``cal_dockq``."""
    p1_lines, _, _ = _records(protein1_pdb, "ATOM")
    p2_lines, p2_indices, p2_atoms = _records(protein2_pdb, "ATOM")
    lig_lines, lig_indices, _ = _records(ligand_pdb, "HETATM")
    p1_ca = _ca_coords(p1_lines)
    p2_ca = _ca_coords(p2_lines)

    # Move TriCoopDiff's shared E3 frame back into the original PDB frame.
    frame_r, frame_t = _fit_rigid(np.asarray(featurized_e3_ca), p1_ca)
    pred_ligand_pdb = np.asarray(pred_ligand) @ frame_r + frame_t
    pred_poi_ca_pdb = np.asarray(pred_poi_ca) @ frame_r + frame_t

    # TriCoopDiff predicts POI as a rigid body at Cα level. Apply that fitted
    # rigid transform to every protein2 atom before invoking full-atom DockQ.
    poi_r, poi_t = _fit_rigid(p2_ca, pred_poi_ca_pdb)
    pred_p2_atoms = p2_atoms @ poi_r + poi_t
    p2_pred_lines = _replace_coords(p2_lines, p2_indices, pred_p2_atoms)
    p1_chain = next(line[21] for line in p1_lines if line.startswith("ATOM"))
    lig_pred_lines = _replace_coords(lig_lines, lig_indices, pred_ligand_pdb, chain=p1_chain)

    deepternary_root = Path(deepternary_root).resolve()
    if str(deepternary_root) not in sys.path:
        sys.path.insert(0, str(deepternary_root))
    from DockQ.dockq_util import cal_dockq

    with tempfile.TemporaryDirectory(prefix="tricoopdiff-dockq-") as temp_dir:
        pred_path = Path(temp_dir) / "predicted_complex.pdb"
        records = [
            *(line for line in p1_lines if line.startswith(("ATOM", "HETATM", "CONECT"))),
            *(line for line in p2_pred_lines if line.startswith(("ATOM", "HETATM", "CONECT"))),
            *(line for line in lig_pred_lines if line.startswith(("ATOM", "HETATM", "CONECT"))),
        ]
        pred_path.write_text("\n".join(records) + "\n")
        fnat, irms, lrms, dockq = cal_dockq(str(pred_path), str(native_complex_pdb))
    return {
        "dockq": float(dockq),
        "fnat": float(fnat),
        "irms": float(irms),
        "lrms": float(lrms),
        "method": "DeepTernary/DockQ/dockq_util.py::cal_dockq",
    }

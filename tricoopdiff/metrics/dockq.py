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


def _ligand_records(path):
    """Return a PDB-template record set for either PDB or SDF input.

    Official PROTAC indexes point to ``ligand.sdf`` because that preserves bond
    and atom ordering for DiffDock. Exact DockQ nevertheless needs a combined
    PDB. Converting the same RDKit molecule to a PDB block preserves conformer
    atom order, so predicted coordinates can be inserted without a second atom
    mapping step.
    """
    ligand_path = Path(path)
    if ligand_path.suffix.lower() in {".sdf", ".mol"}:
        from rdkit import Chem

        molecule = Chem.MolFromMolFile(
            str(ligand_path), removeHs=False, sanitize=False,
        )
        if molecule is None:
            raise ValueError(f"cannot parse ligand structure: {ligand_path}")
        # DiffDock predicts heavy-atom coordinates. Some official SDFs carry
        # explicit H atoms; remove them without reordering the heavy atoms.
        molecule = Chem.RemoveHs(molecule, sanitize=False)
        lines = Chem.MolToPDBBlock(molecule).splitlines()
        indices = [
            index for index, line in enumerate(lines)
            if line.startswith(("ATOM", "HETATM"))
        ]
        coords = np.asarray([
            [
                float(lines[index][30:38]),
                float(lines[index][38:46]),
                float(lines[index][46:54]),
            ]
            for index in indices
        ])
        return lines, indices, coords
    return _records(ligand_path, "HETATM")


def _featurized_ca_coords(path):
    """Use DiffDock's residue/alternate-location selection and atom order."""
    import prody as pr
    from tricoopdiff.diffdock.datasets.parse_chi import get_coords

    return get_coords(pr.parsePDB(str(path)))[:, 1, :]


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


def _dockq_protein_records(model_records, native_records):
    """Keep insertion-coded residues distinct for legacy integer-only fnat."""
    residue_ids = sorted({line[22:27] for line in (*model_records, *native_records)},
                         key=lambda key: (int(key[:4]), key[4]))
    if len(residue_ids) > 9999:
        raise ValueError('DockQ residue numbering exceeds PDB four-column capacity')
    numbering = {key: index + 1 for index, key in enumerate(residue_ids)}
    def normalize(records):
        selected = {}
        for line in records:
            key = (line[21:27], line[12:16])
            # One atom per residue; prefer blank/A alternate location, as the
            # protein featurizer does, without merging inserted residues.
            if key not in selected or line[16] < selected[key][16]:
                selected[key] = line
        return [f'{line[:16]} {line[17:22]}{numbering[line[22:27]]:4d} {line[27:]}'
                for line in selected.values()]
    return normalize(model_records), normalize(native_records)


def compute_fullatom_dockq(
    pred_ligand,
    pred_poi_ca,
    featurized_e3_ca,
    protein1_pdb,
    protein2_pdb,
    ligand_pdb,
    native_complex_pdb,
    deepternary_root,
    metric_protein1_pdb=None,
    metric_protein2_pdb=None,
):
    """Rebuild both proteins and call DeepTernary's full-atom ``cal_dockq``.

    ``pred_ligand``, ``pred_poi_ca``, and ``featurized_e3_ca`` must all use the
    same model-local E3 frame. The E3 coordinates are rigid-fitted to the input
    PDB to map predictions back to its absolute frame.
    """
    p1_lines, _, _ = _records(protein1_pdb, "ATOM")
    p2_lines, p2_indices, p2_atoms = _records(protein2_pdb, "ATOM")
    p1_ca = _featurized_ca_coords(protein1_pdb)
    p2_ca = _featurized_ca_coords(protein2_pdb)

    # Move TriCoopDiff's shared E3 frame back into the original PDB frame.
    frame_r, frame_t = _fit_rigid(np.asarray(featurized_e3_ca), p1_ca)
    pred_poi_ca_pdb = np.asarray(pred_poi_ca) @ frame_r + frame_t

    # TriCoopDiff predicts POI as a rigid body at Cα level. Apply that fitted
    # rigid transform to every protein2 atom before invoking full-atom DockQ.
    poi_r, poi_t = _fit_rigid(p2_ca, pred_poi_ca_pdb)
    pred_p2_atoms = p2_atoms @ poi_r + poi_t
    if (metric_protein1_pdb is None) != (metric_protein2_pdb is None):
        raise ValueError('provide both metric protein templates or neither')
    if metric_protein1_pdb is not None:
        # DeepTernary predict_one_unbound applies the predicted rigid motion
        # of the unbound receptor to its prealigned bound all-atom template.
        # Templates are used for scoring only, never as neural-model inputs.
        p1_lines, _, _ = _records(metric_protein1_pdb, 'ATOM')
        p2_lines, p2_indices, p2_atoms = _records(metric_protein2_pdb, 'ATOM')
        pred_p2_atoms = p2_atoms @ poi_r + poi_t
    p2_pred_lines = _replace_coords(p2_lines, p2_indices, pred_p2_atoms)

    deepternary_root = Path(deepternary_root).resolve()
    if str(deepternary_root) not in sys.path:
        sys.path.insert(0, str(deepternary_root))
    from DockQ.dockq_util import cal_dockq

    with tempfile.TemporaryDirectory(prefix="tricoopdiff-dockq-") as temp_dir:
        pred_path = Path(temp_dir) / "predicted_complex.pdb"
        native_path = Path(temp_dir) / "native_proteins.pdb"
        # DockQ scores only the protein--protein interface, not the ligand.
        records = [
            *(line for line in p1_lines if line.startswith("ATOM")),
            *(line for line in p2_pred_lines if line.startswith("ATOM")),
        ]
        native_records = [line for line in Path(native_complex_pdb).read_text().splitlines()
                          if line.startswith('ATOM')]
        if not native_records:
            raise ValueError('native DockQ structure has no protein atoms')
        records, native_records = _dockq_protein_records(records, native_records)
        pred_path.write_text("\n".join(records) + "\n")
        native_path.write_text('\n'.join(native_records) + '\n')
        fnat, irms, lrms, dockq = cal_dockq(str(pred_path), str(native_path))
    if (not np.isfinite([fnat, irms, lrms, dockq]).all()
            or not 0 <= fnat <= 1 or not 0 <= dockq <= 1 or min(irms, lrms) < 0):
        raise ValueError(f'invalid full-atom DockQ result: {(fnat, irms, lrms, dockq)}')
    return {
        "dockq": float(dockq),
        "fnat": float(fnat),
        "irms": float(irms),
        "lrms": float(lrms),
        "method": "DeepTernary/DockQ/dockq_util.py::cal_dockq",
        "protocol": "protein_protein_fullatom_v2",
    }

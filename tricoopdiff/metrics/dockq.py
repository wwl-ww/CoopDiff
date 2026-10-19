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
    coords = np.asarray(coords)
    if coords.shape != (len(indices), 3) or not np.isfinite(coords).all():
        raise ValueError('PDB coordinates must be a finite [N, 3] array')
    output = list(lines)
    for index, xyz in zip(indices, coords):
        line = output[index]
        fields = [f'{value:8.3f}' for value in xyz]
        if any(len(field) != 8 for field in fields):
            raise ValueError('PDB coordinates exceed eight-column capacity')
        line = f"{line[:30]}{''.join(fields)}{line[54:]}"
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
        # The vendored fnat binary compares contacts in chain-encounter order.
        # Normalize both inputs to the same chain order before invoking it;
        # otherwise a B,A native file against an A,B model can report Fnat=0
        # even when their residue contact pairs overlap.
        chains = sorted({line[21] for line in selected.values()})
        return [f'{line[:16]} {line[17:22]}{numbering[line[22:27]]:4d} {line[27:]}'
                for chain in chains for line in selected.values()
                if line[21] == chain]
    return normalize(model_records), normalize(native_records)


def _align_pose_residue_numbers(model_records, native_records):
    """Map each model chain's sequence onto native residue IDs before DockQ."""
    from Bio.Align import PairwiseAligner
    from Bio.SeqUtils import seq1

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2.
    aligner.mismatch_score = -1.
    aligner.open_gap_score = -5.
    aligner.extend_gap_score = -.5

    output = []
    for chain in ('A', 'B'):
        model = {}
        native = {}
        for line in model_records:
            if line[21] == chain:
                model.setdefault(line[22:27], []).append(line)
        for line in native_records:
            if line[21] == chain:
                native.setdefault(line[22:27], []).append(line)
        model_residues = list(model.values())
        native_residues = list(native.values())
        model_seq = ''.join(seq1(lines[0][17:20]) for lines in model_residues)
        native_seq = ''.join(seq1(lines[0][17:20]) for lines in native_residues)
        alignment = aligner.align(model_seq, native_seq)[0]
        matched = 0
        for model_index, native_index in zip(*alignment.indices):
            if (model_index < 0 or native_index < 0
                    or model_seq[model_index] != native_seq[native_index]):
                continue
            native_number = native_residues[native_index][0][22:27]
            output.extend(line[:22] + native_number + line[27:]
                          for line in model_residues[model_index])
            matched += 1
        if matched < 20:
            raise ValueError(f'only {matched} sequence-matched residues for chain {chain}')
    return output


def compute_fullatom_pose_dockq(protein_pose_pdb, native_complex_pdb, deepternary_root):
    """Score an actual HDOCK unbound all-atom pose, without CA reprojection."""
    from tricoopdiff.diffdock.datasets.constants import three_to_one

    model_records = [line for line in Path(protein_pose_pdb).read_text().splitlines()
                     if line.startswith('ATOM') and line[21] in ('A', 'B')
                     and line[17:20] in three_to_one]
    native_records = [line for line in Path(native_complex_pdb).read_text().splitlines()
                      if line.startswith('ATOM') and line[21] in ('A', 'B')
                      and line[17:20] in three_to_one]
    if ({line[21] for line in model_records} != {'A', 'B'}
            or {line[21] for line in native_records} != {'A', 'B'}):
        raise ValueError('full-atom pose DockQ requires two protein chains A and B')
    model_records = _align_pose_residue_numbers(model_records, native_records)
    model_records, native_records = _dockq_protein_records(model_records, native_records)
    deepternary_root = Path(deepternary_root).resolve()
    if str(deepternary_root) not in sys.path:
        sys.path.insert(0, str(deepternary_root))
    from DockQ.dockq_util import cal_dockq

    with tempfile.TemporaryDirectory(prefix='tricoopdiff-pose-dockq-') as temp_dir:
        model_path = Path(temp_dir)/'model.pdb'
        native_path = Path(temp_dir)/'native.pdb'
        model_path.write_text('\n'.join(model_records)+'\n')
        native_path.write_text('\n'.join(native_records)+'\n')
        fnat, irms, lrms, dockq = cal_dockq(str(model_path), str(native_path))
    if (not np.isfinite((fnat, irms, lrms, dockq)).all()
            or not 0 <= fnat <= 1 or not 0 <= dockq <= 1 or min(irms, lrms) < 0):
        raise ValueError('invalid full-atom pose DockQ result')
    return {'dockq': float(dockq), 'fnat': float(fnat),
            'irms': float(irms), 'lrms': float(lrms),
            'method': 'DeepTernary/DockQ/dockq_util.py::cal_dockq',
            'protocol': 'sequence_aligned_unbound_protein_pose_fullatom_v2'}


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
    pred_poi_rotation=None,
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
    if pred_poi_rotation is None:
        poi_r, poi_t = _fit_rigid(p2_ca, pred_poi_ca_pdb)
    else:
        body = np.asarray(pred_poi_rotation)
        if (body.shape != (3, 3) or not np.isfinite(body).all()
                or not np.allclose(body.T @ body, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(body), 1., atol=1e-4)):
            raise ValueError('supplied POI body rotation must be proper SO(3)')
        # Input featurization translates monomers without changing orientation.
        # The executor's body rotation resolves short/collinear CA ambiguity.
        poi_r = body.T @ frame_r
        poi_t = pred_poi_ca_pdb.mean(0)-p2_ca.mean(0) @ poi_r
    for role, source, target, rotation, translation in (
            ('E3', np.asarray(featurized_e3_ca), p1_ca, frame_r, frame_t),
            ('POI', p2_ca, pred_poi_ca_pdb, poi_r, poi_t)):
        # A least-squares fit is not proof of rigid representability. Silently
        # projecting a distorted prediction would score a different structure.
        # Allow PDB rounding and accumulated float32 integration error only.
        # Candidate poses may accumulate several float32 rigid updates before
        # being mapped back to the PDB frame.  A 1e-3 A floor rejects otherwise
        # exact rigid images by a few microns (observed ~1.4e-3 A); this remains
        # far below any meaningful structural deformation.
        tolerance = 2e-3 + 64 * np.finfo(np.float32).eps * max(
            float(np.abs(source).max()), float(np.abs(target).max()))
        error = np.linalg.norm(source @ rotation + translation - target, axis=-1).max()
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(f'{role} CA coordinates are not a rigid image of the input template')
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
        native_records = [line for line in Path(native_complex_pdb).read_text().splitlines()
                          if line.startswith('ATOM')]
        if not native_records:
            raise ValueError('native DockQ structure has no protein atoms')
        native_chains = {line[21] for line in native_records}
        first_chains = {line[21] for line in p1_lines if line.startswith('ATOM')}
        second_chains = {line[21] for line in p2_pred_lines if line.startswith('ATOM')}
        if len(native_chains) != 2:
            raise ValueError('native DockQ structure must contain exactly two protein chains')
        if len(first_chains) != 1 or len(second_chains) != 1:
            raise ValueError('DockQ protein templates must each contain exactly one chain')
        if (first_chains.isdisjoint(second_chains)
                and first_chains | second_chains == native_chains):
            # Explicit matching chain identities take precedence, including
            # callers that legitimately exchange the two protein roles.
            first_chain, second_chain = next(iter(first_chains)), next(iter(second_chains))
        elif native_chains == {'A', 'B'}:
            # The normalized DeepTernary protocol identifies protein1/E3 as A
            # and protein2/POI as B, regardless of standalone input chain names.
            first_chain, second_chain = 'A', 'B'
        else:
            raise ValueError('cannot identify DockQ protein roles from native chain names')
        records = [*(f'{line[:21]}{first_chain}{line[22:]}' for line in p1_lines if line.startswith('ATOM')),
                   *(f'{line[:21]}{second_chain}{line[22:]}' for line in p2_pred_lines if line.startswith('ATOM'))]
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

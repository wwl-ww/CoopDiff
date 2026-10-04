"""Regression tests that keep metric semantics aligned with DeepTernary."""

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = _load("tricoopdiff_evaluate", ROOT / "tricoopdiff" / "evaluate.py")
exact = _load("deepternary_metrics", ROOT / "tricoopdiff" / "metrics" / "dockq.py")
reference = _load("deepternary_simple_dockq", ROOT / "DeepTernary" / "DockQ" / "simple_dockq.py")


def test_rigid_fit_recovers_known_transform():
    source = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    target = source @ rotation + np.asarray([4., -2., 7.])
    fitted_r, fitted_t = exact._fit_rigid(source, target)
    assert np.max(np.abs(source @ fitted_r + fitted_t - target)) < 1e-10


def test_dockq_preserves_inserted_residues_without_duplicate_altloc_atoms():
    def atom(serial, insertion, alternate=' ', name='CA'):
        return (f'ATOM  {serial:5d} {name:^4}{alternate}GLY A  27{insertion}   '
                f'{serial:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{0.:6.2f}          C ')
    records = [atom(1, ' '), atom(2, 'A'), atom(3, 'B', 'A'), atom(4, 'B', 'B')]
    model, native = exact._dockq_protein_records(records, records)
    assert model == native and len(model) == 3
    assert [int(line[22:26]) for line in model] == [1, 2, 3]
    assert all(line[26] == ' ' and line[16] == ' ' for line in model)


def test_exact_metric_maps_predictions_from_shared_model_frame_to_pdb_frame():
    e3_local = np.asarray([
        [0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.],
    ])
    rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.asarray([12., -7., 3.])
    e3_pdb = e3_local @ rotation + translation
    poi_pdb = np.asarray([[14., -3., 3.], [15., -3., 3.], [14., -2., 3.]])
    ligand_local = np.asarray([[1., 1., 2.], [3., 2., 1.]])
    poi_local = np.asarray([[4., 0., 0.], [5., 0., 0.], [4., 1., 0.]])

    def pdb_records(coords, chain, record, atom_name):
        return "\n".join(
            f"{record:<6}{i:5d} {atom_name:<4} ALA {chain}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          C "
            for i, (x, y, z) in enumerate(coords, 1)
        ) + "\n"

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        p1_path = root / "p1.pdb"
        p2_path = root / "p2.pdb"
        ligand_path = root / "ligand.pdb"
        p1_path.write_text(pdb_records(e3_pdb, "I", "ATOM", "CA"))
        p2_path.write_text(pdb_records(poi_pdb, "J", "ATOM", "CA"))
        ligand_path.write_text(pdb_records(ligand_local, "L", "HETATM", "C1"))
        native_path = root / 'native.pdb'
        native_path.write_text(p1_path.read_text() + p2_path.read_text() + ligand_path.read_text())

        def capture(predicted, native):
            assert all(not line.startswith('HETATM') for line in Path(native).read_text().splitlines())
            poi_lines = [
                line for line in Path(predicted).read_text().splitlines()
                if line.startswith('ATOM') and line[21] == 'J'
            ]
            assert 'HETATM' not in Path(predicted).read_text()
            exported = np.asarray([
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                for line in poi_lines
            ])
            expected = poi_local @ rotation + translation
            assert np.allclose(exported, expected, atol=1e-3, rtol=0)
            return 1.0, 0.0, 0.0, 1.0

        with patch.dict(sys.modules, {
            "DockQ.dockq_util": SimpleNamespace(cal_dockq=capture),
        }), patch.object(
            exact, "_featurized_ca_coords", side_effect=[e3_pdb, poi_pdb],
        ):
            result = exact.compute_fullatom_dockq(
                ligand_local, poi_local, e3_local, p1_path, p2_path,
                ligand_path, native_path, root,
            )
        with patch.dict(sys.modules, {
            'DockQ.dockq_util': SimpleNamespace(cal_dockq=lambda *args: (1.5, 0., 0., .8)),
        }), patch.object(exact, '_featurized_ca_coords', side_effect=[e3_pdb, poi_pdb]):
            try:
                exact.compute_fullatom_dockq(
                    ligand_local, poi_local, e3_local, p1_path, p2_path,
                    ligand_path, native_path, root)
            except ValueError as error:
                assert 'invalid full-atom DockQ' in str(error)
            else:
                raise AssertionError('fnat > 1 must fail even if final DockQ is in range')
    assert result["dockq"] == 1.0


def test_sdf_ligand_records_preserve_atom_order():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(molecule, randomSeed=0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ligand.sdf"
        writer = Chem.SDWriter(str(path))
        writer.write(molecule)
        writer.close()
        lines, indices, coords = exact._ligand_records(path)
    heavy_atoms = [atom.GetIdx() for atom in molecule.GetAtoms()
                   if atom.GetAtomicNum() > 1]
    assert len(indices) == len(heavy_atoms)
    assert coords.shape == (len(heavy_atoms), 3)
    expected = molecule.GetConformer().GetPositions()[heavy_atoms]
    # SDF rounds to four decimals, then PDB to three; keep the heavy-atom order.
    assert np.allclose(coords, expected, atol=6e-4, rtol=0)
    assert all(lines[index].startswith(("ATOM", "HETATM")) for index in indices)


def test_approximate_metric_matches_deepternary_reference():
    ligand = np.asarray([[1., 0., 0.], [1., 1., 0.]])
    e3 = np.asarray([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]])
    poi = np.asarray([[2., 0., 0.], [2., 1., 0.], [2., 2., 0.]])
    pred_ligand = ligand + np.asarray([0.3, 0.1, 0.])
    pred_poi = poi + np.asarray([0.7, 0.2, 0.])
    ours = metrics.compute_deepternary_dockq(
        ligand, e3, poi, pred_ligand, e3, pred_poi,
    )
    expected = reference.cal_approx_dockq(
        tuple(torch.as_tensor(x) for x in (ligand, e3, poi)),
        tuple(torch.as_tensor(x) for x in (pred_ligand, e3, pred_poi)),
    )
    for key, value in zip(("fnat", "lrms", "irms", "dockq"), expected):
        assert abs(ours[key] - float(value)) < 1e-7


def test_perfect_pose_is_one_and_interface_perturbation_degrades():
    ligand = np.asarray([[1., 0., 0.], [1., 1., 0.]])
    e3 = np.asarray([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]])
    poi = np.asarray([[2., 0., 0.], [2., 1., 0.], [2., 2., 0.]])
    perfect = metrics.compute_deepternary_dockq(ligand, e3, poi, ligand, e3, poi)
    displaced = metrics.compute_deepternary_dockq(
        ligand, e3, poi, ligand + 10., e3, poi + 10.,
    )
    assert abs(perfect["dockq"] - 1.0) < 1e-12
    assert abs(perfect["fnat"] - 1.0) < 1e-12
    assert displaced["dockq"] < perfect["dockq"]

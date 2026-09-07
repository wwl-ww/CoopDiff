"""Regression tests that keep metric semantics aligned with DeepTernary."""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = _load("tricoopdiff_evaluate", ROOT / "tricoopdiff" / "evaluate.py")
exact = _load("deepternary_metrics", ROOT / "tricoopdiff" / "deepternary_metrics.py")
reference = _load("deepternary_simple_dockq", ROOT / "DeepTernary" / "DockQ" / "simple_dockq.py")


def test_rigid_fit_recovers_known_transform():
    source = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    target = source @ rotation + np.asarray([4., -2., 7.])
    fitted_r, fitted_t = exact._fit_rigid(source, target)
    assert np.max(np.abs(source @ fitted_r + fitted_t - target)) < 1e-10


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
    heavy = Chem.RemoveHs(molecule)
    assert len(indices) == heavy.GetNumAtoms()
    assert coords.shape == (heavy.GetNumAtoms(), 3)
    assert np.allclose(coords, heavy.GetConformer().GetPositions(), atol=1e-3), 'heavy atom order changed'
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

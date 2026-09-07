"""CPU tests for inference/training endpoint shape metrics."""

import importlib.util
from pathlib import Path
import sys

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_endpoint_support.py'
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location('audit_endpoint_support', _SCRIPT)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def test_shape_metrics_ignore_rigid_pose_and_detect_internal_change():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = points @ rotation.T + np.array([7., -3., 2.])
    assert audit.aligned_rmsd(moved, points) < 1e-12
    assert audit.distance_rmse(moved, points) < 1e-12
    changed = moved.copy()
    changed[-1, -1] += 1
    assert audit.aligned_rmsd(changed, points) > .1
    assert audit.distance_rmse(changed, points) > .1


def test_dihedral_wrap_and_measurement():
    points = np.array([[0., 1., 0.], [0., 0., 0.], [1., 0., 0.], [1., 0., 1.]])
    assert np.isclose(abs(audit._dihedral(points, (0, 1, 2, 3))), np.pi / 2)
    assert np.isclose(audit._wrap_angle(3 * np.pi), -np.pi)


if __name__ == '__main__':
    test_shape_metrics_ignore_rigid_pose_and_detect_internal_change()
    test_dihedral_wrap_and_measurement()
    print('Endpoint-support tests passed')

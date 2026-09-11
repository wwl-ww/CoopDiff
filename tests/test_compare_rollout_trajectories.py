"""Tests for paired finite-step rollout diagnostics."""

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_rollout_trajectories import cosine, rigid_rotvec, summarize


def test_guarded_cosine_and_summary_ignore_zero_motion():
    assert cosine([1, 0, 0], [2, 0, 0]) == 1.0
    assert cosine([0, 0, 0], [2, 0, 0]) is None
    rows = [{
        'center_error_before_A': 2.0,
        'center_error_after_A': 1.0,
        'center_direction_cosine_to_native': 1.0,
        'rotation_direction_cosine_to_native': None,
    }]
    result = summarize(rows)
    assert result['center_error_change_A'] == -1.0
    assert result['positive_center_direction_cosine_to_native_fraction'] == 1.0
    assert result['mean_rotation_direction_cosine_to_native'] is None


def test_rigid_rotvec_recovers_row_vector_rotation_direction():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    expected = np.array([0.2, -0.3, 0.4])
    column_rotation = Rotation.from_rotvec(expected).as_matrix()
    moved = points @ column_rotation.T + np.array([4., -2., 7.])
    actual = rigid_rotvec(points, moved)
    assert np.allclose(actual, expected, atol=1e-12, rtol=0)


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}', flush=True)

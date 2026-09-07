"""Check that failure metrics separate internal shape from rigid placement."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np

from analyze_failed_trajectories import geometry


def test_rigid_displacement_preserves_shape_metrics():
    reference = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    prediction = reference @ rotation + np.array([10., 20., 30.])
    row = geometry(prediction, prediction, reference, reference, reference)
    assert row['ligand_pose_rmsd_A'] > 20
    assert row['ligand_com_error_A'] > 20
    assert row['ligand_aligned_rmsd_A'] < 1e-10
    assert row['poi_aligned_rmsd_A'] < 1e-10
    assert abs(row['poi_rotation_to_input_deg']-90.) < 1e-8
    assert row['e3_ca_ligand_pairs_8A'] == 0
    assert row['poi_ca_ligand_pairs_8A'] > 0


if __name__ == '__main__':
    test_rigid_displacement_preserves_shape_metrics()
    print('Failure geometry test passed')

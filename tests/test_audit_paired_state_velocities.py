"""CPU checks for paired-state checkpoint diagnostics."""

from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_paired_state_velocities import load_state, state_body


def test_load_state_uses_initial_and_one_based_trajectory_steps():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        case = 'example'
        folder = root / case
        folder.mkdir()
        initial_ligand = np.arange(18, dtype=np.float32).reshape(1, 6, 3)
        initial_poi = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
        trajectory_ligand = np.stack(
            (initial_ligand[0] + 1, initial_ligand[0] + 2))
        trajectory_poi = np.stack((initial_poi[0] + 1, initial_poi[0] + 2))
        np.savez(
            folder / 'coordinates.npz', initial_ligand=initial_ligand,
            initial_poi=initial_poi, trajectory_ligand=trajectory_ligand,
            trajectory_poi=trajectory_poi)

        ligand0, poi0 = load_state(root, case, 0, 'cpu')
        ligand2, poi2 = load_state(root, case, 2, 'cpu')
        assert torch.equal(ligand0, torch.from_numpy(initial_ligand))
        assert torch.equal(poi0, torch.from_numpy(initial_poi))
        assert torch.equal(ligand2, torch.from_numpy(trajectory_ligand[1:2]))
        assert torch.equal(poi2, torch.from_numpy(trajectory_poi[1:2]))


def test_state_body_recovers_rigid_rotation_and_rejects_deformation():
    native = torch.tensor(
        [[[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]],
        dtype=torch.float64)
    expected = torch.from_numpy(
        Rotation.from_rotvec([.2, -.3, .4]).as_matrix()).to(torch.float64)
    translation = torch.tensor([[4., -2., 7.]], dtype=torch.float64)
    current = native @ expected.T + translation[:, None]
    actual = state_body(native, current)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=0)

    deformed = current.clone()
    deformed[:, 0, 0] += .1
    try:
        state_body(native, deformed)
    except ValueError as error:
        assert 'not rigid' in str(error)
    else:
        raise AssertionError('non-rigid POI state was accepted')


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}', flush=True)

"""Checks for the two-interface coordinate diffusion formulation."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tricoopdiff.legacy.diffusion import InterfaceDiffusion
from tricoopdiff.inputs.graphs import transform_ligand_to_poi_frame


def _triangle(center):
    return torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
    ).unsqueeze(0) + torch.tensor(center).view(1, 1, 3)


def test_interface_composition_preserves_poi_side_relative_geometry():
    diffusion = InterfaceDiffusion(
        temp_sampling_tr=1.0,
        temp_sampling_rot=1.0,
    )
    ligand = _triangle([2.0, -1.0, 0.5])
    poi = _triangle([-3.0, 4.0, 1.0])
    target_ligand = _triangle([7.0, 2.0, -3.0])
    tr_score = torch.tensor([[0.2, -0.3, 0.1]])
    rot_score = torch.tensor([[0.0, 0.0, 0.4]])
    time = torch.tensor([0.3])
    dt = torch.tensor([0.05])

    poi_side_proposal, _ = diffusion.reverse_step_protac(
        ligand,
        tr_score=tr_score,
        rot_score=rot_score,
        tor_score=torch.empty(1, 0),
        t_tr=time,
        t_rot=time,
        t_tor=time,
        dt_tr=dt,
        dt_rot=dt,
        dt_tor=dt,
        add_noise=False,
    )
    updated_poi = diffusion.reverse_step_poi_ligand_interface(
        poi,
        source_ligand_pos=ligand,
        target_ligand_pos=target_ligand,
        tr_score=tr_score,
        rot_score=rot_score,
        t_tr=time,
        t_rot=time,
        dt_tr=dt,
        dt_rot=dt,
        add_noise=False,
    )
    assert updated_poi.shape == poi.shape
    assert torch.isfinite(updated_poi).all()
    assert torch.equal(ligand, _triangle([2.0, -1.0, 0.5]))
    # Applying one rigid transform to POI and the sub2 proposal preserves all
    # cross-interface distances when that proposal lands on the sub1 target.
    assert torch.allclose(
        torch.cdist(updated_poi, target_ligand),
        torch.cdist(poi, poi_side_proposal),
        atol=1e-5,
    )


def test_interface_pair_noise_reconstructs_finite_frames():
    torch.manual_seed(0)
    diffusion = InterfaceDiffusion()
    ligand = _triangle([1.0, 2.0, 3.0])
    poi = _triangle([8.0, -2.0, 1.0])
    noisy_ligand, noisy_poi, noise, global_to_poi = (
        diffusion.add_noise_interface_pair(
            ligand, poi, torch.tensor([0.4]), torch.tensor([0.4]),
        )
    )
    assert noisy_ligand.shape == ligand.shape
    assert noisy_poi.shape == poi.shape
    assert global_to_poi.shape == (1, 3, 3)
    assert torch.isfinite(noisy_ligand).all()
    assert torch.isfinite(noisy_poi).all()
    assert torch.allclose(
        torch.det(global_to_poi), torch.ones(1), atol=1e-5,
    )
    assert set(noise) == {"e3_ligand", "poi_ligand"}

    # The returned matrix is the inverse/body-frame map: the same rotation
    # that placed the clean POI into the global noisy frame must recover its
    # centered template coordinates when applied on the right to row vectors.
    recovered_poi = (
        noisy_poi - noisy_poi.mean(dim=1, keepdim=True)
    ) @ global_to_poi
    clean_centered = poi - poi.mean(dim=1, keepdim=True)
    assert torch.allclose(recovered_poi, clean_centered, atol=2e-5)

    local_vector = torch.tensor([[0.3, -0.7, 1.1]])
    global_vector = local_vector @ global_to_poi.transpose(-1, -2)
    assert torch.allclose(global_vector @ global_to_poi, local_vector, atol=1e-6)


def test_body_frame_transform_centers_poi_and_roundtrips_ligand():
    rotation = torch.tensor([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
    ])
    e3_center = torch.tensor([0.0, 0.0, 0.0])
    poi_center = torch.tensor([6.0, -2.0, 3.0])
    ligand = torch.tensor([[7.0, -1.0, 3.5], [8.0, -3.0, 2.0]])
    local = transform_ligand_to_poi_frame(
        ligand, poi_center - e3_center, rotation,
    )
    recovered = local @ rotation.T + poi_center
    assert torch.allclose(recovered, ligand, atol=1e-6)


def test_poi_center_metric_uses_the_current_e3_coordinate_frame():
    import numpy as np
    from unittest.mock import patch
    from tricoopdiff.evaluate import compute_poi_com_dist

    e3 = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    poi = e3 + np.array([5., 0., 0.])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([20., -10., 3.])
    current_e3, current_poi = e3 @ rotation.T + translation, poi @ rotation.T + translation
    with patch('tricoopdiff.evaluate._parse_native_complex', return_value={'e3_ca': e3, 'poi_ca': poi}):
        legacy = compute_poi_com_dist(current_poi, 'unused')
        corrected = compute_poi_com_dist(current_poi, 'unused', current_e3)
        perturbed = compute_poi_com_dist(current_poi + np.array([0., 0., 2.]), 'unused', current_e3)
    assert legacy['poi_com_dist'] > 10, 'fixture must expose the legacy frame assumption'
    assert corrected['poi_com_dist'] < 1e-10
    assert abs(corrected['native_e3_poi_com_dist'] - 5.) < 1e-10
    assert abs(perturbed['poi_com_dist'] - 2.) < 1e-10


def test_exact_dockq_maps_shared_standalone_chain_ids_without_exiting():
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    from tricoopdiff.metrics.dockq import compute_fullatom_dockq

    def records(coords, chain):
        return ''.join(f'ATOM  {i:5d}  CA  ALA {chain}{i:4d}    '
                       f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n'
                       for i, (x, y, z) in enumerate(coords, 1))

    e3 = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    poi = e3 + np.array([0., 0., 4.])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        p1, p2, native = (root / name for name in ('p1.pdb', 'p2.pdb', 'native.pdb'))
        p1.write_text(records(e3, 'A'))
        p2.write_text(records(poi, 'A'))
        native.write_text(records(e3, 'A') + records(poi, 'B'))
        visited = []

        def capture(model, reference):
            model_records = Path(model).read_text().splitlines()
            assert {line[21] for line in model_records} == {'A', 'B'}
            assert len(model_records) == 6
            visited.append(True)
            return 1., 0., 0., 1.

        with patch.dict(sys.modules, {'DockQ.dockq_util': SimpleNamespace(cal_dockq=capture)}), \
                patch('tricoopdiff.metrics.dockq._featurized_ca_coords', side_effect=[e3, poi, e3, poi]):
            result = compute_fullatom_dockq(e3, poi, e3, p1, p2, root / 'unused.pdb', native, root)
            assert result['dockq'] == 1. and visited == [True]
            assert {line[21] for line in p2.read_text().splitlines()} == {'A'}, 'metric must not mutate source PDBs'
            native.write_text(native.read_text() + records(e3, 'C'))
            try:
                compute_fullatom_dockq(e3, poi, e3, p1, p2, root / 'unused.pdb', native, root)
            except ValueError as error:
                assert 'exactly two' in str(error)
            else:
                raise AssertionError('unsupported native chain sets must raise, not sys.exit in DockQ')
            assert visited == [True]


def test_exact_dockq_preserves_explicit_native_chain_roles_when_swapped():
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    from tricoopdiff.metrics.dockq import compute_fullatom_dockq

    def records(coords, chain):
        return ''.join(f'ATOM  {i:5d}  CA  ALA {chain}{i:4d}    '
                       f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n'
                       for i, (x, y, z) in enumerate(coords, 1))

    a = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    b = a + np.array([0., 0., 4.])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        p1, p2, native = (root / name for name in ('p1.pdb', 'p2.pdb', 'native.pdb'))
        p1.write_text(records(b, 'B'))
        p2.write_text(records(a, 'A'))
        native.write_text(records(a, 'A') + records(b, 'B'))

        def capture(model, reference):
            def coordinates(path, chain):
                return np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])]
                                 for line in Path(path).read_text().splitlines() if line[21] == chain])
            for chain in ('A', 'B'):
                assert np.allclose(coordinates(model, chain), coordinates(reference, chain)), \
                    'explicit native chain identities must survive protein-role exchange'
            return 1., 0., 0., 1.

        with patch.dict(sys.modules, {'DockQ.dockq_util': SimpleNamespace(cal_dockq=capture)}), \
                patch('tricoopdiff.metrics.dockq._featurized_ca_coords', side_effect=[b, a]):
            result = compute_fullatom_dockq(a, a, b, p1, p2, root / 'unused.pdb', native, root)
        assert result['dockq'] == 1.
        for distorted_role in ('POI', 'E3'):
            distorted_poi = np.tile(a.mean(0), (len(a), 1)) if distorted_role == 'POI' else a
            distorted_e3 = np.tile(b.mean(0), (len(b), 1)) if distorted_role == 'E3' else b
            with patch.dict(sys.modules, {'DockQ.dockq_util': SimpleNamespace(cal_dockq=capture)}), \
                    patch('tricoopdiff.metrics.dockq._featurized_ca_coords', side_effect=[b, a]):
                try:
                    compute_fullatom_dockq(a, distorted_poi, distorted_e3, p1, p2,
                                          root / 'unused.pdb', native, root)
                except ValueError as error:
                    assert f'{distorted_role} CA coordinates are not a rigid image' in str(error)
                else:
                    raise AssertionError('exact DockQ silently reconstructed a different protein shape')


def test_native_contact_parser_keeps_insertion_codes_and_default_altlocs():
    import tempfile
    import numpy as np
    from tricoopdiff.evaluate import _parse_native_complex

    def atom(serial, chain, insertion=' ', alternate=' ', x=0., record='ATOM  '):
        return (f'{record}{serial:5d}  CA {alternate}ALA {chain}  27{insertion}   '
                f'{x:8.3f}{0.:8.3f}{0.:8.3f}  1.00  0.00           C  \n')

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'native.pdb'
        path.write_text(atom(1, 'A') + atom(2, 'B', x=2.)
                        + atom(3, 'B', 'A', 'B', 99.)
                        + atom(4, 'B', 'A', 'A', 3.)
                        + atom(5, 'A', record='HETATM'))
        native = _parse_native_complex(path)
    assert native is not None
    assert np.array_equal(native['poi_ca'], np.array([[2., 0., 0.], [3., 0., 0.]])), \
        'inserted residues must remain separate, and altloc B must not replace featurized altloc A'


if __name__ == "__main__":
    test_interface_composition_preserves_poi_side_relative_geometry()
    test_interface_pair_noise_reconstructs_finite_frames()
    test_body_frame_transform_centers_poi_and_roundtrips_ligand()
    test_poi_center_metric_uses_the_current_e3_coordinate_frame()
    test_exact_dockq_maps_shared_standalone_chain_ids_without_exiting()
    test_exact_dockq_preserves_explicit_native_chain_roles_when_swapped()
    test_native_contact_parser_keeps_insertion_codes_and_default_altlocs()
    print("Interface-coordinate tests passed")

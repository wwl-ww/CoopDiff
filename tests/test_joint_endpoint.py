"""CPU oracle audit of joint rigid endpoints; no learned/blind predictions.

Independent NumPy homogeneous transforms and SVD fits provide expected values.
The target helper is deliberately test-only: it requires native coordinates.
"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tricoopdiff.diffdock_compat  # noqa: F401
from tricoopdiff.geometry.rigid import assemble_interface_pair
from tricoopdiff.sdpi_flow import SDPIFlow


METRICS = {}


def _tensor(value):
    return torch.as_tensor(value, dtype=torch.float64, device='cpu')


def _transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _apply(points, transform):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (transform @ homogeneous.T).T[:, :3]


def _fit(source, target):
    """Independent proper rigid least-squares fit, with known atom order."""
    x, y = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - x).T @ (target - y))
    sign = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ sign @ u.T
    return _transform(rotation, y - rotation @ x)


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    ligand = rng.normal(size=(13, 3)) * [2., 3., 4.] + [4., -7., 11.]
    poi = rng.normal(size=(19, 3)) * [7., 5., 9.] + [-12., 8., 3.]
    rotations = Rotation.random(2, random_state=rng).as_matrix()
    poses = [_transform(r, c - r @ ligand.mean(0))
             for r, c in zip(rotations, rng.normal(size=(2, 3)) * 20)]
    body = poses[0] @ np.linalg.inv(poses[1])
    data = {'coords': _tensor(ligand)[None],
            'edge_index': torch.empty((2, 0), dtype=torch.long),
            'edge_mask': torch.empty(0, dtype=torch.bool),
            'mask_rotate': np.empty((0, len(ligand)), dtype=bool)}
    return ligand, poi, poses, body, data


def _oracle(ligand, poi, native_ligand, native_poi, time):
    """Forward velocities whose reverse dt=time step reaches both fits.

    Let A fit ligand -> native, S fit POI -> native. Assembly applies A B^-1
    to POI, so B = S^-1 A. Both increments rotate about CURRENT ligand center.
    """
    assert time > 0
    a, s = _fit(ligand, native_ligand), _fit(poi, native_poi)
    b = np.linalg.solve(s, a)
    center = ligand.mean(0, keepdims=True)
    velocity = {'torsion': torch.empty((1, 0), dtype=torch.float64)}
    for name, transform in zip(('e3_ligand', 'poi_ligand'), (a, b)):
        velocity[name] = {
            'tr': _tensor((center - _apply(center, transform)) / time),
            'rot': _tensor(-Rotation.from_matrix(transform[:3, :3]).as_rotvec() / time)[None],
        }
    return velocity


def _step(ligand, poi, body, velocity, time, data):
    return SDPIFlow.step(_tensor(ligand)[None], _tensor(poi)[None],
                         _tensor(body)[None], velocity, _tensor([time]), data)


def test_assembly_matches_independent_homogeneous_composition():
    worst = 0.
    for seed in range(20):
        ligand, poi, (a, b), body, _ = _fixture(seed)
        placed, rotation = assemble_interface_pair(
            _tensor(poi), _tensor(a[:3, :3]), _tensor(a[:3, 3]),
            _tensor(b[:3, :3]), _tensor(b[:3, 3]))
        worst = max(worst, np.abs(placed.numpy() - _apply(poi, body)).max())
        assert np.allclose(rotation, body[:3, :3], atol=1e-12, rtol=0)
        # Distances across the B interface must survive assembly.
        dist = lambda x, y: np.linalg.norm(x[:, None] - y[None], axis=-1)
        assert np.allclose(dist(placed.numpy(), _apply(ligand, a)),
                           dist(poi, _apply(ligand, b)), atol=1e-11, rtol=0)
    assert worst < 1e-11, f'assembly coordinate error {worst}'
    METRICS['assembly_max_abs_A'] = float(worst)


def test_joint_oracle_recovers_offpath_endpoints_in_one_step():
    worst = 0.
    for seed in range(20):
        native, target_poi, (a, _), body, data = _fixture(seed)
        ligand, poi = _apply(native, a), _apply(target_poi, body)
        for time in (1., .37, 1e-4):
            velocity = _oracle(ligand, poi, native, target_poi, time)
            result = _step(ligand, poi, body[:3, :3], velocity, time, data)
            for actual, expected in zip(result, (native, target_poi, np.eye(3))):
                worst = max(worst, np.abs(actual[0].numpy() - expected).max())
    assert worst < 1e-9, f'joint endpoint recovery error {worst}'
    METRICS['joint_oracle_max_abs_A_or_body'] = float(worst)


def test_no_torsion_teacher_targets_match_independent_joint_endpoint():
    worst = 0.
    flow = SDPIFlow()
    for seed in range(8):
        native, poi, poses, _, data = _fixture(seed)
        coupling = {'angles': np.empty(0), 'bonds': np.empty((0, 2), dtype=int),
                    'masks': data['mask_rotate'],
                    'omegas': np.array([Rotation.from_matrix(p[:3, :3]).as_rotvec() for p in poses]),
                    'centers': np.array([_apply(native, p).mean(0) for p in poses])}
        for time in (1e-4, .2, .73, 1.):
            ligand, placed, body, teacher = flow.path(data, _tensor(poi)[None], coupling, _tensor([time]))
            oracle = _oracle(ligand[0].numpy(), placed[0].numpy(), native, poi, time)
            for branch in ('e3_ligand', 'poi_ligand'):
                for component in ('tr', 'rot'):
                    error = (oracle[branch][component] - teacher[branch][component]).abs().max().item()
                    worst = max(worst, error)
    assert worst < 1e-8, f'no-torsion teacher/endpoint velocity mismatch {worst}'
    METRICS['teacher_endpoint_max_abs_velocity'] = worst


def test_joint_endpoint_tangent_matches_independent_cartesian_path():
    native, target_poi, (a, _), body, data = _fixture(43)
    ligand, poi = _apply(native, a), _apply(target_poi, body)
    time = .61
    velocity = _oracle(ligand, poi, native, target_poi, time)
    center = ligand.mean(0)

    def expected(h):
        transforms = []
        for name in ('e3_ligand', 'poi_ligand'):
            v = velocity[name]
            r = Rotation.from_rotvec(-h * v['rot'][0].numpy()).as_matrix()
            transforms.append(_transform(r, center - h * v['tr'][0].numpy() - r @ center))
        aa, bb = transforms
        return _apply(ligand, aa), _apply(poi, aa @ np.linalg.inv(bb))

    eps = 1e-5
    tangent = [(p - m) / (2 * eps) for p, m in zip(expected(eps), expected(-eps))]
    errors = []
    for h in (1e-4, 5e-5):
        result = _step(ligand, poi, body[:3, :3], velocity, h, data)
        for actual, reference in zip(result[:2], expected(h)):
            assert np.allclose(actual[0], reference, atol=1e-10, rtol=0)
        errors.append(max(np.abs((actual[0].numpy() - start) / h - derivative).max()
                          for actual, start, derivative in zip(result[:2], (ligand, poi), tangent)))
    assert errors[1] < .55 * errors[0], f'Cartesian tangent failed to converge: {errors}'
    METRICS['tangent_velocity_errors_h_1e4_5e5'] = errors


def test_joint_endpoint_is_globally_equivariant():
    native, target_poi, (a, _), body, _ = _fixture(19)
    ligand, poi = _apply(native, a), _apply(target_poi, body)
    q = Rotation.from_rotvec([.8, -.4, .3]).as_matrix()
    world = _transform(q, [100., -37., 52.])
    original = _oracle(ligand, poi, native, target_poi, .4)
    moved = _oracle(*[_apply(x, world) for x in (ligand, poi, native, target_poi)], .4)
    for branch in ('e3_ligand', 'poi_ligand'):
        for component in ('tr', 'rot'):
            assert np.allclose(moved[branch][component], original[branch][component] @ q.T,
                               atol=1e-10, rtol=0), 'endpoint vectors must rotate with world frame'


def test_e3_only_correction_cannot_recover_joint_endpoint():
    native, target_poi, (a, _), body, data = _fixture(7)
    ligand, poi = _apply(native, a), _apply(target_poi, body)
    velocity = _oracle(ligand, poi, native, target_poi, .5)
    velocity['poi_ligand'] = {k: torch.zeros_like(v) for k, v in velocity['poi_ligand'].items()}
    result = _step(ligand, poi, body[:3, :3], velocity, .5, data)
    assert np.allclose(result[0][0], native, atol=1e-10, rtol=0)
    error = np.sqrt(np.mean(np.sum((result[1][0].numpy() - target_poi)**2, axis=-1)))
    assert error > 1., 'negative control must expose missing B-interface correction'
    METRICS['e3_only_oracle_poi_rmsd_A'] = float(error)


def test_rigid_oracle_cannot_remove_internal_shape_error():
    native, target_poi, (a, _), body, data = _fixture(9)
    changed = native.copy()
    changed[0] += [3., -2., 4.]
    ligand, poi = _apply(changed, a), _apply(target_poi, body)
    velocity = _oracle(ligand, poi, native, target_poi, .5)
    result = _step(ligand, poi, body[:3, :3], velocity, .5, data)
    best_fit = _apply(ligand, _fit(ligand, native))
    assert np.allclose(result[0][0], best_fit, atol=1e-10, rtol=0)
    assert np.allclose(result[1][0], target_poi, atol=1e-10, rtol=0)
    error = np.sqrt(np.mean(np.sum((best_fit - native)**2, axis=-1)))
    assert error > .5, 'rigid fitting must retain irreducible internal-shape error'
    METRICS['changed_shape_oracle_ligand_rmsd_A'] = float(error)


def test_endpoint_small_time_amplification_and_pi_branch():
    native, poi, _, _, data = _fixture(13)
    displaced = native + [3., -4., 0.]
    slow = _oracle(displaced, poi, native, poi, .5)
    fast = _oracle(displaced, poi, native, poi, .005)
    assert torch.allclose(fast['e3_ligand']['tr'], 100 * slow['e3_ligand']['tr'])
    METRICS['fixed_displacement_velocity_norm_t_0.5_0.005'] = [
        float(v['e3_ligand']['tr'].norm()) for v in (slow, fast)]
    spins = []
    for angle in (0., 1e-9, np.pi - 1e-7, np.pi, np.pi + 1e-7):
        r = Rotation.from_rotvec([angle, 0., 0.]).as_matrix()
        ligand = (native - native.mean(0)) @ r.T + native.mean(0)
        velocity = _oracle(ligand, poi, native, poi, .5)
        result = _step(ligand, poi, np.eye(3), velocity, .5, data)
        assert np.allclose(result[0][0], native, atol=1e-9, rtol=0)
        assert np.allclose(result[1][0], poi, atol=1e-9, rtol=0)
        spins.append(velocity['e3_ligand']['rot'][0].numpy())
    assert np.linalg.norm(spins[-1] - spins[-3]) > 12., 'principal SO(3) log jumps at pi'
    METRICS['pi_cut_velocity_jump_norm'] = float(np.linalg.norm(spins[-1] - spins[-3]))


def test_real_trainer_no_torsion_loss_and_gradient_against_oracle():
    """Exercise actual trainer methods with an independent oracle plus offset."""
    from tricoopdiff.train import TriCoopDiffTrainer

    native, poi, poses, _, data = _fixture(23)
    omegas = np.array([Rotation.from_matrix(p[:3, :3]).as_rotvec() for p in poses])
    centers = np.array([_apply(native, p).mean(0) for p in poses])
    coupling = {'angles': np.empty(0), 'bonds': np.empty((0, 2), dtype=int),
                'masks': data['mask_rotate'], 'omegas': omegas, 'centers': centers}
    time = .5
    poses_t = [_transform(r, c - r @ native.mean(0)) for r, c in zip(
        Rotation.from_rotvec(time * omegas).as_matrix(),
        (1 - time) * native.mean(0) + time * centers)]
    state_l = _apply(native, poses_t[0])
    state_p = _apply(poi, poses_t[0] @ np.linalg.inv(poses_t[1]))
    oracle = _oracle(state_l, state_p, native, poi, time)
    e3 = native * 3
    # Calculate prior translation scales independently of SDPIFlow.scales.
    scales = [np.sqrt(np.mean(np.sum((x - x.mean(0))**2, axis=-1)))
              * 1.4601642460337794 / 1.73 for x in (e3, poi)]

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(_tensor(.25))
            self.expert = SimpleNamespace(model=SimpleNamespace(no_torsion=True, dynamic_max_cross=False))

        def forward(self, *args, **kwargs):
            prediction = {'torsion': torch.empty((1, 0), dtype=torch.float64)}
            for index, branch in enumerate(('e3_ligand', 'poi_ligand')):
                prediction[branch] = {
                    key: oracle[branch][key] / scale + self.offset
                    for key, scale in (('tr', scales[index]), ('rot', np.pi))}
                prediction[branch]['tor'] = prediction['torsion']
            return {'normalized_velocity': prediction}

    # Avoid checkpoint/optimizer setup; retain the real class and bound methods.
    trainer = TriCoopDiffTrainer.__new__(TriCoopDiffTrainer)
    trainer.device, trainer.model, trainer.diffusion = 'cpu', Predictor(), SDPIFlow()
    trainer.torsion_expert_aux_weight = .25
    args = ({'coords': _tensor(e3)[None]}, {'coords': _tensor(poi)[None]}, data)
    kwargs = {'t_override': _tensor([time]), 'coupling_override': coupling}
    plain = trainer.compute_loss(*args, **kwargs)
    detail = trainer.compute_loss(*args, diagnostics=True, **kwargs)
    assert torch.equal(plain['loss_total'], detail['loss_total'])
    assert abs(plain['loss_total'].item() - .25) < 1e-10, 'four mean-square offsets must sum to 4*(.25)^2'
    plain['loss_total'].backward()
    assert abs(trainer.model.offset.grad.item() - 2.) < 1e-10, 'gradient must equal 8*offset'
    assert len(detail['rms']) == 14 and all(np.isfinite(v) for v in detail['rms'].values())
    try:
        trainer.compute_loss(*args, t_pol_override=_tensor([.7]), **kwargs)
    except ValueError as error:
        assert 'shared interface clock' in str(error)
    else:
        raise AssertionError('mismatched clocks must fail')
    METRICS['real_trainer_oracle_offset_loss_and_gradient'] = [
        plain['loss_total'].item(), trainer.model.offset.grad.item()]


def test_blind_no_torsion_initialization_retains_oracle_shape_floor():
    """Fresh ETKDG inputs, then native-assisted fitting; NOT blind predictions."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_initialization import ligand_input
    from tricoopdiff.initialization import TernaryInitializer

    rng = np.random.default_rng(93)
    e3 = _tensor(rng.normal(size=(23, 3)) * 8)
    target_poi = _tensor(rng.normal(size=(19, 3)) * 7 + [9., -3., 2.])
    rows = []
    for smiles in ('CCCOCC', 'CC1CCCCC1', 'c1ccccc1'):
        data = ligand_input(smiles)
        data['coords'] = data['coords'].double()
        native = data['coords'][0].numpy()
        for seed in (3, 17, 41):
            initializer = TernaryInitializer(no_torsion=True)
            ligand, poi, body, _ = initializer.initialize(e3, target_poi, data, seed=seed)
            # Verify native coordinates cannot influence the blind initializer.
            changed = {**data, 'coords': data['coords'] + 1000}
            again = initializer.initialize(e3, target_poi, changed, seed=seed)
            assert torch.equal(ligand, again[0]) and torch.equal(poi, again[1])
            oracle = _oracle(ligand[0].numpy(), poi[0].numpy(), native, target_poi.numpy(), 1.)
            recovered = SDPIFlow.step(ligand, poi, body, oracle, _tensor([1.]), data)
            fit = _apply(ligand[0].numpy(), _fit(ligand[0].numpy(), native))
            assert np.allclose(recovered[0][0], fit, atol=1e-9, rtol=0)
            assert torch.allclose(recovered[1][0], target_poi, atol=1e-9, rtol=0)
            rmsd = float(np.sqrt(np.mean(np.sum((fit - native)**2, axis=-1))))
            rows.append({'smiles': smiles, 'seed': seed, 'oracle_ligand_shape_floor_A': rmsd})
    assert max(row['oracle_ligand_shape_floor_A'] for row in rows) > .1, (
        'disabling torsion updates does not force fresh ETKDG to native geometry')
    METRICS['blind_etkdg_no_torsion_then_oracle_fit'] = rows


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}', flush=True)
    if len(sys.argv) > 1:
        output = Path(sys.argv[1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'scope': 'synthetic CPU oracle; no blind quality evaluation',
                                      'metrics': METRICS}, indent=2) + '\n')

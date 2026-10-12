"""CPU checks for explicit Matcha frame policies; no external models loaded."""

import pytest
import torch

from tricoopdiff.models.ordered_matcha_frame import FrozenOrderedResidueMatcha as FrozenMatcha


def _boundary_positions(epsilon):
    return torch.tensor([[1. + epsilon, 0., 0.], [-1., 0., 0.],
                         [0., .7, 0.], [0., -.7, 0.], [0., 0., .2]], dtype=torch.float64)


def test_ordered_matcha_frame_is_continuous_across_argmax_boundary():
    left, right = _boundary_positions(-1e-6), _boundary_positions(1e-6)
    legacy_left = FrozenMatcha.protein_anchors(left)
    legacy_right = FrozenMatcha.protein_anchors(right)
    assert legacy_left[0] != legacy_right[0], 'fixture did not cross the hard argmax boundary'
    frames = [FrozenMatcha.protein_frame(p, FrozenMatcha.protein_anchors(p))[1]
              for p in (left, right)]
    assert (frames[0] - frames[1]).abs().max() > 1.
    policy = 'ordered_residue_com_v1'
    anchors = FrozenMatcha.protein_anchors(left, policy)
    assert anchors == FrozenMatcha.protein_anchors(right, policy) == (0, 3)
    frames = [FrozenMatcha.protein_frame(p, anchors, policy)[1] for p in (left, right)]
    assert (frames[0] - frames[1]).abs().max() < 1e-5


def test_ordered_matcha_frame_is_equivariant_and_differentiable():
    positions = _boundary_positions(.1)
    policy = 'ordered_residue_com_v1'
    anchors = FrozenMatcha.protein_anchors(positions, policy)
    origin, frame = FrozenMatcha.protein_frame(positions, anchors, policy)
    rotation, _ = torch.linalg.qr(torch.tensor([[1., 2., 3.], [4., -2., 1.],
                                               [2., 1., -3.]], dtype=torch.float64))
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1
    moved = positions @ rotation.T + 17.
    moved_origin, moved_frame = FrozenMatcha.protein_frame(moved, anchors, policy)
    assert torch.allclose(moved_frame, rotation @ frame, atol=1e-12)
    assert torch.allclose(moved_origin, origin @ rotation.T + 17., atol=1e-12)
    assert torch.allclose(frame.T @ frame, torch.eye(3, dtype=frame.dtype), atol=1e-12)
    assert torch.det(frame) > .999999
    assert torch.autograd.gradcheck(lambda p: FrozenMatcha.protein_frame(p, anchors, policy)[1],
                                    (positions.requires_grad_(),))


def test_ordered_matcha_frame_rejects_degenerate_anchors_without_fallback():
    policy = 'ordered_residue_com_v1'
    with pytest.raises(ValueError, match='at least three'):
        FrozenMatcha.protein_anchors(torch.zeros(2, 3), policy)
    with pytest.raises(ValueError, match='degenerate first'):
        FrozenMatcha.protein_anchors(torch.zeros(4, 3), policy)
    line = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    with pytest.raises(ValueError, match='collinear'):
        FrozenMatcha.protein_anchors(line, policy)


def test_legacy_matcha_frame_matches_original_formula():
    positions = _boundary_positions(.2)
    centered = positions - positions.mean(0)
    first = int(centered.square().sum(-1).argmax())
    e1 = centered[first] / centered[first].norm()
    perpendicular = centered - (centered @ e1)[:, None] * e1
    second = int(perpendicular.square().sum(-1).argmax())
    e2 = perpendicular[second] / perpendicular[second].norm()
    expected = torch.stack((e1, e2, torch.linalg.cross(e1, e2)), -1)
    assert FrozenMatcha.protein_anchors(positions) == (first, second)
    assert torch.equal(FrozenMatcha.protein_frame(positions, (first, second))[1], expected)


def test_ordered_matcha_prepare_metadata_and_identity_are_explicit(monkeypatch):
    from tricoopdiff.models.expert import FrozenMatcha as LegacyMatcha, expert_base_identity
    old = {'backend': 'matcha', 'adapter_sha256': 'legacy-source-hash',
           'coordinate_contract': 'protein_fixed_anchor_frame_v1'}
    def fake_init(self, *args):
        torch.nn.Module.__init__(self)
        self.provenance = dict(old)
    monkeypatch.setattr(LegacyMatcha, '__init__', fake_init)
    expert = FrozenMatcha('/unused', '/unused', 'cpu')
    identity = expert_base_identity(expert)
    assert identity['adapter_sha256'] == old['adapter_sha256']
    assert identity['frame_policy'] == 'ordered_residue_com_v1'
    assert identity['frame_adapter_sha256']
    assert old == {'backend': 'matcha', 'adapter_sha256': 'legacy-source-hash',
                   'coordinate_contract': 'protein_fixed_anchor_frame_v1'}
    protein = {'coords': _boundary_positions(.1).unsqueeze(0), 'matcha': {'anchors': (1, 2)}}
    ligand = {'matcha': {'features': torch.ones(1, 2, 3)}}
    monkeypatch.setattr(LegacyMatcha, 'prepare_case', lambda *args: (protein, protein, ligand))
    prepared = expert.prepare_case(protein, protein, ligand, ['/a', '/b'])
    assert all(part['matcha']['anchors'] == (0, 3) for part in prepared[:2])
    assert all(part['matcha']['frame_policy'] == expert.frame_policy for part in prepared[:2])
    assert protein['matcha']['anchors'] == (1, 2), 'changed cached legacy metadata in place'
    monkeypatch.setattr(LegacyMatcha, 'read_pair', lambda *args: 'delegated')
    assert expert.read_pair(prepared[0], ligand, .5) == 'delegated'
    with pytest.raises(ValueError, match='metadata frame policy'):
        expert.read_pair(protein, ligand, .5)
    bad = {**prepared[0], 'matcha': {**prepared[0]['matcha'], 'anchors': (1, 2)}}
    with pytest.raises(ValueError, match='noncanonical residue anchors'):
        expert.read_pair(bad, ligand, .5)


def test_matcha_training_factory_uses_explicit_policy_and_saved_resume_identity(tmp_path, monkeypatch):
    import train_entry
    import tricoopdiff.models.expert as legacy
    import tricoopdiff.models.ordered_matcha_frame as ordered
    class SelectedFactory(Exception):
        pass
    def legacy_factory(**kwargs):
        raise SelectedFactory('legacy')
    def ordered_factory(**kwargs):
        raise SelectedFactory('ordered')
    monkeypatch.setattr(legacy, 'FrozenMatcha', legacy_factory)
    monkeypatch.setattr(ordered, 'FrozenOrderedResidueMatcha', ordered_factory)
    base_args = ['--train_data', '/unused/train.json', '--val_data', '/unused/val.json',
                 '--output_dir', str(tmp_path), '--training-arm', 'A1']
    for policy, factory in (('legacy_argmax_v1', 'legacy'), ('ordered_residue_com_v1', 'ordered')):
        args = train_entry.parse_args(base_args + ['--expert-backend', 'matcha', '--matcha-frame-policy', policy])
        assert args.explicit_frame_policy and args.matcha_frame_policy == policy
        with pytest.raises(SelectedFactory, match=factory):
            train_entry.build_trainer(args, torch.device('cpu'))
    checkpoint = tmp_path / 'legacy.pt'
    torch.save({'expert_base_identity': {'backend': 'matcha'}}, checkpoint)
    args = train_entry.parse_args(base_args + ['--expert-backend', 'matcha', '--resume', str(checkpoint)])
    assert not args.explicit_frame_policy
    with pytest.raises(SelectedFactory, match='legacy'):
        train_entry.build_trainer(args, torch.device('cpu'))
    args = train_entry.parse_args(base_args + ['--expert-backend', 'matcha', '--resume', str(checkpoint),
                                  '--matcha-frame-policy', 'ordered_residue_com_v1'])
    with pytest.raises(ValueError, match='cannot change'):
        train_entry.build_trainer(args, torch.device('cpu'))
    torch.save({'expert_base_identity': {'backend': 'matcha',
                                        'frame_policy': 'ordered_residue_com_v1'}}, checkpoint)
    args = train_entry.parse_args(base_args + ['--expert-backend', 'matcha', '--resume', str(checkpoint)])
    with pytest.raises(SelectedFactory, match='ordered'):
        train_entry.build_trainer(args, torch.device('cpu'))
    assert args.matcha_frame_policy == 'ordered_residue_com_v1'


def test_ordered_matcha_frame_rejects_nonfinite_arithmetic_from_finite_coordinates():
    positions = torch.tensor([[3e38, 0., 0.], [-3e38, 0., 0.],
                              [0., 3e38, 0.], [0., -3e38, 0.]])
    assert torch.isfinite(positions).all()
    with pytest.raises(ValueError, match='degenerate first'):
        FrozenMatcha.protein_anchors(positions, 'ordered_residue_com_v1')

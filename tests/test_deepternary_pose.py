"""DeepTernary predictions are reusable without reading a ternary native."""

import torch

from tricoopdiff.deepternary_pose import DeepTernaryPoseInitializer


def _record(kind, serial, name, residue, chain, residue_id, xyz):
    x, y, z = xyz
    return (f'{kind:<6}{serial:5d} {name:^4} {residue:>3} {chain}{residue_id:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  ')


def test_deepternary_pose_is_moved_into_current_e3_frame(tmp_path):
    e3 = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    poi = torch.tensor([[8., 0., 0.], [10., 0., 0.], [8., 3., 0.]])
    ligand = torch.tensor([[1., 1., 0.], [7., 1., 0.]])
    lines = []
    for index, xyz in enumerate(e3, 1):
        lines.append(_record('ATOM', index, 'CA', 'ALA', 'A', index, xyz))
    for index, xyz in enumerate(poi, 1):
        lines.append(_record('ATOM', index + 3, 'CA', 'ALA', 'B', index, xyz))
    for index, xyz in enumerate(ligand, 1):
        lines.append(_record('HETATM', index + 6, f'C{index}', 'LIG', 'A', 1, xyz))
    path = tmp_path / 'prediction.pdb'
    path.write_text('\n'.join(lines) + '\n')
    initializer = DeepTernaryPoseInitializer(path)
    translation = torch.tensor([4., -2., 5.])
    result = initializer.initialize(e3 + translation, poi, {'coords': ligand[None]}, seed=3)
    assert torch.allclose(result[0][0], ligand + translation, atol=1e-5)
    assert torch.allclose(result[1][0], poi + translation, atol=1e-5)
    assert result[3]['native_information_used'] is False
    assert result[3]['deepternary_prediction_used'] is True


def test_deepternary_cli_requires_explicit_refinement_clock(tmp_path):
    from scripts.evaluate_conditioned_entry import main

    try:
        main([
            '--checkpoint', str(tmp_path / 'checkpoint.pt'),
            '--index', str(tmp_path / 'index.json'),
            '--case', 'example',
            '--condition-manifest', str(tmp_path / 'condition.json'),
            '--out', str(tmp_path / 'report.json'),
            '--init-mode', 'deepternary_pose',
            '--deepternary-pose', str(tmp_path / 'prediction.pdb'),
        ])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError('DeepTernary refinement silently defaulted to the full diffusion clock')


def test_condition_guidance_scale_is_bounded(tmp_path):
    from scripts.evaluate_conditioned_entry import main

    try:
        main([
            '--checkpoint', str(tmp_path / 'checkpoint.pt'),
            '--index', str(tmp_path / 'index.json'), '--case', 'example',
            '--condition-manifest', str(tmp_path / 'condition.json'),
            '--out', str(tmp_path / 'report.json'),
            '--condition-guidance-scale', '1.01',
        ])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError('out-of-range condition guidance scale was accepted')

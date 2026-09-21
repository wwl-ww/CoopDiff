"""Post-inference scoring keeps reference access separate from generation."""

import json
from pathlib import Path
import runpy
from unittest.mock import patch


def test_scoring_uses_saved_coordinates_and_refuses_invalid_inputs(tmp_path):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                   / 'scripts/score_anchored_prediction.py'))
    main = namespace['main']
    case = tmp_path / 'example'
    case.mkdir()
    for name in ('unbound_protein1.pdb', 'unbound_protein2.pdb', 'ligand.sdf',
                 'protein1.pdb', 'protein2.pdb', 'gt_complex.pdb'):
        (case / name).write_text('test template')
    prediction = tmp_path / 'prediction.json'
    out = tmp_path / 'score.json'
    payload = {'case': 'example', 'checkpoint_scope': 'training',
               'checkpoint_sha256': 'a' * 64,
               'coordinate_frame': 'featurized E3 frame; E3 coordinates included',
               'ligand_coords': [[1., 2., 3.]], 'poi_coords': [[4., 5., 6.]],
               'e3_coords': [[7., 8., 9.]]}
    prediction.write_text(json.dumps(payload))
    argv = ['score', '--prediction', str(prediction), '--case-dir', str(case),
            '--deepternary-root', str(tmp_path), '--out', str(out)]
    with patch('sys.argv', argv), patch.dict(main.__globals__, {
            'compute_fullatom_dockq': lambda *args, **kwargs: {
                'dockq': .1, 'coordinate_check': args[0].tolist()}}):
        main()
        report = json.loads(out.read_text())
        assert report['metric']['coordinate_check'] == payload['ligand_coords']
        assert report['native_used_only_after_inference'] is True
        assert len(report['reference_sha256']) == 6
        before = out.read_bytes()
        try:
            main()
        except FileExistsError:
            pass
        else:
            raise AssertionError('score must not overwrite existing results')
        assert out.read_bytes() == before
    ensemble_out = tmp_path / 'ensemble_score.json'
    second = {**payload, 'ligand_coords': [[2., 2., 3.]]}
    prediction.write_text(json.dumps({**payload, 'n_walkers': 2,
                                      'predictions': [payload, second]}))
    with patch('sys.argv', argv[:-1] + [str(ensemble_out)]), patch.dict(main.__globals__, {
            'compute_fullatom_dockq': lambda *args, **kwargs: {'dockq': float(args[0][0, 0]) / 10}}):
        main()
    ensemble = json.loads(ensemble_out.read_text())
    assert ensemble['oracle_dockq'] == .2 and ensemble['n_walkers'] == 2
    assert ensemble['model_top1_computed'] is False
    assert 'metric' not in ensemble and len(ensemble['candidate_metrics']) == 2
    for change in ({'case': 'wrong'}, {'checkpoint_scope': 'implementation_smoke'},
                   {'coordinate_frame': 'unknown'}, {'poi_coords': [[float('nan'), 0, 0]]},
                   {'n_walkers': 2}):
        prediction.write_text(json.dumps({**payload, **change}))
        with patch('sys.argv', argv[:-1] + [str(tmp_path / 'invalid.json')]), patch.dict(
                main.__globals__, {'compute_fullatom_dockq': lambda *a, **k: 1 / 0}):
            try:
                main()
            except ValueError:
                pass
            else:
                raise AssertionError('invalid predictions must fail before scoring')

"""Check per-seed metric provenance without running neural models."""

from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from tricoopdiff.evaluate import evaluate_model


def test_walker_metrics_preserve_seeds_and_exact_failures():
    coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    samples = [dict(ligand_coords=coords, poi_coords=coords,
                    valid=valid, initialization={}) for valid in (True, False, True)]
    data = {'coords': torch.tensor(coords).unsqueeze(0),
            'edge_index': torch.tensor([[0, 1], [1, 0]])}
    exact = {'dockq': .5, 'fnat': .4, 'lrms': 2., 'irms': 1.}
    with ExitStack() as stack:
        stack.enter_context(patch('tricoopdiff.sampler.sample_ensemble', return_value=samples))
        stack.enter_context(patch('tricoopdiff.deepternary_metrics.compute_fullatom_dockq',
                                  side_effect=[exact, RuntimeError('metric failed')]))
        for name in ('compute_deepternary_dockq', 'compute_e3_interface_dockq',
                     'compute_poi_dockq', 'compute_poi_com_dist'):
            stack.enter_context(patch(f'tricoopdiff.evaluate.{name}', return_value=None))
        stack.enter_context(patch('tricoopdiff.ensemble_analysis.analyze_ensemble', return_value={}))
        report = evaluate_model(
            None, data, data, data, None, initializer=object(), device='cpu',
            n_walkers=3, seed=7, native_complex_pdb='dummy', deepternary_dockq_paths={})
    assert report['ensemble']['invalid_seeds'] == [8]
    assert report['ensemble']['n_exact_dockq_walkers'] == 1
    assert [row['seed'] for row in report['walkers']] == [7, 9]
    assert report['walkers'][0]['dockq'] == .5
    assert 'metric failed' in report['walkers'][1]['exact_dockq_error']
    assert all('pred' not in row and 'pred_poi' not in row for row in report['walkers'])
    assert report['structure']['dockq_best'] == .5


if __name__ == '__main__':
    test_walker_metrics_preserve_seeds_and_exact_failures()
    print('Evaluation walker provenance test passed')

"""Native-free archive, fixed-clock probe and source-sampling contracts."""

import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from collect_ranking_archive import inference_entries
from tricoopdiff.ranking.ranking_archive import (geometry_features, probe_state,
                                       select_archive_candidates, state_hash)


def test_inference_index_rejects_label_references(tmp_path):
    path = tmp_path / 'index.json'
    entry = dict(case='demo', group='a', split='train', protein1='a', protein2='b', ligand='c')
    path.write_text(json.dumps({'entries': [entry]}))
    assert inference_entries(path) == [entry]
    path.write_text(json.dumps({'entries': [{**entry, 'native_complex_pdb': 'secret'}]}))
    try:
        inference_entries(path)
    except ValueError:
        pass
    else:
        raise AssertionError('native reference must not enter generation index')


def test_archive_sampling_tracks_actual_sources_and_probabilities():
    candidates = [{'candidate_id': [s, i]} for s, n in [('bitseed', 192), ('sdpi', 64)] for i in range(n)]
    selected, probability = select_archive_candidates(candidates, 32, 7)
    assert selected == select_archive_candidates(candidates, 32, 7)[0]
    assert len(set(selected)) == 32
    assert sum(candidates[i]['candidate_id'][0] == 'bitseed' for i in selected) == 16
    assert probability == {'bitseed': 16/192, 'sdpi': 16/64}
    fallback = [{'candidate_id': ['sdpi', i]} for i in range(256)]
    ids, probability = select_archive_candidates(fallback, 32, 7)
    assert len(ids) == 32 and probability == {'bitseed': 0., 'sdpi': .125}


def test_geometry_is_rigid_invariant_and_hash_tracks_frame():
    torch.manual_seed(17)
    coords = [torch.randn(n, 3) for n in (8, 9, 4)]
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert np.allclose(geometry_features(*coords),
                       geometry_features(*(c @ rotation + 3 for c in coords)), atol=1e-5)
    state = {'ligand': coords[2], 'poi': coords[1], 'body_frame': torch.eye(3)}
    assert state_hash(state) != state_hash({**state, 'body_frame': rotation})


def test_probe_uses_saved_coordinates_fixed_clock_and_preserves_rng():
    class Model:
        training = False
        expert = SimpleNamespace(model=SimpleNamespace(dynamic_max_cross=False))

        def __call__(self, e3, poi, ligand, **kwargs):
            assert torch.allclose(ligand['coords'], torch.ones(1, 4, 3) * 7)
            assert kwargs['source'] == 1
            assert abs(float(kwargs['t_e3l']) - .05) < 1e-6
            random.random(); np.random.rand(); torch.rand(1)
            return {'rank_features': torch.ones(1, 8)}

    class Flow:
        def t_to_sigma(self, *args):
            return (torch.ones(1),) * 3

        def conditioning(self, t, dt, e3, poi):
            assert float(dt) == 0
            return torch.zeros(1, 9)

    inputs = {'coords': torch.randn(1, 4, 3)}
    state = {'ligand': torch.ones(4, 3) * 7, 'poi': torch.randn(4, 3), 'body_frame': torch.eye(3)}
    before = state_hash(state)
    torch_state, py_state, np_state = torch.random.get_rng_state(), random.getstate(), np.random.get_state()
    output = probe_state(Model(), Flow(), inputs, inputs, inputs, state, 'bitseed')
    assert output['probe_valid'] and len(output['frozen']) == 8
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert py_state == random.getstate() and np.array_equal(np_state[1], np.random.get_state()[1])
    assert before == state_hash(state)

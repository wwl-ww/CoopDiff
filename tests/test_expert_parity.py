"""Parity diagnostics must distinguish repeat noise from implementation drift."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from expert_parity import audit_binary_preservation


class ToyExperts(torch.nn.Module):
    def __init__(self, film_error=0., repeat_error=0., fail=False):
        super().__init__()
        self.dual_stage = 'soft'
        self.expert = torch.nn.Module()
        self.expert.model = torch.nn.Module()
        self.expert.model.conv_layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.film_error, self.repeat_error, self.fail = film_error, repeat_error, fail
        self.calls = 0

    def forward(self, e3, poi, ligand, **kwargs):
        if self.fail:
            raise RuntimeError('test failure')
        self.calls += 1
        nodes = [ligand['coords'][0], kwargs['ligand_b_coords'][0]]
        offset = self.repeat_error * self.calls + (self.film_error if self.dual_stage == 'annealed' else 0.)
        nodes = [node + offset for node in nodes]
        if self.dual_stage == 'independent':
            for branch in range(2):
                for layer in self.expert.model.conv_layers:
                    nodes[branch] = layer(nodes[branch])
        else:
            for layer in self.expert.model.conv_layers:
                for branch in range(2):
                    nodes[branch] = layer(nodes[branch])
        return {'interface_scores': {name: {'tr': x.mean(0)[None], 'rot': x.mean(0)[None],
                                            'tor': x.new_empty(1, 0)}
                                     for name, x in zip(('e3_ligand', 'poi_ligand'), nodes)}}


def run(model):
    coords = torch.arange(12.).reshape(1, 4, 3)
    data = {'coords': coords}
    return audit_binary_preservation(model, {'e3_data': data, 'poi_data': data, 'ligand_data': data},
                                      {'a': coords, 'b': coords + 2, 'poi': coords,
                                       'body': torch.eye(3)[None]}, torch.tensor([.1]), trace_layers=True)


def test_parity_aligns_sequential_and_interleaved_layers():
    model = ToyExperts()
    report = run(model)
    assert model.dual_stage == 'soft' and model.training
    for comparison in report['comparisons'].values():
        assert comparison['equal'] and comparison['allclose']
        assert len(comparison['layers']) == 4
        assert all(row['identical_inputs'] and row['output_max_abs_error'] == 0 for row in comparison['layers'])


def test_parity_distinguishes_repeat_noise_from_path_error():
    path = run(ToyExperts(film_error=.01))['comparisons']
    assert path['native_repeat']['equal'] and path['film_repeat']['equal']
    assert not path['native_vs_film']['allclose']
    repeat = run(ToyExperts(repeat_error=.01))['comparisons']
    assert not repeat['native_repeat']['allclose']
    assert not repeat['film_repeat']['allclose']


def test_parity_restores_stage_and_hooks_on_failure():
    model = ToyExperts(fail=True)
    try:
        run(model)
    except RuntimeError as error:
        assert str(error) == 'test failure'
    else:
        raise AssertionError('failed forward was swallowed')
    assert model.dual_stage == 'soft' and model.training
    assert all(not layer._forward_hooks for layer in model.expert.model.conv_layers)

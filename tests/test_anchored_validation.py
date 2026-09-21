"""EMA isolation, fixed-cohort selection and pre-filter population separation."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_anchored_trainer import trainer, update
from tricoopdiff.anchored_validation import (VALIDATION_POLICY, VALIDATION_TIMES,
                                           evaluate_panel, validate_index_pair)


def contract():
    return dict(policy=VALIDATION_POLICY, times=list(VALIDATION_TIMES), seed=7315, index_sha256='e' * 64)


def test_ema_context_restores_weights_mode_and_rng_even_on_failure():
    value = trainer(contract())
    update(value)
    state = {key: tensor.clone() for key, tensor in value.model.trifuse.state_dict().items()}
    random_state = torch.get_rng_state().clone()
    value.model.train()
    try:
        with value.ema_mode():
            assert not value.model.training
            for key, tensor in value.model.trifuse.state_dict().items():
                assert torch.equal(tensor, value.ema[key])
            torch.rand(17)
            raise RuntimeError('injected validation failure')
    except RuntimeError:
        pass
    assert value.model.training and not value.model.expert.training
    assert torch.equal(torch.get_rng_state(), random_state)
    assert value.global_step == 1
    for key, tensor in value.model.trifuse.state_dict().items():
        assert torch.equal(tensor, state[key])


def test_validation_selection_rejects_changed_cohorts_and_records_best():
    value = trainer(contract())
    report = {**contract(), 'global_step': 0, 'mean_loss': 1., 'cohort_sha256': 'f' * 64, 'evaluated': 2}
    assert value.record_validation(report)
    value.global_step = 1
    assert not value.record_validation({**report, 'global_step': 1, 'mean_loss': 2.})
    checkpoint = value.checkpoint(epoch=0, next_case=1, seed=7315)
    assert checkpoint['validation_state']['best_step'] == 0
    for changes in ({'cohort_sha256': '0' * 64}, {'evaluated': 0}, {'index_sha256': '0' * 64}):
        try:
            value.record_validation({**report, 'global_step': 1, 'mean_loss': .1, **changes})
        except ValueError:
            pass
        else:
            raise AssertionError('invalid validation report selected a checkpoint')
    assert value.validation_state['best_loss'] == 1.
    restored = trainer(contract())
    restored.restore(checkpoint)
    assert restored.validation_state == value.validation_state


def test_validation_panel_reuses_prior_and_does_not_draw_training_rng():
    value = trainer(contract())
    entries = [{'case': 'eligible'}, {'case': 'rejected'}]

    def prepare(*args, generator, evidence_id):
        if evidence_id.endswith('rejected'):
            return None
        return SimpleNamespace(prior_angles=torch.rand(2, generator=generator),
                               audit={'selected_atom_maps': [[0], [2]]},
                               flow=SimpleNamespace(plan=SimpleNamespace(original_indices=(1, 2))))

    def load(entry):
        return {}, {}, {'coords': torch.zeros(1, 3, 3)}, 16

    rng = torch.get_rng_state().clone()
    with patch('tricoopdiff.anchored_validation.prepare_augmented_anchored_sample', side_effect=prepare):
        with patch.object(value, 'evaluate', return_value={'loss_total': 1.}) as evaluate:
            first = evaluate_panel(value, entries, load, index_sha256='e' * 64, seed=7315)
            assert evaluate.call_count == 3
        with patch.object(value, 'evaluate', return_value={'loss_total': .5}):
            second = evaluate_panel(value, entries, load, index_sha256='e' * 64, seed=7315)
    assert first['cohort_sha256'] == second['cohort_sha256']
    assert first['evaluated'] == first['rejected'] == 1
    assert first['mean_loss'] == 1. and second['mean_loss'] == .5
    assert torch.equal(torch.get_rng_state(), rng)


def test_split_gate_checks_training_cases_removed_by_chemistry_filter():
    training = {'entries': [{'case': '1AAA_A_B_X'}],
                'metadata': {'split': 'train', 'source_index_sha256': 'a' * 64,
                             'excluded_benchmark_pdb_ids': ['9ZZZ']}}
    validation = {'entries': [{'case': '2BBB_A_B_Y'}],
                  'metadata': {'split': 'val', 'disjoint_from_index_sha256': 'a' * 64,
                               'excluded_training_pdb_ids': ['1AAA', '3CCC'],
                               'excluded_benchmark_pdb_ids': ['9ZZZ']}}
    assert validate_index_pair(training, validation)['overlap'] == []
    for forbidden in ('1AAA', '3CCC', '9ZZZ'):
        bad = copy.deepcopy(validation)
        bad['entries'][0]['case'] = forbidden + '_A_B_X'
        try:
            validate_index_pair(training, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('overlap with a filtered-out training or benchmark PDB was missed')

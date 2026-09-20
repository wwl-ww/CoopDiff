"""Completion checks must not silently accept a partial training run."""

from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_conditioned_training_run import audit_metrics


def rows():
    return [
        {'event': 'train_step', 'epoch': 1, 'global_step': 1,
         'source': 'sdpi', 'loss_total': 2.0},
        {'event': 'train_step', 'epoch': 1, 'global_step': 2,
         'source': 'bitseed', 'loss_total': 1.0},
        {'event': 'validation', 'epoch': 1, 'global_step': 2,
         'by_cell': {'sdpi/full': 2.0, 'bitseed_endpoint/full': 4.0},
         'counts': {'sdpi/full': 1, 'bitseed_endpoint/full': 1},
         'selection_loss': 3.0, 'bitseed_fallback_count': 0},
    ]


def check(data, complete=True, epochs=1):
    return audit_metrics(data, epochs=epochs, train_clusters=2,
                         val_clusters=1, complete=complete)


def rejected(data, **kwargs):
    try:
        check(data, **kwargs)
    except ValueError:
        return
    raise AssertionError('invalid training record accepted')


def test_complete_run_audit():
    report = check(rows())
    assert report['training_steps'] == 2
    assert report['validation_epochs'] == 1
    assert report['actual_sources'] == {'sdpi': 1, 'bitseed': 1}


def test_partial_run_requires_explicit_mode():
    rejected(rows(), epochs=2)
    assert check(rows(), complete=False, epochs=2)['validation_epochs'] == 1
    rejected(rows()[:-1])


def test_step_gaps_and_wrong_epochs_rejected():
    for field, value in (('global_step', 3), ('epoch', 2)):
        data = rows()
        data[1][field] = value
        rejected(data)


def test_nonfinite_loss_rejected():
    for value in (float('nan'), float('inf')):
        data = rows()
        data[0]['loss_total'] = value
        rejected(data)


def test_validation_selection_and_coverage_rejected():
    for field, value in (
            ('selection_loss', 2.0),
            ('counts', {'sdpi/full': 1, 'bitseed_endpoint/full': 0}),
            ('bitseed_fallback_count', 2),
            ('global_step', 1)):
        data = deepcopy(rows())
        data[-1][field] = value
        rejected(data)

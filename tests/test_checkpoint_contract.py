"""Checks for external frozen-expert identity in adapter-only checkpoints."""

from pathlib import Path
from types import SimpleNamespace
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tricoopdiff.models.expert import (
    expert_base_identity, validate_expert_base_identity,
)


def fixture():
    return SimpleNamespace(provenance={
        'diffdock_commit': 'abc123', 'score_checkpoint_sha256': 'deadbeef',
        'no_torsion': False, 'score_checkpoint': '/machine-specific/path.pt',
    })


def test_identity_excludes_machine_specific_path():
    assert expert_base_identity(fixture()) == {
        'diffdock_commit': 'abc123', 'score_checkpoint_sha256': 'deadbeef',
        'no_torsion': False,
    }


def test_legacy_is_labeled_and_known_mismatch_rejected():
    expert = fixture()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert validate_expert_base_identity({}, expert) == 'legacy_unrecorded'
        assert len(caught) == 1
    identity = expert_base_identity(expert)
    assert validate_expert_base_identity(
        {'expert_base_identity': identity}, expert) == identity
    try:
        validate_expert_base_identity(
            {'expert_base_identity': {**identity, 'no_torsion': True}}, expert)
    except ValueError as error:
        assert 'does not match' in str(error)
    else:
        raise AssertionError('wrong frozen expert accepted adapter weights')


if __name__ == '__main__':
    test_identity_excludes_machine_specific_path()
    test_legacy_is_labeled_and_known_mismatch_rejected()
    print('Checkpoint-contract tests passed')

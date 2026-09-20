"""Exhaustive archive extension is not a waiver for selector distribution shift."""

import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.structural_ranking_data import audit_datasets


class Dataset:
    def __init__(self, case, policy):
        self.manifest = {'checkpoint_sha256': 'same', 'steps': 20, 'pool': {},
                         'selection': policy, 'expert_assets': {'model': 'same'},
                         'diffusion_seed': 'fixed_index_seed',
                         'runtime_sha256': {name: 'same' for name in (
                             'tricoopdiff/sampler.py', 'tricoopdiff/sdpi_flow.py',
                             'tricoopdiff/consensus.py', 'tricoopdiff/geometry_encoders.py',
                             'scripts/evaluation_runner.py')}}
        self.row = {'case': case, 'group': case, 'pool_size': 256, 'archived_pairs': 256,
                    'labeled_pairs': 256, 'stages': ['initial', 'final'] * 256}

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.row


def example():
    return (Dataset('1ABC_A_B_L', 'complete_existing_pool_v1'),
            Dataset('2ABC_A_B_L', 'uniform_without_replacement_within_actual_source_balanced_v1'))


def rejected(train, val):
    try:
        audit_datasets(train, val)
    except ValueError:
        return
    raise AssertionError('incompatible full-pool protocol passed')


def test_complete_pools_accept_collection_name_difference():
    train, val = example()
    assert len(audit_datasets(train, val)) == 2


def test_missing_or_failed_outcomes_cannot_pass_as_exhaustive():
    for key, value in (('archived_pairs', 64), ('stages', ['initial', 'final'] * 255 + ['initial'])):
        train, val = example()
        train.row[key] = value
        rejected(train, val)


def test_runtime_and_selector_differences_remain_errors():
    for key in ('checkpoint_sha256', 'steps', 'pool', 'expert_assets', 'diffusion_seed', 'selection'):
        train, val = example()
        val.manifest[key] = 'changed'
        rejected(train, val)
    train, val = example()
    for name in train.manifest['runtime_sha256']:
        altered = copy.deepcopy(val)
        altered.manifest['runtime_sha256'][name] = 'changed'
        rejected(train, altered)


if __name__ == '__main__':
    test_complete_pools_accept_collection_name_difference()
    test_missing_or_failed_outcomes_cannot_pass_as_exhaustive()
    test_runtime_and_selector_differences_remain_errors()
    print('Full-pool training protocol: 3 tests passed')

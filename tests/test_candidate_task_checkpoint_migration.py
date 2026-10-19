"""Regression check for explicit legacy candidate task-row migration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tricoopdiff.models.assembly import (
    CANDIDATE_SOURCE_NAMES,
    TernaryAssembly,
    migrate_legacy_candidate_task_embedding,
    production_model_config,
)


def test_legacy_source_rows_are_duplicated_into_both_task_blocks():
    config = production_model_config(
        16, 12, True, expert_backend='matcha', expert_scalar_dim=4,
        expert_vector_dim=3, candidate_driven=True,
    )
    state = TernaryAssembly(**config).state_dict()
    legacy_config = dict(config)
    legacy_config.pop('candidate_task_contract')
    state['controller.source.weight'] = state['controller.source.weight'][:3].clone()

    migrated = migrate_legacy_candidate_task_embedding({
        'model_config': legacy_config,
        'model_state': state,
        'updates': 200,
    })

    weight = migrated['model_state']['controller.source.weight']
    assert weight.shape == (2 * len(CANDIDATE_SOURCE_NAMES), 16)
    assert torch.equal(weight[:3], state['controller.source.weight'])
    assert torch.equal(weight[3:], state['controller.source.weight'])
    assert migrated['model_config']['candidate_task_contract']
    assert migrated['checkpoint_migration']['row_map']['hdock_matcha'] == {
        'assemble': 2, 'refine': 5,
    }


def test_migration_rejects_nonlegacy_task_checkpoint():
    config = production_model_config(
        16, 12, True, expert_backend='matcha', expert_scalar_dim=4,
        expert_vector_dim=3, candidate_driven=True,
    )
    state = TernaryAssembly(**config).state_dict()
    try:
        migrate_legacy_candidate_task_embedding({
            'model_config': config,
            'model_state': state,
        })
    except ValueError as error:
        assert 'not a legacy' in str(error)
    else:
        raise AssertionError('already task-conditioned checkpoints must not be remigrated')

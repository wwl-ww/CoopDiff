"""CPU checks for training-stage contracts that must fail explicitly."""

import sys
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tricoopdiff.train import TriCoopDiffTrainer
from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.sampler import actual_candidate_source
from train_entry import (
    build_trainer, parse_args, prepare_source_batch, swap_protein_reference,
)


def _head():
    return torch.nn.Linear(4, 2)


def test_phase1_freezes_unsupervised_heads():
    trainer = SimpleNamespace(model=SimpleNamespace(
        trifuse=SimpleNamespace(judge_head=_head(), state_head=_head()),
    ))
    TriCoopDiffTrainer.set_phase(trainer, 1)
    assert not any(
        parameter.requires_grad
        for head in (trainer.model.trifuse.judge_head,
                     trainer.model.trifuse.state_head)
        for parameter in head.parameters()
    )


def test_undefined_phase_is_rejected():
    trainer = SimpleNamespace(model=SimpleNamespace(
        trifuse=SimpleNamespace(judge_head=_head(), state_head=_head()),
    ))
    try:
        TriCoopDiffTrainer.set_phase(trainer, 2)
    except ValueError as error:
        assert "labels and losses" in str(error)
    else:
        raise AssertionError("an unsupervised training phase must not run")


def test_protein_role_swap_changes_reference_without_changing_geometry():
    e3 = {'coords': torch.tensor([[[-1.0, 0, 0], [1.0, 0, 0]]]), 'tag': 'e3'}
    poi = {'coords': torch.tensor([[[9.0, 0, 0], [11.0, 0, 0]]]), 'tag': 'poi'}
    ligand = {'coords': torch.tensor([[[4.0, 2.0, 0]]])}
    new_e3, new_poi, new_ligand = swap_protein_reference(e3, poi, ligand)
    assert new_e3['tag'] == 'poi'
    assert new_poi['tag'] == 'e3'
    assert torch.allclose(new_e3['coords'].mean(dim=1), torch.zeros(1, 3))
    assert torch.allclose(
        new_ligand['coords'] - new_poi['coords'][:, :1],
        ligand['coords'] - e3['coords'][:, :1],
    )


def test_trainer_owns_recorded_activation_clamp_runtime():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trifuse = torch.nn.Linear(2, 2)
            self.expert = torch.nn.Module()

    with patch('tricoopdiff.lora.expert_lora_parameter_groups',
               return_value={'early': [], 'late': []}):
        with patch.dict(os.environ, {}, clear=True):
            trainer = TriCoopDiffTrainer(
                DummyModel(), SDPIFlow(), device='cpu', activation_clamp=37)
            assert trainer.activation_clamp == 37
            assert os.environ['NAN_CLAMP'] == '37.0'


def test_consensus_builder_passes_only_active_model_dimensions():
    args = parse_args([
        '--train_data', 'unused.json', '--val_data', 'unused.json',
        '--output_dir', 'unused',
    ])
    expert = SimpleNamespace(num_conv_layers=11)
    with patch('tricoopdiff.expert_factory.build_diffdock_expert', return_value=expert), \
            patch('tricoopdiff.consensus.ConsensusFuse') as fuse, \
            patch('tricoopdiff.consensus.ConsensusCoopDiff') as model, \
            patch('tricoopdiff.consensus_train.ConsensusTrainer') as trainer:
        result, _, trained = build_trainer(args, 'cpu', input_dim=1281)
        fuse.assert_called_once_with(
            node_dim=args.node_dim, input_dim=1281,
            num_transformer_layers=args.num_transformer_layers,
        )
        assert result is model.return_value
        assert trained is trainer.return_value
        assert trainer.call_args.kwargs['activation_clamp_mode'] == 'irrep_norm'
        assert trainer.call_args.kwargs['training_endpoint_protocol'] == 'none'


def test_c_builder_records_online_single_pair_endpoint_protocol():
    args = parse_args([
        '--train_data', 'unused.json', '--val_data', 'unused.json',
        '--output_dir', 'unused', '--training-arm', 'C',
    ])
    args.training_endpoint_protocol = 'online_bitseed_training_c1_m1_v1'
    expert = SimpleNamespace(num_conv_layers=11)
    with patch('tricoopdiff.expert_factory.build_diffdock_expert', return_value=expert), \
            patch('tricoopdiff.consensus.ConsensusFuse'), \
            patch('tricoopdiff.consensus.ConsensusCoopDiff'), \
            patch('tricoopdiff.consensus_train.ConsensusTrainer') as trainer:
        build_trainer(args, 'cpu', input_dim=1281)
        assert trainer.call_args.kwargs['training_endpoint_protocol'] == (
            'online_bitseed_training_c1_m1_v1'
        )


def test_legacy_clamp_mode_is_explicitly_selectable():
    args = parse_args([
        '--train_data', 'unused.json', '--val_data', 'unused.json',
        '--output_dir', 'unused', '--activation-clamp-mode', 'componentwise',
    ])
    assert args.activation_clamp_mode == 'componentwise'


def test_mixed_arms_share_exact_source_and_endpoint_sequence():
    class Store:
        def coupling(self, case, occurrence=0):
            return ({'case_marker': case, 'occurrence': occurrence},
                    {'actual_source': 'bitseed'})

    batch = {'case': 'case-a'}
    for step in range(6):
        arm_b = prepare_source_batch(batch, step, 'B', Store())
        arm_c = prepare_source_batch(batch, step, 'C', Store())
        assert arm_b == arm_c
        assert arm_b['source'] == step % 2
        if step % 2:
            assert arm_b['coupling_override']['occurrence'] == step // 2


def test_a1_never_consumes_cached_endpoint():
    batch = prepare_source_batch({'case': 'case-a'}, 3, 'A1', None)
    assert batch == {'case': 'case-a', 'source': 0}


def test_sampler_conditions_on_actual_fallback_source():
    assert actual_candidate_source('sdpi', {}) == 'sdpi'
    assert actual_candidate_source('expert_anchor', {'fallback': None}) == 'bitseed'
    assert actual_candidate_source(
        'expert_anchor', {'fallback': 'broad_sdpi_clash_filter'},
    ) == 'sdpi'
    assert actual_candidate_source(
        'fixed_candidate', {'candidate_source': 'bitseed'},
    ) == 'bitseed'


def test_mixed_training_relabels_ood_expert_fallback_as_sdpi():
    class Store:
        def coupling(self, case, occurrence=0):
            return {'cached': True}, {'actual_source': 'sdpi'}

    batch = prepare_source_batch({'case': 'ood'}, 1, 'C', Store())
    assert batch['source'] == 0
    assert batch['coupling_override'] == {'cached': True}


def test_consensus_builder_rejects_lora_before_loading_expert():
    with patch('tricoopdiff.expert_factory.build_diffdock_expert') as expert:
        try:
            build_trainer(SimpleNamespace(lora_rank=4), 'cpu')
        except ValueError as error:
            assert 'freezes native experts' in str(error)
        else:
            raise AssertionError('v2 must reject expert adapters')
        expert.assert_not_called()


if __name__ == "__main__":
    test_phase1_freezes_unsupervised_heads()
    test_undefined_phase_is_rejected()
    test_protein_role_swap_changes_reference_without_changing_geometry()
    test_trainer_owns_recorded_activation_clamp_runtime()
    test_consensus_builder_passes_only_active_model_dimensions()
    test_c_builder_records_online_single_pair_endpoint_protocol()
    test_consensus_builder_rejects_lora_before_loading_expert()
    test_legacy_clamp_mode_is_explicitly_selectable()
    test_mixed_arms_share_exact_source_and_endpoint_sequence()
    test_a1_never_consumes_cached_endpoint()
    test_sampler_conditions_on_actual_fallback_source()
    test_mixed_training_relabels_ood_expert_fallback_as_sdpi()
    print("Training-contract tests passed")

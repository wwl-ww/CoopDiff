"""Checkpoint and native-mask contracts for the isolated conditioned trainer."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch
from torch import nn
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from tricoopdiff.conditioning.condition_augmentation import ConditionFeatureDropout
from tricoopdiff.conditioning.conditioned_consensus_train import (
    CHECKPOINT_FORMAT,
    ConditionedConsensusTrainer,
    supervised_condition_batch,
    validate_conditioned_checkpoint,
)
from tricoopdiff.conditioning.conditioned_consensus_train import ConditionedConsensusCoopDiff, ConditionedConsensusFuse
from tricoopdiff.models.assembly import TernaryAssembly
from tricoopdiff.train import validate_assembly_checkpoint
from tricoopdiff.sdpi_flow import SDPIFlow
from evaluation_runner import EvaluationRunner


class FrozenExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_conv_layers = 1
        self.model = SimpleNamespace(dynamic_max_cross=False, no_torsion=True)
        self.provenance = {
            'diffdock_commit': 'fixture',
            'score_checkpoint_sha256': 'b' * 64,
            'no_torsion': True,
        }


def trainer(parent='a' * 64, dropout=ConditionFeatureDropout()):
    base = TernaryAssembly(node_dim=16, input_dim=8, source_conditioning=True)
    fuse = ConditionedConsensusFuse.from_base(base)
    model = ConditionedConsensusCoopDiff(FrozenExpert(), fuse)
    return ConditionedConsensusTrainer(
        model, SDPIFlow(inference_steps=20), device='cpu', lr=1e-4,
        residual_weight=1e-3, ema_decay=.99, activation_clamp_mode='irrep_norm',
        training_source_mix='sdpi_bitseed_1_1',
        training_endpoint_protocol='online_bitseed_training_c1_m1_v1',
        parent_checkpoint_sha256=parent,
        condition_dropout=dropout,
    )


def test_supervised_masks_are_discrete_aligned_and_coordinate_free():
    molecule = Chem.MolFromSmiles('CCO')
    batch = {
        'case': 'fixture',
        'ligand_data': {'mol': molecule.ToBinary(), 'coords': torch.tensor([[
            [0., 0., 0.], [5., 0., 0.], [20., 0., 0.],
        ]])},
        'e3_data': {'coords': torch.tensor([[[0., 1., 0.], [30., 0., 0.]]])},
        'poi_data': {'coords': torch.tensor([[[5., 1., 0.], [40., 0., 0.]]])},
    }
    dropout = ConditionFeatureDropout(0., 0., 0., 0., 0.)
    first, audit = supervised_condition_batch(batch, seed=9, dropout=dropout)
    second, _ = supervised_condition_batch(batch, seed=9, dropout=dropout)
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert first['ligand'][:, :2].tolist() == [[1., 1.], [1., 1.], [0., 0.]]
    assert first['protein1'][:, 0].tolist() == [1., 0.]
    assert first['protein2'][:, 0].tolist() == [1., 0.]
    assert first['ligand'][:, 6:8].tolist() == [[1., 1.], [1., 1.], [0., 0.]]
    assert set(first) == {'ligand', 'protein1', 'protein2', 'sides'}
    assert not any('coord' in key for key in first)
    assert audit['native_coordinates_returned'] is False
    assert audit['flow_state_modified'] is False


def test_conditioned_checkpoint_roundtrip_and_legacy_rejection():
    original = trainer()
    assert original.optimizer_config == {
        'base_trifuse': {'lr': 1e-4, 'weight_decay': 1e-4},
        'condition': {'lr': 1e-3, 'weight_decay': 1e-4},
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'checkpoint.pt'
        original.global_step = 17
        original.save_checkpoint(path, epoch=3, best_val_loss=2.5)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['format'] == CHECKPOINT_FORMAT
        assert payload['runtime_config']['native_coordinates_stored_in_conditions'] is False
        try:
            validate_assembly_checkpoint(payload)
        except ValueError:
            pass
        else:
            raise AssertionError('conditioned checkpoint masquerades as a v20 checkpoint')
        restored = trainer()
        restored.load_checkpoint(path)
        assert restored.global_step == 17 and restored.resume_epoch == 3
        assert all(torch.equal(value, restored.model.trifuse.state_dict()[name])
                   for name, value in original.model.trifuse.state_dict().items())
        try:
            trainer('c' * 64).load_checkpoint(path)
        except ValueError as error:
            assert 'parent checkpoint mismatch' in str(error)
        else:
            raise AssertionError('wrong C40 parent identity accepted')


def test_conditioned_checkpoint_rejects_implicit_policy():
    instance = trainer()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'checkpoint.pt'
        instance.save_checkpoint(path)
        payload = torch.load(path, map_location='cpu', weights_only=False)
    payload['runtime_config'].pop('condition_dropout')
    try:
        validate_conditioned_checkpoint(payload)
    except ValueError as error:
        assert 'dropout policy' in str(error)
    else:
        raise AssertionError('implicit condition dropout accepted')


def test_resume_dropout_change_requires_explicit_opt_in():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'checkpoint.pt'
        trainer().save_checkpoint(path, epoch=1)
        changed = trainer(dropout=ConditionFeatureDropout(pocket_only_probability=.2))
        try:
            changed.load_checkpoint(path)
        except ValueError as error:
            assert 'condition_dropout' in str(error)
        else:
            raise AssertionError('implicit resume augmentation change accepted')
        changed.load_checkpoint(path, allow_dropout_change=True)
        assert changed.resume_epoch == 1


def test_conditioned_evaluator_builds_only_the_isolated_wrapper():
    instance = trainer()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'checkpoint.pt'
        instance.save_checkpoint(path)
        payload = torch.load(path, map_location='cpu', weights_only=False)
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.model = None; runner.input_dim = None
    runner.checkpoint = payload; runner.conditioned_checkpoint = True
    runner.experimental_checkpoint = False; runner.expert = FrozenExpert()
    runner.device = torch.device('cpu')
    runner._ensure_model(8)
    assert isinstance(runner.model, ConditionedConsensusCoopDiff)
    assert runner.input_dim == 8


if __name__ == '__main__':
    test_supervised_masks_are_discrete_aligned_and_coordinate_free()
    test_conditioned_checkpoint_roundtrip_and_legacy_rejection()
    test_conditioned_checkpoint_rejects_implicit_policy()
    test_resume_dropout_change_requires_explicit_opt_in()
    test_conditioned_evaluator_builds_only_the_isolated_wrapper()

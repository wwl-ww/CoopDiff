"""Factorial validation must change conditions without resampling the flow state."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from rdkit import Chem
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
SPEC = importlib.util.spec_from_file_location(
    'train_conditioned_entry', ROOT / 'scripts/train_conditioned_entry.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_resume_requires_paired_best_checkpoint():
    try:
        module.parse_args(['--parent', 'p', '--output-dir', 'o', '--resume', 'r'])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError('unpaired resume checkpoint accepted')
    args = module.parse_args(['--parent', 'p', '--output-dir', 'o',
                              '--resume', 'r', '--resume-best', 'b'])
    assert args.resume == 'r' and args.resume_best == 'b'


class Trainer:
    def __init__(self):
        self.calls = []
        self.role_graph_radius = 0
        self.diffusion = SimpleNamespace(sample_coupling=lambda *args, **kwargs: 'sdpi-coupling')
        self.model = SimpleNamespace(
            expert=SimpleNamespace(model=SimpleNamespace(no_torsion=False)),
            train=lambda: None)

    def validate_step(self, **kwargs):
        self.calls.append(kwargs)
        return {'loss_total': 1.}


class EndpointStore:
    lazy = False

    def coupling(self, case, **kwargs):
        return 'bitseed-coupling', {'actual_source': 'bitseed'}


def test_six_cells_reuse_time_and_pair_each_coupling():
    molecule = Chem.MolFromSmiles('C')
    batch = {
        'case': 'fixture',
        'e3_data': {'coords': torch.zeros(1, 1, 3)},
        'poi_data': {'coords': torch.ones(1, 1, 3)},
        'ligand_data': {'coords': torch.zeros(1, 1, 3), 'mol': molecule.ToBinary()},
    }
    trainer = Trainer()
    result = module.validate_grid(trainer, [batch], EndpointStore(), torch.device('cpu'), 7,
                                  include_robustness=True)
    assert result['counts'] == {name: 1 for name in (
        'sdpi/full', 'sdpi/pocket_only', 'sdpi/absent',
        'bitseed_endpoint/full', 'bitseed_endpoint/pocket_only',
        'bitseed_endpoint/absent')}
    assert len(trainer.calls) == 6
    assert all(call['t_override'] is trainer.calls[0]['t_override'] for call in trainer.calls)
    assert all(call['coupling_override'] == 'sdpi-coupling' for call in trainer.calls[:3])
    assert all(call['coupling_override'] == 'bitseed-coupling' for call in trainer.calls[3:])
    assert not torch.equal(trainer.calls[0]['conditions']['sides'],
                           trainer.calls[1]['conditions']['sides'])
    assert not torch.equal(trainer.calls[1]['conditions']['sides'],
                           trainer.calls[2]['conditions']['sides'])
    trainer.calls.clear()
    result = module.validate_grid(trainer, [batch], EndpointStore(), torch.device('cpu'), 7)
    assert set(result['counts']) == {'sdpi/full', 'bitseed_endpoint/full'}
    assert len(trainer.calls) == 2
    assert 'SDPI/full' in module.CHECKPOINT_SELECTION


if __name__ == '__main__':
    test_six_cells_reuse_time_and_pair_each_coupling()

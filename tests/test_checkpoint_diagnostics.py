"""CPU gates for objective isolation and two-direction condition interventions."""

import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import torch

from diagnose_checkpoint import objective
from flow_condition_audit import opposite_state
from test_sdpi_flow import fixture


def test_staged_objectives_isolate_gradients():
    names = ('loss_e3l_tr', 'loss_pol_tr', 'loss_e3l_rot', 'loss_pol_rot', 'loss_score_tor')
    values = {name: torch.tensor(float(i + 1), requires_grad=True)
              for i, name in enumerate(names)}
    losses = {'components': values, 'loss_total': sum(values.values())}
    for stage, count in (('translation', 2), ('rigid', 4), ('all', 5)):
        gradients = torch.autograd.grad(objective(losses, stage), list(values.values()),
                                        allow_unused=True, retain_graph=True)
        assert all(g is not None and g.item() == 1 for g in gradients[:count])
        assert all(g is None for g in gradients[count:])


def test_opposite_interventions_preserve_clean_endpoint_and_inputs():
    flow, data, e3, poi, coupling = fixture()
    batch = {'ligand_data': data, 'e3_data': {'coords': e3}, 'poi_data': {'coords': poi}}
    before = copy.deepcopy(batch)
    for branch, index, clean in (('e3_ligand', 1, poi), ('poi_ligand', 0, e3)):
        kind, coords = opposite_state(flow, batch, coupling, torch.tensor([0.]), branch)
        assert kind == index
        assert torch.allclose(coords, clean[0], atol=1e-9)
        # Internal independent path checks must also pass away from the endpoint.
        kind, coords = opposite_state(flow, batch, coupling, torch.tensor([.5]), branch)
        assert torch.isfinite(coords).all()
        assert not torch.allclose(coords, clean[0])
    for name in batch:
        assert torch.equal(batch[name]['coords'], before[name]['coords'])


if __name__ == '__main__':
    test_staged_objectives_isolate_gradients()
    test_opposite_interventions_preserve_clean_endpoint_and_inputs()
    print('Checkpoint diagnostic tests passed')

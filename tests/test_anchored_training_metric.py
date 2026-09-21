from pathlib import Path
import sys

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_bounded_binary_flow import bounded
from tricoopdiff.anchored_training_metric import PHYSICAL_METRIC, PARAMETER_METRIC, binary_offset_metric


def test_binary_metric_matches_physical_scale_and_keeps_gradients():
    flow = bounded()
    state = flow.strict.ligand.new_zeros(flow.size)
    error = torch.ones_like(state, requires_grad=True)
    scales = state.new_tensor([20., 30.])
    old = binary_offset_metric(flow, state, error, scales, policy=PARAMETER_METRIC)
    new = binary_offset_metric(flow, state, error, scales, policy=PHYSICAL_METRIC)
    assert old == 1 and 0 < new < old / 100
    new.backward()
    assert torch.isfinite(error.grad).all()
    assert torch.count_nonzero(error.grad[:flow.torsion_count]) == 0
    assert (error.grad[flow.torsion_count:].abs() > 0).all()

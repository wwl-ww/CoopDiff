"""Coupled linker actions preserve prior functions and exchange bond evidence."""

from dataclasses import replace
from pathlib import Path
import sys

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_anchored_torsion import fixture
from tricoopdiff.anchored_action_adapter import AnchoredActionAdapter
from tricoopdiff.anchored_flow import AnchoredTorsionFlow
from tricoopdiff.coupled_linker_action import CoupledLinkerActionAdapter


def test_unified_single_head_zero_initialization_and_cross_bond_gradient():
    flow = AnchoredTorsionFlow(*fixture())
    parent = AnchoredActionAdapter(8).double()
    model = CoupledLinkerActionAdapter(8).double()
    assert not hasattr(model, 'net'), 'old body-aware MLP must not remain in the unified module'
    state = flow.ligand.new_zeros(3)
    context = {'global': torch.randn(1, 8, dtype=state.dtype),
        'bond': torch.randn(1, 7, 8, dtype=state.dtype, requires_grad=True),
        'ligand_directions': torch.randn_like(flow.ligand), 'poi_directions': torch.randn_like(flow.poi),
        'edge_index': torch.stack((torch.arange(7), torch.arange(1, 8)))}
    args = (flow, state, context, state, state.new_tensor([4., 7.]), .5)
    assert torch.count_nonzero(model(*args)) == 0
    with torch.no_grad():
        model.coupled_output.weight.normal_(std=.1)
    output = model(*args)
    output[0].backward()
    assert context['bond'].grad[0, flow.plan.original_indices[2]].abs().sum() > 0
    order = [2, 0, 1]
    plan = replace(flow.plan, **{key: tuple(getattr(flow.plan, key)[i] for i in order)
                                for key in ('original_indices', 'bonds', 'moving_atoms', 'moves_poi')})
    reordered = AnchoredTorsionFlow(flow.ligand, flow.poi, flow.body, plan)
    other = model(reordered, state, context, state, state.new_tensor([4., 7.]), .5)
    assert torch.allclose(output[order], other, atol=1e-12), 'bond ordering changed the coupled action'

"""Projected supervision cannot hide unreachable native internal geometry."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_bounded_binary_flow import bounded
from tricoopdiff.chart_target_projection import project_chart_target


def test_reachable_target_and_reference_immutability():
    flow = bounded()
    state = flow.strict.ligand.new_zeros(flow.size)
    state[:flow.torsion_count] = .2
    state[flow.torsion_count:] = .03
    ligand, poi, _ = flow.decode(state)
    edges = torch.stack((torch.arange(7), torch.arange(1, 8)))
    original = [x.clone() for x in (flow.strict.ligand, flow.strict.poi, flow.strict.body)]
    rng = torch.get_rng_state().clone()
    result = project_chart_target(flow, ligand, poi, edges, restarts=3)
    assert result.audit['ligand_rmsd'] < 1e-3, result.audit
    assert result.audit['poi_rmsd'] < 1e-3, result.audit
    assert torch.equal(rng, torch.get_rng_state())
    repeated = project_chart_target(flow, ligand, poi, edges, restarts=3)
    assert torch.equal(result.state, repeated.state)
    assert result.audit == repeated.audit
    for before, after in zip(original, (flow.strict.ligand, flow.strict.poi, flow.strict.body)):
        assert torch.equal(before, after), 'projection modified the deployment reference'
    assert torch.equal(ligand, flow.decode(state)[0]), 'supervision was overwritten'


def test_unreachable_internal_geometry_remains_explicit():
    flow = bounded()
    ligand, poi, _ = flow.decode(flow.strict.ligand.new_zeros(flow.size))
    ligand = ligand.clone()
    ligand[0] += torch.tensor([.4, -.2, .3], dtype=ligand.dtype)
    edges = torch.stack((torch.arange(7), torch.arange(1, 8)))
    result = project_chart_target(flow, ligand, poi, edges, restarts=3)
    assert result.audit['target_semantics'] == 'projected_endpoint_not_native'
    assert result.audit['ligand_max_error'] > .01
    assert result.audit['ligand_max_error_bond_lower_bound'] > 0
    assert result.audit['ligand_max_error'] + 1e-8 >= result.audit['ligand_max_error_bond_lower_bound']
    assert result.audit['final_equal_body_mse'] <= result.audit['initial_equal_body_mse']
    assert len(result.audit['restart_results']) == 3
    try:
        flow.encode_target(ligand, poi, edges)
    except ValueError:
        pass
    else:
        raise AssertionError('native target incorrectly declared reachable')
    recovered, _ = flow.encode_target(result.ligand, result.poi, edges)
    assert torch.allclose(flow.decode(recovered)[0], result.ligand, atol=1e-3)


if __name__ == '__main__':
    test_reachable_target_and_reference_immutability()
    test_unreachable_internal_geometry_remains_explicit()
    print('Projection audit checks passed')

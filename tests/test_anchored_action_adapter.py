"""Body-aware feature equivariance, transparent migration and executed gradients."""

from pathlib import Path
import sys

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_anchored_torsion import fixture
from tricoopdiff.anchored_action_adapter import AnchoredActionAdapter, install_action_adapter
from tricoopdiff.anchored_flow import AnchoredTorsionFlow


def test_body_aware_head_is_invariant_and_reads_poi_motion():
    flow = AnchoredTorsionFlow(*fixture())
    adapter = AnchoredActionAdapter(8).double()
    state = flow.ligand.new_tensor([.2, -.3, .5])
    context = {'global': torch.randn(1, 8, dtype=state.dtype),
               'bond': torch.randn(1, 7, 8, dtype=state.dtype),
               'ligand_directions': torch.randn_like(flow.ligand, requires_grad=True),
               'poi_directions': torch.randn_like(flow.poi, requires_grad=True)}
    velocity, scales = state.clone(), state.new_tensor([4., 7.])
    assert torch.count_nonzero(adapter(flow, state, context, velocity, scales, .5)) == 0
    with torch.no_grad():
        adapter.net[-1].weight.normal_(std=.1)
    first = adapter(flow, state, context, velocity, scales, .5)
    first.sum().backward()
    assert context['poi_directions'].grad.abs().sum() > 0, 'POI contact direction did not reach torsion head'
    assert context['ligand_directions'].grad.abs().sum() > 0
    rotation = state.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    other = AnchoredTorsionFlow(flow.ligand @ rotation.T + 17., flow.poi @ rotation.T + 17.,
                                rotation @ flow.body, flow.plan)
    transformed = {**context, 'ligand_directions': context['ligand_directions'] @ rotation.T,
                   'poi_directions': context['poi_directions'] @ rotation.T}
    second = adapter(other, state, transformed, velocity, scales, .5)
    assert torch.allclose(first, second, atol=1e-12), 'body-aware torsions depend on the global coordinate frame'


def test_real_consensus_migration_preserves_velocity_and_trains_new_head():
    _check_real_consensus_migration()


def test_real_consensus_coupled_migration_and_gradients():
    from tricoopdiff.coupled_linker_action import COUPLED_ADAPTER_POLICY
    _check_real_consensus_migration(COUPLED_ADAPTER_POLICY)


def _check_real_consensus_migration(policy=None):
    from rdkit import Chem
    from test_consensus import fixture as network_fixture
    from test_anchored_consensus import Expert
    from tricoopdiff.conditional_consensus import ConditionedConsensusFuse, ConditionedConsensusCoopDiff
    from tricoopdiff.condition_features import build_condition_features
    from tricoopdiff.anchored_torsion import compile_anchored_torsions
    from tricoopdiff.anchored_consensus import predict_anchored_velocity, anchored_target_loss

    base, graph, _, _ = network_fixture()
    model = ConditionedConsensusCoopDiff(Expert(), ConditionedConsensusFuse.from_base(base)).eval()
    types = graph['node_types']
    proteins = []
    for side in (0, 1):
        count = int((types == side).sum())
        proteins.append({'coords': graph['node_coords'][types == side][None],
            'features': graph['node_features'][types == side][None],
            'edge_index': torch.stack((torch.arange(count), torch.arange(count).roll(1)))})
    ligand = {'coords': graph['node_coords'][types == 2][None],
        'features': graph['node_features'][types == 2][None], 'edge_index': graph['lig_edge_index'],
        'edge_mask': graph['lig_edge_mask'], 'mask_rotate': graph['lig_mask_rotate'], 'edge_attr': torch.ones(6, 1)}
    plan = compile_anchored_torsions(4, ligand['edge_index'], ligand['edge_mask'], (0, 1), (2, 3))
    flow = AnchoredTorsionFlow(ligand['coords'][0], proteins[1]['coords'][0], torch.eye(3), plan)
    conditions = build_condition_features(Chem.MolFromSmiles('CCCC'), [5, 6])
    state = torch.tensor([.4])
    before, _ = predict_anchored_velocity(model, flow, state, *proteins, ligand, .5, .05,
                                         conditions=conditions, source=0)
    rng = torch.get_rng_state().clone()
    install_action_adapter(model, **({'policy': policy} if policy else {}))
    assert torch.equal(rng, torch.get_rng_state())
    after, _ = predict_anchored_velocity(model, flow, state, *proteins, ligand, .5, .05,
                                        conditions=conditions, source=0)
    assert torch.equal(before, after), 'zero-output adapter altered its parent velocity'
    losses = anchored_target_loss(model, flow, flow.ligand, flow.poi, torch.tensor([.8]),
        *proteins, ligand, .5, .05, conditions=conditions, source=0)
    losses['loss_total'].backward()
    assert model.trifuse.anchored_action_adapter.net[-1].weight.grad.abs().sum() > 0
    if policy:
        assert model.trifuse.anchored_action_adapter.coupled_output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.expert.parameters())

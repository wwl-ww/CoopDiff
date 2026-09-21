"""CPU integration through actual consensus, with a deterministic frozen expert."""

from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_consensus import fixture
from tricoopdiff.anchored_consensus import anchored_target_loss, sample_anchored
from tricoopdiff.anchored_flow import AnchoredTorsionFlow
from tricoopdiff.anchored_torsion import compile_anchored_torsions
from tricoopdiff.consensus import ConsensusCoopDiff


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.num_conv_layers = 1
        self.model = SimpleNamespace(dynamic_max_cross=False)

    def forward(self, graph):
        vector = self.weight * torch.ones(1, 3)
        return vector, vector * .5, self.weight.reshape(1)


def test_real_consensus_backward_and_sampling_share_anchored_motion():
    fuse, graph, _, _ = fixture()
    model = ConsensusCoopDiff(Expert(), fuse)
    types = graph['node_types']
    proteins = []
    for side in (0, 1):
        count = int((types == side).sum())
        proteins.append({'coords': graph['node_coords'][types == side][None],
                         'features': graph['node_features'][types == side][None],
                         'edge_index': torch.stack((torch.arange(count), torch.arange(count).roll(1)))})
    ligand = {'coords': graph['node_coords'][types == 2][None],
              'features': graph['node_features'][types == 2][None],
              'edge_index': graph['lig_edge_index'], 'edge_mask': graph['lig_edge_mask'],
              'mask_rotate': graph['lig_mask_rotate'], 'edge_attr': torch.ones(6, 1)}
    plan = compile_anchored_torsions(4, ligand['edge_index'], ligand['edge_mask'], (0, 1), (2, 3))
    flow = AnchoredTorsionFlow(ligand['coords'][0], proteins[1]['coords'][0], torch.eye(3), plan)
    data, prior = torch.zeros(1), torch.tensor([.8])
    target_ligand, target_poi, _ = flow.decode(data)
    loss = anchored_target_loss(model, flow, target_ligand, target_poi, prior,
                                *proteins, ligand, .5, .1, conditions=None, source=0)
    assert max(loss['target_audit'].values()) < 1e-5
    loss['loss_total'].backward()
    assert model.expert.weight.grad is None
    assert model.trifuse.torsion_coordinator.net[-1].weight.grad.abs().sum() > 0
    assert model.trifuse.residual_head.weight.grad.abs().sum() > 0, \
        'joint motion must propagate the executed-motion loss into rigid heads'
    result = sample_anchored(model, flow, prior, *proteins, ligand, steps=3,
                             conditions=None, source=0)
    assert model.training, 'sampling did not restore model training mode'
    assert torch.equal(result['ligand_coords'][:2], flow.ligand[:2])
    assert torch.allclose(torch.cdist(result['ligand_coords'][2:], result['poi_coords']),
                          torch.cdist(flow.ligand[2:], flow.poi), atol=1e-5)


def test_incompatible_target_fails_before_model_forward():
    from test_anchored_torsion import fixture as geometry_fixture

    flow = AnchoredTorsionFlow(*geometry_fixture())
    edges = torch.stack((torch.arange(7), torch.arange(1, 8)))
    # None deliberately cannot be forwarded: the geometry gate must fire first.
    try:
        anchored_target_loss(None, flow, flow.ligand, flow.poi + 1,
                             flow.ligand.new_zeros(3), None, None,
                             {'edge_index': edges}, .5, .1, conditions=None, source=0)
    except ValueError as error:
        assert 'outside anchored chart' in str(error)
    else:
        raise AssertionError('unreachable target was forwarded to the model')


if __name__ == '__main__':
    test_real_consensus_backward_and_sampling_share_anchored_motion()
    test_incompatible_target_fails_before_model_forward()
    print('Anchored consensus integration checks passed')

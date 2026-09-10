"""Radius fallback contracts, including saturation across source blocks."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.diffdock_compat import _radius_pytorch, _radius_graph_pytorch


def test_radius_keeps_cuda_source_order_and_strict_boundary():
    x = torch.tensor([[0.5, 0.], [0.25, 0.], [0.75, 0.], [1., 0.]])
    y = torch.zeros(1, 2)
    edges = _radius_pytorch(x, y, 1, max_num_neighbors=2)
    assert edges.dtype == torch.long
    assert torch.equal(edges, torch.tensor([[0, 0], [0, 1]]))
    edges = _radius_pytorch(x, y, 1, max_num_neighbors=10)
    assert torch.equal(edges[1], torch.tensor([0, 1, 2])), 'radius boundary is strict'


def test_cap_is_shared_across_source_blocks_and_batches():
    x = torch.zeros(2200, 3)
    x[:1020] = 10
    y = torch.zeros(3, 3)
    bx = torch.tensor([0] * 2100 + [2] * 100)
    by = torch.tensor([0, 1, 2])
    edges = _radius_pytorch(x, y, 1, bx, by, max_num_neighbors=8)
    assert torch.equal(edges[0], torch.tensor([0] * 8 + [2] * 8))
    assert torch.equal(edges[1], torch.cat((torch.arange(1020, 1028),
                                           torch.arange(2100, 2108))))


def test_graph_matches_163_self_filter_and_positional_arguments():
    x = torch.zeros(6, 3)
    # v1.6.3 requests cap+1 before removing self, even if self was not selected.
    edges = _radius_graph_pytorch(x, 1, None, False, 2)
    assert not (edges[0] == edges[1]).any()
    assert torch.equal(edges[0, edges[1] == 5], torch.tensor([0, 1, 2]))
    reverse = _radius_graph_pytorch(x, 1, flow='target_to_source', max_num_neighbors=2)
    assert torch.equal(reverse, edges.flip(0))


def test_empty_zero_cap_and_large_coordinates():
    x = torch.tensor([10000., 10000.25, 10000.5])
    edges = _radius_pytorch(x, x[:1], .4)
    assert torch.equal(edges, torch.tensor([[0, 0], [0, 1]]))
    assert _radius_pytorch(x, x, 1, max_num_neighbors=0).shape == (2, 0)
    assert _radius_pytorch(x[:0], x, 1).shape == (2, 0)
    assert _radius_graph_pytorch(x, 1, max_num_neighbors=0).shape == (2, 0)


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print(name, 'passed')

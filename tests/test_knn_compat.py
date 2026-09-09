"""Directed-neighborhood contracts used by the frozen DiffDock receptor graph."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.diffdock_compat import _knn_graph_pytorch


def test_knn_messages_flow_from_neighbor_to_query():
    coords = torch.tensor([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    edges = _knn_graph_pytorch(coords, 1)
    assert torch.equal(edges, torch.tensor([[1, 0, 1], [0, 1, 2]]))
    reverse = _knn_graph_pytorch(coords, 1, flow='target_to_source')
    assert torch.equal(reverse, edges.flip(0))


def test_small_receptors_do_not_acquire_self_loops():
    coords = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    edges = _knn_graph_pytorch(coords, 24)
    assert edges.shape == (2, 90)
    assert not (edges[0] == edges[1]).any()
    assert torch.equal(torch.bincount(edges[1]), torch.full((10,), 9))
    assert _knn_graph_pytorch(coords[:1], 24).shape == (2, 0)


def test_batches_and_explicit_self_loops():
    coords = torch.tensor([[0., 0.], [1., 0.], [0., 0.], [2., 0.]])
    batch = torch.tensor([0, 0, 1, 1])
    edges = _knn_graph_pytorch(coords, 1, batch=batch)
    assert torch.equal(batch[edges[0]], batch[edges[1]])
    loops = _knn_graph_pytorch(coords, 1, batch=batch, loop=True)
    assert torch.equal(loops, torch.arange(4).repeat(2, 1))


def test_large_translation_preserves_nearest_neighbor_order():
    torch.manual_seed(118)
    coords = torch.randn(40, 3) + 10000
    distances = (coords[:, None] - coords[None, :]).square().sum(-1)
    distances.fill_diagonal_(float('inf'))
    nearest = distances.argmin(-1)
    edges = _knn_graph_pytorch(coords, 1)
    assert torch.equal(edges[0], nearest), 'cancellation changed nearest neighbors'
    assert torch.equal(edges[1], torch.arange(len(coords)))


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print(name, 'passed')

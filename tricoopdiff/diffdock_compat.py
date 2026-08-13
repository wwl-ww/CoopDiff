"""
Pure-PyTorch fallbacks for `torch_cluster` and `torch_scatter`.

The cluster has `torch_cluster` / `torch_scatter` installed as stub packages
whose native CUDA kernels fail to build/load. DiffDock's `models.cg_model`
imports `radius` / `radius_graph` / `knn_graph` from `torch_cluster` and
`scatter` / `scatter_mean` from `torch_scatter` at *import time* (via
`from torch_cluster import radius`), so these fallbacks MUST be applied
BEFORE anything imports `models.cg_model` (i.e. before `get_model`).

Importing this module applies the patches immediately as a side effect.
`apply_patches()` re-applies them explicitly (idempotent).

Convention (matches torch_cluster): `radius(x, y, r)` returns
`edge_index = [y_idx, x_idx]` — index 0 into the query/center set `y`,
index 1 into the source set `x`.
"""

import torch


def _radius_pytorch(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32,
                    batch_size=None):
    """Replacement for torch_cluster.radius.

    For each element in `y`, find all points in `x` within distance `r`.
    Returns [y_indices, x_indices].
    """
    if batch_x is None:
        batch_x = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    if batch_y is None:
        batch_y = torch.zeros(y.shape[0], dtype=torch.long, device=y.device)
    edge_index_list = []
    for bx in batch_x.unique():
        mask_x = (batch_x == bx)
        idx_x = mask_x.nonzero(as_tuple=False).squeeze(-1)
        xi = x[mask_x]
        for by_ in batch_y.unique():
            if bx != by_:
                continue
            mask_y = (batch_y == by_)
            idx_y = mask_y.nonzero(as_tuple=False).squeeze(-1)
            yj = y[mask_y]
            block_size = 1024
            for a in range(0, xi.shape[0], block_size):
                a_end = min(a + block_size, xi.shape[0])
                xi_block = xi[a:a_end]
                for b in range(0, yj.shape[0], block_size):
                    b_end = min(b + block_size, yj.shape[0])
                    yj_block = yj[b:b_end]
                    dist_block = torch.cdist(xi_block, yj_block)
                    rows, cols = (dist_block <= r).nonzero(as_tuple=True)
                    if rows.numel() > 0:
                        # torch_cluster convention: [y_idx, x_idx]
                        edge_index_list.append(
                            torch.stack([idx_y[b + cols], idx_x[a + rows]], dim=0))
    if not edge_index_list:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(edge_index_list, dim=1).int()


def _radius_graph_pytorch(x, r, batch=None, max_num_neighbors=32,
                          flow='source_to_target', loop=False, max_radius=None):
    """Replacement for torch_cluster.radius_graph."""
    assert flow in ['source_to_target', 'target_to_source']
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    ei = _radius_pytorch(x, x, r, batch, batch, max_num_neighbors)
    if flow == 'source_to_target':
        row, col = ei[1], ei[0]  # swap: message flows neighbor→center
    else:
        row, col = ei[0], ei[1]
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0).int()


def _knn_graph_pytorch(x, k, batch=None, loop=False, flow='source_to_target',
                       cosine=False, num_workers=1):
    """Replacement for torch_cluster.knn_graph."""
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    ei_list = []
    for b in batch.unique():
        mask = (batch == b)
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        xi = x[mask]
        dist = torch.cdist(xi, xi)
        if not loop:
            dist.diagonal().fill_(float('inf'))
        kk = min(k, xi.shape[0])
        _, topk = dist.topk(kk, dim=1, largest=False)
        rows = torch.arange(xi.shape[0], device=x.device).unsqueeze(1).expand(-1, kk).reshape(-1)
        cols = topk.reshape(-1)
        valid = cols < xi.shape[0]
        rows, cols = rows[valid], cols[valid]
        ei_list.append(torch.stack([idx[rows], idx[cols]], dim=0))
    if not ei_list:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(ei_list, dim=1).int()  # int32 for torch.bincount


def _scatter_pytorch(src, index, dim_size=None, dim=0, reduce='sum'):
    """Replacement for torch_scatter.scatter."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    if index.numel() == 0:
        shape = list(src.shape)
        shape[dim] = dim_size
        return torch.zeros(shape, dtype=src.dtype, device=src.device)
    if reduce == 'sum':
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.index_add_(0, index, src)
        return out
    elif reduce == 'mean':
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.index_add_(0, index, src)
        count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        count.index_add_(0, index, torch.ones(src.shape[0], dtype=src.dtype, device=src.device))
        return out / count.clamp(min=1).unsqueeze(-1)
    elif reduce == 'max':
        out = torch.full((dim_size, *src.shape[1:]), float('-inf'),
                         dtype=src.dtype, device=src.device)
        return out.scatter_reduce(0, index.unsqueeze(-1).expand_as(src), src, reduce='amax')
    elif reduce == 'min':
        out = torch.full((dim_size, *src.shape[1:]), float('inf'),
                         dtype=src.dtype, device=src.device)
        return out.scatter_reduce(0, index.unsqueeze(-1).expand_as(src), src, reduce='amin')
    else:
        raise ValueError(f"Unknown reduce: {reduce}")


def _scatter_mean_pytorch(src, index, dim_size=None, dim=0):
    return _scatter_pytorch(src, index, dim_size, dim, reduce='mean')


def apply_patches():
    """Patch torch_cluster + torch_scatter with pure-PyTorch fallbacks.

    Idempotent. MUST be called before `models.cg_model` is imported.
    """
    import torch_cluster
    import torch_scatter
    torch_cluster.radius = _radius_pytorch
    torch_cluster.radius_graph = _radius_graph_pytorch
    torch_cluster.knn_graph = _knn_graph_pytorch
    torch_scatter.scatter = _scatter_pytorch
    torch_scatter.scatter_mean = _scatter_mean_pytorch
    return True


# Apply on import (idempotent). This guarantees the patch is in place the
# moment `diffdock_compat` is imported, which expert_factory does before any
# DiffDock import.
apply_patches()

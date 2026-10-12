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

import math

import torch


def _radius_pytorch(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32,
                    num_workers=1, batch_size=None):
    """Replacement for torch_cluster.radius.

    Return [query, source] edges with torch_cluster 1.6.3 CUDA semantics.

    The radius boundary is strict. Saturated neighborhoods keep the first
    source indices, matching the CUDA kernel's scan order; native CPU uses
    a spatial tree and can select a different subset. Counts span all blocks.
    """
    if not math.isfinite(float(r)) or r < 0:
        raise ValueError('radius must be finite and non-negative')
    if max_num_neighbors < 0:
        raise ValueError('max_num_neighbors must be non-negative')
    if x.numel() == 0 or y.numel() == 0 or max_num_neighbors == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    if batch_x is None:
        batch_x = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    if batch_y is None:
        batch_y = torch.zeros(y.shape[0], dtype=torch.long, device=y.device)
    edge_index_list = []
    for bx in batch_x.unique():
        mask_x = (batch_x == bx)
        idx_x = mask_x.nonzero(as_tuple=False).squeeze(-1)
        xi = x[mask_x]
        idx_y = (batch_y == bx).nonzero(as_tuple=False).squeeze(-1)
        for b in range(0, idx_y.numel(), 256):
            query_indices = idx_y[b:b + 256]
            queries = y[query_indices]
            used = torch.zeros(len(queries), dtype=torch.long, device=x.device)
            for a in range(0, xi.shape[0], 1024):
                sources = xi[a:a + 1024]
                # Direct differences avoid cancellation under translation.
                squared = (queries[:, None] - sources[None]).square().sum(-1)
                within = squared < r * r
                rank = within.long().cumsum(dim=1)
                accepted = within & (rank <= (max_num_neighbors - used)[:, None])
                used = used + accepted.sum(dim=1)
                query, source = accepted.nonzero(as_tuple=True)
                if query.numel():
                    edge_index_list.append(torch.stack(
                        [query_indices[query], idx_x[a + source]], dim=0))
    if not edge_index_list:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    edges = torch.cat(edge_index_list, dim=1)
    # Restore query-major/source-major ordering across source blocks.
    return edges[:, torch.argsort(edges[0], stable=True)]


def _radius_graph_pytorch(x, r, batch=None, loop=False, max_num_neighbors=32,
                          flow='source_to_target', num_workers=1, batch_size=None,
                          max_radius=None):
    """Match torch_cluster 1.6.3's radius-plus-self-filter wrapper.

    Its no-loop wrapper requests cap+1 before filtering self edges. If a
    saturated scan excludes self, cap+1 edges can remain, as in that version.
    """
    if flow not in ('source_to_target', 'target_to_source'):
        raise ValueError(f'unsupported flow: {flow}')
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    if max_num_neighbors <= 0:
        if max_num_neighbors < 0:
            raise ValueError('max_num_neighbors must be non-negative')
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    ei = _radius_pytorch(x, x, r, batch, batch,
                         max_num_neighbors if loop else max_num_neighbors + 1,
                         num_workers, batch_size)
    if flow == 'source_to_target':
        row, col = ei[1], ei[0]  # swap: message flows neighbor→center
    else:
        row, col = ei[0], ei[1]
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)


def _knn_graph_pytorch(x, k, batch=None, loop=False, flow='source_to_target',
                       cosine=False, num_workers=1):
    """Replacement for torch_cluster.knn_graph."""
    if flow not in ('source_to_target', 'target_to_source'):
        raise ValueError(f'unsupported flow: {flow}')
    if k < 0:
        raise ValueError('k must be non-negative')
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    ei_list = []
    for b in batch.unique():
        mask = (batch == b)
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        xi = x[mask]
        if cosine:
            normalized = torch.nn.functional.normalize(xi, dim=-1)
            dist = 1 - normalized @ normalized.T
        else:
            # The matrix-multiplication distance formula loses close-neighbor
            # ordering when PDB coordinates have a large common translation.
            dist = torch.cdist(xi, xi, compute_mode='donot_use_mm_for_euclid_dist')
        if not loop:
            dist.diagonal().fill_(float('inf'))
        kk = min(k, xi.shape[0] if loop else max(0, xi.shape[0] - 1))
        if not kk:
            continue
        _, topk = dist.topk(kk, dim=1, largest=False)
        rows = torch.arange(xi.shape[0], device=x.device).unsqueeze(1).expand(-1, kk).reshape(-1)
        cols = topk.reshape(-1)
        valid = cols < xi.shape[0]
        rows, cols = rows[valid], cols[valid]
        # rows are query/center nodes; cols are their nearest neighbors.
        # torch_cluster's default sends messages from neighbor to center.
        source, target = (cols, rows) if flow == 'source_to_target' else (rows, cols)
        ei_list.append(torch.stack([idx[source], idx[target]], dim=0))
    if not ei_list:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(ei_list, dim=1).long()


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
        # Reshape count to [dim_size, 1, 1, ...] matching out's trailing dims so
        # the division broadcasts elementwise along dim 0. A bare
        # count.unsqueeze(-1) is [dim_size, 1]; for a 1-D src (out == [dim_size])
        # that broadcasts to [dim_size, dim_size] instead of [dim_size].
        count = count.clamp(min=1).view(dim_size, *([1] * (src.dim() - 1)))
        return out / count
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


def _scatter_sum_pytorch(src, index, dim_size=None, dim=0):
    return _scatter_pytorch(src, index, dim_size, dim, reduce='sum')


def apply_patches():
    """Patch torch_cluster + torch_scatter with pure-PyTorch fallbacks.

    Idempotent. MUST be called before `models.cg_model` is imported.
    """
    import sys
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    try:
        import torch_cluster
    except ModuleNotFoundError:
        torch_cluster = ModuleType('torch_cluster')
        torch_cluster.__spec__ = ModuleSpec('torch_cluster', loader=None)
        sys.modules['torch_cluster'] = torch_cluster
    try:
        import torch_scatter
    except ModuleNotFoundError:
        torch_scatter = ModuleType('torch_scatter')
        torch_scatter.__spec__ = ModuleSpec('torch_scatter', loader=None)
        sys.modules['torch_scatter'] = torch_scatter
    torch_cluster.radius = _radius_pytorch
    torch_cluster.radius_graph = _radius_graph_pytorch
    torch_cluster.knn_graph = _knn_graph_pytorch
    torch_scatter.scatter = _scatter_pytorch
    torch_scatter.scatter_mean = _scatter_mean_pytorch
    torch_scatter.scatter_sum = _scatter_sum_pytorch
    return True


# Apply on import (idempotent). This guarantees the patch is in place the
# moment the package is imported, before any model binds DiffDock kernels.
apply_patches()

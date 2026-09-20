"""Exact listwise gradients with bounded per-structure activation memory."""

import torch
from torch import nn


def backward_list(model, case, device, loss_fn, *, chunk_size=4, weight=1.):
    """Differentiate one complete list, then replay score Jacobians in chunks.

    No optimizer step may occur between the two passes. This is not minibatch
    ranking: all candidates compete in one loss. Only deterministic pointwise
    encoders are supported; candidate-set interaction needs a different method.
    """
    if type(chunk_size) is not int or chunk_size < 1 or not case['graphs']:
        raise ValueError('positive chunk size and nonempty case required')
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) or (
                isinstance(module, nn.modules.dropout._DropoutNd) and module.training and module.p):
            raise ValueError('gradient replay requires deterministic stateless forwards')

    def forward(start):
        end = start + chunk_size
        graphs = [{k: v.to(device) for k, v in graph.items()} for graph in case['graphs'][start:end]]
        return model.forward_list(graphs, stages=case['stages'][start:end],
                                  sources=case['sources'][start:end], task=case.get('ranking_task', 'quality'))

    with torch.no_grad():
        pieces = [forward(start) for start in range(0, len(case['graphs']), chunk_size)]
        values = {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]}
    leaves = {key: value.detach().requires_grad_() for key, value in values.items()}
    loss = loss_fn(leaves)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite complete-list loss')
    gradients = torch.autograd.grad(weight * loss, tuple(leaves.values()), allow_unused=True)
    active = [(key, grad) for key, grad in zip(leaves, gradients) if grad is not None]
    for start in range(0, len(case['graphs']), chunk_size):
        output = forward(start)
        for key, _ in active:
            if not torch.allclose(output[key].detach(), values[key][start:start + chunk_size], atol=1e-6, rtol=1e-5):
                raise ValueError('forward changed during gradient replay; do not use stochastic/stateful encoders')
        torch.autograd.backward([output[key] for key, _ in active],
                                [grad[start:start + chunk_size] for _, grad in active])
    return loss.detach(), values

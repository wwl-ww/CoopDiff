"""Quality-supervised readout of the existing ternary geometry encoder.

One unbatched pose per forward, with explicit stage and actual source. Complete
case lists receive ordering supervision. No generator forward is needed.
"""

import copy
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .diffdock.constants import lig_feature_dims
from .geometry_encoders import TernaryGeometryEncoder
from .ranking_interface import InterfaceMessagePassing

STAGES = ('initial', 'final')
SOURCES = ('sdpi', 'bitseed')


class StructuralQualityRanker(nn.Module):
    """Independent trainable structural encoder and common state-quality head."""

    def __init__(self, input_dim, hidden_dim=128, *, encoder=None, interface_layers=0, component_aux=False,
                 interface_readout='pooled'):
        super().__init__()
        if input_dim < len(lig_feature_dims[0]) or hidden_dim < 4:
            raise ValueError('invalid structural ranker dimensions')
        if encoder is None:
            self.encoder = TernaryGeometryEncoder(input_dim, hidden_dim)
        else:
            if (not isinstance(encoder, TernaryGeometryEncoder)
                    or encoder.hidden_dim != hidden_dim
                    or encoder.protein_feature_proj.in_features != input_dim):
                raise ValueError('incompatible structural encoder')
            # Never share parameter storage or requires_grad flags with refiner.
            self.encoder = copy.deepcopy(encoder)
        self.encoder.requires_grad_(True)
        if type(interface_layers) is not int or interface_layers < 0:
            raise ValueError('interface_layers must be a nonnegative integer')
        self.input_dim = input_dim
        self.stage_embedding = nn.Embedding(2, hidden_dim)
        self.source_embedding = nn.Embedding(2, hidden_dim)
        self.atom_attention = nn.Linear(hidden_dim, 1, bias=False)
        self.readout = nn.Sequential(
            nn.LayerNorm(6 * hidden_dim), nn.Linear(6 * hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.quality_head = nn.Linear(hidden_dim, 6)
        self.rank_head = nn.Linear(hidden_dim, 1)
        self.future_head = nn.Linear(hidden_dim, 1)
        # Three quality intervals. Their probabilities and conditional means
        # define one distribution's threshold probabilities and expected score.
        self.register_buffer('lower', torch.tensor([0., .23, .49]))
        self.register_buffer('upper', torch.tensor([.23, .49, 1.]))
        # Instantiate after baseline heads so a fixed seed preserves their
        # initialization in the architecture ablation.
        if interface_readout not in ('pooled', 'contacts', 'contacts-independent') or (not interface_layers and interface_readout != 'pooled'):
            raise ValueError('contact readout requires interface layers')
        self.interface = InterfaceMessagePassing(hidden_dim, interface_layers, readout=interface_readout) if interface_layers else None
        self.interface_gate = nn.Parameter(torch.tensor(-2.1972246)) if interface_layers else None
        # Added last so existing encoder/ranking initialization stays unchanged.
        self.component_head = nn.Linear(hidden_dim, 3) if component_aux else None

    def forward(self, *, node_features, node_coords, node_types, lig_edge_index,
                stage, source, task='quality'):
        if stage not in STAGES or source not in SOURCES or task not in ('quality', 'future'):
            raise ValueError('explicit stage, actual source and ranking task required')
        if task == 'future' and stage != 'initial':
            raise ValueError('future ranking accepts initial states only')
        if (node_features.ndim != 2 or node_features.shape[1] != self.input_dim
                or not node_features.is_floating_point()
                or node_coords.shape != (len(node_features), 3)
                or not node_coords.is_floating_point()
                or node_types.shape != (len(node_features),)
                or node_types.dtype != torch.long
                or not torch.isfinite(node_features).all()
                or not torch.isfinite(node_coords).all()
                or not bool(((node_types >= 0) & (node_types <= 2)).all())
                or any(not bool((node_types == i).any()) for i in range(3))):
            raise ValueError('invalid unbatched ternary ranker input')
        atoms = node_features[node_types == 2, :len(lig_feature_dims[0])]
        limits = atoms.new_tensor(lig_feature_dims[0])
        if (not torch.equal(atoms, atoms.round())
                or not bool(((atoms >= 0) & (atoms < limits)).all())):
            raise ValueError('invalid categorical ligand features')
        if (lig_edge_index.dtype != torch.long or lig_edge_index.ndim != 2
                or lig_edge_index.shape[0] != 2
                or bool((lig_edge_index < 0).any())
                or bool((lig_edge_index >= len(atoms)).any())):
            raise ValueError('invalid local ligand bond indices')
        if len({x.device for x in (node_features, node_coords, node_types, lig_edge_index)}) != 1:
            raise ValueError('ranker tensors must share a device')
        pooled = self.encoder(
            node_features.detach(), node_coords.detach(), node_types,
            lig_edge_index=lig_edge_index,
            return_node_features=self.interface is not None,
        )
        global_context = pooled['global']
        if self.interface is not None:
            inputs = (pooled['node_features'], node_coords.detach(), node_types, lig_edge_index)
            # Full-case list supervision retains hundreds of graphs until backward.
            # Recompute local messages instead of retaining every edge activation.
            interface_context = (checkpoint(self.interface, *inputs, use_reentrant=False)
                                 if self.training and torch.is_grad_enabled()
                                 else self.interface(*inputs))
            global_context = global_context + torch.sigmoid(self.interface_gate) * interface_context
        atoms = pooled['ligand_tokens'][0]
        weights = torch.softmax(self.atom_attention(atoms), dim=0)
        atom_context = (weights * atoms).sum(0, keepdim=True)
        context = self.readout(torch.cat(
            (global_context, pooled['e3l'], pooled['pol'], atom_context,
             self.stage_embedding.weight[STAGES.index(stage)][None],
             self.source_embedding.weight[SOURCES.index(source)][None]), -1))
        score = (self.rank_head if task == 'quality' else self.future_head)(context).squeeze(-1)
        if task == 'future':
            return {'score': score}
        logits = self.quality_head(context)
        probabilities = torch.softmax(logits[:, :3], dim=-1)
        means = self.lower + (self.upper - self.lower) * torch.sigmoid(logits[:, 3:])
        quality = (probabilities * means).sum(-1)
        result = {'score': score, 'q_hat': quality,
                'bin_logits': logits[:, :3], 'bin_probabilities': probabilities,
                'bin_means': means, 'p23': probabilities[:, 1:].sum(-1),
                'p49': probabilities[:, 2]}
        if self.component_head is not None:
            result['components'] = torch.sigmoid(self.component_head(context))
        return result

    def forward_list(self, graphs, *, stages, sources, task='quality'):
        """Score one case's valid structures without detaching rank gradients.

        Caller supplies one case only, already excluding unavailable states.
        No labels enter this method. Future mode cannot consume final graphs.
        """
        if not graphs or len(graphs) != len(stages) or len(graphs) != len(sources):
            raise ValueError('nonempty aligned case lists required')
        outputs = [self(**graph, stage=stage, source=source, task=task)
                   for graph, stage, source in zip(graphs, stages, sources)]
        return {key: torch.cat([out[key] for out in outputs], dim=0)
                for key in outputs[0]}


def _validate_ranking_inputs(scores, labels, tie_tolerance):
    if (scores.ndim != 1 or labels.shape != scores.shape or not scores.numel()
            or scores.device != labels.device or not torch.isfinite(scores).all()
            or not torch.isfinite(labels).all()
            or not bool(((labels >= 0) & (labels <= 1)).all())
            or not math.isfinite(tie_tolerance) or tie_tolerance < 0):
        raise ValueError('invalid full-list ranking inputs')


def full_list_ranking_loss(scores, labels, *, tie_tolerance=1e-4):
    """Tie-aware suffix likelihood, using every ordered level of one case.

    Each item competes against every strictly lower relevance tier. Peers in
    the same tier never compete with each other. With unique labels this is
    ListMLE averaged over nontrivial suffixes. Tiers are formed by descending
    label with a fixed tolerance relative to each tier's highest label.
    """
    _validate_ranking_inputs(scores, labels, tie_tolerance)
    order = torch.argsort(labels.detach(), descending=True, stable=True)
    truth = labels.detach()[order].cpu().tolist()
    ordered = scores[order]
    tier_ends, begin = [], 0
    while begin < len(truth):
        end = begin + 1
        while end < len(truth) and truth[begin] - truth[end] <= tie_tolerance:
            end += 1
        if end < len(truth):
            tier_ends.extend([end] * (end - begin))
        begin = end
    if not tier_ends:
        return scores.sum() * 0
    # Every tier reuses the same suffix normalizers. One scan replaces the
    # quadratic repeated reductions without changing tie-aware supervision.
    suffix = torch.logcumsumexp(ordered.flip(0), dim=0).flip(0)
    ends = torch.tensor(tier_ends, dtype=torch.long, device=scores.device)
    return nn.functional.softplus(suffix[ends] - ordered[:len(tier_ends)]).mean()


def best_tier_loss(scores, labels, *, tie_tolerance=1e-4):
    """Negative log probability mass of the best label tier, not one tie winner.

    This optional auxiliary emphasizes the first decision without replacing
    full-list supervision. It is ordinal, not a calibrated quality estimate.
    """
    _validate_ranking_inputs(scores, labels, tie_tolerance)
    truth = labels.detach()
    best = truth.max() - truth <= tie_tolerance
    if bool(best.all()):
        return scores.sum() * 0
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[best], dim=0)


def budget_retention_loss(scores, labels, *, keep=64, tie_tolerance=1e-4):
    """Surrogate for retaining at least one best-tier item under top-k selection.

    A best-tier item must beat the kth non-best score to enter the budget.
    Unlike best_tier_loss, this need not make it top-1. Requires complete outcome
    supervision at the caller: missing candidates cannot be treated as negatives.
    Max/topk are piecewise differentiable; this is not an expected-oracle loss.
    """
    _validate_ranking_inputs(scores, labels, tie_tolerance)
    if type(keep) is not int or keep < 1:
        raise ValueError('positive integer selection budget required')
    best = labels.max() - labels.detach() <= tie_tolerance
    competitors = scores[~best]
    if len(competitors) < keep:
        return scores.sum() * 0
    boundary = competitors.topk(keep).values[-1]
    return nn.functional.softplus(boundary - scores[best].amax())


def structural_ranking_loss(output, labels, *, quality_weight=0., top_weight=0., tie_tolerance=1e-4):
    """Primary full-list ordering, with optional current-quality auxiliary loss.

    Average this scalar across cases externally; never concatenate unrelated
    complexes into one ranked list. Future labels are refined utility and must
    not be applied to the current-quality auxiliary head.
    """
    if any(not math.isfinite(w) or w < 0 for w in (quality_weight, top_weight)):
        raise ValueError('auxiliary weights must be finite and nonnegative')
    ranking = full_list_ranking_loss(output['score'], labels, tie_tolerance=tie_tolerance)
    if top_weight:
        ranking = ranking + top_weight * best_tier_loss(output['score'], labels,
                                                       tie_tolerance=tie_tolerance)
    if quality_weight:
        if 'bin_logits' not in output:
            raise ValueError('future output has no current-quality auxiliary head')
        ranking = ranking + quality_weight * structural_quality_loss(output, labels)
    return ranking


def dockq_regression_loss(output, labels, **kwargs):
    """Same scalar head, sigmoid-MSE supervision; no listwise or auxiliary loss."""
    if kwargs.get('quality_weight', 0) or kwargs.get('top_weight', 0):
        raise ValueError('pure regression requires zero quality/top weights')
    score = output['score']
    if (score.ndim != 1 or labels.shape != score.shape or not labels.numel()
            or not torch.isfinite(labels).all() or not torch.isfinite(score).all()
            or not bool(((labels >= 0) & (labels <= 1)).all())):
        raise ValueError('finite scalar scores and bounded DockQ targets required')
    return nn.functional.mse_loss(torch.sigmoid(score), labels.detach())


def component_quality_loss(output, targets):
    """Supervise fnat and bounded iRMS/LRMS quality, not a second ranking label.

    Components are the three terms whose mean is DockQ. The deployed ordering
    still uses rank_head; this auxiliary must not silently replace its score.
    """
    if ('components' not in output or targets.shape != output['components'].shape
            or targets.ndim != 2 or targets.shape[1] != 3 or not targets.numel()
            or not torch.isfinite(targets).all() or not bool(((targets >= 0) & (targets <= 1)).all())):
        raise ValueError('complete bounded DockQ component targets required')
    return nn.functional.mse_loss(output['components'], targets.detach())


def selected_walker_mask(scores, initial_identities, identities, *, keep=64):
    """Native-free hard Top-K allocation, with canonical pool-index tie breaks.

    Only initial predictions enter allocation. Returned mask covers both ends
    of selected trajectories and does not propagate gradients through Top-K.
    """
    if (type(keep) is not int or keep < 1 or scores.ndim != 1 or len(scores) < keep
            or len(initial_identities) != len(scores) or not torch.isfinite(scores).all()
            or any(stage != 'initial' for _, stage in initial_identities)):
        raise ValueError('finite aligned initial predictions and valid budget required')
    ids = [index for index, _ in initial_identities]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate initial candidate identity')
    canonical = sorted(range(len(ids)), key=lambda i: ids[i])
    order = torch.argsort(scores.detach().cpu()[canonical], descending=True, stable=True)
    selected = {ids[canonical[i]] for i in order[:keep].tolist()}
    return torch.tensor([index in selected for index, _ in identities], dtype=torch.bool)


def decision_ranking_loss(output, labels, stages, union_eligible=None, final_eligible=None, **kwargs):
    """Equal-weight full-list objectives on actual decision sets, not their union.

    Initial ordering uses all initials; Final uses observed finals; Union uses
    only eligible walker states. Each scope retains full-list supervision.
    """
    if len(stages) != labels.numel() or any(s not in STAGES for s in stages):
        raise ValueError('aligned initial/final stages required')
    initial = torch.tensor([s == 'initial' for s in stages], device=labels.device)
    masks = [initial, ~initial]
    if final_eligible is not None:
        final_mask = final_eligible.to(labels.device)
        if final_mask.shape != labels.shape or final_mask.dtype != torch.bool:
            raise ValueError('boolean aligned Final eligibility required')
        masks[1] = masks[1] & final_mask
    if bool((~initial).any()):
        eligible = torch.ones_like(initial) if union_eligible is None else union_eligible.to(labels.device)
        if eligible.shape != labels.shape or eligible.dtype != torch.bool:
            raise ValueError('boolean aligned Union eligibility required')
        masks.append(eligible)
    losses = []
    for mask in masks:
        if bool(mask.any()):
            subset = {key: value[mask] for key, value in output.items()}
            losses.append(structural_ranking_loss(subset, labels[mask], **kwargs))
    return torch.stack(losses).mean()


def structural_quality_loss(output, dockq):
    """Interval probability plus conditional mean supervision.

    This is single-state quality supervision, not a claim of calibration or a
    complete casewise training objective. Case/stage weighting belongs to the
    future trainer. Targets exactly at .23/.49 belong to the upper interval.
    """
    if (dockq.shape != output['score'].shape or not dockq.numel()
            or not torch.isfinite(dockq).all()
            or not bool(((dockq >= 0) & (dockq <= 1)).all())):
        raise ValueError('DockQ must be a matching nonempty vector in [0,1]')
    labels = dockq.detach()
    bins = (labels >= .23).long() + (labels >= .49).long()
    selected_mean = output['bin_means'].gather(-1, bins[:, None]).squeeze(-1)
    return (nn.functional.cross_entropy(output['bin_logits'], bins)
            + nn.functional.mse_loss(selected_mean, labels))


def rank_states(scores, stages, valid, *, scope='union', top_k=None):
    """Return all eligible indices in rank order, optionally truncated to top-k.

    Initial/final arrays share exactly the same scorer. Ties keep input order;
    stable candidate identity ordering is a caller responsibility.
    """
    if (scores.ndim != 1 or len(stages) != scores.numel()
            or valid.shape != scores.shape or valid.dtype != torch.bool
            or valid.device != scores.device or scope not in ('initial', 'final', 'union')
            or any(stage not in ('initial', 'final') for stage in stages)):
        raise ValueError('invalid state selection inputs')
    eligible = valid & torch.tensor(
        [scope == 'union' or stage == scope for stage in stages], device=scores.device)
    ids = eligible.nonzero().flatten()
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError('top_k must be a positive integer')
    if not torch.isfinite(scores[ids]).all():
        raise ValueError('eligible state scores must be finite')
    ordered = ids[torch.argsort(scores[ids], descending=True, stable=True)]
    return ordered if top_k is None else ordered[:top_k]


def select_state(scores, stages, valid, *, scope='union'):
    order = rank_states(scores, stages, valid, scope=scope, top_k=1)
    return int(order[0]) if order.numel() else None

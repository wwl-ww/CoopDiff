"""Permutation-equivariant ranking over all candidates from one complex."""

import torch
from torch import nn

from .pose_ranking import listwise_ranking_loss


SOURCE_NAMES = ('bitseed', 'sdpi')


def source_indices(sources, *, device=None):
    """Encode actual candidate sources without depending on pool ordering."""
    lookup = {name: index for index, name in enumerate(SOURCE_NAMES)}
    try:
        values = [lookup[source] for source in sources]
    except (KeyError, TypeError) as error:
        raise ValueError('candidate sources must be bitseed or sdpi') from error
    return torch.tensor(values, dtype=torch.long, device=device)


def _batched_inputs(features, sources, mask):
    squeeze = features.ndim == 2
    if squeeze:
        features = features.unsqueeze(0)
        sources = sources.unsqueeze(0)
        mask = None if mask is None else mask.unsqueeze(0)
    if features.ndim != 3 or sources.shape != features.shape[:2]:
        raise ValueError('setwise features/sources must be [B,N,D] and [B,N]')
    if not torch.is_floating_point(features) or not torch.isfinite(features).all():
        raise ValueError('setwise features must be finite floating point values')
    if sources.dtype != torch.long or not bool(((sources >= 0) & (sources < len(SOURCE_NAMES))).all()):
        raise ValueError('setwise sources must contain valid integer source IDs')
    if mask is None:
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
    if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('setwise mask must be boolean [B,N]')
    if not bool(mask.any(dim=1).all()):
        raise ValueError('each candidate set must contain at least one valid item')
    return features, sources, mask, squeeze


class SetwiseCandidateRanker(nn.Module):
    """Rank candidates relative to their complete case-level candidate set.

    No positional encoding is used, so reordering candidates only reorders the
    outputs. Source identity is explicit rather than inferred from pool index.
    """

    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2,
                 feedforward_dim=128, dropout=0.):
        super().__init__()
        if (input_dim < 1 or model_dim < 4 or num_heads < 1 or num_layers < 1
                or model_dim % num_heads or feedforward_dim < model_dim
                or not 0 <= dropout < 1):
            raise ValueError('invalid setwise ranker dimensions')
        self.config = {
            'input_dim': input_dim, 'model_dim': model_dim,
            'num_heads': num_heads, 'num_layers': num_layers,
            'feedforward_dim': feedforward_dim, 'dropout': float(dropout),
        }
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.source_embedding = nn.Embedding(len(SOURCE_NAMES), model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        # The pooled case token supplies an explicit global reference in
        # addition to token-to-token attention. Four outputs are: relative
        # score, absolute quality logit, >=.23 logit, and >=.49|>=.23 logit.
        self.head = nn.Sequential(
            nn.LayerNorm(2 * model_dim), nn.Linear(2 * model_dim, model_dim),
            nn.SiLU(), nn.Linear(model_dim, 4),
        )

    def forward(self, features, sources, mask=None):
        features, sources, mask, squeeze = _batched_inputs(features, sources, mask)
        if features.shape[-1] != self.config['input_dim']:
            raise ValueError('setwise feature dimension does not match the model')
        tokens = self.feature_encoder(features.detach()) + self.source_embedding(sources)
        context = self.set_encoder(tokens, src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).to(context.dtype)
        pooled = (context * weights).sum(1) / weights.sum(1).clamp_min(1.)
        logits = self.head(torch.cat((context, pooled.unsqueeze(1).expand_as(context)), -1))
        p23 = torch.sigmoid(logits[..., 2])
        result = {
            'score': logits[..., 0], 'q_hat': torch.sigmoid(logits[..., 1]),
            'p23': p23, 'p49': p23 * torch.sigmoid(logits[..., 3]),
            'threshold_logits': logits[..., 2:4], 'mask': mask,
        }
        if squeeze:
            result = {key: value.squeeze(0) for key, value in result.items()}
        return result


class TrajectorySetRanker(nn.Module):
    """Post-rank paired initial/refined states in the context of all walkers.

    Unlike a state-independent MLP, this model sees the refinement displacement,
    the initial fallback, actual source, failed trajectories, and competing
    walkers simultaneously. It ranks final states and initial/final unions with
    separate heads because those deployment decisions are not interchangeable.
    """

    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2,
                 feedforward_dim=128, dropout=0.):
        super().__init__()
        if (input_dim < 1 or model_dim < 4 or num_heads < 1 or num_layers < 1
                or model_dim % num_heads or feedforward_dim < model_dim
                or not 0 <= dropout < 1):
            raise ValueError('invalid trajectory ranker dimensions')
        self.config = {
            'input_dim': input_dim, 'model_dim': model_dim,
            'num_heads': num_heads, 'num_layers': num_layers,
            'feedforward_dim': feedforward_dim, 'dropout': float(dropout),
        }
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(4 * model_dim), nn.Linear(4 * model_dim, model_dim),
            nn.SiLU(), nn.Linear(model_dim, model_dim),
        )
        self.source_embedding = nn.Embedding(len(SOURCE_NAMES), model_dim)
        self.valid_embedding = nn.Embedding(2, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        # final/union score, final/union q, two ordered logits for each stage,
        # and the probability that union should return the refined state.
        self.head = nn.Sequential(
            nn.LayerNorm(2 * model_dim), nn.Linear(2 * model_dim, model_dim),
            nn.SiLU(), nn.Linear(model_dim, 9),
        )

    def forward(self, initial, final, sources, final_mask, candidate_mask=None):
        initial, sources, candidate_mask, squeeze = _batched_inputs(
            initial, sources, candidate_mask)
        if squeeze:
            final = final.unsqueeze(0)
            final_mask = final_mask.unsqueeze(0)
        if (final.shape != initial.shape or final_mask.shape != initial.shape[:2]
                or final_mask.dtype != torch.bool
                or not torch.is_floating_point(final) or not torch.isfinite(final).all()
                or initial.shape[-1] != self.config['input_dim']
                or bool((final_mask & ~candidate_mask).any())):
            raise ValueError('invalid paired trajectory inputs')
        # A failed refinement carries no hidden information from its placeholder
        # tensor. Its effective final state is the deployable initial fallback.
        effective_final = torch.where(final_mask.unsqueeze(-1), final, initial)
        initial_token = self.state_encoder(initial.detach())
        final_token = self.state_encoder(effective_final.detach())
        difference = final_token - initial_token
        pair = self.pair_encoder(torch.cat(
            (initial_token, final_token, difference, difference.abs()), -1))
        pair = (pair + self.source_embedding(sources)
                + self.valid_embedding(final_mask.long()))
        context = self.set_encoder(pair, src_key_padding_mask=~candidate_mask)
        weights = candidate_mask.unsqueeze(-1).to(context.dtype)
        pooled = (context * weights).sum(1) / weights.sum(1).clamp_min(1.)
        logits = self.head(torch.cat((context, pooled.unsqueeze(1).expand_as(context)), -1))
        final_p23, union_p23 = torch.sigmoid(logits[..., 4]), torch.sigmoid(logits[..., 6])
        result = {
            'final_score': logits[..., 0], 'union_score': logits[..., 1],
            'final_q_hat': torch.sigmoid(logits[..., 2]),
            'union_q_hat': torch.sigmoid(logits[..., 3]),
            'final_p23': final_p23,
            'final_p49': final_p23 * torch.sigmoid(logits[..., 5]),
            'union_p23': union_p23,
            'union_p49': union_p23 * torch.sigmoid(logits[..., 7]),
            'final_threshold_logits': logits[..., 4:6],
            'union_threshold_logits': logits[..., 6:8],
            'choose_final_probability': torch.sigmoid(logits[..., 8]),
            'choose_final_logit': logits[..., 8],
            'final_mask': final_mask, 'candidate_mask': candidate_mask,
        }
        if squeeze:
            result = {key: value.squeeze(0) for key, value in result.items()}
        return result


def setwise_quality_loss(output, dockq, *, supervision_mask=None, ranking_weight=1.,
                         regression_weight=.5, threshold_weight=.5,
                         temperature=.05):
    """Case-balanced future/current quality loss for variable candidate sets."""
    scores, mask = output['score'], output['mask']
    squeeze = scores.ndim == 1
    if squeeze:
        scores, mask, dockq = scores.unsqueeze(0), mask.unsqueeze(0), dockq.unsqueeze(0)
        q_hat = output['q_hat'].unsqueeze(0)
        threshold_logits = output['threshold_logits'].unsqueeze(0)
    else:
        q_hat, threshold_logits = output['q_hat'], output['threshold_logits']
    if supervision_mask is None:
        supervision_mask = mask
    elif squeeze and supervision_mask.ndim == 1:
        supervision_mask = supervision_mask.unsqueeze(0)
    if (dockq.shape != scores.shape or mask.shape != scores.shape
            or supervision_mask.shape != mask.shape or supervision_mask.dtype != torch.bool
            or bool((supervision_mask & ~mask).any())
            or not bool(supervision_mask.any(dim=1).all())
            or not torch.isfinite(dockq).all()
            or not bool(((dockq >= 0) & (dockq <= 1)).all())
            or min(ranking_weight, regression_weight, threshold_weight) < 0):
        raise ValueError('invalid setwise quality supervision')
    losses = []
    for batch in range(scores.shape[0]):
        keep = supervision_mask[batch]
        labels = dockq[batch, keep].detach()
        ranking = (scores[batch, keep].sum() * 0 if labels.numel() == 1 else
                   listwise_ranking_loss(scores[batch, keep], labels, temperature))
        regression = nn.functional.smooth_l1_loss(q_hat[batch, keep], labels)
        threshold_targets = torch.stack((labels >= .23, labels >= .49), -1).float()
        threshold = nn.functional.binary_cross_entropy_with_logits(
            threshold_logits[batch, keep], threshold_targets)
        losses.append(ranking_weight * ranking + regression_weight * regression
                      + threshold_weight * threshold)
    return torch.stack(losses).mean()


def stable_topk(scores, mask, keep):
    """Select valid candidates only; exact ties retain serialized order."""
    scores, mask = torch.as_tensor(scores), torch.as_tensor(mask)
    if (scores.ndim != 1 or mask.shape != scores.shape or mask.dtype != torch.bool
            or not torch.isfinite(scores[mask]).all() or not 1 <= keep <= int(mask.sum())):
        raise ValueError('invalid masked top-k inputs')
    valid = torch.where(mask)[0]
    return valid[torch.argsort(scores[valid], descending=True, stable=True)[:keep]]


def trajectory_ranking_loss(output, initial_dockq, final_dockq, *, temperature=.05):
    """Supervise final ranking, union ranking, calibration, and fallback stage."""
    squeeze = output['union_score'].ndim == 1
    if squeeze:
        output = {key: value.unsqueeze(0) for key, value in output.items()}
        initial_dockq, final_dockq = initial_dockq.unsqueeze(0), final_dockq.unsqueeze(0)
    candidate_mask, final_mask = output['candidate_mask'], output['final_mask']
    if (initial_dockq.shape != candidate_mask.shape or final_dockq.shape != candidate_mask.shape
            or not torch.isfinite(initial_dockq).all() or not torch.isfinite(final_dockq).all()
            or not bool(((initial_dockq >= 0) & (initial_dockq <= 1)).all())
            or not bool(((final_dockq >= 0) & (final_dockq <= 1)).all())):
        raise ValueError('invalid trajectory DockQ supervision')
    losses = []
    for batch in range(candidate_mask.shape[0]):
        candidates = candidate_mask[batch]
        finals = final_mask[batch] & candidates
        initial = initial_dockq[batch]
        final = final_dockq[batch]
        union = torch.where(finals, torch.maximum(initial, final), initial).detach()
        union_labels = union[candidates]
        union_ranking = (output['union_score'][batch, candidates].sum() * 0
                         if union_labels.numel() == 1 else listwise_ranking_loss(
                             output['union_score'][batch, candidates], union_labels, temperature))
        if int(finals.sum()) > 1:
            final_ranking = listwise_ranking_loss(
                output['final_score'][batch, finals], final[finals].detach(), temperature)
        else:
            final_ranking = output['final_score'][batch, candidates].sum() * 0
        final_regression = (nn.functional.smooth_l1_loss(
            output['final_q_hat'][batch, finals], final[finals].detach())
            if bool(finals.any()) else union_ranking * 0)
        union_regression = nn.functional.smooth_l1_loss(
            output['union_q_hat'][batch, candidates], union_labels)
        union_thresholds = torch.stack(
            (union_labels >= .23, union_labels >= .49), -1).float()
        threshold = nn.functional.binary_cross_entropy_with_logits(
            output['union_threshold_logits'][batch, candidates], union_thresholds)
        choose_final = (finals & (final > initial))[candidates].float()
        stage = nn.functional.binary_cross_entropy_with_logits(
            output['choose_final_logit'][batch, candidates], choose_final)
        losses.append(union_ranking + final_ranking + .25 * (
            final_regression + union_regression + threshold + stage))
    return torch.stack(losses).mean()

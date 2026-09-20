"""Replay compute allocation using initial scores and observed paired outcomes."""

import torch


def expected_random_winner(values, priority, keep):
    """Exact outcome of highest-priority item in a uniform k-subset.

    Priority ties use input identity order. This is an evaluation control,
    not a model prediction or an oracle selection policy.
    """
    count = len(values)
    keep = min(keep, count)
    order = torch.argsort(priority, descending=True, stable=True)
    probability, expectation = keep / count, 0.
    for rank in range(count - keep + 1):
        expectation += probability * float(values[order[rank]])
        if rank < count - keep:
            probability *= (count - rank - keep) / (count - rank - 1)
    return expectation


def refinement_selection_metrics(scores, labels, identities, sources, budgets=(8, 32, 64),
                                 selection_scores=None):
    """Report only observed valid pairs; never impute unrefined/failed outcomes.

    Labels are used exclusively for measurement after score-based decisions.
    Initial scores choose walkers; the common quality score chooses the final
    or Union output among those walkers. Pair identity, not array order, joins
    initial/final states. Partial coverage is explicit, not a full-pool claim.
    """
    scores, labels = scores.detach().cpu(), labels.detach().cpu()
    selection_scores = scores if selection_scores is None else selection_scores.detach().cpu()
    if (scores.ndim != 1 or labels.shape != scores.shape or len(identities) != len(scores)
            or len(sources) != len(scores) or not torch.isfinite(scores).all()
            or not torch.isfinite(labels).all() or not bool(((labels >= 0) & (labels <= 1)).all())
            or selection_scores.shape != scores.shape or not torch.isfinite(selection_scores).all()
            or any(type(k) is not int or k < 1 for k in budgets)):
        raise ValueError('invalid selection replay inputs')
    mapping = {}
    for i, (identity, stage) in enumerate(identities):
        if stage not in ('initial', 'final') or sources[i] not in ('sdpi', 'bitseed'):
            raise ValueError('invalid stage/source')
        if (identity, stage) in mapping:
            raise ValueError('duplicate state identity')
        mapping[identity, stage] = i
    pairs = []
    for (identity, stage), final in mapping.items():
        if stage != 'final':
            continue
        initial = mapping.get((identity, 'initial'))
        if initial is None or sources[initial] != sources[final]:
            raise ValueError('final state lacks a same-source initial partner')
        pairs.append((identity, initial, final))
    pairs.sort(key=lambda item: item[0])
    initial_count = sum(stage == 'initial' for _, stage in mapping)
    result = {'scope': 'observed_valid_pairs_only', 'initial_states': initial_count,
              'paired_states': len(pairs), 'unpaired_initials': initial_count - len(pairs),
              'complete_pair_coverage': initial_count == len(pairs) and initial_count > 0,
              'budgets': {}}
    if not pairs:
        return result
    initial = torch.tensor([item[1] for item in pairs])
    final = torch.tensor([item[2] for item in pairs])
    order = torch.argsort(selection_scores[initial], descending=True, stable=True)
    all_union = torch.stack((initial, final), dim=1).flatten()
    final_oracle = float(labels[final].max())
    union_oracle = float(labels[all_union].max())
    for budget in budgets:
        selected = order[:budget]
        selected_initial, selected_final = initial[selected], final[selected]
        # Terminal ties must not depend on which pre-selector produced the set.
        canonical = selected.sort().values
        canonical_final = final[canonical]
        union = torch.stack((initial[canonical], canonical_final), dim=1).flatten()
        best_final = canonical_final[scores[canonical_final].argmax()]
        best_union = union[scores[union].argmax()]
        pair_winner = torch.where(scores[initial] >= scores[final], initial, final)
        union_values = torch.maximum(labels[initial], labels[final])
        random_control = {
            'final_selected_best': expected_random_winner(labels[final], labels[final], budget),
            'union_selected_best': expected_random_winner(union_values, union_values, budget),
            'final_top1': expected_random_winner(labels[final], scores[final], budget),
            'union_top1': expected_random_winner(labels[pair_winner], scores[pair_winner], budget),
            'final_selected_hit_probability_0.23': expected_random_winner(
                (labels[final] >= .23).double(), labels[final], budget),
            'final_top1_hit_probability_0.23': expected_random_winner(
                (labels[final] >= .23).double(), scores[final], budget),
        }
        result['budgets'][str(budget)] = {
            'selected_pairs': len(selected),
            'selected_pool_ids': [pairs[i][0] for i in selected.tolist()],
            'source_counts': {source: sum(sources[i] == source for i in selected_initial.tolist())
                              for source in ('sdpi', 'bitseed')},
            'initial_selected_best': float(labels[selected_initial].max()),
            'final_selected_best': float(labels[selected_final].max()),
            'union_selected_best': float(labels[union].max()),
            'final_top1': float(labels[best_final]), 'union_top1': float(labels[best_union]),
            'paired_final_oracle': final_oracle, 'paired_union_oracle': union_oracle,
            'uniform_random_control': random_control,
        }
    return result

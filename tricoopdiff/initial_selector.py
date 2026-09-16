"""Full-pool initial-state features and compute-allocation retention metrics."""

import math

import torch

from .ranking_archive import geometry_features, state_hash


POOL_LABEL_VERSION = 'exact_initial_pool_labels_v1'


def initial_pool_features(archive):
    """Recompute the 14 native-free geometry features for every pool state."""
    if archive.get('protocol') != 'initial_final_pose_archive_v1':
        raise ValueError('unsupported source archive')
    e3 = archive['input_graphs']['e3_data']['coords'][0]
    rows, seen = [], set()
    for index, candidate in enumerate(archive['pool']):
        if candidate.get('source') not in ('bitseed', 'sdpi'):
            raise ValueError('initial pool has an unknown source')
        state = candidate.get('state')
        if candidate.get('state_sha256') != state_hash(state):
            raise ValueError('initial pool state checksum mismatch')
        candidate_id = tuple(candidate.get('candidate_id', ()))
        if not candidate_id or candidate_id in seen:
            raise ValueError('initial pool candidate IDs must be unique')
        seen.add(candidate_id)
        rows.append({
            'pool_index': index, 'candidate_id': list(candidate_id),
            'source': candidate['source'], 'state_sha256': candidate['state_sha256'],
            'geometry': geometry_features(e3, state['poi'], state['ligand']),
        })
    if len(rows) != 256:
        raise ValueError('first selector protocol requires the fixed 256-candidate pool')
    return rows


def diversity_aware_indices(scores, descriptors, keep=64, penalty=.0):
    """Greedily trade predicted quality against redundancy in a candidate set.

    Descriptors must already be native-free and scaled to comparable units.  An
    RBF similarity uses the within-case median non-zero distance as bandwidth,
    so ``penalty`` remains comparable across cases.  ``penalty=0`` is exactly
    stable top-k selection and is kept as an implementation control.
    """
    scores = torch.as_tensor(scores).float()
    descriptors = torch.as_tensor(descriptors, device=scores.device).float()
    if (scores.ndim != 1 or descriptors.ndim != 2
            or descriptors.shape[0] != scores.numel()
            or not 1 <= keep <= scores.numel() or penalty < 0
            or not torch.isfinite(scores).all()
            or not torch.isfinite(descriptors).all()):
        raise ValueError('invalid diversity-aware selection inputs')
    if penalty == 0:
        return torch.argsort(scores, descending=True, stable=True)[:keep]
    distances = torch.cdist(descriptors, descriptors)
    nonzero = distances[distances > 0]
    bandwidth = torch.median(nonzero) if nonzero.numel() else scores.new_tensor(1.)
    similarity = torch.exp(-.5 * (distances / bandwidth.clamp_min(1e-8)).square())
    selected = []
    available = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
    redundancy = torch.zeros_like(scores)
    for _ in range(keep):
        utility = scores - float(penalty) * redundancy
        utility = utility.masked_fill(~available, -torch.inf)
        index = int(torch.argmax(utility))
        selected.append(index)
        available[index] = False
        redundancy = torch.maximum(redundancy, similarity[index])
    return torch.tensor(selected, dtype=torch.long, device=scores.device)


def selector_retention_at_indices(indices, dockq, sources, threshold=.23):
    """Measure oracle retention for an explicitly selected candidate subset."""
    dockq = torch.as_tensor(dockq)
    indices = torch.as_tensor(indices, dtype=torch.long)
    if (dockq.ndim != 1 or indices.ndim != 1 or not indices.numel()
            or len(sources) != dockq.numel() or indices.unique().numel() != indices.numel()
            or int(indices.min()) < 0 or int(indices.max()) >= dockq.numel()
            or not torch.isfinite(dockq).all()
            or not bool(((dockq >= 0) & (dockq <= 1)).all())):
        raise ValueError('invalid explicit selector retention inputs')
    order = indices.cpu()
    selected = dockq[order.to(dockq.device)]
    oracle = float(dockq.max())
    return {
        'keep': order.numel(), 'selected_best': float(selected.max()),
        'full_pool_oracle': oracle,
        'oracle_retention': float(selected.max()) / oracle if oracle > 0 else None,
        'selected_hit': int(float(selected.max()) >= threshold),
        'full_pool_hit': int(oracle >= threshold),
        'oracle_index_retained': int(int(torch.argmax(dockq)) in set(order.tolist())),
        'selected_source_counts': {
            source: sum(sources[int(i)] == source for i in order)
            for source in ('bitseed', 'sdpi')},
    }


def selector_retention(scores, dockq, sources, keep=64, threshold=.23):
    """Measure whether stable top-score selection preserves useful basins."""
    scores, dockq = torch.as_tensor(scores), torch.as_tensor(dockq)
    if (scores.ndim != 1 or dockq.shape != scores.shape or len(sources) != scores.numel()
            or not 1 <= keep <= scores.numel() or not torch.isfinite(scores).all()
            or not torch.isfinite(dockq).all() or not bool(((dockq >= 0) & (dockq <= 1)).all())):
        raise ValueError('invalid selector retention inputs')
    order = torch.argsort(scores, descending=True, stable=True)[:keep]
    return selector_retention_at_indices(order, dockq, sources, threshold)


def future_selector_retention(scores, initial_dockq, final_dockq, sources,
                              keep=64, threshold=.23):
    """Audit whether a compute selector preserves candidates valuable after refinement."""
    scores = torch.as_tensor(scores)
    initial = torch.as_tensor(initial_dockq)
    final = torch.as_tensor(final_dockq)
    if (scores.ndim != 1 or initial.shape != scores.shape or final.shape != scores.shape
            or len(sources) != scores.numel() or not 1 <= keep <= scores.numel()
            or not torch.isfinite(scores).all() or not torch.isfinite(initial).all()
            or not torch.isfinite(final).all()
            or not bool(((initial >= 0) & (initial <= 1)).all())
            or not bool(((final >= 0) & (final <= 1)).all())):
        raise ValueError('invalid future selector retention inputs')
    order = torch.argsort(scores, descending=True, stable=True)[:keep]
    chosen = set(order.tolist())
    union = torch.maximum(initial, final)
    rescued = (initial < threshold) & (final >= threshold)

    def stage_metrics(values):
        selected_best, oracle = float(values[order].max()), float(values.max())
        return {
            'selected_best': selected_best,
            'full_pool_oracle': oracle,
            'oracle_retention': selected_best / oracle if oracle > 0 else None,
            'selected_hit': int(selected_best >= threshold),
            'full_pool_hit': int(oracle >= threshold),
            'oracle_index_retained': int(int(torch.argmax(values)) in chosen),
        }

    rescued_ids = torch.where(rescued)[0]
    selected_rescued = [int(index) for index in rescued_ids if int(index) in chosen]
    return {
        'keep': keep,
        'initial': stage_metrics(initial),
        'final': stage_metrics(final),
        'union': stage_metrics(union),
        'rescued_candidates': int(rescued.sum()),
        'rescued_candidates_retained': len(selected_rescued),
        'rescued_hit_retained': int(bool(selected_rescued)),
        'selected_source_counts': {
            source: sum(sources[int(index)] == source for index in order)
            for source in ('bitseed', 'sdpi')},
    }


def exact_random_selection(dockq, sources, quotas, threshold=.23):
    """Exact expected best and hit probability for source-quota random selection."""
    values = torch.as_tensor(dockq).float()
    if (values.ndim != 1 or not values.numel() or len(sources) != values.numel()
            or not torch.isfinite(values).all()
            or not bool(((values >= 0) & (values <= 1)).all())):
        raise ValueError('invalid random selection labels')
    quotas = {str(key): int(value) for key, value in quotas.items()}
    counts = {source: sum(value == source for value in sources) for source in quotas}
    if (set(sources) != set(quotas) or any(q < 0 or q > counts[s] for s, q in quotas.items())
            or not 1 <= sum(quotas.values()) <= values.numel()):
        raise ValueError('invalid random selection source quotas')
    denominator = math.prod(math.comb(counts[source], quota)
                            for source, quota in quotas.items())
    remaining = dict(counts)
    expected_best = hit_probability = probability_sum = 0.
    order = torch.argsort(values, descending=True, stable=True).tolist()
    for index in order:
        source = sources[index]
        remaining[source] -= 1
        if quotas[source] == 0:
            continue
        ways = 1
        for group, quota in quotas.items():
            needed = quota - int(group == source)
            ways *= math.comb(remaining[group], needed) if 0 <= needed <= remaining[group] else 0
        probability = ways / denominator
        probability_sum += probability
        expected_best += probability * float(values[index])
        if float(values[index]) >= threshold:
            hit_probability += probability
    if abs(probability_sum - 1.) > 1e-10:
        raise AssertionError('random maximum probabilities do not sum to one')
    oracle = float(values.max())
    return {
        'expected_best': expected_best,
        'full_pool_oracle': oracle,
        'expected_oracle_retention': expected_best / oracle if oracle > 0 else None,
        'hit_probability_0.23': hit_probability,
        'quotas': quotas,
    }

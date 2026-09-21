"""Deterministic cluster-balanced case visits, independent of training RNG."""

import hashlib

import torch


INDEX_ORDER = 'index_order_v1'
CLUSTER_ORDER = 'cluster_80pct_any_20pct_representative_v1'


def epoch_entries(entries, clusters, *, seed, epoch, policy=INDEX_ORDER):
    if type(seed) is not int or seed < 0 or type(epoch) is not int or epoch < 0:
        raise ValueError('data schedule requires nonnegative integer seed and epoch')
    if policy == INDEX_ORDER:
        return list(entries)
    if policy != CLUSTER_ORDER or not clusters:
        raise ValueError('cluster sampling requires explicit nonempty clusters')
    by_case = {row['case']: row for row in entries}
    if len(by_case) != len(entries):
        raise ValueError('duplicate cases in training index')
    members = [case for cluster in clusters for case in cluster['items']]
    if len(set(members)) != len(members) or set(members) != set(by_case):
        raise ValueError('clusters must partition the complete restored training index')
    for cluster in clusters:
        if not cluster['items'] or not set(cluster.get('representative', [])) <= set(cluster['items']):
            raise ValueError('invalid cluster membership or representatives')
    derived = int.from_bytes(hashlib.sha256(f'{seed}:cluster-epoch:{epoch}'.encode()).digest()[:8], 'big')
    generator = torch.Generator().manual_seed(derived)
    order = torch.randperm(len(clusters), generator=generator).tolist()
    selected = []
    for index in order:
        cluster = clusters[index]
        representative = float(torch.rand((), generator=generator)) < .2
        pool = (cluster.get('representative') if representative else None) or cluster['items']
        case = pool[int(torch.randint(len(pool), (), generator=generator))]
        selected.append(by_case[case])
    return selected

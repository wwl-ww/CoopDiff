"""Persistent native-free structural ranking for live candidate allocation."""

from pathlib import Path

import torch

from .ranking_archive import ARCHIVE_VERSION, cpu_tree
from .structural_ranker import StructuralQualityRanker, rank_states
from .structural_ranking_data import adapt_case, adapt_initial_pool, file_digest


def checkpoint_capabilities(checkpoint):
    """Supervision provenance is not efficacy or probability calibration."""
    if (checkpoint.get('architecture') != 'structural_quality_ranker'
            or checkpoint.get('metadata', {}).get('task') != 'quality'
            or type(checkpoint.get('epoch')) is not int or checkpoint['epoch'] < 1):
        raise ValueError('trained current-structure quality checkpoint required')
    supervision = checkpoint.get('head_supervision')
    if supervision is None:
        return {'rank': True, 'quality_aux': False, 'future': False,
                'provenance': 'legacy_quality_task_only', 'probabilities_calibrated': False}
    if (set(supervision) != {'rank', 'quality_aux', 'future'}
            or any(type(value) is not bool for value in supervision.values())
            or not supervision['rank']
            or type(checkpoint.get('optimizer_steps')) is not int
            or checkpoint['optimizer_steps'] < 1):
        raise ValueError('invalid quality-checkpoint supervision contract')
    arguments = checkpoint['metadata']['arguments']
    if supervision['quality_aux'] != (arguments['quality_weight'] > 0):
        raise ValueError('auxiliary supervision contradicts training configuration')
    if supervision['future'] != (arguments.get('future_weight', 0) > 0):
        raise ValueError('future supervision contradicts training configuration')
    return {**supervision, 'provenance': 'explicit_head_supervision',
            'probabilities_calibrated': False}


class StructuralRankingRuntime:
    """Selection policy may change; generator, pool and horizon may not.

    This explicitly evaluates a new selection policy under the trained fixed
    generator. It makes no distribution-shift or deployment-efficacy guarantee.
    Native references, labels and refined coordinates are absent from this API.
    """

    def __init__(self, checkpoint_path, generator_protocol, *, task='quality', device='cpu'):
        if task not in ('quality', 'future'):
            raise ValueError('unknown selection task')
        path = Path(checkpoint_path)
        before = file_digest(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if file_digest(path) != before:
            raise ValueError('checkpoint changed while loading; use a frozen copy')
        capabilities = checkpoint_capabilities(checkpoint)
        if task == 'future' and not capabilities['future']:
            raise ValueError('future selection requires a supervised future head')
        trained = checkpoint['metadata']['generator_protocol']
        for key in ('checkpoint_sha256', 'steps', 'pool'):
            if generator_protocol[key] != trained[key]:
                raise ValueError(f'ranker generator protocol mismatch: {key}')
        self.device, self.task = device, task
        self.final_supported = checkpoint['metadata'].get('training_candidate_scope') != 'initial-pool'
        self.model = StructuralQualityRanker(**checkpoint['config']).to(device).eval()
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.requires_grad_(False)
        self.provenance = {'mode': 'structural_ranker_topk_v1', 'task': task,
                           'future_target': checkpoint['metadata'].get('future_target') if task == 'future' else None,
                           'checkpoint_sha256': before, 'epoch': checkpoint['epoch'],
                           'training_selection': trained['selection'],
                           'selection_distribution_changed': True,
                           'tie_break': 'pool_index_ascending'}

    @torch.no_grad()
    def select(self, inputs, pool, keep):
        if type(keep) is not int or not 1 <= keep <= len(pool):
            raise ValueError('selection budget must fit the available pool')
        case = adapt_initial_pool({'protocol': ARCHIVE_VERSION, 'case': 'live', 'group': 'live',
                                   'pool': cpu_tree(pool), 'input_graphs': self._templates(inputs)})
        graphs = [{key: value.to(self.device) for key, value in graph.items()}
                  for graph in case['graphs']]
        scores = self.model.forward_list(graphs, stages=case['stages'], sources=case['sources'],
                                         task=self.task)['score'].cpu()
        order = rank_states(scores, case['stages'], torch.ones_like(scores, dtype=torch.bool),
                            scope='initial').tolist()
        return order[:keep], {'provenance': self.provenance, 'ordering': order,
                             'scores': scores.tolist(), 'selected': order[:keep],
                             'source_counts': {source: sum(case['sources'][i] == source for i in order[:keep])
                                               for source in ('sdpi', 'bitseed')},
                             'randomized_inclusion_probabilities': False}

    @torch.no_grad()
    def rank_pairs(self, inputs, pool, pairs):
        """Rank available outputs with quality, never with the future head."""
        if not self.final_supported:
            raise ValueError('initial-only checkpoint cannot rank refined outputs')
        case = adapt_case({'protocol': ARCHIVE_VERSION, 'case': 'live', 'group': 'live',
                           'pool': cpu_tree(pool), 'pairs': cpu_tree(pairs),
                           'input_graphs': self._templates(inputs)})
        graphs = [{key: value.to(self.device) for key, value in graph.items()}
                  for graph in case['graphs']]
        scores = self.model.forward_list(graphs, stages=case['stages'], sources=case['sources'],
                                         task='quality')['score'].cpu()
        rankings = {}
        for scope in ('initial', 'final', 'union'):
            order = rank_states(scores, case['stages'], torch.ones_like(scores, dtype=torch.bool),
                                scope=scope).tolist()
            rankings[scope] = [{'pool_index': case['identities'][i][0], 'stage': case['stages'][i],
                                'source': case['sources'][i], 'score': float(scores[i])} for i in order]
        return {'task': 'quality', 'checkpoint_sha256': self.provenance['checkpoint_sha256'],
                'rankings': rankings}

    @staticmethod
    def _templates(inputs):
        # The live generator may hold CUDA templates while saved pool states
        # are on CPU. Canonicalize only the graph builder's allowlisted inputs;
        # template POI/ligand coordinates never enter structural scoring.
        return {name: {key: cpu_tree(inputs[name][key]) for key in keys}
                for name, keys in [('e3_data', ('coords', 'features')),
                                   ('poi_data', ('features',)),
                                   ('ligand_data', ('features', 'edge_index'))]}

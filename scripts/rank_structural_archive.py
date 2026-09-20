"""Rank a sealed case with a quality checkpoint; no labels/native files read."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tricoopdiff.diffdock_compat  # noqa: F401
import torch

from tricoopdiff.structural_ranker import StructuralQualityRanker, rank_states
from tricoopdiff.structural_ranking_data import adapt_case, adapt_initial_pool, file_digest
from tricoopdiff.structural_ranking_runtime import checkpoint_capabilities


@torch.no_grad()
def rank_archive(checkpoint_path, archive_path, *, device='cpu', top_k=None,
                 candidate_scope='archived-pairs', include_quality_estimates=False, task='quality'):
    """Only trusted local PT files; protocol mismatch is a hard error."""
    if candidate_scope not in ('archived-pairs', 'initial-pool'):
        raise ValueError('unknown candidate scope')
    checkpoint_path, archive_path = Path(checkpoint_path), Path(archive_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    capabilities = checkpoint_capabilities(checkpoint)
    if task not in ('quality', 'future'):
        raise ValueError('unknown ranking task')
    if task == 'future' and (not capabilities['future'] or candidate_scope != 'initial-pool'
                             or include_quality_estimates):
        raise ValueError('future requires a supervised head, initial-only input and no quality estimates')
    if (checkpoint['metadata'].get('training_candidate_scope') == 'initial-pool'
            and candidate_scope != 'initial-pool'):
        raise ValueError('initial-only checkpoint cannot rank final/union states')
    if include_quality_estimates and not capabilities['quality_aux']:
        raise ValueError('checkpoint cannot establish auxiliary-quality supervision')
    manifest_path = archive_path.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(archive_path.with_suffix('.complete.json').read_text())
    archive_digest = file_digest(archive_path)
    if receipt['archive_sha256'] != archive_digest:
        raise ValueError('archive seal mismatch')
    for key in ('checkpoint_sha256', 'steps', 'pool', 'selection'):
        if manifest[key] != checkpoint['metadata']['generator_protocol'][key]:
            selection = manifest.get('selection')
            if (key == 'selection' and isinstance(selection, dict)
                    and selection.get('mode') == 'structural_ranker_topk_v1'
                    and selection.get('checkpoint_sha256') == file_digest(checkpoint_path)
                    and selection.get('training_selection') == checkpoint['metadata']['generator_protocol'][key]):
                continue  # Explicit application of this ranker's own new selection policy.
            raise ValueError(f'ranker generator protocol mismatch: {key}')
    archive = torch.load(archive_path, map_location='cpu', weights_only=False)
    if (archive['manifest'] != manifest or archive['case'] != archive_path.stem
            or archive['case'] not in manifest['cases']):
        raise ValueError('archive manifest/case mismatch')
    case = adapt_initial_pool(archive) if candidate_scope == 'initial-pool' else adapt_case(archive)
    model = StructuralQualityRanker(**checkpoint['config']).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    graphs = [{key: value.to(device) for key, value in graph.items()} for graph in case['graphs']]
    output = model.forward_list(graphs, stages=case['stages'], sources=case['sources'], task=task)
    scores = output['score'].cpu()
    future_target = checkpoint['metadata'].get('future_target', 'observed_final_dockq')
    result = {'case': case['case'], 'task': ('future_union_utility' if future_target == 'observed_union_dockq'
                                           else 'future_final_utility') if task == 'future' else 'current_quality',
              'future_target': future_target if task == 'future' else None,
              'candidate_scope': candidate_scope,
              'scope': ('all generated initial states; predicted fixed-protocol utility (see future_target)'
                        if task == 'future' else 'all generated initial states; current quality, not future utility'
                        if candidate_scope == 'initial-pool' else
                        'archived initial/valid-final pairs only; not unrefined pool'),
              'checkpoint_sha256': file_digest(checkpoint_path), 'archive_sha256': archive_digest,
              'pool_size': case['pool_size'],
              'head_capabilities': capabilities,
              'available_states': len(scores), 'top_k': top_k, 'rankings': {}}
    if candidate_scope == 'archived-pairs':
        result['archived_pairs'] = case['archived_pairs']
    for scope in (('initial',) if candidate_scope == 'initial-pool' else ('initial', 'final', 'union')):
        ids = rank_states(scores, case['stages'], torch.ones_like(scores, dtype=torch.bool),
                          scope=scope, top_k=top_k)
        result['rankings'][scope] = [
            {'pool_index': case['identities'][i][0], 'stage': case['stages'][i],
             'source': case['sources'][i], 'score': float(scores[i])} for i in ids.tolist()]
        if include_quality_estimates:
            for row, index in zip(result['rankings'][scope], ids.tolist()):
                estimates = {key: float(output[key][index]) for key in ('q_hat', 'p23', 'p49')}
                if not all(math.isfinite(v) for v in estimates.values()):
                    raise ValueError('nonfinite auxiliary-quality output')
                row['uncalibrated_quality_estimates'] = estimates
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--task', choices=('quality', 'future'), default='quality')
    parser.add_argument('--include-quality-estimates', action='store_true',
                        help='requires supervised auxiliary head; estimates are uncalibrated')
    parser.add_argument('--candidate-scope', choices=('archived-pairs', 'initial-pool'),
                        default='archived-pairs')
    args = parser.parse_args(argv)
    if args.out.exists() or (args.top_k is not None and args.top_k < 1):
        parser.error('use a new output path and positive top-k')
    report = rank_archive(args.checkpoint, args.archive, device=args.device, top_k=args.top_k,
                          candidate_scope=args.candidate_scope,
                          include_quality_estimates=args.include_quality_estimates, task=args.task)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()

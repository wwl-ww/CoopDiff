"""Train current-structure full-list ranking on sealed local archives."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tricoopdiff.diffdock_compat  # noqa: F401
import torch

from tricoopdiff.structural_ranker import (
    StructuralQualityRanker, rank_states, structural_ranking_loss, decision_ranking_loss,
    budget_retention_loss, selected_walker_mask,
)
from tricoopdiff.initial_selector import selector_retention
from tricoopdiff.ranking_backward import backward_list
from tricoopdiff.ranking_selection import refinement_selection_metrics
from tricoopdiff.structural_ranking_data import (
    StructuralArchiveCollection, StructuralArchiveDataset, StructuralJointDataset, audit_datasets, file_digest,
    future_utility_case,
)


def initialize_encoder(model, path, weights):
    """Copy only the geometry encoder from a trusted local refiner checkpoint."""
    if weights not in ('model', 'ema'):
        raise ValueError('explicit model/ema encoder weights required')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    key = 'ema_trifuse_state_dict' if weights == 'ema' else 'trifuse_state_dict'
    if key not in checkpoint or not isinstance(checkpoint[key], dict):
        raise ValueError(f'encoder checkpoint lacks {key}; no implicit fallback')
    state = {name[len('pooling.'):]: value for name, value in checkpoint[key].items()
             if name.startswith('pooling.')}
    expected = model.encoder.state_dict()
    if state.keys() != expected.keys():
        raise ValueError('pretrained geometry encoder keys do not match ranker')
    if any(value.shape != expected[name].shape or not torch.isfinite(value).all()
           for name, value in state.items()):
        raise ValueError('pretrained geometry encoder shapes/values do not match ranker')
    model.encoder.load_state_dict(state, strict=True)
    return {'kind': 'copied_refiner_geometry', 'weights': weights,
            'checkpoint_sha256': file_digest(path)}


def atomic_checkpoint(payload, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_resume(checkpoint, metadata, config, epochs):
    """Epoch-boundary resume only, with identical data/objective/runtime."""
    if ('resume_state' not in checkpoint or checkpoint.get('config') != config
            or type(checkpoint.get('epoch')) is not int or checkpoint['epoch'] < 1
            or epochs <= checkpoint['epoch']):
        raise ValueError('resume needs last.pt and a larger total epoch count')
    if checkpoint.get('optimizer_steps') != checkpoint['epoch'] * len(metadata['coverage'][0]):
        raise ValueError('resume optimizer-step count contradicts completed epochs')
    previous = checkpoint['metadata']
    for key, value in metadata.items():
        if key == 'arguments':
            continue
        if previous.get(key) != value:
            raise ValueError(f'resume contract changed: {key}')
    mutable = {'out', 'epochs', 'resume', 'audit_only', 'max_hours'}
    old_args = {k: v for k, v in previous['arguments'].items() if k not in mutable}
    new_args = {k: v for k, v in metadata['arguments'].items() if k not in mutable}
    if old_args != new_args:
        raise ValueError('resume training arguments changed')
    if checkpoint['resume_state']['torch_version'] != str(torch.__version__):
        raise ValueError('resume PyTorch runtime changed')


def score_case(model, case, device):
    graphs = [{key: value.to(device) for key, value in graph.items()}
              for graph in case['graphs']]
    return model.forward_list(graphs, stages=case['stages'], sources=case['sources'],
                              task=case.get('ranking_task', 'quality'))


def ordering_metrics(scores, labels, *, tie_tolerance=1e-4, groups=None):
    """Within-case pair accuracy; label ties excluded, predicted ties worth .5.

    This diagnostic uses pairwise label tolerance, not the loss's anchored
    tiers. It is descriptive only and does not choose checkpoints.
    """
    differences = labels[:, None] - labels[None, :]
    comparable = differences > tie_tolerance
    if groups is not None:
        if len(groups) != len(scores):
            raise ValueError('ordering groups must match scores')
        same_group = torch.tensor([[a == b for b in groups] for a in groups],
                                  device=scores.device, dtype=torch.bool)
        comparable = comparable & same_group
    count = int(comparable.sum())
    predicted = (scores[:, None] - scores[None, :])[comparable]
    accuracy = float(((predicted > 0).float() + .5 * (predicted == 0).float()).mean()) if count else None
    return {'comparable_pairs': count, 'pair_accuracy': accuracy}


def permute_within_groups(groups, generator):
    """Uniform permutations, including fixed points; preserve stage/source."""
    members = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    permutation = torch.arange(len(groups))
    for indices in members.values():
        ids = torch.tensor(indices)
        permutation[ids] = ids[torch.randperm(len(ids), generator=generator)]
    return permutation


def geometry_shuffle_diagnostic(scores, labels, stages, sources, *, repeats, seed, union_eligible=None):
    """Conditional permutation baseline for independent deterministic scorers.

    Permuting scores equals re-scoring permuted structures within identical
    flags, as long as there is no cross-candidate interaction. Labels stay at
    their original identities. This is not valid for a set-attention scorer.
    """
    if repeats < 1:
        raise ValueError('shuffle diagnostic needs positive repeats')
    scores, labels = scores.detach().cpu(), labels.detach().cpu()
    groups = list(zip(stages, sources))
    if len(groups) != len(scores) or labels.shape != scores.shape:
        raise ValueError('unaligned shuffle inputs')
    rng = torch.Generator().manual_seed(seed)
    valid = torch.ones_like(scores, dtype=torch.bool)
    union_valid = valid if union_eligible is None else union_eligible.cpu()
    if union_valid.shape != scores.shape or union_valid.dtype != torch.bool:
        raise ValueError('invalid Union eligibility mask')
    baseline = {scope: [] for scope in ('initial', 'final', 'union')}
    baseline_within = []
    for _ in range(repeats):
        shuffled = scores[permute_within_groups(groups, rng)]
        accuracy = ordering_metrics(shuffled, labels, groups=groups)['pair_accuracy']
        if accuracy is not None:
            baseline_within.append(accuracy)
        for scope in baseline:
            scope_scores = shuffled
            if scope == 'union' and not bool(union_valid.all()):
                # Do not introduce unrefined initial candidates into the
                # archived-walker Union null distribution either.
                union_groups = [(group, bool(eligible)) for group, eligible in zip(groups, union_valid)]
                scope_scores = scores[permute_within_groups(union_groups, rng)]
            order = rank_states(scope_scores, stages, union_valid if scope == 'union' else valid,
                                scope=scope, top_k=1)
            if len(order):
                baseline[scope].append(float(labels[order[0]]))
    result = {'repeats': repeats, 'seed': seed,
              'movable_states': sum(groups.count(group) > 1 for group in groups)}
    for scope, values in baseline.items():
        order = rank_states(scores, stages, union_valid if scope == 'union' else valid, scope=scope, top_k=1)
        if not values:
            result[scope] = None
            continue
        values = torch.tensor(values, dtype=torch.float64)
        mean = float(values.mean())
        result[scope] = {'shuffled_mean_top1': mean,
                         'shuffled_std_top1': float(values.std(unbiased=False)),
                         'observed_minus_shuffled': float(labels[order[0]]) - mean}
    observed = ordering_metrics(scores, labels, groups=groups)['pair_accuracy']
    mean_within = sum(baseline_within) / len(baseline_within) if baseline_within else None
    result['within_stage_source'] = {
        'shuffled_mean_pair_accuracy': mean_within,
        'observed_minus_shuffled': observed - mean_within if observed is not None else None,
    }
    return result


@torch.no_grad()
def evaluate(model, dataset, device, *, shuffle_repeats=0, shuffle_seed=7315, future_enabled=False,
             future_target='final'):
    model.eval()
    candidate_scope = getattr(dataset, 'candidate_scope', 'archived-pairs')
    initial_only = candidate_scope == 'initial-pool'
    full_initials = candidate_scope in ('initial-pool', 'joint')
    scopes = ('initial',) if initial_only else ('initial', 'final', 'union')
    rows = []
    for i in range(len(dataset)):
        case = dataset[i]
        scores = score_case(model, case, device)['score'].cpu()
        row = {'case': case['case'], 'within_stage_source': ordering_metrics(
            scores, case['labels'], groups=list(zip(case['stages'], case['sources'])))}
        if not initial_only:
            row['refinement_selection'] = refinement_selection_metrics(
                scores, case['labels'], case['identities'], case['sources'])
        if future_enabled:
            future = future_utility_case(case, target=future_target)
            if future is not None:
                predictions = score_case(model, future, device)['score'].cpu()
                row['future_ordering'] = ordering_metrics(predictions, future['labels'])
                row['future_selection'] = selector_retention(
                    predictions, future['labels'], future['sources'], keep=min(64, len(predictions)))
                positions = {tuple(identity): j for j, identity in enumerate(case['identities'])}
                selection_scores = scores.clone()
                for identity, prediction in zip(future['identities'], predictions):
                    selection_scores[positions[tuple(identity)]] = prediction
                row['future_refinement_selection'] = refinement_selection_metrics(
                    scores, case['labels'], case['identities'], case['sources'],
                    selection_scores=selection_scores)
                # Archived-pairs input alone can look complete while omitting
                # unrefined pool members. Compare against the actual pool.
                row['future_refinement_selection']['complete_pair_coverage'] = (
                    len(future['identities']) == case['pool_size'])
        if shuffle_repeats:
            # Stable per-case seed, independent of loader order and global RNG.
            digest = hashlib.sha256(f'{shuffle_seed}:{case["case"]}'.encode()).digest()
            row['geometry_shuffle'] = geometry_shuffle_diagnostic(
                scores, case['labels'], case['stages'], case['sources'],
                repeats=shuffle_repeats, seed=int.from_bytes(digest[:8], 'big'),
                union_eligible=case.get('union_eligible'))
        if full_initials:
            initial_ids = [j for j, stage in enumerate(case['stages']) if stage == 'initial']
            row['initial_selection'] = selector_retention(
                scores[initial_ids], case['labels'][initial_ids], [case['sources'][j] for j in initial_ids],
                keep=min(64, len(initial_ids)))
        for scope in scopes:
            eligible = case.get('union_eligible', torch.ones_like(scores, dtype=torch.bool)) if scope == 'union' else torch.ones_like(scores, dtype=torch.bool)
            order = rank_states(scores, case['stages'], eligible, scope=scope)
            row[scope] = None if not len(order) else {
                'top1': float(case['labels'][order[0]]),
                'oracle': float(case['labels'][order].max()),
                'top1_regret': float(case['labels'][order].max() - case['labels'][order[0]]),
                **ordering_metrics(scores[order], case['labels'][order]),
                'topk_oracle': {str(k): float(case['labels'][order[:k]].max()) for k in (8, 32, 64)},
                'ranking': [case['identities'][j] for j in order.tolist()],
            }
        rows.append(row)
    summary = {'rows': rows, 'cases': len(rows)}
    replay_keys = ['refinement_selection'] if not initial_only else []
    if future_enabled:
        replay_keys.append('future_refinement_selection')
    for replay_key in replay_keys:
        replays = [row[replay_key] for row in rows if replay_key in row]
        summary[replay_key] = {
            'scope': 'observed_valid_pairs_only',
            'complete_pair_coverage_cases': sum(r['complete_pair_coverage'] for r in replays),
            'budgets': {},
        }
        for budget in ('8', '32', '64'):
            available = [r['budgets'][budget] for r in replays if budget in r['budgets']]
            summary[replay_key]['budgets'][budget] = {
                'available_cases': len(available),
                **{f'mean_{key}': sum(r[key] for r in available) / len(available) if available else None
                   for key in ('initial_selected_best', 'final_selected_best', 'union_selected_best',
                               'final_top1', 'union_top1', 'paired_final_oracle', 'paired_union_oracle')},
                **{f'{key}_hits_0.23': sum(r[key] >= .23 for r in available)
                   for key in ('final_selected_best', 'union_selected_best', 'final_top1', 'union_top1')},
                'uniform_random_control': {
                    key: sum(r['uniform_random_control'][key] for r in available) / len(available)
                    if available else None
                    for key in ('final_selected_best', 'union_selected_best', 'final_top1', 'union_top1',
                                'final_selected_hit_probability_0.23', 'final_top1_hit_probability_0.23')},
            }
    if full_initials:
        selection = [row['initial_selection'] for row in rows]
        retention = [r['oracle_retention'] for r in selection if r['oracle_retention'] is not None]
        summary['initial_selection'] = {
            'mean_selected_best': sum(r['selected_best'] for r in selection) / len(rows),
            'mean_oracle_retention': sum(retention) / len(retention) if retention else None,
            'selected_hits_0.23': sum(r['selected_hit'] for r in selection),
            'selected_hits_0.49': sum(r['selected_best'] >= .49 for r in selection),
            'full_pool_hits_0.23': sum(r['full_pool_hit'] for r in selection),
            'oracle_index_retained': sum(r['oracle_index_retained'] for r in selection),
            'mean_selected_source_counts': {
                source: sum(r['selected_source_counts'][source] for r in selection) / len(rows)
                for source in ('sdpi', 'bitseed')},
        }
    within = [row['within_stage_source']['pair_accuracy'] for row in rows
              if row['within_stage_source']['pair_accuracy'] is not None]
    summary['within_stage_source'] = {
        'available_cases': len(within),
        'mean_pair_accuracy': sum(within) / len(within) if within else None,
    }
    if shuffle_repeats:
        summary['geometry_shuffle'] = {'repeats': shuffle_repeats, 'seed': shuffle_seed}
        for scope in (*scopes, 'within_stage_source'):
            entries = [row['geometry_shuffle'][scope] for row in rows]
            deltas = [entry['observed_minus_shuffled'] for entry in entries
                      if entry is not None and entry['observed_minus_shuffled'] is not None]
            summary['geometry_shuffle'][scope] = {
                'available_cases': len(deltas),
                'mean_observed_minus_shuffled': sum(deltas) / len(deltas) if deltas else None,
            }
    for scope in scopes:
        available = [row[scope] for row in rows if row[scope] is not None]
        pair_accuracies = [r['pair_accuracy'] for r in available if r['pair_accuracy'] is not None]
        summary[scope] = {
            'available_cases': len(available), 'missing_cases': len(rows) - len(available),
            'mean_top1_available': (sum(r['top1'] for r in available) / len(available)) if available else None,
            'mean_oracle_available': (sum(r['oracle'] for r in available) / len(available)) if available else None,
            'mean_top1_regret_available': (sum(r['top1_regret'] for r in available) / len(available)) if available else None,
            'hits_0.23': sum(r['top1'] >= .23 for r in available),
            'hits_0.49': sum(r['top1'] >= .49 for r in available),
            'success_rate_0.23_all': sum(r['top1'] >= .23 for r in available) / len(rows),
            'success_rate_0.49_all': sum(r['top1'] >= .49 for r in available) / len(rows),
            'pair_accuracy_cases': len(pair_accuracies),
            'mean_pair_accuracy': sum(pair_accuracies) / len(pair_accuracies) if pair_accuracies else None,
            'mean_topk_oracle_available': {
                str(k): sum(r['topk_oracle'][str(k)] for r in available) / len(available) if available else None
                for k in (8, 32, 64)},
        }
        for threshold in (.23, .49):
            opportunities = sum(r['oracle'] >= threshold for r in available)
            hits = sum(r['top1'] >= threshold for r in available)
            summary[scope][f'oracle_opportunities_{threshold}'] = opportunities
            summary[scope][f'hit_rate_given_oracle_{threshold}'] = hits / opportunities if opportunities else None
    return summary


def selection_score(metrics, candidate_scope, future_target=None, quality_walker_scope='all'):
    if future_target is not None or quality_walker_scope == 'future-top64':
        replay = metrics['future_refinement_selection']
        if replay['complete_pair_coverage_cases'] != metrics['cases']:
            raise ValueError('selector checkpoint selection requires complete validation pair coverage')
        metric = f'mean_{future_target}_selected_best' if future_target is not None else 'mean_union_top1'
        value = replay['budgets']['64'][metric]
        if value is None:
            raise ValueError('selector checkpoint selection needs budget-64 validation')
        return value
    if candidate_scope == 'initial-pool':
        return metrics['initial_selection']['mean_selected_best']
    return metrics['union']['mean_top1_available']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train-archives', 'train-labels'):
        parser.add_argument('--' + name, type=Path, nargs='+', required=True)
    for name in ('val-archives', 'val-labels', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--candidate-scope', choices=('archived-pairs', 'initial-pool', 'joint'), default='archived-pairs')
    parser.add_argument('--train-initial-labels', type=Path, nargs='+')
    parser.add_argument('--val-initial-labels', type=Path)
    parser.add_argument('--resume', type=Path, help='last.pt; --epochs is the total target, --out must be new')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max-hours', type=float, default=0.,
                        help='optional soft training budget; finish/checkpoint the current epoch before stopping')
    parser.add_argument('--save-epoch-checkpoints', action='store_true')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--interface-layers', type=int, default=0,
                        help='local interface message-passing layers; 0 is the pooled baseline')
    parser.add_argument('--encoder-checkpoint', type=Path)
    parser.add_argument('--encoder-weights', choices=('model', 'ema'))
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--quality-weight', type=float, default=0.)
    parser.add_argument('--backward-chunk-size', type=int, default=0,
                        help='exact two-pass list gradient replay; 0 retains ordinary autograd')
    parser.add_argument('--future-weight', type=float, default=0.,
                        help='joint initial-only full ranking of observed final utility')
    parser.add_argument('--future-target', choices=('final', 'union'), default='final')
    parser.add_argument('--budget-weight', type=float, default=0.,
                        help='future top64 best-tier retention; requires complete training pool outcomes')
    parser.add_argument('--objective', choices=('decision', 'legacy'), default='decision',
                        help='decision balances Initial/Final/eligible Union full-list losses')
    parser.add_argument('--checkpoint-task', choices=('quality', 'future'), default='quality')
    parser.add_argument('--quality-walker-scope', choices=('all', 'future-top64'), default='all',
                        help='train Final/Union on current native-free selector allocation; Initial stays full-pool')
    parser.add_argument('--top-weight', type=float, default=1.,
                        help='best-tier mass coefficient within each decision set; 0 disables')
    parser.add_argument('--tie-tolerance', type=float, default=1e-4)
    parser.add_argument('--shuffle-repeats', type=int, default=32,
                        help='within-flag geometry diagnostic draws; 0 disables')
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours < 0:
        parser.error('max hours must be finite and nonnegative')
    if args.backward_chunk_size < 0:
        parser.error('backward chunk size must be nonnegative')
    if args.quality_walker_scope == 'future-top64' and (args.future_weight <= 0 or args.objective != 'decision'):
        parser.error('selector-conditioned quality requires decision objective and trained future head')
    if not math.isfinite(args.budget_weight) or args.budget_weight < 0:
        parser.error('budget weight must be finite and nonnegative')
    if args.budget_weight and args.future_weight <= 0:
        parser.error('budget supervision requires a trained future head')
    if args.checkpoint_task == 'future' and args.future_weight <= 0:
        parser.error('future checkpoint selection requires future supervision')
    if args.future_weight > 0 and args.candidate_scope == 'initial-pool':
        parser.error('future supervision requires observed paired final outcomes')
    if len(args.train_archives) != len(args.train_labels):
        parser.error('each training archive shard needs its matching label directory')
    if args.candidate_scope == 'joint':
        if not args.train_initial_labels or len(args.train_initial_labels) != len(args.train_archives) or args.val_initial_labels is None:
            parser.error('joint mode needs aligned full-initial label directories for train and val')
    elif args.train_initial_labels is not None or args.val_initial_labels is not None:
        parser.error('additional initial label directories are only used by joint mode')
    if (args.encoder_checkpoint is None) != (args.encoder_weights is None):
        parser.error('encoder checkpoint and explicit model/ema choice must be supplied together')
    if (args.out.exists() or args.epochs < 1 or args.hidden_dim < 4 or args.shuffle_repeats < 0 or args.interface_layers < 0
            or not math.isfinite(args.lr) or args.lr <= 0
            or any(not math.isfinite(v) or v < 0 for v in (args.quality_weight, args.future_weight, args.top_weight, args.tie_tolerance))):
        parser.error('use a new output directory and valid hyperparameters')
    def make_dataset(archives, labels, split, initial_labels=None):
        scope = 'archived-pairs' if args.candidate_scope == 'joint' else args.candidate_scope
        dataset = StructuralArchiveDataset(archives, labels, split, candidate_scope=scope)
        if args.candidate_scope == 'joint':
            dataset = StructuralJointDataset(dataset, StructuralArchiveDataset(
                archives, initial_labels, split, candidate_scope='initial-pool'))
        return dataset

    shards = [make_dataset(a, l, 'train', args.train_initial_labels[i] if args.train_initial_labels else None)
              for i, (a, l) in enumerate(zip(args.train_archives, args.train_labels))]
    train = shards[0] if len(shards) == 1 else StructuralArchiveCollection(shards)
    val = make_dataset(args.val_archives, args.val_labels, 'val', args.val_initial_labels)
    coverage = audit_datasets(train, val)
    if args.checkpoint_task == 'future' or args.quality_walker_scope == 'future-top64':
        for index in range(len(val)):
            case = val[index]
            observed = sum(stage == 'final' for stage in case['stages'])
            if observed != case['pool_size'] or observed < 64:
                raise ValueError('future checkpoint selection needs all pool outcomes and at least 64 candidates per validation case')
    future_coverage = []
    if args.future_weight:
        for index in range(len(train)):
            case = train[index]
            future = future_utility_case(case, target=args.future_target)
            if (args.budget_weight or args.quality_walker_scope == 'future-top64') and (future is None or len(future['labels']) != case['pool_size']
                                       or case['pool_size'] <= 64):
                raise ValueError('top64 budget loss requires complete training outcomes for a pool larger than 64; partial trajectories cannot substitute')
            informative = (future is not None and len(future['labels']) > 1
                           and float(future['labels'].max() - future['labels'].min()) > args.tie_tolerance)
            future_coverage.append({'case': train.cases[index],
                                    'paired_labels': len(future['labels']) if future is not None else 0,
                                    'informative': informative})
        if not any(row['informative'] for row in future_coverage):
            raise ValueError('future training has no nontrivial observed final ordering')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = {'input_dim': train[0]['graphs'][0]['node_features'].shape[1], 'hidden_dim': args.hidden_dim,
              'interface_layers': args.interface_layers}
    model = StructuralQualityRanker(**config)
    initialization = (initialize_encoder(model, args.encoder_checkpoint, args.encoder_weights)
                      if args.encoder_checkpoint is not None else 'random')
    metadata = {'scope': ('internal full-pool initial-quality ranking, not future utility'
                          if args.candidate_scope == 'initial-pool' else
                          'internal joint full initials and archived finals' if args.candidate_scope == 'joint' else
                          'internal labeled archived pairs, not full-pool efficacy'),
                'training_candidate_scope': args.candidate_scope,
                'collection_policies': {'train': train.manifest['selection'],
                                        'validation': val.manifest['selection'],
                                        'different_names_require_complete_valid_outcomes': True},
                'supervised_stages': ['initial'] if args.candidate_scope == 'initial-pool' else ['initial', 'final'],
                'implementation_sha256': {str(path.relative_to(Path(__file__).resolve().parents[1])): file_digest(path)
                                         for path in [Path(__file__).resolve()] + [
                                             Path(__file__).resolve().parents[1] / 'tricoopdiff' / name
                                             for name in ('structural_ranker.py', 'structural_ranking_data.py',
                                                          'geometry_encoders.py', 'graphs.py', 'ranking_interface.py',
                                                          'ranking_selection.py', 'ranking_backward.py')]},
                'coverage': coverage, 'train_manifest_sha256': train.manifest_digest,
                'val_manifest_sha256': val.manifest_digest,
                'train_label_manifest_sha256': train.label_manifest_digest,
                'val_label_manifest_sha256': val.label_manifest_digest,
                'train_shard_identities': [{'archive_manifest_sha256': d.manifest_digest,
                                            'label_manifest_sha256': d.label_manifest_digest,
                                            'cases': d.cases} for d in shards],
                'train_case_seals': dict(train.seals), 'val_case_seals': dict(val.seals),
                'generator_protocol': {k: train.manifest[k] for k in
                                       ('checkpoint_sha256', 'steps', 'pool', 'selection')},
                'arguments': {k: ([str(item) for item in v] if isinstance(v, list)
                                  else str(v) if isinstance(v, Path) else v)
                              for k, v in vars(args).items()},
                'task': 'quality', 'primary': ('initial_top64_selected_best' if args.candidate_scope == 'initial-pool'
                                              else 'union_mean_top1'),
                'encoder_initialization': initialization, 'future_head_trained': args.future_weight > 0,
                'future_target': f'observed_{args.future_target}_dockq' if args.future_weight > 0 else None,
                'future_training_coverage': future_coverage}
    selection_target = args.future_target if args.checkpoint_task == 'future' else None
    if selection_target is not None:
        metadata['primary'] = f'future_top64_{selection_target}_selected_best'
    elif args.quality_walker_scope == 'future-top64':
        metadata['primary'] = 'future_top64_quality_union_top1'
    resumed = None
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=False)
        validate_resume(resumed, metadata, config, args.epochs)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'audit.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if args.audit_only:
        print('Archive integrity, label coverage and PDB/group separation passed')
        return
    model = model.to(args.device)
    model.future_head.requires_grad_(args.future_weight > 0)
    if args.quality_weight == 0:
        model.quality_head.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    best, best_payload, first_epoch = -math.inf, None, 1
    if resumed is not None:
        model.load_state_dict(resumed['state_dict'], strict=True)
        optimizer.load_state_dict(resumed['optimizer'])
        state = resumed['resume_state']
        best_payload = state['best_payload']
        best = selection_score(best_payload['validation'], args.candidate_scope, selection_target, args.quality_walker_scope)
        atomic_checkpoint(best_payload, args.out / 'best.pt')
        first_epoch = resumed['epoch'] + 1
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'], torch.device(args.device))
        (args.out / 'resume.json').write_text(json.dumps({
            'checkpoint_sha256': file_digest(args.resume), 'completed_epoch': resumed['epoch'],
            'history': 'prior epochs remain in the original run directory',
        }, indent=2) + '\n')
    training_started = time.monotonic()
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        total = 0.
        quality_total, future_total, future_cases = 0., 0., 0
        budget_total = 0.
        for position, index in enumerate(order, 1):
            case = train[index]
            optimizer.zero_grad(set_to_none=True)
            objective = decision_ranking_loss if args.objective == 'decision' else structural_ranking_loss
            scope_args = {'stages': case['stages'], 'union_eligible': case.get('union_eligible')} if args.objective == 'decision' else {}
            if args.quality_walker_scope == 'future-top64':
                allocation_case = future_utility_case(case, target=args.future_target)
                with torch.no_grad():
                    allocation_scores = score_case(model, allocation_case, args.device)['score']
                eligible = selected_walker_mask(allocation_scores, allocation_case['identities'], case['identities'])
                scope_args.update(union_eligible=eligible, final_eligible=eligible)
            def quality_objective(output):
                return objective(output, case['labels'].to(args.device), **scope_args,
                                 quality_weight=args.quality_weight, top_weight=args.top_weight,
                                 tie_tolerance=args.tie_tolerance)
            if args.backward_chunk_size:
                loss, _ = backward_list(model, case, args.device, quality_objective,
                                        chunk_size=args.backward_chunk_size)
            else:
                loss = quality_objective(score_case(model, case, args.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite ranking loss')
                loss.backward()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite ranking loss')
            quality_total += float(loss.detach())
            if args.future_weight:
                future = future_utility_case(case, target=args.future_target)
                if future is not None:
                    def future_objective(output):
                        value = structural_ranking_loss(
                            output, future['labels'].to(args.device),
                            tie_tolerance=args.tie_tolerance, top_weight=args.top_weight)
                        if args.budget_weight:
                            value = value + args.budget_weight * budget_retention_loss(
                                output['score'], future['labels'].to(args.device),
                                keep=64, tie_tolerance=args.tie_tolerance)
                        return value
                    if args.backward_chunk_size:
                        future_loss, future_output = backward_list(
                            model, future, args.device, future_objective,
                            chunk_size=args.backward_chunk_size, weight=args.future_weight)
                    else:
                        future_output = score_case(model, future, args.device)
                        future_loss = future_objective(future_output)
                        if not torch.isfinite(future_loss):
                            raise FloatingPointError('nonfinite future ranking loss')
                        (args.future_weight * future_loss).backward()
                    if args.budget_weight:
                        budget_total += float(budget_retention_loss(
                            future_output['score'].detach(), future['labels'].to(args.device), keep=64,
                            tie_tolerance=args.tie_tolerance))
                    if not torch.isfinite(future_loss):
                        raise FloatingPointError('nonfinite future ranking loss')
                    future_total += float(future_loss.detach())
                    future_cases += 1
                    loss = loss.detach() + args.future_weight * future_loss.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
            if position == 1 or position % 10 == 0 or position == len(order):
                print(f'epoch={epoch} case={position}/{len(order)} id={case["case"]} '
                      f'states={len(case["graphs"])} loss={float(loss.detach()):.6f}', flush=True)
        metrics = evaluate(model, val, args.device, shuffle_repeats=args.shuffle_repeats,
                           shuffle_seed=args.seed, future_enabled=args.future_weight > 0,
                           future_target=args.future_target)
        record = {'epoch': epoch, 'train_loss': total / len(train), 'validation': metrics}
        if args.future_weight:
            record.update(quality_train_loss=quality_total / len(train),
                          future_train_loss=future_total / future_cases if future_cases else None,
                          future_supervised_cases=future_cases)
            if args.budget_weight:
                record['budget_train_loss'] = budget_total / future_cases
        with (args.out / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        key = selection_score(metrics, args.candidate_scope, selection_target, args.quality_walker_scope)
        payload = {'architecture': 'structural_quality_ranker', 'config': config,
                   'state_dict': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                   'optimizer_steps': epoch * len(train),
                   'head_supervision': {'rank': True, 'quality_aux': args.quality_weight > 0,
                                        'future': args.future_weight > 0},
                   'epoch': epoch, 'metadata': metadata, 'validation': metrics}
        if key > best:
            best = key
            best_payload = payload
            atomic_checkpoint(best_payload, args.out / 'best.pt')
        if args.save_epoch_checkpoints:
            atomic_checkpoint(payload, args.out / f'epoch_{epoch:03d}.pt')
        atomic_checkpoint({**payload, 'optimizer': optimizer.state_dict(), 'resume_state': {
            'best_payload': best_payload, 'python_rng': random.getstate(),
            'torch_rng': torch.get_rng_state(), 'torch_version': str(torch.__version__),
            'cuda_rng': torch.cuda.get_rng_state(torch.device(args.device))
            if torch.device(args.device).type == 'cuda' else None,
        }}, args.out / 'last.pt')
        print(f'epoch={epoch} loss={record["train_loss"]:.6f} {metadata["primary"]}={key:.6f}', flush=True)
        if args.max_hours and time.monotonic() - training_started >= args.max_hours * 3600:
            print(f'Soft time budget reached after checkpointed epoch {epoch}', flush=True)
            break


if __name__ == '__main__':
    main()

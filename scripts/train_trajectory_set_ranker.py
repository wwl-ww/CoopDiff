#!/usr/bin/env python
"""Train a paired initial/refined set ranker on sealed native-free archives."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import torch

from evaluation_common import sha256_file
from extract_expanded_ranking_features import PROTOCOL
from tricoopdiff.setwise_ranking import (TrajectorySetRanker, source_indices,
                                         stable_topk, trajectory_ranking_loss)


def load_shard(archives, features, labels, split):
    archive_manifest_path = archives / 'manifest.json'
    archive_manifest = json.loads(archive_manifest_path.read_text())
    feature_manifest = json.loads((features / 'manifest.json').read_text())
    label_manifest = json.loads((labels / 'manifest.json').read_text())
    if (feature_manifest['protocol'] != PROTOCOL
            or feature_manifest['archive_manifest_sha256'] != sha256_file(archive_manifest_path)
            or label_manifest['archive_manifest_sha256'] != sha256_file(archive_manifest_path)):
        raise ValueError('trajectory features/labels do not match archive manifest')
    cases = []
    for case in archive_manifest['cases']:
        archive_path = archives / f'{case}.pt'
        archive = torch.load(archive_path, map_location='cpu', weights_only=False)
        feature = torch.load(features / f'{case}.pt', map_location='cpu', weights_only=False)
        label = json.loads((labels / f'{case}.json').read_text())
        digest = sha256_file(archive_path)
        if (archive['case'] != case or archive['split'] != split
                or feature['case'] != case or feature['archive_sha256'] != digest
                or label['case'] != case or label['archive_sha256'] != digest):
            raise ValueError('trajectory split/archive identity mismatch')
        feature_rows = {(row['pool_index'], row['stage']): row for row in feature['rows']}
        if len(feature_rows) != len(feature['rows']):
            raise ValueError('duplicate trajectory feature row')
        truth = {row['pool_index']: row for row in label['labels']}
        pairs = {pair['pool_index']: pair for pair in archive['pairs']}
        if len(pairs) != len(archive['pairs']) or set(pairs) != set(truth):
            raise ValueError('trajectory labels do not exactly cover archived pairs')
        initial_rows, final_rows, initial_q, final_q = [], [], [], []
        valid, sources, pool_indices = [], [], []
        for pool_index in sorted(pairs):
            pair, target = pairs[pool_index], truth[pool_index]
            key = (pool_index, 'initial')
            if key not in feature_rows or pair['valid'] != target['valid']:
                raise ValueError('missing initial feature or validity mismatch')
            initial = np.asarray(feature_rows[key]['features'], dtype=np.float32)
            final_key = (pool_index, 'final')
            if pair['valid']:
                if final_key not in feature_rows or target['final'] is None:
                    raise ValueError('valid trajectory lacks final state')
                final = np.asarray(feature_rows[final_key]['features'], dtype=np.float32)
                final_value = float(target['final']['dockq'])
            else:
                if final_key in feature_rows or target['final'] is not None:
                    raise ValueError('failed trajectory unexpectedly has final state')
                final, final_value = initial.copy(), 0.
            if pair['source'] not in ('bitseed', 'sdpi'):
                raise ValueError('unknown actual trajectory source')
            initial_rows.append(initial); final_rows.append(final)
            initial_q.append(float(target['initial']['dockq'])); final_q.append(final_value)
            valid.append(bool(pair['valid'])); sources.append(pair['source'])
            pool_indices.append(pool_index)
        initial = torch.from_numpy(np.stack(initial_rows)).float().log1p()
        final = torch.from_numpy(np.stack(final_rows)).float().log1p()
        if (initial.shape != final.shape or initial.shape[1] != len(feature_manifest['feature_names'])
                or not torch.isfinite(initial).all() or not torch.isfinite(final).all()):
            raise ValueError('invalid trajectory feature matrices')
        cases.append({
            'case': case, 'initial': initial, 'final': final,
            'initial_dockq': torch.tensor(initial_q).float(),
            'final_dockq': torch.tensor(final_q).float(),
            'valid': torch.tensor(valid, dtype=torch.bool),
            'sources': sources, 'pool_indices': pool_indices,
        })
    return archive_manifest, cases


def combine_shards(shards):
    if not shards:
        raise ValueError('at least one trajectory shard is required')
    manifest, cases, seen = shards[0][0], [], set()
    for other, rows in shards:
        for key in ('checkpoint_sha256', 'steps', 'pool', 'selection'):
            if other[key] != manifest[key]:
                raise ValueError(f'trajectory shard generator mismatch: {key}')
        for case in rows:
            pdb = case['case'].split('_')[0].upper()
            if pdb in seen:
                raise ValueError('duplicate trajectory PDB across shards')
            seen.add(pdb); cases.append(case)
    return manifest, cases


def model_inputs(case, mean, scale, device):
    initial = ((case['initial'] - mean) / scale).to(device)
    final = ((case['final'] - mean) / scale).to(device)
    sources = source_indices(case['sources'], device=device)
    return initial, final, sources, case['valid'].to(device)


@torch.no_grad()
def evaluate(model, cases, mean, scale, device):
    model.eval(); rows, losses = [], []
    for case in cases:
        initial, final, sources, valid = model_inputs(case, mean, scale, device)
        output = model(initial, final, sources, valid)
        losses.append(float(trajectory_ranking_loss(
            output, case['initial_dockq'].to(device), case['final_dockq'].to(device))))
        final_ids = stable_topk(output['final_score'].cpu(), case['valid'], 1)
        union_ids = stable_topk(
            output['union_score'].cpu(), torch.ones_like(case['valid']), 1)
        final_index, union_index = int(final_ids[0]), int(union_ids[0])
        choose_final = bool(output['choose_final_probability'][union_index] >= .5)
        if not case['valid'][union_index]:
            choose_final = False
        union_value = float(case['final_dockq'][union_index] if choose_final
                            else case['initial_dockq'][union_index])
        oracle_union_values = torch.where(
            case['valid'], torch.maximum(case['initial_dockq'], case['final_dockq']),
            case['initial_dockq'])
        rows.append({
            'case': case['case'],
            'final_top1': float(case['final_dockq'][final_index]),
            'final_oracle': float(case['final_dockq'][case['valid']].max()),
            'union_top1': union_value,
            'union_oracle_stage_at_selected': float(oracle_union_values[union_index]),
            'union_oracle': float(oracle_union_values.max()),
            'selected_union_stage': 'final' if choose_final else 'initial',
            'selected_pool_index': case['pool_indices'][union_index],
        })
    summary = {'loss': sum(losses) / len(losses), 'cases': len(rows), 'rows': rows}
    for stage in ('final', 'union'):
        summary[stage] = {
            'mean_top1': sum(row[f'{stage}_top1'] for row in rows) / len(rows),
            'mean_oracle': sum(row[f'{stage}_oracle'] for row in rows) / len(rows),
            'hits_0.23': sum(row[f'{stage}_top1'] >= .23 for row in rows),
            'hits_0.49': sum(row[f'{stage}_top1'] >= .49 for row in rows),
        }
    summary['union']['mean_oracle_stage_at_selected'] = sum(
        row['union_oracle_stage_at_selected'] for row in rows) / len(rows)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train-archives', 'train-features', 'train-labels'):
        parser.add_argument('--' + name, type=Path, nargs='+', required=True)
    for name in ('val-archives', 'val-features', 'val-labels', 'split-audit', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--primary', choices=('final', 'union'), default='union')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model-dim', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args(argv)
    counts = (len(args.train_archives), len(args.train_features), len(args.train_labels))
    if len(set(counts)) != 1 or args.epochs < 1 or args.lr <= 0 or args.out.exists():
        parser.error('training shards must match; use positive epochs/lr and a new output')
    train_manifest, train = combine_shards([
        load_shard(a, f, l, 'train') for a, f, l in zip(
            args.train_archives, args.train_features, args.train_labels)])
    _, val = load_shard(args.val_archives, args.val_features, args.val_labels, 'val')
    audit = json.loads(args.split_audit.read_text())
    if [case['case'] for case in train] != audit['train_cases']:
        raise ValueError('trajectory-ranker training order differs from split audit')
    if {case['case'] for case in val} != set(audit['val_cases']):
        raise ValueError('trajectory-ranker validation differs from split audit')
    if args.train_limit is not None:
        if not 1 <= args.train_limit <= len(train):
            raise ValueError('invalid trajectory-ranker training prefix')
        train = train[:args.train_limit]

    random.seed(args.seed); torch.manual_seed(args.seed)
    values = torch.cat([case['initial'] for case in train]
                       + [case['final'][case['valid']] for case in train])
    mean, scale = values.mean(0), values.std(0).clamp_min(.01)
    model = TrajectorySetRanker(
        values.shape[1], args.model_dim, args.num_heads, args.num_layers,
        2 * args.model_dim, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        total = 0.
        for index in order:
            case = train[index]
            initial, final, sources, valid = model_inputs(case, mean, scale, args.device)
            output = model(initial, final, sources, valid)
            loss = trajectory_ranking_loss(
                output, case['initial_dockq'].to(args.device),
                case['final_dockq'].to(args.device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); total += float(loss.detach())
        metrics = evaluate(model, val, mean, scale, args.device)
        key = (metrics[args.primary]['mean_top1'], metrics[args.primary]['hits_0.23'],
               -metrics['loss'])
        history.append({'epoch': epoch, 'train_loss': total / len(train),
                        'validation_loss': metrics['loss'], 'selection_key': list(key),
                        'final': metrics['final'], 'union': metrics['union']})
        if best is None or key > best['key']:
            best = {'epoch': epoch, 'key': key, 'metrics': metrics,
                    'state_dict': {name: value.detach().cpu().clone()
                                   for name, value in model.state_dict().items()}}
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save({
        'architecture': 'trajectory_set_ranker', 'model_config': model.config,
        'state_dict': best['state_dict'], 'mean': mean, 'scale': scale,
        'primary': args.primary, 'best_epoch': best['epoch'],
        'selection_key': list(best['key']),
        'generator_checkpoint_sha256': train_manifest['checkpoint_sha256'],
        'split_audit_sha256': sha256_file(args.split_audit),
        'train_case_ids': [case['case'] for case in train],
    }, args.out / 'best.pt')
    (args.out / 'report.json').write_text(json.dumps({
        'scope': 'internal paired-trajectory post-ranker; not untouched benchmark',
        'architecture': ('paired initial/final/delta encoding plus permutation-equivariant '
                         'walker-set attention and explicit fallback-stage prediction'),
        'primary': args.primary, 'seed': args.seed, 'epochs': args.epochs,
        'train_cases': len(train), 'val_cases': len(val),
        'best_epoch': best['epoch'], 'metrics': best['metrics'], 'history': history,
    }, indent=2) + '\n')
    print(json.dumps({stage: best['metrics'][stage] for stage in ('final', 'union')}, indent=2))


if __name__ == '__main__':
    main()

"""Collect native-free initial/final geometry and fixed-clock C40 probes.

The inference index is an allowlisted schema with no scoring references. Exact
labels are produced later by score_ranking_archive.py in a separate process.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import tricoopdiff.diffdock_compat  # noqa: E402,F401
import torch

from evaluation_common import evaluation_input_signature, resolve_entry_paths, sha256_file
from evaluation_runner import EvaluationRunner
from featurize_common import featurize_complex
from tricoopdiff.expert_anchors import DualExpertAnchorInitializer, FixedCandidateInitializer, generate_hybrid_candidates
from tricoopdiff.ranking_archive import (ARCHIVE_VERSION, GEOMETRY_NAMES, PROBE_TIME, cpu_tree,
                                       probe_state, select_archive_candidates, state_hash)
from tricoopdiff.sampler import sample_single
from tricoopdiff.structural_ranking_runtime import StructuralRankingRuntime


def atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inference_entries(path):
    payload = json.loads(Path(path).read_text())
    if set(payload) != {'entries'}:
        raise ValueError('inference index must only contain entries; labels/metadata belong elsewhere')
    allowed = {'case', 'group', 'split', 'protein1', 'protein2', 'ligand'}
    for entry in payload['entries']:
        if set(entry) != allowed:
            raise ValueError(f'inference entry must contain exactly {sorted(allowed)}')
        if entry['case'] != Path(entry['case']).name or entry['case'] in ('.', '..'):
            raise ValueError('unsafe case name')
    if len({e['case'] for e in payload['entries']}) != len(payload['entries']):
        raise ValueError('duplicate cases')
    return payload['entries']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--walkers', type=int, default=32)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--selector-checkpoint', type=Path,
                        help='experimental native-free structural Top-K; default remains balanced random')
    parser.add_argument('--selector-task', choices=('quality', 'future'), default='quality')
    parser.add_argument('--seed', type=int, default=31000)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--case-limit', type=int, help='Smoke-only limit on the index prefix')
    args = parser.parse_args()
    if args.selector_task == 'future' and args.selector_checkpoint is None:
        parser.error('future selection requires a trained selector checkpoint')
    if not (1 <= args.walkers <= 256 and args.steps > 0 and 0 <= args.shard_index < args.num_shards):
        parser.error('invalid walker/step/shard budget')
    root = args.project_root.resolve()
    # Vendored runtime imports may change CWD; resolve CLI assets at project root.
    os.chdir(root)
    output = args.out.resolve()
    entries = inference_entries(args.index)
    if args.case_limit is not None:
        if args.case_limit < 1:
            parser.error('--case-limit must be positive')
        entries = entries[:args.case_limit]
    paths_by_case = {e['case']: resolve_entry_paths(e, root) for e in entries}
    manifest = {
        'protocol': ARCHIVE_VERSION, 'index_sha256': sha256_file(args.index),
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'input_signatures': {case: evaluation_input_signature(paths) for case, paths in paths_by_case.items()},
        'cases': [e['case'] for e in entries], 'walkers': args.walkers, 'steps': args.steps,
        'proposal_seed': args.seed, 'selection_seed': args.seed + 3_000_000,
        'diffusion_seed': 'proposal_seed + 2000000 + pool_index',
        'pool': {'bitseed': 192, 'sdpi': 64, 'conformers': 4, 'poses_per_side': 8,
                 'binary_steps': 20, 'retry_limit': 2, 'consecutive_failure_limit': 3},
        'selection': 'uniform_without_replacement_within_actual_source_balanced_v1',
        'probe': {'time': PROBE_TIME, 'dt': 0., 'geometry_names': list(GEOMETRY_NAMES),
                  'geometry_scope': 'C-alpha proxies, not full-atom validity',
                  'exact_saved_geometry': True, 'gradient': False},
        'expert_assets': {name: sha256_file(root / 'checkpoints/score_model' / name)
                          for name in ('model_parameters.yml', 'best_ema_inference_epoch_model.pt')},
        'runtime_sha256': {str(p.relative_to(ROOT)): sha256_file(p)
                           for folder in ('tricoopdiff', 'scripts')
                           for p in sorted((ROOT / folder).rglob('*.py'))
                           if 'diffdock' not in p.relative_to(ROOT).parts},
        'native_reference_available_to_generator': False,
    }
    selector = (StructuralRankingRuntime(args.selector_checkpoint.resolve(), manifest,
                                         task=args.selector_task, device=args.device)
                if args.selector_checkpoint is not None else None)
    if selector is not None:
        manifest['selection'] = dict(selector.provenance)
    output.mkdir(parents=True, exist_ok=True)
    existing = output / 'manifest.json'
    if existing.exists() and json.loads(existing.read_text()) != manifest:
        raise ValueError('archive manifest mismatch; choose a new output directory')
    atomic_json(manifest, existing)
    runner = None
    completed, failures = [], []
    started = time.monotonic()
    progress_path = output / f'worker_{args.shard_index}.json'

    def progress(state, case=None, done=0):
        atomic_json({'state': state, 'pid': os.getpid(), 'updated_unix': time.time(),
                     'elapsed_seconds': time.monotonic() - started, 'case': case,
                     'pairs_done': done, 'completed': completed, 'failures': failures,
                     'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}, progress_path)

    progress('running')
    for entry in entries[args.shard_index::args.num_shards]:
        case = entry['case']
        destination = output / f'{case}.pt'
        receipt = output / f'{case}.complete.json'
        try:
            if receipt.exists():
                prior = json.loads(receipt.read_text())
                if prior['archive_sha256'] != sha256_file(destination):
                    raise ValueError('archive receipt/hash mismatch')
                completed.append(case)
                progress('running')
                continue
            progress('generating', case)
            if runner is None:
                runner = EvaluationRunner(root, args.checkpoint.resolve(), device=args.device)
            paths = paths_by_case[case]
            e3, poi, ligand, dim = featurize_complex(
                case, str(paths['protein1']), str(paths['protein2']), str(paths['ligand']),
                runner.score_model_args, args.device, cache_dir=str(args.cache_dir.resolve()))
            runner._ensure_model(dim)
            initializer = DualExpertAnchorInitializer(runner.expert, e3, poi, binary_steps=20,
                                                     proposal_attempts=8, conformer_policy='etkdgv3_random')
            candidates, generation = generate_hybrid_candidates(
                initializer, e3['coords'][0], poi['coords'][0], ligand, seed=args.seed)
            pool = []
            for candidate in candidates:
                state = cpu_tree({'ligand': candidate['ligand'][0], 'poi': candidate['poi'][0],
                                  'body_frame': candidate['rotation'][0]})
                pool.append({'candidate_id': candidate['candidate_id'], 'source': candidate['candidate_id'][0],
                             'state': state, 'state_sha256': state_hash(state),
                             'geometry': candidate.get('geometry'), 'penalty': candidate.get('penalty'),
                             'acceptable': bool(candidate.get('acceptable', False)),
                             'fit_rmsd': candidate.get('fit_rmsd'),
                             'fallback': candidate.get('fallback')})
            selection_details = None
            if selector is None:
                selected, probability = select_archive_candidates(candidates, args.walkers, args.seed + 3_000_000)
            else:
                selected, selection_details = selector.select(
                    {'e3_data': e3, 'poi_data': poi, 'ligand_data': ligand}, pool, args.walkers)
                probability = None
            pairs = []
            for index in selected:
                candidate, initial = candidates[index], pool[index]['state']
                source = pool[index]['source']
                pair_started = time.monotonic()
                try:
                    sampled = sample_single(
                        runner.model, runner.diffusion,
                        FixedCandidateInitializer(candidate, no_torsion=runner.initializer.no_torsion),
                        e3, poi, ligand, steps=args.steps, device=args.device,
                        seed=args.seed + 2_000_000 + index, init_mode='fixed_candidate',
                        t_anchor=1., return_body_frame=True)
                    if sampled['valid'] and not all(torch.isfinite(torch.as_tensor(sampled[key])).all()
                                                    for key in ('ligand_coords', 'poi_coords', 'body_frame')):
                        sampled = {'valid': False, 'termination_reason': 'nonfinite_final_geometry'}
                except FloatingPointError as error:
                    sampled = {'valid': False, 'termination_reason': f'nonfinite_model: {error}'}
                # Probes happen after refinement and cannot affect its coordinate path.
                first_features = probe_state(runner.model, runner.diffusion, e3, poi, ligand, initial, source)
                final, final_features = None, None
                if sampled['valid']:
                    final = cpu_tree({'ligand': torch.as_tensor(sampled['ligand_coords']),
                                      'poi': torch.as_tensor(sampled['poi_coords']),
                                      'body_frame': torch.as_tensor(sampled['body_frame'])})
                    final_features = probe_state(runner.model, runner.diffusion, e3, poi, ligand, final, source)
                pairs.append({'pool_index': index, 'source': source, 'candidate_id': candidate['candidate_id'],
                              'inclusion_probability': probability[source] if probability is not None else None,
                              'initial_state_sha256': state_hash(initial),
                              'initial_features': first_features, 'final': final, 'final_features': final_features,
                              'final_state_sha256': state_hash(final) if final is not None else None,
                              'valid': sampled['valid'], 'termination_reason': sampled['termination_reason'],
                              'seconds': time.monotonic() - pair_started})
                progress('refining', case, len(pairs))
            archive = {'protocol': ARCHIVE_VERSION, 'case': case, 'group': entry['group'],
                       'split': entry['split'], 'manifest': manifest, 'pool': pool, 'pairs': pairs,
                       'probe_feature_dim': runner.model.trifuse.config['node_dim'],
                       'input_graphs': cpu_tree({'e3_data': e3, 'poi_data': poi, 'ligand_data': ligand}),
                       'mapping_contract': 'featurized order with hashed input templates and RDKit mol',
                       'generation': generation}
            if selection_details is not None:
                archive['selection_details'] = selection_details
                archive['structural_post_ranking'] = (selector.rank_pairs(archive['input_graphs'], pool, pairs)
                    if selector.final_supported else {'unavailable': 'initial-only trained checkpoint'})
            temporary = destination.with_suffix(f'.pt.tmp.{os.getpid()}')
            torch.save(archive, temporary)
            temporary.replace(destination)
            atomic_json({'case': case, 'archive_sha256': sha256_file(destination),
                         'pairs': len(pairs), 'valid': sum(p['valid'] for p in pairs)}, receipt)
            completed.append(case)
            progress('running')
            print(f'DONE {case}: {len(pairs)} archived pairs', flush=True)
        except Exception as error:
            failures.append({'case': case, 'error': f'{type(error).__name__}: {error}'})
            progress('running')
            print(f'FAIL {case}: {error}', flush=True)
            if str(args.device).startswith('cuda'):
                torch.cuda.empty_cache()
    progress('failed' if failures else 'complete')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

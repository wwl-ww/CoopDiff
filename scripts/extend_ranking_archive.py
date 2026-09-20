"""Resume fixed-pool training trajectories without regenerating binary poses."""

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import math

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
os.environ.setdefault('TRICOOPDIFF_CACHE_DIR', str(PROJECT_ROOT / 'results/runtime_cache/diffdock'))
bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument('--runtime-root', type=Path, default=ROOT)
ROOT = bootstrap.parse_known_args()[0].runtime_root.resolve()
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import tricoopdiff.diffdock_compat  # noqa: E402,F401
import torch
import numpy as np

from collect_ranking_archive import atomic_json
from evaluation_common import sha256_file
from evaluation_runner import EvaluationRunner
from tricoopdiff.expert_anchors import FixedCandidateInitializer
from tricoopdiff.ranking_archive import ARCHIVE_VERSION, cpu_tree, probe_state, state_hash
from tricoopdiff.sampler import sample_single


def device_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: device_tree(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(device_tree(item, device) for item in value)
    return value


def validate_archive(archive):
    if archive['protocol'] != ARCHIVE_VERSION or archive['split'] != 'train':
        raise ValueError('only training pose archives may be extended')
    pool, pairs = archive['pool'], archive['pairs']
    seen = set()
    for item in pool:
        if state_hash(item['state']) != item['state_sha256']:
            raise ValueError('candidate state hash mismatch')
    for pair in pairs:
        index = pair['pool_index']
        if type(index) is not int or not 0 <= index < len(pool) or index in seen:
            raise ValueError('duplicate or invalid pair index')
        seen.add(index)
        item = pool[index]
        if (pair['initial_state_sha256'] != item['state_sha256']
                or pair['source'] != item['source'] or pair['candidate_id'] != item['candidate_id']):
            raise ValueError('pair identity mismatch')
        if pair['valid']:
            if pair['final'] is None or state_hash(pair['final']) != pair['final_state_sha256']:
                raise ValueError('final state hash mismatch')
        elif pair['final'] is not None or pair['final_state_sha256'] is not None:
            raise ValueError('failed trajectory must not contain a final state')
    return [i for i in range(len(pool)) if i not in seen]


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    torch.save(value, temporary)
    temporary.replace(path)


def same_tree(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_tree(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same_tree(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right, equal_nan=True)
    if hasattr(left, 'ToBinary'):
        return type(left) is type(right) and left.ToBinary() == right.ToBinary()
    if isinstance(left, float) and math.isnan(left):
        return isinstance(right, float) and math.isnan(right)
    return left == right


def validate_extension(parent, extended):
    validate_archive(extended)
    if not same_tree(parent['pool'], extended['pool']):
        raise ValueError('extension changed candidate pool')
    if not same_tree(parent['input_graphs'], extended['input_graphs']):
        raise ValueError('extension changed input templates')
    pairs = {p['pool_index']: p for p in extended['pairs']}
    for pair in parent['pairs']:
        if not same_tree(pair, pairs.get(pair['pool_index'])):
            raise ValueError('extension changed a previously observed trajectory')


def refine_pair(runner, archive, index, inputs):
    item = archive['pool'][index]
    initial, source = item['state'], item['source']
    candidate = {**item, 'ligand': initial['ligand'].to(runner.device)[None],
                 'poi': initial['poi'].to(runner.device)[None],
                 'rotation': initial['body_frame'].to(runner.device)[None]}
    e3, poi, ligand = (inputs[key] for key in ('e3_data', 'poi_data', 'ligand_data'))
    started = time.monotonic()
    manifest = archive['manifest']
    try:
        result = sample_single(runner.model, runner.diffusion,
            FixedCandidateInitializer(candidate, no_torsion=runner.initializer.no_torsion),
            e3, poi, ligand, steps=manifest['steps'], device=runner.device,
            seed=manifest['proposal_seed'] + 2_000_000 + index,
            init_mode='fixed_candidate', t_anchor=1., return_body_frame=True)
        if result['valid'] and not all(torch.isfinite(torch.as_tensor(result[key])).all()
                                      for key in ('ligand_coords', 'poi_coords', 'body_frame')):
            result = {'valid': False, 'termination_reason': 'nonfinite_final_geometry'}
    except FloatingPointError as error:
        result = {'valid': False, 'termination_reason': f'nonfinite_model: {error}'}
    final = (cpu_tree({'ligand': torch.as_tensor(result['ligand_coords']),
                       'poi': torch.as_tensor(result['poi_coords']),
                       'body_frame': torch.as_tensor(result['body_frame'])}) if result['valid'] else None)
    return {'pool_index': index, 'candidate_id': item['candidate_id'], 'source': source,
            'inclusion_probability': 1., 'initial_state_sha256': item['state_sha256'],
            'initial_features': probe_state(runner.model, runner.diffusion, e3, poi, ligand, initial, source),
            'final': final, 'final_state_sha256': state_hash(final) if final is not None else None,
            'final_features': probe_state(runner.model, runner.diffusion, e3, poi, ligand, final, source)
                              if final is not None else None,
            'valid': result['valid'], 'termination_reason': result['termination_reason'],
            'seconds': time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=ROOT,
                        help='original immutable runtime snapshot; imports and refinement hashes use this root')
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--max-new-pairs', type=int, help='smoke limit; incomplete cases get no completion receipt')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    source, output, checkpoint = (p.resolve() for p in (args.archives, args.out, args.checkpoint))
    if (source == output or source in output.parents or output in source.parents
            or not 0 <= args.shard_index < args.num_shards
            or args.max_new_pairs is not None and args.max_new_pairs < 1):
        parser.error('use separate non-nested output and valid shard/smoke limits')
    os.chdir(PROJECT_ROOT)
    parent = json.loads((source / 'manifest.json').read_text())
    if sha256_file(checkpoint) != parent['checkpoint_sha256']:
        raise ValueError('generator checkpoint mismatch')
    if parent['diffusion_seed'] != 'proposal_seed + 2000000 + pool_index':
        raise ValueError('unsupported parent seed protocol')
    for name, digest in parent['expert_assets'].items():
        if sha256_file(PROJECT_ROOT / 'checkpoints/score_model' / name) != digest:
            raise ValueError(f'expert asset changed: {name}')
    # Ranking-only edits are expected, but the refinement path must be identical.
    for name in ('tricoopdiff/sampler.py', 'tricoopdiff/sdpi_flow.py', 'tricoopdiff/consensus.py',
                 'tricoopdiff/geometry_encoders.py', 'scripts/evaluation_runner.py'):
        if sha256_file(ROOT / name) != parent['runtime_sha256'].get(name):
            raise ValueError(f'refinement runtime changed: {name}; use the original runtime snapshot')
    seals = {}
    missing_total = 0
    for case in parent['cases']:
        if Path(case).name != case or case in ('.', '..'):
            raise ValueError('unsafe case name')
        receipt = json.loads((source / f'{case}.complete.json').read_text())
        digest = sha256_file(source / f'{case}.pt')
        if receipt['archive_sha256'] != digest:
            raise ValueError('source receipt mismatch')
        archive = torch.load(source / f'{case}.pt', map_location='cpu', weights_only=False)
        if archive['manifest'] != parent or archive['case'] != case:
            raise ValueError('source manifest/case mismatch')
        if len(archive['pool']) != 256:
            raise ValueError('this extension protocol requires the archived 256-candidate pool')
        missing_total += len(validate_archive(archive))
        seals[case] = digest
    print(f'AUDIT cases={len(seals)} missing_pairs={missing_total}', flush=True)
    if args.audit_only:
        return
    manifest = {**parent, 'selection': 'complete_existing_pool_v1', 'walkers': 256,
                'extension': {'parent_manifest_sha256': sha256_file(source / 'manifest.json'),
                              'parent_archives': seals, 'entrypoint_sha256': sha256_file(Path(__file__))}}
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (output / 'manifest.json').exists():
            if json.loads((output / 'manifest.json').read_text()) != manifest:
                raise ValueError('extension manifest mismatch')
        else:
            atomic_json(manifest, output / 'manifest.json')
    runner, added = None, 0
    for case in parent['cases'][args.shard_index::args.num_shards]:
        with (output / f'{case}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            destination = output / f'{case}.pt'
            receipt = output / f'{case}.complete.json'
            if receipt.exists():
                if json.loads(receipt.read_text())['archive_sha256'] != sha256_file(destination):
                    raise ValueError('completed extension hash mismatch')
                continue
            parent_archive = torch.load(source / f'{case}.pt', map_location='cpu', weights_only=False)
            if sha256_file(source / f'{case}.pt') != seals[case]:
                raise ValueError('source changed during extension')
            archive = torch.load(destination, map_location='cpu', weights_only=False) if destination.exists() else copy.deepcopy(parent_archive)
            if destination.exists() and archive['manifest'] != manifest:
                raise ValueError('partial extension manifest mismatch')
            archive['manifest'] = manifest
            missing = validate_archive(archive)
            validate_extension(parent_archive, archive)
            if runner is None and missing:
                runner = EvaluationRunner(PROJECT_ROOT, checkpoint, device=args.device)
            if missing:
                runner._ensure_model(archive['input_graphs']['e3_data']['features'].shape[-1])
                inputs = device_tree(archive['input_graphs'], runner.device)
            for index in missing:
                archive['pairs'].append(refine_pair(runner, archive, index, inputs))
                atomic_save(archive, destination)
                added += 1
                print(f'PAIR {case} index={index} total={len(archive["pairs"])}', flush=True)
                if args.max_new_pairs is not None and added >= args.max_new_pairs:
                    return
            atomic_save(archive, destination)
            atomic_json({'case': case, 'archive_sha256': sha256_file(destination),
                         'pairs': len(archive['pairs']), 'valid': sum(p['valid'] for p in archive['pairs'])}, receipt)


if __name__ == '__main__':
    main()

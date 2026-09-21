"""Audit chemical-cut training applicability using existing validated caches only."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import tricoopdiff.diffdock_compat  # noqa: F401
from featurize_common import featurize_complex
from train_entry import _load_index, _load_score_model_args
from tricoopdiff.anchored_training import prepare_augmented_anchored_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--expert-model-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--max-cases', type=int)
    parser.add_argument('--single-anchor', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('choose a new audit output')
    if args.max_cases is not None and args.max_cases < 1:
        parser.error('--max-cases must be positive')
    torch.set_num_threads(1)
    entries, clusters, by_case = _load_index(args.index)
    payload = json.loads(args.index.read_text())
    excluded = payload.get('metadata', {}).get('excluded_benchmark_pdb_ids')
    if not excluded:
        raise ValueError('training index must declare excluded benchmark PDB IDs')
    excluded = {case.upper() for case in excluded}
    if any(row['case'].split('_')[0].upper() in excluded for row in entries):
        raise ValueError('benchmark contamination in training index')
    if clusters:
        entries = [by_case[next(case for case in (cluster.get('representative') or cluster['items'])
                                if case in by_case)] for cluster in clusters]
    if args.max_cases:
        entries = entries[:args.max_cases]
    score_args = _load_score_model_args(str(args.expert_model_dir))
    rows = []
    for index, entry in enumerate(entries):
        case = entry['case']
        # Case-local seed makes eligibility independent of traversal/cache order.
        seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{case}'.encode()).digest()[:8], 'big')
        row = {'case': case, 'seed': seed}
        try:
            e3, poi, ligand, _ = featurize_complex(
                case, entry['protein1'], entry['protein2'], entry['ligand'], score_args, 'cpu',
                cache_dir=str(args.cache_dir), cache_read_only=True, cache_wait_seconds=0)
            prepare = prepare_augmented_anchored_sample
            if args.single_anchor:
                from tricoopdiff.single_anchor_training import prepare_single_anchor_sample
                prepare = prepare_single_anchor_sample
            sample = prepare(
                e3, poi, ligand, generator=torch.Generator().manual_seed(seed),
                evidence_id=f'training-index:{case}')
            row.update(status='eligible' if sample is not None else 'no_chemical_cut_pair')
            if sample is not None:
                row['audit'] = sample.audit
                row['active_torsions'] = len(sample.flow.plan.bonds)
        except TimeoutError as error:
            row.update(status='cache_unavailable_or_stale', error=str(error))
        except (ValueError, RuntimeError) as error:
            row.update(status='geometry_or_chemistry_rejected', error=str(error))
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(json.dumps({'processed': index + 1, 'counts': dict(Counter(r['status'] for r in rows))}), flush=True)
    result = {'scope': 'one representative per training cluster; not full dataset coverage',
              'task': 'single_anchor' if args.single_anchor else 'dual_anchor',
              'single_known_side_bounds': {'translation': .5, 'rotation': .2} if args.single_anchor else None,
              'seed': args.seed, 'index_sha256': hashlib.sha256(args.index.read_bytes()).hexdigest(),
              'benchmark_overlap': [], 'cache_policy': 'validated read-only; no featurization/downloads',
              'code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ('scripts/audit_anchored_training.py',
                                           'tricoopdiff/anchored_fragments.py',
                                           'tricoopdiff/anchored_training.py',
                                           'tricoopdiff/anchored_flow.py',
                                           'tricoopdiff/anchored_torsion.py',
                                           'tricoopdiff/single_anchor_flow.py',
                                           'tricoopdiff/single_anchor_training.py')},
              'counts': dict(Counter(row['status'] for row in rows)), 'cases': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result['counts']), flush=True)


if __name__ == '__main__':
    main()

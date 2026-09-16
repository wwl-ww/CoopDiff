"""Two explicitly assigned GPUs: collect, score separately, then fit two R1 heads."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def pipeline_budget(run, train_walkers, val_walkers):
    """Check the declared panel before starting either persistent GPU worker."""
    if not all(1 <= n <= 256 for n in (train_walkers, val_walkers)):
        raise ValueError('walker budgets must be between 1 and 256')
    audit = json.loads((run / 'indexes/audit.json').read_text())
    budget, pdbs = {}, {}
    fields = {'case', 'group', 'split', 'protein1', 'protein2', 'ligand'}
    for split, walkers in (('train', train_walkers), ('val', val_walkers)):
        payload = json.loads((run / 'indexes' / f'{split}_inference.json').read_text())
        if set(payload) != {'entries'} or not payload['entries']:
            raise ValueError('nonempty inference-only panel required')
        entries = payload['entries']
        if any(set(e) != fields or e['split'] != split for e in entries):
            raise ValueError('incorrect inference schema or split')
        cases = [e['case'] for e in entries]
        if (len(set(cases)) != len(cases) or set(cases) != set(audit[f'{split}_cases'])
                or any(c != Path(c).name or c in ('.', '..') for c in cases)):
            raise ValueError('unsafe, duplicate or unapproved cases')
        pdbs[split] = {c.split('_')[0].upper() for c in cases}
        if pdbs[split] & set(audit['benchmark_pdbs_excluded']):
            raise ValueError('benchmark PDB in internal panel')
        budget.update({f'{split}_cases': len(cases), f'{split}_walkers': walkers})
    if pdbs['train'] & pdbs['val']:
        raise ValueError('train/validation PDB overlap')
    budget['total_trajectories'] = sum(budget[f'{s}_cases'] * budget[f'{s}_walkers']
                                       for s in ('train', 'val'))
    return budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs=2, required=True)
    parser.add_argument('--train-walkers', type=int, default=32)
    parser.add_argument('--val-walkers', type=int, default=256)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2:
        parser.error('two distinct authorized physical GPUs are required')
    root, run, runtime = args.project_root.resolve(), args.run_dir.resolve(), args.runtime.resolve()
    status_path = run / 'pipeline_status.json'
    if status_path.exists():
        raise FileExistsError('pipeline status already exists; do not duplicate live workers')
    budget = pipeline_budget(run, args.train_walkers, args.val_walkers)
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTHONDONTWRITEBYTECODE='1', TRICOOPDIFF_CACHE_DIR=str(root / 'results/runtime_cache/diffdock'))
    started = time.time()
    status = {'pid': os.getpid(), 'gpus': args.gpus, 'runtime': str(runtime), 'started_unix': started,
              'budget': budget}

    def write_status(stage, results=None):
        status.update(stage=stage, updated_unix=time.time(), results=results)
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(status_path)
        print(stage, results, flush=True)

    def execute(script, argv, log_name, gpu=None):
        command = [sys.executable, '-u', str(runtime / 'scripts' / script)] + list(map(str, argv))
        env = dict(environment, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
        with (run / log_name).open('x') as log:
            print(json.dumps({'command': command, 'gpu': gpu}), file=log, flush=True)
            return subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode

    def worker(shard):
        results = {}
        for split, walkers in (('train', args.train_walkers), ('val', args.val_walkers)):
            results[split] = execute('collect_ranking_archive.py', [
                '--project-root', root, '--index', run / 'indexes' / f'{split}_inference.json',
                '--checkpoint', run / 'checkpoint_epoch40.pt', '--out', run / f'{split}_archives',
                '--cache-dir', root / 'results/featurize_cache_proposal_oracle',
                '--walkers', walkers, '--num-shards', 2, '--shard-index', shard],
                f'{split}_gpu{args.gpus[shard]}.log', args.gpus[shard])
        return results

    write_status('collecting')
    with ThreadPoolExecutor(max_workers=2) as pool:
        collection = list(pool.map(worker, (0, 1)))
    write_status('scoring', collection)

    def score(split):
        return execute('score_ranking_archive.py', [
            '--project-root', root, '--deepternary-root', root / 'DeepTernary',
            '--archives', run / f'{split}_archives', '--label-refs', run / 'indexes' / f'{split}_label_refs.json',
            '--out', run / f'{split}_labels'], f'score_{split}.log')

    with ThreadPoolExecutor(max_workers=2) as pool:
        scored = list(pool.map(score, ('train', 'val')))
    if any(code for worker_result in collection for code in worker_result.values()) or any(scored):
        write_status('blocked_incomplete_data', {'collection': collection, 'scoring': scored})
        raise SystemExit(1)
    write_status('auditing_labels')

    def audit_labels(split):
        return execute('audit_ranking_label_balance.py', [
            '--archives', run / f'{split}_archives', '--labels', run / f'{split}_labels',
            '--out', run / f'{split}_label_support.json'], f'audit_labels_{split}.log')

    with ThreadPoolExecutor(max_workers=2) as pool:
        audited = list(pool.map(audit_labels, ('train', 'val')))
    if any(audited):
        write_status('blocked_label_audit', audited)
        return 1
    write_status('training_rankers')

    def train(shard):
        mode = 'geometry' if shard == 0 else 'frozen_geometry'
        argv = ['--train-archives', run / 'train_archives', '--train-labels', run / 'train_labels',
                '--val-archives', run / 'val_archives', '--val-labels', run / 'val_labels',
                '--split-audit', run / 'indexes/audit.json', '--out', run / f'ranker_{mode}',
                '--device', 'cuda', '--diagnostics']
        if shard == 0:
            argv.append('--geometry-only')
        return execute('train_state_ranker.py', argv, f'ranker_{mode}.log', args.gpus[shard])

    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = list(pool.map(train, (0, 1)))
    write_status('complete' if not any(trained) else 'ranker_failed', trained)
    return int(any(trained))


if __name__ == '__main__':
    sys.exit(main())

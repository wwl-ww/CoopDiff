"""Score all 256 initial candidates for a separate pre-selector experiment."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import tricoopdiff.diffdock_compat  # noqa: E402,F401
import torch

from tricoopdiff.runtime import atomic_json
from evaluation_common import evaluation_input_signature, sha256_file
from tricoopdiff.deepternary_metrics import compute_fullatom_dockq
from tricoopdiff.ranking.initial_selector import POOL_LABEL_VERSION, initial_pool_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--label-refs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deepternary-root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        parser.error('invalid scoring shard')
    project, deep = args.project_root.resolve(), args.deepternary_root.resolve()
    archives, refs, out = args.archives.resolve(), args.label_refs.resolve(), args.out.resolve()
    os.chdir(project)
    entries = json.loads(refs.read_text())['entries']
    source_manifest = json.loads((archives / 'manifest.json').read_text())
    wanted = source_manifest['cases']
    by_case = {row['case']: row for row in entries}
    if set(wanted) - set(by_case) or len(by_case) != len(entries):
        raise ValueError('missing or duplicate label references')
    manifest = {
        'protocol': POOL_LABEL_VERSION,
        'scope': 'all initial pool states; no refined states',
        'pool_size': 256, 'selection_budget': 64,
        'references_sha256': sha256_file(refs),
        'archive_manifest_sha256': sha256_file(archives / 'manifest.json'),
        'entrypoint_sha256': sha256_file(Path(__file__)),
        'feature_source_sha256': sha256_file(ROOT / 'tricoopdiff/ranking/initial_selector.py'),
        'metric_sources': {str(path): sha256_file(path) for path in
                           [ROOT / 'tricoopdiff/deepternary_metrics.py']
                           + sorted((deep / 'DockQ').rglob('*.py'))},
        'references': {row['case']: evaluation_input_signature(
            {key: row[key] for key in ('protein1', 'protein2', 'ligand')},
            row['native_complex_pdb']) for row in entries},
    }
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('initial-pool label manifest mismatch')
    atomic_json(manifest, manifest_path)
    assigned = [by_case[case] for case in wanted][args.shard_index::args.num_shards]
    progress = out / (f'worker_{args.shard_index}.json' if args.num_shards > 1 else 'progress.json')
    completed, failures, missing = [], [], []
    for entry in assigned:
        case = entry['case']
        archive_path, receipt_path = archives / f'{case}.pt', archives / f'{case}.complete.json'
        if not receipt_path.is_file():
            missing.append(case)
            continue
        try:
            digest = sha256_file(archive_path)
            if json.loads(receipt_path.read_text())['archive_sha256'] != digest:
                raise ValueError('archive checksum mismatch')
            destination = out / f'{case}.json'
            if destination.exists():
                if json.loads(destination.read_text()).get('archive_sha256') != digest:
                    raise ValueError('existing initial-pool label checksum mismatch')
                completed.append(case)
                continue
            archive = torch.load(archive_path, map_location='cpu', weights_only=False)
            expected = evaluation_input_signature({key: entry[key] for key in ('protein1', 'protein2', 'ligand')})
            if expected != archive['manifest']['input_signatures'][case]:
                raise ValueError('scorer templates do not match generation inputs')
            rows = initial_pool_features(archive)
            e3 = archive['input_graphs']['e3_data']['coords'][0].numpy()
            for row in rows:
                state = archive['pool'][row['pool_index']]['state']
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    metrics = compute_fullatom_dockq(
                        state['ligand'].numpy(), state['poi'].numpy(), e3,
                        entry['protein1'], entry['protein2'], entry['ligand'],
                        entry['native_complex_pdb'], str(deep))
                row['geometry'] = row['geometry'].tolist()
                row['dockq'] = metrics
            atomic_json({'case': case, 'archive_sha256': digest, 'rows': rows}, destination)
            completed.append(case)
            print(f'SCORED INITIAL POOL {case}: {len(rows)}', flush=True)
        except Exception as error:
            failures.append({'case': case, 'error': f'{type(error).__name__}: {error}'})
            print(f'FAIL {case}: {error}', flush=True)
        atomic_json({'completed': completed, 'failures': failures, 'missing': missing}, progress)
    atomic_json({'completed': completed, 'failures': failures, 'missing': missing}, progress)
    if failures or missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

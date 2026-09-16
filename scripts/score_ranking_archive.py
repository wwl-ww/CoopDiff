"""Label immutable pose archives; native references never enter the generator."""

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

from collect_ranking_archive import atomic_json
from evaluation_common import evaluation_input_signature, sha256_file
from tricoopdiff.deepternary_metrics import compute_fullatom_dockq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--label-refs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deepternary-root', type=Path, default=ROOT / 'DeepTernary')
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error('invalid scoring shard')
    os.chdir(args.project_root.resolve())
    entries = json.loads(args.label_refs.read_text())['entries']
    wanted = json.loads((args.archives / 'manifest.json').read_text())['cases']
    by_case = {e['case']: e for e in entries}
    if set(wanted) - set(by_case):
        raise ValueError('missing label references for archive cases')
    entries = [by_case[case] for case in wanted]
    if args.out.resolve() == args.archives.resolve():
        parser.error('labels must be written separately from inference archives')
    manifest = {'protocol': 'exact_archive_labels_v1', 'references_sha256': sha256_file(args.label_refs),
                'archive_manifest_sha256': sha256_file(args.archives / 'manifest.json'),
                'entrypoint_sha256': sha256_file(Path(__file__)),
                'metric_sources': {str(p): sha256_file(p) for p in [ROOT / 'tricoopdiff/deepternary_metrics.py']
                                   + sorted((args.deepternary_root / 'DockQ').rglob('*.py'))},
                'references': {e['case']: evaluation_input_signature(
                    {k: e[k] for k in ('protein1', 'protein2', 'ligand')}, e['native_complex_pdb']) for e in entries}}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('label manifest mismatch; use a new directory')
    atomic_json(manifest, manifest_path)
    entries = entries[args.shard_index::args.num_shards]
    progress_path = args.out / ('progress.json' if args.num_shards == 1 else f'worker_{args.shard_index}.json')
    completed, failures, missing = [], [], []
    for entry in entries:
        case = entry['case']
        archive_path, receipt_path = args.archives / f'{case}.pt', args.archives / f'{case}.complete.json'
        if not receipt_path.is_file():
            missing.append(case)
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
            archive_sha = sha256_file(archive_path)
            if receipt['archive_sha256'] != archive_sha:
                raise ValueError('archive checksum mismatch')
            destination = args.out / f'{case}.json'
            if destination.exists():
                previous = json.loads(destination.read_text())
                if previous.get('archive_sha256') != archive_sha:
                    raise ValueError('existing label archive checksum mismatch')
                completed.append(case)
                continue
            archive = torch.load(archive_path, map_location='cpu', weights_only=False)
            expected_input = evaluation_input_signature({k: entry[k] for k in ('protein1', 'protein2', 'ligand')})
            if expected_input != archive['manifest']['input_signatures'][case]:
                raise ValueError('scorer templates do not match generation input signature')
            e3 = archive['input_graphs']['e3_data']['coords'][0].numpy()

            def score(state):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    return compute_fullatom_dockq(
                        state['ligand'].numpy(), state['poi'].numpy(), e3,
                        entry['protein1'], entry['protein2'], entry['ligand'],
                        entry['native_complex_pdb'], str(args.deepternary_root))

            labels = []
            for pair in archive['pairs']:
                index = pair['pool_index']
                initial = score(archive['pool'][index]['state'])
                final = score(pair['final']) if pair['valid'] else None
                labels.append({'pool_index': index, 'initial': initial, 'final': final,
                               'valid': pair['valid'], 'union_dockq': max(initial['dockq'], final['dockq'])
                               if final is not None else initial['dockq']})
            atomic_json({'case': case, 'archive_sha256': archive_sha, 'labels': labels}, destination)
            completed.append(case)
            print(f'SCORED {case}: {len(labels)} pairs', flush=True)
        except Exception as error:
            failures.append({'case': case, 'error': f'{type(error).__name__}: {error}'})
            print(f'FAIL {case}: {error}', flush=True)
        atomic_json({'completed': completed, 'failures': failures, 'missing': missing}, progress_path)
    atomic_json({'completed': completed, 'failures': failures, 'missing': missing}, progress_path)
    if failures or missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Create a fixed-seed, label-blind internal ranking panel without editing source indexes."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluation_common import resolve_entry_paths, sha256_file
from make_deepternary_index import read_case_list


def select_groups(payload, count, excluded_pdbs, seed):
    if count < 1:
        raise ValueError('group count must be positive')
    by_case = {e['case']: e for e in payload['entries']}
    if len(by_case) != len(payload['entries']):
        raise ValueError('duplicate source cases')
    clusters = payload.get('clusters') or [{'items': [c]} for c in sorted(by_case)]
    eligible = []
    for i, cluster in enumerate(clusters):
        members = cluster['items']
        if any(c.split('_')[0].upper() in excluded_pdbs for c in members):
            continue
        entries = [by_case[c] for c in sorted(members) if c in by_case]
        if entries:
            eligible.append((i, entries[0]))
    random.Random(seed).shuffle(eligible)
    selected, pdbs = [], set()
    for i, entry in eligible:
        pdb = entry['case'].split('_')[0].upper()
        if pdb not in pdbs:
            selected.append((i, entry))
            pdbs.add(pdb)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f'only {len(selected)} eligible groups for requested {count}')
    return selected


def prepare_extension(base_indexes, out):
    """Declare only unseen suffix cases while retaining immutable base archives."""
    base_audit = json.loads((base_indexes / 'audit.json').read_text())
    full_audit = json.loads((out / 'audit.json').read_text())
    for key in ('seed', 'source_sha256', 'benchmark_pdbs_excluded'):
        if base_audit[key] != full_audit[key]:
            raise ValueError(f'panel extension audit mismatch: {key}')
    payloads = {}
    for split in ('train', 'val'):
        for kind in ('inference', 'label_refs'):
            name = f'{split}_{kind}.json'
            before = json.loads((base_indexes / name).read_text())['entries']
            after = json.loads((out / name).read_text())['entries']
            if (split == 'train' and (after[:len(before)] != before or len(after) <= len(before))
                    or split == 'val' and before != after):
                raise ValueError('extension must keep training prefix and exact validation panel')
            if split == 'train':
                payloads[f'extra_train_{kind}.json'] = {'entries': after[len(before):]}
    if any((out / name).exists() for name in payloads):
        raise FileExistsError('extension declarations already exist')
    for name, payload in payloads.items():
        with (out / name).open('x') as handle:
            json.dump(payload, handle, indent=2)
            handle.write('\n')
    return len(payloads['extra_train_inference.json']['entries'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=16)
    parser.add_argument('--val-count', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--reuse-indexes', type=Path,
                        help='Require this exact training prefix/validation set; also declare an extra-only index')
    args = parser.parse_args()
    if min(args.train_count, args.val_count) < 1:
        parser.error('counts must be positive')
    payloads = {split: json.loads((ROOT / f'data/ternary_{split}.json').read_text())
                for split in ('train', 'val')}
    train_pdbs = {e['case'].split('_')[0].upper() for e in payloads['train']['entries']}
    val_pdbs = {e['case'].split('_')[0].upper() for e in payloads['val']['entries']}
    benchmark_paths = [ROOT / 'DeepTernary/data/PROTAC/protac22.txt',
                       ROOT / 'DeepTernary/data/MolecularGlue/test_all.txt']
    excluded = {c.split('_')[0].upper() for p in benchmark_paths for c in read_case_list(p)}
    chosen = {
        'train': select_groups(payloads['train'], args.train_count, excluded, args.seed),
        'val': select_groups(payloads['val'], args.val_count, excluded | train_pdbs, args.seed),
    }
    documents = {}
    for split, selection in chosen.items():
        inputs, labels = [], []
        for group, entry in selection:
            paths = resolve_entry_paths(entry, ROOT)
            native = ROOT / 'DeepTernary/data/TernaryDB/pdbs' / entry['case'] / 'gt_complex.pdb'
            if not native.is_file():
                raise FileNotFoundError(native)
            inputs.append({'case': entry['case'], 'split': split, 'group': f'{split}_cluster_{group}',
                           **{key: str(value) for key, value in paths.items()}})
            labels.append({'case': entry['case'], 'native_complex_pdb': str(native),
                           **{key: str(value) for key, value in paths.items()}})
        documents[f'{split}_inference.json'] = {'entries': inputs}
        documents[f'{split}_label_refs.json'] = {'entries': labels}
    documents['audit.json'] = {
        'protocol': 'ranking_pipeline_smoke_v1', 'seed': args.seed,
        'budget': {'train_cases': args.train_count, 'val_cases': args.val_count},
        'scope': 'pipeline smoke, not family/ligand-disjoint or untouched generator holdout',
        'c40_exposure': 'train groups seen by generator; validation used for generator model selection',
        'original_train_val_shared_pdbs': sorted(train_pdbs & val_pdbs),
        'selection': 'shuffle eligible clusters, first sorted member, unique PDB, no pose labels',
        'train_cases': [e['case'] for _, e in chosen['train']],
        'val_cases': [e['case'] for _, e in chosen['val']],
        'validation_excludes_all_original_training_pdbs': True,
        'benchmark_pdbs_excluded': sorted(excluded),
        'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p)
                          for p in benchmark_paths + [ROOT / f'data/ternary_{s}.json' for s in ('train', 'val')]},
    }
    args.out.mkdir(parents=True, exist_ok=False)
    for name, document in documents.items():
        (args.out / name).write_text(json.dumps(document, indent=2) + '\n')
    if args.reuse_indexes:
        prepare_extension(args.reuse_indexes, args.out)
    print(json.dumps(documents['audit.json'], indent=2))


if __name__ == '__main__':
    main()

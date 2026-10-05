"""Registered eight-train/four-validation pilot using the production entry.

Only index selection, epoch-zero checkpointing and fixed validation starts are
wrapped. Optimization, endpoint losses, EMA, scheduler and split audits remain
owned by train_entry. This is an A1/SDPI development test, not generalization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch
import train_entry as entry
from tricoopdiff.initialization import TernaryInitializer


def state_digest(values):
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        value = value.detach().cpu().contiguous()
        digest.update(str((index, tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prepare_protocol(output, train_cases=None, validation_cases=None, epochs=5):
    source = ROOT / 'results/model45_small_sample/v39_full/cache_subset.json'
    old = json.loads(source.read_text())
    selected = {key: old[key]['selected_entries'] for key in ('train', 'validation')}
    if train_cases is not None:
        selected['train'] = list(train_cases)
    if validation_cases is not None:
        selected['validation'] = list(validation_cases)
    if not selected['train'] or not selected['validation']:
        raise ValueError('pilot requires non-empty train and validation case lists')
    selection_hash = hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, source_name in (('train', 'ternary_train.json'), ('validation', 'ternary_val.json')):
        parent = ROOT / 'results/model45_partition' / source_name
        payload = json.loads(parent.read_text())
        entries = {value['case']: value for value in payload['entries']}
        subset = {'entries': [entries[case] for case in selected[split]], 'rejected': [],
            'clusters': [{'representative': [case], 'items': [case]} for case in selected[split]],
            'metadata': {**payload['metadata'], 'pilot_selection_protocol': 'registered_fixed_representatives_8_4_v1',
                         'subset_parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest()}}
        paths[split] = output / source_name
        paths[split].write_text(json.dumps(subset, indent=2))
    from make_deepternary_index import audit_index_files
    audit = audit_index_files(paths['train'], paths['validation'])
    original = ROOT.parent / 'CoopDiff'
    argv = ['--train_data', str(paths['train']), '--val_data', str(paths['validation']),
        '--output_dir', str(output), '--training-arm', 'A1', '--no-random-flip-proteins',
        '--epochs', str(epochs), '--batch_size', '1', '--seed', '73', '--lr', '1e-4',
        '--weight_decay', '1e-4', '--nan-clamp', '100', '--node_dim', '128',
        '--expert_model_dir', str(original / 'checkpoints/score_model'),
        '--cache_dir', str(original / 'results/featurize_cache_proposal_oracle'), '--cache-only',
        '--native-root', str(original / 'DeepTernary/data/TernaryDB/pdbs'),
        '--log_interval', '1', '--save_interval', '1', '--device', 'cuda']
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    registered_source = '70f1e9258bcce6aaff49d6a1716fbed9433e189e'
    if subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor',
                       registered_source, commit], check=False).returncode:
        raise ValueError('pilot must descend from the registered v41 source commit')
    protocol = {'source_commit': commit, 'selected_cases': selected, 'selection_sha256': selection_hash,
        'selection_source': str(source), 'selection_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'data_partition': audit, 'entry_argv': argv, 'model_profile': 'ternary_local_compatibility_v1',
        'checkpoint_version': 41, 'training_arm': 'A1', 'initialization': 'SDPI',
        'epochs': epochs, 'updates_expected': epochs * len(selected['train']), 'seed': 73, 'lr': 1e-4, 'node_dim': 128, 'batch_size': 1,
        'zero_local_baseline': 'GPU1 evaluates epoch0 frozen experts with local cost heads zeroed',
        'validation_initial_state_file': str(output / 'validation_initial_states.pt'),
        'validation_initialization': 'case_seed(case,0); actual tensors shared with independent evaluation',
        'final_evaluation': ('independent GPU1 paired raw checkpoint versus zero-local on the same actual states; '
                             'EMA is exploratory only because short-run averaging lags training'),
        'scope': 'cross-system small development pilot; no generalization or BiTSeed coverage claim',
        'code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            'tricoopdiff/models/consensus.py', 'tricoopdiff/models/local_compatibility.py',
            'tricoopdiff/train.py', 'tricoopdiff/sdpi_flow.py', 'scripts/train_entry.py')},
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps({'protocol': str(output / 'protocol.json'), 'selection_sha256': selection_hash,
                      'selected_cases': selected, 'audit': audit}), flush=True)
    return protocol


def install_hooks(output):
    original_build = entry.build_trainer
    def build(args, device, input_dim=None):
        model, flow, trainer = original_build(args, device, input_dim=input_dim)
        trainer.data_partition = args.data_partition
        trainer.save_checkpoint(output / 'epoch0.pt', epoch=0)
        original_validate = trainer.validate_step
        states = None
        def validate_step(**batch):
            nonlocal states
            if states is None:
                path = output / 'validation_initial_states.pt'
                deadline = time.monotonic() + 900
                while not path.exists():
                    if time.monotonic() > deadline:
                        raise RuntimeError('independent evaluator did not publish validation starts')
                    time.sleep(1)
                states = torch.load(path, map_location=device, weights_only=False)
            case = batch['case']
            start = states[case]['initial_state']
            digest = state_digest([start[key] for key in ('ligand', 'poi', 'body_frame')])
            with (output / 'validation_state_hashes.jsonl').open('a') as handle:
                handle.write(json.dumps({'case': case, 'global_step': trainer.global_step,
                                         'state_sha256': digest}) + '\n')
            return original_validate(**{**batch, 'coupling_override': {'initial_state': start}})
        trainer.validate_step = validate_step
        return model, flow, trainer
    entry.build_trainer = build
    active_case = {'case': None}
    original_train_batch = entry._guarded_train_batch
    def train_batch(trainer, batches):
        if len(batches) != 1:
            raise ValueError('registered pilot uses batch size one')
        active_case['case'] = batches[0]['case']
        try:
            return original_train_batch(trainer, batches)
        finally:
            active_case['case'] = None
    entry._guarded_train_batch = train_batch
    original_initialize = TernaryInitializer.initialize
    def initialize(self, *args, **kwargs):
        result = original_initialize(self, *args, **kwargs)
        if active_case['case'] is not None:
            with (output / 'training_initial_state_hashes.jsonl').open('a') as handle:
                handle.write(json.dumps({'case': active_case['case'], 'seed': kwargs.get('seed'),
                                         'state_sha256': state_digest(result[:3])}) + '\n')
        return result
    TernaryInitializer.initialize = initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/model41_cross_system_pilot')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--train-case', action='append', dest='train_cases')
    parser.add_argument('--validation-case', action='append', dest='validation_cases')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    output = (ROOT / args.out).resolve()
    if (output / 'metrics.jsonl').exists():
        raise ValueError('refusing to overwrite an existing pilot')
    protocol = prepare_protocol(output, args.train_cases, args.validation_cases, args.epochs)
    if args.prepare_only:
        return
    (output / 'train.pid').write_text(str(os.getpid()))
    install_hooks(output)
    sys.argv = ['train_entry.py', *protocol['entry_argv']]
    entry.main()


if __name__ == '__main__':
    main()

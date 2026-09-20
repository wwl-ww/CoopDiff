"""Wait for fixed-epoch checkpoints, then evaluate each on the same sealed data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tricoopdiff.diffdock_compat  # noqa: F401
import torch
from train_structural_ranker import evaluate
from tricoopdiff.structural_ranker import StructuralQualityRanker
from tricoopdiff.structural_ranking_data import (
    StructuralArchiveDataset, StructuralJointDataset, audit_datasets, file_digest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--wait-hours', type=float, default=12.)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.epoch < 1 or not 0 < args.wait_hours <= 24:
        parser.error('new output directory, positive epoch and 0<wait-hours<=24 required')
    args.out.mkdir(parents=True)
    pending, complete = list(args.runs), []
    print(f'waiting for epoch={args.epoch}: {[str(run) for run in pending]}', flush=True)
    (args.out / 'progress.json').write_text(json.dumps({'complete': [],
        'pending': [str(run) for run in pending], 'epoch': args.epoch}, indent=2) + '\n')
    deadline = time.monotonic() + args.wait_hours * 3600
    while pending:
        if time.monotonic() >= deadline:
            raise TimeoutError(f'fixed-epoch checkpoints did not complete: {pending}')
        progressed = False
        for run in pending[:]:
            path = run / 'last.pt'
            if not path.exists():
                continue
            # Loading every model at every poll is wasteful. The history is
            # appended before last.pt; always check the checkpoint epoch too.
            history = run / 'history.jsonl'
            if not history.exists():
                continue
            lines = history.read_text().splitlines()
            try:
                ready = bool(lines) and json.loads(lines[-1])['epoch'] >= args.epoch
            except json.JSONDecodeError:
                continue  # Writer may still be appending the final line.
            if not ready:
                continue
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint['epoch'] < args.epoch:
                continue
            if checkpoint['epoch'] != args.epoch:
                raise ValueError(f'{run}: requested fixed epoch no longer available')
            checkpoint_digest = file_digest(path)
            if checkpoint['architecture'] != 'structural_quality_ranker':
                raise ValueError('wrong checkpoint architecture')
            config = checkpoint['metadata']['arguments']
            if config['candidate_scope'] != 'joint':
                raise ValueError('comparison requires joint training protocol')

            def dataset(split):
                if split == 'train':
                    # Deliberate first shard only: diagnostic, not all train160.
                    archives, labels, initial = (config[key][0] for key in
                        ('train_archives', 'train_labels', 'train_initial_labels'))
                else:
                    archives, labels, initial = (config[key] for key in
                        ('val_archives', 'val_labels', 'val_initial_labels'))
                return StructuralJointDataset(
                    StructuralArchiveDataset(Path(archives), Path(labels), split),
                    StructuralArchiveDataset(Path(archives), Path(initial), split, candidate_scope='initial-pool'))

            train, val = dataset('train'), dataset('val')
            audit_datasets(train, val)
            for split, data in [('train', train), ('val', val)]:
                original = checkpoint['metadata'][split + '_case_seals']
                if any(original.get(case) != seal for case, seal in data.seals.items()):
                    raise ValueError('evaluation archive/label seals differ from training')
            model = StructuralQualityRanker(**checkpoint['config']).to(args.device).eval()
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            future = checkpoint['head_supervision']['future']
            print(f'evaluating {run} epoch={args.epoch} future={future}', flush=True)
            report = {'run': str(run.resolve()), 'epoch': args.epoch,
                      'checkpoint_sha256': checkpoint_digest,
                      'scope': 'fixed epoch; first train shard diagnostic and reused internal validation',
                      'head_supervision': checkpoint['head_supervision'],
                      'config': checkpoint['config'], 'arguments': config,
                      'train_cases': train.cases, 'val_cases': val.cases,
                      'train': evaluate(model, train, args.device, future_enabled=future),
                      'val': evaluate(model, val, args.device, future_enabled=future)}
            name = hashlib.sha256(str(run.resolve()).encode()).hexdigest()[:12] + '.json'
            temporary = args.out / (name + '.tmp')
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            temporary.replace(args.out / name)
            complete.append({'run': str(run.resolve()), 'report': name})
            (args.out / 'progress.json').write_text(json.dumps({'complete': complete,
                'pending': [str(p) for p in pending if p != run]}, indent=2) + '\n')
            print(f'completed {run}: {args.out / name}', flush=True)
            pending.remove(run)
            del model, checkpoint, report
            if str(args.device).startswith('cuda'):
                torch.cuda.empty_cache()
            progressed = True
        if pending and not progressed:
            time.sleep(30)
    print('All fixed-epoch evaluations complete', flush=True)


if __name__ == '__main__':
    main()

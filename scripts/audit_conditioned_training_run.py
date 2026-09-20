"""Read-only completion audit for a sealed, non-resumed conditioned run."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_metrics(rows, *, epochs, train_clusters, val_clusters, complete):
    """Reject missing steps, nonfinite losses and incomplete validation cells."""
    steps = [row for row in rows if row['event'] == 'train_step']
    validations = [row for row in rows if row['event'] == 'validation']
    require([r['global_step'] for r in steps] == list(range(1, len(steps) + 1)),
            'training steps must be contiguous from one')
    require(len(steps) <= epochs * train_clusters, 'too many training steps')
    for row in steps:
        require(row['epoch'] == (row['global_step'] - 1) // train_clusters + 1,
                'epoch/step mismatch')
        require(row['source'] in ('sdpi', 'bitseed'), 'unknown actual source')
        require(all(math.isfinite(value) for key, value in row.items()
                    if key.startswith('loss_')), 'nonfinite training loss')
    require([r['epoch'] for r in validations] == list(range(1, len(validations) + 1)),
            'validation epochs must be contiguous from one')
    cells = {'sdpi/full', 'bitseed_endpoint/full'}
    for row in validations:
        require(row['global_step'] == row['epoch'] * train_clusters <= len(steps),
                'validation epoch/step mismatch')
        require(set(row['by_cell']) == cells and set(row['counts']) == cells,
                'expected exactly two full-condition validation cells')
        require(all(count == val_clusters for count in row['counts'].values()),
                'incomplete validation coverage')
        require(all(math.isfinite(value) for value in row['by_cell'].values()),
                'nonfinite validation loss')
        require(math.isclose(row['selection_loss'], sum(row['by_cell'].values()) / 2,
                             rel_tol=1e-9, abs_tol=1e-9), 'selection loss mismatch')
        require(0 <= row['bitseed_fallback_count'] <= val_clusters,
                'invalid validation fallback count')
    if complete:
        require(len(steps) == epochs * train_clusters, 'training is not complete')
        require(len(validations) == epochs, 'validation is not complete')
    return {'training_steps': len(steps), 'validation_epochs': len(validations),
            'actual_sources': dict(Counter(row['source'] for row in steps)),
            'latest_validation': validations[-1] if validations else None}


def audit(root, *, complete=True):
    load = lambda path: json.loads(path.read_text())
    preparation = load(root / 'preparation.json')
    run = root / 'run'
    config = load(run / 'config.json')
    require(config['resume'] is None, 'this audit expects a non-resumed run')
    for relative, expected in preparation['runtime_sha256'].items():
        require(sha256(root / 'runtime' / relative) == expected,
                f'runtime source changed: {relative}')
    indexes = {name: load(root / f'{name}_index.json') for name in ('train', 'val')}
    for name, index in indexes.items():
        require(Path(config[f'{name}_data']).resolve() ==
                (root / f'{name}_index.json').resolve(), 'configured index path mismatch')
        for field in ('entries', 'clusters'):
            require(len(index[field]) == preparation[f'{name}_{field}'],
                    f'{name} {field} count changed')
    pdbs = {name: {r['case'].split('_')[0].upper() for r in index['entries']}
            for name, index in indexes.items()}
    excluded = {p.upper() for p in indexes['train']['metadata']['excluded_benchmark_pdb_ids']}
    require(not pdbs['train'] & pdbs['val'], 'train/validation PDB overlap')
    require(not (pdbs['train'] | pdbs['val']) & excluded, 'benchmark PDB overlap')
    raw_lines = (run / 'metrics.jsonl').read_text().splitlines(keepends=True)
    # A live writer may not yet have finished its last JSON line.
    if not complete and raw_lines and not raw_lines[-1].endswith('\n'):
        raw_lines.pop()
    rows = [json.loads(line) for line in raw_lines]
    report = audit_metrics(rows, epochs=config['epochs'],
                           train_clusters=preparation['train_clusters'],
                           val_clusters=preparation['val_clusters'], complete=complete)
    report['completion_required'] = complete
    report['runtime_files_verified'] = len(preparation['runtime_sha256'])
    report['scope'] = preparation['validation_scope']
    if complete:
        import torch

        parent = Path(config['parent'])
        require(sha256(parent) == config['parent_checkpoint_sha256'], 'parent changed')
        parent_payload = torch.load(parent, map_location='cpu', weights_only=False)
        validations = [r for r in rows if r['event'] == 'validation']
        best = min(validations, key=lambda row: row['selection_loss'])
        for filename, epoch in (('final_model.pt', config['epochs']),
                                ('best_model.pt', best['epoch'])):
            path = run / filename
            payload = torch.load(path, map_location='cpu', weights_only=False)
            require(payload['training_state']['epoch'] == epoch, f'{filename}: wrong epoch')
            require(payload['global_step'] == epoch * preparation['train_clusters'],
                    f'{filename}: wrong step')
            require(payload['parent_checkpoint_sha256'] == config['parent_checkpoint_sha256'],
                    f'{filename}: wrong parent')
            require(payload['expert_base_identity'] == parent_payload['expert_base_identity'],
                    f'{filename}: expert identity changed')
            require(payload['runtime_config']['validation_selection'] ==
                    'full_conditions_two_sources_v1', f'{filename}: wrong selection protocol')
            require(math.isclose(payload['training_state']['best_val_loss'],
                                 best['selection_loss']), f'{filename}: wrong best loss')
            for field in ('trifuse_state_dict', 'ema_trifuse_state_dict'):
                require(bool(payload[field]), f'{filename}: empty {field}')
                require(all(torch.isfinite(tensor).all() for tensor in payload[field].values()),
                        f'{filename}: nonfinite {field}')
            report[filename] = {'sha256': sha256(path), 'epoch': epoch,
                                'architecture': payload['architecture']}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--in-progress', action='store_true',
                        help='Check existing logs without asserting training completion')
    args = parser.parse_args()
    print(json.dumps(audit(args.run_root, complete=not args.in_progress), indent=2))

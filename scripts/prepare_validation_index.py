"""Copy training unchanged and exclude contaminated validation clusters.

Extracted from the retired model-v2 overnight workflow. This is an explicit
preparation step, not an automatic change to existing experiment splits.
"""

import argparse
import hashlib
import json
from pathlib import Path


def prepare_training_index(config, output):
    """Write fresh run-local indexes; never alter membership of training data."""
    output = Path(output)
    train_path, val_path = output / 'train_index.json', output / 'val_index.json'
    if train_path.exists() or val_path.exists():
        raise FileExistsError('preserve existing indexes; choose a new output directory')
    source_path = Path(config['val_index'])
    source = json.loads(source_path.read_text())
    inputs = {key: json.loads(Path(config[key]).read_text())
              for key in ('train_index', 'protac_index', 'mgd_index')}
    excluded = {entry['case'].split('_')[0].upper()
                for payload in inputs.values() for entry in payload['entries']}
    if source.get('clusters'):
        clusters = [cluster for cluster in source['clusters'] if not any(
            case.split('_')[0].upper() in excluded for case in cluster['items'])]
        retained = {case for cluster in clusters for case in cluster['items']}
        prepared = {**source, 'clusters': clusters,
                    'entries': [entry for entry in source['entries'] if entry['case'] in retained]}
    else:
        prepared = {**source, 'entries': [entry for entry in source['entries']
                    if entry['case'].split('_')[0].upper() not in excluded]}
    if not prepared['entries']:
        raise ValueError('no validation samples remain after split decontamination')
    report = {
        'source': str(source_path),
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'policy': 'retain full training; drop validation clusters sharing training/benchmark PDB',
        'original_entries': len(source['entries']), 'retained_entries': len(prepared['entries']),
        'removed_cases': sorted({entry['case'] for entry in source['entries']} -
                                {entry['case'] for entry in prepared['entries']}),
    }
    prepared['split_provenance'] = report
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also guards against another preparer overwriting these files.
    for path, payload in ((train_path, inputs['train_index']), (val_path, prepared)):
        with path.open('x') as handle:
            json.dump(payload, handle, indent=2)
            handle.write('\n')
    return {**config, 'train_index': str(train_path), 'val_index': str(val_path)}, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('train', 'val', 'protac', 'mgd'):
        parser.add_argument(f'--{key}-index', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepared, report = prepare_training_index(vars(args), args.out)
    print(json.dumps({'indexes': {key: prepared[key] for key in ('train_index', 'val_index')},
                      'validation_filter': report}, indent=2))


if __name__ == '__main__':
    main()

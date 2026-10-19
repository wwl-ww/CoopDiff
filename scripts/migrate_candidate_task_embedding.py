#!/usr/bin/env python
"""Make legacy source-only candidate checkpoints explicitly task-compatible."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tricoopdiff.models.assembly import migrate_legacy_candidate_task_embedding
from tricoopdiff.runtime import atomic_json, atomic_torch_save, sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    source = args.input.resolve()
    destination = args.output.resolve()
    if not source.is_file():
        parser.error(f'checkpoint does not exist: {source}')
    if source == destination or destination.exists():
        parser.error('output must be a new path, distinct from the source checkpoint')

    payload = torch.load(source, map_location='cpu', weights_only=False)
    source_digest = sha256_file(source)
    migrated = migrate_legacy_candidate_task_embedding(payload)
    migrated['checkpoint_migration']['source_checkpoint_sha256'] = source_digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(migrated, destination)
    audit = {
        'source_checkpoint': str(source),
        'source_checkpoint_sha256': source_digest,
        'migrated_checkpoint': str(destination),
        'migrated_checkpoint_sha256': sha256_file(destination),
        'migration': migrated['checkpoint_migration'],
        'updates': migrated.get('updates'),
        'training_options': migrated.get('training_options'),
    }
    atomic_json(audit, destination.with_suffix(destination.suffix + '.migration.json'))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

"""Prove source indexes differ only in path spelling; retain immutable artifacts."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(index, root):
    value = copy.deepcopy(index)
    for entry in value['entries']:
        for key in ('protein1', 'protein2', 'ligand'):
            path = Path(entry[key])
            entry[key] = str((path if path.is_absolute() else root / path).resolve())
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('restored-index', 'source-index', 'canonical-index', 'project-root',
                 'prior-manifest', 'out-index', 'out-prior-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out_index.exists() or args.out_prior_manifest.exists():
        raise FileExistsError('index rebind must publish new files, never overwrite an artifact')
    source, canonical, restored, prior = [json.loads(path.read_text()) for path in
        (args.source_index, args.canonical_index, args.restored_index, args.prior_manifest)]
    if normalized(source, args.project_root) != normalized(canonical, args.project_root):
        raise ValueError('source indexes have substantive differences beyond path spelling')
    if source['metadata']['split'] != 'train':
        raise ValueError('only training source identity can be rebound here')
    if restored['metadata']['source_index_sha256'] != digest(args.source_index):
        raise ValueError('restored index does not identify the supplied original source')
    if prior['index_sha256'] != digest(args.restored_index):
        raise ValueError('prior manifest does not identify the supplied restored index')
    proof = {'policy': 'exact_index_equality_after_absolute_path_normalization_v1',
             'source_index': str(args.source_index.resolve()), 'source_sha256': digest(args.source_index),
             'canonical_index': str(args.canonical_index.resolve()), 'canonical_sha256': digest(args.canonical_index),
             'project_root': str(args.project_root.resolve()), 'entries_changed': False,
             'clusters_changed': False, 'coordinate_artifacts_changed': False}
    restored['metadata'].update(source_index_sha256=digest(args.canonical_index), source_identity_equivalence=proof)
    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    with args.out_index.open('x') as handle:
        json.dump(restored, handle, indent=2)
    prior.update(index_sha256=digest(args.out_index), previous_manifest_sha256=digest(args.prior_manifest),
                 source_identity_equivalence=proof)
    args.out_prior_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_prior_manifest.open('x') as handle:
        json.dump(prior, handle, indent=2)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()

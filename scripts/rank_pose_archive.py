"""Select saved poses without loading DockQ labels or native structure files."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch

from evaluation_common import sha256_file
from tricoopdiff.ranking_inference import INFERENCE_FIELDS, load_quality_checkpoint, predict_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--ranker', type=Path, help='Exported inference.pt only; omit for fixed R0 C-alpha rule')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.archives / 'manifest.json').read_text())
    checkpoint = torch.load(args.ranker, map_location='cpu', weights_only=False) if args.ranker else None
    if checkpoint is not None and set(checkpoint) != INFERENCE_FIELDS:
        raise ValueError('export inference-only weights before ranking; training metadata not accepted')
    if not manifest['cases'] or len(set(manifest['cases'])) != len(manifest['cases']):
        raise ValueError('empty or duplicate archive cases')
    model = load_quality_checkpoint(checkpoint, manifest) if checkpoint else None
    args.out.mkdir(parents=True, exist_ok=False)
    configuration = {
        'protocol': 'native_free_saved_state_ranking_v1',
        'method': 'quality_mlp' if model else 'ca_balanced_contact_minus_clash_v1',
        'archive_manifest_sha256': sha256_file(args.archives / 'manifest.json'),
        'checkpoint_sha256': sha256_file(args.ranker) if args.ranker else None,
        'code_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                       (Path(__file__), ROOT / 'tricoopdiff/ranking_inference.py', ROOT / 'tricoopdiff/pose_ranking.py')},
        'cases': manifest['cases'], 'native_or_labels_read': False,
        'output_set': 'archived paired states only; unsampled pool initial states excluded',
        'tie_break': 'pool_index ascending, then initial before final',
    }
    (args.out / 'manifest.json').write_text(json.dumps(configuration, indent=2) + '\n')
    for case in manifest['cases']:
        if case != Path(case).name or case in ('.', '..'):
            raise ValueError('unsafe case identifier')
        path = args.archives / f'{case}.pt'
        receipt = json.loads((args.archives / f'{case}.complete.json').read_text())
        digest = sha256_file(path)
        if digest != receipt['archive_sha256']:
            raise ValueError('archive receipt mismatch')
        archive = torch.load(path, map_location='cpu', weights_only=False)
        if archive['case'] != case or archive['manifest'] != manifest:
            raise ValueError('case/archive manifest mismatch')
        result = {'case': case, 'archive_sha256': digest, **predict_archive(archive, model, checkpoint)}
        (args.out / f'{case}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(f'RANKED {case}: {len(result["rows"])} states', flush=True)
    (args.out / 'complete.json').write_text(json.dumps({'cases': manifest['cases']}) + '\n')


if __name__ == '__main__':
    main()

"""Score a saved prediction after inference; native structures never enter sampling."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tricoopdiff.diffdock_compat  # noqa: F401
from tricoopdiff.deepternary_metrics import compute_fullatom_dockq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prediction', 'case-dir', 'deepternary-root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true',
                        help='Score implementation-smoke predictions explicitly; not a formal accuracy claim')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('choose a new score output')
    raw = args.prediction.read_bytes()
    prediction = json.loads(raw)
    if prediction['case'] != args.case_dir.name:
        raise ValueError('prediction and reference case do not match')
    if prediction['checkpoint_scope'] != 'training' and not args.allow_smoke:
        raise ValueError('implementation smoke is not a trained prediction')
    if prediction['coordinate_frame'] != 'featurized E3 frame; E3 coordinates included':
        raise ValueError('unsupported prediction coordinate frame')
    candidates = prediction.get('predictions', [prediction])
    if (not isinstance(candidates, list) or not candidates
            or len(candidates) != prediction.get('n_walkers', 1)):
        raise ValueError('declared walker count must match complete saved predictions')
    coordinates = []
    for candidate in candidates:
        coords = {key: np.asarray(candidate[key], dtype=np.float64) for key in ('ligand_coords', 'poi_coords')}
        coords['e3_coords'] = np.asarray(prediction['e3_coords'], dtype=np.float64)
        if any(value.ndim != 2 or value.shape[1] != 3 or not len(value)
               or not np.isfinite(value).all() for value in coords.values()):
            raise ValueError('prediction must contain finite nonempty coordinate arrays')
        coordinates.append(coords)
    paths = {name: args.case_dir / name for name in (
        'unbound_protein1.pdb', 'unbound_protein2.pdb', 'ligand.sdf',
        'protein1.pdb', 'protein2.pdb', 'gt_complex.pdb')}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in paths.items()}
    metrics = [compute_fullatom_dockq(
        coords['ligand_coords'], coords['poi_coords'], coords['e3_coords'],
        str(paths['unbound_protein1.pdb']), str(paths['unbound_protein2.pdb']),
        str(paths['ligand.sdf']), str(paths['gt_complex.pdb']),
        str(args.deepternary_root), metric_protein1_pdb=str(paths['protein1.pdb']),
        metric_protein2_pdb=str(paths['protein2.pdb'])) for coords in coordinates]
    summary = ({'metric': metrics[0]} if len(metrics) == 1 else {
        'candidate_metrics': metrics, 'oracle_dockq': max(row['dockq'] for row in metrics),
        'mean_dockq': float(np.mean([row['dockq'] for row in metrics])), 'model_top1_computed': False})
    report = {'case': prediction['case'],
              'prediction_stage': prediction.get('prediction_stage', 'final'),
              'scope': ('single saved prediction; not ensemble oracle' if len(metrics) == 1
                        else 'saved candidate-pool oracle; no learned ranking'),
              'n_walkers': len(metrics), 'ensemble_policy': prediction.get('ensemble_policy'),
              'prediction_sha256': hashlib.sha256(raw).hexdigest(),
              'checkpoint_sha256': prediction['checkpoint_sha256'],
              'reference_sha256': hashes, **summary,
              'native_used_only_after_inference': True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

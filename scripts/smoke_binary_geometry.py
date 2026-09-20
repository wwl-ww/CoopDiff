"""Check real-input conditioned SDPI/BiTSeed trajectories without native scoring."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from evaluation_runner import EvaluationRunner
from featurize_common import featurize_complex
from tricoopdiff.condition_manifest import load_condition_manifest
from tricoopdiff.expert_anchors import DualExpertAnchorInitializer
from tricoopdiff.sampler import sample_single


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--condition-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--anchor-attempts', type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('choose a new smoke output')
    case = args.case_dir.name
    runner = EvaluationRunner(args.project_root, args.checkpoint, device=args.device,
                              allow_conditioned_checkpoint=True)
    e3, poi, ligand, dim = featurize_complex(
        case, str(args.case_dir / 'unbound_protein1.pdb'),
        str(args.case_dir / 'unbound_protein2.pdb'), str(args.case_dir / 'ligand.sdf'),
        runner.score_model_args, args.device,
        cache_dir=str(args.project_root / 'results/featurize_cache'))
    runner._ensure_model(dim)
    from rdkit import Chem
    features, audit = load_condition_manifest(
        args.condition_manifest, case=case, molecule=Chem.Mol(ligand['mol']),
        protein_sizes=[e3['coords'].shape[1], poi['coords'].shape[1]])
    reports = []
    for mode in ('sdpi', 'expert_anchor'):
        initializer = (runner.initializer if mode == 'sdpi' else
                       DualExpertAnchorInitializer(
                           runner.expert, e3, poi, binary_steps=20, proposal_attempts=args.anchor_attempts,
                           conformer_policy=runner.initializer.conformer_policy))
        result = sample_single(
            runner.model, runner.diffusion, initializer, e3, poi, ligand,
            device=args.device, steps=20, seed=args.seed, init_mode=mode,
            conditions=features, condition_guidance='full', return_trajectory=True)
        for key in ('ligand_coords', 'poi_coords'):
            if not np.isfinite(result[key]).all():
                raise AssertionError(f'{mode}: nonfinite {key}')
        if not result['valid'] or len(result['trajectory']) != 20:
            raise AssertionError('incomplete trajectory')
        reports.append({'mode': mode, 'steps': 20, 'finite': True,
                        'initialization': result.get('initialization'),
                        'ligand_atoms': len(result['ligand_coords']),
                        'poi_nodes': len(result['poi_coords'])})
        print(json.dumps({'mode': mode, 'finite': True,
                          'actual_source': result['initialization']['candidate_source'],
                          'fallback': result['initialization'].get('fallback')}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump({'scope': 'implementation smoke; no accuracy claim', 'case': case,
                   'architecture': runner.checkpoint['architecture'],
                   'checkpoint_sha256': runner.checkpoint_sha256,
                   'condition_audit': audit, 'runs': reports}, handle, indent=2)


if __name__ == '__main__':
    main()

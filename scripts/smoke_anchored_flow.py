"""Real frozen-expert integration smoke; synthetic target, no accuracy claim."""

import argparse
import json
from pathlib import Path
import sys

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import tricoopdiff.diffdock_compat  # noqa: F401
from evaluation_runner import EvaluationRunner
from featurize_common import featurize_complex
from tricoopdiff.anchored_consensus import anchored_target_loss, sample_anchored
from tricoopdiff.anchored_flow import AnchoredTorsionFlow
from tricoopdiff.binary_pose_constraints import BinaryPoseInitializer
from tricoopdiff.condition_manifest import load_binary_pose_sides, load_condition_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'checkpoint', 'case-dir', 'condition-manifest', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--max-fit-rmsd', type=float, default=.15)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('choose a new smoke output')
    torch.manual_seed(args.seed)
    runner = EvaluationRunner(args.project_root, args.checkpoint.resolve(), device=args.device,
                              allow_conditioned_checkpoint=True)
    case = args.case_dir.name
    e3, poi, ligand, dim = featurize_complex(
        case, str(args.case_dir / 'unbound_protein1.pdb'),
        str(args.case_dir / 'unbound_protein2.pdb'), str(args.case_dir / 'ligand.sdf'),
        runner.score_model_args, args.device,
        cache_dir=str(args.project_root / 'results/featurize_cache'))
    runner._ensure_model(dim)
    options = dict(case=case, molecule=Chem.Mol(ligand['mol']),
                   protein_sizes=[e3['coords'].shape[1], poi['coords'].shape[1]])
    conditions, condition_audit = load_condition_manifest(args.condition_manifest, **options)
    sides, _ = load_binary_pose_sides(args.condition_manifest, **options)
    flow, audit = AnchoredTorsionFlow.from_binary_initializer(
        BinaryPoseInitializer(*sides, max_candidates=2),
        e3['coords'][0], poi['coords'][0], ligand,
        seed=args.seed, max_fit_rmsd=args.max_fit_rmsd)
    print(json.dumps({'stage': 'chart', 'audit': audit}), flush=True)
    generator = torch.Generator(device=flow.ligand.device).manual_seed(args.seed)
    prior = flow.sample_prior(generator=generator)
    runner.model.train()
    runner.model.zero_grad(set_to_none=True)
    losses = anchored_target_loss(
        runner.model, flow, flow.ligand, flow.poi, prior, e3, poi, ligand, .5, .1,
        conditions=conditions, source=0)
    losses['loss_total'].backward()
    gradient = runner.model.trifuse.torsion_coordinator.net[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(p.grad is None for p in runner.model.expert.parameters()), 'expert must remain frozen'
    report = {key: float(value.detach()) for key, value in losses.items()
              if key.startswith('loss_')}
    report['torsion_head_gradient_norm'] = float(gradient.norm())
    report['target_audit'] = losses['target_audit']
    print(json.dumps({'stage': 'backward', **report}), flush=True)
    runner.model.zero_grad(set_to_none=True)
    del losses
    result = sample_anchored(runner.model, flow, prior, e3, poi, ligand,
                             steps=3, conditions=conditions, source=0)
    left, right = audit['selected_atom_maps']
    anchor_error = (result['ligand_coords'][left] - flow.ligand[left]).abs().max()
    binary_error = (torch.cdist(result['ligand_coords'][right], result['poi_coords'])
                    - torch.cdist(flow.ligand[right], flow.poi)).abs().max()
    assert anchor_error < 1e-4 and binary_error < 1e-3, 'binary geometry drift'
    assert all(torch.isfinite(value).all() for value in result.values() if torch.is_tensor(value))
    report.update(anchor_max_error=float(anchor_error), binary_distance_max_error=float(binary_error))
    payload = {'scope': 'implementation smoke; synthetic zero-angle target; no accuracy claim',
               'case': case, 'seed': args.seed, 'generative_process': flow.version,
               'checkpoint_sha256': runner.checkpoint_sha256,
               'compatibility_source': 'sdpi embedding (0); not a trained anchored checkpoint',
               'condition_audit': condition_audit, 'initialization': audit, 'checks': report}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({'stage': 'complete', **report}), flush=True)


if __name__ == '__main__':
    main()

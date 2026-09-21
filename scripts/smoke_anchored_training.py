"""One optimizer update on one real training target; no evaluation/accuracy claim."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import tricoopdiff.diffdock_compat  # noqa: F401
from evaluation_runner import EvaluationRunner
from featurize_common import featurize_complex
from train_entry import _load_index
from tricoopdiff.anchored_consensus import anchored_target_loss
from tricoopdiff.anchored_training import prepare_augmented_anchored_sample
from tricoopdiff.anchored_action_adapter import install_action_adapter, ADAPTER_POLICY
from tricoopdiff.coupled_linker_action import COUPLED_ADAPTER_POLICY
from tricoopdiff.anchored_trainer import warm_start_anchored_weights
from tricoopdiff.anchored_training_metric import PARAMETER_METRIC, PHYSICAL_METRIC
from tricoopdiff.chemical_prior_contract import ChemicalPriorLibrary, chemical_prior_config, apply_training_prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'checkpoint', 'index', 'cache-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--binary-translation-bound', type=float)
    parser.add_argument('--binary-rotation-bound', type=float)
    parser.add_argument('--single-anchor', action='store_true')
    parser.add_argument('--dual-core', action='store_true')
    parser.add_argument('--body-aware-action', action='store_true')
    parser.add_argument('--coupled-linker-action', action='store_true')
    parser.add_argument('--binary-metric', choices=(PARAMETER_METRIC, PHYSICAL_METRIC), default=PARAMETER_METRIC)
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--reset-action-adapter', action='store_true')
    parser.add_argument('--chemical-prior-manifest', type=Path)
    args = parser.parse_args()
    if args.single_anchor and args.dual_core:
        parser.error('select a single conditioning task')
    if args.body_aware_action and args.coupled_linker_action:
        parser.error('select one action adapter')
    adapter_policy = (COUPLED_ADAPTER_POLICY if args.coupled_linker_action else
                      ADAPTER_POLICY if args.body_aware_action else None)
    if (args.binary_translation_bound is None) != (args.binary_rotation_bound is None):
        parser.error('provide both binary bounds')
    bounds = (None if args.binary_translation_bound is None else
              {'translation': [args.binary_translation_bound] * 2,
               'rotation': [args.binary_rotation_bound] * 2})
    if args.out.exists():
        raise FileExistsError('choose a new training smoke output')
    payload = json.loads(args.index.read_text())
    excluded = payload.get('metadata', {}).get('excluded_benchmark_pdb_ids')
    if not excluded or args.case.split('_')[0].upper() in {x.upper() for x in excluded}:
        raise ValueError('require explicit benchmark exclusions; never train on benchmark cases')
    _, _, by_case = _load_index(args.index)
    entry = by_case[args.case]
    torch.manual_seed(args.seed)
    runner = EvaluationRunner(args.project_root, args.checkpoint.resolve(), device=args.device,
                              allow_conditioned_checkpoint=True)
    e3, poi, ligand, dim = featurize_complex(
        args.case, entry['protein1'], entry['protein2'], entry['ligand'],
        runner.score_model_args, args.device, cache_dir=str(args.cache_dir),
        cache_read_only=True, cache_wait_seconds=0)
    runner._ensure_model(dim)
    if adapter_policy:
        install_action_adapter(runner.model, policy=adapter_policy)
    if args.warm_start:
        warm_start_anchored_weights(runner.model, torch.load(args.warm_start, map_location='cpu', weights_only=False),
                                   reset_action_adapter=args.reset_action_adapter)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    if args.dual_core:
        from tricoopdiff.core_training import prepare_augmented_core_sample
        sample = prepare_augmented_core_sample(e3, poi, ligand, generator=generator,
            evidence_id=f'training-smoke:{args.case}', binary_bounds=bounds)
    elif args.single_anchor:
        from tricoopdiff.single_anchor_training import prepare_single_anchor_sample
        sample = prepare_single_anchor_sample(
            e3, poi, ligand, generator=generator,
            evidence_id=f'training-smoke:{args.case}',
            translation_bound=bounds['translation'][0] if bounds else 0.,
            rotation_bound=bounds['rotation'][0] if bounds else 0.)
    else:
        sample = prepare_augmented_anchored_sample(
            e3, poi, ligand, generator=generator,
            evidence_id=f'training-smoke:{args.case}', binary_bounds=bounds)
    if sample is None:
        raise ValueError('this case has no applicable chemical-cut augmentation')
    if args.chemical_prior_manifest:
        config = chemical_prior_config(seed=args.seed)
        library = ChemicalPriorLibrary(args.chemical_prior_manifest, config,
                                       index_sha256=hashlib.sha256(args.index.read_bytes()).hexdigest())
        sample = apply_training_prior(sample, ligand, library.get(args.case, ligand['mol']),
                                       generator=generator, config=config)
    model = runner.model
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    head = model.trifuse.torsion_coordinator.net[-1].weight
    before = head.detach().clone()
    rigid_head = model.trifuse.residual_head.weight
    rigid_before = rigid_head.detach().clone()
    adapter_head = (model.trifuse.anchored_action_adapter.net[-1].weight
                    if adapter_policy == ADAPTER_POLICY else None)
    adapter_before = adapter_head.detach().clone() if adapter_head is not None else None
    coupled_head = model.trifuse.anchored_action_adapter.coupled_output.weight if args.coupled_linker_action else None
    coupled_before = coupled_head.detach().clone() if coupled_head is not None else None
    losses = anchored_target_loss(
        model, sample.flow, ligand['coords'][0], poi['coords'][0], sample.prior_angles,
        e3, poi, ligand, .5, .1, conditions=sample.conditions, source=0, binary_metric=args.binary_metric)
    losses['loss_total'].backward()
    assert all(p.grad is None and not p.requires_grad for p in model.expert.parameters())
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.,
                                         error_if_nonfinite=True)
    if len(sample.flow.plan.bonds):
        assert head.grad is not None and head.grad.abs().sum() > 0
    assert rigid_head.grad is not None and rigid_head.grad.abs().sum() > 0
    optimizer.step()
    assert torch.isfinite(head).all()
    if len(sample.flow.plan.bonds):
        assert not torch.equal(head, before)
    assert torch.isfinite(rigid_head).all() and not torch.equal(rigid_head, rigid_before)
    if adapter_head is not None and len(sample.flow.plan.bonds):
        assert torch.isfinite(adapter_head).all() and not torch.equal(adapter_head, adapter_before)
    if coupled_head is not None and len(sample.flow.plan.bonds):
        assert torch.isfinite(coupled_head).all() and not torch.equal(coupled_head, coupled_before)
    report = {'scope': 'one real training-target optimizer update; no accuracy claim; no checkpoint saved',
              'case': args.case, 'seed': args.seed, 'audit': sample.audit,
              'generative_process': sample.flow.version,
              'action_policy': losses['model_output']['anchored_joint_motion']['policy'],
              'joint_motion': losses['model_output']['anchored_joint_motion'],
              'compatibility_source': 'old SDPI embedding 0; experimental adapter only',
              'parent_checkpoint_sha256': runner.checkpoint_sha256,
              'warm_start_sha256': hashlib.sha256(args.warm_start.read_bytes()).hexdigest() if args.warm_start else None,
              'binary_metric': args.binary_metric, 'action_adapter': adapter_policy,
              'index_sha256': hashlib.sha256(args.index.read_bytes()).hexdigest(),
              'losses': {key: float(value.detach()) for key, value in losses.items()
                         if key.startswith('loss_')},
              'gradient_norm_before_clipping': float(norm),
              'torsion_head_update_norm': float((head.detach() - before).norm()),
              'rigid_head_update_norm': float((rigid_head.detach() - rigid_before).norm()),
              'body_aware_head_update_norm': float((adapter_head.detach() - adapter_before).norm())
                                             if adapter_head is not None else None,
              'coupled_head_update_norm': float((coupled_head.detach() - coupled_before).norm())
                                          if coupled_head is not None else None,
              'expert_frozen': True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

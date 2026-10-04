"""Synthetic condition-availability/noise stress on real EMA weights; CPU only."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from smoke_conditioned_fuse import synthetic_case, velocity_vector
import torch
from rdkit import Chem

from evaluation_common import sha256_file
from tricoopdiff.conditioning.condition_augmentation import ConditionAugmentation, augment_conditions
from tricoopdiff.conditioning.condition_features import SideCondition, build_condition_features
from tricoopdiff.conditioning.conditioned_consensus_train import ConditionedConsensusFuse
from tricoopdiff.models.consensus import ConsensusFuse
from tricoopdiff.train import validate_consensus_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.out.exists():
        raise FileExistsError('use a new condition stress output')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    validate_consensus_checkpoint(checkpoint)
    base = ConsensusFuse(**checkpoint['model_config']).eval()
    base.load_state_dict(checkpoint['ema_trifuse_state_dict'], strict=True)
    model = ConditionedConsensusFuse.from_base(base).eval()
    graph, proposals, state, molecule = synthetic_case(base.config['input_dim'])
    left = SideCondition('template', 'synthetic:left', fragment=Chem.MolFromSmiles('CC'), atom_map=(0, 1),
                         pocket_probabilities=[.8, .3, .1, 0., 0.],
                         binary_protein_coords=graph['node_coords'][:5].numpy(),
                         binary_fragment_coords=graph['node_coords'][11:13].numpy())
    right = SideCondition('template', 'synthetic:right', fragment=Chem.MolFromSmiles('ON'), atom_map=(2, 3),
                          pocket_probabilities=[0., 0., .2, .8, .9, .1],
                          binary_protein_coords=graph['node_coords'][5:11].numpy(),
                          binary_fragment_coords=graph['node_coords'][13:].numpy())
    without_binary = lambda s: replace(s, binary_fragment_coords=None, binary_protein_coords=None)
    roles_only = lambda s: replace(without_binary(s), pocket_probabilities=None, provenance='chemical_definition')
    modes = {'none': ((None, None), None), 'roles_only': ((roles_only(left), roles_only(right)), None),
             'pockets': ((without_binary(left), without_binary(right)), None), 'binary': ((left, right), None),
             'left_only': ((left, None), None), 'right_only': ((None, right), None),
             'wrong_pocket': ((replace(left, pocket_probabilities=[0., 0., 0., 1., 1.]), right), None)}
    for name, config in {
        'drop_all': ConditionAugmentation(drop_all_probability=1.),
        'drop_pockets': ConditionAugmentation(drop_pocket_probability=1.),
        'drop_binary': ConditionAugmentation(drop_binary_probability=1.),
        'noisy_pockets': ConditionAugmentation(pocket_noise_std=.2, perturbed_confidence_scale=.5),
        'noisy_binary': ConditionAugmentation(binary_translation_std=.5, binary_rotation_std=.2,
                                             perturbed_confidence_scale=.5),
    }.items():
        sides, _, audit = augment_conditions((left, right), seed=7315, config=config)
        modes[name] = (sides, audit)
    original = {key: value.clone() for key, value in graph.items()}
    rows = []
    for source in (0, 1):
        reference = velocity_vector(base(graph, proposals, state, source=source)).detach()
        for name, (sides, audit) in modes.items():
            features = build_condition_features(molecule, (5, 6), sides)
            model.zero_grad(set_to_none=True)
            velocity = velocity_vector(model(graph, proposals, state, source=source, conditions=features))
            if not torch.isfinite(velocity).all():
                raise AssertionError(f'{source}/{name}: non-finite conditional velocity')
            (velocity - .5).square().mean().backward()
            gradient = sum(float(p.grad.detach().norm()) for p in model.condition_encoder.parameters()
                           if p.grad is not None)
            scale_gradient = float(model.condition_scale.grad.detach().abs())
            available = bool(features['sides'][:, :3].any())
            if available and not scale_gradient > 0:
                raise AssertionError(f'{source}/{name}: missing zero-gate condition gradient')
            if gradient != 0:
                raise AssertionError(f'{source}/{name}: encoder opened before its zero gate')
            if not available and (not torch.equal(velocity.detach(), reference) or gradient != 0):
                raise AssertionError('absent conditions changed base output or trained missing features')
            if not torch.equal(velocity.detach(), reference):
                raise AssertionError('untrained conditions changed the migrated parent function')
            rows.append({'source': source, 'mode': name, 'finite': True,
                         'velocity_delta_l2': float((velocity.detach() - reference).norm()),
                         'condition_scale_gradient_norm': scale_gradient,
                         'condition_encoder_gradient_norm_sum': gradient, 'augmentation': audit})
    if any(not torch.equal(graph[key], value) for key, value in original.items()):
        raise AssertionError('condition stress mutated input state/chemistry')
    report = {'scope': 'synthetic zero-migration/availability stress; not evidence of learned robustness',
              'parent_checkpoint_sha256': sha256_file(args.checkpoint), 'optimizer_steps': 0,
              'device': 'cpu', 'modes': list(modes), 'checks': rows,
              'code_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in (
                  Path(__file__), ROOT / 'scripts/smoke_conditioned_fuse.py',
                  ROOT / 'tricoopdiff/conditioning/condition_augmentation.py', ROOT / 'tricoopdiff/conditioning/condition_features.py',
                  ROOT / 'tricoopdiff/conditioning/conditioned_consensus_train.py', ROOT / 'tricoopdiff/models/consensus.py',
                  ROOT / 'tricoopdiff/models/consensus.py')}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'checks': len(rows), 'all_finite': True, 'optimizer_steps': 0}))


if __name__ == '__main__':
    main()

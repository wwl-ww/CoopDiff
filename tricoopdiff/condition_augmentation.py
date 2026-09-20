"""Deterministic, label-free condition corruption for future training/stress tests.

Only conditions are changed: never the flow state, its velocity target or the
candidate's SDPI/BiTSeed source. Defaults are identity, not a chosen training policy.
"""

import copy
from dataclasses import asdict, dataclass, replace
import math

import numpy as np
import torch


@dataclass(frozen=True)
class ConditionAugmentation:
    drop_all_probability: float = 0.
    drop_side_probability: float = 0.
    drop_pocket_probability: float = 0.
    drop_binary_probability: float = 0.
    pocket_noise_std: float = 0.
    binary_translation_std: float = 0.  # Angstrom, protein-local frame.
    binary_rotation_std: float = 0.  # Radians per rotation-vector component.
    perturbed_confidence_scale: float = 1.

    def validate(self):
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid augmentation parameter: {name}')
            if (name.endswith('probability') or name.endswith('scale')) and value > 1:
                raise ValueError(f'augmentation probability/scale exceeds one: {name}')


@dataclass(frozen=True)
class ConditionFeatureDropout:
    """Training-only availability corruption after native contact masks exist."""

    drop_all_probability: float = 0.1
    drop_side_probability: float = 0.2
    drop_pocket_probability: float = 0.1
    drop_ligand_role_probability: float = 0.1
    drop_binary_probability: float = 0.1
    pocket_only_probability: float = 0.

    def validate(self):
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'invalid feature dropout parameter: {name}')


def dropout_condition_features(features, *, seed, config=ConditionFeatureDropout()):
    """Drop whole/one-sided native-derived masks without changing coordinates.

    Fixed RNG slots make replay independent of whether a particular side has
    any positive contact. Unknown is represented by availability masks, never
    by interpreting an all-zero pocket as known negative.
    """
    config.validate()
    if type(seed) is not int or seed < 0 or set(features) != {
            'ligand', 'protein1', 'protein2', 'sides', 'binary1', 'binary2'}:
        raise ValueError('feature dropout requires the versioned condition fields and a nonnegative seed')
    result = {name: torch.as_tensor(value).detach().cpu().clone() for name, value in features.items()}
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError('condition features must be finite')
    rng = np.random.default_rng(seed)
    audit = {'protocol': 'condition_feature_dropout_v2', 'seed': seed, 'config': asdict(config),
             'native_coordinates_returned': False, 'flow_state_modified': False, 'events': []}

    def drop_role(side):
        result['ligand'][:, side] = 0
        result['sides'][side, 0] = 0
        result['sides'][side, 4] = 0

    def drop_pocket(side):
        result[f'protein{side + 1}'][:, :3] = 0
        result['sides'][side, 1] = 0

    def drop_binary(side):
        result[f'binary{side + 1}'].zero_()
        start = 4 + 5 * side
        result['ligand'][:, start:start + 5] = 0
        result[f'protein{side + 1}'][:, 3:6] = 0
        result['sides'][side, 2] = 0

    if rng.random() < config.drop_all_probability:
        for side in (0, 1):
            drop_role(side); drop_pocket(side); drop_binary(side)
        result['sides'].zero_()
        audit['events'].append({'action': 'drop_all'})
    elif (config.pocket_only_probability > 0
          and rng.random() < config.pocket_only_probability):
        for side in (0, 1):
            drop_role(side); drop_binary(side)
        audit['events'].append({'action': 'keep_pocket_only_both_sides'})
    else:
        for side in (0, 1):
            side_rng = np.random.default_rng(rng.integers(0, 2**63))
            if side_rng.random() < config.drop_side_probability:
                drop_role(side); drop_pocket(side); drop_binary(side); result['sides'][side].zero_()
                audit['events'].append({'side': side, 'action': 'drop_side'})
                continue
            if side_rng.random() < config.drop_pocket_probability:
                drop_pocket(side)
                audit['events'].append({'side': side, 'action': 'drop_pocket_to_unknown'})
            if side_rng.random() < config.drop_ligand_role_probability:
                drop_role(side)
                audit['events'].append({'side': side, 'action': 'drop_ligand_role_to_unknown'})
            if side_rng.random() < config.drop_binary_probability:
                drop_binary(side)
                audit['events'].append({'side': side, 'action': 'drop_binary_to_unknown'})
    result['ligand'][:, 3] = (result['ligand'][:, :3].sum(1) == 0).to(result['ligand'])
    # A side with no remaining condition must have no confidence/provenance.
    for side in (0, 1):
        if not result['sides'][side, :3].any():
            result['sides'][side].zero_()
    return result, audit


def protein_local_basis(coords):
    """Deterministic right-handed axes from geometry; reject degenerate frames.

Ties follow the existing receptor order. This is rigid-frame equivariant, not a
claim of permutation invariance for exact symmetric/tied protein configurations.
"""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 3 or not np.isfinite(coords).all():
        raise ValueError('binary augmentation requires at least three finite protein coordinates')
    centered = coords - coords.mean(0)
    norms = np.linalg.norm(centered, axis=1)
    if norms.max() < 1e-8:
        raise ValueError('degenerate binary protein frame')
    first = centered[np.argmax(norms)] / norms.max()
    transverse = centered - (centered @ first)[:, None] * first
    lengths = np.linalg.norm(transverse, axis=1)
    if lengths.max() < 1e-8:
        raise ValueError('collinear binary protein frame')
    second = transverse[np.argmax(lengths)] / lengths.max()
    return np.stack((first, second, np.cross(first, second)))


def perturb_binary_pose(fragment, protein, rng, translation_std, rotation_std):
    """Rigidly perturb the fragment relative to its own protein; preserve bonds."""
    coords = np.asarray(fragment, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords) or not np.isfinite(coords).all():
        raise ValueError('invalid binary fragment coordinates')
    basis = protein_local_basis(protein)
    omega = rng.normal(size=3) * rotation_std
    theta = np.linalg.norm(omega)
    rotation = np.eye(3)
    if theta > 1e-12:
        x, y, z = omega / theta
        cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        rotation += math.sin(theta) * cross + (1 - math.cos(theta)) * (cross @ cross)
    translation = rng.normal(size=3) * translation_std
    center = coords.mean(0)
    moved = ((coords - center) @ basis.T @ rotation.T + translation) @ basis + center
    return moved, {'translation_local_angstrom': translation.tolist(),
                   'rotation_vector_local_radians': omega.tolist()}


def augment_conditions(sides, *, seed, config=ConditionAugmentation(), linker_indices=()):
    """Return new sides, explicit linker annotations and a replayable audit record."""
    config.validate()
    if len(sides) != 2 or type(seed) is not int or seed < 0:
        raise ValueError('two optional sides and a nonnegative integer seed required')
    rng = np.random.default_rng(seed)  # Never consume Python/NumPy/Torch global RNG.
    audit = {'protocol': 'condition_augmentation_v1', 'seed': seed, 'config': asdict(config),
             'flow_state_modified': False, 'candidate_source_modified': False, 'events': []}
    if rng.random() < config.drop_all_probability:
        audit['events'].append({'action': 'drop_all_including_linker_roles'})
        return (None, None), (), audit
    result = []
    for index, original in enumerate(sides):
        # Fixed per-side draw slots keep side 2's decisions independent of side 1's availability.
        side_rng = np.random.default_rng(rng.integers(0, 2**63))
        if original is None:
            result.append(None)
            continue
        if side_rng.random() < config.drop_side_probability:
            result.append(None)
            audit['events'].append({'side': index, 'action': 'drop_side'})
            continue
        side = copy.deepcopy(original)
        events, perturbed = [], False
        if side_rng.random() < config.drop_pocket_probability:
            if side.pocket_probabilities is not None:
                side = replace(side, pocket_probabilities=None)
                events.append({'action': 'drop_pocket_to_unknown'})
        if side_rng.random() < config.drop_binary_probability:
            if side.binary_fragment_coords is not None or side.binary_protein_coords is not None:
                side = replace(side, binary_fragment_coords=None, binary_protein_coords=None)
                events.append({'action': 'drop_binary_pose'})
        if side.pocket_probabilities is not None and config.pocket_noise_std > 0:
            probabilities = np.asarray(side.pocket_probabilities, dtype=np.float64)
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError('invalid input pocket probabilities')
            noisy = np.clip(probabilities + side_rng.normal(size=probabilities.shape) * config.pocket_noise_std, 0., 1.)
            side = replace(side, pocket_probabilities=noisy)
            events.append({'action': 'perturb_pocket_probabilities'})
            perturbed = True
        if side.binary_fragment_coords is not None and (config.binary_translation_std > 0 or config.binary_rotation_std > 0):
            if side.binary_protein_coords is None:
                raise ValueError('binary perturbation requires the local protein frame')
            coords, record = perturb_binary_pose(side.binary_fragment_coords, side.binary_protein_coords, side_rng,
                                                 config.binary_translation_std, config.binary_rotation_std)
            side = replace(side, binary_fragment_coords=coords)
            events.append({'action': 'perturb_binary_rigid_pose', **record})
            perturbed = True
        if perturbed:
            side = replace(side, confidence=side.confidence * config.perturbed_confidence_scale,
                           evidence_id=f'{side.evidence_id}#synthetic-condition-noise-v1:{seed}:{index}')
        if side.fragment is None and side.pocket_probabilities is None and side.binary_fragment_coords is None:
            side = None
        result.append(side)
        audit['events'].extend({'side': index, **event} for event in events)
    return tuple(result), tuple(linker_indices), audit

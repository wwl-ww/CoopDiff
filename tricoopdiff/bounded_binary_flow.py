"""Explicit bounded binary-pose uncertainty, separate from the strict chart.

This geometry module is not implicitly enabled by a manifest confidence value.
Translation bounds refer to the protected-fragment centroid, not a protein
centroid far from the binding site. It cannot repair internal fragment shapes.
"""

import math

import torch

from .anchored_flow import wrap_angles
from .pose_anchors import proper_rigid_fit
from .rigid import apply_rigid


BOUNDED_FLOW_VERSION = 'bounded_binary_torsion_chart_v1'


def ball_map(latent, radius):
    """Map R^3 smoothly into an open norm ball; never accumulate stepwise slack."""
    if not math.isfinite(radius) or radius < 0 or latent.shape != (3,):
        raise ValueError('bounded offset requires a 3-vector and finite nonnegative radius')
    if not torch.isfinite(latent).all():
        raise ValueError('nonfinite bounded offset')
    scale = latent.detach().abs().max().clamp_min(1)
    scaled = latent / scale
    return radius * scaled / torch.sqrt(scale.reciprocal().square() + scaled.square().sum())


def inverse_ball(value, radius):
    if (value.shape != (3,) or not torch.isfinite(value).all()
            or not math.isfinite(radius) or radius < 0):
        raise ValueError('invalid inverse bounded offset')
    if radius == 0:
        if value.norm() > 1e-7:
            raise ValueError('target deviates from an exact binary condition')
        return torch.zeros_like(value)
    if value.norm() >= radius:
        raise ValueError('target exceeds the declared total binary-offset bound')
    return value / torch.sqrt(radius ** 2 - value.square().sum())


def rotation_exp(vector):
    x, y, z = vector
    zero = x * 0
    matrix = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    return torch.linalg.matrix_exp(matrix)


class BoundedBinaryTorsionFlow:
    """State = K torsions + E3-side (translation, rotation) + POI-side offsets.

    E3-side offset moves the entire ligand/POI assembly. POI-side offset moves
    only POI around its protected fragment, in the transported reference frame.
    Both are ALWAYS applied to a fresh strict decode, never to the last step.
    """

    version = BOUNDED_FLOW_VERSION

    def __init__(self, strict, end1, end2, *, translation_bounds, rotation_bounds):
        self.strict = strict
        self.end1, self.end2 = tuple(end1), tuple(end2)
        self.translation_bounds = tuple(float(x) for x in translation_bounds)
        self.rotation_bounds = tuple(float(x) for x in rotation_bounds)
        if (len(self.translation_bounds) != 2 or len(self.rotation_bounds) != 2
                or any(not math.isfinite(x) or x < 0 for x in self.translation_bounds)
                or any(not math.isfinite(x) or not 0 <= x < math.pi for x in self.rotation_bounds)):
            raise ValueError('declare two translation bounds in Angstrom and rotation bounds below pi')
        if (len(set(self.end1)) != len(self.end1) or len(set(self.end2)) != len(self.end2)
                or len(self.end1) < 3 or len(self.end2) < 3 or set(self.end1) & set(self.end2)
                or any(type(i) is not int or not 0 <= i < len(strict.ligand)
                       for i in (*self.end1, *self.end2))):
            raise ValueError('bounded chart requires disjoint mapped cores with at least three atoms')
        for indices in (self.end1, self.end2):
            proper_rigid_fit(strict.ligand[list(indices)], strict.ligand[list(indices)])
        for moving, follows in zip(strict.plan.moving_atoms, strict.plan.moves_poi):
            if set(moving) & set(self.end1) or (
                    bool(set(moving) & set(self.end2)) != bool(follows)) or (
                    follows and not set(self.end2) <= set(moving)):
                raise ValueError('offset cores disagree with the strict torsion plan')
        self.torsion_count = len(strict.plan.bonds)
        self.size = self.torsion_count + 12
        self.plan = strict.plan

    def consensus_velocity(self, normalized_torsion):
        return self.strict.consensus_velocity(normalized_torsion)

    @staticmethod
    def offset_map(value, scale):
        return ball_map(value, scale)

    @staticmethod
    def offset_inverse(value, scale):
        return inverse_ball(value, scale)

    @staticmethod
    def offset_derivatives(latent, translation, rotation):
        from .bounded_joint_motion import _offset_derivatives
        return _offset_derivatives(latent, translation, rotation)

    def cartesian_velocity(self, state, velocity, *, create_graph=False):
        self._state(state)
        self._state(velocity)
        return torch.autograd.functional.jvp(
            lambda value: self.decode(value)[:2], state, velocity,
            create_graph=create_graph)[1]

    def _state(self, state):
        if (state.shape != (self.size,) or state.device != self.strict.ligand.device
                or state.dtype != self.strict.ligand.dtype or not torch.isfinite(state).all()):
            raise ValueError('bounded state shape/dtype/device mismatch')

    def decode(self, state):
        self._state(state)
        ligand, poi, body = self.strict.decode(state[:self.torsion_count])
        dtype = ligand.dtype
        ligand, poi, body = ligand.double(), poi.double(), body.double()
        offsets = state[self.torsion_count:].double().reshape(2, 2, 3)
        translations = [self.offset_map(offsets[i, 0], self.translation_bounds[i]) for i in range(2)]
        rotations = [rotation_exp(self.offset_map(offsets[i, 1], self.rotation_bounds[i])) for i in range(2)]
        transported = body @ self.strict.body.double().T
        rp = transported @ rotations[1] @ transported.T
        cp = ligand[list(self.end2)].mean(0)
        poi = (poi - cp) @ rp.T + cp + transported @ translations[1]
        body = rp @ body
        ce = self.strict.ligand[list(self.end1)].double().mean(0)
        re = rotations[0]
        ligand = (ligand - ce) @ re.T + ce + translations[0]
        poi = (poi - ce) @ re.T + ce + translations[0]
        body = re @ body
        return tuple(value.to(dtype) for value in (ligand, poi, body))

    def sample_prior(self, *, generator):
        torsions = self.strict.sample_prior(generator=generator)
        offsets = torch.randn((2, 2, 3), generator=generator, device=torsions.device, dtype=torsions.dtype)
        for side in range(2):
            if self.translation_bounds[side] == 0:
                offsets[side, 0].zero_()
            if self.rotation_bounds[side] == 0:
                offsets[side, 1].zero_()
        return torch.cat((torsions, offsets.flatten()))

    def path(self, target, prior, time):
        self._state(target)
        self._state(prior)
        if not math.isfinite(float(time)) or not 0 <= float(time) <= 1:
            raise ValueError('invalid bounded flow time')
        velocity = prior - target
        velocity = torch.cat((wrap_angles(velocity[:self.torsion_count]), velocity[self.torsion_count:]))
        state = target + float(time) * velocity
        state = torch.cat((wrap_angles(state[:self.torsion_count]), state[self.torsion_count:]))
        return state, self.decode(state), velocity

    def step(self, state, velocity, dt):
        self._state(state)
        self._state(velocity)
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError('invalid bounded integration step')
        updated = state - float(dt) * velocity
        updated = torch.cat((wrap_angles(updated[:self.torsion_count]), updated[self.torsion_count:]))
        return updated, self.decode(updated)

    @torch.no_grad()
    def encode_target(self, ligand, poi, edge_index, *, tolerance=1e-3):
        """Invert only reachable targets; reference fitting never replaces a target."""
        from scipy.spatial.transform import Rotation

        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('invalid bounded target tolerance')
        for actual, expected in ((ligand, self.strict.ligand), (poi, self.strict.poi)):
            if (actual.shape != expected.shape or actual.device != expected.device
                    or actual.dtype != expected.dtype or not torch.isfinite(actual).all()):
                raise ValueError('bounded target must match reference shape/dtype/device')
        ref = self.strict.ligand.double()
        re, te, _ = proper_rigid_fit(ref[list(self.end1)], ligand[list(self.end1)].double())
        ligand_base = (ligand.double() - te) @ re
        poi_base = (poi.double() - te) @ re
        r2, t2, _ = proper_rigid_fit(ref[list(self.end2)], ligand_base[list(self.end2)])
        compatible_poi = apply_rigid(self.strict.poi.double(), r2, t2)
        angles, _ = self.strict.encode_target(ligand_base.to(ligand), compatible_poi.to(poi),
                                               edge_index, tolerance=tolerance)
        base_ligand, base_poi, base_body = (value.double() for value in self.strict.decode(angles))
        rp, tp, _ = proper_rigid_fit(base_poi, poi_base)
        transported = base_body @ self.strict.body.double().T
        ce, cp = ref[list(self.end1)].mean(0), base_ligand[list(self.end2)].mean(0)
        physical_translations = [re @ ce + te - ce, transported.T @ (rp @ cp + tp - cp)]
        physical_rotations = [re, transported.T @ rp @ transported]
        offsets = []
        for i in range(2):
            rotvec = torch.as_tensor(Rotation.from_matrix(physical_rotations[i].cpu().numpy()).as_rotvec(),
                                     device=ref.device, dtype=ref.dtype)
            offsets.extend((self.offset_inverse(physical_translations[i], self.translation_bounds[i]),
                            self.offset_inverse(rotvec, self.rotation_bounds[i])))
        state = torch.cat((angles, torch.cat(offsets).to(angles)))
        decoded = self.decode(state)
        errors = {'ligand_max_error': float((decoded[0] - ligand).norm(dim=-1).max()),
                  'poi_max_error': float((decoded[1] - poi).norm(dim=-1).max())}
        if max(errors.values()) > tolerance:
            raise ValueError(f'target outside bounded binary chart: {errors}')
        return state, errors

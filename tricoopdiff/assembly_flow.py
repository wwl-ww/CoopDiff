"""Soft binary assembly: local poses are evidence, not hard displacement bounds."""

import math

import torch

from .bounded_binary_flow import BoundedBinaryTorsionFlow


ASSEMBLY_FLOW = 'unbounded_binary_pose_assembly_chart_v1'
ASSEMBLY_TRAINING = 'chemical_binary_pose_corruption_assembly_training_v1'


class AssemblyFlow(BoundedBinaryTorsionFlow):
    """Reuse one joint decoder with linear translation and exponential rotation.

    Inherited `*_bounds` attributes store unit scales here, NOT displacement
    bounds. End internal geometry remains rigid; each binary relative pose is
    free. This does not solve non-torsion fragment shape mismatch.
    """

    version = ASSEMBLY_FLOW

    def __init__(self, strict, end1, end2):
        super().__init__(strict, end1, end2, translation_bounds=(10., 10.), rotation_bounds=(1., 1.))

    @staticmethod
    def offset_map(value, scale):
        return value * scale

    @staticmethod
    def offset_inverse(value, scale):
        return value / scale

    @staticmethod
    def offset_derivatives(latent, translation, rotation):
        identity = torch.eye(3, device=latent.device, dtype=latent.dtype)
        vector = latent[1] * rotation
        x, y, z = vector.unbind()
        zero = x * 0
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
        theta = vector.norm()
        squared = theta.square()
        a = .5 * torch.sinc(theta / (2 * math.pi)).square()
        b = torch.where(theta < 1e-4, 1/6 - squared/120 + squared.square()/5040,
                        (1 - torch.sinc(theta / math.pi)) / squared.clamp_min(1e-16))
        return identity * translation, (identity + a * skew + b * skew @ skew) * rotation


def prepare_assembly_sample(e3, poi, ligand, *, generator, evidence_id):
    from .anchored_training import prepare_augmented_anchored_sample

    sample = prepare_augmented_anchored_sample(e3, poi, ligand, generator=generator,
                                               evidence_id=evidence_id)
    if sample is None:
        return None
    sample.flow = AssemblyFlow(sample.flow, *sample.audit['selected_atom_maps'])
    sample.data_angles, target = sample.flow.encode_target(
        ligand['coords'][0], poi['coords'][0], ligand['edge_index'])
    sample.prior_angles = sample.flow.sample_prior(generator=generator)
    sample.audit.update(training_contract=ASSEMBLY_TRAINING, target=target,
                        binary_bounds=None, local_pose_is_soft_condition=True,
                        independent_binary_shape_mismatch_solved=False)
    return sample

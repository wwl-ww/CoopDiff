"""Training-only perturbed binary templates with reachable nonzero offsets.

Native is supervision, never deployment input. Synthetic perturbations cover a
declared rigid uncertainty task; they do not reproduce arbitrary template shape
differences or establish a calibrated experimental uncertainty distribution.
"""

import math

import torch
from rdkit import Chem

from .anchored_flow import AnchoredTorsionFlow
from .anchored_training import AnchoredTrainingSample
from .bounded_binary_flow import BoundedBinaryTorsionFlow
from .condition_features import SideCondition, build_training_condition_features


BOUNDED_TRAINING_CONTRACT = 'perturbed_binary_native_supervision_bounded_v1'


def validate_binary_bounds(bounds):
    if bounds is None:
        return
    if not isinstance(bounds, dict) or set(bounds) != {'translation', 'rotation'}:
        raise ValueError('binary bounds must declare translation and rotation for both sides')
    for name, values in bounds.items():
        if (not isinstance(values, (tuple, list)) or len(values) != 2
                or any(not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0
                       or (name == 'rotation' and x >= math.pi) for x in values)):
            raise ValueError('bounded training requires positive finite bounds, rotation below pi')


def augment_binary_uncertainty(sample, e3, poi, ligand, *, fragments, atom_maps,
                                generator, evidence_id, bounds):
    validate_binary_bounds(bounds)
    if bounds is None:
        raise ValueError('uncertainty augmentation requires explicit bounds')
    # Make perturbed template references first. Half-radius perturbations keep
    # the inverse target strictly inside the declared full-radius chart.
    perturbation = BoundedBinaryTorsionFlow(
        sample.flow, *atom_maps, translation_bounds=[x / 2 for x in bounds['translation']],
        rotation_bounds=[x / 2 for x in bounds['rotation']])
    state = sample.flow.ligand.new_zeros(perturbation.size)
    state[perturbation.torsion_count:] = torch.randn(
        12, device=state.device, dtype=state.dtype, generator=generator)
    reference_ligand, reference_poi, body = perturbation.decode(state)
    strict = AnchoredTorsionFlow(reference_ligand, reference_poi, body, sample.flow.plan)
    flow = BoundedBinaryTorsionFlow(strict, *atom_maps,
                                   translation_bounds=bounds['translation'], rotation_bounds=bounds['rotation'])
    target, target_audit = flow.encode_target(ligand['coords'][0], poi['coords'][0], ligand['edge_index'])
    molecule = Chem.Mol(ligand['mol'])
    sides = []
    for index, (protein, fragment, mapping) in enumerate(zip(
            (e3['coords'][0], reference_poi), fragments, atom_maps)):
        center = protein.mean(0)
        local_protein = protein - center
        local_fragment = reference_ligand[list(mapping)] - center
        pocket = (torch.cdist(local_protein, local_fragment).min(1).values <= 6.).float()
        graph = Chem.Mol(fragment)
        graph.RemoveAllConformers()
        sides.append(SideCondition(
            provenance='training_native_supervision', evidence_id=f'{evidence_id}:perturbed_side{index + 1}',
            fragment=graph, atom_map=tuple(mapping), pocket_probabilities=pocket,
            binary_fragment_coords=local_fragment, binary_protein_coords=local_protein))
    conditions = build_training_condition_features(
        molecule, [e3['coords'].shape[1], poi['coords'].shape[1]], sides)
    audit = {**sample.audit, 'training_contract': BOUNDED_TRAINING_CONTRACT,
             'binary_bounds': bounds, 'template_perturbation_radius_fraction': .5,
             'reference_geometry': 'native_internal_geometry_with_synthetic_binary_rigid_offsets',
             'target': target_audit, 'target_offset_latents': target[flow.torsion_count:].detach().cpu().tolist()}
    return AnchoredTrainingSample(flow, target, flow.sample_prior(generator=generator), conditions, audit)

"""Explicit-map, native-supervised sample preparation for experimental training.

This is not a deployment initializer. Chemical fragments/maps must be supplied;
we never label contact-nearest atoms as warheads or all remaining atoms linker.
"""

from dataclasses import dataclass

import torch
from rdkit import Chem

from .anchored_flow import AnchoredTorsionFlow, audit_bond_geometry
from .anchored_torsion import compile_anchored_torsions
from .condition_features import SideCondition, build_training_condition_features


TRAINING_CONTRACT = 'explicit_mapped_binary_native_supervision_anchored_v1'


@dataclass
class AnchoredTrainingSample:
    flow: AnchoredTorsionFlow
    data_angles: torch.Tensor
    prior_angles: torch.Tensor
    conditions: dict
    audit: dict


def prepare_augmented_anchored_sample(e3_data, poi_data, ligand_data, *, generator,
                                      evidence_id, max_trials=64, binary_bounds=None):
    """Training-only chemical-cut augmentation; None means inapplicable, not fallback."""
    from .anchored_fragments import select_training_fragments

    selection = select_training_fragments(
        ligand_data['mol'], ligand_data['edge_index'], ligand_data['edge_mask'],
        ligand_data['coords'][0], (e3_data['coords'][0], poi_data['coords'][0]),
        generator=generator, max_trials=max_trials)
    if selection is None:
        return None
    sample = prepare_anchored_training_sample(
        e3_data, poi_data, ligand_data, fragments=selection['fragments'],
        atom_maps=selection['atom_maps'], generator=generator, evidence_id=evidence_id)
    sample.audit['fragment_augmentation'] = selection['audit']
    if binary_bounds is not None:
        from .bounded_binary_training import augment_binary_uncertainty
        sample = augment_binary_uncertainty(
            sample, e3_data, poi_data, ligand_data, fragments=selection['fragments'],
            atom_maps=selection['atom_maps'], generator=generator,
            evidence_id=evidence_id, bounds=binary_bounds)
    return sample


def prepare_anchored_training_sample(e3_data, poi_data, ligand_data, *,
                                     fragments, atom_maps, generator,
                                     evidence_id, pocket_cutoff=6.):
    """Create consistent local conditions, chart, prior and reachable supervision.

    Input coordinates are one native training complex in a common frame. Each
    chemical fragment has exactly one explicit selected map for this trajectory.
    Side-local distance features contain no cross-protein native transform.
    The chart's fixed geometry is native-supervised, unlike inference's binary
    assembly: this mismatch remains explicit in the audit, not silently solved.
    """
    if len(fragments) != 2 or len(atom_maps) != 2:
        raise ValueError('two explicit chemical fragments and maps are required')
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError('native supervision requires an explicit evidence identifier')
    if not 0 < pocket_cutoff < 20:
        raise ValueError('invalid training pocket cutoff')
    for data in (e3_data, poi_data, ligand_data):
        coords = data['coords']
        if (coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[2] != 3
                or not coords.shape[1] or not torch.isfinite(coords).all()):
            raise ValueError('anchored training expects one finite unpadded complex')
    ligand = ligand_data['coords'][0].detach()
    proteins = [data['coords'][0].detach() for data in (e3_data, poi_data)]
    molecule = Chem.Mol(ligand_data['mol'])
    plan = compile_anchored_torsions(
        len(ligand), ligand_data['edge_index'], ligand_data['edge_mask'], *atom_maps,
        molecule=molecule)
    if not plan.bonds:
        raise ValueError('explicit fragments leave no trainable anchored torsion')
    bond_audit = audit_bond_geometry(molecule, ligand)
    sides = []
    for index, (protein, fragment, mapping) in enumerate(zip(proteins, fragments, atom_maps)):
        graph = Chem.Mol(fragment)
        graph.RemoveAllConformers()
        # Remove even the shared translation before handing coordinates to the
        # condition builder, which emits only each side's internal distances.
        center = protein.mean(0)
        local_protein = protein - center
        local_fragment = ligand[list(mapping)] - center
        pocket = (torch.cdist(local_protein, local_fragment).min(1).values <= pocket_cutoff).float()
        sides.append(SideCondition(
            provenance='training_native_supervision', evidence_id=f'{evidence_id}:side{index + 1}',
            fragment=graph, atom_map=tuple(mapping), pocket_probabilities=pocket,
            binary_fragment_coords=local_fragment, binary_protein_coords=local_protein))
    conditions = build_training_condition_features(molecule, [len(p) for p in proteins], sides)
    body = torch.eye(3, dtype=ligand.dtype, device=ligand.device)
    native_chart = AnchoredTorsionFlow(ligand, proteins[1], body, plan)
    # A randomized chart origin avoids encoding every training target as q=0.
    # The prior is independently uniform over the same allowed periodic angles.
    origin = native_chart.sample_prior(generator=generator)
    flow = AnchoredTorsionFlow(*native_chart.decode(origin), plan)
    data_angles, target_audit = flow.encode_target(
        ligand, proteins[1], ligand_data['edge_index'])
    prior = flow.sample_prior(generator=generator)
    audit = {'training_contract': TRAINING_CONTRACT, 'evidence_id': evidence_id,
             'condition_provenance': 'training_native_supervision',
             'reference_geometry': 'native_training_geometry_not_independent_binary_assembly',
             'cross_protein_native_transform_in_features': False,
             'selected_atom_maps': [list(mapping) for mapping in atom_maps],
             'allowed_torsion_indices': list(plan.original_indices),
             'bond_geometry': bond_audit, 'target': target_audit}
    return AnchoredTrainingSample(flow, data_angles, prior, conditions, audit)

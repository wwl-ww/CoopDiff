"""Core-protected training using the existing joint decoder and motion heads."""

import torch
from rdkit import Chem

from .anchored_fragments import select_training_fragments
from .anchored_training import prepare_anchored_training_sample
from .binding_core import CORE_POLICY, CONTACT_CORE_POLICY, select_binding_core
from .bounded_binary_training import augment_binary_uncertainty
from .condition_features import SideCondition, build_training_condition_features
from .pose_anchors import proper_rigid_fit
from .rigid import apply_rigid


CORE_TRAINING_CONTRACT = 'binary_core_protected_flexible_periphery_training_v1'
CONTACT_CORE_TRAINING_CONTRACT = 'binary_contact_blocks_protected_training_v1'


def prepare_core_training_sample(e3, poi, ligand, *, fragments, atom_maps,
                                 generator, evidence_id, binary_bounds, all_contact_blocks=False):
    if len(fragments) != 2 or len(atom_maps) != 2 or binary_bounds is None:
        raise ValueError('core training requires two mapped fragments and explicit binary bounds')
    cores = [select_binding_core(fragment, mapping, ligand['coords'][0, list(mapping)], protein['coords'][0],
                                all_contact_blocks=all_contact_blocks)
             for fragment, mapping, protein in zip(fragments, atom_maps, (e3, poi))]
    core_fragments = tuple(core['fragment'] for core in cores)
    core_maps = tuple(core['atom_map'] for core in cores)
    sample = prepare_anchored_training_sample(e3, poi, ligand,
        fragments=core_fragments, atom_maps=core_maps, generator=generator, evidence_id=evidence_id)
    sample = augment_binary_uncertainty(sample, e3, poi, ligand,
        fragments=core_fragments, atom_maps=core_maps, generator=generator,
        evidence_id=evidence_id, bounds=binary_bounds)
    # Only the core is hard-protected; retain ALL observed fragment evidence as
    # soft conditioning. Transport each complete local template by its core fit,
    # not by the random chart's peripheral torsions (which are unknown motion).
    strict = sample.flow.strict
    soft_sides = []
    for index, (fragment, mapping, core_map, protein) in enumerate(zip(
            fragments, atom_maps, core_maps, (e3['coords'][0], strict.poi))):
        rotation, translation, error = proper_rigid_fit(
            ligand['coords'][0, list(core_map)], strict.ligand[list(core_map)])
        if error > 1e-3:
            raise ValueError('protected core deformed during template transport')
        xyz = apply_rigid(ligand['coords'][0, list(mapping)], rotation, translation).to(protein)
        center = protein.mean(0)
        graph = Chem.Mol(fragment)
        graph.RemoveAllConformers()
        soft_sides.append(SideCondition(provenance='training_native_supervision',
            evidence_id=f'{evidence_id}:full_binary_side{index + 1}', fragment=graph,
            atom_map=tuple(mapping), binary_fragment_coords=xyz - center,
            binary_protein_coords=protein - center,
            pocket_probabilities=(torch.cdist(protein, xyz).min(1).values <= 6.).float()))
    sample.conditions = build_training_condition_features(Chem.Mol(ligand['mol']),
        [len(e3['coords'][0]), len(poi['coords'][0])], soft_sides)
    sample.audit.update(training_contract=CONTACT_CORE_TRAINING_CONTRACT if all_contact_blocks else CORE_TRAINING_CONTRACT,
                        fragment_augmentation={'policy': CONTACT_CORE_POLICY if all_contact_blocks else CORE_POLICY},
                        binding_cores=[core['audit'] for core in cores],
                        flexible_periphery='chemical_rotors_outside_protected_cores',
                        full_binary_template_retained_as_soft_condition=True,
                        independent_binary_shape_mismatch_solved=False)
    return sample


def prepare_augmented_core_sample(e3, poi, ligand, *, generator, evidence_id, binary_bounds,
                                  all_contact_blocks=False):
    selection = select_training_fragments(ligand['mol'], ligand['edge_index'], ligand['edge_mask'],
        ligand['coords'][0], (e3['coords'][0], poi['coords'][0]), generator=generator)
    if selection is None:
        return None
    sample = prepare_core_training_sample(e3, poi, ligand, fragments=selection['fragments'],
        atom_maps=selection['atom_maps'], generator=generator, evidence_id=evidence_id,
        binary_bounds=binary_bounds, all_contact_blocks=all_contact_blocks)
    sample.audit['source_fragment_augmentation'] = selection['audit']
    return sample

"""Native-supervised one-sided completion; no fabricated unknown-side pose."""

import torch
from rdkit import Chem

from .anchored_flow import AnchoredTorsionFlow, audit_bond_geometry
from .anchored_training import AnchoredTrainingSample
from .condition_features import SideCondition, _build_condition_features, validate_atom_maps
from .initialization import receptor_prior
from .single_anchor_flow import SingleAnchorTorsionFlow, compile_single_anchor_torsions


SINGLE_TRAINING_CONTRACT = 'single_anchor_native_completion_v1'
SINGLE_FRAGMENT_POLICY = 'one_bridge_cut_or_complete_ligand_training_augmentation_v1'


def select_single_training_fragment(molecule, ligand, receptor, *, generator):
    """Select a whole connected chemical fragment, not nearest-atom labels."""
    mol = Chem.Mol(molecule)
    candidates = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        mappings = []
        try:
            fragments = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=False),
                                         asMols=True, sanitizeFrags=True, fragsMolAtomMapping=mappings)
        except (ValueError, RuntimeError):
            continue
        for fragment, mapping in zip(fragments, mappings):
            if len(mapping) >= 3:
                candidates.append((fragment, tuple(mapping), bond.GetIdx()))
    order = torch.randperm(len(candidates), generator=generator, device=generator.device).cpu().tolist()
    # A rigid ligand can still supply a known binary pose for free protein docking.
    ordered = [candidates[i] for i in order] + [(Chem.Mol(mol), tuple(range(mol.GetNumAtoms())), None)]
    distances = torch.cdist(ligand, receptor).min(1).values
    for fragment, mapping, cut in ordered:
        if len(mapping) < 3 or not (distances[list(mapping)] <= 6.).any():
            continue
        try:
            validate_atom_maps(mol, SideCondition('training_native_supervision', SINGLE_FRAGMENT_POLICY,
                                                fragment=fragment, atom_map=mapping))
        except ValueError:
            continue
        return fragment, mapping, cut
    return None


def prepare_single_anchor_sample(e3, poi, ligand, *, generator, evidence_id,
                                 translation_bound=.5, rotation_bound=.2):
    from scipy.spatial.transform import Rotation

    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError('single-anchor native supervision requires an evidence identifier')
    for data in (e3, poi, ligand):
        coords = data['coords']
        if (coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[2] != 3
                or not coords.shape[1] or not torch.isfinite(coords).all()):
            raise ValueError('single-anchor training requires one finite unpadded complex')
    native = ligand['coords'][0].detach()
    bond_audit = audit_bond_geometry(ligand['mol'], native)
    selected = select_single_training_fragment(ligand['mol'], native, e3['coords'][0], generator=generator)
    if selected is None:
        return None
    fragment, mapping, cut = selected
    plan = compile_single_anchor_torsions(ligand['mol'], ligand['edge_index'], ligand['edge_mask'], mapping)
    identity = torch.eye(3, device=native.device, dtype=native.dtype)
    # Randomize both chart origins; the free POI target must not always be R=I.
    native_chart = AnchoredTorsionFlow(native, poi['coords'][0], identity, plan)
    reference_ligand = native_chart.decode(native_chart.sample_prior(generator=generator))[0]
    quaternion = torch.randn(4, generator=generator, device=native.device, dtype=native.dtype)
    rotation = torch.as_tensor(Rotation.from_quat(quaternion.cpu().numpy()).as_matrix(),
                               dtype=native.dtype, device=native.device)
    reference_poi = (poi['coords'][0] - poi['coords'][0].mean(0)) @ rotation.T
    center, _, _ = receptor_prior(e3['coords'][0])
    _, _, scale = receptor_prior(poi['coords'][0])
    strict = AnchoredTorsionFlow(reference_ligand, reference_poi, rotation, plan)
    perturb = SingleAnchorTorsionFlow(strict, mapping, center, scale,
                                     translation_bound=translation_bound / 2, rotation_bound=rotation_bound / 2)
    state = native.new_zeros(perturb.size)
    state[perturb.torsion_count:perturb.torsion_count + 6] = torch.randn(
        6, generator=generator, device=native.device, dtype=native.dtype)
    perturbed_ligand = perturb.decode(state)[0]
    strict = AnchoredTorsionFlow(perturbed_ligand, reference_poi, rotation, plan)
    flow = SingleAnchorTorsionFlow(strict, mapping, center, scale,
                                   translation_bound=translation_bound, rotation_bound=rotation_bound)
    target, target_audit = flow.encode_target(native, poi['coords'][0], ligand['edge_index'])
    fragment = Chem.Mol(fragment)
    fragment.RemoveAllConformers()
    local_protein = e3['coords'][0] - center
    local_fragment = perturbed_ligand[list(mapping)] - center
    pocket = (torch.cdist(local_protein, local_fragment).min(1).values <= 6.).float()
    side = SideCondition('training_native_supervision', evidence_id, fragment=fragment, atom_map=mapping,
                          pocket_probabilities=pocket, binary_fragment_coords=local_fragment,
                          binary_protein_coords=local_protein)
    conditions = _build_condition_features(Chem.Mol(ligand['mol']),
        [e3['coords'].shape[1], poi['coords'].shape[1]], (side, None),
        allowed_provenance=('training_native_supervision',))
    audit = {'training_contract': SINGLE_TRAINING_CONTRACT, 'evidence_id': evidence_id,
             'fragment_augmentation': {'policy': SINGLE_FRAGMENT_POLICY, 'cut_bond_index': cut},
             'selected_atom_maps': [list(mapping), None], 'conditioned_sides': [True, False],
             'known_side_bounds': {'translation': translation_bound, 'rotation': rotation_bound},
             'allowed_torsion_indices': list(plan.original_indices), 'target': target_audit,
             'bond_geometry': bond_audit, 'cross_protein_native_transform_in_features': False}
    return AnchoredTrainingSample(flow, target, flow.sample_prior(generator=generator), conditions, audit)

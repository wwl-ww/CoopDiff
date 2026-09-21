"""Build deployment-style references before any target fitting or selection.

The caller supplies protein-local binary evidence with explicit provenance.
Synthetic training binary evidence must remain labeled synthetic; this builder
does not manufacture independent experimental evidence from native structures.
"""

import torch
from rdkit import Chem

from .binary_chart_bank import build_binary_chart_bank
from .binary_pose_constraints import BinaryPoseInitializer
from .bounded_binary_flow import BoundedBinaryTorsionFlow
from .bounded_binary_training import validate_binary_bounds


def build_independent_training_reference(sides, e3, poi, ligand, *, seed, generator,
                                         binary_bounds, evidence_provenance,
                                         max_candidates=24, max_fit_rmsd=.15):
    """Pick a chart independently of native targets; strip all ligand conformers."""
    if evidence_provenance not in ('independent_binary', 'synthetic_native_local_pairs'):
        raise ValueError('explicit independent or synthetic local binary provenance required')
    if len(sides) != 2 or any(side is None for side in sides):
        raise ValueError('BiAnchor reference requires two mapped binary sides')
    validate_binary_bounds(binary_bounds)
    if binary_bounds is None:
        raise ValueError('training reference needs explicit binary uncertainty bounds')
    molecule = Chem.Mol(ligand['mol'])
    molecule.RemoveAllConformers()
    chemistry = {key: ligand[key] for key in ('edge_index', 'edge_mask', 'mask_rotate')}
    chemistry['mol'] = molecule
    bank, audit = build_independent_training_reference_bank(
        sides, e3, poi, ligand, seed=seed, binary_bounds=binary_bounds,
        evidence_provenance=evidence_provenance, max_candidates=max_candidates,
        max_fit_rmsd=max_fit_rmsd)
    index = int(torch.randint(len(bank), (), generator=generator, device=generator.device))
    return bank[index][0], {**audit, 'reference_index': index, 'selected': bank[index][1]}


def build_independent_training_reference_bank(sides, e3, poi, ligand, *, seed,
                                              binary_bounds, evidence_provenance,
                                              max_candidates=24, max_fit_rmsd=.15):
    """Return every valid deployment reference; selection stays separate."""
    if evidence_provenance not in ('independent_binary', 'synthetic_native_local_pairs'):
        raise ValueError('explicit independent or synthetic local binary provenance required')
    if len(sides) != 2 or any(side is None for side in sides):
        raise ValueError('BiAnchor reference requires two mapped binary sides')
    validate_binary_bounds(binary_bounds)
    if binary_bounds is None:
        raise ValueError('training reference needs explicit binary uncertainty bounds')
    molecule = Chem.Mol(ligand['mol'])
    molecule.RemoveAllConformers()
    chemistry = {key: ligand[key] for key in ('edge_index', 'edge_mask', 'mask_rotate')}
    chemistry['mol'] = molecule
    strict_bank, audit = build_binary_chart_bank(
        BinaryPoseInitializer(*sides, max_candidates=max_candidates), e3, poi, chemistry,
        seed=seed, max_fit_rmsd=max_fit_rmsd)
    # The generic deployment bank's native_information_used flag concerns its
    # algorithm only. Explicitly qualify it for native-derived training evidence.
    audit = {**audit, 'native_information_used': evidence_provenance == 'synthetic_native_local_pairs',
             'binary_evidence_provenance': evidence_provenance,
             'native_cross_protein_transform_used': False,
             'full_ligand_input_coordinates_used': False,
             'reference_selection': 'uniform_valid_reference_before_target_fitting'}
    bank = []
    for strict, selected in strict_bank:
        flow = BoundedBinaryTorsionFlow(strict, *selected['selected_atom_maps'],
                                       translation_bounds=binary_bounds['translation'],
                                       rotation_bounds=binary_bounds['rotation'])
        bank.append((flow, selected))
    return bank, audit

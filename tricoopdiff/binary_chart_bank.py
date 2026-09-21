"""Retain native-free binary reference alternatives for constrained generation.

Reference internal geometry and atom maps are discrete conditioning variables.
Keeping their validated bank avoids collapsing all walkers to one arbitrary
reference before the model has considered alternative assemblies.
"""

import math

import torch

from .anchored_flow import AnchoredTorsionFlow, audit_bond_geometry
from .anchored_torsion import compile_anchored_torsions
from .binary_pose_constraints import BinaryPoseInitializer
from .pose_anchors import proper_rigid_fit


BANK_POLICY = 'all_valid_binary_reference_charts_v1'


@torch.no_grad()
def build_binary_chart_bank(initializer, e3, poi, ligand, *, seed, max_fit_rmsd=.15):
    if (not isinstance(initializer, BinaryPoseInitializer) or initializer.no_torsion
            or not math.isfinite(max_fit_rmsd) or max_fit_rmsd <= 0):
        raise ValueError('binary chart bank requires a torsion-enabled initializer and finite fit bound')
    collected = []

    def collect(candidate, maps):
        if candidate['e3_frame_rmsd'] > 1e-3:
            raise ValueError('binary side1 protein coordinates differ from current E3 input')
        fits = [candidate['side1_fit_rmsd'], candidate['side2_fit_rmsd']]
        if max(fits) > max_fit_rmsd:
            raise ValueError(f'binary assembly exceeds declared fit tolerance: {fits}')
        bonds = audit_bond_geometry(ligand['mol'], candidate['ligand_coords'])
        plan = compile_anchored_torsions(candidate['ligand_coords'].shape[0], ligand['edge_index'],
            ligand['edge_mask'], *maps, molecule=ligand['mol'])
        audit = {'policy': BANK_POLICY, 'candidate_index': candidate['map_pair_index'],
                 'selected_atom_maps': [list(mapping) for mapping in maps],
                 'side_fit_rmsd': fits, 'e3_frame_rmsd': candidate['e3_frame_rmsd'],
                 'bond_geometry': bonds, 'allowed_torsion_indices': list(plan.original_indices)}
        collected.append((candidate, plan, audit))
        return audit

    # Existing initializer verifies both input receptor identities as well as
    # the candidate gate. Its legacy clash-selected winner is not used here.
    _, _, _, source_audit = initializer.initialize(e3, poi, ligand, seed=seed, candidate_validator=collect)
    binary_poi = torch.as_tensor(initializer.side2.protein_coords, dtype=poi.dtype, device=poi.device)
    input_rotation, _, error = proper_rigid_fit(binary_poi, poi)
    if error > 1e-3:
        raise ValueError('binary side2 protein coordinates differ from current POI input')
    charts = []
    for candidate, plan, audit in collected:
        body = candidate['poi_rotation'] @ input_rotation.to(candidate['poi_rotation']).T
        flow = AnchoredTorsionFlow(candidate['ligand_coords'], candidate['poi_coords'], body, plan)
        charts.append((flow, {**audit, 'generative_process': flow.version}))
    return charts, {'policy': BANK_POLICY, 'seed': seed, 'candidate_budget': initializer.max_candidates,
                    'valid_references': len(charts), 'native_information_used': False,
                    'embedding_failures': source_audit['embedding_failures'],
                    'constraint_rejections': source_audit['constraint_rejections'],
                    'reference_selection': 'retain_all_valid_not_clash_ranked'}


def reference_conformer_schedule(reference_count, conformer_count, walkers, *, generator):
    """Cover references cyclically and never repeat a discrete proposal pair."""
    if (any(type(x) is not int or x < 1 for x in (reference_count, conformer_count, walkers))
            or walkers > reference_count * conformer_count):
        raise ValueError('walker budget exceeds distinct binary-reference/conformer combinations')
    reference_order = torch.randperm(reference_count, generator=generator, device=generator.device).cpu().tolist()
    conformer_orders = [torch.randperm(conformer_count, generator=generator, device=generator.device).cpu().tolist()
                       for _ in range(reference_count)]
    return [(reference_order[i % reference_count], conformer_orders[reference_order[i % reference_count]][i // reference_count])
            for i in range(walkers)]

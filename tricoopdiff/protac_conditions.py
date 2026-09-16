"""Chemical mapping and protein-local pocket audits, with no ternary native input."""

import numpy as np


CONDITION_SCHEMA_VERSION = 'protac_condition_asset_audit_v1'


def fragment_matches(full_molecule, fragment, limit=128):
    """Retain stereo-aware graph matches; do not resolve symmetry using coordinates."""
    if type(limit) is not int or limit < 1:
        raise ValueError('positive integer mapping limit required')
    matches = full_molecule.GetSubstructMatches(
        fragment, useChirality=True, uniquify=False, maxMatches=limit + 1)
    atom_sets = sorted({tuple(sorted(match)) for match in matches[:limit]})
    return {'atom_maps': [list(match) for match in matches[:limit]],
            'truncated': len(matches) > limit,
            'role_atom_sets': [list(indices) for indices in atom_sets],
            'unambiguous_role_atoms': list(atom_sets[0]) if len(atom_sets) == 1 and len(matches) <= limit else None,
            'mapping_status': ('unmatched' if not matches else
                               'unique' if len(matches) == 1 else 'ambiguous'),
            'mapping_method': 'exact_stereo_chemical_subgraph_no_spatial_matching'}


def attachment_bonds(molecule, atom_map):
    mapped = set(atom_map)
    return [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in molecule.GetBonds()
            if (bond.GetBeginAtomIdx() in mapped) != (bond.GetEndAtomIdx() in mapped)]


def mapping_diagnostics(full_molecule, fragment):
    """Identify matching obstacles; relaxed graphs must never authorize atom maps."""
    from collections import Counter
    from rdkit import Chem

    def connectivity_only(molecule):
        graph = Chem.RWMol()
        for atom in molecule.GetAtoms():
            graph.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        for bond in molecule.GetBonds():
            graph.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
        result = graph.GetMol()
        result.UpdatePropertyCache(strict=False)
        return result

    no_stereo = full_molecule.HasSubstructMatch(fragment, useChirality=False)
    topology = connectivity_only(full_molecule).HasSubstructMatch(connectivity_only(fragment))
    return {
        'chemistry_match_ignoring_stereo': bool(no_stereo),
        'element_connectivity_match': bool(topology),
        'fragment_bond_types': dict(Counter(str(b.GetBondType()) for b in fragment.GetBonds())),
        'full_bond_types': dict(Counter(str(b.GetBondType()) for b in full_molecule.GetBonds())),
        'interpretation': ('stereo_or_symmetry_requires_review' if no_stereo else
                           'bond_order_aromaticity_or_charge_requires_review' if topology else
                           'connectivity_or_capping_requires_review'),
        'relaxed_mapping_usable_for_inference': False,
    }


def local_pocket(protein_coords, residue_ids, fragment_coords, radius=8.):
    """Each binary side is measured only in its own protein/fragment frame."""
    protein, fragment = np.asarray(protein_coords), np.asarray(fragment_coords)
    if protein.shape != (len(residue_ids), 3) or fragment.ndim != 2 or fragment.shape[1] != 3:
        raise ValueError('invalid protein/fragment coordinates')
    if not len(protein) or not len(fragment) or not np.isfinite(protein).all() or not np.isfinite(fragment).all():
        raise ValueError('empty or non-finite local binary condition')
    distances = np.linalg.norm(protein[:, None] - fragment[None], axis=-1).min(1)
    pocket = sorted({residue_ids[i] for i in np.flatnonzero(distances < radius)})
    return {'residue_ids': pocket, 'radius_angstrom': radius,
            'minimum_heavy_atom_distance': float(distances.min())}

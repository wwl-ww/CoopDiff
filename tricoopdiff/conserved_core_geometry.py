"""Binary-local geometric evidence for conserved chemical cores.

These diagnostics cannot establish transfer accuracy in a ternary structure.
They preserve every chemical hypothesis; contact counts do not select a map.
"""

import ast

import numpy as np
from rdkit import Chem


def core_geometry_evidence(core, binary_fragment, protein_coordinates, *, cutoff=6.):
    """Keep canonical graph order, correspondence and binary coordinates aligned."""
    if binary_fragment.GetNumConformers() != 1 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('one binary conformer and positive finite contact cutoff required')
    protein = np.asarray(protein_coordinates, dtype=float)
    coordinates = np.asarray(binary_fragment.GetConformer().GetPositions(), dtype=float)
    if (protein.ndim != 2 or protein.shape[1] != 3 or not len(protein)
            or not np.isfinite(protein).all() or not np.isfinite(coordinates).all()):
        raise ValueError('invalid binary-local coordinates')
    selected = tuple(core['source_atom_indices'])
    fragment = Chem.Mol(core['fragment'])
    smiles = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
    order = ast.literal_eval(fragment.GetProp('_smilesAtomOutputOrder'))
    if sorted(order) != list(range(len(selected))):
        raise ValueError('incomplete canonical core atom ordering')
    reparsed = Chem.MolFromSmiles(smiles)
    if reparsed is None or reparsed.GetNumAtoms() != len(selected):
        raise ValueError('core graph serialization failed')
    distance = np.linalg.norm(coordinates[:, None] - protein[None], axis=-1)
    contact_atoms = distance.min(1) <= cutoff
    core_coordinates = coordinates[list(selected)][order]
    singular_values = np.linalg.svd(core_coordinates - core_coordinates.mean(0), compute_uv=False)
    return {
        'scope': 'conserved_analogue_core_not_complete_warhead',
        'fragment_smiles': smiles,
        'source_atom_indices_in_smiles_order': [selected[index] for index in order],
        'atom_maps': [[mapping[index] for index in order] for mapping in core['atom_maps']],
        'binary_fragment_coords': core_coordinates.tolist(),
        'binary_protein_coords': protein.tolist(),
        'pocket_probabilities': (distance[list(selected)].min(0) <= cutoff).astype(float).tolist(),
        'core_contact_atoms': int(contact_atoms[list(selected)].sum()),
        'source_contact_atoms': int(contact_atoms.sum()),
        'source_contact_coverage': (float(contact_atoms[list(selected)].sum() / contact_atoms.sum())
                                    if contact_atoms.any() else 0.),
        'core_centered_singular_values': singular_values.tolist(),
        'orientation_defined': bool(singular_values[1] > 1e-3),
        'contact_cutoff_angstrom': cutoff,
        'native_ternary_coordinates_used': False,
        'map_selection_by_coordinates': False,
        'deployment_eligible': False,
    }

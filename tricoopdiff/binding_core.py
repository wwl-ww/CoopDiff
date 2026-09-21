"""Select protected rigid blocks from one mapped binary fragment only.

No ternary alignment, cross-protein distances, or model/native scores enter this
selection. Flexible substituents are omitted from the protected core, not
relabeled as linker. A rigid block is a kinematic approximation, not proof of
experimental rigidity or a calibrated confidence estimate.
"""

import torch
from rdkit import Chem

from .anchored_torsion import chemical_rotatable_pairs


CORE_POLICY = 'binary_local_contact_rigid_block_v1'
CONTACT_CORE_POLICY = 'binary_local_all_contact_blocks_v1'


def select_binding_core(fragment, atom_map, fragment_coords, protein_coords, *, cutoff=6., all_contact_blocks=False):
    """Return a connected chemical block and its explicit parent atom mapping.

    Split only allowed chemical rotors. Prefer blocks with more contact atoms,
    then more atoms. Ambiguous best blocks are rejected, not atom-index selected.
    At least three contacting atoms and a rank-two frame are required.
    """
    graph = Chem.Mol(fragment)
    graph.RemoveAllConformers()
    mapping = tuple(atom_map)
    coords = torch.as_tensor(fragment_coords).detach().cpu().double()
    protein = torch.as_tensor(protein_coords).detach().cpu().double()
    if (len(mapping) != graph.GetNumAtoms() or len(set(mapping)) != len(mapping)
            or any(type(i) is not int or i < 0 for i in mapping)
            or coords.shape != (len(mapping), 3) or protein.ndim != 2
            or protein.shape[1] != 3 or not len(protein)
            or not torch.isfinite(coords).all() or not torch.isfinite(protein).all()
            or not 0 < cutoff < 20 or len(Chem.GetMolFrags(graph)) != 1):
        raise ValueError('invalid mapped binary fragment or local geometry')
    rotors = chemical_rotatable_pairs(graph)
    cuts = [b.GetIdx() for b in graph.GetBonds()
            if frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) in rotors]
    cut = Chem.FragmentOnBonds(graph, cuts, addDummies=False) if cuts else graph
    maps = []
    blocks = Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=maps)
    contact = torch.cdist(coords, protein).min(1).values <= cutoff
    candidates = []
    for block, indices in zip(blocks, maps):
        if len(indices) < 3 or int(contact[list(indices)].sum()) < 3:
            continue
        xyz = coords[list(indices)]
        if float(torch.linalg.svdvals(xyz - xyz.mean(0))[1]) < 1e-3:
            continue
        candidates.append(((int(contact[list(indices)].sum()), len(indices)), block, indices))
    if not candidates:
        raise ValueError('no contact-supported orientable rigid binding core')
    if all_contact_blocks:
        kept = {i for _, _, group in candidates for i in group}
        # Retain the chemical connector between blocks too. Restrict connectors
        # to the rotor-block tree: removing allowed bridge rotors cannot cut a
        # ring, and a branch on the minimal spanning subtree is unambiguous.
        groups = [set(group) for group in maps]
        edges = []
        for cut_index in cuts:
            bond = graph.GetBondWithIdx(cut_index)
            left = next(i for i, group in enumerate(groups) if bond.GetBeginAtomIdx() in group)
            right = next(i for i, group in enumerate(groups) if bond.GetEndAtomIdx() in group)
            if left != right:
                edges.append((left, right))
        active = set(range(len(groups)))
        terminals = {i for i, group in enumerate(groups) if group & kept}
        while True:
            removable = {i for i in active - terminals
                         if sum(i in edge and all(j in active for j in edge) for edge in edges) <= 1}
            if not removable:
                break
            active -= removable
        kept = set().union(*(groups[i] for i in active))
        boundary = [b.GetIdx() for b in graph.GetBonds()
                    if (b.GetBeginAtomIdx() in kept) != (b.GetEndAtomIdx() in kept)]
        separated = Chem.FragmentOnBonds(graph, boundary, addDummies=False) if boundary else graph
        separated_maps = []
        separated_blocks = Chem.GetMolFrags(separated, asMols=True, sanitizeFrags=True,
                                            fragsMolAtomMapping=separated_maps)
        chosen = [(part, ids) for part, ids in zip(separated_blocks, separated_maps) if set(ids) == kept]
        if len(chosen) != 1:
            raise ValueError('contact blocks do not define one connected protected scaffold')
        block, indices = chosen[0]
        score = (int(contact[list(indices)].sum()), len(indices))
    else:
        score = max(row[0] for row in candidates)
        best = [row for row in candidates if row[0] == score]
        if len(best) != 1:
            raise ValueError('ambiguous binding core; retain alternatives upstream rather than index-select')
        _, block, indices = best[0]
    return {'fragment': block, 'atom_map': tuple(mapping[i] for i in indices),
            'local_indices': tuple(indices),
            'audit': {'policy': CONTACT_CORE_POLICY if all_contact_blocks else CORE_POLICY,
                      'protected_blocks': len(candidates) if all_contact_blocks else 1,
                      'source_atom_map': list(mapping),
                      'protected_atom_map': [mapping[i] for i in indices],
                      'released_atom_map': [mapping[i] for i in range(len(mapping)) if i not in indices],
                      'contact_cutoff': cutoff, 'contact_atoms': score[0],
                      'cross_protein_geometry_used': False,
                      'uncertainty_calibrated': False}}

"""Chemical-cut augmentation for generic ternary training, never warhead inference."""

from itertools import combinations

import torch
from rdkit import Chem

from .condition_features import SideCondition, validate_atom_maps
from .anchored_torsion import chemical_rotatable_pairs


FRAGMENT_POLICY = 'two_bridge_cuts_connected_fragment_training_augmentation_v1'


def select_training_fragments(molecule, edge_index, edge_mask, ligand, proteins, *,
                              generator, min_fragment_atoms=3, cutoff=6., max_trials=64):
    """Sample chemical cuts, then orient intact ends by their own training contacts.

    Two bridge cuts must leave three connected components arranged end/middle/end.
    Geometry only orients/accepts a whole fragment; it never labels atoms one by
    one. No result is a claim of biological warhead identity or known linker.
    None means no applicable augmentation was found within the explicit budget.
    """
    if type(min_fragment_atoms) is not int or min_fragment_atoms < 3:
        raise ValueError('training fragments require at least three atoms')
    if type(max_trials) is not int or max_trials < 1 or not 0 < cutoff < 20:
        raise ValueError('invalid chemical-cut search budget/cutoff')
    mol = Chem.Mol(molecule)
    mol.RemoveAllConformers()
    count = mol.GetNumAtoms()
    ligand = torch.as_tensor(ligand).detach().cpu()
    proteins = [torch.as_tensor(p).detach().cpu() for p in proteins]
    if (ligand.shape != (count, 3) or not torch.isfinite(ligand).all()
            or len(proteins) != 2 or any(p.ndim != 2 or p.shape[1] != 3 or not len(p)
                                        or not torch.isfinite(p).all() for p in proteins)):
        raise ValueError('invalid training geometry')
    edges = torch.as_tensor(edge_index).detach().cpu()
    mask = torch.as_tensor(edge_mask).detach().cpu()
    if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long
            or mask.shape != (edges.shape[1],) or mask.dtype != torch.bool):
        raise ValueError('invalid chemical torsion graph')
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('training ligand must be connected')
    candidates = []
    allowed = chemical_rotatable_pairs(mol)
    for a, b in edges[:, mask].T.tolist():
        if not 0 <= a < count or not 0 <= b < count:
            raise ValueError('torsion atom outside molecule')
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            raise ValueError('torsion edge is not a chemical bond')
        if frozenset((a, b)) in allowed:
            candidates.append(bond.GetIdx())
    if len(set(candidates)) != len(candidates):
        raise ValueError('duplicate selected torsion bond')
    pairs = list(combinations(candidates, 2))
    order = torch.randperm(len(pairs), generator=generator,
                           device=generator.device).cpu().tolist()[:max_trials]
    nearest = [torch.cdist(ligand, p.to(ligand)).min(1).values for p in proteins]
    for trial, pair_index in enumerate(order, start=1):
        cuts = pairs[pair_index]
        cut_mol = Chem.FragmentOnBonds(mol, cuts, addDummies=False)
        mappings = []
        try:
            fragments = Chem.GetMolFrags(cut_mol, asMols=True, sanitizeFrags=True,
                                         fragsMolAtomMapping=mappings)
        except (ValueError, RuntimeError):
            continue
        if len(fragments) != 3:
            continue
        degrees = [0, 0, 0]
        for cut in cuts:
            bond = mol.GetBondWithIdx(cut)
            for atom in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                degrees[next(i for i, mapping in enumerate(mappings) if atom in mapping)] += 1
        ends = [i for i, degree in enumerate(degrees) if degree == 1]
        if len(ends) != 2 or any(len(mappings[i]) < min_fragment_atoms for i in ends):
            continue
        # Reject unsupported fragment stereo/valence rather than weaken mapping.
        try:
            for index in ends:
                validate_atom_maps(mol, SideCondition(
                    'training_native_supervision', FRAGMENT_POLICY,
                    fragment=fragments[index], atom_map=tuple(mappings[index])))
        except ValueError:
            continue
        left, right = ends
        assignments = []
        for first, second in ((left, right), (right, left)):
            own = [nearest[0][list(mappings[first])], nearest[1][list(mappings[second])]]
            other = [nearest[1][list(mappings[first])], nearest[0][list(mappings[second])]]
            if any(not (distances <= cutoff).any() for distances in own):
                continue
            preference = sum(float((foreign - local).mean()) for local, foreign in zip(own, other))
            assignments.append((preference, first, second))
        if not assignments:
            continue
        preference, first, second = max(assignments)
        return {
            'fragments': (fragments[first], fragments[second]),
            'atom_maps': (tuple(mappings[first]), tuple(mappings[second])),
            'audit': {'policy': FRAGMENT_POLICY, 'chemical_warhead_identity_claimed': False,
                      'cut_bond_indices': list(cuts), 'trials': trial,
                      'max_trials': max_trials, 'min_fragment_atoms': min_fragment_atoms,
                      'contact_cutoff': cutoff, 'side_preference_angstrom': preference,
                      'middle_atoms': list(mappings[degrees.index(2)])},
        }
    return None

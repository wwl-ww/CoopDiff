"""Exact conserved subfragments of a supplied binary ligand, not relaxed MCS.

Enumerate connected components from non-ring single-bond cuts. Every retained
atom/bond and the original parents' mapped stereocenters must agree. These are
partial analogue-template hypotheses, never claims of a complete warhead match.
No coordinates participate in correspondence or hypothesis selection.
"""

import itertools
import math

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


CORE_POLICY = 'exact_parent_checked_nonring_cut_binary_core_v1'


def _parent_stereo_agrees(source, target, mapping):
    tetra = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    for a, b in mapping.items():
        left, right = source.GetAtomWithIdx(a), target.GetAtomWithIdx(b)
        ls, rs = left.GetChiralTag(), right.GetChiralTag()
        if ls == rs == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        if ls not in tetra or rs not in tetra:
            return False
        ln = [atom.GetIdx() for atom in left.GetNeighbors()]
        rn = [atom.GetIdx() for atom in right.GetNeighbors()]
        if len(ln) != len(rn) or left.GetTotalNumHs() != right.GetTotalNumHs():
            return False
        assigned = {index: mapping[index] for index in ln if index in mapping}
        unknown = [index for index in ln if index not in assigned]
        remaining = [index for index in rn if index not in assigned.values()]
        # A unique external neighbour can preserve tetrahedral parity even
        # when its substituent chemistry changes. More ambiguity is withheld.
        if len(unknown) != len(remaining) or len(unknown) > 1:
            return False
        if unknown:
            assigned[unknown[0]] = remaining[0]
        permutation = [rn.index(assigned[index]) for index in ln]
        odd = sum(permutation[i] > permutation[j] for i in range(len(ln))
                  for j in range(i + 1, len(ln))) % 2
        if (ls == rs) != (odd == 0):
            return False
    return True


def _parent_atoms_agree(source, target, mapping):
    for a, b in mapping.items():
        left, right = source.GetAtomWithIdx(a), target.GetAtomWithIdx(b)
        if any(getattr(left, field)() != getattr(right, field)() for field in
               ('GetAtomicNum', 'GetFormalCharge', 'GetIsAromatic', 'GetHybridization', 'GetIsotope')):
            return False
    for bond in source.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in mapping or b not in mapping:
            continue
        other = target.GetBondBetweenAtoms(mapping[a], mapping[b])
        if (other is None or bond.GetBondType() != other.GetBondType()
                or bond.GetIsAromatic() != other.GetIsAromatic()):
            return False
    reverse = {target_index: source_index for source_index, target_index in mapping.items()}
    for bond in target.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in reverse and b in reverse and source.GetBondBetweenAtoms(reverse[a], reverse[b]) is None:
            return False
    return _parent_stereo_agrees(source, target, mapping)


def _subfragment(source, selected):
    selected = tuple(sorted(selected))
    editable = Chem.RWMol(source)
    keep = set(selected)
    for atom in editable.GetAtoms():
        if atom.GetIdx() in keep and any(n.GetIdx() not in keep for n in atom.GetNeighbors()):
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for index in reversed(range(source.GetNumAtoms())):
        if index not in keep:
            editable.RemoveAtom(index)
    result = editable.GetMol()
    result.RemoveAllConformers()
    Chem.SanitizeMol(result)
    Chem.AssignStereochemistry(result, cleanIt=True, force=True)
    return result


def conserved_binary_cores(full_molecule, binary_fragment, *, min_atoms=8,
                           min_fraction=.3, max_cuts=3, max_combinations=4096, max_maps=128):
    """Return all largest admissible core hypotheses, or an explicit rejection.

    The scope is uncertain analogue transfer. Even an exact conserved chemical
    core does not prove that the analogue binds identically in the new molecule.
    """
    if (type(min_atoms) is not int or min_atoms < 3 or not 0 < min_fraction <= 1
            or type(max_cuts) is not int or max_cuts < 0 or max_combinations < 1 or max_maps < 1):
        raise ValueError('invalid conserved-core enumeration bounds')
    uncharger = rdMolStandardize.Uncharger()
    full, fragment = [uncharger.uncharge(Chem.RemoveHs(Chem.Mol(mol)))
                       for mol in (full_molecule, binary_fragment)]
    full.RemoveAllConformers()
    fragment.RemoveAllConformers()
    if len(Chem.GetMolFrags(full)) != 1 or len(Chem.GetMolFrags(fragment)) != 1:
        raise ValueError('conserved cores require connected parent molecules')
    cuts = [bond.GetIdx() for bond in fragment.GetBonds()
            if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE]
    total = sum(math.comb(len(cuts), size) for size in range(min(max_cuts, len(cuts)) + 1))
    audit = {'policy': CORE_POLICY, 'scope': 'partial analogue-template hypotheses, not complete warheads',
             'coordinates_used_for_mapping': False, 'combinations': total,
             'min_atoms': min_atoms, 'min_fraction': min_fraction, 'max_cuts': max_cuts,
             'source_atom_count': fragment.GetNumAtoms(), 'deployment_eligible': False}
    if total > max_combinations:
        return [], {**audit, 'rejection': 'enumeration_budget_exceeded'}
    minimum = max(min_atoms, math.ceil(fragment.GetNumAtoms() * min_fraction))
    rings = [set(ring) for ring in fragment.GetRingInfo().AtomRings() if len(ring) >= 5]
    subsets = {tuple(range(fragment.GetNumAtoms()))}
    for size in range(1, min(max_cuts, len(cuts)) + 1):
        for selected_cuts in itertools.combinations(cuts, size):
            broken = Chem.FragmentOnBonds(fragment, list(selected_cuts), addDummies=False)
            subsets.update(tuple(sorted(group)) for group in Chem.GetMolFrags(broken)
                           if len(group) >= minimum)
    hypotheses, maximum, truncated = [], 0, False
    for selected in sorted(subsets, key=lambda value: (-len(value), value)):
        if len(selected) < max(minimum, maximum) or not any(ring <= set(selected) for ring in rings):
            continue
        try:
            core = _subfragment(fragment, selected)
        except ValueError:
            continue
        matches = full.GetSubstructMatches(core, useChirality=True, uniquify=False, maxMatches=max_maps + 1)
        if len(matches) > max_maps:
            truncated = True
            continue
        matches = sorted(set(match for match in matches
            if _parent_atoms_agree(fragment, full, dict(zip(selected, match)))))
        if not matches:
            continue
        maximum = len(selected)
        boundary = [bond.GetIdx() for bond in fragment.GetBonds()
                    if (bond.GetBeginAtomIdx() in selected) != (bond.GetEndAtomIdx() in selected)]
        hypotheses.append({'source_atom_indices': selected, 'fragment': core, 'atom_maps': matches,
            'boundary_bonds': boundary, 'source_coverage': len(selected) / fragment.GetNumAtoms(),
            'parent_hybridization_checked': True, 'parent_stereo_checked': True})
    return hypotheses, {**audit, 'hypotheses': len(hypotheses), 'core_atoms': maximum,
                        'some_match_enumerations_truncated': truncated,
                        'rejection': None if hypotheses else 'no_exact_parent_consistent_core'}

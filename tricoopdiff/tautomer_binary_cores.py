"""Explicit tautomer-normalized core hypotheses, separate from exact chemistry.

This does not authorize a different oxidation state or a different scaffold.
An incomplete tautomer enumeration cannot certify a canonical correspondence.
"""

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .conserved_binary_cores import conserved_binary_cores, _parent_stereo_agrees


TAUTOMER_CORE_POLICY = 'complete_canonical_tautomer_parent_stereo_core_v1'


def canonical_tautomer_with_identity(molecule):
    original = Chem.RemoveHs(Chem.Mol(molecule))
    original.RemoveAllConformers()
    neutral = rdMolStandardize.Uncharger().uncharge(original)
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetRemoveSp3Stereo(False)
    enumerator.SetRemoveBondStereo(False)
    enumerator.SetMaxTautomers(1000)
    enumerator.SetMaxTransforms(10000)
    result = enumerator.Enumerate(neutral)
    if result.status != rdMolStandardize.TautomerEnumeratorStatus.Completed:
        raise ValueError(f'incomplete tautomer enumeration: {result.status}')
    canonical = enumerator.PickCanonical(result)
    if canonical.GetNumAtoms() != original.GetNumAtoms():
        raise ValueError('tautomer normalization changed atom count')
    for left, right in zip(original.GetAtoms(), canonical.GetAtoms()):
        if (left.GetAtomicNum() != right.GetAtomicNum()
                or left.GetIsotope() != right.GetIsotope()
                or sorted(n.GetIdx() for n in left.GetNeighbors())
                != sorted(n.GetIdx() for n in right.GetNeighbors())):
            raise ValueError('tautomer normalization changed atom identity or connectivity')
        if left.GetChiralTag() != right.GetChiralTag():
            raise ValueError('tautomer normalization changed stereochemistry')
    return canonical, {'enumerated_tautomers': len(result), 'complete': True,
                       'canonical_smiles': Chem.MolToSmiles(canonical),
                       'atom_order_and_stereocenters_preserved': True}


def tautomer_conserved_binary_cores(full_molecule, binary_fragment, **kwargs):
    """Find partial hypotheses in explicitly normalized parent graphs only."""
    try:
        full, full_audit = canonical_tautomer_with_identity(full_molecule)
        source, source_audit = canonical_tautomer_with_identity(binary_fragment)
    except ValueError as error:
        return [], {'policy': TAUTOMER_CORE_POLICY, 'rejection': str(error),
                    'deployment_eligible': False}
    cores, audit = conserved_binary_cores(full, source, **kwargs)
    original_full = Chem.RemoveHs(Chem.Mol(full_molecule))
    original_source = Chem.RemoveHs(Chem.Mol(binary_fragment))
    retained = []
    for core in cores:
        maps = [mapping for mapping in core['atom_maps']
                if _parent_stereo_agrees(original_source, original_full,
                                         dict(zip(core['source_atom_indices'], mapping)))]
        if maps:
            retained.append({**core, 'atom_maps': maps, 'tautomer_normalization_required': True})
    return retained, {**audit, 'policy': TAUTOMER_CORE_POLICY,
                      'full_normalization': full_audit, 'binary_normalization': source_audit,
                      'hypotheses': len(retained), 'deployment_eligible': False,
                      'coordinates_used_for_mapping': False,
                      'rejection': None if retained else 'no_parent_stereo_consistent_tautomer_core'}

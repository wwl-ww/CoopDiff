"""Strict deploy-time PROTAC condition manifests with no native ternary inputs."""

import json
from pathlib import Path

from rdkit import Chem

from .condition_features import SideCondition, build_condition_features
from .binary_pose_constraints import BinaryPoseSide


MANIFEST_VERSION = 'protac_condition_manifest_v1'
SIDE_FIELDS = {
    'provenance', 'evidence_id', 'confidence', 'fragment_smiles', 'atom_map',
    'atom_maps', 'mapping_method',
    'audit_only',
    'pocket_probabilities', 'binary_fragment_coords', 'binary_protein_coords',
}


def canonical_isomeric_smiles(molecule):
    """Stable chemistry identity used to prevent cross-case condition reuse."""
    return Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True, isomericSmiles=True)


def load_condition_manifest(path, *, case, molecule, protein_sizes):
    """Validate a case-specific manifest and return aligned tensor features.

    Protein pocket arrays must already follow featurization node order. Binary
    geometry, when present, remains in each protein's local frame. This loader
    has no field for native ternary coordinates by construction.
    """
    source = Path(path)
    payload = json.loads(source.read_text(encoding='utf-8'))
    allowed = {'schema', 'case', 'ligand_atom_count', 'ligand_canonical_isomeric_smiles',
               'sides', 'linker_indices', 'protac'}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError('condition manifest has unknown or invalid fields')
    if payload.get('schema') != MANIFEST_VERSION or payload.get('case') != case:
        raise ValueError('condition manifest schema/case mismatch')
    ligand = Chem.RemoveHs(molecule)
    if (payload.get('ligand_atom_count') != ligand.GetNumAtoms()
            or payload.get('ligand_canonical_isomeric_smiles') != canonical_isomeric_smiles(ligand)):
        raise ValueError('condition manifest ligand identity mismatch')
    raw_sides = payload.get('sides')
    if not isinstance(raw_sides, list) or len(raw_sides) != 2:
        raise ValueError('condition manifest requires exactly two side entries')
    sides = []
    for raw in raw_sides:
        if raw is None:
            sides.append(None)
            continue
        if not isinstance(raw, dict) or set(raw) - SIDE_FIELDS:
            raise ValueError('condition manifest side has unknown fields')
        fragment_smiles = raw.get('fragment_smiles')
        fragment = Chem.MolFromSmiles(fragment_smiles) if isinstance(fragment_smiles, str) else None
        if fragment_smiles is not None and fragment is None:
            raise ValueError('condition manifest fragment SMILES is invalid')
        sides.append(SideCondition(
            provenance=raw.get('provenance'), evidence_id=raw.get('evidence_id'),
            confidence=raw.get('confidence', 1.), fragment=fragment,
            atom_map=tuple(raw.get('atom_map', ())),
            atom_maps=tuple(tuple(value) for value in raw.get('atom_maps', ())),
            mapping_method=raw.get('mapping_method', 'exact_stereo'),
            audit_only=raw.get('audit_only', False),
            pocket_probabilities=raw.get('pocket_probabilities'),
            binary_fragment_coords=raw.get('binary_fragment_coords'),
            binary_protein_coords=raw.get('binary_protein_coords'),
        ))
    features = build_condition_features(
        ligand, list(protein_sizes), tuple(sides),
        linker_indices=tuple(payload.get('linker_indices', ())),
        protac=payload.get('protac', True),
    )
    audit = {
        'schema': MANIFEST_VERSION, 'case': case, 'path': str(source.resolve()),
        'ligand_identity_verified': True, 'native_ternary_geometry_read': False,
        'available_sides': [side is not None for side in sides],
    }
    return features, audit


def load_binary_pose_sides(path, *, case, molecule, protein_sizes):
    """Load the raw local binary poses after the normal manifest validation."""
    _, audit = load_condition_manifest(
        path, case=case, molecule=molecule, protein_sizes=protein_sizes)
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    result = []
    for raw in payload['sides']:
        if (raw is None or raw.get('binary_fragment_coords') is None
                or raw.get('binary_protein_coords') is None):
            result.append(None)
            continue
        maps = raw.get('atom_maps')
        if not maps and raw.get('atom_map'):
            maps = [raw['atom_map']]
        result.append(BinaryPoseSide(
            protein_coords=raw['binary_protein_coords'],
            fragment_coords=raw['binary_fragment_coords'],
            atom_maps=tuple(tuple(value) for value in maps or ()),
        ))
    return tuple(result), {**audit, 'binary_pose_sides': [side is not None for side in result]}

"""Explicit PROTAC condition inputs, independent of candidate initialization source.

Protein arrays must already follow the receptor node order. Binary coordinates
are used only within each side's own frame; no cross-side coordinate distances
are exposed. This is a standalone encoder, not enabled in C40 inference.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem

import torch
from torch import nn


PROVENANCE = ('chemical_definition', 'experimental_binary', 'template', 'docked', 'predicted',
              'training_native_supervision')
DEPLOYABLE_PROVENANCE = PROVENANCE[:-1]


def condition_feature_shapes(protein_sizes, ligand_atoms):
    """Versioned unbatched tensor shapes shared by builders and the fuse."""
    if (len(protein_sizes) != 2 or any(type(n) is not int or n < 1 for n in protein_sizes)
            or type(ligand_atoms) is not int or ligand_atoms < 1):
        raise ValueError('condition shapes require two proteins and a nonempty ligand')
    # Provenance is mandatory audit metadata but is deliberately not a learned
    # feature: generic training only observes native-derived supervision and
    # cannot calibrate separate template/docked/predicted embedding columns.
    return {'protein1': (protein_sizes[0], 3), 'protein2': (protein_sizes[1], 3),
            'ligand': (ligand_atoms, 8), 'sides': (2, 4)}


@dataclass(frozen=True)
class SideCondition:
    provenance: str
    evidence_id: str
    confidence: float = 1.
    fragment: Optional[object] = None
    atom_map: tuple = ()  # Fragment atom order -> complete ligand atom order.
    pocket_probabilities: Optional[object] = None  # None means unknown, not all zeros.
    binary_fragment_coords: Optional[object] = None
    binary_protein_coords: Optional[object] = None  # Same ordering as receptor nodes.


def finite_tensor(value, shape, name):
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().clone()
    if tuple(tensor.shape) != tuple(shape) or not torch.isfinite(tensor).all():
        raise ValueError(f'{name}: expected finite shape {shape}')
    return tensor


def validate_atom_map(molecule, side):
    mapping = tuple(side.atom_map)
    if side.fragment is None:
        if mapping:
            raise ValueError('atom map requires an explicit fragment chemical graph')
        return mapping
    if (not mapping or len(mapping) != side.fragment.GetNumAtoms() or len(set(mapping)) != len(mapping)
            or any(type(i) is not int or not 0 <= i < molecule.GetNumAtoms() for i in mapping)):
        raise ValueError('invalid fragment atom map')
    # Explicit mapping may choose a chemically equivalent symmetry, but cannot
    # change connectivity, stereo or use native coordinates to construct a map.
    matches = molecule.GetSubstructMatches(side.fragment, useChirality=True, uniquify=False, maxMatches=4096)
    if mapping not in matches:
        raise ValueError('atom map is not a verified stereo-aware chemical subgraph (or exceeds match limit)')
    return mapping


def build_condition_features(molecule, protein_sizes, sides=(None, None), *, linker_indices=(), protac=True):
    """Build role/mask/local-distance features; unspecified ligand atoms stay unknown.

Roles are multi-hot E3 end / POI end / linker / unknown. Overlapping ends are
allowed only for generic ternary ligands. Linker atoms are explicit design
annotations, never inferred as every atom outside an incomplete fragment.
"""
    if len(protein_sizes) != 2 or any(type(n) is not int or n < 1 for n in protein_sizes) or len(sides) != 2:
        raise ValueError('two nonempty proteins and two optional side conditions required')
    count = molecule.GetNumAtoms()
    if not count:
        raise ValueError('empty ligand')
    ligand = torch.zeros(count, 8)
    proteins = [torch.zeros(n, 3) for n in protein_sizes]
    metadata = torch.zeros(2, 4)
    maps = []
    for index, side in enumerate(sides):
        if side is None:
            maps.append(())
            continue
        if side.provenance not in DEPLOYABLE_PROVENANCE or not isinstance(side.evidence_id, str) or not side.evidence_id.strip():
            raise ValueError('conditions require explicit allowed provenance and evidence ID')
        if not 0 <= side.confidence <= 1:
            raise ValueError('condition confidence must be finite in [0,1]')
        mapping = validate_atom_map(molecule, side)
        maps.append(mapping)
        if mapping:
            ligand[list(mapping), index] = 1.
            metadata[index, 0] = 1.
        if side.pocket_probabilities is not None:
            if side.provenance == 'chemical_definition':
                raise ValueError('chemical definition alone cannot establish a protein pocket')
            pocket = finite_tensor(side.pocket_probabilities, (protein_sizes[index],), 'pocket')
            if (pocket < 0).any() or (pocket > 1).any():
                raise ValueError('pocket probabilities must be in [0,1]')
            proteins[index][:, 0] = pocket
            proteins[index][:, 1] = 1.  # Known negative pockets differ from absent pockets.
            proteins[index][:, 2] = side.confidence
            metadata[index, 1] = 1.
        geometry = (side.binary_fragment_coords is not None, side.binary_protein_coords is not None)
        if any(geometry):
            if not all(geometry) or not mapping or side.provenance == 'chemical_definition':
                raise ValueError('binary geometry requires a mapped fragment, its protein frame and pose provenance')
            fragment = finite_tensor(side.binary_fragment_coords, (len(mapping), 3), 'binary fragment')
            protein = finite_tensor(side.binary_protein_coords, (protein_sizes[index], 3), 'binary protein')
            nearest = torch.cdist(fragment, protein).min(1).values
            ligand[list(mapping), 4 + index] = torch.log1p(nearest)
            ligand[list(mapping), 6 + index] = 1.
            metadata[index, 2] = 1.
        if not metadata[index, :3].any():
            raise ValueError('empty condition must be represented by None')
        metadata[index, 3] = side.confidence
    if protac and set(maps[0]) & set(maps[1]):
        raise ValueError('overlapping PROTAC ends; use generic ternary mode if intended')
    linker = tuple(linker_indices)
    if (len(set(linker)) != len(linker) or any(type(i) is not int or not 0 <= i < count for i in linker)
            or set(linker) & (set(maps[0]) | set(maps[1]))):
        raise ValueError('invalid or overlapping linker annotation')
    if linker:
        ligand[list(linker), 2] = 1.
    ligand[:, 3] = (ligand[:, :3].sum(1) == 0).float()
    return {'ligand': ligand, 'protein1': proteins[0], 'protein2': proteins[1], 'sides': metadata}


def build_supervised_contact_features(molecule, protein_coords, ligand_coords, cutoff=6.):
    """Build generic ternary pocket masks from native training coordinates.

    This mirrors the discrete ligand/protein pocket masks exposed by
    DeepTernary's PROTAC path. Coordinates are consumed only to create binary
    supervision and are not returned. Generic ligands may contact both sides.
    """
    if not isinstance(cutoff, (int, float)) or not 0 < cutoff < 20:
        raise ValueError('training pocket cutoff must be finite and in (0,20) Å')
    proteins = [torch.as_tensor(value, dtype=torch.float32).detach().cpu()
                for value in protein_coords]
    ligand = torch.as_tensor(ligand_coords, dtype=torch.float32).detach().cpu()
    if (len(proteins) != 2 or ligand.ndim != 2 or ligand.shape[1] != 3
            or ligand.shape[0] != molecule.GetNumAtoms() or not torch.isfinite(ligand).all()
            or any(p.ndim != 2 or p.shape[1] != 3 or not len(p) or not torch.isfinite(p).all()
                   for p in proteins)):
        raise ValueError('native condition coordinates do not match the chemical graph/proteins')
    shapes = condition_feature_shapes([len(p) for p in proteins], len(ligand))
    result = {name: torch.zeros(shape) for name, shape in shapes.items()}
    for index, protein in enumerate(proteins):
        distances = torch.cdist(ligand, protein)
        contacts = distances <= float(cutoff)
        result['ligand'][:, index] = contacts.any(1).float()
        nearest = distances.min(1).values
        role = contacts.any(1)
        result['ligand'][role, 4 + index] = torch.log1p(nearest[role])
        result['ligand'][role, 6 + index] = 1.
        result[f'protein{index + 1}'][:, 0] = contacts.any(0).float()
        result[f'protein{index + 1}'][:, 1:] = 1.  # known mask + confidence
        # Roles and pockets are supervised including known negatives; local
        # geometry exists only when a mapped/contacting end was identified.
        result['sides'][index, 0:2] = 1.
        result['sides'][index, 2] = float(role.any())
        result['sides'][index, 3] = 1.
    result['ligand'][:, 3] = (result['ligand'][:, :3].sum(1) == 0).float()
    return result


class ProtacConditionEncoder(nn.Module):
    """Optional residual node/side embeddings; absent conditions are exactly zero."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ligand_encoder = nn.Sequential(nn.Linear(8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.protein_encoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.side_encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.SiLU(),
                                          nn.Linear(hidden_dim, hidden_dim))

    def forward(self, features):
        ligand, sides = features['ligand'], features['sides']
        result = {'ligand': self.ligand_encoder(ligand) * (ligand[:, :3].sum(-1, keepdim=True) > 0),
                  'sides': self.side_encoder(sides) * (sides[:, :3].sum(-1, keepdim=True) > 0)}
        for name in ('protein1', 'protein2'):
            protein = features[name]
            result[name] = self.protein_encoder(protein) * protein[:, 1:2]
        return result


MANIFEST_VERSION = 'protac_condition_manifest_v1'
SIDE_FIELDS = {
    'provenance', 'evidence_id', 'confidence', 'fragment_smiles', 'atom_map',
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

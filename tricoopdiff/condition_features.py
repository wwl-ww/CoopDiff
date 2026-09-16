"""Explicit PROTAC condition inputs, independent of candidate initialization source.

Protein arrays must already follow the receptor node order. Binary coordinates
are used only within each side's own frame; no cross-side coordinate distances
are exposed. This is a standalone encoder, not enabled in C40 inference.
"""

from dataclasses import dataclass
from typing import Optional

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
    return {'protein1': (protein_sizes[0], 3), 'protein2': (protein_sizes[1], 3),
            'ligand': (ligand_atoms, 8), 'sides': (2, 4 + len(PROVENANCE))}


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
    metadata = torch.zeros(2, 4 + len(PROVENANCE))
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
        metadata[index, 4 + PROVENANCE.index(side.provenance)] = 1.
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
        result['sides'][index, 4 + PROVENANCE.index('training_native_supervision')] = 1.
    result['ligand'][:, 3] = (result['ligand'][:, :3].sum(1) == 0).float()
    return result


class ProtacConditionEncoder(nn.Module):
    """Optional residual node/side embeddings; absent conditions are exactly zero."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ligand_encoder = nn.Sequential(nn.Linear(8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.protein_encoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.side_encoder = nn.Sequential(nn.Linear(4 + len(PROVENANCE), hidden_dim), nn.SiLU(),
                                          nn.Linear(hidden_dim, hidden_dim))

    def forward(self, features):
        ligand, sides = features['ligand'], features['sides']
        result = {'ligand': self.ligand_encoder(ligand) * (ligand[:, :3].sum(-1, keepdim=True) > 0),
                  'sides': self.side_encoder(sides) * (sides[:, :3].sum(-1, keepdim=True) > 0)}
        for name in ('protein1', 'protein2'):
            protein = features[name]
            result[name] = self.protein_encoder(protein) * protein[:, 1:2]
        return result

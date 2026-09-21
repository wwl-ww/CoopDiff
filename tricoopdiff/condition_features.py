"""Explicit PROTAC condition inputs, independent of candidate initialization source.

Protein arrays must already follow the receptor node order. Binary coordinates
are used only within each side's own frame; no cross-side coordinate distances
are exposed. The conditioned model also consumes receptor-indexed distance
alternatives; the original C40 architecture does not accept these features.
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
    # Provenance is mandatory audit metadata but is deliberately not a learned
    # feature: generic training only observes native-derived supervision and
    # cannot calibrate separate template/docked/predicted embedding columns.
    return {'protein1': (protein_sizes[0], 6), 'protein2': (protein_sizes[1], 6),
            'ligand': (ligand_atoms, 14), 'sides': (2, 5),
            'binary1': (1, ligand_atoms, protein_sizes[0], 3),
            'binary2': (1, ligand_atoms, protein_sizes[1], 3)}


@dataclass(frozen=True)
class SideCondition:
    provenance: str
    evidence_id: str
    confidence: float = 1.
    fragment: Optional[object] = None
    atom_map: tuple = ()  # Fragment atom order -> complete ligand atom order.
    atom_maps: tuple = ()  # All symmetry-equivalent maps; preferred for geometry.
    mapping_method: str = 'exact_stereo'
    audit_only: bool = False
    pocket_probabilities: Optional[object] = None  # None means unknown, not all zeros.
    binary_fragment_coords: Optional[object] = None
    binary_protein_coords: Optional[object] = None  # Same ordering as receptor nodes.


def finite_tensor(value, shape, name):
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().clone()
    if tuple(tensor.shape) != tuple(shape) or not torch.isfinite(tensor).all():
        raise ValueError(f'{name}: expected finite shape {shape}')
    return tensor


def validate_atom_maps(molecule, side):
    mapping = tuple(side.atom_map)
    mappings = tuple(tuple(value) for value in side.atom_maps) if side.atom_maps else ((mapping,) if mapping else ())
    if side.fragment is None:
        if mappings:
            raise ValueError('atom map requires an explicit fragment chemical graph')
        return mappings
    if (not mappings or any(len(value) != side.fragment.GetNumAtoms()
                            or len(set(value)) != len(value)
                            or any(type(i) is not int or not 0 <= i < molecule.GetNumAtoms()
                                   for i in value) for value in mappings)):
        raise ValueError('invalid fragment atom map')
    # Explicit mapping may choose a chemically equivalent symmetry, but cannot
    # change connectivity, stereo or use native coordinates to construct a map.
    reference, fragment = molecule, side.fragment
    if side.mapping_method == 'charge_normalized_exact_stereo':
        from rdkit.Chem.MolStandardize import rdMolStandardize
        uncharger = rdMolStandardize.Uncharger()
        reference, fragment = uncharger.uncharge(molecule), uncharger.uncharge(side.fragment)
    elif side.mapping_method == 'conservative_charge_normalized_mcs':
        if not side.audit_only or side.confidence > .5:
            raise ValueError('conservative MCS mappings require an explicit low-confidence audit side')
        from rdkit.Chem.MolStandardize import rdMolStandardize
        uncharger = rdMolStandardize.Uncharger()
        reference, fragment = uncharger.uncharge(molecule), uncharger.uncharge(side.fragment)
        # MCS evidence deliberately does not claim stereo or exact aromatic
        # bond typing. Still verify atom identity and every mapped adjacency;
        # the low-confidence audit gate above prevents deployment use.
        for mapping in mappings:
            if any(fragment.GetAtomWithIdx(index).GetAtomicNum()
                   != reference.GetAtomWithIdx(mapping[index]).GetAtomicNum()
                   for index in range(fragment.GetNumAtoms())):
                raise ValueError('conservative MCS atom identity mismatch')
            if any(reference.GetBondBetweenAtoms(
                    mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]) is None
                   for bond in fragment.GetBonds()):
                raise ValueError('conservative MCS connectivity mismatch')
        return mappings
    elif side.mapping_method != 'exact_stereo':
        raise ValueError('unsupported deploy-time atom mapping method')
    matches = reference.GetSubstructMatches(fragment, useChirality=True, uniquify=False,
                                            maxMatches=4096)
    if any(value not in matches for value in mappings):
        raise ValueError('atom map is not a verified stereo-aware chemical subgraph (or exceeds match limit)')
    return mappings


def build_condition_features(molecule, protein_sizes, sides=(None, None), *, linker_indices=(), protac=True):
    """Build deployment features; native-supervised conditions are forbidden."""
    return _build_condition_features(molecule, protein_sizes, sides,
                                     linker_indices=linker_indices, protac=protac,
                                     allowed_provenance=DEPLOYABLE_PROVENANCE)


def build_training_condition_features(molecule, protein_sizes, sides):
    """Explicit training-only entry using the same geometry encoding as deployment."""
    if any(side is None or side.provenance != 'training_native_supervision' for side in sides):
        raise ValueError('training-only builder requires two native-supervised sides')
    return _build_condition_features(molecule, protein_sizes, sides,
                                     allowed_provenance=('training_native_supervision',))


def _build_condition_features(molecule, protein_sizes, sides=(None, None), *,
                              linker_indices=(), protac=True, allowed_provenance):
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
    ligand = torch.zeros(count, 14)
    proteins = [torch.zeros(n, 6) for n in protein_sizes]
    metadata = torch.zeros(2, 5)
    maps = []
    binary = [torch.zeros(1, count, n, 3) for n in protein_sizes]
    for index, side in enumerate(sides):
        if side is None:
            maps.append(())
            continue
        if side.provenance not in allowed_provenance or not isinstance(side.evidence_id, str) or not side.evidence_id.strip():
            raise ValueError('conditions require explicit allowed provenance and evidence ID')
        if not 0 <= side.confidence <= 1:
            raise ValueError('condition confidence must be finite in [0,1]')
        mappings = validate_atom_maps(molecule, side)
        role_atoms = tuple(sorted({atom for mapping in mappings for atom in mapping}))
        maps.append(role_atoms)
        if role_atoms:
            ligand[list(role_atoms), index] = 1.
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
            if not all(geometry) or not mappings or side.provenance == 'chemical_definition':
                raise ValueError('binary geometry requires a mapped fragment, its protein frame and pose provenance')
            fragment = finite_tensor(side.binary_fragment_coords, (side.fragment.GetNumAtoms(), 3),
                                     'binary fragment')
            protein = finite_tensor(side.binary_protein_coords, (protein_sizes[index], 3), 'binary protein')
            distances = torch.cdist(fragment, protein)
            # Keep complete assignments together. Independent per-pair ranges
            # could admit a hybrid that corresponds to no legal chemical map.
            unique_maps = sorted(set(mappings))
            binary[index] = torch.zeros(len(unique_maps), count, protein_sizes[index], 3)
            for alternative, mapping in enumerate(unique_maps):
                binary[index][alternative, list(mapping), :, 0] = distances
                binary[index][alternative, list(mapping), :, 1] = distances
                binary[index][alternative, list(mapping), :, 2] = side.confidence
            # A small radial fingerprint retains substantially more of the
            # known binary pose than one nearest-neighbor scalar while staying
            # invariant to an independent rigid transform of either side.
            ligand_columns = 4 + 5 * index
            radial = [torch.log1p(distances.min(1).values)]
            for offset, neighbors in enumerate((4, 8), start=1):
                k = min(neighbors, distances.shape[1])
                radial.append(torch.log1p(
                    distances.topk(k, dim=1, largest=False).values.mean(1)))
            radial = torch.stack(radial, dim=1)
            accumulated = ligand.new_zeros((count, 3))
            occurrences = ligand.new_zeros((count, 1))
            for mapping in mappings:
                accumulated[list(mapping)] += radial
                occurrences[list(mapping)] += 1
            present = occurrences[:, 0] > 0
            ligand[present, ligand_columns:ligand_columns + 3] = (
                accumulated[present] / occurrences[present])
            ligand[present, ligand_columns + 3] = 1.
            ligand[present, ligand_columns + 4] = side.confidence
            protein_distances = distances.transpose(0, 1)
            proteins[index][:, 3] = torch.log1p(protein_distances.min(1).values)
            k = min(4, protein_distances.shape[1])
            proteins[index][:, 4] = torch.log1p(
                protein_distances.topk(k, dim=1, largest=False).values.mean(1))
            proteins[index][:, 5] = 1.
            metadata[index, 2] = 1.
        if not metadata[index, :3].any():
            raise ValueError('empty condition must be represented by None')
        metadata[index, 3] = side.confidence
        metadata[index, 4] = len(role_atoms) / count
    if protac and set(maps[0]) & set(maps[1]):
        raise ValueError('overlapping PROTAC ends; use generic ternary mode if intended')
    linker = tuple(linker_indices)
    if (len(set(linker)) != len(linker) or any(type(i) is not int or not 0 <= i < count for i in linker)
            or set(linker) & (set(maps[0]) | set(maps[1]))):
        raise ValueError('invalid or overlapping linker annotation')
    if linker:
        ligand[list(linker), 2] = 1.
    ligand[:, 3] = (ligand[:, :3].sum(1) == 0).float()
    return {'ligand': ligand, 'protein1': proteins[0], 'protein2': proteins[1],
            'sides': metadata, 'binary1': binary[0], 'binary2': binary[1]}


def _dilate_ligand_mask(molecule, mask, radius):
    """Expand a contact mask by a fixed number of covalent graph edges."""
    result = torch.as_tensor(mask, dtype=torch.bool).clone()
    frontier = result.clone()
    for _ in range(radius):
        expanded = result.clone()
        for bond in molecule.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if frontier[left]:
                expanded[right] = True
            if frontier[right]:
                expanded[left] = True
        frontier = expanded & ~result
        result = expanded
    return result


def build_supervised_contact_features(molecule, protein_coords, ligand_coords, cutoff=6.,
                                      role_graph_radius=0, disjoint_roles=False,
                                      role_margin=1.):
    """Build generic ternary pocket masks from native training coordinates.

    This mirrors the discrete ligand/protein pocket masks exposed by
    DeepTernary's PROTAC path. Coordinates are consumed only to create binary
    supervision and protein-local distances. Absolute coordinates are not
    returned. Generic ligands may contact both sides.
    """
    if (not isinstance(cutoff, (int, float)) or not 0 < cutoff < 20
            or type(role_graph_radius) is not int or role_graph_radius < 0
            or not isinstance(role_margin, (int, float)) or role_margin < 0):
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
    distances_by_side = [torch.cdist(ligand, protein) for protein in proteins]
    roles = []
    if disjoint_roles:
        nearest = [distance.min(1).values for distance in distances_by_side]
        roles = [((nearest[0] <= float(cutoff))
                  & (nearest[0] + float(role_margin) < nearest[1])),
                 ((nearest[1] <= float(cutoff))
                  & (nearest[1] + float(role_margin) < nearest[0]))]
        roles = [_dilate_ligand_mask(molecule, role, role_graph_radius) for role in roles]
        # Dilation can meet in the linker. Ambiguous atoms stay linker rather
        # than becoming impossible overlapping PROTAC ends.
        overlap = roles[0] & roles[1]
        roles = [role & ~overlap for role in roles]
    for index, protein in enumerate(proteins):
        distances = distances_by_side[index]
        all_contacts = distances <= float(cutoff)
        nearest = distances.min(1).values
        role = roles[index] if disjoint_roles else _dilate_ligand_mask(
            molecule, all_contacts.any(1), role_graph_radius)
        contacts = all_contacts & role[:, None]
        if disjoint_roles and not role.any():
            continue
        binary = result[f'binary{index + 1}']
        binary[0, role, :, 0] = distances[role]
        binary[0, role, :, 1] = distances[role]
        binary[0, role, :, 2] = 1.
        result['ligand'][:, index] = role.float()
        columns = 4 + 5 * index
        result['ligand'][role, columns] = torch.log1p(nearest[role])
        for offset, neighbors in enumerate((4, 8), start=1):
            k = min(neighbors, distances.shape[1])
            result['ligand'][role, columns + offset] = torch.log1p(
                distances[role].topk(k, dim=1, largest=False).values.mean(1))
        result['ligand'][role, columns + 3:columns + 5] = 1.
        result[f'protein{index + 1}'][:, 0] = contacts.any(0).float()
        result[f'protein{index + 1}'][:, 1:3] = 1.  # known mask + confidence
        if role.any():
            # At deployment this profile is computed against the mapped binary
            # fragment, so native supervision uses only this side's contact-role
            # atoms rather than leaking the other end or linker into the profile.
            reciprocal = distances[role].transpose(0, 1)
            result[f'protein{index + 1}'][:, 3] = torch.log1p(reciprocal.min(1).values)
            k = min(4, reciprocal.shape[1])
            result[f'protein{index + 1}'][:, 4] = torch.log1p(
                reciprocal.topk(k, dim=1, largest=False).values.mean(1))
            result[f'protein{index + 1}'][:, 5] = 1.
        # Roles and pockets are supervised including known negatives; local
        # geometry exists only when a mapped/contacting end was identified.
        result['sides'][index, 0:2] = 1.
        result['sides'][index, 2] = float(role.any())
        result['sides'][index, 3] = 1.
        result['sides'][index, 4] = role.float().mean()
    if disjoint_roles:
        result['ligand'][:, 2] = (~roles[0] & ~roles[1]).float()
    result['ligand'][:, 3] = (result['ligand'][:, :3].sum(1) == 0).float()
    return result


class ProtacConditionEncoder(nn.Module):
    """Optional residual node/side embeddings; absent conditions are exactly zero."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ligand_encoder = nn.Sequential(nn.Linear(14, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.protein_encoder = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.side_encoder = nn.Sequential(nn.Linear(5, hidden_dim), nn.SiLU(),
                                          nn.Linear(hidden_dim, hidden_dim))

    def forward(self, features):
        ligand, sides = features['ligand'], features['sides']
        result = {'ligand': self.ligand_encoder(ligand) * (ligand[:, :3].sum(-1, keepdim=True) > 0),
                  'sides': self.side_encoder(sides) * (sides[:, :3].sum(-1, keepdim=True) > 0)}
        for name in ('protein1', 'protein2'):
            protein = features[name]
            availability = torch.maximum(protein[:, 1:2], protein[:, 5:6])
            result[name] = self.protein_encoder(protein) * availability
        return result

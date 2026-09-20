"""Native-free assembly from two protein-local binary PROTAC poses."""

from dataclasses import dataclass

import torch

from .pose_anchors import proper_rigid_fit
from .rigid import apply_rigid


@dataclass(frozen=True)
class BinaryPoseSide:
    """One mapped warhead pose in its own receptor coordinate frame."""

    protein_coords: object
    fragment_coords: object
    atom_maps: tuple


def _coords(value, name, *, dtype=torch.float64, device=None):
    result = torch.as_tensor(value, dtype=dtype, device=device)
    if result.ndim != 2 or result.shape[1] != 3 or not len(result) or not torch.isfinite(result).all():
        raise ValueError(f'{name} must be finite nonempty [N,3] coordinates')
    return result


def _side(value, ligand_atoms, name, *, dtype, device):
    protein = _coords(value.protein_coords, f'{name}.protein', dtype=dtype, device=device)
    fragment = _coords(value.fragment_coords, f'{name}.fragment', dtype=dtype, device=device)
    mappings = tuple(tuple(mapping) for mapping in value.atom_maps)
    if (not mappings or any(len(mapping) != len(fragment)
                            or len(set(mapping)) != len(mapping)
                            or any(type(index) is not int or not 0 <= index < ligand_atoms
                                   for index in mapping) for mapping in mappings)):
        raise ValueError(f'{name}: invalid atom-map ensemble')
    return protein, fragment, mappings


def assemble_binary_pose_candidates(ligand_conformer, current_e3, side1, side2,
                                    *, max_candidates=32):
    """Place ligand and POI while preserving both supplied binary complexes.

    Side 1 is first brought into the current E3 receptor frame.  Each legal
    chemical symmetry map then aligns the ligand conformer to that warhead.
    Side 2's receptor is placed by aligning its local warhead onto the placed
    ligand.  No ternary reference or cross-protein distance is consumed.
    """
    reference = torch.as_tensor(ligand_conformer)
    dtype = torch.float64
    ligand = _coords(ligand_conformer, 'ligand conformer', dtype=dtype,
                     device=reference.device)
    e3 = _coords(current_e3, 'current E3', dtype=dtype, device=reference.device)
    protein1, fragment1, maps1 = _side(
        side1, len(ligand), 'side1', dtype=dtype, device=reference.device)
    protein2, fragment2, maps2 = _side(
        side2, len(ligand), 'side2', dtype=dtype, device=reference.device)
    if protein1.shape != e3.shape:
        raise ValueError('binary side1 protein must follow current E3 node order')
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError('max_candidates must be a positive integer')

    e3_rotation, e3_translation, e3_rmsd = proper_rigid_fit(protein1, e3)
    target1 = apply_rigid(fragment1, e3_rotation, e3_translation)
    rows = []
    for map1_index, mapping1 in enumerate(maps1):
        ligand_rotation, ligand_translation, side1_rmsd = proper_rigid_fit(
            ligand[list(mapping1)], target1)
        placed_ligand = apply_rigid(ligand, ligand_rotation, ligand_translation)
        for map2_index, mapping2 in enumerate(maps2):
            poi_rotation, poi_translation, side2_rmsd = proper_rigid_fit(
                fragment2, placed_ligand[list(mapping2)])
            placed_poi = apply_rigid(protein2, poi_rotation, poi_translation)
            rows.append({
                'ligand_coords': placed_ligand.to(torch.float32),
                'poi_coords': placed_poi.to(torch.float32),
                'poi_rotation': poi_rotation.to(torch.float32),
                'map1_index': map1_index, 'map2_index': map2_index,
                'e3_frame_rmsd': float(e3_rmsd),
                'side1_fit_rmsd': float(side1_rmsd),
                'side2_fit_rmsd': float(side2_rmsd),
                'fit_rmsd_sum': float(side1_rmsd + side2_rmsd),
            })
    rows.sort(key=lambda row: (row['fit_rmsd_sum'], row['map1_index'], row['map2_index']))
    return rows[:max_candidates]


def embed_with_fixed_binary_ends(molecule, side1, side2, maps, seed):
    """Generate ETKDG coordinates, rigidly replace both ends, then relax linker."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if type(seed) is not int or seed < 0:
        raise ValueError('invalid constrained conformer seed')
    mol = Chem.Mol(molecule)
    mol.RemoveAllConformers()
    heavy_atoms = mol.GetNumAtoms()
    embedded = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed % (2**31 - 1))
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(embedded, params) < 0:
        raise ValueError('unconstrained ETKDG embedding failed')
    conformer = embedded.GetConformer()
    fixed = set()
    for side, mapping in zip((side1, side2), maps):
        fragment = _coords(side.fragment_coords, 'binary fragment')
        if len(mapping) != len(fragment):
            raise ValueError('binary fragment/map size mismatch')
        current = torch.as_tensor(conformer.GetPositions(), dtype=torch.float64)[list(mapping)]
        rotation, translation, _ = proper_rigid_fit(fragment, current)
        target = apply_rigid(fragment, rotation, translation).cpu().numpy()
        for fragment_index, ligand_index in enumerate(mapping):
            conformer.SetAtomPosition(ligand_index, target[fragment_index])
            fixed.add(ligand_index)
    properties = AllChem.MMFFGetMoleculeProperties(embedded)
    forcefield = (AllChem.MMFFGetMoleculeForceField(embedded, properties)
                  if properties is not None else AllChem.UFFGetMoleculeForceField(embedded))
    if forcefield is None:
        raise ValueError('no force field supports fixed-end linker relaxation')
    for index in fixed:
        forcefield.AddFixedPoint(index)
    forcefield.Initialize()
    forcefield.Minimize(maxIts=200)
    coords = conformer.GetPositions()[:heavy_atoms]
    return torch.as_tensor(coords, dtype=torch.float32)


class BinaryPoseInitializer:
    """Initialize from ETKDG while exactly retaining two supplied binary poses."""

    def __init__(self, side1, side2, *, no_torsion=False, max_candidates=24):
        if side1 is None or side2 is None:
            raise ValueError('binary-pose initialization requires both sides')
        self.side1, self.side2 = side1, side2
        self.no_torsion = bool(no_torsion)
        self.max_candidates = max_candidates

    @torch.no_grad()
    def initialize(self, e3_coords, poi_coords, ligand_data, seed=None):
        if seed is None:
            raise ValueError('binary-pose initialization requires an explicit seed')
        from rdkit import Chem
        from .initialization import validate_ligand_graph

        molecule = ligand_data.get('mol')
        molecule = Chem.Mol(molecule) if isinstance(molecule, bytes) else Chem.Mol(molecule)
        validate_ligand_graph(molecule, ligand_data['edge_index'], ligand_data['edge_mask'],
                              ligand_data['mask_rotate'])
        map_pairs = [(left, right) for left in self.side1.atom_maps
                     for right in self.side2.atom_maps]
        candidates = []
        failures = []
        # Chemical symmetry maps and linker conformers are independent sources
        # of ambiguity.  Cycle over every legal map pair while drawing multiple
        # seeded ETKDG conformers, rather than producing only one conformer when
        # each end has a unique mapping.
        for index in range(self.max_candidates):
            pair = map_pairs[index % len(map_pairs)]
            try:
                conformer = embed_with_fixed_binary_ends(
                    molecule, self.side1, self.side2, pair, seed + index)
                left = BinaryPoseSide(self.side1.protein_coords, self.side1.fragment_coords,
                                      (pair[0],))
                right = BinaryPoseSide(self.side2.protein_coords, self.side2.fragment_coords,
                                       (pair[1],))
                candidate = assemble_binary_pose_candidates(
                    conformer.to(e3_coords.device), e3_coords, left, right,
                    max_candidates=1)[0]
                candidate['map_pair_index'] = index
                candidates.append(candidate)
            except ValueError as error:
                failures.append({'map_pair_index': index, 'error': str(error)})
        if not candidates:
            raise ValueError(f'all two-end constrained embeddings failed: {failures}')
        binary_poi = _coords(self.side2.protein_coords, 'side2.protein',
                             dtype=poi_coords.dtype, device=poi_coords.device)
        if binary_poi.shape != poi_coords.shape:
            raise ValueError('binary side2 protein must follow current POI node order')
        _, _, poi_input_rmsd = proper_rigid_fit(binary_poi, poi_coords)
        if float(poi_input_rmsd) > 1e-3:
            raise ValueError('binary side2 protein coordinates differ from current POI input')

        e3 = e3_coords.to(candidates[0]['ligand_coords'])
        scored = []
        for candidate in candidates:
            ligand, poi = candidate['ligand_coords'], candidate['poi_coords']
            protein_distances = torch.cdist(e3, poi)
            e3_ligand = torch.cdist(e3, ligand)
            poi_ligand = torch.cdist(poi, ligand)
            clashes = int((protein_distances < 3.).sum())
            e3_ligand_clashes = int((e3_ligand < 2.5).sum())
            poi_ligand_clashes = int((poi_ligand < 2.5).sum())
            interface_contacts = int((protein_distances < 8.).sum())
            geometry = {
                'protein_ca_pairs_below_3A': clashes,
                'protein_ca_pairs_below_8A': interface_contacts,
                'e3_ca_ligand_pairs_below_2_5A': e3_ligand_clashes,
                'poi_ca_ligand_pairs_below_2_5A': poi_ligand_clashes,
                'minimum_protein_ca_distance': float(protein_distances.min()),
                'minimum_e3_ca_ligand_distance': float(e3_ligand.min()),
                'minimum_poi_ca_ligand_distance': float(poi_ligand.min()),
            }
            severe_clashes = clashes + e3_ligand_clashes + poi_ligand_clashes
            penalty = (severe_clashes, clashes, -interface_contacts,
                       abs(geometry['minimum_e3_ca_ligand_distance'] - 3.5)
                       + abs(geometry['minimum_poi_ca_ligand_distance'] - 3.5),
                       candidate['fit_rmsd_sum'], candidate['map1_index'],
                       candidate['map2_index'])
            scored.append((penalty, candidate, geometry))
        _, selected, geometry = min(scored, key=lambda value: value[0])
        metadata = {
            'init_mode': 'binary_pose',
            'version': 'known_binary_pose_initializer_v1',
            'seed': seed, 'conformer_method': 'ETKDGv3_rigid_ends_fixed_linker_relaxation',
            'no_torsion': self.no_torsion,
            'native_information_used': False,
            'binary_pose_knowledge': True,
            'candidate_count': len(candidates),
            'embedding_failures': failures,
            'selected_map_pair_index': selected['map_pair_index'],
            'selected_maps': [selected['map1_index'], selected['map2_index']],
            'side_fit_rmsd': [selected['side1_fit_rmsd'], selected['side2_fit_rmsd']],
            'poi_input_rmsd': float(poi_input_rmsd), 'geometry': geometry,
        }
        return (selected['ligand_coords'].unsqueeze(0),
                selected['poi_coords'].unsqueeze(0),
                selected['poi_rotation'].unsqueeze(0), metadata)

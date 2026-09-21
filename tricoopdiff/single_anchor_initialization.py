"""Native-free embedding with exactly one supplied binary-pose constraint."""

import math

import torch

from .anchored_flow import AnchoredTorsionFlow, audit_bond_geometry
from .binary_pose_constraints import BinaryPoseSide
from .initialization import receptor_prior, validate_ligand_graph
from .pose_anchors import proper_rigid_fit
from .rigid import apply_rigid
from .single_anchor_flow import SingleAnchorTorsionFlow, compile_single_anchor_torsions


def _embed_one_side(molecule, side, mapping, seed):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.Mol(molecule)
    mol.RemoveAllConformers()
    count = mol.GetNumAtoms()
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed % (2 ** 31 - 1)
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol, params) < 0:
        raise ValueError('single-anchor ETKDG embedding failed')
    conf = mol.GetConformer()
    current = torch.as_tensor(conf.GetPositions(), dtype=torch.float64)[list(mapping)]
    fragment = torch.as_tensor(side.fragment_coords, dtype=torch.float64, device='cpu')
    rotation, translation, _ = proper_rigid_fit(fragment, current)
    placed = apply_rigid(fragment, rotation, translation).numpy()
    for row, atom in enumerate(mapping):
        conf.SetAtomPosition(atom, placed[row])
    properties = AllChem.MMFFGetMoleculeProperties(mol)
    forcefield = (AllChem.MMFFGetMoleculeForceField(mol, properties) if properties is not None
                  else AllChem.UFFGetMoleculeForceField(mol))
    if forcefield is None:
        raise ValueError('no force field supports the single fixed binary core')
    for atom in mapping:
        forcefield.AddFixedPoint(atom)
    forcefield.Initialize()
    forcefield.Minimize(maxIts=200)
    return torch.as_tensor(conf.GetPositions()[:count], dtype=torch.float64)


@torch.no_grad()
def initialize_single_anchor(side, e3, poi, ligand_data, *, seed, max_candidates=24,
                              max_fit_rmsd=.15, translation_bound=.5, rotation_bound=.2):
    """Select the first chemically valid mapped conformer; POI is not scored/fitted."""
    from rdkit import Chem

    if (not isinstance(side, BinaryPoseSide) or type(seed) is not int or seed < 0
            or type(max_candidates) is not int or max_candidates < 1
            or not math.isfinite(max_fit_rmsd) or max_fit_rmsd <= 0):
        raise ValueError('single-anchor initialization requires one pose and explicit valid budget/seed')
    mol = Chem.Mol(ligand_data['mol'])
    validate_ligand_graph(mol, ligand_data['edge_index'], ligand_data['edge_mask'], ligand_data['mask_rotate'])
    protein = torch.as_tensor(side.protein_coords, dtype=torch.float64, device=e3.device)
    fragment = torch.as_tensor(side.fragment_coords, dtype=torch.float64, device=e3.device)
    rotation, translation, error = proper_rigid_fit(protein, e3.double())
    if error > 1e-3:
        raise ValueError('known-side receptor differs from input E3 geometry')
    target = apply_rigid(fragment, rotation, translation)
    maps = tuple(tuple(mapping) for mapping in side.atom_maps)
    if (not maps or any(len(m) != len(fragment) or len(set(m)) != len(m)
                        or any(type(i) is not int or not 0 <= i < mol.GetNumAtoms() for i in m) for m in maps)):
        raise ValueError('invalid single-anchor maps')
    failures = []
    for index in range(max_candidates):
        mapping = maps[index % len(maps)]
        try:
            coords = _embed_one_side(mol, side, mapping, seed + index).to(e3.device)
            r, t, fit = proper_rigid_fit(coords[list(mapping)], target)
            if fit > max_fit_rmsd:
                raise ValueError('single binary fit exceeds declared tolerance')
            coords = apply_rigid(coords, r, t).to(e3)
            bonds = audit_bond_geometry(mol, coords)
            plan = compile_single_anchor_torsions(mol, ligand_data['edge_index'], ligand_data['edge_mask'], mapping)
            strict = AnchoredTorsionFlow(coords, poi, torch.eye(3, dtype=e3.dtype, device=e3.device), plan)
            center, _, _ = receptor_prior(e3)
            _, _, scale = receptor_prior(poi)
            flow = SingleAnchorTorsionFlow(strict, mapping, center, scale,
                                           translation_bound=translation_bound, rotation_bound=rotation_bound)
            return flow, {'init_mode': 'single_binary_pose', 'seed': seed,
                          'policy': 'first_chemically_valid_single_binary_embedding_v1',
                          'native_information_used': False, 'conditioned_sides': [True, False],
                          'selected_atom_maps': [list(mapping), None], 'candidate_index': index,
                          'candidate_budget': max_candidates, 'failures': failures,
                          'known_side_fit_rmsd': float(fit), 'bond_geometry': bonds,
                          'generative_process': flow.version}
        except ValueError as error:
            failures.append({'candidate': index, 'error': str(error)})
    raise ValueError(f'no valid single-binary embedding: {failures}')

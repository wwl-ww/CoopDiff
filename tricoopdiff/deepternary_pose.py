"""Initialization from an explicitly supplied DeepTernary prediction."""

from pathlib import Path

import torch

from .pose_anchors import proper_rigid_fit
from .rigid import apply_rigid


_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'MSE',
}


def _pdb_coords(path, record, chain=None, atom_name=None):
    rows, seen_residues = [], set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.startswith(record):
            continue
        if chain is not None and line[21] != chain:
            continue
        if atom_name is not None and line[12:16].strip() != atom_name:
            continue
        if record == 'ATOM' and line[17:20].strip() not in _AMINO_ACIDS:
            continue
        if record == 'ATOM' and atom_name == 'CA':
            residue = (line[21], line[22:26], line[26])
            if residue in seen_residues:
                continue
            seen_residues.add(residue)
        rows.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    value = torch.tensor(rows, dtype=torch.float32)
    if value.ndim != 2 or value.shape[1:] != (3,) or not len(value):
        raise ValueError(f'no matching coordinates in {path}')
    return value


class DeepTernaryPoseInitializer:
    """Rigidly move a DeepTernary pose into the current E3 reference frame."""

    def __init__(self, complex_pdb, *, no_torsion=False):
        self.path = Path(complex_pdb).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.e3 = _pdb_coords(self.path, 'ATOM', chain='A', atom_name='CA')
        self.poi = _pdb_coords(self.path, 'ATOM', chain='B', atom_name='CA')
        self.ligand = _pdb_coords(self.path, 'HETATM')
        self.no_torsion = bool(no_torsion)

    @torch.no_grad()
    def initialize(self, e3_coords, poi_coords, ligand_data, seed=None):
        ligand_atoms = int(ligand_data['coords'].shape[1])
        if self.e3.shape != e3_coords.shape or self.poi.shape != poi_coords.shape:
            raise ValueError('DeepTernary protein C-alpha order does not match model inputs')
        if len(self.ligand) != ligand_atoms:
            raise ValueError('DeepTernary ligand atom order does not match model input')
        device, dtype = e3_coords.device, e3_coords.dtype
        rotation, translation, e3_rmsd = proper_rigid_fit(
            self.e3.to(device=device, dtype=dtype), e3_coords)
        ligand = apply_rigid(self.ligand.to(device=device, dtype=dtype),
                             rotation, translation)
        poi = apply_rigid(self.poi.to(device=device, dtype=dtype), rotation, translation)
        poi_rotation, _, poi_fit = proper_rigid_fit(poi_coords, poi)
        distances = torch.cdist(e3_coords, poi)
        metadata = {
            'init_mode': 'deepternary_pose',
            'version': 'deepternary_pose_initializer_v1',
            'prediction_path': str(self.path),
            'native_information_used': False,
            'deepternary_prediction_used': True,
            'e3_frame_rmsd': float(e3_rmsd),
            'poi_rigid_fit_rmsd': float(poi_fit),
            'geometry': {
                'protein_ca_pairs_below_3A': int((distances < 3.).sum()),
                'protein_ca_pairs_below_8A': int((distances < 8.).sum()),
                'minimum_protein_ca_distance': float(distances.min()),
            },
        }
        return ligand.unsqueeze(0), poi.unsqueeze(0), poi_rotation.unsqueeze(0), metadata

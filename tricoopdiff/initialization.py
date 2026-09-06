"""Shared full-ligand, receptor-size-adaptive dual binary initialization."""

import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .rigid import apply_rigid, assemble_interface_pair


INITIAL_NOISE_STD_PROPORTION = 1.4601642460337794
INITIALIZATION_VERSION = "dual_native_v1"


def receptor_prior(coords):
    """Use all receptor Cα atoms, centered on their own arithmetic mean."""
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords):
        raise ValueError("receptor must contain [N, 3] coordinates")
    if not torch.isfinite(coords).all():
        raise ValueError("non-finite receptor coordinates")
    center = coords.mean(0)
    radius = (coords - center).square().sum(-1).mean().sqrt()
    if radius <= 0:
        raise ValueError("receptor RMS radius must be positive")
    return center, radius, radius * INITIAL_NOISE_STD_PROPORTION / 1.73


def validate_ligand_graph(mol, edge_index, edge_mask, mask_rotate):
    """Require exact indexed full-molecule bonds and directed torsion masks."""
    from rdkit import Chem

    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("dual_native requires one connected complete RDKit molecule")
    n = mol.GetNumAtoms()
    edges = torch.as_tensor(edge_index).detach().cpu()
    selected = torch.as_tensor(edge_mask).detach().cpu()
    if edges.ndim != 2 or edges.shape[0] != 2 or selected.dtype != torch.bool:
        raise ValueError("invalid ligand edge_index or edge_mask")
    if selected.shape != (edges.shape[1],) or edges.numel() == 0:
        raise ValueError("invalid full-ligand bond map")
    if edges.min() < 0 or edges.max() >= n:
        raise ValueError("ligand atom map out of range")
    expected = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()}
    expected |= {(v, u) for u, v in expected.copy()}
    if set(map(tuple, edges.T.tolist())) != expected:
        raise ValueError("ligand graph does not match indexed chemical graph")
    if isinstance(mask_rotate, (list, tuple)):
        if len(mask_rotate) != 1:
            raise ValueError("expected one full-ligand torsion map")
        mask_rotate = mask_rotate[0]
    masks = torch.as_tensor(mask_rotate, dtype=torch.bool).cpu()
    bonds = edges.T[selected]
    if masks.shape != (len(bonds), n):
        raise ValueError("torsion mask shape/order does not match canonical bonds")
    seen = set()
    for (u, v), mask in zip(bonds.tolist(), masks):
        bond_id = tuple(sorted((u, v)))
        if bond_id in seen or bool(mask[u]) or not bool(mask[v]):
            raise ValueError("duplicate/reversed torsion bond: rotate side must contain v, not u")
        seen.add(bond_id)
        # Removing the axis bond must split the graph into exactly these sides.
        component, pending = {v}, [v]
        while pending:
            atom = pending.pop()
            for neighbor in mol.GetAtomWithIdx(atom).GetNeighbors():
                j = neighbor.GetIdx()
                if {atom, j} == {u, v}:
                    continue
                if j not in component:
                    component.add(j)
                    pending.append(j)
        if component != set(torch.where(mask)[0].tolist()) or u in component:
            raise ValueError("torsion rotation side is not the canonical bond component")
    return bonds, masks.numpy()


class TernaryInitializer:
    """Only dual_native is supported; old shell/fixed-Å/local priors are removed."""

    def __init__(self, initial_noise_std_proportion=INITIAL_NOISE_STD_PROPORTION,
                 no_torsion=False):
        if initial_noise_std_proportion != INITIAL_NOISE_STD_PROPORTION:
            raise ValueError("dual_native requires the fixed positive DiffDock proportion")
        self.no_torsion = bool(no_torsion)

    def initialize(self, e3_coords, poi_coords, ligand_data, seed=None):
        """Return B=1 ligand, POI, inverse body frame and JSON-safe provenance.

        Fresh ETKDG geometry is generated from the indexed chemical graph, never
        from the native coordinates. No atom matching or Kabsch inference occurs.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from .diffdock.utils.torsion import modify_conformer_torsion_angles

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        mol = ligand_data.get("mol")
        if isinstance(mol, bytes):
            mol = Chem.Mol(mol)
        bonds, masks = validate_ligand_graph(
            mol, ligand_data["edge_index"], ligand_data["edge_mask"],
            ligand_data["mask_rotate"],
        )
        mol = Chem.Mol(mol)
        mol.RemoveAllConformers()
        conformer_seed = int(np.random.randint(0, 2**31 - 1))
        embedded = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = conformer_seed
        if AllChem.EmbedMolecule(embedded, params) != 0:
            raise ValueError("full-ligand ETKDG embedding failed; no native-pose fallback")
        # AddHs appends atoms; retain the original graph's exact atom order.
        shared = torch.as_tensor(
            embedded.GetConformer().GetPositions()[:mol.GetNumAtoms()],
            dtype=e3_coords.dtype, device=e3_coords.device,
        )
        torsions = np.zeros(len(bonds))
        if not self.no_torsion and len(bonds):
            torsions = np.random.uniform(-np.pi, np.pi, len(bonds))
            shared = modify_conformer_torsion_angles(
                shared.cpu(), bonds, masks, torsions, as_numpy=False,
            ).to(device=e3_coords.device, dtype=e3_coords.dtype)
        ligand_center = shared.mean(0)
        transforms, sides = [], {}
        for name, receptor in (("e3", e3_coords), ("poi", poi_coords)):
            center, radius, sigma = receptor_prior(receptor)
            rotation = torch.as_tensor(Rotation.random().as_matrix(),
                                       device=shared.device, dtype=shared.dtype)
            delta = torch.randn(3, device=shared.device, dtype=shared.dtype) * sigma
            translation = center + delta - rotation @ ligand_center
            transforms.append((rotation, translation))
            sides[name] = {
                "receptor_center": center.tolist(), "rms_radius": radius.item(),
                "sigma_init": sigma.item(), "rotation": rotation.tolist(),
                "delta": delta.tolist(), "translation": translation.tolist(),
                "center_definition": "all_receptor_ca_mean",
            }
        (ra, ta), (rb, tb) = transforms
        ligand = apply_rigid(shared, ra, ta)
        poi, body_rotation = assemble_interface_pair(poi_coords, ra, ta, rb, tb)
        metadata = {
            "init_mode": "dual_native", "version": INITIALIZATION_VERSION,
            "initial_noise_std_proportion": INITIAL_NOISE_STD_PROPORTION,
            "no_torsion": self.no_torsion, "pocket_knowledge": False,
            "choose_residue": False, "collision_filter": False,
            "rng_seeds": dict(python=seed, numpy=seed, scipy=seed, pytorch=seed,
                              rdkit=conformer_seed),
            "torsion_updates": torsions.tolist(),
            "torsion_bonds": bonds.tolist(), "sides": sides,
        }
        return ligand.unsqueeze(0), poi.unsqueeze(0), body_rotation.unsqueeze(0), metadata

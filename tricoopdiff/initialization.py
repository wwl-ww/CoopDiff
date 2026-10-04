"""SDPI: Shared-Ligand Dual-Pose Initialization with receptor-adaptive priors."""

import platform
import random

import numpy as np
import scipy
import torch
from scipy.spatial.transform import Rotation

from tricoopdiff.pose_anchors import apply_rigid, assemble_interface_pair


INITIAL_NOISE_STD_PROPORTION = 1.4601642460337794
INITIALIZATION_VERSION = "sdpi_v1"


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


def sample_binary_pose(shared, receptor):
    """Sample and retain one Haar rotation and receptor-adaptive translation."""
    center, radius, sigma = receptor_prior(receptor)
    rotation = torch.as_tensor(Rotation.random().as_matrix(),
                               device=shared.device, dtype=shared.dtype)
    delta = torch.randn(3, device=shared.device, dtype=shared.dtype) * sigma
    translation = center + delta - rotation @ shared.mean(0)
    return rotation, translation, {
        "receptor_center": center.tolist(), "rms_radius": radius.item(),
        "sigma_init": sigma.item(), "rotation": rotation.tolist(),
        "delta": delta.tolist(), "translation": translation.tolist(),
        "center_definition": "all_receptor_ca_mean",
    }


def validate_ligand_graph(mol, edge_index, edge_mask, mask_rotate):
    """Require exact indexed full-molecule bonds and directed torsion masks."""
    from rdkit import Chem

    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("sdpi requires one connected complete RDKit molecule")
    n = mol.GetNumAtoms()
    edges = torch.as_tensor(edge_index).detach().cpu()
    selected = torch.as_tensor(edge_mask).detach().cpu()
    if (edges.ndim != 2 or edges.shape[0] != 2 or selected.dtype != torch.bool
            or edges.dtype not in (torch.int32, torch.int64)):
        raise ValueError("invalid ligand edge_index or edge_mask")
    if selected.shape != (edges.shape[1],) or edges.numel() == 0:
        raise ValueError("invalid full-ligand bond map")
    if edges.min() < 0 or edges.max() >= n:
        raise ValueError("ligand atom map out of range")
    expected = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()}
    expected |= {(v, u) for u, v in expected.copy()}
    if set(map(tuple, edges.T.tolist())) != expected or edges.shape[1] != len(expected):
        raise ValueError("ligand graph does not match indexed chemical graph")
    if isinstance(mask_rotate, (list, tuple)):
        if len(mask_rotate) != 1:
            raise ValueError("expected one full-ligand torsion map")
        mask_rotate = mask_rotate[0]
    masks = torch.as_tensor(mask_rotate).cpu()
    if masks.dtype != torch.bool:
        raise ValueError("torsion mask_rotate must be boolean")
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
    """Only sdpi is supported; old shell/fixed-Å/local priors are removed."""

    def __init__(self, initial_noise_std_proportion=INITIAL_NOISE_STD_PROPORTION,
                 no_torsion=False, conformer_policy="etkdgv3_random"):
        if initial_noise_std_proportion != INITIAL_NOISE_STD_PROPORTION:
            raise ValueError("sdpi requires the fixed positive DiffDock proportion")
        self.no_torsion = bool(no_torsion)
        if conformer_policy not in ("etkdgv3_random", "diffdock_v1_1"):
            raise ValueError("unknown conformer policy")
        self.conformer_policy = conformer_policy

    def initialize(self, e3_coords, poi_coords, ligand_data, seed=None):
        """Return B=1 ligand, POI, inverse body frame and JSON-safe provenance.

        Fresh ETKDG geometry is generated from the indexed chemical graph, never
        from the native coordinates. No atom matching or Kabsch inference occurs.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from tricoopdiff.diffdock.utils.torsion import modify_conformer_torsion_angles

        if seed is None:
            seed = random.SystemRandom().randrange(2**32)
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
        for field in ("coords", "features"):
            if field in ligand_data and ligand_data[field].shape[-2] != mol.GetNumAtoms():
                raise ValueError(f"full-ligand atom count disagrees with {field}")
        mol = Chem.Mol(mol)
        mol.RemoveAllConformers()
        conformer_seed = int(np.random.randint(0, 2**31 - 1))
        embedded = Chem.AddHs(mol)
        params = (AllChem.ETKDGv2() if self.conformer_policy == "diffdock_v1_1"
                  else AllChem.ETKDGv3())
        params.randomSeed = conformer_seed
        used_random_coords = self.conformer_policy == "etkdgv3_random"
        params.useRandomCoords = used_random_coords
        conformer_attempts = 0
        embedded_id = -1
        max_attempts = 3 if self.conformer_policy == "diffdock_v1_1" else 1
        while conformer_attempts < max_attempts and embedded_id == -1:
            embedded_id = AllChem.EmbedMolecule(embedded, params)
            conformer_attempts += 1
        # DiffDock v1.1's generate_conformer() reaches this branch only after
        # three ordinary ETKDGv2 failures, then optimizes the random embedding.
        if embedded_id == -1 and self.conformer_policy == "diffdock_v1_1":
            params.useRandomCoords = True
            used_random_coords = True
            embedded_id = AllChem.EmbedMolecule(embedded, params)
            if embedded_id != -1:
                AllChem.MMFFOptimizeMolecule(embedded, confId=embedded_id)
        if embedded_id == -1:
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
        transforms, sides = [], {}
        for name, receptor in (("e3", e3_coords), ("poi", poi_coords)):
            rotation, translation, sides[name] = sample_binary_pose(shared, receptor)
            transforms.append((rotation, translation))
        (ra, ta), (rb, tb) = transforms
        ligand = apply_rigid(shared, ra, ta)
        poi, body_rotation = assemble_interface_pair(poi_coords, ra, ta, rb, tb)
        metadata = {
            "init_mode": "sdpi", "version": INITIALIZATION_VERSION,
            "initial_noise_std_proportion": INITIAL_NOISE_STD_PROPORTION,
            "no_torsion": self.no_torsion, "pocket_knowledge": False,
            "choose_residue": False, "collision_filter": False,
            "rng_seeds": dict(python=seed, numpy=seed, scipy=seed, pytorch=seed,
                              rdkit=conformer_seed),
            "torsion_updates": torsions.tolist(),
            "torsion_bonds": bonds.tolist(), "sides": sides,
            "conformer": {
                "policy": self.conformer_policy,
                "method": ("ETKDGv2" if self.conformer_policy == "diffdock_v1_1"
                           else "ETKDGv3"),
                "useRandomCoords": used_random_coords,
                "attempts_before_fallback": conformer_attempts,
                "enforceChirality": True,
            },
            "software": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "pytorch": torch.__version__,
                         "rdkit": Chem.rdBase.rdkitVersion},
            "device": str(shared.device), "dtype": str(shared.dtype),
        }
        return ligand.unsqueeze(0), poi.unsqueeze(0), body_rotation.unsqueeze(0), metadata

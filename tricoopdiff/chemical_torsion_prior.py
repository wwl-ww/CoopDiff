"""Native-free conformer torsions transferred into a binary-constrained chart.

Only active dihedrals are transferred. Bond lengths, angles, ring geometry and
binary fragments remain those of the chart. This is a chemical proposal prior,
not a guarantee that transferred conformers are low energy or clash free.
"""

import math
import hashlib

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from .anchored_flow import audit_bond_geometry, wrap_angles


CHEMICAL_PRIOR_POLICY = 'etkdgv3_mmff_relaxed_active_dihedral_transfer_v2'


def ordered_graph_sha256(molecule):
    """Identity includes atom indices and stereochemistry, never coordinates."""
    mol = Chem.Mol(molecule)
    mol.RemoveAllConformers()
    for index, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(index + 1)
    graph = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return hashlib.sha256(graph.encode()).hexdigest()


def independent_conformers(molecule, *, seed, count=32, max_iterations=500):
    """Embed from graph only; never retain input conformers or coordinate maps.

    Atom order is unchanged. Finite, bond-valid relaxed conformers are proposals,
    not necessarily energy minima. Record convergence separately; never fall
    back to input coordinates when embedding or force-field setup fails.
    """
    if (type(seed) is not int or not 0 <= seed < 2**31
            or type(count) is not int or count < 1
            or type(max_iterations) is not int or max_iterations < 1):
        raise ValueError('invalid chemical prior seed or conformer budget')
    mol = Chem.Mol(molecule)
    if not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('chemical prior requires a connected molecule')
    atom_count = mol.GetNumAtoms()
    mol.RemoveAllConformers()
    mol = Chem.AddHs(mol)
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise ValueError('chemical prior requires complete MMFF parameters')
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.clearConfs = True
    params.pruneRmsThresh = -1.
    identifiers = list(AllChem.EmbedMultipleConfs(mol, numConfs=count, params=params))
    if not identifiers:
        raise ValueError('chemical prior ETKDG embedding failed')
    optimized = AllChem.MMFFOptimizeMoleculeConfs(
        mol, numThreads=1, maxIters=max_iterations, mmffVariant='MMFF94s')
    coordinates, energies, failed, convergence = [], [], [], []
    for identifier, (status, energy) in zip(identifiers, optimized):
        xyz = torch.tensor(mol.GetConformer(identifier).GetPositions()[:atom_count], dtype=torch.float64)
        if status not in (0, 1) or not math.isfinite(energy) or not torch.isfinite(xyz).all():
            failed.append({'conformer': identifier, 'optimization_status': status})
            continue
        try:
            audit_bond_geometry(molecule, xyz)
        except ValueError as error:
            failed.append({'conformer': identifier, 'reason': str(error)})
            continue
        coordinates.append(xyz)
        energies.append(float(energy))
        convergence.append(status == 0)
    if not coordinates:
        raise ValueError('chemical prior has no finite bond-valid relaxed conformer')
    return torch.stack(coordinates), {
        'policy': CHEMICAL_PRIOR_POLICY, 'seed': seed, 'requested': count,
        'embedded': len(identifiers), 'accepted': len(coordinates),
        'rejected': failed, 'mmff94s_energies': energies,
        'mmff_converged': convergence, 'mmff_max_iterations': max_iterations,
        'rdkit_version': Chem.rdBase.rdkitVersion,
        'input_coordinates_used': False,
        'selection': 'uniform_over_bond_valid_relaxed_conformers_not_energy_ranked',
    }


def transfer_dihedrals(reference, plan, edge_index, conformers):
    """Return [conformer, active torsion] increments in the chart convention.

    This is intentionally NOT target encoding: arbitrary conformers need not
    belong to the chart. Do not use it to silently project training labels.
    """
    if (reference.shape != (plan.atom_count, 3) or not torch.isfinite(reference).all()
            or conformers.ndim != 3 or conformers.shape[1:] != reference.shape
            or not conformers.shape[0] or not torch.isfinite(conformers).all()):
        raise ValueError('chemical prior coordinate shape or finiteness mismatch')
    if (edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long
            or (edge_index < 0).any() or (edge_index >= plan.atom_count).any()):
        raise ValueError('chemical prior requires valid bonded edges')
    neighbors = [set() for _ in range(plan.atom_count)]
    for u, v in edge_index.T.tolist():
        neighbors[u].add(v)
        neighbors[v].add(u)
    coords = torch.cat((reference[None].double(), conformers.to(reference.device).double()))

    def angle(a, u, v, b):
        axis = coords[:, u] - coords[:, v]
        length = axis.norm(dim=-1, keepdim=True)
        if (length < 1e-8).any():
            raise ValueError('degenerate chemical prior bond')
        axis = axis / length
        left, right = coords[:, a] - coords[:, u], coords[:, b] - coords[:, v]
        left = left - (left * axis).sum(-1, keepdim=True) * axis
        right = right - (right * axis).sum(-1, keepdim=True) * axis
        if (left.norm(dim=-1) < 1e-8).any() or (right.norm(dim=-1) < 1e-8).any():
            raise ValueError('collinear chemical prior dihedral')
        return torch.atan2((torch.cross(left, right, dim=-1) * axis).sum(-1), (left * right).sum(-1))

    transferred = []
    for (u, v), moving in zip(plan.bonds, plan.moving_atoms):
        left = sorted(neighbors[u] - {v} - set(moving))
        right = sorted((neighbors[v] - {u}) & set(moving))
        if not left or not right:
            raise ValueError('chemical prior torsion needs bonded reference atoms')
        values = angle(left[0], u, v, right[0])
        transferred.append(wrap_angles(values[1:] - values[0]))
    return (torch.stack(transferred, dim=-1).to(reference) if transferred
            else reference.new_zeros((len(conformers), 0)))


def sample_transferred_prior(flow, edge_index, conformers, *, generator):
    """Replace only prior torsions, retaining the flow's declared other priors."""
    strict = getattr(flow, 'strict', flow)
    proposals = transfer_dihedrals(strict.ligand, flow.plan, edge_index, conformers)
    index = int(torch.randint(len(proposals), (), device=strict.ligand.device, generator=generator))
    state = flow.sample_prior(generator=generator)
    state[:len(flow.plan.bonds)] = proposals[index]
    return state, {'policy': CHEMICAL_PRIOR_POLICY, 'conformer_index': index,
                   'conformer_count': len(proposals), 'transferred_torsions': len(flow.plan.bonds)}

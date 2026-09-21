"""Rebuild cached ligand chemistry without changing coordinates or receptor features."""

import torch
from rdkit import Chem

from . import diffdock_compat  # noqa: F401


def rebuild_ligand_graph(molecule, cached_ligand):
    """Return an unbatched ligand graph after checking atom order and translation.

    The caller must validate the cache against its original source files before
    reuse. The corrected molecule must retain deposited heavy-atom order/coords.
    No graph tensors from the old chemistry survive, except checked coordinates.
    """
    from torch_geometric.data import HeteroData
    from .diffdock.datasets.process_mols import get_lig_graph_with_matching

    molecule = Chem.Mol(molecule)
    original = Chem.Mol(cached_ligand['mol'])
    if (molecule.GetNumConformers() != 1 or molecule.GetNumAtoms() != original.GetNumAtoms()
            or any(a.GetAtomicNum() != b.GetAtomicNum()
                   for a, b in zip(molecule.GetAtoms(), original.GetAtoms()))):
        raise ValueError('corrected molecule changed heavy-atom count/order/elements')
    positions = torch.tensor(molecule.GetConformer().GetPositions(), dtype=torch.float64)
    cached = cached_ligand['coords'].detach().cpu()
    if cached.shape != positions.shape or not torch.isfinite(cached).all():
        raise ValueError('chemistry migration expects a finite unbatched ligand cache')
    offsets = positions - cached.double()
    error = float((offsets - offsets.mean(0)).norm(dim=-1).max())
    if error > 1e-4:
        raise ValueError(f'corrected molecule moved/reordered deposited coordinates: {error}')
    graph = HeteroData()
    get_lig_graph_with_matching(molecule, graph, popsize=None, maxiter=None,
                               matching=False, keep_original=False,
                               num_conformers=1, remove_hs=True)
    if not torch.allclose(graph['ligand'].pos.double(), positions, atol=1e-4, rtol=0):
        raise ValueError('graph construction changed corrected coordinates or atom order')
    result = {'mol': molecule.ToBinary(), 'coords': cached.clone(),
              'features': graph['ligand'].x,
              'edge_index': graph['ligand', 'ligand'].edge_index,
              'edge_attr': graph['ligand', 'ligand'].edge_attr,
              'edge_mask': graph['ligand'].edge_mask,
              'mask_rotate': graph['ligand'].mask_rotate}
    return result, {'coordinate_max_error_after_translation': error,
                    'coordinates_changed': False, 'ligand_graph_rebuilt': True,
                    'old_smiles': Chem.MolToSmiles(original),
                    'new_smiles': Chem.MolToSmiles(molecule),
                    'old_torsions': int(cached_ligand['edge_mask'].sum()),
                    'new_torsions': int(result['edge_mask'].sum())}

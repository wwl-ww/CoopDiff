"""``crop_beyond`` extracted from DiffDock's ``utils/utils.py``.

Kept standalone so the vendored sampling loop does not need to import
``utils/utils.py``, whose top-level imports (RDKit, torch_geometric DataParallel,
AAModel / old-* models) are irrelevant to the frozen-expert path.
"""
import torch
from torch_geometric.utils import subgraph


def crop_beyond(complex_graph, cutoff, all_atoms):
    ligand_pos = complex_graph['ligand'].pos
    receptor_pos = complex_graph['receptor'].pos
    residues_to_keep = torch.any(torch.sum((ligand_pos.unsqueeze(0) - receptor_pos.unsqueeze(1)) ** 2, -1) < cutoff ** 2, dim=1)

    if all_atoms:
        #print(complex_graph['atom'].x.shape, complex_graph['atom'].pos.shape, complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index.shape)
        atom_to_res_mapping = complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index[1]
        atoms_to_keep = residues_to_keep[atom_to_res_mapping]
        rec_remapper = (torch.cumsum(residues_to_keep.long(), dim=0) - 1)
        atom_to_res_new_mapping = rec_remapper[atom_to_res_mapping][atoms_to_keep]
        atom_res_edge_index = torch.stack([torch.arange(len(atom_to_res_new_mapping), device=atom_to_res_new_mapping.device), atom_to_res_new_mapping])

    complex_graph['receptor'].pos = complex_graph['receptor'].pos[residues_to_keep]
    complex_graph['receptor'].x = complex_graph['receptor'].x[residues_to_keep]
    complex_graph['receptor'].side_chain_vecs = complex_graph['receptor'].side_chain_vecs[residues_to_keep]
    complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = \
        subgraph(residues_to_keep, complex_graph['receptor', 'rec_contact', 'receptor'].edge_index, relabel_nodes=True)[0]

    if all_atoms:
        complex_graph['atom'].x = complex_graph['atom'].x[atoms_to_keep]
        complex_graph['atom'].pos = complex_graph['atom'].pos[atoms_to_keep]
        complex_graph['atom', 'atom_contact', 'atom'].edge_index = subgraph(atoms_to_keep, complex_graph['atom', 'atom_contact', 'atom'].edge_index, relabel_nodes=True)[0]
        complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index = atom_res_edge_index

    #print("cropped", 1-torch.mean(residues_to_keep.float()), 'residues', 1-torch.mean(atoms_to_keep.float()), 'atoms')

"""
Subgraph construction for TriCoopDiff.

Two physically disconnected subgraphs (E3-L and POI-L) are constructed to match
DiffDock's pretraining input distribution exactly (zero OOD).
A third full ternary graph connects all three molecules for TriBrain coordination.
"""

import torch
from torch_geometric.data import HeteroData


def build_subgraph_e3l(
    e3_coords: torch.Tensor,
    e3_features: torch.Tensor,
    ligand_coords: torch.Tensor,
    ligand_features: torch.Tensor,
    rec_edge_index: torch.Tensor = None,
    lig_edge_index: torch.Tensor = None,
    lig_edge_attr: torch.Tensor = None,
    lig_edge_mask: torch.Tensor = None,
    lig_mask_rotate: torch.Tensor = None,
) -> HeteroData:
    """
    Build E3-L two-body subgraph.

    Nodes: E3 residues + PROTAC atoms ONLY. No POI nodes.
    Edges: E3 internal, PROTAC internal, E3-PROTAC cross edges.
    This matches DiffDock's standard protein-ligand docking input format exactly.

    Args:
        e3_coords: [N_e3, 3] E3 ligase residue coordinates (Cα positions)
        e3_features: [N_e3, D_rec] E3 residue features
        ligand_coords: [N_lig, 3] PROTAC atom coordinates
        ligand_features: [N_lig, D_lig] PROTAC atom features
        rec_edge_index: [2, E_rec] precomputed receptor internal edges
        lig_edge_index: [2, E_lig] ligand internal edges
        lig_edge_attr: [E_lig, D_edge] ligand edge attributes
        lig_edge_mask: [E_lig] bool mask for rotatable bond edges
        lig_mask_rotate: list of torsional update masks

    Returns:
        HeteroData graph compatible with DiffDock's CGModel forward pass.
    """
    graph = HeteroData()

    graph['receptor'].pos = e3_coords
    graph['receptor'].x = e3_features

    graph['ligand'].pos = ligand_coords
    graph['ligand'].x = ligand_features

    # Single-complex batching: DiffDock's CGModel indexes `data['ligand'].batch`
    # and `data.num_graphs` in the score heads, so these MUST be set. Batch must
    # live on the same device as the coords (tor_sigma[ligand.batch] indexing and
    # the radius() fallback both reject cross-device index tensors on CUDA).
    graph['receptor'].batch = torch.zeros(e3_coords.shape[0], dtype=torch.long, device=e3_coords.device)
    graph['ligand'].batch = torch.zeros(ligand_coords.shape[0], dtype=torch.long, device=ligand_coords.device)
    graph.num_graphs = 1

    if rec_edge_index is not None:
        graph['receptor', 'receptor'].edge_index = rec_edge_index

    if lig_edge_index is not None:
        graph['ligand', 'ligand'].edge_index = lig_edge_index
    if lig_edge_attr is not None:
        graph['ligand', 'ligand'].edge_attr = lig_edge_attr
    if lig_edge_mask is not None:
        graph['ligand'].edge_mask = lig_edge_mask
    if lig_mask_rotate is not None:
        graph['ligand'].mask_rotate = lig_mask_rotate

    # NOTE: Cross edges (receptor-ligand) are built dynamically by CGModel.build_cross_conv_graph()
    # at each forward pass based on current coordinates and cross_max_distance cutoff.
    # We do NOT precompute them here.

    return graph


def build_subgraph_pol(
    poi_coords: torch.Tensor,
    poi_features: torch.Tensor,
    ligand_coords_poi_frame: torch.Tensor,
    ligand_features: torch.Tensor,
    rec_edge_index: torch.Tensor = None,
    lig_edge_index: torch.Tensor = None,
    lig_edge_attr: torch.Tensor = None,
    lig_edge_mask: torch.Tensor = None,
    lig_mask_rotate: torch.Tensor = None,
) -> HeteroData:
    """
    Build POI-L two-body subgraph. Symmetric to E3-L subgraph.

    The ligand coordinates here must be expressed in the POI reference frame
    (transformed from the global E3 frame via the current T_E3→POI estimate).

    Args:
        poi_coords: [N_poi, 3] POI residue coordinates
        poi_features: [N_poi, D_rec] POI residue features
        ligand_coords_poi_frame: [N_lig, 3] PROTAC coords transformed to POI frame
        ligand_features: [N_lig, D_lig] PROTAC atom features
        (remaining args same as build_subgraph_e3l)

    Returns:
        HeteroData graph for POI-L subgraph.
    """
    graph = HeteroData()

    graph['receptor'].pos = poi_coords
    graph['receptor'].x = poi_features

    graph['ligand'].pos = ligand_coords_poi_frame
    graph['ligand'].x = ligand_features

    # Single-complex batching (see build_subgraph_e3l). Batch on the coords'
    # device to avoid cross-device indexing on CUDA.
    graph['receptor'].batch = torch.zeros(poi_coords.shape[0], dtype=torch.long, device=poi_coords.device)
    graph['ligand'].batch = torch.zeros(ligand_coords_poi_frame.shape[0], dtype=torch.long, device=ligand_coords_poi_frame.device)
    graph.num_graphs = 1

    if rec_edge_index is not None:
        graph['receptor', 'receptor'].edge_index = rec_edge_index

    if lig_edge_index is not None:
        graph['ligand', 'ligand'].edge_index = lig_edge_index
    if lig_edge_attr is not None:
        graph['ligand', 'ligand'].edge_attr = lig_edge_attr
    if lig_edge_mask is not None:
        graph['ligand'].edge_mask = lig_edge_mask
    if lig_mask_rotate is not None:
        graph['ligand'].mask_rotate = lig_mask_rotate

    return graph


def build_tribrain_graph(
    e3_coords: torch.Tensor,
    e3_features: torch.Tensor,
    poi_coords: torch.Tensor,
    poi_features: torch.Tensor,
    ligand_coords: torch.Tensor,
    ligand_features: torch.Tensor,
) -> dict:
    """
    Build the full ternary graph for TriBrain processing.

    Unlike the two subgraphs, this graph contains ALL nodes and ALL edge types:
    E3 residues, POI residues, PROTAC atoms, plus all pairwise edges including
    direct E3-POI protein-protein edges. This is where TriBrain learns three-body
    coordination.

    Returns a flat dictionary (not HeteroData) for the custom TriBrain architecture.

    Args:
        e3_coords: [N_e3, 3]
        e3_features: [N_e3, D_e3]
        poi_coords: [N_poi, 3]
        poi_features: [N_poi, D_poi]
        ligand_coords: [N_lig, 3]
        ligand_features: [N_lig, D_lig]

    Returns:
        dict with keys: node_coords, node_features, node_types, batch
    """
    n_e3 = e3_coords.shape[0]
    n_poi = poi_coords.shape[0]
    n_lig = ligand_coords.shape[0]

    # Concatenate all nodes
    node_coords = torch.cat([e3_coords, poi_coords, ligand_coords], dim=0)

    # For features, we need to handle potentially different feature dimensions.
    # Pad to max dim with zeros and store a mask.
    max_dim = max(e3_features.shape[-1], poi_features.shape[-1], ligand_features.shape[-1])

    def pad_features(feat, target_dim):
        if feat.shape[-1] < target_dim:
            return torch.cat([
                feat,
                torch.zeros(*feat.shape[:-1], target_dim - feat.shape[-1], device=feat.device, dtype=feat.dtype)
            ], dim=-1)
        return feat

    e3_feat_padded = pad_features(e3_features, max_dim)
    poi_feat_padded = pad_features(poi_features, max_dim)
    lig_feat_padded = pad_features(ligand_features, max_dim)

    node_features = torch.cat([e3_feat_padded, poi_feat_padded, lig_feat_padded], dim=0)

    # Node type encoding: 0=E3, 1=POI, 2=PROTAC
    node_types = torch.cat([
        torch.zeros(n_e3, dtype=torch.long),
        torch.ones(n_poi, dtype=torch.long),
        2 * torch.ones(n_lig, dtype=torch.long),
    ], dim=0)

    return {
        'node_coords': node_coords,          # [N_total, 3]
        'node_features': node_features,      # [N_total, max_dim]
        'node_types': node_types,            # [N_total]
        'n_e3': n_e3,
        'n_poi': n_poi,
        'n_lig': n_lig,
    }


def transform_ligand_to_poi_frame(
    ligand_coords: torch.Tensor,
    t_e3_to_poi: torch.Tensor,
    r_e3_to_poi: torch.Tensor,
) -> torch.Tensor:
    """
    Transform PROTAC coordinates from E3 reference frame to POI reference frame.

    Args:
        ligand_coords: [N_lig, 3] PROTAC coords in E3 frame
        t_e3_to_poi: [3] translation vector (POI center relative to E3 center)
        r_e3_to_poi: [3, 3] rotation matrix from E3 frame to POI frame

    Returns:
        [N_lig, 3] PROTAC coords in POI frame
    """
    return (ligand_coords - t_e3_to_poi) @ r_e3_to_poi

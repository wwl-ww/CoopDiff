"""
Subgraph construction for TriCoopDiff.

Two physically disconnected subgraphs (E3-L and POI-L) use DiffDock's input
schema. Chemistry, graph construction and pose distributions require separate
validation against the pretrained expert's input contract.
A third full ternary graph connects all three molecules for TriFuse coordination.
"""

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph


def build_binary_graph(
    receptor_coords: torch.Tensor,
    receptor_features: torch.Tensor,
    ligand_coords: torch.Tensor,
    ligand_features: torch.Tensor,
    rec_edge_index: torch.Tensor = None,
    lig_edge_index: torch.Tensor = None,
    lig_edge_attr: torch.Tensor = None,
    lig_edge_mask: torch.Tensor = None,
    lig_mask_rotate: torch.Tensor = None,
) -> HeteroData:
    """Build one DiffDock-compatible receptor/ligand graph in a shared frame.

    Coordinates are [N,3], features [N,D], internal edges [2,E]. Cross edges
    are rebuilt by the frozen expert from the current coordinates.
    """
    graph = HeteroData()

    graph['receptor'].pos = receptor_coords
    graph['receptor'].x = receptor_features

    graph['ligand'].pos = ligand_coords
    graph['ligand'].x = ligand_features

    # Single-complex batching: DiffDock's CGModel indexes `data['ligand'].batch`
    # and `data.num_graphs` in the score heads, so these MUST be set. Batch must
    # live on the same device as the coords (tor_sigma[ligand.batch] indexing and
    # the radius() fallback both reject cross-device index tensors on CUDA).
    graph['receptor'].batch = torch.zeros(receptor_coords.shape[0], dtype=torch.long, device=receptor_coords.device)
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


def build_subgraph_e3l(e3_coords, e3_features, *args, **kwargs):
    """Historical E3-L API; graph construction is shared between both experts."""
    return build_binary_graph(e3_coords, e3_features, *args, **kwargs)


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
    return build_binary_graph(
        poi_coords, poi_features, ligand_coords_poi_frame, ligand_features,
        rec_edge_index, lig_edge_index, lig_edge_attr, lig_edge_mask, lig_mask_rotate)


def build_trifuse_graph(
    e3_coords: torch.Tensor,
    e3_features: torch.Tensor,
    poi_coords: torch.Tensor,
    poi_features: torch.Tensor,
    ligand_coords: torch.Tensor,
    ligand_features: torch.Tensor,
    lig_edge_index: torch.Tensor = None,
    lig_edge_mask: torch.Tensor = None,
    lig_mask_rotate=None,
) -> dict:
    """
    Build the full ternary graph for TriFuse processing.

    Unlike the two DiffDock subgraphs, this structure contains all E3 residues,
    POI residues, and PROTAC atoms with explicit component labels. TriFuse
    constructs invariant descriptors for all three cross-component pairs from
    these coordinates; no dense edge tensor is materialized here.

    Returns a flat dictionary (not HeteroData) for the custom TriFuse architecture.

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
        torch.zeros(n_e3, dtype=torch.long, device=node_coords.device),
        torch.ones(n_poi, dtype=torch.long, device=node_coords.device),
        2 * torch.ones(n_lig, dtype=torch.long, device=node_coords.device),
    ], dim=0)

    return {
        'node_coords': node_coords,          # [N_total, 3]
        'node_features': node_features,      # [N_total, max_dim]
        'node_types': node_types,            # [N_total]
        'n_e3': n_e3,
        'n_poi': n_poi,
        'n_lig': n_lig,
        'lig_edge_index': lig_edge_index,
        'lig_edge_mask': lig_edge_mask,
        'lig_mask_rotate': lig_mask_rotate,
    }


def crop_receptor_to_ligand(
    receptor_coords: torch.Tensor,
    receptor_features: torch.Tensor,
    ligand_coords: torch.Tensor,
    rec_edge_index: torch.Tensor = None,
    cutoff: float | torch.Tensor = None,
    min_residues: int = 1,
    min_residues_with_edges: int = 8,
    require_internal_edges: bool = True,
) -> dict:
    """
    Lightweight TriCoopDiff-specific receptor crop.

    Mirrors the native DiffDock sampling driver semantics at the minimal schema
    our subgraph builders actually consume: keep receptor residues within
    ``cutoff`` of any ligand atom, then relabel ``('receptor','receptor')``
    edges. This intentionally omits native-only fields such as
    ``side_chain_vecs`` / ``rec_contact``.

    Args:
        receptor_coords: [N_rec, 3] receptor residue coordinates
        receptor_features: [N_rec, D_rec] receptor residue features
        ligand_coords: [N_lig, 3] ligand coordinates in the SAME frame
        rec_edge_index: [2, E_rec] receptor internal edges (optional)
        cutoff: scalar crop radius. ``None`` disables cropping.
        min_residues: fallback minimum number of kept residues. If the distance
            crop would keep fewer than this many residues (e.g. ligand starts far
            away from the protein), fall back to the ``min_residues`` closest
            residues so the expert never sees an empty receptor graph.
        min_residues_with_edges: stronger fallback used when receptor internal
            edges are available. Keeping a single isolated residue leaves the
            receptor-only edge groups empty, which drives DiffDock's grouped
            tensor-product conv into a divide-by-zero path. Promote the crop to
            the ``min_residues_with_edges`` nearest residues so the receptor
            subgraph retains a small but connected local neighborhood.
        require_internal_edges: when True and ``rec_edge_index`` is available,
            keep enlarging the nearest-residue fallback until the cropped
            receptor subgraph has at least one internal edge (or all residues are
            exhausted). This prevents the ``n_rec=1, n_rec_e=0, n_cross_e=0``
            degenerate graphs that the parity trace showed exploding inside the
            grouped tensor-product convolutions.

    Returns:
        dict with cropped ``coords``, ``features``, ``edge_index`` and the
        boolean ``mask`` over the original receptor residues.
    """
    if cutoff is None:
        return {
            'coords': receptor_coords,
            'features': receptor_features,
            'edge_index': rec_edge_index,
            'mask': torch.ones(receptor_coords.shape[0], dtype=torch.bool, device=receptor_coords.device),
        }

    if torch.is_tensor(cutoff):
        cutoff = float(cutoff.detach().cpu().item())

    if receptor_coords.shape[0] == 0:
        return {
            'coords': receptor_coords,
            'features': receptor_features,
            'edge_index': rec_edge_index,
            'mask': torch.zeros(0, dtype=torch.bool, device=receptor_coords.device),
        }

    dist2 = torch.sum(
        (receptor_coords.unsqueeze(1) - ligand_coords.unsqueeze(0)) ** 2,
        dim=-1,
    )
    min_dist2 = dist2.min(dim=1).values
    distance_order = torch.argsort(min_dist2)
    keep_mask = torch.any(dist2 < cutoff ** 2, dim=1)

    keep_count = int(keep_mask.sum().item())
    target_min = min_residues
    has_rec_graph = rec_edge_index is not None and receptor_coords.shape[0] > 1
    if has_rec_graph:
        target_min = max(target_min, min_residues_with_edges)

    if keep_count < target_min:
        k = min(max(target_min, 1), receptor_coords.shape[0])
        keep_mask = torch.zeros_like(keep_mask)
        keep_mask[distance_order[:k]] = True

    cropped_edge_index = rec_edge_index
    if rec_edge_index is not None:
        cropped_edge_index = subgraph(keep_mask, rec_edge_index, relabel_nodes=True)[0]

        if has_rec_graph and require_internal_edges:
            max_k = receptor_coords.shape[0]
            k = int(keep_mask.sum().item())
            while k < max_k and cropped_edge_index.shape[1] == 0:
                k = min(max_k, max(k + 1, target_min))
                keep_mask = torch.zeros_like(keep_mask)
                keep_mask[distance_order[:k]] = True
                cropped_edge_index = subgraph(keep_mask, rec_edge_index, relabel_nodes=True)[0]

    # Align with native crop_beyond semantics: once the crop is stable, do not
    # over-prune to a single residue if the geometric cutoff happened to miss all
    # residues at this step. The interface expert already has a native-style
    # fallback for fully empty cross graphs, so the crop should remain a genuine
    # receptor neighborhood whenever internal edges exist.
    if has_rec_graph and cropped_edge_index is not None and cropped_edge_index.shape[1] == 0 and keep_mask.sum() > 1:
        keep_mask = torch.zeros_like(keep_mask)
        keep_mask[distance_order[:min(receptor_coords.shape[0], max(target_min, 2))]] = True
        cropped_edge_index = subgraph(keep_mask, rec_edge_index, relabel_nodes=True)[0]

    return {
        'coords': receptor_coords[keep_mask],
        'features': receptor_features[keep_mask],
        'edge_index': cropped_edge_index,
        'mask': keep_mask,
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

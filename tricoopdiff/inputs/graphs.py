"""
Subgraph construction for TriCoopDiff.

Two physically disconnected subgraphs (E3-L and POI-L) use DiffDock's input
schema. Chemistry, graph construction and pose distributions require separate
validation against the pretrained expert's input contract.
A third full ternary graph connects all three molecules for TriFuse coordination.
"""

from collections import OrderedDict
import math

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph

from tricoopdiff.inputs.static_cache import tensor_view_key


_TRIFUSE_STATIC_CACHE = OrderedDict()
_TRIFUSE_STATIC_CACHE_LIMIT = 8


def _validate_component(coords, features, name):
    """Check graph metadata without synchronizing coordinate values on GPU."""
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError(f'{name} coordinates must be nonempty [N, 3]')
    if features is not None and (features.ndim != 2 or features.shape[0] != len(coords)
                                 or features.shape[1] == 0):
        raise ValueError(f'{name} features must match its coordinate node count')


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
    _validate_component(receptor_coords, receptor_features, 'receptor')
    _validate_component(ligand_coords, ligand_features, 'ligand')
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
    lig_edge_attr: torch.Tensor = None,
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
    _validate_component(e3_coords, e3_features, 'E3')
    _validate_component(poi_coords, poi_features, 'POI')
    _validate_component(ligand_coords, ligand_features, 'ligand')
    n_e3 = e3_coords.shape[0]
    n_poi = poi_coords.shape[0]
    n_lig = ligand_coords.shape[0]

    # Coordinates remain live; only labels, identity features and topology can
    # survive rollout steps. Source references prevent storage-key reuse.
    sources = (e3_features, poi_features, ligand_features, lig_edge_index)
    signatures = tuple(tensor_view_key(value) if value is not None else None for value in sources)
    cacheable = (all(value is None or not value.requires_grad for value in sources)
                 and all(value is None or key is not None for value, key in zip(sources, signatures)))
    static_key = (str(e3_coords.device), signatures) if cacheable else None
    cached = _TRIFUSE_STATIC_CACHE.get(static_key) if cacheable else None
    node_coords = torch.cat([e3_coords, poi_coords, ligand_coords], dim=0)
    if cached is not None and any(tensor_view_key(value) != key
                                  for value, key in cached['versions']):
        cached = None
    if cached is not None:
        _TRIFUSE_STATIC_CACHE.move_to_end(static_key)
        static = cached['static']
        return {**static, 'node_coords': node_coords,
                'lig_edge_index': lig_edge_index, 'lig_edge_mask': lig_edge_mask,
                'lig_mask_rotate': lig_mask_rotate, 'lig_edge_attr': lig_edge_attr}

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

    component_indices = tuple(torch.arange(start, stop, device=node_coords.device)
        for start, stop in ((0, n_e3), (n_e3, n_e3 + n_poi),
                            (n_e3 + n_poi, n_e3 + n_poi + n_lig)))
    static = {'node_features': node_features, 'node_types': node_types,
              'n_e3': n_e3, 'n_poi': n_poi, 'n_lig': n_lig,
              'component_indices': component_indices,
              'global_ligand_edges': (component_indices[2][lig_edge_index]
                                      if lig_edge_index is not None else None)}
    if cacheable:
        tensors = (node_features, node_types, *component_indices, static['global_ligand_edges'])
        versions = tuple((value, tensor_view_key(value)) for value in tensors if value is not None)
        # Inference-mode outputs cannot safely be reused by a later training
        # forward, even when their source tensors have version counters.
        if all(key is not None for _, key in versions):
            _TRIFUSE_STATIC_CACHE[static_key] = {
                'sources': sources, 'static': static, 'versions': versions}
            while len(_TRIFUSE_STATIC_CACHE) > _TRIFUSE_STATIC_CACHE_LIMIT:
                _TRIFUSE_STATIC_CACHE.popitem(last=False)
    return {
        **static,
        'node_coords': node_coords,          # [N_total, 3]
        'node_features': node_features,      # [N_total, max_dim]
        'node_types': node_types,            # [N_total]
        'n_e3': n_e3,
        'n_poi': n_poi,
        'n_lig': n_lig,
        'lig_edge_index': lig_edge_index,
        'lig_edge_mask': lig_edge_mask,
        'lig_mask_rotate': lig_mask_rotate,
        'lig_edge_attr': lig_edge_attr,
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
    _validate_component(receptor_coords, receptor_features, 'receptor')
    _validate_component(ligand_coords, None, 'ligand')
    for value in (min_residues, min_residues_with_edges):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError('receptor crop minimum counts must be positive integers')
    if cutoff is None:
        return {
            'coords': receptor_coords,
            'features': receptor_features,
            'edge_index': rec_edge_index,
            'mask': torch.ones(receptor_coords.shape[0], dtype=torch.bool, device=receptor_coords.device),
        }

    if torch.is_tensor(cutoff):
        if cutoff.numel() != 1 or cutoff.dtype == torch.bool:
            raise ValueError('receptor crop cutoff must be a finite nonnegative scalar')
        cutoff = float(cutoff.detach().cpu().item())
    if isinstance(cutoff, bool) or not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError('receptor crop cutoff must be a finite nonnegative scalar')

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

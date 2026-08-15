"""
TriCoopDiff: Top-level model orchestrating two-body experts and TriBrain.

The forward pass at each diffusion step:
1. Build two disconnected subgraphs (E3-L, POI-L) + full ternary graph
2. TriBrain processes the ternary graph → FiLM params + s_SE3
3. Two-body experts run forward with per-layer FiLM modulation
4. Scores from two subgraphs are merged (direct addition)
5. Output: combined PROTAC scores + E3-POI rigid body score
"""

import torch
import torch.nn as nn
import numpy as np

from .graphs import (
    build_subgraph_e3l,
    build_subgraph_pol,
    build_tribrain_graph,
    transform_ligand_to_poi_frame,
)
from .expert import FrozenDiffDock
from .tribrain import TriBrain
from .film import apply_film


class TriCoopDiff(nn.Module):
    """
    PROTAC ternary complex cooperative diffusion model.

    Components:
    - Two frozen DiffDock experts (shared weights, two subgraphs)
    - One trainable TriBrain (~2.2M params)
    - Multi-rate diffusion process
    - Score merging by direct addition
    """

    def __init__(
        self,
        frozen_expert: FrozenDiffDock,
        tribrain: TriBrain,
        num_conv_layers: int = 11,
        use_film: bool = True,
        use_learned_merge: bool = False,
    ):
        """
        Args:
            frozen_expert: FrozenDiffDock wrapping DiffDock CGModel
            tribrain: TriBrain coordination network
            num_conv_layers: number of conv layers in the expert (11 for DiffDock-L)
            use_film: if False, skip FiLM modulation (L0 ablation)
            use_learned_merge: if True, use LearnedCoordinationLayer instead
                              of direct addition (Ablation B)
        """
        super().__init__()
        self.expert = frozen_expert
        self.tribrain = tribrain
        self.num_conv_layers = num_conv_layers
        self.use_film = use_film
        self.use_learned_merge = use_learned_merge

        if use_learned_merge:
            from .merging import LearnedCoordinationLayer
            self.merge_layer = LearnedCoordinationLayer(hidden_dim=tribrain.node_dim)

        self._layer_cache = {}

    def forward(
        self,
        e3_data: dict,
        poi_data: dict,
        ligand_data: dict,
        t: torch.Tensor,
        t_se3: torch.Tensor = None,
        r_e3_to_poi: torch.Tensor = None,
        pocket_guide: torch.Tensor = None,
    ) -> dict:
        """
        Single diffusion step forward pass.

        Args:
            e3_data: dict with 'coords' [B,N_e3,3], 'features' [B,N_e3,D_rec],
                     'edge_index' [2,E_rec] (optional)
            poi_data: dict with 'coords' [B,N_poi,3], 'features' [B,N_poi,D_rec]
            ligand_data: dict with 'coords' [B,N_lig,3], 'features' [B,N_lig,D_lig],
                         'edge_index', 'edge_attr', 'edge_mask', 'mask_rotate' (optional)
            t: [B] diffusion time for PROTAC DOFs
            t_se3: [B] diffusion time for SE3 DOFs (defaults to t if None)
            r_e3_to_poi: [B,3,3] rotation from E3 frame to POI frame. Defaults to
                identity (backward-compatible). This is the transpose of the
                cumulative SE3 rotation applied to the POI (tracked by the
                sampler / training loop), so subgraph2 sees the correct
                PROTAC-POI relative geometry.
            pocket_guide: [B,3] optional pocket center for guidance potential

        Returns:
            dict with:
              's_q': dict with 'tr', 'rot', 'tor' PROTAC scores
              's_se3': [B,6] E3-POI rigid body score
              'film_params': list of FiLM params (for analysis)
              'confidence': [B,1] clash confidence
              'state': [B,2] thermodynamic state predictions
        """
        if t_se3 is None:
            t_se3 = t

        B = t.shape[0]
        device = t.device

        # ------------------------------------------------------------------
        # 1-5: Iterate over batch (HeteroData subgraphs are per-complex)
        # ------------------------------------------------------------------
        all_s_q = {'tr': [], 'rot': [], 'tor': []}
        all_s_se3 = []
        all_s_se3_components = {'tr': [], 'rot': []}
        all_confidence = []
        all_state = []

        for b in range(B):
            # Compute POI→E3 relative transform
            e3_center = e3_data['coords'][b].mean(dim=0)
            poi_center = poi_data['coords'][b].mean(dim=0)
            t_e3_to_poi = poi_center - e3_center

            r_e3 = (
                r_e3_to_poi[b]
                if r_e3_to_poi is not None
                else torch.eye(3, device=device)
            )
            ligand_coords_poi = transform_ligand_to_poi_frame(
                ligand_data['coords'][b], t_e3_to_poi, r_e3,
            )

            # Build subgraphs
            sub1 = build_subgraph_e3l(
                e3_coords=e3_data['coords'][b],
                e3_features=e3_data['features'][b],
                ligand_coords=ligand_data['coords'][b],
                ligand_features=ligand_data['features'][b],
                rec_edge_index=e3_data.get('edge_index'),
                lig_edge_index=ligand_data.get('edge_index'),
                lig_edge_attr=ligand_data.get('edge_attr'),
                lig_edge_mask=ligand_data.get('edge_mask'),
                lig_mask_rotate=ligand_data.get('mask_rotate'),
            )

            sub2 = build_subgraph_pol(
                poi_coords=poi_data['coords'][b],
                poi_features=poi_data['features'][b],
                ligand_coords_poi_frame=ligand_coords_poi,
                ligand_features=ligand_data['features'][b],
                rec_edge_index=poi_data.get('edge_index'),
                lig_edge_index=ligand_data.get('edge_index'),
                lig_edge_attr=ligand_data.get('edge_attr'),
                lig_edge_mask=ligand_data.get('edge_mask'),
                lig_mask_rotate=ligand_data.get('mask_rotate'),
            )

            # Build ternary graph for TriBrain
            tri_graph = build_tribrain_graph(
                e3_coords=e3_data['coords'][b],
                e3_features=e3_data['features'][b],
                poi_coords=poi_data['coords'][b],
                poi_features=poi_data['features'][b],
                ligand_coords=ligand_data['coords'][b],
                ligand_features=ligand_data['features'][b],
            )

            # TriBrain forward (node features are 2D [N, D]; no batch dim)
            tri_out = self.tribrain(
                node_features=tri_graph['node_features'],
                node_coords=tri_graph['node_coords'],
                node_types=tri_graph['node_types'],
            )

            film_params_b = tri_out['film_params']

            # Expert forward with (optional) FiLM
            if self.use_film:
                scores_1 = _expert_forward_with_film(
                    self.expert, sub1, t[b:b+1], film_params_b[:self.num_conv_layers]
                )
                scores_2 = _expert_forward_with_film(
                    self.expert, sub2, t[b:b+1], film_params_b[self.num_conv_layers:]
                )
            else:
                # L0 ablation: no FiLM, use native expert forward
                raw_out1 = self.expert(sub1)
                raw_out2 = self.expert(sub2)
                scores_1 = {'tr_score': raw_out1[0], 'rot_score': raw_out1[1], 'tor_score': raw_out1[2]}
                scores_2 = {'tr_score': raw_out2[0], 'rot_score': raw_out2[1], 'tor_score': raw_out2[2]}

            # Merge scores
            if self.use_learned_merge:
                merged = self.merge_layer(scores_1, scores_2, tri_out['h_cond'])
            else:
                from .merging import merge_direct_addition
                merged = merge_direct_addition(scores_1, scores_2)

            all_s_q['tr'].append(merged['tr_score'])
            all_s_q['rot'].append(merged['rot_score'])
            all_s_q['tor'].append(merged['tor_score'])

            all_s_se3.append(tri_out['s_se3'])
            all_s_se3_components['tr'].append(tri_out['s_se3'][:, :3])
            all_s_se3_components['rot'].append(tri_out['s_se3'][:, 3:])
            all_confidence.append(tri_out['confidence'])
            all_state.append(tri_out['state'])

        # Stack batch results
        s_q = {
            k: torch.stack(v, dim=0).squeeze(0) if len(v[0].shape) <= 2
            else torch.cat(v, dim=0)
            for k, v in all_s_q.items()
        }

        return {
            's_q': s_q,
            's_se3': torch.cat(all_s_se3, dim=0),
            's_se3_components': {
                'tr': torch.cat(all_s_se3_components['tr'], dim=0),
                'rot': torch.cat(all_s_se3_components['rot'], dim=0),
            },
            'confidence': torch.cat(all_confidence, dim=0),
            'state': torch.cat(all_state, dim=0),
        }

    def _expert_forward_with_film(
        self,
        data,
        t: torch.Tensor,
        film_params: list,
    ) -> dict:
        """Method retained for backwards-compatible callers; delegates to the
        module-level `_expert_forward_with_film`."""
        return _expert_forward_with_film(self.expert, data, t, film_params)


def _expert_forward_with_film(
    expert: FrozenDiffDock,
    data,
    t: torch.Tensor,
    film_params: list,
) -> dict:
    """
    Run expert forward pass with per-layer FiLM modulation.

    Replicates CGModel.forward() logic but inserts FiLM between conv layers.

    Args:
        expert: FrozenDiffDock wrapping the CGModel
        data: HeteroData subgraph
        t: [B] diffusion time
        film_params: list of (gamma, beta) tuples, one per conv layer

    Returns:
        dict with 'tr_score', 'rot_score', 'tor_score'
    """
    model = expert.model  # the underlying CGModel

    # Set time embeddings on data
    _set_time_embeddings(data, t, model)

    # Get noise levels
    tr_sigma, rot_sigma, tor_sigma = model.t_to_sigma(
        *[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']]
    )

    # --- Embedding phase ---
    lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight, \
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh, rec_edge_weight = \
        model.embedding(data)

    # --- Build cross graph ---
    if model.dynamic_max_cross:
        cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1)
    else:
        cross_cutoff = model.cross_max_distance

    lr_edge_index, lr_edge_attr, lr_edge_sh, rev_lr_edge_sh, lr_edge_weight = \
        model.build_cross_conv_graph(data, cross_cutoff)
    lr_edge_attr = model.cross_edge_embedding(lr_edge_attr)

    # --- Combine nodes and edges ---
    node_attr = torch.cat([lig_node_attr, rec_node_attr], dim=0)
    n_lig = len(lig_node_attr)
    lr_edge_index = lr_edge_index.clone()
    lr_edge_index[1] = lr_edge_index[1] + n_lig

    edge_index = torch.cat([
        lig_edge_index, lr_edge_index,
        rec_edge_index + n_lig,
        torch.flip(lr_edge_index, dims=[0]),
    ], dim=1)
    edge_attr = torch.cat([
        lig_edge_attr, lr_edge_attr, rec_edge_attr, lr_edge_attr
    ], dim=0)
    edge_sh = torch.cat([
        lig_edge_sh, lr_edge_sh, rec_edge_sh, rev_lr_edge_sh
    ], dim=0)
    edge_weight = torch.cat([
        lig_edge_weight, lr_edge_weight, rec_edge_weight, lr_edge_weight
    ], dim=0) if torch.is_tensor(lig_edge_weight) else torch.ones(
        (len(edge_index[0]), 1), device=edge_index.device
    )

    s1 = len(lig_edge_index[0])
    s2 = s1 + len(lr_edge_index[0])
    s3 = s2 + len(rec_edge_index[0])

    # --- Per-layer conv with FiLM ---
    for l in range(len(model.conv_layers)):
        # Prepare edge_attr for this layer
        if l < len(model.conv_layers) - 1:
            edge_attr_ = torch.cat([
                edge_attr,
                node_attr[edge_index[0], :model.ns],
                node_attr[edge_index[1], :model.ns],
            ], -1)
            if model.differentiate_convolutions:
                edge_attr_ = [
                    edge_attr_[:s1], edge_attr_[s1:s2],
                    edge_attr_[s2:s3], edge_attr_[s3:]
                ]
            node_attr = model.conv_layers[l](
                node_attr, edge_index, edge_attr_, edge_sh,
                edge_weight=edge_weight
            )
        else:
            edge_attr_ = torch.cat([
                edge_attr[:s2],
                node_attr[edge_index[0, :s2], :model.ns],
                node_attr[edge_index[1, :s2], :model.ns],
            ], -1)
            if model.differentiate_convolutions:
                edge_attr_ = [edge_attr_[:s1], edge_attr_[s1:s2]]
            node_attr = model.conv_layers[l](
                node_attr, edge_index[:, :s2], edge_attr_, edge_sh[:s2],
                edge_weight=edge_weight[:s2]
            )

        # --- FiLM modulation (scalar block only, ligand nodes only) ---
        gamma, beta = film_params[l]
        lig_mask = torch.zeros(node_attr.shape[0], dtype=torch.bool, device=node_attr.device)
        lig_mask[:n_lig] = True
        # Modulate only the leading scalar (0e) channels to preserve SE(3)
        # equivariance. gamma/beta are [B, ns]; node_attr is [N, D] with
        # D = ns + vector channels (118 for DiffDock-L).
        node_attr = apply_film(
            node_attr, gamma, beta, node_mask=lig_mask, scalar_dim=expert.ns
        )

    # Extract ligand features
    lig_node_attr = node_attr[:n_lig]

    # --- Final score computation ---
    graph_sigma_emb = model.timestep_emb_func(data.complex_t['tr'])

    scores = expert.compute_scores(
        data, lig_node_attr, graph_sigma_emb,
        tr_sigma, rot_sigma, tor_sigma
    )

    return scores


def _set_time_embeddings(data, t, model):
    """
    Set diffusion time information on the HeteroData object, matching DiffDock's
    `set_time` semantics: `complex_t` holds per-graph [batchsize] tensors and
    `ligand.node_t` holds per-node [num_nodes] tensors. CGModel's
    `timestep_emb_func` (GaussianFourierProjection) requires 1-D tensor inputs,
    so these must be tensors, not Python floats.
    """
    device = data['ligand'].pos.device
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device, dtype=torch.float32)
    t = t.to(device).reshape(-1)  # [1] (single-complex mode)

    batchsize = int(data.num_graphs)
    num_nodes = data['ligand'].pos.shape[0]

    data.complex_t = {
        'tr': t.expand(batchsize),
        'rot': t.expand(batchsize),
        'tor': t.expand(batchsize),
    }
    data['ligand'].node_t = {
        'tr': t.expand(num_nodes),
        'rot': t.expand(num_nodes),
        'tor': t.expand(num_nodes),
    }


def zero_film_verification(expert, data, tribrain, tri_graph, t=None, atol=1e-5):
    """
    Verify that zero-initialized FiLM is transparent: the expert output under
    zero FiLM (gamma=beta=0 from the zero-initialized FiLMHead) must equal the
    native expert output numerically.

    Args:
        expert: FrozenDiffDock wrapping the CGModel
        data: HeteroData subgraph (single protein + PROTAC), already on device
        tribrain: TriBrain (zero-initialized FiLM heads)
        tri_graph: dict from build_tribrain_graph (node_coords/features/types)
        t: [1] diffusion time tensor (defaults to 0.5)
        atol: absolute tolerance for torch.allclose

    Returns:
        (passed: bool, max_abs_diff: float)
    """
    import copy

    device = data['ligand'].pos.device
    if t is None:
        t = torch.tensor([0.5], device=device, dtype=torch.float32)
    t = t.to(device).reshape(-1)

    # --- Native forward (no FiLM) ---
    data_native = copy.deepcopy(data)
    _set_time_embeddings(data_native, t, expert.model)
    with torch.no_grad():
        native_out = expert.model(data_native)
    native_tr, native_rot, native_tor = native_out[:3]

    # --- FiLM forward with (guaranteed-zero) FiLM params ---
    with torch.no_grad():
        tri_out = tribrain(
            node_features=tri_graph['node_features'],
            node_coords=tri_graph['node_coords'],
            node_types=tri_graph['node_types'],
        )
    film_params_1 = tri_out['film_params'][:expert.num_conv_layers]

    data_film = copy.deepcopy(data)
    with torch.no_grad():
        film_scores = _expert_forward_with_film(
            expert, data_film, t, film_params_1
        )

    # --- Compare component-wise ---
    comparisons = [
        ('tr', native_tr, film_scores['tr_score']),
        ('rot', native_rot, film_scores['rot_score']),
        ('tor', native_tor, film_scores['tor_score']),
    ]
    max_abs_diff = 0.0
    passed = True
    for name, native, film in comparisons:
        if native.numel() != film.numel():
            print(f"[zero_film] {name}: SHAPE MISMATCH native={tuple(native.shape)} "
                  f"film={tuple(film.shape)}")
            passed = False
            continue
        if native.numel() == 0:
            continue
        d = (native - film).abs().max().item()
        max_abs_diff = max(max_abs_diff, d)
        ok = torch.allclose(native, film, atol=atol)
        if not ok:
            passed = False
        print(f"[zero_film] {name}: allclose={ok} max_abs_diff={d:.3e}")

    return passed, max_abs_diff

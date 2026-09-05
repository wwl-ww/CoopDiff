"""
TriCoopDiff: Top-level model orchestrating two-body experts and TriFuse.

The forward pass at each diffusion step:
1. Build two disconnected subgraphs (E3-L, POI-L) + full ternary graph
2. TriFuse processes the ternary graph → branch-specific FiLM params
3. Two-body experts run forward with per-layer FiLM modulation
4. Return both interface score fields without merging their rigid coordinates
5. The sampler updates E3--Ligand directly and applies the inverse POI--Ligand
   rigid update to POI
"""

import torch
import torch.nn as nn

from .graphs import (
    build_subgraph_e3l,
    build_subgraph_pol,
    build_trifuse_graph,
    crop_receptor_to_ligand,
    transform_ligand_to_poi_frame,
)
from .expert import FrozenDiffDock
from .trifuse import TriFuse
from .film import apply_film


class TriCoopDiff(nn.Module):
    """
    PROTAC ternary complex cooperative diffusion model.

    Components:
    - Two frozen DiffDock experts (shared weights, two subgraphs)
    - One trainable TriFuse (~2.2M params)
    - Multi-rate diffusion process
    - Explicit E3--Ligand and POI--Ligand interface score fields
    """

    def __init__(
        self,
        frozen_expert: FrozenDiffDock,
        trifuse: TriFuse,
        num_conv_layers: int = 11,
        use_film: bool = True,
    ):
        """
        Args:
            frozen_expert: FrozenDiffDock wrapping DiffDock CGModel
            trifuse: TriFuse coordination network
            num_conv_layers: number of conv layers in the expert (11 for DiffDock-L)
            use_film: if False, skip FiLM modulation (L0 ablation)
        """
        super().__init__()
        self.expert = frozen_expert
        self.trifuse = trifuse
        self.num_conv_layers = num_conv_layers
        self.use_film = use_film

        self._layer_cache = {}

    def forward(
        self,
        e3_data: dict,
        poi_data: dict,
        ligand_data: dict,
        t_e3l: torch.Tensor,
        t_pol: torch.Tensor = None,
        r_e3_to_poi: torch.Tensor = None,
        pocket_guide: torch.Tensor = None,
        receptor_crop_cutoff: torch.Tensor = None,
    ) -> dict:
        """
        Single diffusion step forward pass.

        Args:
            e3_data: dict with 'coords' [B,N_e3,3], 'features' [B,N_e3,D_rec],
                     'edge_index' [2,E_rec] (optional)
            poi_data: dict with 'coords' [B,N_poi,3], 'features' [B,N_poi,D_rec]
            ligand_data: dict with 'coords' [B,N_lig,3], 'features' [B,N_lig,D_lig],
                         'edge_index', 'edge_attr', 'edge_mask', 'mask_rotate' (optional)
            t_e3l: [B] diffusion time for the E3--Ligand interface.
            t_pol: [B] diffusion time for the POI--Ligand interface. Defaults
                to ``t_e3l``.
            r_e3_to_poi: [B,3,3] rotation from E3 frame to POI frame. Defaults to
                identity (backward-compatible). The sampler tracks this inverse
                body-frame map directly so subgraph2 sees the correct
                PROTAC-POI relative geometry.
            pocket_guide: [B,3] optional pocket center for guidance potential
            receptor_crop_cutoff: [B] optional native-style receptor crop radius
                per complex. When provided, each two-body subgraph keeps only the
                receptor residues within this cutoff of the current ligand pose
                (with a nearest-residue fallback), mirroring DiffDock's
                step-level ``crop_beyond(tr_sigma * 3 + 20)`` driver behavior.

        Returns:
            dict with:
              'interface_scores': raw E3--Ligand and mapped POI--Ligand scores
              'torsion_score': TriFuse-coordinated shared Ligand torsion score
              'confidence': [B,1] clash confidence
              'state': [B,2] thermodynamic state predictions
        """
        if t_pol is None:
            t_pol = t_e3l

        B = t_e3l.shape[0]
        device = t_e3l.device

        # ------------------------------------------------------------------
        # 1-5: Iterate over batch (HeteroData subgraphs are per-complex)
        # ------------------------------------------------------------------
        all_expert_scores = {
            'sub1': {'tr': [], 'rot': [], 'tor': []},
            'sub2': {'tr': [], 'rot': [], 'tor': []},
        }
        all_torsion_scores = []
        all_torsion_weights = []
        all_torsion_residuals = []
        all_confidence = []
        all_state = []

        for b in range(B):
            # Compute E3↔POI relative transform. `r_e3_to_poi` maps E3-frame
            # row vectors into the current POI body frame; the inverse map used to
            # rotate POI-frame scores back for merging is therefore its transpose.
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
            # Both sides of the POI-L expert graph must use the same local POI
            # frame. Previously only the ligand was transformed while the POI
            # receptor remained in the global E3 frame. At noisy initial poses
            # that artificial frame split produced sparse, extreme-distance
            # graphs and NaNs in both native and inline expert forwards.
            poi_coords_poi = transform_ligand_to_poi_frame(
                poi_data['coords'][b], t_e3_to_poi, r_e3,
            )

            # Native DiffDock's driver only does step-level receptor cropping when
            # the expert itself is in dynamic-cross mode (`dynamic_max_cross=True`).
            # If the frozen expert was built with static cross edges, forcing an
            # extra node-level crop here creates a non-native hybrid path: the
            # cross-edge cutoff stays static while the receptor node set shrinks.
            # Keep the full receptor graph in that case and let the expert's own
            # cross-edge builder handle far-away poses, exactly like vendored
            # `sampling.py` when `args.crop_beyond is None`.
            use_native_step_crop = bool(getattr(self.expert.model, 'dynamic_max_cross', False))
            crop_cutoff_b = (
                receptor_crop_cutoff[b]
                if receptor_crop_cutoff is not None and use_native_step_crop
                else None
            )
            # Native DiffDock's driver only crops the receptor graph when the
            # underlying expert is in dynamic-cross mode. If the checkpoint is
            # running static cross edges, keep the full receptor graph here and let
            # the expert handle far-away poses the same way its own build_cross_*
            # path does.
            crop_e3 = crop_cutoff_b if use_native_step_crop else None
            crop_poi = crop_cutoff_b if use_native_step_crop else None

            e3_crop = crop_receptor_to_ligand(
                receptor_coords=e3_data['coords'][b],
                receptor_features=e3_data['features'][b],
                ligand_coords=ligand_data['coords'][b],
                rec_edge_index=e3_data.get('edge_index'),
                cutoff=crop_e3,
                min_residues=1,
                min_residues_with_edges=8,
                require_internal_edges=True,
            )
            poi_crop = crop_receptor_to_ligand(
                receptor_coords=poi_coords_poi,
                receptor_features=poi_data['features'][b],
                ligand_coords=ligand_coords_poi,
                rec_edge_index=poi_data.get('edge_index'),
                cutoff=crop_poi,
                min_residues=1,
                min_residues_with_edges=8,
                require_internal_edges=True,
            )

            # Build subgraphs
            sub1 = build_subgraph_e3l(
                e3_coords=e3_crop['coords'],
                e3_features=e3_crop['features'],
                ligand_coords=ligand_data['coords'][b],
                ligand_features=ligand_data['features'][b],
                rec_edge_index=e3_crop['edge_index'],
                lig_edge_index=ligand_data.get('edge_index'),
                lig_edge_attr=ligand_data.get('edge_attr'),
                lig_edge_mask=ligand_data.get('edge_mask'),
                lig_mask_rotate=ligand_data.get('mask_rotate'),
            )

            sub2 = build_subgraph_pol(
                poi_coords=poi_crop['coords'],
                poi_features=poi_crop['features'],
                ligand_coords_poi_frame=ligand_coords_poi,
                ligand_features=ligand_data['features'][b],
                rec_edge_index=poi_crop['edge_index'],
                lig_edge_index=ligand_data.get('edge_index'),
                lig_edge_attr=ligand_data.get('edge_attr'),
                lig_edge_mask=ligand_data.get('edge_mask'),
                lig_mask_rotate=ligand_data.get('mask_rotate'),
            )

            # Build ternary graph for TriFuse
            tri_graph = build_trifuse_graph(
                e3_coords=e3_data['coords'][b],
                e3_features=e3_data['features'][b],
                poi_coords=poi_data['coords'][b],
                poi_features=poi_data['features'][b],
                ligand_coords=ligand_data['coords'][b],
                ligand_features=ligand_data['features'][b],
            )

            # TriFuse forward (node features are 2D [N, D]; no batch dim).
            # TriFuse receives both interface clocks and builds branch-specific
            # FiLM conditions. The experts are never conditioned with a clock
            # belonging to a different relative pose.
            tri_out = self.trifuse(
                node_features=tri_graph['node_features'],
                node_coords=tri_graph['node_coords'],
                node_types=tri_graph['node_types'],
                t_e3l=t_e3l[b:b + 1],
                t_pol=t_pol[b:b + 1],
            )

            film_params_b = tri_out['film_params']

            # Expert forward with (optional) FiLM
            debug_step0 = (
                b == 0
                and bool(getattr(self, '_debug_step0_probe', False))
            )
            sub1_probe = {} if debug_step0 else None
            sub2_probe = {} if debug_step0 else None
            if self.use_film:
                from .lora import set_expert_lora_branch
                set_expert_lora_branch(self.expert, "sub1")
                scores_1 = _expert_forward_with_film(
                    self.expert, sub1, t_e3l[b:b+1], film_params_b[:self.num_conv_layers],
                    trace_prefix=f"sub1/b{b}",
                    debug_probe=sub1_probe,
                )
                set_expert_lora_branch(self.expert, "sub2")
                scores_2 = _expert_forward_with_film(
                    self.expert, sub2, t_pol[b:b+1], film_params_b[self.num_conv_layers:],
                    trace_prefix=f"sub2/b{b}",
                    debug_probe=sub2_probe,
                )
            else:
                # L0 ablation: no FiLM, use native expert forward
                raw_out1 = self.expert(sub1)
                raw_out2 = self.expert(sub2)
                scores_1 = {'tr_score': raw_out1[0], 'rot_score': raw_out1[1], 'tor_score': raw_out1[2]}
                scores_2 = {'tr_score': raw_out2[0], 'rot_score': raw_out2[1], 'tor_score': raw_out2[2]}

            if debug_step0:
                def _score_debug(scores):
                    out = {}
                    for name in ('tr_score', 'rot_score', 'tor_score'):
                        value = scores.get(name)
                        if value is None or value.numel() == 0:
                            out[name] = {'finite': True, 'absmax': 0.0}
                        else:
                            out[name] = {
                                'finite': bool(torch.isfinite(value).all().item()),
                                'absmax': float(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()),
                            }
                    return out

                self._last_step0_probe = {
                    'batch_index': int(b),
                    'receptor_crop_cutoff': (
                        float(crop_cutoff_b.detach().item())
                        if torch.is_tensor(crop_cutoff_b) else crop_cutoff_b
                    ),
                    'e3_crop': {
                        'n_residues': int(e3_crop['coords'].shape[0]),
                        'n_edges': int(e3_crop['edge_index'].shape[1]) if e3_crop.get('edge_index') is not None else 0,
                    },
                    'poi_crop': {
                        'n_residues': int(poi_crop['coords'].shape[0]),
                        'n_edges': int(poi_crop['edge_index'].shape[1]) if poi_crop.get('edge_index') is not None else 0,
                    },
                    'ligand_frame': {
                        'e3_center': e3_center.detach().cpu().tolist(),
                        'poi_center': poi_center.detach().cpu().tolist(),
                        't_e3_to_poi': t_e3_to_poi.detach().cpu().tolist(),
                        'r_e3_to_poi': r_e3.detach().cpu().tolist(),
                        'ligand_com_e3': ligand_data['coords'][b].mean(dim=0).detach().cpu().tolist(),
                        'ligand_com_poi_frame': ligand_coords_poi.mean(dim=0).detach().cpu().tolist(),
                        'poi_com_poi_frame': poi_coords_poi.mean(dim=0).detach().cpu().tolist(),
                    },
                    'sub1': sub1_probe,
                    'sub2': sub2_probe,
                    'scores_sub1': _score_debug(scores_1),
                    'scores_sub2': _score_debug(scores_2),
                }

            # --- Frame fix: rotate sub2's tr/rot scores back to E3 frame ---
            # `transform_ligand_to_poi_frame()` uses row-vector convention:
            # x_poi = (x_e3 - t_e3_to_poi) @ r_e3_to_poi. Therefore sub2's vector
            # scores also live in the POI frame as row vectors, and the inverse
            # map back to the E3 frame is right-multiplication by r_e3_to_poiᵀ.
            # (tor is an internal torsion DOF, so it is frame-invariant.)
            r_poi_to_e3 = r_e3.transpose(-1, -2)
            scores_2['tr_score'] = scores_2['tr_score'] @ r_poi_to_e3
            scores_2['rot_score'] = scores_2['rot_score'] @ r_poi_to_e3

            # Native DiffDock drops the batch dimension from torsion scores for
            # a single graph. Restore the model-level [B,K] contract before
            # coordination and aggregation. Besides serving the coordinator,
            # this prevents a future B>1 loop from concatenating torsions into
            # one flat [B*K] vector.
            for scores in (scores_1, scores_2):
                if scores['tor_score'].ndim == 1:
                    scores['tor_score'] = scores['tor_score'].unsqueeze(0)

            # Preserve both expert fields after mapping them into the common E3
            # frame.  Downstream joint coordination must be able to distinguish
            # the two interface constraints. Both remain available to the
            # sampler and loss in a common E3 frame.
            for score_name, output_name in (
                ('tr_score', 'tr'),
                ('rot_score', 'rot'),
                ('tor_score', 'tor'),
            ):
                all_expert_scores['sub1'][output_name].append(
                    scores_1[score_name]
                )
                all_expert_scores['sub2'][output_name].append(
                    scores_2[score_name]
                )

            if debug_step0:
                self._last_step0_probe['scores_sub2_mapped'] = _score_debug(scores_2)

            # Torsion is the only coordinate shared by both interface states.
            # The experts provide two interface-local opinions; TriFuse makes
            # the single executable decision for every rotatable bond.
            if scores_1['tor_score'].numel() and scores_2['tor_score'].numel():
                torsion_score, torsion_weight, torsion_residual = (
                    self.trifuse.torsion_coordinator(
                        tri_out['h_cond'],
                        scores_1['tor_score'],
                        scores_2['tor_score'],
                    )
                )
            elif scores_1['tor_score'].numel():
                torsion_score = scores_1['tor_score']
                torsion_weight = torch.ones_like(torsion_score)
                torsion_residual = torch.zeros_like(torsion_score)
            else:
                torsion_score = scores_2['tor_score']
                torsion_weight = torch.zeros_like(torsion_score)
                torsion_residual = torch.zeros_like(torsion_score)
            all_torsion_scores.append(torsion_score)
            all_torsion_weights.append(torsion_weight)
            all_torsion_residuals.append(torsion_residual)
            all_confidence.append(tri_out['confidence'])
            all_state.append(tri_out['state'])

        # Stack batch results. Each v[b] carries a leading batch dim of 1
        # (e.g. tr/rot are [1,3], tor is [1,K]), so torch.cat along dim 0
        # yields [B,3]/[B,K] for both B=1 and B>1. (The previous
        # torch.stack(...).squeeze(0) left a stray [B,1,3] for B>1.)
        expert_scores = {
            branch: {
                name: torch.cat(values, dim=0)
                for name, values in branch_scores.items()
            }
            for branch, branch_scores in all_expert_scores.items()
        }

        return {
            'interface_scores': {
                'e3_ligand': expert_scores['sub1'],
                'poi_ligand': expert_scores['sub2'],
            },
            'torsion_score': torch.cat(all_torsion_scores, dim=0),
            'torsion_coordination': {
                'e3l_weight': torch.cat(all_torsion_weights, dim=0),
                'residual': torch.cat(all_torsion_residuals, dim=0),
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
    trace_prefix: str | None = None,
    debug_probe: dict | None = None,
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
    import os
    import sys

    model = expert.model  # the underlying CGModel
    # Zero-FiLM parity debugging is about the sampler path, not the experts
    # themselves. Keep trace spam opt-in; a non-empty prefix is attached only when
    # NAN_TRACE is exported in the environment.
    nan_trace = bool(os.environ.get("NAN_TRACE"))
    lora_branch = "sub2" if str(trace_prefix).startswith("sub2") else "sub1"

    def run_conv(layer, current_node_attr, current_edge_index,
                 current_edge_attr, current_edge_sh, current_edge_weight):
        """Checkpoint LoRA-enabled convolutions to bound training memory."""
        if bool(getattr(expert, "lora_enabled", False)) and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            def forward(node_attr_input):
                # Checkpoint recomputation happens during backward, after the
                # shared expert may have run the other branch. Re-select the
                # adapter so recomputation uses the same parameters as forward.
                from .lora import set_expert_lora_branch
                set_expert_lora_branch(expert, lora_branch)
                return layer(
                    node_attr_input, current_edge_index, current_edge_attr,
                    current_edge_sh, edge_weight=current_edge_weight,
                )

            return checkpoint(forward, current_node_attr, use_reentrant=False)
        return layer(
            current_node_attr, current_edge_index, current_edge_attr,
            current_edge_sh, edge_weight=current_edge_weight,
        )

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

    lr_edge_index, lr_edge_attr_raw, lr_edge_sh, rev_lr_edge_sh, lr_edge_weight = \
        model.build_cross_conv_graph(data, cross_cutoff)
    lr_edge_attr = model.cross_edge_embedding(lr_edge_attr_raw)

    fallback_triggered = False
    fallback_cross_edges = 0
    fallback_cutoff_value = None
    if lr_edge_index.shape[1] == 0 and rec_node_attr.shape[0] > 1 and rec_edge_index.shape[1] > 0:
        # Zero-FiLM parity must be measured against the true vendored DiffDock
        # path. Do NOT silently expand the cross cutoff here: native sampling.py
        # sanitizes NaNs after model(...), but it does not retry with a larger
        # radius when the dynamic crop leaves zero cross edges. Instead, expose
        # this degeneracy via the optional debug_probe so we can compare the exact
        # step-0 graph state against native DiffDock on the same pose.
        fallback_cutoff = (
            torch.full_like(tr_sigma.unsqueeze(1), float(model.cross_max_distance))
            if torch.is_tensor(tr_sigma) else model.cross_max_distance
        )
        fb_lr_edge_index, _, _, _, _ = model.build_cross_conv_graph(data, fallback_cutoff)
        fallback_triggered = True
        fallback_cross_edges = int(fb_lr_edge_index.shape[1])
        fallback_cutoff_value = fallback_cutoff
        if nan_trace:
            prefix = trace_prefix or "expert"
            sys.stderr.write(
                f"[TRACE {prefix} zero-cross] dynamic_cutoff={(tr_sigma * 3 + 20).unsqueeze(1).detach().cpu().tolist() if model.dynamic_max_cross else cross_cutoff} "
                f"fallback_cutoff={fallback_cutoff.detach().cpu().tolist() if torch.is_tensor(fallback_cutoff) else fallback_cutoff} "
                f"fallback_n_cross_e={fallback_cross_edges}\n"
            )
            sys.stderr.flush()

    if nan_trace:
        prefix = trace_prefix or "expert"
        n_cross = int(lr_edge_index.shape[1])
        n_lig = int(lig_edge_index.shape[1])
        n_rec = int(rec_edge_index.shape[1])
        cross_attr_absmax = float(lr_edge_attr.abs().max().item()) if lr_edge_attr.numel() else 0.0
        lig_absmax = float(lig_node_attr.abs().max().item()) if lig_node_attr.numel() else 0.0
        rec_absmax = float(rec_node_attr.abs().max().item()) if rec_node_attr.numel() else 0.0
        sys.stderr.write(
            f"[TRACE {prefix} init] tr_sigma={tr_sigma.detach().cpu().tolist()} "
            f"cross_cutoff={cross_cutoff.detach().cpu().tolist() if torch.is_tensor(cross_cutoff) else cross_cutoff} "
            f"n_lig={len(lig_node_attr)} n_rec={len(rec_node_attr)} "
            f"n_lig_e={n_lig} n_cross_e={n_cross} n_rec_e={n_rec} "
            f"cross_attr_absmax={cross_attr_absmax:.3e} "
            f"lig_emb_absmax={lig_absmax:.3e} rec_emb_absmax={rec_absmax:.3e}"
            "\n"
        )
        sys.stderr.flush()

    if debug_probe is not None:
        debug_probe.update({
            'complex_t': {
                'tr': float(data.complex_t['tr'][0].item()),
                'rot': float(data.complex_t['rot'][0].item()),
                'tor': float(data.complex_t['tor'][0].item()),
            },
            'sigma': {
                'tr': float(tr_sigma[0].item()) if tr_sigma.ndim > 0 else float(tr_sigma.item()),
                'rot': float(rot_sigma[0].item()) if rot_sigma.ndim > 0 else float(rot_sigma.item()),
                'tor': float(tor_sigma[0].item()) if tor_sigma.ndim > 0 else float(tor_sigma.item()),
            },
            'node_counts': {
                'ligand': int(lig_node_attr.shape[0]),
                'receptor': int(rec_node_attr.shape[0]),
            },
            'edge_counts': {
                'ligand': int(lig_edge_index.shape[1]),
                'cross': int(lr_edge_index.shape[1]),
                'receptor': int(rec_edge_index.shape[1]),
            },
            'cross_cutoff': (
                cross_cutoff.detach().cpu().view(-1).tolist()
                if torch.is_tensor(cross_cutoff) else float(cross_cutoff)
            ),
            'fallback_zero_cross': {
                'triggered': bool(fallback_triggered),
                'fallback_cross_edges': int(fallback_cross_edges),
                'fallback_cutoff': (
                    fallback_cutoff_value.detach().cpu().view(-1).tolist()
                    if torch.is_tensor(fallback_cutoff_value)
                    else (float(fallback_cutoff_value) if fallback_cutoff_value is not None else None)
                ),
            },
            'embedding_finite': {
                'ligand': bool(torch.isfinite(lig_node_attr).all().item()),
                'receptor': bool(torch.isfinite(rec_node_attr).all().item()),
                'cross_attr': bool(torch.isfinite(lr_edge_attr).all().item()) if lr_edge_attr.numel() else True,
                'lig_edge_attr': bool(torch.isfinite(lig_edge_attr).all().item()) if lig_edge_attr.numel() else True,
                'rec_edge_attr': bool(torch.isfinite(rec_edge_attr).all().item()) if rec_edge_attr.numel() else True,
            },
            'embedding_absmax': {
                'ligand': float(lig_node_attr.abs().max().item()) if lig_node_attr.numel() else 0.0,
                'receptor': float(rec_node_attr.abs().max().item()) if rec_node_attr.numel() else 0.0,
                'cross_attr': float(lr_edge_attr.abs().max().item()) if lr_edge_attr.numel() else 0.0,
            },
        })

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
            node_attr = run_conv(
                model.conv_layers[l], node_attr, edge_index, edge_attr_,
                edge_sh, edge_weight,
            )
        else:
            edge_attr_ = torch.cat([
                edge_attr[:s2],
                node_attr[edge_index[0, :s2], :model.ns],
                node_attr[edge_index[1, :s2], :model.ns],
            ], -1)
            if model.differentiate_convolutions:
                edge_attr_ = [edge_attr_[:s1], edge_attr_[s1:s2]]
            node_attr = run_conv(
                model.conv_layers[l], node_attr, edge_index[:, :s2],
                edge_attr_, edge_sh[:s2], edge_weight[:s2],
            )

        # Keep the inline expert path numerically identical to the vendored
        # CGModel.forward() path. The production entry point enables NAN_CLAMP
        # after every convolution; omitting it here meant zero-FiLM native and
        # inline forwards diverged only on the high-magnitude POI branch.
        nan_clamp = os.environ.get("NAN_CLAMP")
        if nan_clamp:
            node_attr = torch.clamp(
                node_attr, min=-float(nan_clamp), max=float(nan_clamp)
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

        if nan_trace:
            prefix = trace_prefix or "expert"
            lig_na = node_attr[:n_lig]
            sys.stderr.write(
                f"[TRACE {prefix} l={l}] finite={torch.isfinite(node_attr).all().item()} "
                f"absmax={node_attr.abs().max().item():.3e} "
                f"lig_absmax={lig_na.abs().max().item():.3e}\n"
            )
            sys.stderr.flush()

    # Extract ligand features
    lig_node_attr = node_attr[:n_lig]

    # --- Final score computation ---
    graph_sigma_emb = model.timestep_emb_func(data.complex_t['tr'])

    scores = expert.compute_scores(
        data, lig_node_attr, graph_sigma_emb,
        tr_sigma, rot_sigma, tor_sigma
    )

    if nan_trace:
        prefix = trace_prefix or "expert"
        for name, value in scores.items():
            finite = bool(torch.isfinite(value).all().item()) if value.numel() else True
            absmax = float(value.abs().max().item()) if value.numel() else 0.0
            sys.stderr.write(
                f"[TRACE {prefix} score:{name}] finite={finite} absmax={absmax:.3e}"
                "\n"
            )
        sys.stderr.flush()

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
    if 'receptor' in data.node_types:
        num_rec_nodes = data['receptor'].pos.shape[0]
        data['receptor'].node_t = {
            'tr': t.expand(num_rec_nodes),
            'rot': t.expand(num_rec_nodes),
            'tor': t.expand(num_rec_nodes),
        }


def zero_film_verification(expert, data, trifuse, tri_graph, t=None, atol=1e-5):
    """
    Verify that zero-initialized FiLM is transparent: the expert output under
    zero FiLM (gamma=beta=0 from the zero-initialized FiLMHead) must equal the
    native expert output numerically.

    Args:
        expert: FrozenDiffDock wrapping the CGModel
        data: HeteroData subgraph (single protein + PROTAC), already on device
        trifuse: TriFuse (zero-initialized FiLM heads)
        tri_graph: dict from build_trifuse_graph (node_coords/features/types)
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
        tri_out = trifuse(
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

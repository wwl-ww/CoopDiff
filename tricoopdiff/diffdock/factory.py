"""Vendored DiffDock CGModel factory.

Replicates the ``model_class = CGModel`` branch of DiffDock's
``utils/utils.py:get_model`` verbatim, minus the AAModel / old-model /
DataParallel / RDKit machinery the frozen expert does not need. Parameter
mapping must stay identical to ``get_model`` so the pretrained weights load
with ``strict=True``.

Upstream: github.com/gcorso/DiffDock commit 47eb91a (+ uncommitted
nan-stability-fixes working-tree changes).
"""
from .models.cg_model import CGModel
from .utils.diffusion_utils import get_timestep_embedding


def build_cg_model(args, device, t_to_sigma):
    timestep_emb_func = get_timestep_embedding(
        embedding_type=args.embedding_type if 'embedding_type' in args else 'sinusoidal',
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale if 'embedding_type' in args else 10000)

    lm_embedding_type = None
    if ('moad_esm_embeddings_path' in args and args.moad_esm_embeddings_path is not None) or \
        ('pdbbind_esm_embeddings_path' in args and args.pdbbind_esm_embeddings_path is not None) or \
        ('pdbsidechain_esm_embeddings_path' in args and args.pdbsidechain_esm_embeddings_path is not None) or \
        ('esm_embeddings_path' in args and args.esm_embeddings_path is not None):
        lm_embedding_type = 'precomputed'
    if 'esm_embeddings_model' in args and args.esm_embeddings_model is not None:
        lm_embedding_type = args.esm_embeddings_model

    model = CGModel(
        t_to_sigma=t_to_sigma,
        device=device,
        no_torsion=args.no_torsion,
        timestep_emb_func=timestep_emb_func,
        num_conv_layers=args.num_conv_layers,
        lig_max_radius=args.max_radius,
        scale_by_sigma=args.scale_by_sigma,
        sigma_embed_dim=args.sigma_embed_dim,
        norm_by_sigma='norm_by_sigma' in args and args.norm_by_sigma,
        ns=args.ns, nv=args.nv,
        distance_embed_dim=args.distance_embed_dim,
        cross_distance_embed_dim=args.cross_distance_embed_dim,
        batch_norm=not args.no_batch_norm,
        dropout=args.dropout,
        use_second_order_repr=args.use_second_order_repr,
        cross_max_distance=args.cross_max_distance,
        dynamic_max_cross=args.dynamic_max_cross,
        smooth_edges=args.smooth_edges if "smooth_edges" in args else False,
        odd_parity=args.odd_parity if "odd_parity" in args else False,
        lm_embedding_type=lm_embedding_type,
        confidence_mode=False,
        affinity_prediction=args.affinity_prediction if 'affinity_prediction' in args else False,
        parallel=args.parallel if "parallel" in args else 1,
        num_confidence_outputs=len(
            args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
            args.rmsd_classification_cutoff, list) else 1,
        atom_num_confidence_outputs=len(
            args.atom_rmsd_classification_cutoff) + 1 if 'atom_rmsd_classification_cutoff' in args and isinstance(
            args.atom_rmsd_classification_cutoff, list) else 1,
        parallel_aggregators=args.parallel_aggregators if "parallel_aggregators" in args else "",
        fixed_center_conv=not args.not_fixed_center_conv if "not_fixed_center_conv" in args else False,
        no_aminoacid_identities=args.no_aminoacid_identities if "no_aminoacid_identities" in args else False,
        include_miscellaneous_atoms=args.include_miscellaneous_atoms if hasattr(args, 'include_miscellaneous_atoms') else False,
        sh_lmax=args.sh_lmax if 'sh_lmax' in args else 2,
        differentiate_convolutions=not args.no_differentiate_convolutions if "no_differentiate_convolutions" in args else True,
        tp_weights_layers=args.tp_weights_layers if "tp_weights_layers" in args else 2,
        num_prot_emb_layers=args.num_prot_emb_layers if "num_prot_emb_layers" in args else 0,
        reduce_pseudoscalars=args.reduce_pseudoscalars if "reduce_pseudoscalars" in args else False,
        embed_also_ligand=args.embed_also_ligand if "embed_also_ligand" in args else False,
        atom_confidence=args.atom_confidence_loss_weight > 0.0 if "atom_confidence_loss_weight" in args else False,
        sidechain_pred=(hasattr(args, 'sidechain_loss_weight') and args.sidechain_loss_weight > 0) or
                       (hasattr(args, 'backbone_loss_weight') and args.backbone_loss_weight > 0),
        depthwise_convolution=args.depthwise_convolution if hasattr(args, 'depthwise_convolution') else False)

    model.to(device)
    return model

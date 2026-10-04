"""
TriCoopDiff training entry point.

Supervised training:
  Direct ternary: executed endpoint and protein-interface supervision.
  Explicit legacy mode: SDPI conditional velocity matching with frozen experts.
  Auxiliary confidence/state training is intentionally unavailable until the
  dataset and loss expose explicit labels.

Current scope: B=1 graph forwards with effective-batch gradient accumulation. The
featurized data is produced on the fly from a JSON index (``make_deepternary_index.py``)
via ``featurize_common.featurize_complex`` and cached to disk so subsequent epochs
skip the expensive ESM2 embedding step. This is not padded graph parallelism.

Usage:
    # Build the index once
    python scripts/make_deepternary_index.py

    # Phase 1 smoke: 3 epochs, B=1, single complex per step
    python scripts/train_entry.py \
        --train_data data/protac_index.json \
        --val_data data/protac_index.json \
        --output_dir results/train_smoke \
        --epochs 3 --batch_size 1 --device cuda
"""

import argparse
import random
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

# Make the project root importable for `import tricoopdiff.*`, and scripts/
# importable for the shared featurization module (sibling of this file).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from featurize_common import featurize_complex  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TriCoopDiff training")

    # Data
    p.add_argument("--train_data", required=True, help="Training data JSON")
    p.add_argument("--val_data", required=True, help="Validation data JSON")
    p.add_argument(
        "--max-train-clusters", type=int, default=None,
        help="Bound training clusters for an allocated smoke run only",
    )
    p.add_argument(
        "--max-val-clusters", type=int, default=None,
        help="Bound validation clusters for an allocated smoke run only",
    )
    p.add_argument(
        "--random-flip-proteins", action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly swap fixed/moving protein roles during training "
             "(use --no-random-flip-proteins for an ablation)",
    )

    # Training
    p.add_argument("--output_dir", required=True, help="Checkpoint output directory")
    p.add_argument("--epochs", type=int, default=850)       # paper: 850 epochs
    p.add_argument(
        "--batch_size", type=int, default=1,
        help="Effective batch size via mean-gradient microbatch accumulation, not padded graph parallelism",
    )
    p.add_argument("--lr", type=float, default=1e-3)         # paper: lr=1e-3, Plateau scheduler
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--nan-clamp", type=float, default=100.0,
        help="Per-layer expert activation clamp recorded in checkpoints",
    )
    p.add_argument(
        '--activation-clamp-mode', choices=('irrep_norm', 'componentwise'),
        default='irrep_norm',
        help='Use componentwise explicitly when resuming legacy v2 checkpoints',
    )
    p.add_argument(
        "--torsion-expert-aux-weight", type=float, default=0.25,
        help="Weight for per-expert torsion losses in addition to joint torsion",
    )
    p.add_argument('--residual-weight', type=float, default=1e-3)
    p.add_argument('--interface-loss-weight', type=float, default=1.0)
    p.add_argument('--interface-cutoff', type=float, default=10.0,
                   help='Native C-alpha interface cutoff in Angstrom, not full-atom DockQ contact membership')
    p.add_argument('--steric-loss-weight', type=float, default=None,
                   help='Coarse short-range repulsion: production default .05, custom default 0')
    p.add_argument('--model-profile', choices=('production', 'custom'), default=None,
                   help='Default is the fixed production recipe; explicit legacy overrides select custom')
    p.add_argument('--dropout', type=float, default=None,
                   help='Production uses 0; custom models retain .1 unless explicitly set')
    p.add_argument('--protein-pair-motion', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Add local protein-contact motion; disable for legacy checkpoint resume')
    p.add_argument('--direct-ternary-motion', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Joint geometry owns independent ligand/POI motion; disable for old checkpoints')
    p.add_argument('--global-node-context', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Broadcast component summaries during joint message passing; disable for older direct checkpoints')
    p.add_argument('--spatial-geometry', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Use current-state center distances, angles and handedness in joint geometry')
    p.add_argument('--motion-attention', action=argparse.BooleanOptionalAction, default=True,
                   help='Learn separate translation/rotation interface attention instead of uniform edge averaging')
    p.add_argument('--global-node-attention', action=argparse.BooleanOptionalAction, default=True,
                   help='Add unpooled node-to-all-node attention after local geometric messages')
    p.add_argument('--checkpoint-ternary', action=argparse.BooleanOptionalAction, default=True,
                   help='Recompute ternary activations during backward to reduce rollout memory')
    p.add_argument('--native-root', default='DeepTernary/data/TernaryDB/pdbs',
                   help='Native complexes for exact terminal training-validation DockQ; not the MGD benchmark protocol')
    p.add_argument('--expert-input-grad', action=argparse.BooleanOptionalAction,
                   default=True, help='Keep frozen expert input derivatives during rollouts')
    p.add_argument('--ema-decay', type=float, default=0.999)
    # Phase
    p.add_argument("--phase", type=int, default=1, choices=[1],
                   help="Frozen experts with trainable ternary motion (the supervised phase)")
    p.add_argument("--resume", default=None, help="Resume from checkpoint")

    # Model config
    # TriFuse params
    p.add_argument("--ns", type=int, default=32, help="TriFuse scalar channels")
    p.add_argument("--nv", type=int, default=4, help="TriFuse vector channels")
    p.add_argument("--num_transformer_layers", type=int, default=3)
    p.add_argument("--node_dim", type=int, default=128)
    # Frozen DiffDock-L expert (ns=48, nv=10, num_conv_layers=11). These are read
    # from the checkpoint's model_parameters.yml, not hand-written here.
    p.add_argument("--expert_model_dir", default="checkpoints/score_model",
                   help="Dir with model_parameters.yml + best_ema_inference_epoch_model.pt")

    # Featurization
    p.add_argument("--cache_dir", default="results/featurize_cache",
                   help="Dir for on-disk featurization cache (per-case .pt)")

    # Hardware
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--num_workers", type=int, default=0, choices=[0],
        help="Reserved for a future batched loader; currently must be 0",
    )
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for model initialization, shuffling, and diffusion noise")
    p.add_argument(
        "--training-arm", choices=("A1", "B", "C"), default="A1",
        help="A1=SDPI only, B=mixed without source embedding, C=mixed with it",
    )
    p.add_argument(
        "--bitseed-endpoint-cache", default=None,
        help="Retired full-pool cache option; mixed training now generates lazily",
    )
    p.add_argument(
        "--bitseed-lazy-cache-dir", default=None,
        help="Per-case C=1/M=1 endpoint cache (defaults inside output_dir)",
    )

    args = p.parse_args(argv)
    tokens = sys.argv[1:] if argv is None else argv
    args.explicit_loss_options = [token.split('=')[0] for token in tokens
                                 if token.split('=')[0] in ('--interface-loss-weight',
                                     '--interface-cutoff', '--steric-loss-weight')]
    if args.batch_size < 1:
        p.error('--batch_size must be positive')
    if args.dropout is not None and not 0 <= args.dropout < 1:
        p.error('--dropout must be in [0, 1)')
    custom = (args.protein_pair_motion or not all((args.direct_ternary_motion,
              args.global_node_context, args.spatial_geometry, args.motion_attention,
              args.global_node_attention, args.expert_input_grad))
              or args.dropout not in (None, 0.))
    if args.model_profile is None:
        args.model_profile = 'custom' if custom or args.resume else 'production'
    if args.model_profile == 'production' and custom:
        p.error('production architecture is fixed; use --model-profile custom for legacy overrides')
    return args


def _load_score_model_args(expert_model_dir):
    """Load DiffDock's model_parameters.yml as an argparse.Namespace."""
    import yaml
    from argparse import Namespace
    with open(Path(expert_model_dir) / "model_parameters.yml") as f:
        return Namespace(**yaml.full_load(f))


def _load_index(data_json):
    """Read an index and resolve portable or stale-checkout input paths."""
    with open(data_json) as f:
        index = json.load(f)
    index_root = Path(data_json).resolve().parent.parent
    entries = index.get("entries", []) if isinstance(index, dict) else index
    resolved = []
    for raw_entry in entries:
        entry = dict(raw_entry)
        for key in ("protein1", "protein2", "ligand"):
            path = Path(entry[key])
            if not path.is_absolute():
                local_path = _PROJECT_ROOT / path
                index_path = index_root / path
                path = local_path if local_path.is_file() else index_path
            if not path.is_file():
                # Backward compatibility for indices created in another clone:
                # reconstruct the canonical in-repo case path from stable fields.
                path = (
                    _PROJECT_ROOT / "DeepTernary" / "output" / "protac22"
                    / entry["case"] / Path(entry[key]).name
                )
            entry[key] = str(path.resolve())
        resolved.append(entry)
    entries_by_case = {entry['case']: entry for entry in resolved}
    clusters = index.get('clusters') if isinstance(index, dict) else None
    return resolved, clusters, entries_by_case


def _load_entries(data_json):
    """Backward-compatible flat-entry accessor."""
    return _load_index(data_json)[0]


def swap_protein_reference(e3_data, poi_data, ligand_data):
    """Swap protein roles and express all coordinates in the new E3 frame."""
    new_origin = poi_data['coords'].mean(dim=1, keepdim=True)
    return (
        {**poi_data, 'coords': poi_data['coords'] - new_origin},
        {**e3_data, 'coords': e3_data['coords'] - new_origin},
        {**ligand_data, 'coords': ligand_data['coords'] - new_origin},
    )


def seed_training(seed, device):
    """Seed training RNGs independently of cache/featurization side effects."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def append_metric(output_dir, payload):
    """Append one flush-safe structured training or validation event."""
    path = Path(output_dir) / 'metrics.jsonl'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, allow_nan=False) + '\n')
        handle.flush()


def main():
    args = parse_args()
    args.generative_process = 'sdpi_flow_v1'
    if args.nan_clamp <= 0:
        raise SystemExit("FATAL: --nan-clamp must be positive")
    os.environ["NAN_CLAMP"] = str(args.nan_clamp)
    device = torch.device(args.device)
    endpoint_store = None
    if args.training_arm in ("B", "C"):
        if args.random_flip_proteins:
            raise SystemExit(
                "FATAL: mixed endpoint caches require --no-random-flip-proteins"
            )
        if args.bitseed_endpoint_cache:
            raise SystemExit(
                "FATAL: full-pool endpoint caches are retired for arms B/C; "
                "use --bitseed-lazy-cache-dir"
            )
        if args.bitseed_lazy_cache_dir is None:
            args.bitseed_lazy_cache_dir = str(
                Path(args.output_dir).resolve() / "bitseed_c1m1_cache"
            )
        from tricoopdiff.initialization.endpoints import ONLINE_TRAINING_ENDPOINT_PROTOCOL
        args.training_endpoint_protocol = ONLINE_TRAINING_ENDPOINT_PROTOCOL
    else:
        if args.bitseed_endpoint_cache or args.bitseed_lazy_cache_dir:
            raise SystemExit("FATAL: arm A1 must not consume a BiTSeed endpoint cache")
        args.training_endpoint_protocol = "none"

    seed_training(args.seed, device)

    # SO(3)/torus cache lookup resolves relative to CWD; chdir to project root.
    os.chdir(_PROJECT_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[TriCoopDiff] Training Phase {args.phase}")
    print(f"  Train data: {args.train_data}")
    print(f"  Val data: {args.val_data}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Seed: {args.seed}")
    print(f"  Expert activation clamp: {args.nan_clamp:g}")

    # ----------------------------------------------------------------
    # 0. Featurize the first sample to derive the receptor feature dim
    #    (TriFuse's input_dim must match the featurized ternary nodes).
    # ----------------------------------------------------------------
    entries = _load_entries(args.train_data)
    if not entries:
        raise SystemExit(f"FATAL: no entries in {args.train_data}")

    print("[0/6] Determining receptor feature dim from first sample...")
    score_model_args = _load_score_model_args(args.expert_model_dir)
    first = entries[0]
    _, _, _, input_dim = featurize_complex(
        first["case"], first["protein1"], first["protein2"], first["ligand"],
        score_model_args, str(device), cache_dir=args.cache_dir,
    )
    print(f"  receptor_feat_dim={input_dim} (from {first['case']})")
    # A cold ESM/graph cache can consume framework RNG state while a warm cache
    # does not. Reset here so model initialization and epoch sampling depend
    # only on --seed, never on whether the first feature was already cached.
    seed_training(args.seed, device)

    # ----------------------------------------------------------------
    # 1. Build model
    # ----------------------------------------------------------------
    print("[1/6] Building model...")
    model, diffusion, trainer = build_trainer(args, device, input_dim=input_dim)
    print(f"  Model profile: {args.model_profile} | "
          f"dropout={args.resolved_model_config['dropout'] if 'dropout' in args.resolved_model_config else .1:g}")
    # Include resolved resume/profile settings, not defaults that were ignored.
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as handle:
        json.dump(vars(args), handle, indent=2)
    if args.training_arm in ("B", "C"):
        from tricoopdiff.initialization.endpoints import LazyTrainingEndpointStore
        endpoint_store = LazyTrainingEndpointStore(
            args.bitseed_lazy_cache_dir, model.expert,
        )

    # ----------------------------------------------------------------
    # 2. Load data
    # ----------------------------------------------------------------
    print("[2/6] Loading data...")
    train_loader = TernaryDataLoader(
        args.train_data, score_model_args,
        shuffle=True, device=str(device), cache_dir=args.cache_dir,
        skip_on_error=False, max_clusters=args.max_train_clusters,
        random_flip_proteins=args.random_flip_proteins,
    )
    val_loader = TernaryDataLoader(
        args.val_data, score_model_args,
        shuffle=False, device=str(device), cache_dir=args.cache_dir,
        skip_on_error=False, max_clusters=args.max_val_clusters,
    )
    if trainer.training_objective == 'executed_endpoint_geometry_v1':
        # Check scoring dependencies before spending an epoch on training.
        for entry in val_loader._epoch_entries():
            validation_metric_context(entry, args.native_root)

    # ----------------------------------------------------------------
    # 3. Resume if specified
    # ----------------------------------------------------------------
    if args.resume:
        print(f"[3/6] Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)
    else:
        print("[3/6] Training from scratch...")

    # Set phase (freeze/unfreeze heads)
    trainer.set_phase(args.phase)

    # ----------------------------------------------------------------
    # 4. Training loop
    # ----------------------------------------------------------------
    print(f"[4/6] Training {args.epochs} epochs...")
    start_epoch = trainer.resume_epoch + 1
    best_val_loss = trainer.resume_best_val_loss
    best_val_dockq = trainer.resume_best_val_dockq
    if start_epoch > args.epochs:
        raise RuntimeError(
            f"resume checkpoint already completed epoch {trainer.resume_epoch}, "
            f"which meets/exceeds target --epochs={args.epochs}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_losses = []
        n_skipped = 0
        t0 = time.time()
        for batch_idx, microbatches in enumerate(iter_training_batches(train_loader, args.batch_size)):
            microbatches = [prepare_source_batch(
                batch, trainer.samples_seen + offset, args.training_arm, endpoint_store,
            ) for offset, batch in enumerate(microbatches)]
            group_losses = _guarded_train_batch(trainer, microbatches)
            train_losses.extend(group_losses)
            for batch, loss_dict in zip(microbatches, group_losses):
                append_metric(args.output_dir, {
                    'event': 'train_step', 'epoch': epoch,
                    'global_step': trainer.global_step, 'case': batch['case'],
                    'source': ('bitseed' if batch['source'] == 1 else 'sdpi'),
                    'training_objective': trainer.training_objective,
                    **loss_dict,
                })
            loss_dict = {key: float(np.mean([loss[key] for loss in group_losses]))
                         for key in group_losses[0]}

            if batch_idx % args.log_interval == 0:
                if 'loss_ligand_geometry' in loss_dict:
                    print(f"  Epoch {epoch} | Batch {batch_idx} | "
                          f"loss={loss_dict['loss_total']:.4f} | "
                          f"ligand_geometry={loss_dict['loss_ligand_geometry']:.4f} | "
                          f"protein_geometry={loss_dict['loss_protein_geometry']:.4f}")
                else:
                    print(f"  Epoch {epoch} | Batch {batch_idx} | "
                      f"loss={loss_dict['loss_total']:.4f} | "
                      f"e3l=(tr={loss_dict['loss_e3l_tr']:.4f}, "
                      f"rot={loss_dict['loss_e3l_rot']:.4f}) | "
                      f"pol=(tr={loss_dict['loss_pol_tr']:.4f}, "
                      f"rot={loss_dict['loss_pol_rot']:.4f}) | "
                      f"tor=(joint={loss_dict['loss_score_tor']:.4f}, "
                      f"e3l={loss_dict['loss_e3l_tor']:.4f}, "
                      f"pol={loss_dict['loss_pol_tor']:.4f})")

        if n_skipped:
            raise RuntimeError(
                f"epoch {epoch}: {n_skipped} training batches were non-finite; "
                "refusing to validate or save a partial-data checkpoint"
            )
        if not train_losses:
            raise RuntimeError(f"epoch {epoch}: training loader produced no batches")

        # Validate
        validation = validate(
            trainer, val_loader, device, args.training_arm, endpoint_store,
            native_root=args.native_root,
        )
        val_loss = validation['selection_loss']
        append_metric(args.output_dir, {
            'event': 'validation', 'epoch': epoch,
            'global_step': trainer.global_step, **validation,
            'training_objective': trainer.training_objective,
        })
        if not np.isfinite(val_loss):
            raise RuntimeError(
                f"epoch {epoch}: validation produced no finite batches; "
                "refusing to save a checkpoint"
            )
        trainer.scheduler_step(val_loss)

        # Log
        train_loss_avg = np.mean([d['loss_total'] for d in train_losses]) if train_losses else float('nan')
        endpoint_training = 'loss_ligand_geometry' in train_losses[0]
        train_e3l_avg = np.mean([
            d['loss_ligand_geometry'] if endpoint_training else d['loss_e3l_tr'] + d['loss_e3l_rot']
            for d in train_losses])
        train_pol_avg = np.mean([
            d['loss_protein_geometry'] if endpoint_training else d['loss_pol_tr'] + d['loss_pol_rot']
            for d in train_losses])
        dt = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss_avg:.4f} | "
              f"{'ligand_geometry' if endpoint_training else 'e3l'}={train_e3l_avg:.4f} | "
              f"{'protein_geometry' if endpoint_training else 'pol'}={train_pol_avg:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"skipped={n_skipped} | "
              f"lr={trainer.optimizer.param_groups[0]['lr']:.2e} | "
              f"time={dt:.1f}s")
        print(f"  validation_by_source={validation['by_source']}")
        val_dockq = validation.get('selection_dockq')
        if val_dockq is not None:
            print(f"  terminal_validation_DockQ={val_dockq:.4f} (full atom, training-coupling protocol)")

        # Save best & periodic
        improved_loss = val_loss < best_val_loss
        improved_dockq = val_dockq is not None and val_dockq > best_val_dockq
        if improved_dockq:
            best_val_dockq = val_dockq
        if improved_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                os.path.join(args.output_dir, "best_loss_model.pt"),
                epoch=epoch, best_val_loss=best_val_loss, best_val_dockq=best_val_dockq,
            )
        if improved_dockq or (val_dockq is None and improved_loss):
            trainer.save_checkpoint(
                os.path.join(args.output_dir, 'best_model.pt'), epoch=epoch,
                best_val_loss=best_val_loss, best_val_dockq=best_val_dockq)
            print('  -> Saved best by ' + ('DockQ' if val_dockq is not None else 'legacy loss'))

        if epoch % args.save_interval == 0:
            trainer.save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"),
                epoch=epoch, best_val_loss=best_val_loss, best_val_dockq=best_val_dockq,
            )

    # ----------------------------------------------------------------
    # 5. Final save
    # ----------------------------------------------------------------
    print("[5/6] Saving final model...")
    final_path = os.path.join(args.output_dir, "tricoopdiff_zero.pt")
    trainer.save_checkpoint(
        final_path, epoch=args.epochs, best_val_loss=best_val_loss, best_val_dockq=best_val_dockq,
    )
    print(f"Done. Model saved to {final_path}")


def build_trainer(args, device, input_dim=None):
    """Build model, diffusion, and trainer."""
    from tricoopdiff.models.consensus import ConsensusCoopDiff, ConsensusFuse, production_model_config
    from tricoopdiff.models.expert import build_diffdock_expert
    from tricoopdiff.sdpi_flow import SDPIFlow
    from tricoopdiff.train import ConsensusTrainer

    # Build frozen DiffDock-L expert from pretrained weights (self-contained
    # vendored core — no ../DiffDock dependency). num_conv_layers (11) and the
    # full CGModel architecture come from model_parameters.yml.
    frozen_expert = build_diffdock_expert(
        model_dir=args.expert_model_dir,
        device=str(device),
    )

    # input_dim must match the ternary node feature dim, which is the receptor
    # feature dim (1281 = 1 amino-acid index + 1280 ESM2 for DiffDock-L). It is
    # derived at runtime from the featurized data, never hardcoded.
    if input_dim is None:
        input_dim = args.node_dim
        print(f"[warn] input_dim not provided; defaulting to node_dim={input_dim}")

    # Consensus learns after expert inference; no FiLM dimensions are needed.
    model_config = {
        'node_dim': args.node_dim, 'input_dim': input_dim,
        'num_transformer_layers': args.num_transformer_layers,
    }
    if getattr(args, 'training_arm', 'A1') == 'C':
        model_config['source_conditioning'] = True
    if getattr(args, 'protein_pair_motion', False):
        model_config['protein_pair_motion'] = True
    if getattr(args, 'direct_ternary_motion', False):
        model_config['direct_ternary_motion'] = True
        if getattr(args, 'global_node_context', True):
            model_config['global_node_context'] = True
        if getattr(args, 'spatial_geometry', True):
            model_config['spatial_geometry'] = True
        if getattr(args, 'motion_attention', True):
            model_config['motion_attention'] = True
        if getattr(args, 'global_node_attention', True):
            model_config['global_node_attention'] = True
    if args.model_profile == 'production':
        model_config = production_model_config(args.node_dim, input_dim,
                                               args.num_transformer_layers,
                                               args.training_arm == 'C')
    elif args.dropout is not None:
        model_config['dropout'] = args.dropout
    if args.resume:
        # Resume means reconstruct the saved architecture, not reinterpret it
        # using today's defaults. The trainer subsequently checks every runtime
        # loss/source/optimizer contract before loading trainable state.
        from tricoopdiff.train import validate_consensus_checkpoint
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        validate_consensus_checkpoint(saved)
        model_config = dict(saved['model_config'])
        if args.model_profile == 'production' and not model_config.get('model_profile'):
            raise ValueError('an older custom checkpoint cannot resume as the production profile')
        if args.dropout is not None and args.dropout != model_config.get('dropout', .1):
            raise ValueError('resume cannot override the saved dropout')
        if model_config['input_dim'] != input_dim:
            raise ValueError('resume checkpoint input features do not match the data')
        args.model_profile = 'production' if model_config.get('model_profile') else 'custom'
        for key, option, old_default in (('interface_loss_weight', '--interface-loss-weight', 0.),
                                         ('interface_cutoff', '--interface-cutoff', 10.),
                                         ('steric_loss_weight', '--steric-loss-weight', 0.)):
            if option not in args.explicit_loss_options:
                setattr(args, key, saved['runtime_config'].get(key, old_default))
    args.resolved_model_config = dict(model_config)
    if args.steric_loss_weight is None:
        args.steric_loss_weight = .05 if args.model_profile == 'production' else 0.
    trifuse = ConsensusFuse(**model_config)

    model = ConsensusCoopDiff(frozen_expert, trifuse,
                             expert_input_grad=getattr(args, 'expert_input_grad', True),
                             checkpoint_ternary=getattr(args, 'checkpoint_ternary', True))

    # Build diffusion
    diffusion = SDPIFlow(inference_steps=20)

    # Build trainer
    trainer = ConsensusTrainer(
        model=model,
        diffusion=diffusion,
        lr=args.lr,
        weight_decay=args.weight_decay,
        activation_clamp=args.nan_clamp,
        activation_clamp_mode=args.activation_clamp_mode,
        torsion_expert_aux_weight=args.torsion_expert_aux_weight,
        residual_weight=args.residual_weight,
        ema_decay=args.ema_decay,
        interface_loss_weight=args.interface_loss_weight,
        interface_cutoff=args.interface_cutoff,
        effective_batch_size=args.batch_size,
        steric_loss_weight=args.steric_loss_weight,
        training_source_mix=(
            'sdpi_only' if getattr(args, 'training_arm', 'A1') == 'A1'
            else 'sdpi_bitseed_1_1'
        ),
        training_endpoint_protocol=getattr(
            args, 'training_endpoint_protocol', 'none',
        ),
        device=device,
    )

    return model, diffusion, trainer


def _guarded_train_batch(trainer, batches):
    from tricoopdiff.train import NonFiniteTrainingError

    try:
        return trainer.train_batch(batches)
    except NonFiniteTrainingError as error:
        raise RuntimeError(f'training aborted without updating the partial microbatch group: {error}') from error


def iter_training_batches(loader, size):
    """Group variable-size complexes without retaining their autograd graphs."""
    if size < 1:
        raise ValueError('effective batch size must be positive')
    pending = []
    for batch in loader:
        pending.append(batch)
        if len(pending) == size:
            yield pending
            pending = []
    if pending:
        yield pending


def validation_metric_context(entry, native_root):
    """Resolve scoring-only templates; missing natives must not fall back to a proxy."""
    root = Path(native_root)
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    candidates = [root / entry['case'] / name for name in ('gt_complex.pdb', 'complex.pdb')]
    input_dir = Path(entry['protein1']).parent
    if input_dir.name == entry['case']:
        candidates += [input_dir / name for name in ('gt_complex.pdb', 'complex.pdb')]
    if entry.get('native_complex_pdb'):
        path = Path(entry['native_complex_pdb'])
        candidates.insert(0, path if path.is_absolute() else _PROJECT_ROOT / path)
    native = next((path for path in candidates if path.is_file()), None)
    if native is None:
        raise FileNotFoundError(f"case={entry['case']}: full-atom validation requires a native complex")
    return {'protein1_pdb': entry['protein1'], 'protein2_pdb': entry['protein2'],
            'ligand_pdb': entry['ligand'], 'native_complex_pdb': str(native),
            'deepternary_root': str(_PROJECT_ROOT / 'DeepTernary')}


def prepare_source_batch(batch, global_step, training_arm, endpoint_store):
    """Attach the actual source and matching coupling for one training step."""
    if training_arm == 'A1':
        return {**batch, 'source': 0}
    from tricoopdiff.initialization.endpoints import mixed_source_for_step
    source = mixed_source_for_step(global_step)
    if source == 0:
        return {**batch, 'source': 0}
    if endpoint_store is None:
        raise ValueError('mixed training requires an endpoint store')
    coupling, record = endpoint_store.coupling(
        batch['case'], occurrence=global_step // 2,
        **({'batch': batch} if getattr(endpoint_store, 'lazy', False) else {}),
    )
    actual_source = 1 if record['actual_source'] == 'bitseed' else 0
    return {**batch, 'source': actual_source, 'coupling_override': coupling}


def validate(trainer, val_loader, device, training_arm='A1', endpoint_store=None,
             native_root='DeepTernary/data/TernaryDB/pdbs'):
    """Run a deterministic validation pass without perturbing training RNG."""
    from tricoopdiff.train import VALIDATION_DOCKQ_PROTOCOL
    cuda_devices = []
    if device.type == 'cuda':
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    exact_dockq = getattr(trainer, 'training_objective', None) == 'executed_endpoint_geometry_v1'
    scores = {'sdpi': [], 'bitseed': []}
    per_case = []
    ema_context = trainer.ema_parameters() if hasattr(trainer, 'ema_parameters') else nullcontext()
    try:
        with torch.random.fork_rng(devices=cuda_devices), ema_context:
            torch.manual_seed(0)
            if cuda_devices:
                torch.cuda.manual_seed_all(0)
            np.random.seed(0)
            random.seed(0)
            trainer.model.eval()
            losses = {'sdpi': [], 'bitseed': []}
            fallback_count = 0

            for batch in val_loader:
                metric_kwargs = ({'dockq_context': validation_metric_context(
                    val_loader.entries_by_case[batch['case']], native_root)} if exact_dockq else {})
                try:
                    loss_dict = trainer.validate_step(**{**batch, 'source': 0}, **metric_kwargs)
                except FloatingPointError as error:
                    raise RuntimeError(f"validation failed: {error}") from error
                if not np.isfinite(loss_dict['loss_total']):
                    raise RuntimeError(
                        f"validation failed: case={batch.get('case', 'unknown')} "
                        "produced a non-finite loss"
                    )
                losses['sdpi'].append(loss_dict['loss_total'])
                if exact_dockq:
                    scores['sdpi'].append(loss_dict['dockq'])
                    per_case.append({'case': batch['case'], 'source': 'sdpi',
                                     'loss': loss_dict['loss_total'],
                                     **{key: loss_dict[key] for key in ('dockq', 'fnat', 'irms', 'lrms')}})
                if training_arm in ('B', 'C'):
                    if endpoint_store is None:
                        raise ValueError('mixed validation requires an endpoint store')
                    coupling, record = endpoint_store.coupling(
                        batch['case'],
                        **({'batch': batch} if getattr(endpoint_store, 'lazy', False)
                           else {}),
                    )
                    actual_source = record['actual_source']
                    try:
                        endpoint_loss = trainer.validate_step(**{
                            **batch,
                            'source': 1 if actual_source == 'bitseed' else 0,
                            'coupling_override': coupling,
                        }, **metric_kwargs)
                    except FloatingPointError as error:
                        raise RuntimeError(
                            f"BiTSeed validation failed: {error}"
                        ) from error
                    if not np.isfinite(endpoint_loss['loss_total']):
                        raise RuntimeError(
                            f"validation failed: case={batch.get('case', 'unknown')} "
                            "produced a non-finite BiTSeed loss"
                        )
                    losses[actual_source].append(endpoint_loss['loss_total'])
                    if exact_dockq:
                        scores[actual_source].append(endpoint_loss['dockq'])
                        per_case.append({'case': batch['case'], 'source': actual_source,
                                         'loss': endpoint_loss['loss_total'],
                                         **{key: endpoint_loss[key] for key in ('dockq', 'fnat', 'irms', 'lrms')}})
                    fallback_count += int(actual_source == 'sdpi')
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        trainer.model.train()
    by_source = {
        source: float(np.mean(values))
        for source, values in losses.items() if values
    }
    selection = float(np.mean(list(by_source.values()))) if by_source else float('nan')
    result = {
        'selection_loss': selection, 'by_source': by_source,
        'source_counts': {source: len(values) for source, values in losses.items()},
        'bitseed_fallback_count': fallback_count,
    }
    if exact_dockq:
        dockq_by_source = {source: float(np.mean(values)) for source, values in scores.items() if values}
        if not dockq_by_source:
            raise RuntimeError('full-atom validation produced no DockQ measurements')
        result.update(selection_dockq=float(np.mean(list(dockq_by_source.values()))),
                      dockq_by_source=dockq_by_source, per_case=per_case,
                      dockq_protocol=VALIDATION_DOCKQ_PROTOCOL)
    return result


class TernaryDataLoader:
    """
    Real data loader for PROTAC ternary complexes.

    Reads the JSON index written by ``scripts/make_deepternary_index.py``::
        {"entries": [
            {"case": ..., "protein1": ".../protein1.pdb",
             "protein2": ".../protein2.pdb", "ligand": ".../ligand.sdf"}]}

    Each entry is featurized on the fly via ``featurize_complex`` (with an
    on-disk cache so subsequent epochs skip ESM2). The loader yields a **B=1**
    batch dict whose keys match ``TriCoopDiffTrainer.compute_loss``::
        {'e3_data': ..., 'poi_data': ..., 'ligand_data': ...}

    By default featurization failures abort training. Silently changing the
    effective dataset would make validation/checkpoint selection misleading.
    """

    def __init__(self, data_json, score_model_args, shuffle=True, device="cuda",
                 cache_dir=None, skip_on_error=False, max_clusters=None,
                 random_flip_proteins=False, cache_read_only=False,
                 cache_wait_seconds=3600.):
        self.entries, self.clusters, self.entries_by_case = _load_index(data_json)
        if max_clusters is not None:
            if max_clusters <= 0:
                raise ValueError('max_clusters must be positive')
            if self.clusters:
                self.clusters = self.clusters[:max_clusters]
            else:
                self.entries = self.entries[:max_clusters]
                self.entries_by_case = {
                    entry['case']: entry for entry in self.entries
                }
        self.score_model_args = score_model_args
        self.shuffle = shuffle
        self.device = device
        self.cache_dir = cache_dir
        self.skip_on_error = skip_on_error
        self.random_flip_proteins = bool(random_flip_proteins)
        self.cache_read_only = bool(cache_read_only)
        self.cache_wait_seconds = float(cache_wait_seconds)
        self.skipped = []

    def __iter__(self):
        entries = self._epoch_entries()
        for i, entry in enumerate(entries):
            try:
                yield self._load_one(entry)
            except Exception as e:
                if not self.skip_on_error:
                    raise
                self.skipped.append(entry.get("case", str(i)))
                print(f"[data] SKIP {entry.get('case', i)}: {e}", flush=True)

    def __len__(self):
        return len(self.clusters) if self.clusters else len(self.entries)

    def _epoch_entries(self):
        """Select one case per cluster, following DeepTernary's 80/20 rule."""
        if not self.clusters:
            selected = list(self.entries)
        elif self.shuffle:
            selected = []
            for cluster in self.clusters:
                items = [case for case in cluster['items']
                         if case in self.entries_by_case]
                representatives = [case for case in cluster.get('representative', [])
                                   if case in self.entries_by_case]
                if not items:
                    raise RuntimeError('cluster has no materialized entries')
                candidates = items if np.random.random() < 0.8 else (
                    representatives or items)
                selected.append(self.entries_by_case[np.random.choice(candidates)])
        else:
            selected = []
            for cluster in self.clusters:
                candidates = [case for case in cluster.get('representative', [])
                              if case in self.entries_by_case]
                if not candidates:
                    candidates = [case for case in cluster['items']
                                  if case in self.entries_by_case]
                if not candidates:
                    raise RuntimeError('cluster has no materialized entries')
                selected.append(self.entries_by_case[candidates[0]])
        if self.shuffle:
            np.random.shuffle(selected)
        return selected

    def _load_one(self, entry):
        # ESM/model construction on a cold cache can consume framework RNG
        # while a warm-cache read does not. Feature cache state must never
        # alter diffusion couplings, dropout or later epoch order.
        numpy_state, python_state = np.random.get_state(), random.getstate()
        device = torch.device(self.device)
        cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                        if device.type == 'cuda' else [])
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                e3_data, poi_data, ligand_data, _ = featurize_complex(
                    entry["case"], entry["protein1"], entry["protein2"], entry["ligand"],
                    self.score_model_args, self.device, cache_dir=self.cache_dir,
                    cache_read_only=self.cache_read_only,
                    cache_wait_seconds=self.cache_wait_seconds,
                )
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        if self.random_flip_proteins and np.random.random() < 0.5:
            # Re-express the entire ternary state in the old POI-centered frame,
            # then exchange branch identities. This matches DeepTernary's
            # training-only random protein flip while preserving the invariant
            # that the current E3/reference partner is centered at the origin.
            e3_data, poi_data, ligand_data = swap_protein_reference(
                e3_data, poi_data, ligand_data,
            )
        return {
            'case': entry['case'],
            'e3_data': e3_data,
            'poi_data': poi_data,
            'ligand_data': ligand_data,
        }


if __name__ == "__main__":
    main()

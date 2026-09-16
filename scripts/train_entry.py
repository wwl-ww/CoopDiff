"""
TriCoopDiff training entry point.

Supervised training:
  Phase 1: SDPI conditional velocity matching with frozen experts.
  Auxiliary confidence/state training is intentionally unavailable until the
  dataset and loss expose explicit labels.

Current scope: **B=1 sequential training** on real ternary complexes. The
featurized data is produced on the fly from a JSON index (``make_deepternary_index.py``)
via ``featurize_common.featurize_complex`` and cached to disk so subsequent epochs
skip the expensive ESM2 embedding step. Padded B>1 batching is a follow-up.

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
        "--batch_size", type=int, default=1, choices=[1],
        help="Only B=1 sequential training is currently implemented",
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
    p.add_argument('--ema-decay', type=float, default=0.999)
    # Phase
    p.add_argument("--phase", type=int, default=1, choices=[1],
                   help="SDPI velocity matching (the only defined phase)")
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

    return p.parse_args(argv)


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
        from tricoopdiff.online_endpoints import ONLINE_TRAINING_ENDPOINT_PROTOCOL
        args.training_endpoint_protocol = ONLINE_TRAINING_ENDPOINT_PROTOCOL
    else:
        if args.bitseed_endpoint_cache or args.bitseed_lazy_cache_dir:
            raise SystemExit("FATAL: arm A1 must not consume a BiTSeed endpoint cache")
        args.training_endpoint_protocol = "none"

    seed_training(args.seed, device)

    # SO(3)/torus cache lookup resolves relative to CWD; chdir to project root.
    os.chdir(_PROJECT_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save run config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

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
    if args.training_arm in ("B", "C"):
        from tricoopdiff.online_endpoints import LazyTrainingEndpointStore
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
    if start_epoch > args.epochs:
        raise RuntimeError(
            f"resume checkpoint already completed epoch {trainer.resume_epoch}, "
            f"which meets/exceeds target --epochs={args.epochs}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_losses = []
        n_skipped = 0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            batch = prepare_source_batch(
                batch, trainer.global_step, args.training_arm, endpoint_store,
            )
            loss_dict = _guarded_train_step(trainer, batch)
            train_losses.append(loss_dict)
            append_metric(args.output_dir, {
                'event': 'train_step', 'epoch': epoch,
                'global_step': trainer.global_step, 'case': batch['case'],
                'source': ('bitseed' if batch['source'] == 1 else 'sdpi'),
                **loss_dict,
            })

            if batch_idx % args.log_interval == 0:
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
        )
        val_loss = validation['selection_loss']
        append_metric(args.output_dir, {
            'event': 'validation', 'epoch': epoch,
            'global_step': trainer.global_step, **validation,
        })
        if not np.isfinite(val_loss):
            raise RuntimeError(
                f"epoch {epoch}: validation produced no finite batches; "
                "refusing to save a checkpoint"
            )
        trainer.scheduler_step(val_loss)

        # Log
        train_loss_avg = np.mean([d['loss_total'] for d in train_losses]) if train_losses else float('nan')
        train_e3l_avg = np.mean([
            d['loss_e3l_tr'] + d['loss_e3l_rot'] for d in train_losses
        ])
        train_pol_avg = np.mean([
            d['loss_pol_tr'] + d['loss_pol_rot'] for d in train_losses
        ])
        dt = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss_avg:.4f} | "
              f"e3l={train_e3l_avg:.4f} | "
              f"pol={train_pol_avg:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"skipped={n_skipped} | "
              f"lr={trainer.optimizer.param_groups[0]['lr']:.2e} | "
              f"time={dt:.1f}s")
        print(f"  validation_by_source={validation['by_source']}")

        # Save best & periodic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                os.path.join(args.output_dir, "best_model.pt"),
                epoch=epoch, best_val_loss=best_val_loss,
            )
            print(f"  -> Saved best (val_loss={val_loss:.4f})")

        if epoch % args.save_interval == 0:
            trainer.save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"),
                epoch=epoch, best_val_loss=best_val_loss,
            )

    # ----------------------------------------------------------------
    # 5. Final save
    # ----------------------------------------------------------------
    print("[5/6] Saving final model...")
    final_path = os.path.join(args.output_dir, "tricoopdiff_zero.pt")
    trainer.save_checkpoint(
        final_path, epoch=args.epochs, best_val_loss=best_val_loss,
    )
    print(f"Done. Model saved to {final_path}")


def build_trainer(args, device, input_dim=None):
    """Build model, diffusion, and trainer."""
    from tricoopdiff.consensus import ConsensusCoopDiff, ConsensusFuse
    from tricoopdiff.expert_factory import build_diffdock_expert
    from tricoopdiff.sdpi_flow import SDPIFlow
    from tricoopdiff.consensus_train import ConsensusTrainer

    # Build frozen DiffDock-L expert from pretrained weights (self-contained
    # vendored core — no ../DiffDock dependency). num_conv_layers (11) and the
    # full CGModel architecture come from model_parameters.yml.
    frozen_expert = build_diffdock_expert(
        model_dir=args.expert_model_dir,
        device=str(device),
    )
    num_conv_layers = frozen_expert.num_conv_layers

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
    trifuse = ConsensusFuse(**model_config)

    model = ConsensusCoopDiff(frozen_expert, trifuse, num_conv_layers=num_conv_layers)

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


def _guarded_train_step(trainer, batch):
    """Abort immediately when a batch is numerically invalid."""
    from tricoopdiff.train import NonFiniteTrainingError

    try:
        return trainer.train_step(**batch)
    except NonFiniteTrainingError as error:
        raise RuntimeError(f"training aborted on first non-finite batch: {error}") from error


def prepare_source_batch(batch, global_step, training_arm, endpoint_store):
    """Attach the actual source and matching coupling for one training step."""
    if training_arm == 'A1':
        return {**batch, 'source': 0}
    from tricoopdiff.dual_init import mixed_source_for_step
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


def validate(trainer, val_loader, device, training_arm='A1', endpoint_store=None):
    """Run a deterministic validation pass without perturbing training RNG."""
    cuda_devices = []
    if device.type == 'cuda':
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(0)
            if cuda_devices:
                torch.cuda.manual_seed_all(0)
            np.random.seed(0)
            trainer.model.eval()
            losses = {'sdpi': [], 'bitseed': []}
            fallback_count = 0

            for batch in val_loader:
                try:
                    loss_dict = trainer.validate_step(**{**batch, 'source': 0})
                except FloatingPointError as error:
                    raise RuntimeError(f"validation failed: {error}") from error
                if not np.isfinite(loss_dict['loss_total']):
                    raise RuntimeError(
                        f"validation failed: case={batch.get('case', 'unknown')} "
                        "produced a non-finite loss"
                    )
                losses['sdpi'].append(loss_dict['loss_total'])
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
                        })
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
                    fallback_count += int(actual_source == 'sdpi')
    finally:
        np.random.set_state(numpy_state)
        trainer.model.train()
    by_source = {
        source: float(np.mean(values))
        for source, values in losses.items() if values
    }
    selection = float(np.mean(list(by_source.values()))) if by_source else float('nan')
    return {
        'selection_loss': selection, 'by_source': by_source,
        'source_counts': {source: len(values) for source, values in losses.items()},
        'bitseed_fallback_count': fallback_count,
    }


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

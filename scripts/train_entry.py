"""
TriCoopDiff training entry point.

Two-phase training:
  Phase 1: Interface score matching (branch FiLM + expert LoRA)
  Phase 2: PROTAC fine-tuning (all TriBrain heads, low LR, LoRA)

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


def parse_args():
    p = argparse.ArgumentParser(description="TriCoopDiff training")

    # Data
    p.add_argument("--train_data", required=True, help="Training data JSON")
    p.add_argument("--val_data", required=True, help="Validation data JSON")

    # Training
    p.add_argument("--output_dir", required=True, help="Checkpoint output directory")
    p.add_argument("--epochs", type=int, default=850)       # paper: 850 epochs
    p.add_argument(
        "--batch_size", type=int, default=1, choices=[1],
        help="Only B=1 sequential training is currently implemented",
    )
    p.add_argument("--lr", type=float, default=1e-3)         # paper: lr=1e-3, Plateau scheduler
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lora_rank", type=int, default=0,
                   help="Rank of independent sub1/sub2 expert adapters; 0 disables LoRA")
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_start_layer", type=int, default=0)
    p.add_argument("--lora_early_layers", type=int, default=4)
    p.add_argument("--lora_early_rank", type=int, default=2)
    p.add_argument("--lora_early_alpha", type=float, default=4.0)
    p.add_argument("--lora_lr", type=float, default=1e-5)
    p.add_argument("--lora_early_lr", type=float, default=5e-6)

    # Phase
    p.add_argument("--phase", type=int, default=1, choices=[1, 2],
                   help="1=interface score matching, 2=finetune auxiliary heads")
    p.add_argument("--resume", default=None, help="Resume from checkpoint")

    # Model config
    # TriBrain params
    p.add_argument("--ns", type=int, default=32, help="TriBrain scalar channels")
    p.add_argument("--nv", type=int, default=4, help="TriBrain vector channels")
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

    return p.parse_args()


def _load_score_model_args(expert_model_dir):
    """Load DiffDock's model_parameters.yml as an argparse.Namespace."""
    import yaml
    from argparse import Namespace
    with open(Path(expert_model_dir) / "model_parameters.yml") as f:
        return Namespace(**yaml.full_load(f))


def _load_entries(data_json):
    """Read an index and resolve portable or stale-checkout input paths."""
    with open(data_json) as f:
        index = json.load(f)
    entries = index.get("entries", []) if isinstance(index, dict) else index
    resolved = []
    for raw_entry in entries:
        entry = dict(raw_entry)
        for key in ("protein1", "protein2", "ligand"):
            path = Path(entry[key])
            if not path.is_absolute():
                path = _PROJECT_ROOT / path
            if not path.is_file():
                # Backward compatibility for indices created in another clone:
                # reconstruct the canonical in-repo case path from stable fields.
                path = (
                    _PROJECT_ROOT / "DeepTernary" / "output" / "protac22"
                    / entry["case"] / Path(entry[key]).name
                )
            entry[key] = str(path.resolve())
        resolved.append(entry)
    return resolved


def main():
    args = parse_args()
    device = torch.device(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

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

    # ----------------------------------------------------------------
    # 0. Featurize the first sample to derive the receptor feature dim
    #    (TriBrain's input_dim must match the featurized ternary nodes).
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

    # ----------------------------------------------------------------
    # 1. Build model
    # ----------------------------------------------------------------
    print("[1/6] Building model...")
    model, diffusion, trainer = build_trainer(args, device, input_dim=input_dim)

    # ----------------------------------------------------------------
    # 2. Load data
    # ----------------------------------------------------------------
    print("[2/6] Loading data...")
    train_loader = TernaryDataLoader(
        args.train_data, score_model_args,
        shuffle=True, device=str(device), cache_dir=args.cache_dir,
        skip_on_error=False,
    )
    val_loader = TernaryDataLoader(
        args.val_data, score_model_args,
        shuffle=False, device=str(device), cache_dir=args.cache_dir,
        skip_on_error=False,
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
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_losses = []
        n_skipped = 0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            loss_dict = _guarded_train_step(trainer, batch)
            if loss_dict is None:
                n_skipped += 1
                print(f"  Epoch {epoch} | Batch {batch_idx} | SKIP non-finite loss",
                      flush=True)
                continue
            train_losses.append(loss_dict)

            if batch_idx % args.log_interval == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx} | "
                      f"loss={loss_dict['loss_total']:.4f} | "
                      f"e3l=(tr={loss_dict['loss_e3l_tr']:.4f}, "
                      f"rot={loss_dict['loss_e3l_rot']:.4f}) | "
                      f"pol=(tr={loss_dict['loss_pol_tr']:.4f}, "
                      f"rot={loss_dict['loss_pol_rot']:.4f}) | "
                      f"tor={loss_dict['loss_score_tor']:.4f}")

        if n_skipped:
            raise RuntimeError(
                f"epoch {epoch}: {n_skipped} training batches were non-finite; "
                "refusing to validate or save a partial-data checkpoint"
            )
        if not train_losses:
            raise RuntimeError(f"epoch {epoch}: training loader produced no batches")

        # Validate
        val_loss = validate(trainer, val_loader, device)
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

        # Save best & periodic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> Saved best (val_loss={val_loss:.4f})")

        if epoch % args.save_interval == 0:
            trainer.save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
            )

    # ----------------------------------------------------------------
    # 5. Final save
    # ----------------------------------------------------------------
    print("[5/6] Saving final model...")
    final_path = os.path.join(
        args.output_dir,
        "tricoopdiff_zero.pt" if args.phase == 1 else "tricoopdiff_ft.pt"
    )
    trainer.save_checkpoint(final_path)
    print(f"Done. Model saved to {final_path}")


def build_trainer(args, device, input_dim=None):
    """Build model, diffusion, and trainer."""
    from tricoopdiff import TriCoopDiff
    from tricoopdiff.tribrain import TriBrain
    from tricoopdiff.expert_factory import build_diffdock_expert
    from tricoopdiff.diffusion import InterfaceDiffusion
    from tricoopdiff.train import TriCoopDiffTrainer

    # Build frozen DiffDock-L expert from pretrained weights (self-contained
    # vendored core — no ../DiffDock dependency). num_conv_layers (11) and the
    # full CGModel architecture come from model_parameters.yml.
    frozen_expert = build_diffdock_expert(
        model_dir=args.expert_model_dir,
        device=str(device),
    )
    if getattr(args, 'lora_rank', 0) > 0:
        from tricoopdiff.lora import inject_expert_lora
        count = inject_expert_lora(
            frozen_expert, rank=args.lora_rank, alpha=args.lora_alpha,
            start_layer=args.lora_start_layer,
            early_layers=args.lora_early_layers,
            early_rank=args.lora_early_rank,
            early_alpha=args.lora_early_alpha,
        )
        print(f"  expert LoRA: {count} linear layers, early rank="
              f"{args.lora_early_rank}, late rank={args.lora_rank}, "
              f"start_layer={args.lora_start_layer}")
    num_conv_layers = frozen_expert.num_conv_layers

    # input_dim must match the ternary node feature dim, which is the receptor
    # feature dim (1281 = 1 amino-acid index + 1280 ESM2 for DiffDock-L). It is
    # derived at runtime from the featurized data, never hardcoded.
    if input_dim is None:
        input_dim = args.node_dim
        print(f"[warn] input_dim not provided; defaulting to node_dim={input_dim}")

    # Build TriBrain (the only trainable module). film_dim MUST equal the expert's
    # scalar channel count (ns=48 for DiffDock-L) — it is independent of TriBrain's
    # own internal width (args.ns).
    tribrain = TriBrain(
        node_dim=args.node_dim,
        input_dim=input_dim,
        ns=args.ns,
        nv=args.nv,
        num_transformer_layers=args.num_transformer_layers,
        num_conv_layers_per_expert=num_conv_layers,
        film_dim=frozen_expert.ns,
    )

    model = TriCoopDiff(frozen_expert, tribrain, num_conv_layers=num_conv_layers)

    # Build diffusion
    diffusion = InterfaceDiffusion(inference_steps=20)

    # Build trainer
    trainer = TriCoopDiffTrainer(
        model=model,
        diffusion=diffusion,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lora_lr=getattr(args, 'lora_lr', 1e-5),
        lora_early_lr=getattr(args, 'lora_early_lr', 5e-6),
        lora_early_layers=getattr(args, 'lora_early_layers', 4),
        device=device,
    )

    return model, diffusion, trainer


def _guarded_train_step(trainer, batch):
    """Return ``None`` without stepping when a batch is numerically invalid."""
    from tricoopdiff.train import NonFiniteTrainingError

    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    try:
        loss_dict = trainer.compute_loss(**batch)
    except NonFiniteTrainingError as error:
        print(f"[train] SKIP {error}", flush=True)
        return None
    total = loss_dict['loss_total']
    if not torch.isfinite(total):
        print(f"[train] SKIP case={batch.get('case', 'unknown')}: non-finite total loss", flush=True)
        return None
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainer.model.tribrain.parameters(), max_norm=1.0,
    )
    if not torch.isfinite(grad_norm):
        trainer.optimizer.zero_grad(set_to_none=True)
        print(
            f"[train] SKIP case={batch.get('case', 'unknown')}: non-finite gradients",
            flush=True,
        )
        return None
    trainer.optimizer.step()
    trainer.global_step += 1
    result = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    result['lr'] = trainer.optimizer.param_groups[0]['lr']
    return result


def validate(trainer, val_loader, device):
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
            total_loss = 0.0
            n_batches = 0

            for batch in val_loader:
                try:
                    loss_dict = trainer.validate_step(**batch)
                except FloatingPointError as error:
                    raise RuntimeError(f"validation failed: {error}") from error
                if not np.isfinite(loss_dict['loss_total']):
                    raise RuntimeError(
                        f"validation failed: case={batch.get('case', 'unknown')} "
                        "produced a non-finite loss"
                    )
                total_loss += loss_dict['loss_total']
                n_batches += 1
    finally:
        np.random.set_state(numpy_state)
        trainer.model.train()
    return total_loss / n_batches if n_batches else float('nan')


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
                 cache_dir=None, skip_on_error=False):
        self.entries = _load_entries(data_json)
        self.score_model_args = score_model_args
        self.shuffle = shuffle
        self.device = device
        self.cache_dir = cache_dir
        self.skip_on_error = skip_on_error
        self.skipped = []

    def __iter__(self):
        indices = list(range(len(self.entries)))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in indices:
            entry = self.entries[i]
            try:
                yield self._load_one(entry)
            except Exception as e:
                if not self.skip_on_error:
                    raise
                self.skipped.append(entry.get("case", str(i)))
                print(f"[data] SKIP {entry.get('case', i)}: {e}", flush=True)

    def __len__(self):
        return len(self.entries)

    def _load_one(self, entry):
        e3_data, poi_data, ligand_data, _ = featurize_complex(
            entry["case"], entry["protein1"], entry["protein2"], entry["ligand"],
            self.score_model_args, self.device, cache_dir=self.cache_dir,
        )
        return {
            'case': entry['case'],
            'e3_data': e3_data,
            'poi_data': poi_data,
            'ligand_data': ligand_data,
        }


if __name__ == "__main__":
    main()

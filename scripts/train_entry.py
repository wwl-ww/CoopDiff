"""
TriCoopDiff training entry point.

Two-phase training:
  Phase 1: General ternary pretraining (FiLM + SE3 heads only)
  Phase 2: PROTAC fine-tuning (all TriBrain heads, low LR, LoRA)

Current scope: **B=1 sequential training** on real PROTAC complexes. The
featurized data is produced on the fly from a JSON index (``make_protac_index.py``)
via ``featurize_common.featurize_complex`` and cached to disk so subsequent epochs
skip the expensive ESM2 embedding step. Padded B>1 batching is a follow-up.

Usage:
    # Build the index once
    python scripts/make_protac_index.py

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
    p.add_argument("--batch_size", type=int, default=16)     # paper: batch_size=16
    p.add_argument("--lr", type=float, default=1e-3)         # paper: lr=1e-3, Plateau scheduler
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Phase
    p.add_argument("--phase", type=int, default=1, choices=[1, 2],
                   help="1=pretrain (FiLM+SE3), 2=finetune (all heads)")
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
    p.add_argument("--receptor_feat_dim", type=int, default=1281,
                   help="Fallback receptor feature dim (1281 for DiffDock-L) if "
                        "the first-sample featurization cannot determine it")
    p.add_argument("--cache_dir", default="results/featurize_cache",
                   help="Dir for on-disk featurization cache (per-case .pt)")

    # Hardware
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=10)

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
    os.makedirs(args.output_dir, exist_ok=True)

    # SO(3)/torus cache lookup resolves relative to CWD; chdir to project root.
    os.chdir(_PROJECT_ROOT)

    # Save run config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[TriCoopDiff] Training Phase {args.phase}")
    print(f"  Train data: {args.train_data}")
    print(f"  Val data: {args.val_data}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # ----------------------------------------------------------------
    # 0. Featurize the first sample to derive the receptor feature dim
    #    (TriBrain's input_dim must match the featurized ternary nodes).
    # ----------------------------------------------------------------
    entries = _load_entries(args.train_data)
    if not entries:
        raise SystemExit(f"FATAL: no entries in {args.train_data}")

    print("[0/6] Determining receptor feature dim from first sample...")
    score_model_args = _load_score_model_args(args.expert_model_dir)
    input_dim = None
    first = entries[0]
    try:
        _, _, _, input_dim = featurize_complex(
            first["case"], first["protein1"], first["protein2"], first["ligand"],
            score_model_args, str(device), cache_dir=args.cache_dir,
        )
        print(f"  receptor_feat_dim={input_dim} (from {first['case']})")
    except Exception as e:
        print(f"[warn] first-sample featurize failed ({e}); "
              f"falling back to --receptor_feat_dim={args.receptor_feat_dim}")
    if input_dim is None:
        input_dim = args.receptor_feat_dim

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
    )
    val_loader = TernaryDataLoader(
        args.val_data, score_model_args,
        shuffle=False, device=str(device), cache_dir=args.cache_dir,
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
                      f"score={loss_dict['loss_score']:.4f} "
                      f"(tr={loss_dict['loss_score_tr']:.4f}, "
                      f"rot={loss_dict['loss_score_rot']:.4f}, "
                      f"tor={loss_dict['loss_score_tor']:.4f}) | "
                      f"se3={loss_dict['loss_se3']:.4f} "
                      f"(tr={loss_dict['loss_se3_tr']:.4f}, "
                      f"rot={loss_dict['loss_se3_rot']:.4f}) | "
                      f"phys={loss_dict['loss_phys']:.4f}")

        # Validate
        val_loss = validate(trainer, val_loader, device)
        trainer.scheduler_step()

        # Log
        train_loss_avg = np.mean([d['loss_total'] for d in train_losses]) if train_losses else float('nan')
        train_se3_avg = np.mean([d['loss_se3'] for d in train_losses]) if train_losses else float('nan')
        train_se3_tr_avg = np.mean([d['loss_se3_tr'] for d in train_losses]) if train_losses else float('nan')
        train_se3_rot_avg = np.mean([d['loss_se3_rot'] for d in train_losses]) if train_losses else float('nan')
        train_phys_avg = np.mean([d['loss_phys'] for d in train_losses]) if train_losses else float('nan')
        dt = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss_avg:.4f} | "
              f"se3={train_se3_avg:.4f} "
              f"(tr={train_se3_tr_avg:.4f}, rot={train_se3_rot_avg:.4f}) | "
              f"phys={train_phys_avg:.4f} | "
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
    from tricoopdiff.diffusion import MultiRateDiffusion
    from tricoopdiff.train import TriCoopDiffTrainer

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
    diffusion = MultiRateDiffusion(inference_steps=20)

    # Build trainer
    trainer = TriCoopDiffTrainer(
        model=model,
        diffusion=diffusion,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    return model, diffusion, trainer


def _guarded_train_step(trainer, batch):
    """``trainer.train_step`` with a non-finite loss guard.

    Skips (returns None) instead of stepping on NaN/Inf losses, so a handful of
    known-NaN complexes can't poison the optimizer.
    """
    loss_dict = trainer.compute_loss(**batch)
    total = loss_dict['loss_total']
    if not torch.isfinite(total):
        return None
    trainer.optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.tribrain.parameters(), max_norm=1.0)
    trainer.optimizer.step()
    trainer.global_step += 1
    result = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    result['lr'] = trainer.optimizer.param_groups[0]['lr']
    return result


def validate(trainer, val_loader, device):
    """Run validation pass — no backward, no optimizer."""
    trainer.model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        loss_dict = trainer.validate_step(**batch)
        if not np.isfinite(loss_dict['loss_total']):
            continue
        total_loss += loss_dict['loss_total']
        n_batches += 1

    trainer.model.train()
    return total_loss / max(n_batches, 1)


class TernaryDataLoader:
    """
    Real data loader for PROTAC ternary complexes.

    Reads the JSON index written by ``scripts/make_protac_index.py``::
        {"entries": [
            {"case": ..., "protein1": ".../protein1.pdb",
             "protein2": ".../protein2.pdb", "ligand": ".../ligand.sdf"}]}

    Each entry is featurized on the fly via ``featurize_complex`` (with an
    on-disk cache so subsequent epochs skip ESM2). The loader yields a **B=1**
    batch dict whose keys match ``TriCoopDiffTrainer.compute_loss``::
        {'e3_data': ..., 'poi_data': ..., 'ligand_data': ..., 't_se3_data': None}

    Featurization failures (e.g. the known NaN cases) are skipped with a log
    rather than aborting training.
    """

    def __init__(self, data_json, score_model_args, shuffle=True, device="cuda",
                 cache_dir=None, skip_on_error=True):
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
            'e3_data': e3_data,
            'poi_data': poi_data,
            'ligand_data': ligand_data,
            't_se3_data': None,
        }


if __name__ == "__main__":
    main()

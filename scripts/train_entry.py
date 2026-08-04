"""
TriCoopDiff training entry point.

Two-phase training:
  Phase 1: General ternary pretraining (FiLM + SE3 heads only)
  Phase 2: PROTAC fine-tuning (all TriBrain heads, low LR, LoRA)

Usage:
    # Phase 1: pretrain on general ternary data
    python scripts/train_entry.py \
        --train_data data/ternary_train.json \
        --val_data data/ternary_val.json \
        --output_dir checkpoints/phase1 \
        --epochs 100 --batch_size 32 --lr 1e-4

    # Phase 2: fine-tune on PROTAC data
    python scripts/train_entry.py \
        --train_data data/protac_benchmark.json \
        --val_data data/protac_benchmark_test.json \
        --output_dir checkpoints/phase2 \
        --epochs 50 --batch_size 8 --lr 2e-5 \
        --resume checkpoints/phase1/tricoopdiff_zero.pt \
        --phase 2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="TriCoopDiff training")

    # Data
    p.add_argument("--train_data", required=True, help="Training data JSON")
    p.add_argument("--val_data", required=True, help="Validation data JSON")

    # Training
    p.add_argument("--output_dir", required=True, help="Checkpoint output directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Phase
    p.add_argument("--phase", type=int, default=1, choices=[1, 2],
                   help="1=pretrain (FiLM+SE3), 2=finetune (all heads)")
    p.add_argument("--resume", default=None, help="Resume from checkpoint")

    # Model config
    p.add_argument("--ns", type=int, default=32, help="Scalar channels")
    p.add_argument("--nv", type=int, default=4, help="Vector channels")
    p.add_argument("--num_transformer_layers", type=int, default=3)
    p.add_argument("--num_conv_layers", type=int, default=6)
    p.add_argument("--node_dim", type=int, default=128)

    # Hardware
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save run config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[TriCoopDiff] Training Phase {args.phase}")
    print(f"  Train data: {args.train_data}")
    print(f"  Val data: {args.val_data}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # ----------------------------------------------------------------
    # 1. Build model
    # ----------------------------------------------------------------
    print("[1/5] Building model...")
    model, diffusion, trainer = build_trainer(args, device)

    # ----------------------------------------------------------------
    # 2. Load data
    # ----------------------------------------------------------------
    print("[2/5] Loading data...")
    train_loader = TernaryDataLoader(
        args.train_data, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )
    val_loader = TernaryDataLoader(
        args.val_data, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    # ----------------------------------------------------------------
    # 3. Resume if specified
    # ----------------------------------------------------------------
    if args.resume:
        print(f"[3/5] Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)
    else:
        print("[3/5] Training from scratch...")

    # Set phase (freeze/unfreeze heads)
    trainer.set_phase(args.phase)

    # ----------------------------------------------------------------
    # 4. Training loop
    # ----------------------------------------------------------------
    print(f"[4/5] Training {args.epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        train_losses = []
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            loss_dict = trainer.train_step(**batch)
            train_losses.append(loss_dict)

            if batch_idx % args.log_interval == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx} | "
                      f"loss={loss_dict['loss_total']:.4f} | "
                      f"score={loss_dict['loss_score']:.4f}")

        # Validate
        val_loss = validate(trainer, val_loader, device)
        trainer.scheduler_step()

        # Log
        train_loss_avg = np.mean([d['loss_total'] for d in train_losses])
        dt = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss_avg:.4f} | "
              f"val_loss={val_loss:.4f} | "
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
    print("[5/5] Saving final model...")
    final_path = os.path.join(
        args.output_dir,
        "tricoopdiff_zero.pt" if args.phase == 1 else "tricoopdiff_ft.pt"
    )
    trainer.save_checkpoint(final_path)
    print(f"Done. Model saved to {final_path}")


def build_trainer(args, device):
    """Build model, diffusion, and trainer."""
    from tricoopdiff import TriCoopDiff
    from tricoopdiff.tribrain import TriBrain
    from tricoopdiff.expert import FrozenDiffDock
    from tricoopdiff.diffusion import MultiRateDiffusion
    from tricoopdiff.train import TriCoopDiffTrainer

    # Build TriBrain
    tribrain = TriBrain(
        node_dim=args.node_dim,
        ns=args.ns,
        nv=args.nv,
        num_transformer_layers=args.num_transformer_layers,
        num_conv_layers_per_expert=args.num_conv_layers,
    )

    # Build frozen expert
    diffdock_root = str(Path(__file__).parent.parent / "DiffDock")
    sys.path.insert(0, diffdock_root)
    from models.cg_model import CGModel

    cg_model = CGModel(
        t_to_sigma=lambda tr, rot, tor: (
            torch.tensor(0.1) * (30.0 / 0.1) ** tr,
            torch.tensor(0.1) * (1.6 / 0.1) ** rot,
            torch.tensor(0.2) * (2.0 / 0.2) ** tor,
        ),
        device=device,
        timestep_emb_func=_make_timestep_emb(),
        ns=args.ns, nv=args.nv,
        num_conv_layers=args.num_conv_layers,
        scale_by_sigma=True,
        separate_noise_schedule=False,
    )

    frozen_expert = FrozenDiffDock(cg_model, num_conv_layers=args.num_conv_layers)
    model = TriCoopDiff(frozen_expert, tribrain, num_conv_layers=args.num_conv_layers)

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


def _make_timestep_emb():
    """Minimal timestep embedding function."""
    import torch.nn as nn
    emb = nn.Embedding(1, 32)  # placeholder
    def fn(t):
        return emb(torch.zeros(1, dtype=torch.long, device=t.device if hasattr(t, 'device') else 'cpu'))
    return fn


def validate(trainer, val_loader, device):
    """Run validation pass."""
    trainer.model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            loss_dict = trainer.train_step(**batch)  # same step, no optimizer.step
            total_loss += loss_dict['loss_total']
            n_batches += 1

    trainer.model.train()
    return total_loss / max(n_batches, 1)


class TernaryDataLoader:
    """
    Minimal data loader for ternary complex data.

    Reads a JSON index file listing paths to processed HeteroData pickle files.
    In production, use diffdock's DataLoader + our ternary wrapper.
    """

    def __init__(self, data_json, batch_size=32, shuffle=True, num_workers=0):
        with open(data_json) as f:
            self.entries = json.load(f)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._idx = 0

    def __iter__(self):
        indices = list(range(len(self.entries)))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch_entries = [self.entries[i] for i in batch_indices]
            yield self._collate(batch_entries)

    def __len__(self):
        return max(1, len(self.entries) // self.batch_size)

    def _collate(self, entries):
        """Collate entries into batch dict. Override for actual data format."""
        # Placeholder: real implementation loads HeteroData from pickle/disk
        # For now returns dummy tensors matching expected shapes
        B = len(entries)
        return {
            'e3_data': {
                'coords': torch.randn(B, 200, 3),
                'features': torch.randn(B, 200, 1),
            },
            'poi_data': {
                'coords': torch.randn(B, 300, 3),
                'features': torch.randn(B, 300, 1),
            },
            'ligand_data': {
                'coords': torch.randn(B, 100, 3),
                'features': torch.randn(B, 100, 1),
            },
            't_se3_data': {
                'tr': torch.randn(B, 3),
                'rot': torch.eye(3).unsqueeze(0).repeat(B, 1, 1),
            },
        }


if __name__ == "__main__":
    main()

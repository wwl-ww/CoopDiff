"""Reusable model runtime for in-process evaluation of many ternary cases."""

import json
from argparse import Namespace
from pathlib import Path

import torch
import yaml

from evaluation_common import validate_checkpoint_payload
from featurize_common import featurize_complex
from tricoopdiff.diffusion import MultiRateDiffusion, TernaryInitializer
from tricoopdiff.evaluate import evaluate_model
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.lora import (
    inject_expert_lora,
    load_expert_lora_state_dict,
)
from tricoopdiff.model import TriCoopDiff
from tricoopdiff.tribrain import TriBrain


class EvaluationRunner:
    """Load model weights once and evaluate multiple same-feature-schema cases."""

    def __init__(self, project_root, checkpoint, device="cuda",
                 model_dir="checkpoints/score_model",
                 expert_checkpoint="best_ema_inference_epoch_model.pt"):
        self.root = Path(project_root).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = self.root / self.checkpoint_path
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        validate_checkpoint_payload(self.checkpoint)

        self.model_dir = Path(model_dir)
        if not self.model_dir.is_absolute():
            self.model_dir = self.root / self.model_dir
        with (self.model_dir / "model_parameters.yml").open() as handle:
            self.score_model_args = Namespace(**yaml.full_load(handle))
        self.expert = build_diffdock_expert(
            model_dir=str(self.model_dir), device=str(self.device),
            ckpt=expert_checkpoint,
            num_conv_layers=self.score_model_args.num_conv_layers,
        )
        inject_expert_lora(
            self.expert, **self.checkpoint["expert_lora_config"],
        )
        load_expert_lora_state_dict(
            self.expert, self.checkpoint["expert_lora_state_dict"],
        )
        self.model = None
        self.input_dim = None
        self.diffusion = MultiRateDiffusion()
        self.initializer = TernaryInitializer()

    def _ensure_model(self, input_dim):
        if self.model is not None:
            if input_dim != self.input_dim:
                raise ValueError(
                    f"receptor feature dimension changed from {self.input_dim} "
                    f"to {input_dim}"
                )
            return
        tribrain = TriBrain(
            node_dim=128, input_dim=input_dim,
            num_conv_layers_per_expert=self.expert.num_conv_layers,
            film_dim=self.expert.ns,
        )
        self.model = TriCoopDiff(
            frozen_expert=self.expert,
            tribrain=tribrain,
            num_conv_layers=self.expert.num_conv_layers,
            use_film=True,
        ).to(self.device)
        self.model.tribrain.load_state_dict(self.checkpoint["tribrain_state_dict"])
        self.model.eval()
        self.input_dim = input_dim

    def evaluate_case(self, case, paths, cache_dir, native_complex_pdb=None,
                      n_walkers=32, steps=20, seed=0,
                      init_mode="coupled5a", t_anchor=None, freeze_poi=False,
                      debug_trace=False, debug_seed=0):
        """Featurize and evaluate one case with the resident model."""
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        e3_data, poi_data, ligand_data, input_dim = featurize_complex(
            case, str(paths["protein1"]), str(paths["protein2"]),
            str(paths["ligand"]), self.score_model_args, str(self.device),
            cache_dir=str(cache_dir),
        )
        self._ensure_model(input_dim)
        report = evaluate_model(
            self.model, e3_data, poi_data, ligand_data, self.diffusion,
            initializer=self.initializer, device=str(self.device),
            n_walkers=n_walkers, steps=steps,
            native_complex_pdb=(str(native_complex_pdb)
                                if native_complex_pdb else None),
            freeze_poi=freeze_poi, init_mode=init_mode, t_anchor=t_anchor,
            debug_trace=debug_trace, debug_seed=debug_seed,
            deepternary_dockq_paths={
                "protein1_pdb": str(paths["protein1"]),
                "protein2_pdb": str(paths["protein2"]),
                "ligand_pdb": str(paths["ligand"]),
                "deepternary_root": str(self.root / "DeepTernary"),
            },
        )
        report["case"] = case
        report["run_config"] = {
            "seed": seed,
            "n_walkers": n_walkers,
            "steps": (steps if steps is not None
                      else self.diffusion.inference_steps),
            "freeze_poi": freeze_poi,
            "init_mode": init_mode,
            "t_anchor": (t_anchor if t_anchor is not None else
                         (0.7 if init_mode == "anchor_local" else 1.0)),
            "tribrain_checkpoint": str(self.checkpoint_path.resolve()),
            "expert_model_dir": str(self.model_dir),
            "dockq_implementation": (
                "DeepTernary/DockQ/dockq_util.py::cal_dockq"
                if native_complex_pdb else None
            ),
        }
        return report


def write_report_atomic(report, destination):
    """Atomically publish one completed case report."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)

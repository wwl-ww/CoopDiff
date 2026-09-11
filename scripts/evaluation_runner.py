"""Reusable model runtime for in-process evaluation of many ternary cases."""

import json
import os
from argparse import Namespace
from pathlib import Path

import torch
import yaml

from evaluation_common import (
    evaluation_input_signature, sha256_file, validate_checkpoint_payload,
)
from featurize_common import featurize_complex
from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.evaluate import evaluate_model
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.consensus import ConsensusCoopDiff, ConsensusFuse
from tricoopdiff.reproducibility import configure_inference_reproducibility
from tricoopdiff.checkpoint_contract import validate_expert_base_identity
from tricoopdiff.diffdock.utils.torus import (
    SCORE_NORM_VERSION, validate_score_norm_version,
)


class EvaluationRunner:
    """Load model weights once and evaluate multiple same-feature-schema cases."""

    def __init__(self, project_root, checkpoint, device="cuda",
                 model_dir="checkpoints/score_model",
                 expert_checkpoint="best_ema_inference_epoch_model.pt",
                 deepternary_root=None, deterministic=True,
                 conformer_policy="etkdgv3_random",
                 allow_experimental_checkpoint=False):
        self.reproducibility = configure_inference_reproducibility(deterministic)
        self.reproducibility['torus_score_norm_version'] = SCORE_NORM_VERSION
        self.root = Path(project_root).resolve()
        self.deepternary_root = (
            self.root / (deepternary_root or 'DeepTernary')
        ).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = self.root / self.checkpoint_path
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.experimental_checkpoint = (
            self.checkpoint.get('architecture') == 'residual_consensus_factorial_v1'
        )
        if self.experimental_checkpoint:
            if not allow_experimental_checkpoint:
                raise ValueError(
                    'experimental factorial checkpoint requires explicit '
                    'allow_experimental_checkpoint=True'
                )
            contract = self.checkpoint.get('experiment_contract', {})
            if self.checkpoint.get('checkpoint_version') != 1:
                raise ValueError('unsupported experimental checkpoint version')
            if contract.get('chemistry') != 'ccd' or contract.get('torsion') != 'single_bond':
                raise ValueError('experimental checkpoint chemistry/torsion contract differs')
            if contract.get('adaptation') != 'frozen':
                raise ValueError('only frozen-expert experimental checkpoints are supported')
            if self.checkpoint.get('expert_lora_config') is not None:
                raise ValueError('frozen experimental checkpoint unexpectedly contains expert LoRA')
            if 'ema_trifuse_state_dict' not in self.checkpoint:
                raise ValueError('experimental checkpoint has no EMA TriFuse weights')
        else:
            validate_checkpoint_payload(self.checkpoint)
        self.reproducibility['checkpoint_torus_score_norm_version'] = (
            validate_score_norm_version(self.checkpoint['runtime_config'])
        )
        self.activation_clamp = float(
            self.checkpoint["runtime_config"]["activation_clamp"]
        )
        os.environ["NAN_CLAMP"] = str(self.activation_clamp)
        self.activation_clamp_mode = self.checkpoint['runtime_config'].get(
            'activation_clamp_mode', 'componentwise')
        os.environ['NAN_CLAMP_MODE'] = self.activation_clamp_mode

        self.model_dir = Path(model_dir)
        if not self.model_dir.is_absolute():
            self.model_dir = self.root / self.model_dir
        with (self.model_dir / "model_parameters.yml").open() as handle:
            self.score_model_args = Namespace(**yaml.full_load(handle))
        self.expert_config_sha256 = sha256_file(self.model_dir / "model_parameters.yml")
        self.expert = build_diffdock_expert(
            model_dir=str(self.model_dir), device=str(self.device),
            ckpt=expert_checkpoint,
            num_conv_layers=self.score_model_args.num_conv_layers,
        )
        self.reproducibility['checkpoint_expert_base_identity'] = (
            validate_expert_base_identity(self.checkpoint, self.expert)
        )
        self.model = None
        self.input_dim = None
        self.diffusion = SDPIFlow()
        self.initializer = TernaryInitializer(
            no_torsion=getattr(self.score_model_args, "no_torsion", False),
            conformer_policy=conformer_policy,
        )

    def _ensure_model(self, input_dim):
        if self.model is not None:
            if input_dim != self.input_dim:
                raise ValueError(
                    f"receptor feature dimension changed from {self.input_dim} "
                    f"to {input_dim}"
                )
            return
        if self.checkpoint['model_config']['input_dim'] != input_dim:
            raise ValueError('checkpoint input feature dimension mismatch')
        trifuse = ConsensusFuse(**self.checkpoint['model_config'])
        if self.experimental_checkpoint:
            from tricoopdiff.consensus_experiment import ExperimentalConsensus

            self.model = ExperimentalConsensus(
                self.expert, trifuse, self.checkpoint['expert_lora_config'],
            ).to(self.device)
        else:
            self.model = ConsensusCoopDiff(
                frozen_expert=self.expert,
                trifuse=trifuse,
                num_conv_layers=self.expert.num_conv_layers,
                use_film=False,
            ).to(self.device)
        self.model.trifuse.load_state_dict(self.checkpoint["ema_trifuse_state_dict"])
        self.model.eval()
        self.input_dim = input_dim

    def evaluate_case(self, case, paths, cache_dir, native_complex_pdb=None,
                      n_walkers=32, steps=20, seed=0,
                      init_mode="sdpi", t_anchor=None, freeze_poi=False,
                      debug_trace=False, debug_seed=0,
                      translation_velocity_scale=1.0,
                      rotation_velocity_scale=1.0,
                      return_rank_features=False):
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
            seed=seed,
            native_complex_pdb=(str(native_complex_pdb)
                                if native_complex_pdb else None),
            freeze_poi=freeze_poi, init_mode=init_mode, t_anchor=t_anchor,
            debug_trace=debug_trace, debug_seed=debug_seed,
            translation_velocity_scale=translation_velocity_scale,
            rotation_velocity_scale=rotation_velocity_scale,
            return_rank_features=return_rank_features,
            deepternary_dockq_paths={
                "protein1_pdb": str(paths["protein1"]),
                "protein2_pdb": str(paths["protein2"]),
                "ligand_pdb": str(paths["ligand"]),
                "deepternary_root": str(self.deepternary_root),
                **{key + '_pdb': str(paths[key]) for key in
                   ('metric_protein1', 'metric_protein2') if key in paths},
            },
        )
        report["case"] = case
        report["run_config"] = {
            **self.reproducibility,
            "conformer_policy": self.initializer.conformer_policy,
            "translation_velocity_scale": float(translation_velocity_scale),
            "rotation_velocity_scale": float(rotation_velocity_scale),
            "seed": seed,
            "n_walkers": n_walkers,
            "steps": (steps if steps is not None
                      else self.diffusion.inference_steps),
            "freeze_poi": freeze_poi,
            "init_mode": init_mode,
            "initialization_version": "sdpi_v1",
            "generative_process": "sdpi_flow_v1",
            "t_anchor": (t_anchor if t_anchor is not None else
                         1.0),
            "trifuse_checkpoint": str(self.checkpoint_path.resolve()),
            "checkpoint_schema": (
                "experimental_factorial_v1_ema" if self.experimental_checkpoint
                else "production_v18_ema"
            ),
            "rank_features_saved": bool(return_rank_features),
            "checkpoint_sha256": self.checkpoint_sha256,
            "expert_checkpoint_sha256": self.expert.provenance["score_checkpoint_sha256"],
            "expert_config_sha256": self.expert_config_sha256,
            "input_signature": evaluation_input_signature(paths, native_complex_pdb),
            "activation_clamp": self.activation_clamp,
            "activation_clamp_mode": self.activation_clamp_mode,
            "dockq_reconstruction": (
                "deepternary_bound_template_rigid_motion"
                if 'metric_protein1' in paths else "input_allatom_rigid_motion"
            ),
            "expert_model_dir": str(self.model_dir),
            "deepternary_root": str(self.deepternary_root),
            "dockq_source_sha256": sha256_file(self.deepternary_root / 'DockQ/dockq_util.py'),
            "dockq_binary_sha256": sha256_file(self.deepternary_root / 'DockQ/fnat'),
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

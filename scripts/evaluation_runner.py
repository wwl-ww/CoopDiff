"""Reusable model runtime for in-process evaluation of many ternary cases."""

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
from tricoopdiff.models.expert import FrozenFABind, build_diffdock_expert
from tricoopdiff.models.assembly import (
    DIRECT_RANK_FEATURE_STATE, LEGACY_RANK_FEATURE_STATE,
    TernaryAssemblyModel, TernaryAssembly,
)
from tricoopdiff.runtime import configure_inference_reproducibility, atomic_json as write_report_atomic
from tricoopdiff.models.expert import validate_expert_base_identity
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
                 allow_experimental_checkpoint=False,
                 allow_conditioned_checkpoint=False, expert_source_dir=None,
                 featurizer_model_dir='checkpoints/score_model'):
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
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.experimental_checkpoint = (
            self.checkpoint.get('architecture') == 'residual_consensus_factorial_v1'
        )
        self.conditioned_checkpoint = (
            self.checkpoint.get('format') == 'conditioned_consensus_training_v1'
        )
        if self.conditioned_checkpoint:
            raise ValueError('retired condition-token prototype checkpoint; use the nodewise assembly model')
        elif self.experimental_checkpoint:
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
                raise ValueError('adapted-expert experimental checkpoints are unsupported')
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
        self.activation_clamp_mode = self.checkpoint['runtime_config'].get(
            'activation_clamp_mode', 'componentwise')

        self.model_dir = Path(model_dir)
        if not self.model_dir.is_absolute():
            self.model_dir = self.root / self.model_dir
        backend = self.checkpoint.get('model_config', {}).get('expert_backend', 'diffdock')
        feature_dir = self.model_dir if backend == 'diffdock' else Path(featurizer_model_dir)
        if not feature_dir.is_absolute():
            feature_dir = self.root / feature_dir
        with (feature_dir / "model_parameters.yml").open() as handle:
            self.score_model_args = Namespace(**yaml.full_load(handle))
        self.expert_config_sha256 = sha256_file(feature_dir / "model_parameters.yml")
        if backend == 'fabind':
            if not expert_source_dir:
                raise ValueError('FABind evaluation requires its explicit official source directory')
            self.expert = FrozenFABind(expert_source_dir, self.model_dir, str(self.device))
        elif backend == 'diffdock':
            self.expert = build_diffdock_expert(
                model_dir=str(self.model_dir), device=str(self.device),
                ckpt=expert_checkpoint, num_conv_layers=self.score_model_args.num_conv_layers)
        else:
            raise ValueError(f'evaluation backend {backend!r} is not supported by this entrypoint')
        from tricoopdiff.models.expert import configure_expert_policy
        configure_expert_policy(self.expert, self.activation_clamp, self.activation_clamp_mode)
        if backend == 'diffdock' and self.checkpoint.get('model_config', {}).get('model_profile') == 'ternary_endpoint_v14':
            self.expert.model.smooth_cross_edges = True
        if self.checkpoint.get('checkpoint_version', 0) >= 24:
            from tricoopdiff.models.expert import expert_runtime_identity
            if self.checkpoint['runtime_config'].get('expert_runtime_identity') != expert_runtime_identity(self.expert):
                raise ValueError('evaluation expert runtime differs from the training checkpoint')
        self.reproducibility['checkpoint_expert_base_identity'] = (
            validate_expert_base_identity(self.checkpoint, self.expert)
        )
        self.model = None
        self.input_dim = None
        self.diffusion = SDPIFlow(inference_steps=self.checkpoint['runtime_config'].get('inference_steps', 20))
        self.initializer = TernaryInitializer(
            no_torsion=bool(getattr(self.expert.model, 'no_torsion', False)),
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
        trifuse = TernaryAssembly(**self.checkpoint['model_config'])
        if self.experimental_checkpoint:
            from tricoopdiff.experiments.consensus_experiment import ExperimentalConsensus

            model = ExperimentalConsensus(self.expert, trifuse).to(self.device)
        else:
            model = TernaryAssemblyModel(
                frozen_expert=self.expert,
                trifuse=trifuse,
            ).to(self.device)
        weights = getattr(model.trifuse, 'active_state_dict', dict)(self.checkpoint['ema_trifuse_state_dict'])
        current = model.trifuse.state_dict()
        if set(weights) != set(current) or any(
                not torch.is_tensor(weights[name]) or weights[name].shape != current[name].shape
                or not torch.isfinite(weights[name].to(current[name])).all() for name in current):
            raise ValueError('evaluation EMA weights must match the finite active model dtype')
        model.trifuse.load_state_dict(weights, strict=True)
        self.model = model.eval()
        self.input_dim = input_dim

    def evaluate_case(self, case, paths, cache_dir, native_complex_pdb=None,
                      n_walkers=32, steps=20, seed=0,
                      init_mode="sdpi", t_anchor=None, freeze_poi=False,
                      binary_anchor_steps=5, anchor_proposal_attempts=4,
                      anchor_selection="geometry",
                      debug_trace=False, debug_seed=0,
                      translation_velocity_scale=1.0,
                      rotation_velocity_scale=1.0,
                      return_rank_features=False,
                      condition_manifest=None):
        """Featurize and evaluate one case with the resident model."""
        if getattr(self.expert, 'backend', 'diffdock') == 'fabind' and init_mode != 'sdpi':
            raise ValueError('FABind evaluation currently supports SDPI initialization, not DiffDock binary sampling')
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        e3_data, poi_data, ligand_data, input_dim = featurize_complex(
            case, str(paths["protein1"]), str(paths["protein2"]),
            str(paths["ligand"]), self.score_model_args, str(self.device),
            cache_dir=str(cache_dir),
        )
        if getattr(self.expert, 'backend', 'diffdock') == 'fabind':
            e3_data, poi_data, ligand_data = self.expert.prepare_cached_case(e3_data, poi_data, ligand_data)
        self._ensure_model(input_dim)
        conditions, condition_audit, condition_path = None, None, None
        if condition_manifest is not None:
            raise ValueError('assembly checkpoints cannot consume retired condition-token manifests')
        case_initializer = self.initializer
        initialization_version = "sdpi_v1"
        if init_mode == "expert_anchor":
            from tricoopdiff.initialization.candidates import (
                DualExpertAnchorInitializer, EXPERT_ANCHOR_CONSENSUS_VERSION,
                EXPERT_ANCHOR_VERSION,
            )
            case_initializer = DualExpertAnchorInitializer(
                self.expert, e3_data, poi_data,
                binary_steps=binary_anchor_steps,
                proposal_attempts=anchor_proposal_attempts,
                conformer_policy=self.initializer.conformer_policy,
                selection=anchor_selection,
            )
            initialization_version = (
                EXPERT_ANCHOR_CONSENSUS_VERSION
                if anchor_selection == "consensus" else EXPERT_ANCHOR_VERSION
            )
        report = evaluate_model(
            self.model, e3_data, poi_data, ligand_data, self.diffusion,
            initializer=case_initializer, device=str(self.device),
            n_walkers=n_walkers, steps=steps,
            seed=seed,
            native_complex_pdb=(str(native_complex_pdb)
                                if native_complex_pdb else None),
            freeze_poi=freeze_poi, init_mode=init_mode, t_anchor=t_anchor,
            debug_trace=debug_trace, debug_seed=debug_seed,
            translation_velocity_scale=translation_velocity_scale,
            rotation_velocity_scale=rotation_velocity_scale,
            return_rank_features=return_rank_features,
            conditions=conditions,
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
            "initialization_version": initialization_version,
            "binary_anchor_steps": (binary_anchor_steps
                                      if init_mode == "expert_anchor" else None),
            "anchor_proposal_attempts": (anchor_proposal_attempts
                                          if init_mode == "expert_anchor" else None),
            "anchor_selection": (anchor_selection
                                  if init_mode == "expert_anchor" else None),
            "generative_process": "sdpi_flow_v1",
            "t_anchor": (t_anchor if t_anchor is not None else
                         1.0),
            "trifuse_checkpoint": str(self.checkpoint_path.resolve()),
            "checkpoint_schema": (
                "experimental_factorial_v1_ema" if self.experimental_checkpoint else
                f"production_v{self.checkpoint['checkpoint_version']}_ema"),
            "architecture": self.checkpoint['architecture'],
            "rigid_output_mode": ("independent_centroids_v1"
                                  if self.checkpoint['model_config'].get('direct_ternary_motion', False)
                                  else "dual_interfaces_v1"),
            "weight_origin": ("raw_no_ema"
                              if self.checkpoint.get('checkpoint_kind') == 'inference_only'
                              and self.checkpoint['runtime_config'].get('ema_decay') == 0
                              else "ema"),
            "condition_manifest": (str(condition_path.resolve()) if condition_path else None),
            "condition_manifest_sha256": (sha256_file(condition_path) if condition_path else None),
            "condition_audit": condition_audit,
            "rank_features_saved": bool(return_rank_features),
            "rank_feature_state": (
                DIRECT_RANK_FEATURE_STATE
                if return_rank_features and self.checkpoint['model_config'].get('direct_ternary_motion', False)
                else LEGACY_RANK_FEATURE_STATE if return_rank_features else None
            ),
            "checkpoint_sha256": self.checkpoint_sha256,
            "expert_checkpoint_sha256": self.expert.provenance["score_checkpoint_sha256"],
            "expert_config_sha256": self.expert_config_sha256,
            "input_signature": evaluation_input_signature(
                {**paths, **({'condition_manifest': condition_path} if condition_path else {})},
                native_complex_pdb),
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

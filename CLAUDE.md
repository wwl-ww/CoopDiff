# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TriCoopDiff** is a deep learning research project: a physically-constrained diffusion generative model for PROTAC ternary complex structure prediction and functional ranking. The core idea is freezing a pretrained two-body expert (DiffDock-L, ~20M params) and training a lightweight coordination network (TriBrain, ~2.2M params) that modulates the expert via per-layer FiLM, enabling explicit three-body cooperative effects while preserving the expert's pretrained capabilities with zero input-distribution OOD.

**Current status**: The MVP loop is wired end-to-end on real PROTAC data. Frame handling, residual-gated score merge, vendored DiffDock featurization, and both native/inline expert branches are implemented. On case `5T35_D_A_759`, `coupled5a` initialization with no activation clamp passes sub1/sub2 tr/rot/tor parity and the four Phase 1 readiness checks. Training remains B=1. The active work is Phase 1 smoke training and exact full-atom DockQ alignment with DeepTernary.

`doc/README.md` is the authoritative documentation index. The maintained documents cover architecture, current status, initialization/frame conventions, final sampling-parity findings, training/SLURM operation, DeepTernary baseline and DockQ semantics, DiffDock internals, formulas, and the execution roadmap. Do not rely on deleted historical agent briefs or patch handoffs; recover them from Git history only when auditing an old decision.

## Build / Test / Lint

There is **no build system, dependency file, or linter** — no `setup.py`, `pyproject.toml`, `requirements.txt`, or `Makefile`. `tricoopdiff` is **not pip-installed** — entry scripts insert the project root onto `sys.path` manually. There is one standalone smoke-test file (`tests/test_model_build.py`, plain `assert`s, no pytest): `python tests/test_model_build.py`. The real validation gate is `scripts/mvp_smoke_test.py` + the SLURM entry points below.

## Environment & Setup

```bash
conda create -n tricoopdiff python=3.10
pip install torch torch_geometric e3nn torchdiffeq torch_scatter rdkit biopython scipy scikit-learn transformers
# Download DiffDock-L weights to checkpoints/score_model/ (model_parameters.yml + best_ema_inference_epoch_model.pt)
```

- **DiffDock is fully vendored** at `tricoopdiff/diffdock/` — both the model core (`factory`, `models/`, `utils/{so3,torus,torsion,crop,diffusion_utils}`, `sampling.py`) **and** the data-featurization modules (`datasets/` + `utils/inference_utils.py`: `InferenceDataset`, `get_sequences`, `process_mols`, etc.). The frozen expert is built from the vendored core via `expert_factory.build_diffdock_expert` (`load_state_dict(strict=True)`); featurization imports `tricoopdiff.diffdock.datasets.*` / `tricoopdiff.diffdock.utils.inference_utils`. **No external `../DiffDock` checkout is needed anywhere** (BUG-20 fully resolved). Only `scripts/experiment3_ood_verify.py` still references `../DiffDock` by design (it compares vendored vs *native* DiffDock via `get_model`); `scripts/inference.py` / `scripts/eval_entry.py` are BUG-19 legacy (exit non-zero) with a stale `../DiffDock` import.
- **`DeepTernary/`** (git-ignored) holds the baseline; data lives at `DeepTernary/output/protac22/<case>/` with `protein1.pdb` (E3), `protein2.pdb` (POI), `ligand.sdf`/`ligand.pdb` (PROTAC).
- **conda env name is `tricoopdiff`** (SLURM scripts hardcode `/data/home/scwb792/run/miniconda3/bin/conda`).
- **HuggingFace is offline + mirrored**: SLURM scripts `export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. ESM2-650M must already be cached locally (`local_files_only=True` in `esm_embedder.py`).
- **`torch_cluster` / `torch_scatter` are broken stub packages** on this cluster (native CUDA kernels fail to build). `tricoopdiff/diffdock_compat.py` monkey-patches them with pure-PyTorch fallbacks. The patch MUST run before `models.cg_model` is imported; `tricoopdiff/__init__.py` imports it first, and entry scripts import it explicitly.
- **SO(3)/torus cache `.npy` files** (`.so3_score_norms4.npy`, `.so3_omegas_array4.npy`, etc.) resolve relative to CWD; they live in the project root. Entry scripts `os.chdir(_PROJECT_ROOT)` before the model forward so these lookups succeed.

## Architecture

### Data Flow (one diffusion step)

```
ternary graph (E3+POI+PROTAC)
    │
    ├─→ subgraph E3-L (HeteroData) ─→ FrozenDiffDock + FiLM ─→ s_q1
    │
    ├─→ subgraph POI-L (HeteroData) ─→ FrozenDiffDock + FiLM ─→ s_q2
    │                                        ↑
    └─→ TriBrain ─→ FiLM params (γ,β) + merge gate g ─┘
            │
            └─→ s_SE3, confidence, state (ΔΔG, p_productive)

s_q = s_q1 + g·(s_q2 - s_q1)   (residual-gated merge, g zero-init → s_q = s_q1 at init; see merging.py)
```

- **Frozen expert**: DiffDock-L v1.1, `ns=48`, `nv=10`, `num_conv_layers=11` (read from `model_parameters.yml`, not hardcoded in new code). ~20M params, all frozen.
- **Trainable TriBrain**: the only module with gradients. `node_dim=128`, `input_dim` = receptor feature dim (**1281** = 1 amino-acid index + 1280 ESM2-650M; derived at runtime from the first featurized sample), `film_dim` = expert `ns` (48).
- **FiLM heads**: one (γ,β) pair per conv layer per subgraph, zero-initialized (adaLN-Zero) so `h' = h*(1+0)+0 = h` at init.

### Modules

| Module | Purpose |
|--------|---------|
| `expert.py` | `FrozenDiffDock` — wraps the vendored `CGModel`, freezes all params, exposes `embedding()`, `compute_scores()`, `forward()`. Owns `torus_score_norm`. |
| `expert_factory.py` | `build_diffdock_expert(model_dir, device, ...)` — single source of truth for constructing the frozen expert from the vendored core + pretrained weights. Disables dynamic receptor cropping (critical for POI subgraphs). |
| `diffdock_compat.py` | Pure-PyTorch fallbacks for `torch_cluster` / `torch_scatter`; patches on import. |
| `diffdock/` | **Vendored** DiffDock — model core (`factory`, `models/cg_model`, `utils/so3`, `utils/torus`, `utils/torsion`, `utils/crop`, `utils/diffusion_utils`, `sampling.py`) **and** data featurization (`datasets/` + `utils/inference_utils.py`). The model + featurization run entirely on this; no external DiffDock checkout. |
| `esm_embedder.py` | `ESMEmbedder` — ESM2-650M per-residue embeddings (1280-dim) via HF transformers. Not directly imported by the pipeline (featurization uses `scripts/featurize_common.py`). |
| `graphs.py` | `build_subgraph_e3l`, `build_subgraph_pol`, `build_tribrain_graph`, `transform_ligand_to_poi_frame`. Constructs the two disconnected protein+ligand HeteroData subgraphs + one flat ternary dict. |
| `tribrain.py` | `TriBrain`, `MultiScalePooling`, `EquivariantTransformer`, `EquivariantTransformerLayer`, `SE3ScoreHead`, `JudgeHead`, `StateHead`, `FiLMHead`, `MergeGateHead`. The only trainable module. `MergeGateHead` predicts the zero-init residual merge gate `g`. |
| `model.py` | `TriCoopDiff`, `_expert_forward_with_film`, `zero_film_verification`. Orchestrates subgraphs → TriBrain → two expert FiLM passes → merge. `_expert_forward_with_film` replicates `CGModel.forward()` inline, accessing `model.conv_layers[l]` directly. |
| `film.py` | `apply_film` only (used by `model.py:_expert_forward_with_film`). The old `FiLMSharedMLP`/`FiLMLayerHead` classes were removed (`959867a`). |
| `diffusion.py` | `MultiRateDiffusion`, `TernaryInitializer`, `so3_exp`, `sample_so3_noise`, `so3_score_norm`, `so3_score_vec`. Exponential σ(t) schedules for PROTAC (fast) and SE3 rigid body (slow); forward noise (`add_noise_protac`/`add_noise_se3`/`add_torsion_noise`) and reverse SDE steps (`reverse_step_protac`/`reverse_step_torsion`/`reverse_step_se3`). The old λ_q/λ_SE3 threshold masking is **removed** (full-range σ for all t); `get_lambda_q`/`get_lambda_se3` + the threshold params are dead code kept only for legacy diagnostic scripts. |
| `sampler.py` | `sample_single` / `sample_ensemble` — composes the reverse steps into a t=1→0 denoising loop, rebuilding subgraphs inside `model.forward` at each step. Self-contained (does not import `../DiffDock`). |
| `merging.py` | `merge_residual_gated` (**default**, `s_q = s_q1 + g·(s_q2 - s_q1)`), `merge_direct_addition` (L0 ablation only — double-counts the shared PROTAC DOFs). `LearnedCoordinationLayer` is the old Ablation B, now superseded. |
| `reflect.py` | `tricoop_reflect`, `detect_clashes`, `detect_bond_anomalies`, `detect_linker_strain`, `local_minimization`. Post-processing clash/geometry checks + simple repulsive minimization. |
| `train.py` | `TriCoopDiffTrainer` — multi-task loss (score + phys + se3), two-phase training, AdamW + CosineAnnealingLR. `compute_loss`/`train_step`/`validate_step`, `set_phase`, `save/load_checkpoint`. |
| `evaluate.py` | Metric functions (`compute_dockq`, `compute_rmsd`, `compute_clash_rate`, `compute_ensemble_diversity`, `compute_cooperativity_ranking`, `kabsch_align`, `compute_poi_dockq`, `compute_e3_interface_dockq`, `compute_poi_com_dist`, …) + `evaluate_model` + `EnsembleAnalyzer`. POI/E3-interface metrics need the native ternary PDB (`native_complex_pdb`); otherwise they report `None`, never NaN. |
| `scripts/featurize_common.py` | Shared featurization: `featurize_complex`, `generate_esm_embeddings`, `featurize_subgraph`, `extract_receptor`, `extract_ligand`. Produces B=1 dicts with `coords`/`features`/`edge_index`/`edge_attr`/`edge_mask`/`mask_rotate`. Re-express POI + PROTAC in the shared E3 frame (BUG-18 fix: recovers true crystal-frame inter-molecule translations, since `InferenceDataset` centers each molecule independently); caches carry a `cache_version` to invalidate pre-fix geometry. |
| `scripts/make_protac_index.py` | Scans `DeepTernary/output/protac22/` → `data/protac_index.json` (absolute paths, consumed by `train_entry.py`). |

### Diffusion Variables

| DOF | Dim | Noise Source | Score Source |
|-----|-----|-------------|-------------|
| PROTAC translation | 3 | `add_noise_protac` | merged from both subgraphs |
| PROTAC rotation | 3 (so(3)) | `add_noise_protac` | merged from both subgraphs |
| PROTAC torsion | K | `add_torsion_noise` | merged from both subgraphs |
| E3-POI rigid body | 6 (3T+3R) | `add_noise_se3` | `s_se3` from TriBrain SE3 head |

Reference frame: E3 fixed at its featurized coordinates (Cα centroid at origin). After the BUG-18 fix, POI and PROTAC are re-expressed in this shared E3 frame (true crystal-frame translations), so `t_e3_to_poi = poi_center - e3_center` recovers the real E3→POI offset rather than 0. PROTAC coords are sampled once in the E3 frame; subgraph 2 uses `transform_ligand_to_poi_frame`.

## Running Commands

All entry points are `python scripts/*.py` run from the project root (SLURM wrappers `cd "$PROJECT_ROOT"`). Pipeline: index → train → evaluate, with a smoke test for fast sanity checks.

### Build the data index

```bash
python scripts/make_protac_index.py            # → data/protac_index.json
```

### Smoke test (two gates, ~30 min on 1 GPU)

```bash
sbatch scripts/mvp_smoke_test.slurm
# or manually:
python scripts/mvp_smoke_test.py --device cuda --case 5T35_D_A_759 --seed 0
```
GATE 1: zero-FiLM transparency (zero-initialized FiLM reproduces native expert output within `--atol`). GATE 2: finite grads on FiLM + SE3 heads while the frozen expert stays grad-free. Exit 0 = pass.

### Training (B=1 sequential)

```bash
# Phase 1 smoke (3 epochs, single complex per step)
sbatch scripts/train_entry_smoke.slurm
# or manually:
python scripts/train_entry.py \
    --train_data data/protac_index.json --val_data data/protac_index.json \
    --output_dir results/train_smoke --epochs 3 --batch_size 1 \
    --phase 1 --expert_model_dir checkpoints/score_model \
    --cache_dir results/featurize_cache --device cuda
```
`--phase 1` = pretrain (FiLM + SE3 heads only), `--phase 2` = unfreeze all TriBrain heads. Override epochs/LR via `sbatch --export=EPOCHS=100,LR=1e-4`. Checkpoints write to `--output_dir`.

### Evaluation (ensemble sampling)

```bash
sbatch scripts/evaluate_entry.slurm
# or manually (zero-FiLM baseline: omit --tribrain-ckpt):
python scripts/evaluate_entry.py --case 5T35_D_A_759 --n-walkers 32 \
    --out results/eval_5T35.json
```
Reports PROTAC RMSD / pocket distance / clash rate / ensemble diversity / E3-interface & POI DockQ (when the native PDB is available) as JSON. The zero-FiLM baseline (no `--tribrain-ckpt`) is the Go/No-Go reference: it should reproduce native DiffDock-L two-body PROTAC distribution. `evaluate_entry.py` is single-complex; `scripts/eval_entry.py` and `scripts/inference.py` are **legacy and exit non-zero** (BUG-19) — don't use them for the full-benchmark comparison.

### DeepTernary baseline audit

```bash
sbatch scripts/audit_slurm.sh            # Phase 1 predictions (GPU) + Phase 2 metrics (CPU)
# or two-phase manually — see scripts/README_audit.md
```
Output: `deepternary_audit.json`, `audit_results/`, `logs/`. Full instructions in `scripts/README_audit.md`.

### SLURM gotchas (from `doc/训练入口 SLURM 与 Phase1 smoke 交接文档.md`)

- **Real partitions**: `gpu_a800`, `gpu_h100`, `gpu_h200`. **`hp_a800` does not exist** — the `sinfo` wrapper reads a stale static file. Check with `/usr/local/bin/slurm/sinfo` or `scontrol show partition`.
- **CPU quota**: `job_submit.lua` enforces **exactly 6 cores/GPU** on non-h100/h200 partitions. `--cpus-per-task` must be ≤6 (the SLURM scripts use 4). `--cpus-per-task=8` fails with `Unspecified error`.
- `gpu_a800` is congested; 1-GPU jobs often backfill ~30 min before starting (normal, not an error).

## Critical Design Constraints

- **Two subgraphs must be physically disconnected**: no protein-protein edges, no ligand-mediated bridging. Each subgraph = exactly one protein + PROTAC. Cross-edges are built dynamically by `CGModel` per forward pass.
- **PROTAC coordinate sync**: PROTAC sampled once globally (E3 frame). Subgraph 1 uses it directly; subgraph 2 transforms via `t_e3_to_poi = poi_center - e3_center` and the `r_e3_to_poi` rotation (defaults to identity; the sampler/trainer passes the cumulative SE3 rotation so subgraph 2 sees the correct PROTAC-POI relative geometry). Subgraph 2's tr/rot scores are rotated back into the E3 frame before merging (torsion is frame-invariant).
- **FiLM zero-init is mandatory**: `FiLMHead` uses `nn.init.zeros_`. At training start `h' = h`. Verified by `zero_film_verification()` (GATE 1 of the smoke test).
- **Gradient flow**: the two-body expert has `requires_grad=False` everywhere. `loss_phys` backprops through `_estimate_denoised()` (Tweedie) → predicted scores → `model.forward()` → TriBrain along two paths: (1) the PROTAC path passes *through* FiLM (`x̂_ligand` depends on `s_q`, which is FiLM-modulated); (2) the POI/SE3 path goes directly through the SE3 head. The old "not through FiLM" note in `train.py`'s docstring is stale.
- **Score merging**: residual-gated `s_q = s_q1 + g·(s_q2 - s_q1)` is the default (`use_learned_merge=True`). The gate `g` is zero-initialized so at init `s_q = s_q1` (pure E3-L, zero-FiLM transparent); `g=1` → pure POI-L. `merge_direct_addition` is the L0 ablation (`use_learned_merge=False`) and re-introduces double-counting.
- **No multi-rate λ masking anymore**: the stepwise λ_q (t=0.7) / λ_SE3 (t=0.5) cutoffs were removed in the debug-se3-loss work. Forward noise, reverse denoising, and the DSM target now all use the same full-range σ(t) schedule; the λ threshold params/functions are dead code.
- **`t` is a noise-schedule variable in [0,1]**, not physical time.
- **`_expert_forward_with_film` bypasses `FrozenDiffDock.step_conv_layer()`**: it replicates `CGModel.forward()` inline and directly accesses `model.conv_layers[l]`. Intentional — avoids refactoring the edge-group splitting that differs between intermediate and final conv layers.
- **Import order**: `tricoopdiff.diffdock_compat` must be imported before anything that binds `torch_cluster`/`torch_scatter`. Importing any `tricoopdiff.*` submodule runs `__init__.py`, which does this first.
- **FiLM modulates only ligand nodes, only the scalar (0e) channels**, to preserve SE(3) equivariance (`apply_film` with `node_mask` + `scalar_dim=expert.ns`).

## Known Incomplete / Open Items

- **Exact DockQ alignment**: the primary metric must rebuild a full-atom ternary PDB and call DeepTernary's `DockQ/dockq_util.py::cal_dockq`. Cα-only and interface-specific metrics remain diagnostics.
- **Phase 1 smoke**: regenerate parity/frame reports after result cleanup, rerun readiness, then validate all loss components, finite gradients, and checkpoint loading.
- **Initialization semantics**: `coupled5a` passes ternary native/inline parity without clamp; `native` remains the DiffDock-compatible two-body prior and can create OOD POI-L graphs in ternary use.
- **B=1 only**: `model.forward` remains single-complex per step; padded B>1 batching needs a separate validation cycle.
- **Pooling geometry**: `MultiScalePooling` still applies its branches to the same node set; true distance-scale partitioning is not implemented.
- **Physical proxies**: torsional strain and local minimization remain lightweight geometric approximations rather than a force field.

## Code Duplication to Be Aware Of

- **Single FiLM head**: `tribrain.py:FiLMHead` (production — Linear, zero-init). `film.py` holds only `apply_film`; the standalone `FiLMSharedMLP`/`FiLMLayerHead` classes were removed (`959867a`).
- **Score norm split**: `so3_score_norm` lives in `diffusion.py` (imported by `train.py` via `from .diffusion import so3_score_norm`); `expert.py` has a separate `torus_score_norm` (torsion). If the DiffDock normalization formula changes, update `diffusion.py`.
- **Stale/divergent entry points** (BUG-19): `scripts/inference.py` and `scripts/eval_entry.py` use a `../DiffDock` legacy `CGModel` import, zero-feature placeholder proteins, and their own reverse-diffusion loop — they have **fully forked** from the real pipeline and now **refuse to run** (LEGACY banner + `sys.exit(2)`). The real inference/eval path is `scripts/evaluate_entry.py` → `featurize_common.featurize_complex` → `expert_factory.build_diffdock_expert` → `sampler.sample_ensemble`.
- **Dead λ code**: `diffusion.py`'s `get_lambda_q`/`get_lambda_se3` + `lambda_q_threshold`/`lambda_se3_threshold` are deprecated (production never consults them); only `scripts/diagnose_loss_se3.py`, `scripts/verify_se3_sigma_consistency.py`, and `tests/test_model_build.py` still reference them.

## Training Strategy

- **Phase 1**: freeze judge/state heads (`trainer.set_phase(1)`), train FiLM + SE3 heads. 50-100 epochs, batch=32, lr=1e-4.
- **Phase 2**: unfreeze all TriBrain heads (`trainer.set_phase(2)`), same hyperparams.
- **PROTAC fine-tuning**: 30-50 epochs, batch=8, lr=2e-5, LoRA rank=4 or strong weight decay.
- Loss weights: `score=1.0, phys=0.5, se3=0.3` (configurable in `TriCoopDiffTrainer.__init__`).
- Batch sizes above are the execution-plan targets; the current code only supports **B=1** (see "B=1 only" in Open Items).

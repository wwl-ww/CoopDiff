# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TriCoopDiff** is a deep learning research project: a physically-constrained diffusion generative model for PROTAC ternary complex structure prediction and functional ranking. The core idea is freezing a pretrained two-body expert (DiffDock-L, ~20M params) and training a lightweight coordination network (TriBrain, ~2.2M params) that modulates the expert via per-layer FiLM, enabling explicit three-body cooperative effects while preserving the expert's pretrained capabilities with zero input-distribution OOD.

**Current status**: The MVP integration loop is wired end-to-end on real PROTAC data: featurize → train → sample → evaluate. `scripts/mvp_smoke_test.py` passes its two gates (zero-FiLM transparency + gradient flow to TriBrain). Training runs B=1 (sequential per-complex). The "Agent B" task (E3→POI rotation + POI/DockQ metrics) is **done** — `model.forward` takes an `r_e3_to_poi` rotation (defaults to identity) and `evaluate.py` reports POI/E3-interface DockQ. The SE3-loss debug work is **merged to `master`** (was branch `debug-se3-loss`); current branch is `bug-fix`.

The two original "open problem #1" root causes — score-merge double-counting + frame mismatch — are **fixed**: merge is now residual-gated (`s_q = s_q1 + g·(s_q2 - s_q1)`, `g` zero-init, commit `cacbb7f`). The **active open problem is now sampling fidelity**: the zero-FiLM baseline E3-interface DockQ is **0.0925** vs pure DiffDock E3-L **0.844** (best of 32; was 0.215 pre-`959867a`), i.e. TriCoopDiff's reverse-sampling loop does not yet reproduce native DiffDock inference even when the merge is transparent. A code-review also surfaced two 🔴 critical bugs — **BUG-18** (featurization independently centers each molecule, destroying inter-molecule relative geometry) and **BUG-20** (external `../DiffDock` path) — now fully resolved by vendoring DiffDock featurization in-repo. `doc/README.md` is the authoritative doc index — consult it first; the current problem snapshot is `doc/问题清单与修复状态 交接文档.md`.

Key design documents in `doc/` (Chinese). **`doc/README.md` is the authoritative index**, tiering each doc as current / partially-stale / superseded; consult it before trusting a doc:
- `TriCoopDiff 模型交接文档.md` (v2.4) — the sole authoritative architecture doc (per-layer FiLM, multi-rate diffusion, SE3 rigid body, TriReflect)
- `问题清单与修复状态 交接文档.md` — **current problem snapshot (08-17): "read this first"**. Which fixes are committed vs still working-tree, plus the sampling-fidelity gap.
- `代码审查 Bug 清单.md` — 19 code-review findings (🔴/🟡/🟢), with a post-merge re-audit; BUG-18/20 are the critical ones.
- `采样保真度诊断报告.md` / `采样保真度调查任务 brief.md` — line-by-line comparison of TriCoopDiff reverse-sampling vs native DiffDock inference (torsion σ mismatch + missing torsion temperature sampling are the top hypotheses for the 0.215 vs 0.844 gap).
- `参考系与数值梯度专项分析.md` — rotation-convention / frame cross-checks; confirms BUG-6/7 are false alarms, upgrades BUG-18.
- `TriCoopDiff 执行计划.md` — 9-week execution plan with phase gates
- `三分子初始化策略方案.md` — Three-molecule initialization strategy
- `DeepTernary 审计完成交接文档.md` — baseline audit (mean best DockQ **0.6588**, 22 cases × 40 seeds)
- `Score合并 双重计数与零-FiLM基线 交接文档.md` — open problem #1 (now **resolved** via residual-gated merge); historical A/B/C discussion.
- `DiffDock 审计文档.md` / `DiffDock 实验交接文档.md` — two-body expert reference
- `公式推导参考.md` — formula derivations
- `训练入口 SLURM 与 Phase1 smoke 交接文档.md` — SLURM gotchas + Phase1 smoke observations

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

### 🔴 Active: sampling fidelity (open problem #1, re-scoped)

The score-merge double-counting + frame mismatch are **fixed** (residual-gated merge, commit `cacbb7f`), but the zero-FiLM baseline E3-interface DockQ is still **0.0925** vs pure DiffDock E3-L **0.844** (best of 32; pre-`959867a` value was 0.215, old pre-merge-fix value was 0.153). Since the merge is transparent at init, this means **TriCoopDiff's E3-L reverse-sampling loop does not reproduce native DiffDock inference**. The top hypotheses (see `doc/采样保真度诊断报告.md`) are, in confidence order — **#1/#2 are now fixed** (committed in `959867a`, `diffusion.py`) and **#3 is now applied** (working tree):

1. **Torsion σ mismatch** (✅ fixed): `diffusion.py` hardcoded `tor_sigma_min/max = 0.03/1.55`, but the actual DiffDock-L weights use `model_parameters.yml`'s `0.0314/3.14` — the score is normalized on one σ range and the reverse step integrates another, so `tor_g²` was ~4.8× too weak. Now uses `0.0314/3.14`.
2. **Missing torsion temperature sampling** (✅ fixed): `reverse_step_torsion` lacked the `temp_sampling[2]=7.04` branch that tr/rot have (the constant was defined but never read) → ~25× too-weak torsion drift, the likely driver of the 50 Å COM drift. Now mirrors the tr/rot temperature branch.
3. **Initialization mismatch** (✅ applied, working tree — `diffusion.py` + `sampler.py`, patch `doc/假设3_初始化对齐.patch`): `initialize_protac` starts from the template pose (0–10 Å shell, template torsions) vs native uniform-random torsions + receptor-COM Gaussian (std = `tr_sigma_max` = 19.0). Now uses receptor COM (`e3_coords.mean(0)`) instead of the template-COM `pocket_center`, plus native uniform torsions + Gaussian.

**2026-08-17 regression note**: eval job 139693 (only #1/#2, clean of #3) regressed to **0.0925** (from 0.215) — `protac_pocket_dist_mean` 52.5 Å, best 21.2 Å. Root cause: commit `959867a` bundled #1/#2 with the BUG-18 featurization re-expression, which moved the PROTAC template COM from the E3 origin (~0) to its true crystal position (~20–40 Å out), exposing `sampler.py`'s use of the template COM (not receptor COM) as the init center. That is exactly hypothesis #3, now fixed by the applied patch — **awaiting a re-run** to test whether it recovers ~0.844. Caveat: even pre-BUG-18 (with correct-ish init) the baseline was only 0.215, so a deeper tr/rot/tor reverse-loop mismatch may remain beyond #3.

**Acceptance criterion**: zero-FiLM baseline E3-interface DockQ ≈ 0.844 (within 0.02, same seed). See `doc/问题清单与修复状态 交接文档.md` §2.

### 🔴 Critical bugs from the code review (`doc/代码审查 Bug 清单.md`)

- **BUG-18 — featurization destroys relative geometry** (confirmed): `featurize_complex` uses DiffDock's `InferenceDataset`, which centers each molecule at its own centroid, so E3, POI and PROTAC each end up at the origin and `t_e3_to_poi = 0`. Training would learn to denoise everything onto the origin. **Fixed** (committed in `959867a`): `featurize_common.py` now recomputes true Cα/heavy-atom centroids from the raw PDB/SDF files and re-expresses POI + PROTAC in the shared E3 frame (with a `cache_version` bump to invalidate stale caches).
- **BUG-20 — DiffDock path** (confirmed): code referenced the sibling `../DiffDock` checkout for featurization. **Resolved by full vendoring** (committed `d297eb9`): `datasets/` + `utils/inference_utils.py` are now vendored under `tricoopdiff/diffdock/`, and `_resolve_diffdock_dir()` was removed entirely — no external checkout needed.
- The other 🟡 findings (BUG-1/2/5/8/11/12/13/14/15/16/17/19) are now fixed (committed `959867a`). BUG-10 (`_physical_loss` → per-element VDW radii) is fixed in the working tree. 2 🟢 items remain open: BUG-4/N-4 (`EquivariantTransformer`/`MultiScalePooling` geometry-blind, compensated by `SE3GeometryEncoder`), N-2 (dead λ code).

### Resolved (merged to master)

- **`loss_se3` (was ≈0 / π-floored)** — root causes fixed on `debug-se3-loss` (now merged): (a) wrapped SO(3) sampler/score mismatch → Euclidean (R³) DSM for SE3 rotation; (b) λ_SE3 cutoff dropped → full t∈[0,1] schedule; (c) SE3 geometry encoder gained rotation-sensitive covariance features. Verif run 138510: `loss_se3_tr` 54→~3, `loss_se3_rot` active (avg ~1.3, below the π≈3.14 floor). A pre-existing score-loss blow-up on large PROTACs (`NAN_CLAMP=1e4` → loss ~1e6-1e8) was masking SE3 convergence; fixed with a per-component `score_loss_cap=100.0`. See commits `b20c238`/`4511ab0`/`cbbd5f0`/`f3b2736`/`c1f1dd0`.

### Still open / follow-up

- **`train.py:_physical_loss`**: now element-aware — decodes PROTAC atom numbers from `ligand_features` into Bondi VDW radii (`_ligand_vdw_radii`), uses carbon radius (1.7 Å) for protein Cα, and penalizes `F.relu((r_i+r_j) - dist)²`. Falls back to the single `contact_distance` (3.4 Å == C···C sum) when ligand features are absent. Still no per-element protein radii (receptor is Cα-level by construction), and no true force-field minimization (see `reflect.py`).
- **B=1 only**: `model.forward` is single-complex per step; padded B>1 batching is a follow-up.
- **`scripts/prepare_data.py:cluster_by_sequence_identity`**: greedy fallback; production should use MMseqs2.
- **`reflect.py:local_minimization`**: simple repulsive gradient descent; production should use RDKit MMFF94.
- **POI-L FiLM gradient starvation** (open decision): `d(s_q)/d(s_q2) = g = 0` at init, so the POI-L FiLM head gets no gradient until the merge gate moves off zero. Accepted for now; watch whether the gate moves during training.
- **Checkpoint bloat** (fixed): `save_checkpoint` no longer writes the ~131 MB frozen-expert `model_state_dict`, only `tribrain_state_dict` + optimizer + scheduler; `load_checkpoint` reads with `.get()` for backward compatibility.

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

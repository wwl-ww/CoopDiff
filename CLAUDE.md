# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TriCoopDiff** is a deep learning research project: a physically-constrained diffusion generative model for PROTAC ternary complex structure prediction and functional ranking. The core idea is freezing a pretrained two-body expert (DiffDock-L, ~20M params) and training a lightweight coordination network (TriBrain, ~2.2M params) that modulates the expert via per-layer FiLM, enabling explicit three-body cooperative effects while preserving the expert's pretrained capabilities with zero input-distribution OOD.

**Current status**: The MVP integration loop is wired end-to-end on real PROTAC data: featurize → train → sample → evaluate. `scripts/mvp_smoke_test.py` passes its two gates (zero-FiLM transparency + gradient flow to TriBrain). Training currently runs B=1 (sequential per-complex). The E3→POI rotation in `model.forward` is still an identity placeholder (open "Agent B" task). See `doc/` for the two most recent handoff notes.

Key design documents in `doc/` (Chinese):
- `TriCoopDiff 模型交接文档.md` — Complete model architecture and design rationale
- `TriCoopDiff 执行计划.md` — 9-week execution plan with phase gates
- `TriCoopDiff vs DeepTernary 竞争定位与实验计划.md` — Competitive positioning vs DeepTernary
- `三分子初始化策略方案.md` — Three-molecule initialization strategy
- `DeepTernary 审计交接文档.md` / `审计完成交接文档.md` / `论文分析与审计对比.md` — DeepTernary baseline audit
- `AgentB 交接文档.md` — Open task: fix `model.py` E3→POI identity rotation + add POI/DockQ metrics
- `训练入口 SLURM 与 Phase1 smoke 交接文档.md` — SLURM gotchas + Phase1 smoke observations (loss_phys/loss_se3 ≈ 0)

## Build / Test / Lint

There is **no build system, test suite, or linter**. No `setup.py`, `pyproject.toml`, `requirements.txt`, or `Makefile`. `tricoopdiff` is **not pip-installed** — entry scripts insert the project root onto `sys.path` manually. Validation is via the smoke test + SLURM entry points below.

## Environment & Setup

```bash
conda create -n tricoopdiff python=3.10
pip install torch torch_geometric e3nn torchdiffeq torch_scatter rdkit biopython scipy scikit-learn transformers
# Download DiffDock-L weights to checkpoints/score_model/ (model_parameters.yml + best_ema_inference_epoch_model.pt)
```

- **DiffDock is vendored** at `tricoopdiff/diffdock/` (factory + models + utils). The frozen expert is built from this vendored core via `expert_factory.build_diffdock_expert`, which does `load_state_dict(strict=True)`. The sibling `../DiffDock` checkout is used **only for data featurization** (`InferenceDataset`, `get_sequences`, `process_mols`) and must still be cloned alongside this repo.
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
    └─→ TriBrain ─→ FiLM params (γ,β) ──────┘
            │
            └─→ s_SE3, confidence, state (ΔΔG, p_productive)

s_q = s_q1 + s_q2  (direct addition, see merging.py)
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
| `diffdock/` | **Vendored** DiffDock core (`factory`, `models/cg_model`, `utils/so3`, `utils/torus`, `utils/torsion`, `utils/crop`, `utils/diffusion_utils`). The model runs entirely on this. |
| `esm_embedder.py` | `ESMEmbedder` — ESM2-650M per-residue embeddings (1280-dim) via HF transformers. Not directly imported by the pipeline (featurization uses `scripts/featurize_common.py`). |
| `graphs.py` | `build_subgraph_e3l`, `build_subgraph_pol`, `build_tribrain_graph`, `transform_ligand_to_poi_frame`. Constructs the two disconnected protein+ligand HeteroData subgraphs + one flat ternary dict. |
| `tribrain.py` | `TriBrain`, `MultiScalePooling`, `EquivariantTransformer`, `EquivariantTransformerLayer`, `SE3ScoreHead`, `JudgeHead`, `StateHead`, `FiLMHead`. The only trainable module. |
| `model.py` | `TriCoopDiff`, `_expert_forward_with_film`, `zero_film_verification`. Orchestrates subgraphs → TriBrain → two expert FiLM passes → merge. `_expert_forward_with_film` replicates `CGModel.forward()` inline, accessing `model.conv_layers[l]` directly. |
| `film.py` | `FiLMSharedMLP`, `FiLMLayerHead`, `apply_film`. Only `apply_film` is actually used (by `model.py:_expert_forward_with_film`); the MLP classes are unused. |
| `diffusion.py` | `MultiRateDiffusion`, `TernaryInitializer`, `so3_exp`, `sample_so3_noise`, `so3_score_norm`, `so3_score_vec`. Multi-rate schedules (PROTAC fast λ_q, SE3 slow λ_SE3); forward noise (`add_noise_protac`/`add_noise_se3`/`add_torsion_noise`) and reverse SDE steps (`reverse_step_protac`/`reverse_step_torsion`/`reverse_step_se3`). |
| `sampler.py` | `sample_single` / `sample_ensemble` — composes the reverse steps into a t=1→0 denoising loop, rebuilding subgraphs inside `model.forward` at each step. Self-contained (does not import `../DiffDock`). |
| `merging.py` | `merge_direct_addition` (MVP default), `LearnedCoordinationLayer` (Ablation B). |
| `reflect.py` | `tricoop_reflect`, `detect_clashes`, `detect_bond_anomalies`, `detect_linker_strain`, `local_minimization`. Post-processing clash/geometry checks + simple repulsive minimization. |
| `train.py` | `TriCoopDiffTrainer` — multi-task loss (score + phys + se3), two-phase training, AdamW + CosineAnnealingLR. `compute_loss`/`train_step`/`validate_step`, `set_phase`, `save/load_checkpoint`. |
| `evaluate.py` | Metric functions (`compute_dockq`, `compute_rmsd`, `compute_clash_rate`, `compute_ensemble_diversity`, `compute_cooperativity_ranking`, `kabsch_align`, …) + `evaluate_model` (MVP: PROTAC pose only) + `EnsembleAnalyzer`. |
| `scripts/featurize_common.py` | Shared featurization: `featurize_complex`, `generate_esm_embeddings`, `featurize_subgraph`, `extract_receptor`, `extract_ligand`. Produces B=1 dicts with `coords`/`features`/`edge_index`/`edge_attr`/`edge_mask`/`mask_rotate`. |
| `scripts/make_protac_index.py` | Scans `DeepTernary/output/protac22/` → `data/protac_index.json` (absolute paths, consumed by `train_entry.py`). |

### Diffusion Variables

| DOF | Dim | Noise Source | Score Source |
|-----|-----|-------------|-------------|
| PROTAC translation | 3 | `add_noise_protac` | merged from both subgraphs |
| PROTAC rotation | 3 (so(3)) | `add_noise_protac` | merged from both subgraphs |
| PROTAC torsion | K | `add_torsion_noise` | merged from both subgraphs |
| E3-POI rigid body | 6 (3T+3R) | `add_noise_se3` | `s_se3` from TriBrain SE3 head |

Reference frame: E3 fixed at its featurized coordinates. PROTAC coords sampled once in the E3 frame; subgraph 2 uses `transform_ligand_to_poi_frame`.

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
Reports PROTAC RMSD / pocket distance / clash rate / ensemble diversity as JSON. The zero-FiLM baseline (no `--tribrain-ckpt`) is the Go/No-Go reference: it should reproduce native DiffDock-L two-body PROTAC distribution.

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
- **PROTAC coordinate sync**: PROTAC sampled once globally (E3 frame). Subgraph 1 uses it directly; subgraph 2 transforms via `t_e3_to_poi = poi_center - e3_center` (currently **identity rotation** — see known issues).
- **FiLM zero-init is mandatory**: `FiLMHead` uses `nn.init.zeros_`. At training start `h' = h`. Verified by `zero_film_verification()` (GATE 1 of the smoke test).
- **Gradient flow**: `loss_phys` backprops through `_estimate_denoised()` (Tweedie) → predicted scores → `model.forward()` → TriBrain — a direct path, not through FiLM. The two-body expert has `requires_grad=False` everywhere.
- **Score merging**: direct addition `s_q = s_q1 + s_q2` is the MVP. `LearnedCoordinationLayer` is Ablation B.
- **Multi-rate λ**: stepwise threshold functions (non-differentiable at t=0.7, t=0.5); accepted engineering approximation.
- **`t` is a noise-schedule variable in [0,1]**, not physical time.
- **`_expert_forward_with_film` bypasses `FrozenDiffDock.step_conv_layer()`**: it replicates `CGModel.forward()` inline and directly accesses `model.conv_layers[l]`. Intentional — avoids refactoring the edge-group splitting that differs between intermediate and final conv layers.
- **Import order**: `tricoopdiff.diffdock_compat` must be imported before anything that binds `torch_cluster`/`torch_scatter`. Importing any `tricoopdiff.*` submodule runs `__init__.py`, which does this first.
- **FiLM modulates only ligand nodes, only the scalar (0e) channels**, to preserve SE(3) equivariance (`apply_film` with `node_mask` + `scalar_dim=expert.ns`).

## Known Incomplete / Open Items

- **`model.py:forward` E3→POI rotation is `torch.eye(3)`** (identity). SE3 diffusion rotates POI, but subgraph 2 is rebuilt with an identity rotation, so the PROTAC-in-POI-frame orientation is wrong once POI is rotated/initialized randomly. This is the open "Agent B" task — see `doc/AgentB 交接文档.md`. Fix must default to identity so existing callers are unaffected.
- **`evaluate.py:evaluate_model` reports PROTAC (E3-L) pose only**. POI RMSD / DockQ / E3-POI interface metrics are intentionally absent because the POI reference is featurized in a different frame. Also Agent B's task.
- **`train.py:_physical_loss`**: fixed 3.4 Å contact distance (per-atom via `torch.cdist`), not true VDW radii. In Phase-1 smoke this makes `loss_phys ≈ 0` (well-separated complexes have no clash at noise level) — the physical term produces no gradient until real VDW radii are used.
- **`loss_se3 ≈ 0` in Phase-1 smoke** — needs investigation (SE3 head is zero-init so output≈0; either σ weighting or target scale is wrong). Phase 1 effectively only trains score-matching through FiLM. See the handoff doc.
- **B=1 only**: `model.forward` is single-complex per step; padded B>1 batching is a follow-up.
- **Checkpoint bloat**: `save_checkpoint` writes full `model_state_dict` (~131 MB, includes frozen expert) but `load_checkpoint` only reads `tribrain_state_dict` (~9 MB). `load_checkpoint` also does not restore optimizer/scheduler.
- **`scripts/prepare_data.py:cluster_by_sequence_identity`**: greedy fallback; production should use MMseqs2.
- **`reflect.py:local_minimization`**: simple repulsive gradient descent; production should use RDKit MMFF94.

## Code Duplication to Be Aware Of

- **Two FiLM head implementations**: `tribrain.py:FiLMHead` (production — Linear, zero-init) and `film.py:FiLMLayerHead`/`FiLMSharedMLP` (standalone, unused). Only `apply_film` from `film.py` is called.
- **Score norm split**: `so3_score_norm` lives in `diffusion.py` (imported by `train.py` via `from .diffusion import so3_score_norm`); `expert.py` has a separate `torus_score_norm` (torsion). If the DiffDock normalization formula changes, update `diffusion.py`.

## Training Strategy

- **Phase 1**: freeze judge/state heads (`trainer.set_phase(1)`), train FiLM + SE3 heads. 50-100 epochs, batch=32, lr=1e-4.
- **Phase 2**: unfreeze all TriBrain heads (`trainer.set_phase(2)`), same hyperparams.
- **PROTAC fine-tuning**: 30-50 epochs, batch=8, lr=2e-5, LoRA rank=4 or strong weight decay.
- Loss weights: `score=1.0, phys=0.5, se3=0.3` (configurable in `TriCoopDiffTrainer.__init__`).

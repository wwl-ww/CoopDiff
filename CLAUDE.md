# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TriCoopDiff** is a deep learning research project: a physically-constrained diffusion generative model for PROTAC ternary complex structure prediction and functional ranking. The core idea is freezing a pretrained two-body expert (DiffDock, 20M params) and training a lightweight coordination network (TriBrain, ~2.2M params) that modulates the expert via per-layer FiLM, enabling explicit three-body cooperative effects while preserving the expert's pretrained capabilities with zero input-distribution OOD.

**Current status**: Phase 0-1 complete. All 11 core modules implemented across 3 git commits (scaffold → modules → audit doc). No training has been run yet; this is ready for Phase 1 MVP integration testing. DeepTernary baseline audit has been run (results in `deepternary_audit.json`).

Key design documents in `doc/` (all in Chinese):
- `TriCoopDiff 模型交接文档.md` — Complete model architecture and design rationale (v2.4)
- `TriCoopDiff 执行计划.md` — 9-week execution plan with phase gates and Go/No-Go criteria
- `TriCoopDiff vs DeepTernary 竞争定位与实验计划.md` — Competitive positioning vs DeepTernary
- `三分子初始化策略方案.md` — Three-molecule initialization strategy
- `DeepTernary 审计交接文档.md` — DeepTernary baseline audit results

## Build / Test / Lint

**There is no build system, test suite, or linter configuration.** This repo has no `setup.py`, `pyproject.toml`, `requirements.txt`, or `Makefile`. There are no test files anywhere. All modules are import-only (no CLI entry points). The code is validated by running the audit scripts (see below).

## Environment & Setup

Requires a working DiffDock installation cloned alongside this repo.

```bash
conda create -n tricoopdiff python=3.10
pip install torch torch_geometric e3nn torchdiffeq torch_scatter rdkit biopython scipy scikit-learn
git clone https://github.com/gcorso/DiffDock.git  # expected at ../DiffDock
# Download DiffDock-L pretrained weights to checkpoints/
```

The `DeepTernary/` subdirectory is git-ignored (`DeepTernary/` is in `.gitignore`) — it must be cloned separately from `github.com/youqingxiaozhua/DeepTernary` for baseline comparisons.

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

### Modules and What They Do

| Module | Key classes/functions | Purpose |
|--------|----------------------|---------|
| `graphs.py` | `build_subgraph_e3l`, `build_subgraph_pol`, `build_tribrain_graph`, `transform_ligand_to_poi_frame` | Construct two disconnected protein+ligand HeteroData subgraphs + one flat ternary dict. Cross-edges built dynamically by CGModel at each forward pass. |
| `tribrain.py` | `TriBrain`, `MultiScalePooling`, `EquivariantTransformer`, `EquivariantTransformerLayer`, `SE3ScoreHead`, `JudgeHead`, `StateHead`, `FiLMHead` | The ONLY trainable module. Pools ternary graph → equivariant transformer → FiLM params (12 heads: 6 layers × 2 subgraphs) + SE3 score + confidence + state. FiLM heads zero-initialized (adaLN-Zero). ~2.2M params in MVP config. |
| `expert.py` | `FrozenDiffDock` | Wraps DiffDock CGModel, freezes all params, exposes `embedding()`, `step_conv_layer()`, `compute_scores()`, `build_cross_graph()`, `build_center_graph()`, `build_bond_graph()`. |
| `film.py` | `FiLMSharedMLP`, `FiLMLayerHead`, `apply_film` | Standalone FiLM utilities. Note: these are NOT used by the main model — TriBrain has its own inline `FiLMHead` and `film_shared` MLP in `tribrain.py`. Only `apply_film` from this module is actually called (by `model.py:_expert_forward_with_film`). |
| `model.py` | `TriCoopDiff`, `_expert_forward_with_film`, `zero_film_verification` | Top-level orchestrator: builds subgraphs, runs TriBrain, runs both expert passes with per-layer FiLM, merges scores. `_expert_forward_with_film` replicates CGModel.forward() internals by directly accessing `model.conv_layers[l]` (it does NOT use `FrozenDiffDock.step_conv_layer()`). |
| `diffusion.py` | `MultiRateDiffusion`, `TernaryInitializer`, `so3_exp`, `sample_so3_noise`, `vector_to_skew` | Multi-rate noise schedules (PROTAC fast λ_q, SE3 slow λ_SE3), forward/reverse SDE steps, SO(3) Lie algebra utilities. `TernaryInitializer` places PROTAC near E3 pocket and POI in 30-60Å shell at t=1. |
| `merging.py` | `merge_direct_addition`, `LearnedCoordinationLayer` | Score merging. Default is direct addition (MVP); `LearnedCoordinationLayer` with attention-based weighting is Ablation B. |
| `reflect.py` | `tricoop_reflect`, `detect_clashes`, `detect_bond_anomalies`, `detect_linker_strain`, `local_minimization` | Post-processing: KD-tree clash detection (VDW radii), bond length anomaly check, linker torsional strain detection, simple repulsive gradient descent minimization (max 3 rounds). MMFF94 integration requires RDKit. |
| `train.py` | `TriCoopDiffTrainer` | Multi-task loss (score + physical + SE3), two-phase training (Phase 1: FiLM+SE3 only, Phase 2: unfreeze all), AdamW + CosineAnnealingLR, gradient clipping, checkpoint save/load. |
| `evaluate.py` | `compute_dockq`, `compute_rmsd`, `compute_rmsd_by_region`, `compute_clash_rate`, `compute_ensemble_energy_probability_correlation`, `compute_ensemble_diversity`, `compute_cooperativity_ranking`, `kabsch_align` | Structure quality (DockQ, RMSD, clash rate), ensemble quality (Spearman ρ, diversity, clustering), functional ranking (α-ranking, Top-20% hit rate, AUROC). |

### Diffusion Variables

| DOF | Dim | Noise Source | Score Source |
|-----|-----|-------------|-------------|
| PROTAC translation | 3 | `add_noise_protac` | Merged from both subgraphs |
| PROTAC rotation | 3 (so(3)) | `add_noise_protac` | Merged from both subgraphs |
| PROTAC torsion | K | `add_noise_protac` | Merged from both subgraphs |
| E3-POI rigid body | 6 (3T+3R) | `add_noise_se3` | `s_se3` from TriBrain SE3 head |

Reference frame: E3 fixed at origin (COM=0). PROTAC coords sampled once in E3 frame, transformed to POI frame for subgraph 2 via `transform_ligand_to_poi_frame`.

## Running Commands

### Data Preparation

```bash
# Data preparation (Phase 0)
python scripts/prepare_data.py --step all --pdb-dir data/pdb_mmcif --protac-dir data/protac_raw --output-dir data

# Individual data steps
python scripts/prepare_data.py --step query    # Query RCSB PDB for ternary complexes
python scripts/prepare_data.py --step filter   # Filter raw PDB lists to confirmed ternaries
python scripts/prepare_data.py --step cluster  # Sequence identity clustering (40%)
python scripts/prepare_data.py --step protac   # Prepare PROTAC benchmark dataset
python scripts/prepare_data.py --step pseudo   # Generate pseudo-ternary data (fallback)
```

### DeepTernary Baseline Audit

The audit evaluates DeepTernary on the PROTAC benchmark (22 structures, 40 seeds each) to establish a baseline for TriCoopDiff comparison.

```bash
# One-shot SLURM submission (GPU for Phase 1, CPU for Phase 2)
sbatch scripts/audit_slurm.sh

# Or manual two-phase run:
# Phase 1 — predictions (needs GPU, ~3-6 hours)
python scripts/audit_run_predictions.py \
    --work-dir DeepTernary/output/checkpoints/PROTAC \
    --out-dir audit_results --seeds 40 --device cuda

# Phase 2 — metrics (CPU, ~5-10 minutes)
python scripts/audit_compute_metrics.py \
    --pred-dir audit_results --output deepternary_audit.json

# Resume interrupted predictions
python scripts/audit_run_predictions.py --resume-from audit_results/prediction_progress.json

# Single case test (~5 min)
python scripts/audit_run_predictions.py --case 5T35_D_A_759 --out-dir audit_results --seeds 10 --device cuda
```

Output: `deepternary_audit.json` (baseline metrics), `audit_results/` (per-case PDBs and CSV summaries), `logs/` (SLURM stdout/stderr).

Full instructions: [scripts/README_audit.md](scripts/README_audit.md)

### Training & Evaluation

```bash
# Training (Phase 1-2)
# Requires: DiffDock pretrained weights at checkpoints/best_model.pt
python tricoopdiff/train.py  # (CLI interface TBD — currently import-only)

# Evaluation
python tricoopdiff/evaluate.py  # (CLI interface TBD)
```

## Known Incomplete / Placeholder Code

These functions are skeletons that need real implementations before training:

- **`train.py:_compute_tor_score_target`** (line 222): Returns zeros. Needs DiffDock's `torus.score()` integration.
- **`train.py:_physical_loss`** (line 236): Simplified center-of-mass clash penalty. Should use per-atom VDW repulsion via KD-tree.
- **`model.py:zero_film_verification`** (line 337): Skeleton — returns `True` without actually verifying. **Critical** to implement before any training to confirm FiLM transparency.
- **`model.py:forward`** (line 58): Currently single-complex mode only (B=1). Multi-complex batching needs DataLoader/padding.
- **`model.py:forward`** (line 110): Uses identity rotation for `ligand_coords_poi` transform. Needs proper SE(3) tracking during diffusion (current `R_E3→POI` estimate).
- **`evaluate.py:evaluate_model`** (line 314): Skeleton placeholder. Needs actual multi-walker sampling loop.
- **`scripts/prepare_data.py:cluster_by_sequence_identity`**: Uses greedy method as fallback. Production should use MMseqs2.
- **`reflect.py:local_minimization`** (line 159): Simple repulsive gradient descent. Production should use RDKit MMFF94.

## Critical Design Constraints

- **Two subgraphs must be physically disconnected**: No protein-protein edges, no ligand-mediated bridging. Each subgraph = exactly one protein + PROTAC. Cross-edges are built dynamically by CGModel per forward pass based on distance cutoffs.
- **PROTAC coordinate sync**: PROTAC is sampled once globally (E3 reference frame). Subgraph 1 uses it directly; Subgraph 2 transforms via current T_E3→POI estimate. Never independently initialize PROTAC in each subgraph.
- **FiLM zero-init is mandatory**: `FiLMHead.__init__` (in `tribrain.py`) uses `nn.init.zeros_` on weight+bias. At training start, `h' = h*(1+0)+0 = h`. Verify with `zero_film_verification()` before training.
- **Gradient flow**: `loss_phys` backprops through `_estimate_denoised()` (Tweedie's formula) → predicted scores → model.forward() → TriBrain. This creates a direct gradient path to TriBrain, NOT through the FiLM indirect path. Two-body expert has `requires_grad=False` on all params.
- **Score merging**: Direct addition `s_q = s_q1 + s_q2` is the MVP. `LearnedCoordinationLayer` in `merging.py` is Ablation B — only use when MVP works.
- **Multi-rate λ**: Stepwise threshold functions (non-differentiable at t=0.7 and t=0.5). This is acceptable as an engineering approximation.
- **Diffusion `t`**: This is a noise schedule variable in [0,1], NOT physical time.
- **`_expert_forward_with_film` bypasses `step_conv_layer()`**: Despite `FrozenDiffDock` exposing `step_conv_layer()` for orchestrated per-layer calls, `model.py:_expert_forward_with_film` directly accesses `model.conv_layers[l]` and replicates the full CGModel.forward() logic inline. This is intentional — it avoids refactoring the complex edge-group splitting logic that differs between intermediate and final conv layers.

## Code Duplication to Be Aware Of

- **`so3_score_norm`** exists in two places: `train.py` (top-level function) and `expert.py` (`FrozenDiffDock.so3_score_norm` method). The `train.py` version is the one actually used at training time. If the DiffDock normalization formula changes, both must be updated.
- **Two FiLM head implementations**: `tribrain.py:FiLMHead` (used in production — simple Linear with zero-init) and `film.py:FiLMLayerHead` (standalone utility with MLP, currently unused by the main model). The `film.py:FiLMSharedMLP` class is also unused — TriBrain has its own `film_shared` sequential MLP inline.

## Training Strategy

- **Phase 1**: Freeze judge/state heads (`trainer.set_phase(1)`), train FiLM heads + SE3 head. 50-100 epochs, batch=32, lr=1e-4.
- **Phase 2**: Unfreeze all TriBrain heads (`trainer.set_phase(2)`). Continue with same hyperparams.
- **PROTAC fine-tuning**: 30-50 epochs, batch=8, lr=2e-5, LoRA rank=4 or strong weight decay.
- Loss weights: score=1.0, phys=0.5, se3=0.3 (configurable in `TriCoopDiffTrainer.__init__`).

## External Dependencies

- **DiffDock** (`github.com/gcorso/DiffDock`): Must be cloned at `../DiffDock`. CGModel accessed via `FrozenDiffDock` wrapper. Key reference files: `models/diffusion.py` (SDE/ODE), `models/score_model.py` (EGNN conv layers), `utils/so3.py` (SO(3) ops), `utils/torus.py` (torsion ops).
- **DeepTernary** (`github.com/youqingxiaozhua/DeepTernary`): Primary baseline. Cloned at `DeepTernary/` (git-ignored). Audit completed; results in `deepternary_audit.json` and `doc/DeepTernary 审计交接文档.md`. Audit scripts in `scripts/audit_*.py`.
- **torch_geometric**: Used for `HeteroData` graph construction. `torch_scatter` for scatter ops.
- **e3nn**: Referenced in `expert.py` for spherical harmonics (`o3.spherical_harmonics`) but used via DiffDock's CGModel internals, not directly by TriCoopDiff.
- **RDKit**: Required for MMFF94 minimization and molecule I/O. Not directly imported in current code (deferred to production).
- **SciPy**: Used in `evaluate.py` (`spearmanr`, `cdist`, `cluster.hierarchy`) and `reflect.py` (`KDTree`, `spatial.transform.Rotation`).

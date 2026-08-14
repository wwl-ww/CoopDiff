# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TriCoopDiff** is a deep learning research project: a physically-constrained diffusion generative model for PROTAC ternary complex structure prediction and functional ranking. The core idea is freezing a pretrained two-body expert (DiffDock, 20M params) and training a lightweight coordination network (TriBrain, ~2.2M params) that modulates the expert via per-layer FiLM, enabling explicit three-body cooperative effects while preserving the expert's pretrained capabilities with zero input-distribution OOD.

**Current status**: Core modules implemented and smoke-tested. The focus has shifted to **baseline auditing and experiment scaffolding**: DiffDock-L baseline inference (Experiment 1), DeepTernary SOTA audit (complete, `deepternary_audit.json`), and OOD verification (Experiment 3). TriCoopDiff training has not yet run — the train/eval entry points exist but several data-loading pieces are still placeholders (see "Known Incomplete / Placeholder Code").

Key design documents in `doc/` (all in Chinese, most recent first):
- `TriCoopDiff 模型交接文档.md` — Model architecture and design rationale (v2.4, canonical)
- `TriCoopDiff 执行计划.md` — 9-week execution plan with phase gates and Go/No-Go criteria
- `TriCoopDiff vs DeepTernary 竞争定位与实验计划.md` — Competitive positioning vs DeepTernary
- `三分子初始化策略方案.md` — Three-molecule initialization strategy (t=1 prior)
- `DeepTernary 论文分析与审计对比.md` — DeepTernary paper analysis + audit comparison (2026-08-08)
- `DeepTernary 审计交接文档.md` / `DeepTernary 审计完成交接文档.md` — baseline audit handoff + results
- `DeepTernary_paper.md` / `DeepTernary_paper.pdf` — DeepTernary full paper (arXiv:2502.18875)
- `DiffDock 审计文档.md` — DiffDock source + paper (Corso et al. ICLR 2023) analysis
- `DiffDock 实验交接文档.md` — DiffDock experiment handoff
- `Experiment1 失败案例交接文档.md` — Experiment 1 NaN failure-case investigation + fixes

## Build / Test / Lint

**There is no build system or linter.** No `setup.py`, `pyproject.toml`, `requirements.txt`, or `Makefile`. Modules are imported directly by adding the repo root to `sys.path`.

**Smoke tests exist but are not a pytest suite** — `tests/test_model_build.py` is a plain script with `test_*()` functions and a `__main__` runner:

```bash
python tests/test_model_build.py
```

It verifies (in order): TriBrain build/shape, FiLM zero-init transparency, diffusion schedules, graph construction, TriReflect clash detection, and score merging. It does NOT require DiffDock to be present. Run it after any change to `tricoopdiff/` to catch regressions.

## Environment & Setup

Requires DiffDock **cloned inside the repo** (git-ignored). Scripts locate it at `./DiffDock` (repo root), NOT `../DiffDock`:

```bash
conda create -n tricoopdiff python=3.10
pip install torch torch_geometric e3nn torchdiffeq torch_scatter rdkit biopython scipy scikit-learn transformers
git clone https://github.com/gcorso/DiffDock.git   # into ./DiffDock (repo root)
# Download DiffDock-L pretrained weights to checkpoints/
```

`DeepTernary/` is also git-ignored and cloned separately from `github.com/youqingxiaozhua/DeepTernary` for baseline comparisons.

**HF endpoint**: `tricoopdiff/esm_embedder.py` sets `HF_ENDPOINT=https://hf-mirror.com` if unset — the cluster cannot reach huggingface.co directly. ESM2 uses `local_files_only=True`, so models must be pre-downloaded. See `scripts/generate_esm_embeddings.py` + `scripts/submit_esm_embeddings.slurm`.

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

DiffDock-L v1.1 (frozen): ns=48, nv=10, num_conv_layers=6, ~20M params (Corso et al. ICLR 2023). Two physically disconnected subgraphs with zero input-distribution OOD.

### DiffDock-L Paper Parameters

| Parameter | Value |
|-----------|-------|
| ns / nv | 48 / 10 |
| num_conv_layers | 6 |
| sigma_embed_dim / distance_embed_dim / cross_distance_embed_dim | 64 / 64 / 64 |
| dropout | 0.1 |
| receptor_radius | 15 |
| c_alpha_max_neighbors | 24 |
| tr_sigma | 0.1 → 19 |
| rot/tor_sigma | 0.03 → 1.55 |
| train batch_size / lr / epochs | 16 / 1e-3 (Plateau) / 850 |

### Modules and What They Do

`tricoopdiff/` package (only `__init__.py` exports are importable from top-level: `TriCoopDiff`, `build_subgraph_e3l`, `build_subgraph_pol`, `build_tribrain_graph`, `MultiRateDiffusion` — everything else is imported as `tricoopdiff.<module>`):

| Module | Key classes/functions | Purpose |
|--------|----------------------|---------|
| `graphs.py` | `build_subgraph_e3l`, `build_subgraph_pol`, `build_tribrain_graph`, `transform_ligand_to_poi_frame` | Construct two disconnected protein+ligand HeteroData subgraphs + one flat ternary dict. Cross-edges built dynamically by CGModel at each forward pass. |
| `tribrain.py` | `TriBrain`, `MultiScalePooling`, `EquivariantTransformer`, `EquivariantTransformerLayer`, `SE3ScoreHead`, `JudgeHead`, `StateHead`, `FiLMHead` | The ONLY trainable module. Pools ternary graph → equivariant transformer → FiLM params (12 heads: 6 layers × 2 subgraphs) + SE3 score + confidence + state. FiLM heads zero-initialized (adaLN-Zero). ~2.2M params in MVP config. |
| `expert.py` | `FrozenDiffDock` | Wraps DiffDock CGModel, freezes all params, exposes `embedding()`, `step_conv_layer()`, `compute_scores()`, `build_cross_graph()`, `build_center_graph()`, `build_bond_graph()`. |
| `esm_embedder.py` | `ESMEmbedder` | Extracts per-residue ESM2-650M embeddings (1280-dim) for E3/POI sequences via HuggingFace transformers. `embed_from_fasta`, `embed_from_pdb`, `embed_from_sequence_file`. |
| `film.py` | `FiLMSharedMLP`, `FiLMLayerHead`, `apply_film` | Standalone FiLM utilities. Note: these are NOT used by the main model — TriBrain has its own inline `FiLMHead` and `film_shared` MLP in `tribrain.py`. Only `apply_film` from this module is actually called (by `model.py:_expert_forward_with_film`). |
| `model.py` | `TriCoopDiff`, `_expert_forward_with_film`, `zero_film_verification` | Top-level orchestrator: builds subgraphs, runs TriBrain, runs both expert passes with per-layer FiLM, merges scores. `_expert_forward_with_film` replicates CGModel.forward() internals by directly accessing `model.conv_layers[l]` (it does NOT use `FrozenDiffDock.step_conv_layer()`). |
| `diffusion.py` | `MultiRateDiffusion`, `TernaryInitializer`, `so3_score_norm`, `so3_exp`, `sample_so3_noise`, `vector_to_skew` | Multi-rate noise schedules (PROTAC fast λ_q, SE3 slow λ_SE3), forward/reverse SDE steps, SO(3) Lie algebra utilities. `TernaryInitializer` places PROTAC near E3 pocket and POI in 30-60Å shell at t=1. **`so3_score_norm` lives here and is the single source of truth** (imported by `train.py` and `expert.py`). |
| `merging.py` | `merge_direct_addition`, `LearnedCoordinationLayer` | Score merging. Default is direct addition (MVP); `LearnedCoordinationLayer` with attention-based weighting is Ablation B. |
| `reflect.py` | `tricoop_reflect`, `detect_clashes`, `detect_bond_anomalies`, `detect_linker_strain`, `local_minimization` | Post-processing: KD-tree clash detection (VDW radii), bond length anomaly check, linker torsional strain detection, simple repulsive gradient descent minimization (max 3 rounds). MMFF94 integration requires RDKit. |
| `train.py` | `TriCoopDiffTrainer` | Multi-task loss (score + physical + SE3), two-phase training (Phase 1: FiLM+SE3 only, Phase 2: unfreeze all), AdamW + CosineAnnealingLR, gradient clipping, checkpoint save/load. |
| `evaluate.py` | `compute_dockq`, `compute_rmsd`, `compute_rmsd_by_region`, `compute_clash_rate`, `compute_ensemble_energy_probability_correlation`, `compute_ensemble_diversity`, `compute_cooperativity_ranking`, `kabsch_align` | Structure quality (DockQ, RMSD, clash rate), ensemble quality (Spearman ρ, diversity, clustering), functional ranking (α-ranking, Top-20% hit rate, AUROC). |

`scripts/` entry points and experiment drivers:
- `inference.py` — full inference pipeline (parse PDB/SDF → init → reverse diffusion → TriReflect → PDB output). The canonical reference for how all pieces fit together.
- `train_entry.py` — training CLI (builds model/trainer, data loader, train loop).
- `eval_entry.py` — evaluation CLI (multi-walker sampling → metrics report).
- `prepare_data.py` — data preparation (query/filter/cluster/protac/pseudo steps).
- `experiment1_diffdock_baseline.py` — Experiment 1: frozen DiffDock-L baseline inference on the PROTAC benchmark (46 e3l/poil pairs → DockQ/RMSD/clash).
- `experiment3_ood_verify.py` — Experiment 3: input-distribution OOD verification (NaN-aware).
- `verify_sota_weights.py` / `test_official_predict.py` — DeepTernary SOTA weight verification.
- `compare_audit_results.py` — audit comparison + variance test + DiffDock baseline comparison.
- `generate_esm_embeddings.py` — batch ESM2 embedding generation.

### Diffusion Variables

| DOF | Dim | Noise Source | Score Source |
|-----|-----|-------------|-------------|
| PROTAC translation | 3 | `add_noise_protac` | Merged from both subgraphs |
| PROTAC rotation | 3 (so(3)) | `add_noise_protac` | Merged from both subgraphs |
| PROTAC torsion | K | `add_noise_protac` | Merged from both subgraphs |
| E3-POI rigid body | 6 (3T+3R) | `add_noise_se3` | `s_se3` from TriBrain SE3 head |

Reference frame: E3 fixed at origin (COM=0). PROTAC coords sampled once in E3 frame, transformed to POI frame for subgraph 2 via `transform_ligand_to_poi_frame`.

## Running Commands

### Smoke Test

```bash
python tests/test_model_build.py
```

### Inference (needs a trained checkpoint + DiffDock weights)

```bash
python scripts/inference.py \
    --e3_pdb data/protac_cases/5t35/e3.pdb \
    --poi_pdb data/protac_cases/5t35/poi.pdb \
    --lig_sdf data/protac_cases/5t35/protac.sdf \
    --checkpoint checkpoints/tricoopdiff_ft.pt \
    --outdir results/5t35 --n_walkers 32 --steps 20
# Ablations: --no_film (L0), --no_trireflect, --single_rate (ablation C)
```

### Training / Evaluation (Phase 1 & 2)

```bash
python scripts/train_entry.py \
    --train_data data/ternary_train.json --val_data data/ternary_val.json \
    --output_dir checkpoints/phase1 --epochs 100 --batch_size 32 --lr 1e-4

python scripts/eval_entry.py \
    --data data/protac_benchmark_test.json \
    --checkpoint checkpoints/tricoopdiff_ft.pt \
    --outdir results/eval_phase2 --n_walkers 32 --steps 20 \
    --deepternary_audit deepternary_audit.json
```

### Data Preparation

```bash
python scripts/prepare_data.py --step all --pdb-dir data/pdb_mmcif --protac-dir data/protac_raw --output-dir data
```

### DeepTernary Baseline Audit

```bash
sbatch scripts/audit_slurm.sh            # one-shot SLURM (GPU Phase 1, CPU Phase 2)
python scripts/audit_run_predictions.py --work-dir DeepTernary/output/checkpoints/PROTAC --out-dir audit_results --seeds 40 --device cuda
python scripts/audit_compute_metrics.py --pred-dir audit_results --output deepternary_audit.json
```

Output: `deepternary_audit.json` (baseline metrics), `audit_results/` (per-case PDBs/CSVs), `logs/` (SLURM stdout/stderr). Full instructions: [scripts/README_audit.md](scripts/README_audit.md).

## Known Incomplete / Placeholder Code

These functions are skeletons that need real implementations before end-to-end training/sampling can run correctly:

- **`train.py:_compute_tor_score_target`** (line 188): Returns zeros. Needs DiffDock's `torus.score()` integration.
- **`train.py:_physical_loss`** (line 206): Simplified center-of-mass clash penalty. Should use per-atom VDW repulsion via KD-tree.
- **`model.py:zero_film_verification`** (line 357): Skeleton — returns `True` without actually verifying. **Critical** to implement before any training to confirm FiLM transparency.
- **`model.py:forward`** (line 68): Iterates per-complex in a Python loop (no true batching), and uses identity rotation for `ligand_coords_poi` transform (line 121). Needs proper SE(3) tracking during diffusion (current `R_E3→POI` estimate).
- **`evaluate.py:evaluate_model`** (line 389): Skeleton — returns empty dict. Needs actual multi-walker sampling loop.
- **`scripts/train_entry.py:TernaryDataLoader._collate`** (line 287): Returns dummy random tensors. The training CLI will not train on real data until this loads HeteroData from disk.
- **`scripts/inference.py:parse_protein`** (line 356): Protein features are a placeholder (zeros, `[N,1]`). Production needs DiffDock's `process_mols` feature extraction or ESM2 embeddings from `esm_embedder.py`.
- **`scripts/prepare_data.py:cluster_by_sequence_identity`**: Uses greedy method as fallback. Production should use MMseqs2.
- **`reflect.py:local_minimization`** (line 159): Simple repulsive gradient descent. Production should use RDKit MMFF94.

## Critical Design Constraints

- **Two subgraphs must be physically disconnected**: No protein-protein edges, no ligand-mediated bridging. Each subgraph = exactly one protein + PROTAC. Cross-edges are built dynamically by CGModel per forward pass based on distance cutoffs.
- **PROTAC coordinate sync**: PROTAC is sampled once globally (E3 reference frame). Subgraph 1 uses it directly; Subgraph 2 transforms via current T_E3→POI estimate. Never independently initialize PROTAC in each subgraph.
- **FiLM zero-init is mandatory**: `FiLMHead.__init__` (in `tribrain.py`) uses `nn.init.zeros_` on weight+bias. At training start, `h' = h*(1+0)+0 = h`. Verify with `zero_film_verification()` before training (and note this verification itself is currently a stub).
- **Gradient flow**: `loss_phys` backprops through `_estimate_denoised()` (Tweedie's formula) → predicted scores → model.forward() → TriBrain. This creates a direct gradient path to TriBrain, NOT through the FiLM indirect path. Two-body expert has `requires_grad=False` on all params.
- **Score merging**: Direct addition `s_q = s_q1 + s_q2` is the MVP. `LearnedCoordinationLayer` in `merging.py` is Ablation B — only use when MVP works.
- **Multi-rate λ**: Stepwise threshold functions (non-differentiable at t=0.7 and t=0.5). This is acceptable as an engineering approximation.
- **Diffusion `t`**: This is a noise schedule variable in [0,1], NOT physical time.
- **`_expert_forward_with_film` bypasses `step_conv_layer()`**: Despite `FrozenDiffDock` exposing `step_conv_layer()` for orchestrated per-layer calls, `model.py:_expert_forward_with_film` directly accesses `model.conv_layers[l]` and replicates the full CGModel.forward() logic inline. This is intentional — it avoids refactoring the complex edge-group splitting logic that differs between intermediate and final conv layers.

## Code Duplication to Be Aware Of

- **Two FiLM head implementations**: `tribrain.py:FiLMHead` (used in production — simple Linear with zero-init) and `film.py:FiLMLayerHead` (standalone utility with MLP, currently unused by the main model). The `film.py:FiLMSharedMLP` class is also unused — TriBrain has its own `film_shared` sequential MLP inline.

## Training Strategy

- **Phase 1**: Freeze judge/state heads (`trainer.set_phase(1)`), train FiLM heads + SE3 head. 50-100 epochs, batch=32, lr=1e-4.
- **Phase 2**: Unfreeze all TriBrain heads (`trainer.set_phase(2)`). Continue with same hyperparams.
- **PROTAC fine-tuning**: 30-50 epochs, batch=8, lr=2e-5, LoRA rank=4 or strong weight decay.
- Loss weights: score=1.0, phys=0.5, se3=0.3 (configurable in `TriCoopDiffTrainer.__init__`).

## External Dependencies

- **DiffDock** (`github.com/gcorso/DiffDock`): Cloned **inside the repo at `./DiffDock`** (git-ignored). CGModel accessed via `FrozenDiffDock` wrapper. Key reference files: `models/diffusion.py` (SDE/ODE), `models/score_model.py` (EGNN conv layers), `utils/so3.py` (SO(3) ops), `utils/torus.py` (torsion ops).
- **DeepTernary** (`github.com/youqingxiaozhua/DeepTernary`): Primary baseline. Cloned at `./DeepTernary` (git-ignored). Audit completed; results in `deepternary_audit.json` and `doc/DeepTernary 审计完成交接文档.md`. Audit scripts in `scripts/audit_*.py`.
- **ESM2** via `transformers` (`facebook/esm2_t33_650M_UR50D`, 1280-dim): per-residue protein embeddings. Requires `HF_ENDPOINT` mirror and `local_files_only=True` (see `esm_embedder.py`).
- **torch_geometric**: `HeteroData` graph construction. `torch_scatter` for scatter ops.
- **e3nn**: Referenced in `expert.py` for spherical harmonics but used via DiffDock's CGModel internals, not directly by TriCoopDiff.
- **RDKit**: Required for MMFF94 minimization and molecule I/O (ligand SDF parsing in `inference.py`).
- **BioPython**: PDB/mmCIF/FASTA parsing (`esm_embedder.py`, `inference.py`).
- **SciPy**: `evaluate.py` (`spearmanr`, `cdist`, `cluster.hierarchy`) and `reflect.py` (`KDTree`, `spatial.transform.Rotation`).

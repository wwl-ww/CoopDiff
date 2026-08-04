# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TriCoopDiff** is a deep learning research project: a physically-constrained diffusion generative model for PROTAC ternary complex structure prediction and functional ranking. The core idea is freezing a pretrained two-body expert (DiffDock, 20M params) and training a lightweight coordination network (TriBrain, ~2.2M params) that modulates the expert via per-layer FiLM, enabling explicit three-body cooperative effects while preserving the expert's pretrained capabilities with zero input-distribution OOD.

Key documents (all in Chinese):
- `TriCoopDiff 模型交接文档.md` — Complete model architecture and design rationale (v2.4)
- `TriCoopDiff 执行计划.md` — 9-week execution plan with phase gates, Go/No-Go criteria, and file-level implementation specs
- `TriCoopDiff vs DeepTernary 竞争定位与实验计划.md` — Competitive positioning vs DeepTernary baseline
- `三分子初始化策略方案.md` — Three-molecule initialization strategy for diffusion

**Current status**: Planning phase complete. No code written yet, no git commits.

## Environment Setup

```bash
conda create -n tricoopdiff python=3.10
pip install torch e3nn torchdiffeq rdkit biopython
git clone https://github.com/gcorso/DiffDock.git
# Download DiffDock-L pretrained weights
```

## Architecture

### Core Design (from 模型交接文档 §2)

1. **Two-body expert (frozen)**: DiffDock-L v1.1 split into two physically disconnected subgraphs:
   - Subgraph 1 (E3-L): E3 residues + PROTAC atoms only — no POI nodes
   - Subgraph 2 (POI-L): POI residues + PROTAC atoms only — no E3 nodes
   - Each subgraph receives standard protein-ligand docking input (zero OOD)
   - All 20M params frozen, no fine-tuning

2. **TriBrain (~2.2M trainable)**: Operates on the full ternary graph (E3+POI+PROTAC, all edges including E3-POI).
   - MultiScalePooling → EquivariantTransformer (3 layers, single-head cross-attention, 32 scalar + 4 vector channels) → SE3 head + judge head + state head
   - Outputs per-layer FiLM parameters (γ, β) via 12 independent heads (6 layers × 2 subgraphs), zero-initialized (adaLN-Zero paradigm)
   - Also outputs E3-POI rigid body scores (s_SE3)

3. **FiLM modulation**: Each DiffDock Conv layer output `h` becomes `h' = h * (1+γ) + β`. At initialization, FiLM is transparent (output numerically identical to native DiffDock).

4. **Multi-rate diffusion**: PROTAC degrees of freedom use fast noise decay (λ_q, threshold at t=0.7); E3-POI rigid body uses slow decay (λ_SE3, threshold at t=0.5). This is an empirical noise schedule trick, not a strict SDE derivation.

5. **TriReflect post-processing**: KD-tree clash detection, bond anomaly detection, linker strain detection, local MMFF94 minimization (max 3 rounds).

6. **Multi-walker sampling**: 32 independent diffusion trajectories → conformational ensemble → population ratios → cooperativity ranking.

### Diffusion Variables

| DOF | Symbol | Dim | Module |
|-----|--------|-----|--------|
| PROTAC translation | t_q | 3 | DiffDock |
| PROTAC rotation | R_q | 3 (so(3)) | DiffDock |
| PROTAC torsion angles | χ | K (rotatable bonds) | DiffDock |
| E3-POI rigid body | SE3 | 6 (3 trans + 3 rot) | TriBrain |

Reference frame: E3 fixed at origin (COM=0). PROTAC coordinates are globally sampled once, then transformed into each subgraph's reference frame.

### Planned Package Structure

```
tricoopdiff/
  graphs.py      — Subgraph construction (E3-L, POI-L) + full ternary graph
  tribrain.py    — MultiScalePooling, EquivariantTransformer, TriBrain, FiLM heads
  expert.py      — FrozenDiffDock wrapper exposing per-layer Conv outputs
  film.py        — apply_film(), FiLM parameter generation
  diffusion.py   — SDE/ODE diffusion process, multi-rate noise schedules
  model.py       — TriCoopDiff top-level module
  merging.py     — Score merging from two subgraphs
  reflect.py     — TriReflect: clash detection, bond checks, local minimization
  train.py       — Training loop with score loss + physical loss + SE3 loss
  evaluate.py    — DockQ, RMSD, clash rate, linker strain, ensemble quality metrics
```

## Key External Dependencies

- **DiffDock** (`github.com/gcorso/DiffDock`): Foundation model. Key files to understand before modifying: `datasets/pdbbind.py` (graph construction reference), `models/diffusion.py` (SDE/ODE), `models/score_model.py` (EGNN score network — FiLM injection point), `utils/sampling.py` (multi-walker reference).
- **DeepTernary** (`github.com/youqingxiaozhua/DeepTernary`): Primary baseline for structural comparison. Audit before Phase 2.

## Execution Plan (9 weeks)

| Phase | Week | Goal | Key Decision Gate |
|-------|------|------|-------------------|
| 0 | 1 | Environment + data prep + DeepTernary audit | Data < 1000 → adjust expectations |
| 1 | 2-3 | MVP: end-to-end ternary structure generation | Runs or not → debug architecture |
| 2 | 4-5 | Training + structure validation (FT DockQ ≥ 0.5) | DockQ < 0.4 → investigate architecture |
| 3 | 6 | Ensemble quality (energy-probability ρ > 0.3) | ρ < 0.2 → skip functional ranking |
| 4 | 7 | Functional ranking (α-ranking ρ > 0.4) | ρ < 0.3 → downgrade to structure-only |
| 5 | 8-9 | Ablation (6 core) + paper draft | — |

## Critical Design Constraints

- **The two subgraphs must be physically disconnected** — no protein-protein edges, no ligand-mediated bridging. Each subgraph contains exactly one protein + PROTAC, matching DiffDock's pretraining input distribution exactly.
- **PROTAC coordinate sync**: PROTAC is sampled once globally (in E3 reference frame). Subgraph 1 uses it directly; Subgraph 2 transforms via current T_E3→POI estimate. Never independently initialize PROTAC in each subgraph.
- **FiLM zero-init is mandatory**: At training start, FiLM must be numerically transparent — output must match native DiffDock exactly. Verify this before any training.
- **Gradient flow**: Physical loss gradients flow through `s_tri_residual → TriBrain`, not through the FiLM indirect path. Two-body expert receives no gradients.
- **MVP-first philosophy**: Build the minimal version first (3-layer TriBrain, 32 channels, simple FiLM), verify it works end-to-end, then upgrade capacity if needed (6 layers, 64 channels).

## Important Implementation Notes

- Score merging from two subgraphs is **direct addition**: `s_q = s_q1 + s_q2`. A learned coordination layer is an ablation upgrade (Ablation B).
- Multi-rate diffusion λ functions use stepwise thresholds — these are non-differentiable at the threshold, which is acceptable as an engineering approximation.
- The diffusion `t` parameter is a noise schedule variable, NOT physical time.
- Training strategy: Phase 1-2 freeze judge/state heads, train only FiLM heads + SE3 head. Unfreeze remaining heads later.
- PROTAC fine-tuning uses LoRA (rank=4) or strong weight decay; batch size drops to 8.

# TriFuse v13 架构与审计

## 模型架构

```mermaid
flowchart LR
    E3[E3 residues] --> G[Typed geometry encoder]
    POI[POI residues] --> G
    L[PROTAC atoms + covalent graph] --> G
    G --> T[5-token negotiator<br/>global · E3-L · POI-L · linker · diffusion]
    T --> F1[Per-atom, per-layer FiLM: E3-L]
    T --> F2[Per-atom, per-layer FiLM: POI-L]
    X1[Frozen DiffDock + sub1 LoRA] --> B[Lock-step scalar bridge]
    X2[Frozen DiffDock + sub2 LoRA] --> B
    F1 --> B
    F2 --> B
    B --> S1[E3-L translation / rotation / torsion]
    B --> S2[POI-L translation / rotation / torsion]
    S1 --> C[Per-bond torsion coordinator]
    S2 --> C
    B --> M[Invariant layer telemetry]
    M --> C
    S1 --> R[Joint coordinate update]
    S2 --> R
    C --> R
```

TriFuse first encodes protein identities, ligand chemistry, covalent neighbors,
ligand-centric contacts, distance distributions, interface angles and signed
handedness. Five compact invariant tokens negotiate the two interfaces and the
physical diffusion state. They generate atom-specific scalar FiLM for each of
11 layers in each expert. The two DiffDock streams then advance in lock-step; a
zero-initialized bounded bridge exchanges only aligned ligand scalar irreps.
Independent LoRA adapters let the two interface experts specialize.

Rigid Cartesian directions always come from the equivariant experts. TriFuse
only calibrates their bounded magnitudes. A separate per-bond head combines both
torsion opinions using local bond context and current expert telemetry. During
sampling, E3 remains fixed, the ligand follows the E3-L proposal, and POI receives
the exact rigid transform that maps the POI-L proposal onto that ligand pose.

## Frames and training contract

E3 defines the global frame. The POI-L graph is rebuilt in the current POI body
frame, and its vector scores are rotated back before use. Training applies two
native DiffDock forward marginals at one shared diffusion clock, then reconstructs
one consistent ternary pose. Phase 1 optimizes both interface score losses, joint
and auxiliary torsion losses, TriFuse, and—after warmup—branch-specific LoRA.
Judge/state heads remain frozen because no labels or losses currently exist.

## Known limitations

- `coupled5a` and `anchor_local` are stable search proposals, not exact forward
  marginals. Their `t_anchor` values therefore remain empirical.
- Protein context uses residue-level coordinates; side-chain orientation and
  explicit backbone frames are not represented inside TriFuse.
- Hard nearest-neighbor contact selection is discontinuous when neighbors swap.
- Confidence and productive-state heads are intentionally unsupervised/frozen;
  walkers are ranked by downstream structural metrics, not learned confidence.
- Evaluation runs walkers sequentially and has no batched graph execution.
- Validation so far is structural/unit testing plus small GPU smoke runs, not a
  converged full-dataset accuracy claim.

## Non-negotiable invariants

Do not exchange vector irreps between expert frames. Do not replace non-finite
scores with zeros. Both experts must score the same pre-update ligand state, all
POI coordinates/frame metadata must receive the same rigid transform, and new
checkpoints must satisfy the exact architecture/runtime contract.

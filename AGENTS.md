# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `tricoopdiff/`. The main orchestration, diffusion, sampling, training, and evaluation modules are at that package root; the frozen, vendored DiffDock implementation is under `tricoopdiff/diffdock/`. Executable data preparation, diagnostics, local GPU training, and evaluation entry points live in `scripts/`; older SLURM wrappers are reference material. Keep lightweight automated checks in `tests/`. Research notes and architecture handoffs are in `doc/`; consult `doc/README.md` before relying on older notes. Treat `data/`, `checkpoints/`, `results/`, and `logs/` as generated artifacts, not source code.

## Build, Test, and Development Commands

There is no package build system or project-level dependency manifest. Run commands from the repository root because SO(3)/torus cache files are resolved relative to the current directory.

```bash
python tests/test_model_build.py
python tests/test_training_contract.py
python scripts/make_deepternary_index.py
python scripts/audit_ternary_inputs.py --index data/ternary_train.json --out results/input_audit.json
scripts/launch_dual_init_factorial_local.sh 3 5 7 /mnt/disk/wl/MoRE-tri/CoopDiff
python scripts/evaluate_entry.py --case 5T35_D_A_759 --n-walkers 32 --out results/eval.json
```

The first two commands run fast assertion-based model and training-contract checks. The index command creates the official DeepTernary indexes. Training and evaluation require configured checkpoints and physical GPU IDs. This repository runs directly on local GPUs; do not introduce SLURM commands. The production B/C path keeps both DiffDock experts frozen and does not support LoRA.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, PEP 8 naming, `snake_case` functions and modules, `PascalCase` classes, and uppercase constants. Add type hints and short docstrings where they clarify tensor shapes, coordinate frames, or non-obvious behavior. Keep imports ordered standard library, third party, then local. No formatter or linter is configured, so preserve surrounding style and keep changes focused. Import `tricoopdiff.diffdock_compat` before vendored DiffDock model code when adding entry points.

## Testing Guidelines

Tests use plain functions and `assert`, although `test_*.py` naming remains pytest-compatible. Name new checks `test_<behavior>` and make failures describe the violated invariant. Run `python tests/test_model_build.py` for model changes and `python tests/test_prepare_data.py` for data-pipeline changes. Use the DeepTernary smoke for gradient-flow or expert-integration changes. Avoid tests that silently download models; cluster execution expects cached Hugging Face assets.

## Commit & Pull Request Guidelines

Recent history favors short imperative Conventional Commit subjects such as `fix: ...`, `feat: ...`, and `docs: ...`. Keep each commit scoped and avoid `tmp` messages. Pull requests should summarize the change, motivation, commands run, and affected datasets/checkpoints. Link relevant issues or design documents; include metric deltas and log/result paths for experiment changes, plus screenshots only for visual outputs.

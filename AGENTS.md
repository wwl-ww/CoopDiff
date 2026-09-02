# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `tricoopdiff/`. The main orchestration, diffusion, sampling, training, and evaluation modules are at that package root; the frozen, vendored DiffDock implementation is under `tricoopdiff/diffdock/`. Executable data preparation, diagnostics, training, and evaluation entry points live in `scripts/`, alongside their SLURM wrappers. Keep lightweight automated checks in `tests/`. Research notes and architecture handoffs are in `doc/`; consult `doc/README.md` before relying on older notes. Treat `data/`, `checkpoints/`, `results/`, and `logs/` as datasets or generated artifacts, not source code.

## Build, Test, and Development Commands

There is no package build system or project-level dependency manifest. Run commands from the repository root because SO(3)/torus cache files are resolved relative to the current directory.

```bash
python tests/test_model_build.py
python scripts/make_protac_index.py
python scripts/mvp_smoke_test.py --device cuda --case 5T35_D_A_759 --seed 0
sbatch scripts/train_entry_smoke.slurm
python scripts/evaluate_entry.py --case 5T35_D_A_759 --n-walkers 32 --out results/eval.json
```

The first command runs the fast assertion-based model checks. The index command creates `data/protac_index.json`; smoke, training, and evaluation workflows require the configured model checkpoints and usually a GPU. Do not use legacy `scripts/inference.py` or `scripts/eval_entry.py`.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, PEP 8 naming, `snake_case` functions and modules, `PascalCase` classes, and uppercase constants. Add type hints and short docstrings where they clarify tensor shapes, coordinate frames, or non-obvious behavior. Keep imports ordered standard library, third party, then local. No formatter or linter is configured, so preserve surrounding style and keep changes focused. Import `tricoopdiff.diffdock_compat` before vendored DiffDock model code when adding entry points.

## Testing Guidelines

Tests use plain functions and `assert`, although `test_*.py` naming remains pytest-compatible. Name new checks `test_<behavior>` and make failures describe the violated invariant. Run `python tests/test_model_build.py` for model-level changes and the MVP smoke test for FiLM, gradient-flow, sampling, or expert-integration changes. Avoid tests that silently download models; cluster execution expects cached Hugging Face assets.

## Commit & Pull Request Guidelines

Recent history favors short imperative Conventional Commit subjects such as `fix: ...`, `feat: ...`, and `docs: ...`. Keep each commit scoped and avoid `tmp` messages. Pull requests should summarize the change, motivation, commands run, and affected datasets/checkpoints. Link relevant issues or design documents; include metric deltas and log/result paths for experiment changes, plus screenshots only for visual outputs.

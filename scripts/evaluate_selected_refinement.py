#!/usr/bin/env python
"""Measure whether CoopDiff refinement rescues native-free proposal selection."""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

# Keep assets in this repository while using an explicitly selected model runtime.
# This avoids editing a worktree whose training jobs are still running.
_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--runtime-root")
_bootstrap.add_argument("--project-root")
_runtime_args, _ = _bootstrap.parse_known_args()
RUNTIME_ROOT = Path(_runtime_args.runtime_root or Path(__file__).resolve().parent.parent).resolve()


def configure_runtime_cache(project_root):
    """Keep numerical DiffDock tables tied to the asset repository."""
    root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    os.environ.setdefault(
        'TRICOOPDIFF_CACHE_DIR', str(root / 'results/runtime_cache/diffdock'))


configure_runtime_cache(_runtime_args.project_root)
sys.path.insert(0, str(RUNTIME_ROOT / "scripts"))
sys.path.insert(0, str(RUNTIME_ROOT))
import tricoopdiff.diffdock_compat  # noqa: E402,F401

import numpy as np
import torch

from evaluation_common import evaluation_input_signature, resolve_entry_paths, sha256_file
from evaluation_runner import EvaluationRunner
from featurize_common import featurize_complex
from tricoopdiff.metrics.dockq import compute_fullatom_dockq
from tricoopdiff.initialization.candidates import (
    DualExpertAnchorInitializer, FixedCandidateInitializer,
    generate_hybrid_candidates, select_hybrid_walkers,
)
from tricoopdiff.sampler import sample_single
from tricoopdiff.runtime import atomic_json


ROOT = Path(_runtime_args.project_root or Path(__file__).resolve().parent.parent).resolve()
PROTOCOL = "bitseed_hybrid_selected_refinement_v4"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--index", required=True)
    result.add_argument("--out-dir", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--runtime-root", help="Explicit code worktree; assets remain relative to this script's repository")
    result.add_argument("--project-root", help="Asset/output root when executing a frozen code snapshot")
    result.add_argument("--smoke-one-per-source", action="store_true",
                        help="Refine only the first BiTSeed and first SDPI candidate; verify source at every model call")
    result.add_argument("--cases", help="Comma-separated case IDs; default is the full index")
    result.add_argument("--device", default="cuda")
    result.add_argument("--cache-dir", default="results/featurize_cache_proposal_oracle")
    result.add_argument("--deepternary-root", default="DeepTernary")
    result.add_argument("--native-root", default="DeepTernary/data/TernaryDB/pdbs")
    result.add_argument("--seed", type=int, default=31_000)
    result.add_argument("--steps", type=int, default=20)
    result.add_argument("--binary-steps", type=int, default=20)
    result.add_argument("--conformers", type=int, default=4)
    result.add_argument("--poses-per-side", type=int, default=8)
    result.add_argument("--bitseed-candidates", type=int, default=192)
    result.add_argument("--sdpi-candidates", type=int, default=64)
    result.add_argument("--binary-retry-limit", type=int, default=2)
    result.add_argument("--binary-consecutive-failure-limit", type=int, default=3)
    result.add_argument("--max-refinements", type=int, help="Smoke-only cap on the union")
    result.add_argument(
        "--arms",
        default="sdpi_64,bitseed_diversity_64,geometry_64,diversity_64,hybrid_48_16,hybrid_56_8",
        help="Comma-separated selector arms to refine and report",
    )
    result.add_argument(
        "--refine-full-pool", action="store_true",
        help="Refine all 256 candidates so every selector has a counterfactual final label",
    )
    result.add_argument("--allow-experimental-checkpoint", action="store_true")
    result.add_argument("--allow-conditioned-checkpoint", action="store_true")
    result.add_argument(
        "--condition-manifest-dir",
        help="Directory containing one <case>.json deploy manifest per case",
    )
    result.add_argument("--no-resume", action="store_true",
                        help="Run without reusing case reports; requires no existing case outputs")
    return result




def source(candidate):
    return "sdpi" if candidate["candidate_id"][0] == "sdpi" else "bitseed"


def candidate_hash(candidate):
    digest = hashlib.sha256()
    digest.update(json.dumps(candidate["candidate_id"]).encode())
    for key in ("ligand", "poi", "rotation"):
        value = candidate[key].detach().cpu().contiguous().numpy()
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def exact_dockq(candidate_ligand, candidate_poi, *, e3_data, paths, native,
                deepternary_root, metric_paths):
    with (contextlib.redirect_stdout(io.StringIO()),
          contextlib.redirect_stderr(io.StringIO())):
        return compute_fullatom_dockq(
            pred_ligand=candidate_ligand,
            pred_poi_ca=candidate_poi,
            featurized_e3_ca=e3_data["coords"][0].detach().cpu().numpy(),
            protein1_pdb=str(paths["protein1"]),
            protein2_pdb=str(paths["protein2"]),
            ligand_pdb=str(paths["ligand"]),
            native_complex_pdb=str(native),
            deepternary_root=str(deepternary_root),
            **metric_paths,
        )


def summarize(rows, metric):
    values = [float(row[metric]) for row in rows
              if isinstance(row.get(metric), (int, float))
              and math.isfinite(row[metric])]
    return {
        "count": len(values),
        "best": max(values) if values else None,
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "hits_0.23": sum(value >= 0.23 for value in values),
        "hits_0.49": sum(value >= 0.49 for value in values),
    }


def resolve_case_inputs(entry, project_root, native_root):
    """Resolve model inputs and every structure used by exact scoring."""
    root = Path(project_root)
    paths = resolve_entry_paths(entry, root)
    metric_paths = {}
    for key in ("metric_protein1", "metric_protein2"):
        if key in entry:
            value = Path(entry[key])
            value = value if value.is_absolute() else root / value
            if not value.is_file():
                raise FileNotFoundError(value)
            metric_paths[f"{key}_pdb"] = str(value.resolve())
    if len(metric_paths) not in (0, 2):
        raise ValueError("provide both metric protein templates or neither")
    native_value = entry.get("complex")
    native = (Path(native_value) if native_value else
              root / native_root / entry["case"] / "gt_complex.pdb")
    native = native if native.is_absolute() else root / native
    if not native.is_file():
        raise FileNotFoundError(native)
    return paths, native.resolve(), metric_paths


def preflight_condition_manifests(condition_paths, case_inputs):
    """Validate every deploy manifest against actual graph inputs up front.

    This is intentionally CPU-only and does not read native ternary structures.
    It mirrors DiffDock's ProDy receptor-node selection so a modified residue or
    stale manifest cannot waste completed refinements before failing on shape.
    """
    if set(condition_paths) != set(case_inputs):
        raise ValueError('condition preflight requires one manifest per case')
    from rdkit import Chem
    import prody as pr
    from tricoopdiff.conditioning.condition_features import load_condition_manifest
    from tricoopdiff.diffdock.datasets.parse_chi import get_coords
    from tricoopdiff.diffdock.datasets.process_mols import read_molecule

    result = {}
    for case, condition_path in condition_paths.items():
        paths = case_inputs[case][0]
        molecule = read_molecule(str(paths['ligand']), sanitize=True, remove_hs=True)
        if molecule is None or not molecule.GetNumAtoms():
            raise ValueError(f'{case}: condition preflight could not read ligand')
        sizes = []
        for key in ('protein1', 'protein2'):
            protein = pr.parsePDB(str(paths[key]))
            if protein is None or protein.ca is None:
                raise ValueError(f'{case}: condition preflight found no {key} nodes')
            coords = get_coords(protein)[:, 1, :]
            if (coords.ndim != 2 or coords.shape[1:] != (3,) or not len(coords)
                    or not np.isfinite(coords).all()):
                raise ValueError(f'{case}: condition preflight found invalid {key} nodes')
            sizes.append(len(coords))
        _, audit = load_condition_manifest(
            condition_path, case=case, molecule=Chem.Mol(molecule),
            protein_sizes=tuple(sizes),
        )
        result[case] = {
            'protein_sizes': sizes,
            'ligand_atoms': molecule.GetNumAtoms(),
            'available_sides': audit['available_sides'],
        }
    return result


def validate_resume_output(output, manifest, *, resume=True):
    """Fail before any output writes if existing results cannot be safely reused."""
    output = Path(output)

    def read(path):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot verify {path}; use a new --out-dir") from error
        if not isinstance(value, dict):
            raise ValueError(f"Invalid report {path}; use a new --out-dir")
        return value

    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError("Existing manifest differs or is legacy; use a new --out-dir")
    completed = {}
    for case in manifest["cases"]:
        path = output / f"{case}.json"
        if not path.exists():
            continue
        if not resume:
            raise ValueError(f"{path} already exists; --no-resume requires a new --out-dir")
        report = read(path)
        if report.get("case") != case or report.get("config") != manifest:
            raise ValueError(f"{case}: existing report contract mismatch; use a new --out-dir")
        if report.get("status") == "complete":
            completed[case] = report
    return completed


def refinement_metric_fields(initial, refined=None):
    """The oracle union retains a valid initialization even if refinement fails."""
    for metrics in (initial, refined):
        if metrics is not None and not all(
                math.isfinite(float(metrics[key])) for key in ("dockq", "fnat", "lrms", "irms")):
            raise ValueError("exact scoring returned non-finite metrics")
    fields = {f"initial_{key}": initial[key] for key in ("dockq", "fnat", "lrms", "irms")}
    fields.update(safe_dockq=initial["dockq"], safe_stage="initial")
    if refined is not None:
        fields.update({f"refined_{key}": refined[key]
                       for key in ("dockq", "fnat", "lrms", "irms")})
        fields["refinement_delta_dockq"] = refined["dockq"] - initial["dockq"]
        if refined["dockq"] >= initial["dockq"]:
            fields.update(safe_dockq=refined["dockq"], safe_stage="refined")
    return fields


def main():
    # Some legacy runtime imports change CWD to their own checkout. Keep all
    # relative data/output/cache paths anchored to this asset repository.
    os.chdir(ROOT)
    args = parser().parse_args()
    if args.smoke_one_per_source and args.max_refinements is not None:
        raise ValueError("source smoke cannot be combined with max-refinements")
    if bool(args.allow_conditioned_checkpoint) != bool(args.condition_manifest_dir):
        raise ValueError(
            "conditioned checkpoints require both --allow-conditioned-checkpoint "
            "and --condition-manifest-dir")
    if args.steps < 1 or args.max_refinements is not None and args.max_refinements < 1:
        raise ValueError("steps and max-refinements must be positive")
    requested_arms = [name.strip() for name in args.arms.split(',') if name.strip()]
    known_arms = {
        'sdpi_64', 'bitseed_diversity_64', 'geometry_64',
        'diversity_64', 'hybrid_48_16', 'hybrid_56_8',
    }
    if not requested_arms or len(set(requested_arms)) != len(requested_arms):
        raise ValueError('at least one unique selector arm is required')
    if set(requested_arms) - known_arms:
        raise ValueError(f'unknown selector arms: {sorted(set(requested_arms) - known_arms)}')
    index_path = Path(args.index).resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    if args.cases:
        requested = [case.strip() for case in args.cases.split(",") if case.strip()]
        by_case = {entry["case"]: entry for entry in entries}
        missing = set(requested) - set(by_case)
        if missing:
            raise ValueError(f"case IDs absent from index: {sorted(missing)}")
        entries = [by_case[case] for case in requested]
    if len({entry["case"] for entry in entries}) != len(entries):
        raise ValueError("duplicate case IDs cannot share an output directory")

    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
    deepternary_root = Path(args.deepternary_root)
    if not deepternary_root.is_absolute():
        deepternary_root = (ROOT / deepternary_root).resolve()
    native_root = (ROOT / args.native_root).resolve()
    case_inputs = {
        entry["case"]: resolve_case_inputs(entry, ROOT, native_root)
        for entry in entries
    }
    condition_paths = {}
    condition_preflight = {}
    if args.condition_manifest_dir:
        condition_dir = Path(args.condition_manifest_dir)
        if not condition_dir.is_absolute():
            condition_dir = (ROOT / condition_dir).resolve()
        condition_paths = {
            entry['case']: condition_dir / f'{entry["case"]}.json'
            for entry in entries
        }
        missing = [str(path) for path in condition_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'missing condition manifests: {missing}')
        condition_preflight = preflight_condition_manifests(condition_paths, case_inputs)
    manifest = {
        "protocol": PROTOCOL,
        "scope": "native-free selection followed by oracle-scored CoopDiff refinement",
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "cases": [entry["case"] for entry in entries],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "runtime_root": str(RUNTIME_ROOT),
        "project_root": str(ROOT),
        "native_root": str(native_root),
        "deepternary_root": str(deepternary_root),
        "input_signatures": {
            case: evaluation_input_signature({**paths, **metric_paths}, native)
            for case, (paths, native, metric_paths) in case_inputs.items()
        },
        "expert_assets_sha256": {
            name: sha256_file(ROOT / "checkpoints/score_model" / name)
            for name in ("model_parameters.yml", "best_ema_inference_epoch_model.pt")
        },
        "dockq_source_sha256": {
            str(path.relative_to(deepternary_root)): sha256_file(path)
            for path in sorted((deepternary_root / "DockQ").rglob("*.py"))
        },
        "entrypoint_sha256": sha256_file(Path(__file__)),
        "runtime_source_sha256": {
            name: sha256_file(RUNTIME_ROOT / name) for name in (
                "tricoopdiff/models/consensus.py", "tricoopdiff/sampler.py",
                "tricoopdiff/sdpi_flow.py", "tricoopdiff/initialization/candidates.py",
                "scripts/evaluation_runner.py", "scripts/featurize_common.py",
                "scripts/evaluation_common.py", "tricoopdiff/metrics/dockq.py",
            ) + (("tricoopdiff/conditioning/conditioned_consensus_train.py",
                  "tricoopdiff/conditioning/condition_features.py")
                 if args.allow_conditioned_checkpoint else ())
        },
        "smoke_one_per_source": args.smoke_one_per_source,
        "allow_experimental_checkpoint": args.allow_experimental_checkpoint,
        "allow_conditioned_checkpoint": args.allow_conditioned_checkpoint,
        "condition_manifests": {
            case: {"path": str(path), "sha256": sha256_file(path)}
            for case, path in condition_paths.items()
        },
        "condition_preflight": condition_preflight,
        "candidate_pool": {
            "bitseed": args.bitseed_candidates, "sdpi": args.sdpi_candidates,
            "conformers": args.conformers, "poses_per_side": args.poses_per_side,
            "binary_steps": args.binary_steps,
            "binary_retry_limit": args.binary_retry_limit,
            "binary_consecutive_failure_limit": args.binary_consecutive_failure_limit,
        },
        "arms": {name: value for name, value in {
            "sdpi_64": {"sdpi": 64},
            "bitseed_diversity_64": {"bitseed": 64},
            "geometry_64": {"source_quota": None, "diversity": False},
            "diversity_64": {"source_quota": None, "diversity": True},
            "hybrid_48_16": {"bitseed_quota": 48, "sdpi_quota": 16},
            "hybrid_56_8": {"bitseed_quota": 56, "sdpi_quota": 8},
        }.items() if name in requested_arms},
        "steps": args.steps,
        "proposal_seed": args.seed,
        "diffusion_seed_schedule": "S_plus_2000000_plus_pool_index",
        "max_refinements": args.max_refinements,
        "refine_full_pool": args.refine_full_pool,
        "shared_trajectory_policy": (
            "refine_every_pool_candidate_once" if args.refine_full_pool else
            "refine_each_unique_selected_candidate_once"
        ),
        "native_information_used_for_selection": False,
        "native_information_used_for_reported_metrics_only": True,
    }
    if not manifest["dockq_source_sha256"]:
        raise FileNotFoundError(f"DockQ scoring sources missing under {deepternary_root}")
    previous_reports = validate_resume_output(output, manifest, resume=not args.no_resume)
    atomic_json(manifest, output / "manifest.json")

    runner = None
    failures = []
    completed_cases = []
    run_started = time.monotonic()

    def progress(state, case=None, refinement=None, requested=None):
        atomic_json({
            "state": state, "pid": os.getpid(),
            "updated_unix": time.time(),
            "elapsed_seconds": time.monotonic() - run_started,
            "requested_cases": len(entries),
            "completed_cases": list(completed_cases),
            "failures": list(failures), "current_case": case,
            "refinements_done": refinement, "refinements_requested": requested,
        }, output / "progress.json")

    progress("running")
    for position, entry in enumerate(entries, start=1):
        case = entry["case"]
        destination = output / f"{case}.json"
        if case in previous_reports:
            completed_cases.append(case)
            progress("running")
            print(f"[{position}/{len(entries)}] SKIP {case}", flush=True)
            continue
        print(f"[{position}/{len(entries)}] RUN {case}", flush=True)
        progress("generating", case)
        started = time.monotonic()
        try:
            paths, native, metric_paths = case_inputs[case]
            if runner is None:
                runner = EvaluationRunner(
                    ROOT, checkpoint, device=args.device, deepternary_root=deepternary_root,
                    allow_experimental_checkpoint=args.allow_experimental_checkpoint,
                    allow_conditioned_checkpoint=args.allow_conditioned_checkpoint,
                )
            e3_data, poi_data, ligand_data, input_dim = featurize_complex(
                case, str(paths["protein1"]), str(paths["protein2"]),
                str(paths["ligand"]), runner.score_model_args, args.device,
                cache_dir=str((ROOT / args.cache_dir).resolve()),
            )
            runner._ensure_model(input_dim)
            conditions, condition_audit = None, None
            if condition_paths:
                from rdkit import Chem
                from tricoopdiff.conditioning.condition_features import load_condition_manifest
                conditions, condition_audit = load_condition_manifest(
                    condition_paths[case], case=case,
                    molecule=Chem.Mol(ligand_data['mol']),
                    protein_sizes=(e3_data['coords'].shape[1],
                                   poi_data['coords'].shape[1]),
                )
            proposal_initializer = DualExpertAnchorInitializer(
                runner.expert, e3_data, poi_data, binary_steps=args.binary_steps,
                proposal_attempts=args.poses_per_side,
                conformer_policy=runner.initializer.conformer_policy,
            )
            candidates, generation = generate_hybrid_candidates(
                proposal_initializer, e3_data["coords"][0], poi_data["coords"][0],
                ligand_data, seed=args.seed, conformer_count=args.conformers,
                bitseed_candidate_budget=args.bitseed_candidates,
                sdpi_candidate_budget=args.sdpi_candidates,
                binary_retry_limit=args.binary_retry_limit,
                binary_consecutive_failure_limit=args.binary_consecutive_failure_limit,
            )
            sdpi = [candidate for candidate in candidates if source(candidate) == "sdpi"]
            if len(sdpi) < 64:
                raise RuntimeError(f"only {len(sdpi)} SDPI candidates available")
            selected_48, meta_48 = select_hybrid_walkers(
                candidates, 64, bitseed_quota=48,
            )
            selected_56, meta_56 = select_hybrid_walkers(
                candidates, 64, bitseed_quota=56,
            )
            selected_geometry, meta_geometry = select_hybrid_walkers(
                candidates, 64, use_diversity=False, protect_sources=False,
            )
            selected_diversity, meta_diversity = select_hybrid_walkers(
                candidates, 64, use_diversity=True, protect_sources=False,
            )
            bitseed = [candidate for candidate in candidates
                       if source(candidate) == "bitseed"]
            if len(bitseed) >= 64:
                selected_bitseed, meta_bitseed = select_hybrid_walkers(
                    bitseed, 64, bitseed_quota=64,
                    use_diversity=True, protect_sources=False,
                )
            else:
                selected_bitseed = list(bitseed)
                meta_bitseed = {
                    "walker_count": 64,
                    "status": "insufficient_bitseed_candidates",
                    "available_count": len(bitseed),
                    "selected_candidate_ids": [
                        candidate["candidate_id"] for candidate in bitseed
                    ],
                }
            arms = {
                "sdpi_64": sdpi[:64],
                "bitseed_diversity_64": selected_bitseed,
                "geometry_64": selected_geometry,
                "diversity_64": selected_diversity,
                "hybrid_48_16": selected_48,
                "hybrid_56_8": selected_56,
            }
            arms = {name: arms[name] for name in requested_arms}
            pool_index = {repr(candidate["candidate_id"]): index
                          for index, candidate in enumerate(candidates)}
            selected_ids = {repr(candidate["candidate_id"])
                            for arm in arms.values() for candidate in arm}
            union = (list(candidates) if args.refine_full_pool else [
                candidate for candidate in candidates
                if repr(candidate["candidate_id"]) in selected_ids
            ])
            union_requested_count = len(union)
            if args.smoke_one_per_source:
                if not bitseed or not sdpi:
                    raise RuntimeError("source smoke requires a real candidate from each source")
                union = [bitseed[0], sdpi[0]]
            if args.max_refinements is not None:
                union = union[:args.max_refinements]
            rows = []
            progress("refining", case, 0, len(union))
            for union_index, candidate in enumerate(union, start=1):
                key = repr(candidate["candidate_id"])
                index = pool_index[key]
                diffusion_seed = (args.seed + 2_000_000 + index) % (2**31 - 1)
                print(
                    f"[{position}/{len(entries)}] {case} refinement "
                    f"{union_index}/{len(union)} id={candidate['candidate_id']}",
                    flush=True,
                )
                initial_ligand = candidate["ligand"][0].detach().cpu().numpy()
                initial_poi = candidate["poi"][0].detach().cpu().numpy()
                initial = exact_dockq(
                    initial_ligand, initial_poi, e3_data=e3_data, paths=paths,
                    native=native, deepternary_root=deepternary_root,
                    metric_paths=metric_paths,
                )
                fixed = FixedCandidateInitializer(
                    candidate, no_torsion=runner.initializer.no_torsion,
                )
                seen_sources = []
                hook = None
                if args.smoke_one_per_source:
                    def observe_source(module, positional, keyword):
                        value = keyword.get("source")
                        seen_sources.append(None if value is None else int(value))
                    hook = runner.model.register_forward_pre_hook(observe_source, with_kwargs=True)
                refinement_started = time.monotonic()
                try:
                    sampled = sample_single(
                        runner.model, runner.diffusion, fixed,
                        e3_data, poi_data, ligand_data, steps=args.steps,
                        device=args.device, seed=diffusion_seed,
                        init_mode="fixed_candidate", t_anchor=1.0,
                        conditions=conditions,
                    )
                finally:
                    if hook is not None:
                        hook.remove()
                refinement_seconds = time.monotonic() - refinement_started
                if args.smoke_one_per_source:
                    expected_source = 0 if source(candidate) == "sdpi" else 1
                    if len(seen_sources) != args.steps or any(value != expected_source for value in seen_sources):
                        raise RuntimeError(f"source route mismatch: expected {expected_source}, got {seen_sources}")
                    if not sampled.get("valid", True):
                        raise RuntimeError(f"source smoke refinement failed: {sampled.get('termination_reason')}")
                row = {
                    "candidate_id": candidate["candidate_id"],
                    "pool_index": index,
                    "source": source(candidate),
                    "initial_state_sha256": candidate_hash(candidate),
                    "model_source_trace": seen_sources if args.smoke_one_per_source else None,
                    "refinement_seconds": refinement_seconds,
                    "acceptable": bool(candidate.get("acceptable", False)),
                    "penalty": list(candidate.get("penalty", ())),
                    "geometry": candidate.get("geometry"),
                    "fit_rmsd": candidate.get("fit_rmsd"),
                    "diffusion_seed": diffusion_seed,
                    "valid": bool(sampled.get("valid", True)),
                    "termination_reason": sampled.get("termination_reason"),
                }
                refined = None
                if row["valid"]:
                    refined = exact_dockq(
                        sampled["ligand_coords"], sampled["poi_coords"],
                        e3_data=e3_data, paths=paths, native=native,
                        deepternary_root=deepternary_root,
                        metric_paths=metric_paths,
                    )
                row.update(refinement_metric_fields(initial, refined))
                rows.append(row)
                if union_index % 16 == 0 or union_index == len(union):
                    progress("refining", case, union_index, len(union))
            by_id = {repr(row["candidate_id"]): row for row in rows}
            arm_reports = {}
            for name, selected in arms.items():
                selected_rows = [by_id[repr(candidate["candidate_id"])]
                                 for candidate in selected
                                 if repr(candidate["candidate_id"]) in by_id]
                arm_reports[name] = {
                    "requested_count": len(selected),
                    "evaluated_count": len(selected_rows),
                    "selected_candidate_ids": [candidate["candidate_id"]
                                               for candidate in selected],
                    "initial": summarize(selected_rows, "initial_dockq"),
                    "refined": summarize(selected_rows, "refined_dockq"),
                    "safe_union": summarize(selected_rows, "safe_dockq"),
                }
            if args.refine_full_pool:
                arm_reports["full_pool_256"] = {
                    "requested_count": len(candidates),
                    "evaluated_count": len(rows),
                    "selected_candidate_ids": [candidate["candidate_id"]
                                               for candidate in candidates],
                    "initial": summarize(rows, "initial_dockq"),
                    "refined": summarize(rows, "refined_dockq"),
                    "safe_union": summarize(rows, "safe_dockq"),
                }
            source_reports = {}
            for candidate_source in ("bitseed", "sdpi"):
                source_rows = [row for row in rows
                               if row["source"] == candidate_source]
                source_reports[candidate_source] = {
                    "count": len(source_rows),
                    "initial": summarize(source_rows, "initial_dockq"),
                    "refined": summarize(source_rows, "refined_dockq"),
                    "safe_union": summarize(source_rows, "safe_dockq"),
                    "improved_count": sum(
                        row.get("refinement_delta_dockq", 0.0) > 0.0
                        for row in source_rows
                    ),
                    "damaged_count": sum(
                        row.get("refinement_delta_dockq", 0.0) < 0.0
                        for row in source_rows
                    ),
                }
            report = {
                "status": "complete",
                "case": case,
                "config": manifest,
                "generation": generation,
                "proposal_pool_sha256": hashlib.sha256(
                    "".join(candidate_hash(candidate) for candidate in candidates).encode()
                ).hexdigest(),
                "evaluation_scope": "two_source_smoke" if args.smoke_one_per_source else "refinement_audit",
                "selector_metadata": {
                    "bitseed_diversity_64": meta_bitseed,
                    "geometry_64": meta_geometry,
                    "diversity_64": meta_diversity,
                    "hybrid_48_16": meta_48,
                    "hybrid_56_8": meta_56,
                },
                "condition_audit": condition_audit,
                "union_requested_count": union_requested_count,
                "union_evaluated_count": len(rows),
                "arms": arm_reports,
                "sources": source_reports,
                "refinements": rows,
                "timing_seconds": {"total": time.monotonic() - started},
            }
            atomic_json(report, destination)
            completed_cases.append(case)
            progress("running")
            print(f"[{position}/{len(entries)}] DONE {case}", flush=True)
        except Exception as error:
            failures.append({"case": case, "error": f"{type(error).__name__}: {error}"})
            progress("running")
            print(f"[{position}/{len(entries)}] FAIL {case}: {error}", flush=True)
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
    atomic_json({"config": manifest, "failures": failures}, output / "failures.json")
    case_reports = [json.loads((output / f"{case}.json").read_text())
                    for case in completed_cases]
    arm_names = sorted({name for report in case_reports for name in report["arms"]})
    leaderboards = {}
    for name in arm_names:
        leaderboards[name] = {}
        for stage in ("initial", "refined", "safe_union"):
            values = [{"value": report["arms"][name][stage]["best"]}
                      for report in case_reports if name in report["arms"]]
            leaderboards[name][stage] = summarize(values, "value")
    atomic_json({
        "scope": "case-best oracle; not native-free ranked top1",
        "config": manifest, "requested_cases": len(entries),
        "completed_cases": completed_cases, "failures": failures,
        "leaderboards": leaderboards,
        "valid_refinements": sum(row["valid"] for report in case_reports
                                 for row in report["refinements"]),
        "evaluated_refinements": sum(len(report["refinements"]) for report in case_reports),
    }, output / "summary.json")
    progress("failed" if failures else "complete")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Lazy C=1/M=1 BiTSeed endpoints used only by mixed-source training."""

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch

from .checkpoint_contract import expert_base_identity
from .dual_init import validate_endpoint_record
from .endpoint_kinematics import project_endpoint_coupling
from .expert_anchors import DualExpertAnchorInitializer, generate_hybrid_candidates
from .sdpi_flow import SDPIFlow


ONLINE_TRAINING_ENDPOINT_PROTOCOL = "online_bitseed_training_c1_m1_v1"
ONLINE_TRAINING_BASE_SEED = 41_000


def _cpu_array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _case_seed(case, base_seed):
    digest = hashlib.sha256(case.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


def _atomic_save(payload, destination):
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


class LazyTrainingEndpointStore:
    """Generate one two-expert endpoint on first case access, then replay it."""

    lazy = True

    def __init__(self, cache_dir, expert, *, binary_steps=20,
                 base_seed=ONLINE_TRAINING_BASE_SEED):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expert = expert
        self.binary_steps = int(binary_steps)
        self.base_seed = int(base_seed)
        self.protocol = {
            "version": ONLINE_TRAINING_ENDPOINT_PROTOCOL,
            "conformers": 1,
            "poses_per_side": 1,
            "binary_steps": self.binary_steps,
            "binary_retry_limit": 1,
            "selection": "single_pair_no_ranking",
            "fallback": "sdpi_v1_on_any_invalid_expert_side",
            "base_seed": self.base_seed,
            "frozen_expert": expert_base_identity(expert),
        }
        self._records = {}

    def _path(self, case):
        safe = "".join(char if char.isalnum() or char in "-_." else "_"
                       for char in case)
        suffix = hashlib.sha256(case.encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"{safe}.{suffix}.pt"

    def _load(self, case):
        path = self._path(case)
        if not path.is_file():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol") != self.protocol or payload.get("case") != case:
            raise ValueError(f"{case}: lazy endpoint cache contract mismatch")
        record = payload.get("record")
        validate_endpoint_record(record)
        return record

    def _generate(self, case, batch):
        e3, poi, ligand = batch["e3_data"], batch["poi_data"], batch["ligand_data"]
        initializer = DualExpertAnchorInitializer(
            self.expert, e3, poi, binary_steps=self.binary_steps,
            proposal_attempts=1, conformer_policy="etkdgv3_random",
        )
        seed = _case_seed(case, self.base_seed)
        device = ligand["coords"].device
        devices = ([device.index if device.index is not None
                    else torch.cuda.current_device()] if device.type == "cuda" else [])
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=devices):
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                if devices:
                    torch.cuda.manual_seed_all(seed)
                candidates, generation = generate_hybrid_candidates(
                    initializer, e3["coords"][0], poi["coords"][0], ligand,
                    seed=seed, conformer_count=1, bitseed_candidate_budget=1,
                    sdpi_candidate_budget=0, binary_retry_limit=1,
                    binary_consecutive_failure_limit=1,
                )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        if len(candidates) != 1:
            raise AssertionError(f"{case}: C=1/M=1 must produce exactly one endpoint")
        candidate = candidates[0]
        actual_source = candidate["candidate_id"][0]
        coupling, projected, projection = project_endpoint_coupling(
            ligand, poi["coords"], candidate["ligand"], candidate["poi"],
            no_torsion=initializer.no_torsion,
        )
        ligand_one, poi_one, body_one, targets = SDPIFlow().path(
            ligand, poi["coords"], coupling,
            torch.ones(1, device=device),
        )
        if not all(torch.isfinite(value).all() for branch in targets.values()
                   for value in (branch.values() if isinstance(branch, dict) else (branch,))):
            raise FloatingPointError(f"{case}: non-finite projected velocity target")
        conformer_index = candidate.get("conformer_index")
        record = {
            "case": case,
            "actual_source": actual_source,
            "candidate_id": candidate["candidate_id"],
            "conformer_seed": (initializer._conformer_seed(seed, int(conformer_index))
                               if conformer_index is not None else None),
            "binary_pose_ids": list(candidate["attempt"]),
            "fallback": candidate.get("fallback"),
            "native_used_in_candidate_generation": False,
            "native_used_in_candidate_selection": False,
            "native_used_for_training_projection": True,
            "endpoint": {
                "ligand": _cpu_array(candidate["ligand"][0]),
                "poi": _cpu_array(candidate["poi"][0]),
                "body_frame": _cpu_array(candidate["rotation"][0]),
            },
            "training_endpoint": {
                "ligand": _cpu_array(ligand_one[0]),
                "poi": _cpu_array(poi_one[0]),
                "body_frame": _cpu_array(body_one[0]),
            },
            "coupling": {name: _cpu_array(value) for name, value in coupling.items()},
            "projection": projection,
            "generation": {
                "case_seed": seed,
                "fallback_events": generation["fallback_events"],
                "geometry": candidate["geometry"],
                "fit_rmsd": candidate["fit_rmsd"],
            },
        }
        if (not np.allclose(_cpu_array(ligand_one[0]), projected["ligand"], atol=1e-5)
                or not np.allclose(_cpu_array(poi_one[0]), projected["poi"], atol=1e-5)):
            raise AssertionError(f"{case}: projected endpoint failed t=1 round-trip")
        validate_endpoint_record(record)
        _atomic_save({"case": case, "protocol": self.protocol, "record": record},
                     self._path(case))
        return record

    def coupling(self, case, occurrence=0, *, batch=None):
        del occurrence  # One immutable endpoint per case and experiment seed.
        record = self._records.get(case)
        if record is None:
            record = self._load(case)
        if record is None:
            if batch is None:
                raise ValueError(f"{case}: batch is required for first endpoint generation")
            record = self._generate(case, batch)
        self._records[case] = record
        coupling = {name: np.asarray(value).copy()
                    for name, value in record["coupling"].items()}
        return coupling, record

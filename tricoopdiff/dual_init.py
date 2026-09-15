"""Runtime and cache contracts for shared SDPI/BiTSeed training."""

from pathlib import Path

import numpy as np
import torch

from .endpoint_kinematics import ENDPOINT_PROJECTION_VERSION


DUAL_INIT_CACHE_VERSION = "bitseed_training_endpoints_v1"
TRAINING_ARMS = ("A1", "B", "C")


def _finite_array(value, name, shape_tail=None):
    array = np.asarray(value)
    if shape_tail is not None and array.shape[-len(shape_tail):] != shape_tail:
        raise ValueError(f"{name} has invalid shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_endpoint_record(record):
    """Validate one native-free-selected, training-target-projected endpoint."""
    required = {
        "case", "actual_source", "candidate_id", "conformer_seed",
        "binary_pose_ids", "endpoint", "training_endpoint", "coupling", "projection",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"endpoint record missing {sorted(missing)}")
    if record["actual_source"] not in ("sdpi", "bitseed"):
        raise ValueError("endpoint actual_source must be sdpi or bitseed")
    if record["actual_source"] == "bitseed" and record.get("fallback"):
        raise ValueError("an SDPI fallback cannot be labelled BiTSeed")
    if record["projection"].get("contract") != ENDPOINT_PROJECTION_VERSION:
        raise ValueError("endpoint has an unknown projection contract")
    endpoint = record["endpoint"]
    _finite_array(endpoint["ligand"], "endpoint ligand", (3,))
    _finite_array(endpoint["poi"], "endpoint POI", (3,))
    _finite_array(endpoint["body_frame"], "endpoint body frame", (3, 3))
    training_endpoint = record["training_endpoint"]
    _finite_array(training_endpoint["ligand"], "training endpoint ligand", (3,))
    _finite_array(training_endpoint["poi"], "training endpoint POI", (3,))
    _finite_array(training_endpoint["body_frame"], "training endpoint body frame", (3, 3))
    coupling = record["coupling"]
    for name in ("angles", "omegas", "centers", "bonds", "masks"):
        if name not in coupling:
            raise ValueError(f"endpoint coupling missing {name}")
    _finite_array(coupling["angles"], "coupling angles")
    _finite_array(coupling["omegas"], "coupling omegas", (2, 3))
    _finite_array(coupling["centers"], "coupling centers", (2, 3))
    if record.get("native_used_in_candidate_generation") is not False:
        raise ValueError("cache must attest native-free candidate generation")
    if record.get("native_used_in_candidate_selection") is not False:
        raise ValueError("cache must attest native-free candidate selection")


def validate_endpoint_cache(payload):
    if payload.get("cache_version") != DUAL_INIT_CACHE_VERSION:
        raise ValueError("unknown BiTSeed training endpoint cache version")
    protocol = payload.get("generation_protocol", {})
    if not protocol.get("frozen_expert_checkpoint_sha256"):
        raise ValueError("cache lacks frozen expert checkpoint hash")
    if not payload.get("entries"):
        raise ValueError("endpoint cache is empty")
    for record in payload["entries"]:
        validate_endpoint_record(record)


class CachedEndpointStore:
    """Load deterministic per-case BiTSeed couplings for arms B and C."""

    def __init__(self, path):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        validate_endpoint_cache(payload)
        self.payload = payload
        self.by_case = {}
        for record in payload["entries"]:
            self.by_case.setdefault(record["case"], []).append(record)
        if not any(record["actual_source"] == "bitseed"
                   for records in self.by_case.values() for record in records):
            raise ValueError("cache contains no actual BiTSeed endpoints")

    def coupling(self, case, occurrence=0):
        records = self.by_case.get(case, ())
        if not records:
            raise KeyError(f"case {case!r} has no cached BiTSeed endpoint")
        record = records[int(occurrence) % len(records)]
        coupling = {
            name: np.asarray(value).copy()
            for name, value in record["coupling"].items()
        }
        return coupling, record


def mixed_source_for_step(global_step):
    """Fixed 1:1 batch sequence shared exactly by arms B and C."""
    return int(global_step) % 2

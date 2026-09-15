#!/usr/bin/env python
"""Validate and merge deterministic BiTSeed endpoint-cache partitions."""

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tricoopdiff.dual_init import validate_endpoint_cache  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", nargs="+", required=True)
    result.add_argument("--out", required=True)
    return result


def atomic_torch_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def main():
    args = parser().parse_args()
    caches = [torch.load(path, map_location="cpu", weights_only=False)
              for path in args.input]
    if not caches:
        raise ValueError("no input caches")
    for cache in caches:
        validate_endpoint_cache(cache)
    reference = caches[0]
    for cache in caches[1:]:
        for key in ("cache_version", "indexes", "generation_protocol"):
            if cache.get(key) != reference.get(key):
                raise ValueError(f"partition mismatch for {key}")

    partitions = [cache.get("case_partition") for cache in caches]
    count = len(caches)
    if any(not isinstance(partition, dict) for partition in partitions):
        raise ValueError("every input must declare case_partition")
    if {partition["count"] for partition in partitions} != {count}:
        raise ValueError("partition count does not match number of inputs")
    if {partition["index"] for partition in partitions} != set(range(count)):
        raise ValueError("partition indexes are incomplete or duplicated")
    totals = {partition["total_cases"] for partition in partitions}
    if len(totals) != 1:
        raise ValueError("partition total_cases values disagree")

    by_case = {}
    for cache in caches:
        for record in cache["entries"]:
            case = record["case"]
            if case in by_case:
                raise ValueError(f"duplicate endpoint case across partitions: {case}")
            by_case[case] = record
    total_cases = totals.pop()
    if len(by_case) != total_cases:
        raise ValueError(
            f"partition union has {len(by_case)} cases; expected {total_cases}"
        )

    # Global index order is preserved by interleaving modulo partitions.
    partition_entries = {
        cache["case_partition"]["index"]: iter(cache["entries"])
        for cache in caches
    }
    entries = [next(partition_entries[index % count]) for index in range(total_cases)]
    for iterator in partition_entries.values():
        try:
            next(iterator)
        except StopIteration:
            continue
        raise ValueError("partition contains more entries than declared")
    if {record["case"] for record in entries} != set(by_case):
        raise ValueError("interleaved partition order lost or duplicated cases")

    payload = {
        "cache_version": reference["cache_version"],
        "indexes": reference["indexes"],
        "generation_protocol": reference["generation_protocol"],
        "entries": entries,
    }
    validate_endpoint_cache(payload)
    atomic_torch_save(payload, args.out)
    print(f"[merge] wrote {len(entries)} endpoints to {args.out}", flush=True)


if __name__ == "__main__":
    main()

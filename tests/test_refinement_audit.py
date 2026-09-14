"""CPU tests for full-pool refinement aggregation and geometry gates."""

from tricoopdiff.refinement_audit import (
    aggregate_reports,
    geometry_from_coordinates,
    geometry_gate_violations,
    hard_safety_violations,
    validate_reports,
)


def row(candidate_id, source, initial, final, *, geometry, acceptable):
    return {
        "candidate_id": candidate_id,
        "source": source,
        "valid": True,
        "geometry": geometry,
        "acceptable": acceptable,
        "initial_dockq": initial,
        "refined_dockq": final,
        "safe_dockq": max(initial, final),
    }


GOOD = {
    "protein_ca_pairs_below_3A": 0,
    "minimum_protein_ca_distance": 5.0,
    "minimum_e3_ca_ligand_distance": 3.5,
    "minimum_poi_ca_ligand_distance": 3.5,
    "poi_outward_separation": 0.0,
}
CLASH = {**GOOD, "protein_ca_pairs_below_3A": 2,
         "minimum_protein_ca_distance": 2.5}


def report(case, shift=0.0):
    rows = [
        row(["bitseed", 0], "bitseed", 0.30 + shift, 0.20 + shift,
            geometry=GOOD, acceptable=True),
        row(["bitseed", 1], "bitseed", 0.10 + shift, 0.40 + shift,
            geometry=CLASH, acceptable=False),
        row(["sdpi", 0], "sdpi", 0.20 + shift, 0.35 + shift,
            geometry=GOOD, acceptable=True),
        row(["sdpi", 1], "sdpi", 0.05 + shift, 0.10 + shift,
            geometry=CLASH, acceptable=False),
    ]
    arms = {
        "sdpi_64": [["sdpi", 0], ["sdpi", 1]],
        "hybrid_48_16": [["bitseed", 0], ["sdpi", 0]],
        "full_pool_256": [item["candidate_id"] for item in rows],
    }
    by_id = {repr(item["candidate_id"]): item for item in rows}
    arm_reports = {}
    for name, identifiers in arms.items():
        selected = [by_id[repr(identifier)] for identifier in identifiers]
        arm_reports[name] = {
            "selected_candidate_ids": identifiers,
            "initial": {"best": max(item["initial_dockq"] for item in selected)},
            "refined": {"best": max(item["refined_dockq"] for item in selected)},
            "safe_union": {"best": max(item["safe_dockq"] for item in selected)},
        }
    return {
        "status": "complete",
        "case": case,
        "config": {"protocol": "fixture_v1"},
        "union_evaluated_count": len(rows),
        "refinements": rows,
        "arms": arm_reports,
        "timing_seconds": {"total": 10.0 + shift},
    }


def test_geometry_gate_boundaries_and_causes():
    assert geometry_gate_violations(GOOD) == []
    assert geometry_gate_violations(CLASH) == [
        "protein_protein_clash_below_3A",
    ]
    assert hard_safety_violations(CLASH) == [
        "protein_protein_clash_below_3A",
    ]
    assert geometry_gate_violations({
        **GOOD,
        "minimum_e3_ca_ligand_distance": 8.1,
        "minimum_poi_ca_ligand_distance": 1.4,
        "minimum_protein_ca_distance": 15.1,
    }) == [
        "e3_ligand_above_8A",
        "poi_ligand_below_1.5A",
        "protein_separation_above_15A",
    ]
    assert hard_safety_violations({
        **GOOD,
        "minimum_e3_ca_ligand_distance": 8.1,
        "minimum_poi_ca_ligand_distance": 1.4,
        "minimum_protein_ca_distance": 15.1,
    }) == ["poi_ligand_below_1.5A"]
    coordinates = geometry_from_coordinates(
        [[0.0, 0.0, 0.0]], [[5.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]],
    )
    assert coordinates["protein_ca_pairs_below_3A"] == 0
    assert coordinates["minimum_e3_ca_ligand_distance"] == 2.0
    assert coordinates["minimum_poi_ca_ligand_distance"] == 3.0
    assert geometry_gate_violations(coordinates) == []


def test_full_pool_aggregation_reproduces_selection_and_gate_audits():
    reports = {"a": report("a"), "b": report("b", 0.01)}
    integrity = validate_reports(
        reports, expected_cases=2, expected_pool=4,
        expected_bitseed=2, expected_sdpi=2,
    )
    result = aggregate_reports(reports, integrity=integrity)
    assert result["integrity"]["candidates"] == 8
    assert result["leaderboards"]["sdpi_64"]["final"]["mean"] == 0.355
    assert result["leaderboards"]["hybrid_48_16"]["initial"]["mean"] == 0.305
    assert result["sources"]["bitseed"]["case_best"]["final_minus_initial"]["wins"] == 2
    retention = result["selector_retention"]["hybrid_48_16"]["final"]
    assert retention["exact_oracle_candidate_retained"] == 0
    assert result["acceptability"]["overall"]["acceptable"] == 4
    assert result["acceptability"]["violation_counts_nonexclusive"] == {
        "protein_protein_clash_below_3A": 4,
    }
    assert result["acceptability"]["arms"]["hybrid_48_16"][
        "cases_with_acceptable_candidate"
    ] == 2
    assert result["acceptability"]["arms"]["hybrid_48_16"][
        "selected_source_counts"
    ] == {"bitseed": 2, "sdpi": 2}
    assert result["acceptability"]["hard_safety_only_diagnostic"]["overall"][
        "retained"
    ] == 4


if __name__ == "__main__":
    test_geometry_gate_boundaries_and_causes()
    test_full_pool_aggregation_reproduces_selection_and_gate_audits()
    print("PASS refinement audit")

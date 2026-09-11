import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_ranker_indexes.py"
SPEC = importlib.util.spec_from_file_location("prepare_ranker_indexes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_relocate_entry_preserves_membership_and_moves_legacy_paths(tmp_path):
    artifacts = tmp_path / "artifacts"
    relative = Path("data/case/protein1.pdb")
    paths = {}
    for key, name in (("protein1", "protein1.pdb"),
                      ("protein2", "protein2.pdb"), ("ligand", "ligand.sdf")):
        path = artifacts / "data/case" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key)
        paths[key] = str(MODULE.LEGACY_ROOT / "data/case" / name)
    result = MODULE.relocate_entry({"case": "case", **paths}, artifacts)
    assert result["case"] == "case"
    assert Path(result["protein1"]) == (artifacts / relative).resolve()


def test_prepare_supports_predeclared_disjoint_offset(tmp_path):
    artifacts = tmp_path / "artifacts"
    entries = []
    for index in range(5):
        paths = {}
        for key, suffix in (("protein1", "p1.pdb"),
                            ("protein2", "p2.pdb"), ("ligand", "lig.sdf")):
            path = artifacts / f"case_{index}" / suffix
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(key)
            paths[key] = str(path)
        entries.append({"case": f"case_{index}", **paths})
    source = tmp_path / "source.json"
    source.write_text(__import__("json").dumps({"entries": entries}))

    result = MODULE.prepare(source, artifacts, 2, "test", offset=2)
    assert [entry["case"] for entry in result["entries"]] == ["case_2", "case_3"]
    assert result["metadata"]["selection"].startswith("entries_3_through_4")


def test_prepare_pdb_disjoint_excludes_and_deduplicates(tmp_path):
    artifacts = tmp_path / "artifacts"
    entries = []
    for case in ("1AAA_A_B_X", "2BBB_A_B_X", "2BBB_C_D_Y", "3CCC_A_B_X"):
        paths = {}
        for key, suffix in (("protein1", "p1.pdb"),
                            ("protein2", "p2.pdb"), ("ligand", "lig.sdf")):
            path = artifacts / case / suffix
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(key)
            paths[key] = str(path)
        entries.append({"case": case, **paths})
    source = tmp_path / "source_disjoint.json"
    source.write_text(__import__("json").dumps({"entries": entries}))

    result = MODULE.prepare_pdb_disjoint(
        source, artifacts, 2, "val", {"1AAA"},
    )
    assert [entry["case"] for entry in result["entries"]] == [
        "2BBB_A_B_X", "3CCC_A_B_X",
    ]

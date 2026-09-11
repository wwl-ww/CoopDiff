import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_local_mgd_panel.py"
SPEC = importlib.util.spec_from_file_location("prepare_local_mgd_panel", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_panel_uses_predeclared_evenly_spaced_positions():
    cases = [f"case_{index:02d}" for index in range(94)]
    assert MODULE.select_panel(cases) == [
        "case_00", "case_23", "case_46", "case_69", "case_93",
    ]


def test_select_panel_rejects_changed_official_cohort():
    try:
        MODULE.select_panel(["case"] * 94)
    except ValueError as error:
        assert "94 unique" in str(error)
    else:
        raise AssertionError("duplicate official MGD cases must be rejected")


def test_full_entries_record_explicit_pdb_fallback(tmp_path):
    ternary = tmp_path / "ternary"
    restored = tmp_path / "restored"
    case = "case"
    (ternary / case).mkdir(parents=True)
    for name in ("protein1.pdb", "protein2.pdb", "ligand.pdb", "gt_complex.pdb"):
        (ternary / case / name).write_text("placeholder")
    entries, fallbacks = MODULE.build_entries(
        [case], ternary, restored, allow_pdb_fallback=True,
    )
    assert fallbacks == [case]
    assert entries[0]["ligand"].endswith("ligand.pdb")
    assert entries[0]["ligand_chemistry"] == "pdb_inferred_fallback"

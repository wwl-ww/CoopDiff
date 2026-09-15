import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_mgd_oracle_matrix.py"
SPEC = importlib.util.spec_from_file_location("summarize_mgd_oracle_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_oracle_matrix_is_paired(tmp_path):
    offsets = {"none": 0.0, "translations": 0.1, "rotations": 0.02, "all": 0.15}
    for mode, offset in offsets.items():
        path = tmp_path / f"mode_{mode}/seed_0/report.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "status": "complete", "oracle_rigid": mode, "seed": 0,
            "baseline": [
                {"case": "a", "dockq": 0.01 + offset},
                {"case": "b", "dockq": 0.02 + offset},
            ],
        }))
    result = MODULE.summarize(tmp_path, seeds=[0])
    assert result["paired_rows"] == 2
    assert result["interpretation"]["translations_dominant"]
    assert abs(result["modes"]["all"]["mean_delta_vs_none"] - 0.15) < 1e-12

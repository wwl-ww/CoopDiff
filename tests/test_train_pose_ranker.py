import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_pose_ranker.py"
SPEC = importlib.util.spec_from_file_location("train_pose_ranker", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_cases_requires_rank_feature_provenance(tmp_path):
    payload = {
        "case": "1ABC_A_B_LIG",
        "run_config": {"rank_features_saved": True},
        "walkers": [
            {"dockq": 0.1 + index / 100, "rank_features": [index, index + 1]}
            for index in range(8)
        ],
    }
    (tmp_path / "case.json").write_text(json.dumps(payload))
    cases = MODULE.load_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0]["features"].shape == (8, 2)

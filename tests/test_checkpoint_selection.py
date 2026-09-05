"""CPU-only tests for formal best-checkpoint log selection."""

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_best_checkpoint.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("select_best_checkpoint", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_parse_training_log_finds_minimum(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "Epoch 1/3 | train_loss=5 | val_loss=4.0000 | skipped=0\n"
        "  -> Saved best (val_loss=4.0000)\n"
        "Epoch 2/3 | train_loss=3 | val_loss=2.0000 | skipped=0\n"
        "  -> Saved best (val_loss=2.0000)\n"
        "Epoch 3/3 | train_loss=3 | val_loss=2.5000 | skipped=0\n"
    )
    epochs, saves = module.parse_training_log(log)
    assert min(epochs, key=lambda item: item["val_loss"])["epoch"] == 2
    assert saves[-1] == 2.0

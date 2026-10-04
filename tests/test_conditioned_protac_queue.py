from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from run_conditioned_protac_queue import evaluation_command, supported_checkpoint_name


def test_conditioned_queue_command_is_explicit_and_fixed_arm(tmp_path):
    args = SimpleNamespace(
        runtime_root=tmp_path / 'runtime', project_root=tmp_path,
        index=tmp_path / 'index.json', cache_dir=tmp_path / 'cache',
        seed=31000, steps=20, arms='hybrid_56_8')
    command = evaluation_command(
        args, ('pocket', tmp_path / 'best_model.pt', tmp_path / 'manifests'),
        tmp_path / 'out')
    assert '--allow-conditioned-checkpoint' in command
    assert command[command.index('--arms') + 1] == 'hybrid_56_8'
    assert command[command.index('--device') + 1] == 'cuda:0'
    assert command[command.index('--steps') + 1] == '20'


def test_conditioned_queue_accepts_only_explicit_checkpoint_names():
    assert supported_checkpoint_name('best_model.pt')
    assert supported_checkpoint_name('final_model.pt')
    assert supported_checkpoint_name('checkpoint_epoch5.pt')
    assert supported_checkpoint_name('checkpoint_epoch40.pt')
    assert not supported_checkpoint_name('checkpoint.pt')
    assert not supported_checkpoint_name('checkpoint_epoch0.pt')
    assert not supported_checkpoint_name('checkpoint_epoch5.pt.bak')


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory(prefix='conditioned-queue-test-') as directory:
        test_conditioned_queue_command_is_explicit_and_fixed_arm(Path(directory))

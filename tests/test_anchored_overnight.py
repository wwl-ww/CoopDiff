"""Night queues construct serial argument lists, never shell command strings."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_anchored_overnight import build_training_command


def test_wall_clock_budget_and_boolean_flags_are_explicit():
    config = {'scope': 'training', 'seed': 7315, 'coupled_linker_action': True,
              'reset_action_adapter': True, 'resume': None, 'body_aware_action': False,
              'conditioning_task': 'dual_contact_core'}
    command = build_training_command(Path('/runtime'), config, Path('/output'), 33000)
    assert command[-2:] == ['--max-seconds', '33000']
    assert command.count('--coupled-linker-action') == 1
    assert '--resume' not in command and '--body-aware-action' not in command
    assert command[command.index('--conditioning-task') + 1] == 'dual_contact_core'
    assert 'out' not in config and 'max_seconds' not in config


def test_unlimited_command_has_no_time_or_update_ceiling():
    command = build_training_command(Path('/runtime'), {'max_updates': 0}, Path('/output'), None)
    assert '--max-seconds' not in command
    assert command[command.index('--max-updates') + 1] == '0'

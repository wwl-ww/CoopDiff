from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.anchored_data_schedule import CLUSTER_ORDER, epoch_entries


def test_cluster_schedule_visits_each_cluster_once_and_is_resume_stable():
    entries = [{'case': str(i)} for i in range(101)]
    clusters = [{'items': ['0'], 'representative': ['0']},
                {'items': [str(i) for i in range(1, 101)], 'representative': ['1']}]
    before = torch.get_rng_state().clone()
    draw = lambda epoch: epoch_entries(entries, clusters, seed=7315, epoch=epoch, policy=CLUSTER_ORDER)
    assert draw(0) == draw(0)
    assert torch.equal(torch.get_rng_state(), before)
    selections = [draw(i) for i in range(100)]
    assert all(len(row) == 2 and sum(x['case'] == '0' for x in row) == 1 for row in selections)
    assert len({x['case'] for row in selections for x in row}) > 20
    assert epoch_entries(entries, clusters, seed=7315, epoch=0) == entries

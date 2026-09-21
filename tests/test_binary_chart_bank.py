"""Discrete reference uncertainty must not silently become duplicate walkers."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.binary_chart_bank import reference_conformer_schedule


def test_schedule_covers_references_and_never_repeats_pairs():
    draw = lambda: reference_conformer_schedule(3, 4, 11, generator=torch.Generator().manual_seed(7315))
    actual = draw()
    assert actual == draw() and len(set(actual)) == 11
    assert {a for a, _ in actual[:3]} == {0, 1, 2}
    assert max(sum(a == ref for a, _ in actual) for ref in range(3)) == 4
    try:
        reference_conformer_schedule(1, 4, 5, generator=torch.Generator().manual_seed(7315))
    except ValueError:
        pass
    else:
        raise AssertionError('discrete pool was silently repeated to meet the walker budget')

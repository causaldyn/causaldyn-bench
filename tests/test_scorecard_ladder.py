"""The ladder of lift tests: its top rung is the budgets run's experiments bit for bit, and its
rungs nest, each test's readout the same draw at every rung that holds it."""

import numpy as np
import pytest

from causaldyn_bench.budget_regret import digest, experiments, lift_rows
from causaldyn_bench.lift_calibration import STARTS
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.ladder import RUNGS, rung, starts


@pytest.fixture(scope="module")
def world():
    return drawn(905).simulate(905)


def _arrays(test):
    return [test.treated.spend, test.treated.outcome, test.control.spend, test.control.outcome]


def test_the_top_rung_is_the_budgets_runs_experiments_bit_for_bit(world):
    top, committed = rung(world, 905, len(STARTS)), experiments(world, 905)
    assert list(top) == list(committed) == list(world.channels)
    for name in world.channels:
        for ours, theirs in zip(top[name], committed[name], strict=True):
            for a, b in zip(_arrays(ours), _arrays(theirs), strict=True):
                np.testing.assert_array_equal(a, b)
    assert digest(world, lift_rows(top)) == digest(world, lift_rows(committed))


def test_each_rung_holds_the_latest_tests_and_reads_them_as_the_top_rung_does(world):
    top = rung(world, 905, len(STARTS))
    for k in RUNGS[1:]:
        assert starts(k) == STARTS[len(STARTS) - k :]
        held = rung(world, 905, k)
        for name in world.channels:
            assert len(held[name]) == k
            for ours, theirs in zip(held[name], top[name][len(STARTS) - k :], strict=True):
                np.testing.assert_array_equal(ours.difference, theirs.difference)
                np.testing.assert_array_equal(ours.control.spend, theirs.control.spend)


def test_rung_nought_has_no_tests_and_no_lift_rows(world):
    assert rung(world, 905, 0) == {}
    rows = lift_rows(rung(world, 905, 0))
    assert (rows.channel, rows.x.size, rows.dropped) == ((), 0, 0)


@pytest.mark.parametrize("k", [-1, len(STARTS) + 1])
def test_a_rung_off_the_ladder_is_refused(world, k):
    with pytest.raises(ValueError, match="a rung holds"):
        rung(world, 905, k)

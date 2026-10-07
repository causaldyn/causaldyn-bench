"""The ladder of lift tests: its top rung is the budgets run's experiments bit for bit, and its
rungs nest, each test's readout the same draw at every rung that holds it. An overlapping rung runs
one channel's tests in the market whose history an arm reads: at a share of 1 the history holds the
gap's noise over each readout, at a share of 1/2 none of it."""

import numpy as np
import pytest

from causaldyn_bench.budget_regret import digest, experiments, lift_rows
from causaldyn_bench.endogenous_mmm import _media
from causaldyn_bench.lift_calibration import STARTS, TEST
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.ladder import RUNGS, SHARES, overlap, rung, starts


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


def _readouts(tests):
    """Each test's readout weeks, as a slice of the history."""
    return [slice(t.control.history, t.control.history + t.control.outcome.size) for t in tests]


@pytest.mark.parametrize("share", SHARES)
@pytest.mark.parametrize("k", RUNGS[1:])
def test_an_overlapping_rung_s_tests_are_the_rung_s_own_and_its_history_holds_them(world, k, share):
    held = overlap(world, 905, k, "meta", share)
    assert list(held.tests) == ["meta"] and held.share == share
    for ours, theirs in zip(held.tests["meta"], rung(world, 905, k)["meta"], strict=True):
        for a, b in zip(_arrays(ours), _arrays(theirs), strict=True):
            np.testing.assert_array_equal(a, b)
    c = world.channels.index("meta")
    np.testing.assert_array_equal(held.planned, world.spend[:, c])
    others = [j for j in range(len(world.channels)) if j != c]
    np.testing.assert_array_equal(held.history.spend[:, others], world.spend[:, others])
    np.testing.assert_array_equal(held.history.media[:, others], world.media[:, others])
    dark = np.zeros(world.week.size, dtype=bool)
    for first in starts(k):
        dark[first - 1 : first - 1 + TEST] = True
    np.testing.assert_array_equal(held.history.spend[~dark, c], world.spend[~dark, c])
    np.testing.assert_allclose(
        held.history.spend[dark, c], (1.0 - share) * world.spend[dark, c], rtol=1e-15, atol=0.0
    )
    gap = (
        _media(
            np.where(dark, 0.0, world.spend[:, c]),
            world.retention[c],
            world.saturation[c],
            world.effect[c],
            world.kernel_length,
        )
        - world.media[:, c]
    )
    np.testing.assert_allclose(
        held.history.media[:, c] - world.media[:, c], share * gap, rtol=0.0, atol=1e-9
    )


def test_at_a_share_of_one_the_history_holds_twice_each_gap_s_noise_over_its_readout(world):
    held = overlap(world, 905, len(STARTS), "pla", 1.0)
    noise = held.history.sales - world.sales - (held.history.media[:, 0] - world.media[:, 0])
    assert noise.std() > 0.005 * world.sales.mean()
    for test, readout in zip(held.tests["pla"], _readouts(held.tests["pla"]), strict=True):
        gap = (held.history.media[:, 0] - world.media[:, 0])[readout]
        np.testing.assert_allclose(test.difference - gap, 2.0 * noise[readout], rtol=0.0, atol=1e-9)


def test_at_a_share_of_one_half_the_history_holds_the_response_and_none_of_the_noise(world):
    held = overlap(world, 905, len(STARTS), "pla", 0.5)
    response = held.history.media[:, 0] - world.media[:, 0]
    assert np.abs(response).max() > 1.0
    np.testing.assert_allclose(held.history.sales, world.sales + response, rtol=0.0, atol=1e-9)
    for test, readout in zip(held.tests["pla"], _readouts(held.tests["pla"]), strict=True):
        assert np.abs(test.difference - 2.0 * response[readout]).max() > 1.0


def test_the_noise_share_reaches_the_tests_and_the_history(world):
    low = overlap(world, 905, 2, "tv", 1.0)
    high = overlap(world, 905, 2, "tv", 1.0, noise_share=0.10)
    c = world.channels.index("tv")

    def noise(held):
        return held.history.sales - world.sales - (held.history.media[:, c] - world.media[:, c])

    np.testing.assert_allclose(noise(high), 10.0 * noise(low), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    ("k", "channel", "share", "match"),
    [
        (0, "pla", 1.0, "runs no test"),
        (len(STARTS) + 1, "pla", 1.0, "a rung holds"),
        (2, "radio", 1.0, "no channel 'radio'"),
        (2, "pla", 0.25, "a share of"),
        (2, "pla", 0.0, "a share of"),
    ],
)
def test_an_overlapping_rung_refuses_what_it_cannot_run(world, k, channel, share, match):
    with pytest.raises(ValueError, match=match):
        overlap(world, 905, k, channel, share)

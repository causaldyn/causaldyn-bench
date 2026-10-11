"""Family 7: an arm reads the last L weeks of a drawn world and nothing else of its history, with
its tests renumbered from the first of them; the deepest rung is the count of tests whose readout
falls inside those weeks, and a deeper one is refused; the world, its quarter and its best plan are
the drawn world's."""

import dataclasses

import numpy as np
import pytest

from causaldyn_bench.endogenous_mmm import YEAR, MediaMixWorld
from causaldyn_bench.lift_calibration import COOLDOWN, PRE, STARTS, TEST
from causaldyn_bench.scorecard.short import DEEPEST, LENGTHS, SHORT, deepest
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

WEEKS = 156


def test_the_deepest_rung_counts_the_tests_whose_readout_the_window_holds():
    """Read off the world's own geo test: a test counts where its readout's first week is one of
    the last L."""
    world = SHORT.world("L-52", SHORT.pilots["L-52"][0]).history
    experiment = world.geo_test("pla", STARTS, test=TEST)
    opens = [experiment.readout(start, pre=PRE, cooldown=COOLDOWN).start for start in STARTS]
    assert opens == [start - 1 - PRE for start in STARTS]
    for name, length in LENGTHS.items():
        assert name == f"L-{length}"
        assert DEEPEST[name] == sum(first >= WEEKS - length for first in opens)
    for length in range(1, WEEKS + 1):  # every length, the readouts' edges among them
        assert deepest(length) == sum(first >= WEEKS - length for first in opens)
    assert dict(DEEPEST) == {"L-52": 1, "L-78": 2, "L-104": 2}


@pytest.mark.parametrize("name", LENGTHS)
def test_an_arm_reads_the_last_l_weeks_and_its_tests_renumbered(name):
    length, top = LENGTHS[name], DEEPEST[name]
    seed = SHORT.pilots[name][0]
    world = SHORT.world(name, seed)
    full = TRACK_M2.observe(TRACK_M2.world("drawn", seed), top)
    short = SHORT.observe(world, top)
    cut = WEEKS - length
    assert short.sales.shape == (length,) and short.spend.shape == (length, 3)
    np.testing.assert_array_equal(short.sales, full.sales[cut:])
    np.testing.assert_array_equal(short.spend, full.spend[cut:])
    for control, series in full.controls.items():
        np.testing.assert_array_equal(short.controls[control], series[cut:])
    assert short.roi_window == (length - YEAR + 1, length)
    assert short.future_controls is not None and full.future_controls is not None
    for control, series in full.future_controls.items():
        np.testing.assert_array_equal(short.future_controls[control], series)
    for field in ("budget", "lower", "upper", "status_quo", "kernel_length", "planned"):
        assert np.array_equal(getattr(short, field), getattr(full, field)), field
    # the same tests, each dated in the weeks the arm reads, its readout inside them
    for field in ("channel", "x", "delta_x", "delta_y", "sigma", "dropped"):
        assert np.array_equal(getattr(short.lift, field), getattr(full.lift, field)), field
    assert short.lift.start == tuple(start - cut for start in full.lift.start)
    assert len(short.lift.channel) + short.lift.dropped == 3 * top
    for row, (channel, start) in enumerate(zip(short.lift.channel, short.lift.start, strict=True)):
        assert start - PRE >= 1 and start + TEST + COOLDOWN - 1 <= length
        dark = short.spend[start - 1 : start - 1 + TEST, short.channels.index(channel)]
        assert short.lift.x[row] == pytest.approx(float(np.mean(dark)), rel=1e-12, abs=0.0)
    with pytest.raises(ValueError, match=f"inside the last {length} weeks, not {top + 1}"):
        SHORT.observe(world, top + 1)


def test_the_world_its_quarter_and_its_best_plan_are_the_drawn_worlds():
    seed = SHORT.pilots["L-78"][2]
    world, base = SHORT.world("L-78", seed), TRACK_M2.world("drawn", seed)
    for field in dataclasses.fields(MediaMixWorld):
        f = field.name
        assert np.array_equal(getattr(world.history, f), getattr(base.history, f)), f
    ours, theirs = SHORT.truth(world), TRACK_M2.truth(base)
    np.testing.assert_array_equal(ours.best.weekly, theirs.best.weekly)
    assert ours.best.worth == theirs.best.worth
    np.testing.assert_array_equal(ours.target, theirs.target)
    np.testing.assert_array_equal(ours.returns.roi, theirs.returns.roi)
    np.testing.assert_array_equal(ours.returns.marginal, theirs.returns.marginal)
    assert theirs.returns.window == (WEEKS - YEAR + 1, WEEKS)
    assert ours.returns.window == (78 - YEAR + 1, 78) == SHORT.observe(world, 0).roi_window

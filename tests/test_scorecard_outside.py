"""Family 6: each channel's curve is the drawn world's over the range of adstock its history shows,
so the history is the drawn world's bit for bit, and the bench's own outside it, continuous at both
ends, with the cliff's exponent uniform on [2, 4] within its Monte Carlo error; the bend bounds the
replaced curve's concavity between its kinks; the tests read the cliff in their dark weeks; the box
is widened for the arm and the oracle alike; no plan in it beats the oracle; and a plan made on the
drawn world's curve loses to it."""

import dataclasses
import math

import numpy as np
import pytest

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.endogenous_mmm import CURVES, MediaMixWorld, _media
from causaldyn_bench.lift_calibration import STARTS
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard.outside import OUTSIDE, SIDES, WIDENED, Replaced
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from causaldyn_bench.scorecard.truth import GAP, _gridded, cells, oracle, worth

TANH = CURVES["tanh"]


def _replaced(name: str, seed: int) -> Replaced:
    history = OUTSIDE.world(name, seed).history
    assert isinstance(history, Replaced)
    return history


def _one(function, a: float, lam: float) -> float:
    """A curve's value or slope at one adstock."""
    return float(function(np.array([a]), lam)[0])


def _adstock(world: MediaMixWorld, c: int) -> np.ndarray:
    """The channel's adstock over the history, as :func:`_media` reads it, at a unit effect."""
    alpha, lam = world.retention[c], world.saturation[c]
    unit = _media(world.spend[:, c], alpha, lam, 1.0, world.kernel_length)
    return 2.0 * np.arctanh(unit) / lam


@pytest.mark.parametrize("name", SIDES)
def test_the_history_is_the_drawn_worlds_and_its_range_the_adstocks(name):
    for seed in OUTSIDE.pilots[name][:3]:
        world, base = _replaced(name, seed), TRACK_M2.world("drawn", seed).history
        for field in dataclasses.fields(MediaMixWorld):
            f = field.name
            assert np.array_equal(getattr(world, f), getattr(base, f)), f
        for c in range(3):
            np.testing.assert_array_equal(world._effect_of(c, world.spend[:, c]), world.media[:, c])
            adstock = _adstock(world, c)
            low, high = world.observed[c]
            assert low == pytest.approx(adstock.min(), rel=1e-9, abs=0.0)
            assert high == pytest.approx(adstock.max(), rel=1e-9, abs=0.0)


@pytest.mark.parametrize("name", SIDES)
def test_the_curve_is_the_drawn_worlds_inside_and_the_benchs_outside(name):
    below, above = SIDES[name]
    world = _replaced(name, OUTSIDE.pilots[name][0])
    for c, curve in enumerate(world.curves()):
        lam, (low, high) = world.saturation[c], world.observed[c]
        inside = np.linspace(low, high, 101)
        np.testing.assert_array_equal(curve.value(inside, lam), TANH.value(inside, lam))
        under, over = np.linspace(0.0, low, 101)[:-1], np.linspace(high, 3.0 * high, 101)[1:]
        power = math.log(_one(curve.value, low / 2, lam) / _one(TANH.value, low, lam)) / math.log(
            0.5
        )
        if below:
            assert 2.0 <= power <= 4.0
            cliff = _one(TANH.value, low, lam) * (under / low) ** power
            np.testing.assert_allclose(curve.value(under, lam), cliff, rtol=1e-12, atol=0.0)
            assert curve.kinks == (low,) and not curve.concave
            # its slope drops at the join, by p sinh(2x) / 2x > 2 (Maxima), x = lam low / 2
            drop = power * math.sinh(lam * low) / (lam * low)
            left = _one(curve.slope, low * (1.0 - 1e-12), lam)
            assert left / _one(TANH.slope, low, lam) == pytest.approx(drop, rel=1e-9, abs=0.0)
        else:
            np.testing.assert_array_equal(curve.value(under, lam), TANH.value(under, lam))
            assert curve.kinks == () and curve.concave
        if above:
            h = high / 4.0
            rise = _one(TANH.slope, high, lam) * h * -np.expm1(-(over - high) / h)
            np.testing.assert_allclose(
                curve.value(over, lam), _one(TANH.value, high, lam) + rise, rtol=1e-14, atol=0.0
            )
            ahead = _one(curve.slope, high * (1.0 + 1e-12), lam)
            assert ahead == pytest.approx(_one(TANH.slope, high, lam), rel=1e-9, abs=0.0)
        else:
            np.testing.assert_array_equal(curve.value(over, lam), TANH.value(over, lam))
        for edge in (low, high):  # continuous at both joins
            near = curve.value(np.array([edge * (1 - 1e-12), edge * (1 + 1e-12)]), lam)
            assert near[0] == pytest.approx(near[1], rel=1e-9, abs=0.0)
        # its slope is its value's, read off central differences a millionth of the top apart
        step = 1e-6 * high
        a = np.concatenate([under[1:], inside[1:-1], over])
        a = a[np.abs(a - low) > 2 * step]
        rise = (curve.value(a + step, lam) - curve.value(a - step, lam)) / (2 * step)
        # near nought the cliff's slope falls to 0: a floor of two millionths of his slope there
        np.testing.assert_allclose(curve.slope(a, lam), rise, rtol=1e-6, atol=1e-6 * lam)


def test_the_cliffs_exponent_is_uniform_on_2_to_4_within_its_monte_carlo_error():
    """Read back off the curve, at half the range's bottom, over the cliff block's 100 pilot worlds
    and their three channels: its mean lies within four standard errors of 3, the error
    ``(2 / sqrt 12) / sqrt 300 = 0.0333``, and its sd within a tenth of ``2 / sqrt 12``."""
    powers = []
    for seed in OUTSIDE.pilots["cliff"]:
        world = _replaced("cliff", seed)
        for c, curve in enumerate(world.curves()):
            lam, low = world.saturation[c], world.observed[c][0]
            powers.append(
                math.log(_one(curve.value, low / 2, lam) / _one(TANH.value, low, lam))
                / math.log(0.5)
            )
    powers = np.array(powers)
    error = 2.0 / math.sqrt(12.0) / math.sqrt(powers.size)
    assert powers.size == 300 and powers.min() >= 2.0 and powers.max() <= 4.0
    assert abs(powers.mean() - 3.0) <= 4.0 * error
    assert powers.std(ddof=1) == pytest.approx(2.0 / math.sqrt(12.0), rel=0.1, abs=0.0)


@pytest.mark.parametrize("name", SIDES)
def test_the_bend_bounds_the_curves_concavity_between_its_kinks(name):
    """``-value''`` from the slope's central differences a millionth of the range's top apart, on
    either side of each kink, stays below the bend; above the range it is greatest at its top."""
    world = _replaced(name, OUTSIDE.pilots[name][1])
    for c, curve in enumerate(world.curves()):
        lam, (low, high) = world.saturation[c], world.observed[c]
        step = 1e-6 * high
        a = np.linspace(step, 3.0 * high, 600_001)
        a = a[np.all(np.abs(a[:, None] - np.array([low, high])) > 2 * step, axis=1)]
        concavity = -(curve.slope(a + step, lam) - curve.slope(a - step, lam)) / (2 * step)
        assert curve.bend is not None and TANH.bend is not None
        bend = curve.bend(lam)
        assert concavity.max() <= bend * (1.0 + 1e-6)
        top = _one(TANH.slope, high, lam) / (high / 4.0)
        assert bend == (max(top, TANH.bend(lam)) if SIDES[name][1] else TANH.bend(lam))
        if SIDES[name][1]:
            # the first point read past the top lies within 7e-6 of it, of a_max, where -value''
            # is e^(-4 * 7e-6) of its greatest
            assert concavity[a > high].max() == pytest.approx(top, rel=3e-5, abs=0.0)


def test_a_geo_test_reads_the_cliff_in_its_dark_weeks():
    """Going dark takes the adstock below the range: the cliff's tests read more lost sales from the
    same spend; the saturation's, where the cliff is not, read the drawn world's, bit for bit."""
    for name in SIDES:
        seed = OUTSIDE.pilots[name][0]
        ours = OUTSIDE.observe(OUTSIDE.world(name, seed), len(STARTS)).lift
        theirs = TRACK_M2.observe(TRACK_M2.world("drawn", seed), len(STARTS)).lift
        np.testing.assert_array_equal(ours.x, theirs.x)
        if SIDES[name][0]:
            assert np.all(ours.delta_y <= theirs.delta_y) and np.any(ours.delta_y < theirs.delta_y)
        else:
            np.testing.assert_array_equal(ours.delta_y, theirs.delta_y)


def test_the_box_is_widened_for_the_arm_and_the_oracle_alike():
    seed = OUTSIDE.pilots["both"][0]
    world = OUTSIDE.world("both", seed)
    narrow = Quarter.after(world.history)
    for quarter in (OUTSIDE.observe(world, 0), OUTSIDE.truth(world).quarter):
        np.testing.assert_array_equal(quarter.lower, WIDENED[0] * narrow.status_quo)
        np.testing.assert_array_equal(quarter.upper, WIDENED[1] * narrow.status_quo)
        np.testing.assert_array_equal(quarter.status_quo, narrow.status_quo)
        assert quarter.budget == narrow.budget


def _dense(on, quarter: Quarter, points: int) -> float:
    """The best worth over a grid of ``points`` a side of the first two cells' boxes, the third
    spending the rest."""
    rate = quarter.budget / PLANNED
    low, high = quarter.lower, quarter.upper
    second = np.linspace(low[1], high[1], points)
    seconds = on[1].worths(second)
    first = np.linspace(low[0], high[0], points)
    best = -np.inf
    for spent, value in zip(first, on[0].worths(first), strict=True):
        rest = rate - spent - second
        inside = (rest >= low[2]) & (rest <= high[2])
        total = value + seconds + on[2].worths(np.clip(rest, low[2], high[2]))
        best = max(best, float(np.max(np.where(inside, total, -np.inf))))
    return best


@pytest.mark.parametrize("name", SIDES)
def test_no_plan_in_the_box_beats_the_oracle(name):
    """A grid of 801 points a side and the searched plan, on three pilot worlds each."""
    for seed in OUTSIDE.pilots[name][:3]:
        truth = OUTSIDE.truth(OUTSIDE.world(name, seed))
        quarter = truth.quarter
        assert quarter.feasible(truth.best.weekly)
        assert all(cell.concave != SIDES[name][0] for cell in truth.cells)
        allowance = GAP * quarter.budget
        assert _dense(truth.cells, quarter, 801) <= truth.best.worth + allowance
        assert _gridded(truth.cells, quarter).worth <= truth.best.worth + allowance


def test_a_plan_made_on_the_drawn_worlds_curve_loses_to_the_oracle():
    """On the pilot blocks' first ten worlds, the plan best on the drawn world's own curve in the
    widened box loses 0.082 per euro on the mean with the cliff, 0.0072 with the saturation and
    0.072 with both; never less than nothing."""
    lost = {}
    for name in SIDES:
        lost[name] = []
        for seed in OUTSIDE.pilots[name][:10]:
            world = OUTSIDE.world(name, seed)
            truth = OUTSIDE.truth(world)
            quarter = truth.quarter
            drawn = oracle(cells(TRACK_M2.world("drawn", seed).history, quarter), quarter)
            gap = truth.best.worth - worth(truth.cells, drawn.weekly)
            lost[name].append(gap / quarter.budget)
    assert min(min(v) for v in lost.values()) >= -TIE
    assert np.mean(lost["cliff"]) > 0.05 and np.mean(lost["both"]) > 0.05
    assert np.mean(lost["saturation"]) > 0.005

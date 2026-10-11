"""Family 4: each world's media are its S-shaped curve's, whose shape the media give back to
rounding; beneath the curve the world is Track M v2's, its spend included; its tests read the
curve; and no plan in the box beats its oracle, whose best plan sits where a concave curve's
would not."""

import numpy as np
import pytest
from scipy.special import expit, logit

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.endogenous_mmm import _MIDPOINT, CURVES, LN3, STEEPNESS
from causaldyn_bench.lift_calibration import STARTS
from causaldyn_bench.scorecard.s_curves import S_CURVES, SHAPES
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from causaldyn_bench.scorecard.truth import worth


def _slope(y: np.ndarray, x: np.ndarray) -> float:
    x = x - x.mean()
    return float(x @ (y - y.mean()) / (x @ x))


def _shape(name: str, g: np.ndarray, z: np.ndarray) -> float:
    """The curve's shape read off its values ``g`` at ``z`` half-saturations: Hill's slope and
    Weibull's shape as log-log slopes of their odds and their cumulative hazard, the logistic's
    steepness as its log-odds' slope in its midpoints."""
    if name == "hill-2":
        return _slope(np.log(g / (1.0 - g)), np.log(z))
    if name == "weibull-2":
        return _slope(np.log(-np.log1p(-g)), np.log(z))
    return _slope(logit(g * expit(STEEPNESS) + expit(-STEEPNESS)), z / _MIDPOINT)


@pytest.mark.parametrize("name", SHAPES)
def test_the_media_give_back_the_curves_shape(name):
    target = {"hill-2": 2.0, "weibull-2": 2.0, "logistic-4": STEEPNESS}[name]
    assert not CURVES[name].concave
    for seed in S_CURVES.pilots[name][:5]:
        history = S_CURVES.world(name, seed).history
        assert history.curve == name
        for c in range(len(history.channels)):
            kernel = history.kernel(c)
            adstock = np.convolve(history.spend[:, c], kernel)[: history.week.size]
            g = history.media[:, c] / history.effect[c]
            z = adstock * history.saturation[c] / LN3
            inside = (g > 1e-6) & (g < 1.0 - 1e-6)
            assert inside.sum() > 100
            assert _shape(name, g[inside], z[inside]) == pytest.approx(target, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("name", SHAPES)
def test_the_curve_moves_the_media_the_sales_and_the_tests_alone(name):
    seed = S_CURVES.pilots[name][1]
    world = S_CURVES.world(name, seed)
    base = TRACK_M2.world("drawn", seed)
    for field in ("spend", "premedia", "plan_noise", "budget", "bidding", "observed_price"):
        assert np.array_equal(getattr(world.history, field), getattr(base.history, field)), field
    assert not np.allclose(world.history.media, base.history.media)
    np.testing.assert_array_equal(
        world.history.sales, world.history.premedia + world.history.media.sum(axis=1)
    )
    ours, theirs = (
        f.observe(w, len(STARTS)).lift for f, w in ((S_CURVES, world), (TRACK_M2, base))
    )
    np.testing.assert_array_equal(ours.x, theirs.x)  # the same spend went dark
    assert not np.allclose(ours.delta_y, theirs.delta_y)  # and returned the S-curve's sales


@pytest.mark.parametrize("name", SHAPES)
def test_no_plan_in_the_box_beats_the_oracle(name):
    rng = np.random.default_rng(4)
    for seed in S_CURVES.pilots[name][:3]:
        truth = S_CURVES.truth(S_CURVES.world(name, seed))
        quarter = truth.quarter
        assert quarter.feasible(truth.best.weekly)
        assert all(not cell.concave and cell.bend is not None for cell in truth.cells)
        for _ in range(200):
            weekly = quarter.project(rng.uniform(quarter.lower, quarter.upper))
            assert worth(truth.cells, weekly) <= truth.best.worth + TIE * quarter.budget


def test_the_best_plan_holds_channels_at_the_box_where_a_concave_curves_would_not():
    """On the S-curve worlds of the pilot blocks' first ten seeds, the best plan holds at least one
    channel on a bound of the box in every world, and the plan best had the curve been his
    ``tanh`` loses at least a tenth of a euro per euro in some."""
    held, lost = [], []
    for name in SHAPES:
        for seed in S_CURVES.pilots[name][:10]:
            world = S_CURVES.world(name, seed)
            truth = S_CURVES.truth(world)
            quarter = truth.quarter
            ends = np.isclose(truth.best.weekly, quarter.lower, rtol=1e-9)
            ends |= np.isclose(truth.best.weekly, quarter.upper, rtol=1e-9)
            held.append(int(ends.sum()))
            concave = TRACK_M2.truth(TRACK_M2.world("drawn", seed)).best.weekly
            gap = truth.best.worth - worth(truth.cells, quarter.project(concave))
            lost.append(gap / quarter.budget)
    assert min(held) >= 1
    assert max(lost) > 0.1 and min(lost) >= -TIE

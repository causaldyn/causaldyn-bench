"""Family 1: each law draws the path its table gives and expects the continuation its formula
gives, which a Monte Carlo of the law's own paths confirms; the drift moves the media, the sales and
the tests and leaves the spend; and the static law is Track M v2 bit for bit."""

import dataclasses
import math

import numpy as np
import pytest

from causaldyn_bench.endogenous_mmm import YEAR, MediaMixWorld
from causaldyn_bench.lift_calibration import STARTS
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard.continuation import Shadow
from causaldyn_bench.scorecard.drift import (
    DRIFT,
    LAWS,
    Revert,
    Season,
    Step,
    Trend,
    Walk,
    expected,
    law,
    onward,
    path,
)
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.scoring import score_world
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from causaldyn_bench.scorecard.truth import cells

WEEKS, CHANNELS = 156, 3
HORIZON = PLANNED + 6 - 1
DRIFTING = [name for name in LAWS if name != "static"]


def _seed(name):
    return DRIFT.scored[name][0]


@pytest.fixture(scope="module")
def worlds():
    return {name: DRIFT.world(name, _seed(name)) for name in LAWS}


def _same(a, b, kind):
    for field in dataclasses.fields(kind):
        left, right = getattr(a, field.name), getattr(b, field.name)
        assert np.array_equal(left, right), field.name


@pytest.mark.parametrize("seed", [_seed("static"), _seed("static") + 1])
def test_the_static_law_is_track_m2_bit_for_bit(seed):
    static, base = DRIFT.world("static", seed), TRACK_M2.world("drawn", seed)
    _same(static.history, base.history, MediaMixWorld)
    _same(static.shadow, base.shadow, Shadow)
    assert np.array_equal(static.expected, base.expected)
    assert np.array_equal(static.realised, base.realised)
    for k in (0, len(STARTS)):
        ours = dataclasses.replace(DRIFT.observe(static, k), family=0, environment="drawn")
        assert ours.digest() == TRACK_M2.observe(base, k).digest()
    ours, theirs = DRIFT.truth(static), TRACK_M2.truth(base)
    assert ours.realised is None and ours.best.worth == theirs.best.worth
    assert np.array_equal(ours.best.weekly, theirs.best.weekly)


@pytest.mark.parametrize("name", DRIFTING)
def test_a_drift_moves_the_media_and_the_sales_and_leaves_the_spend(worlds, name):
    drifted = worlds[name].history
    base = TRACK_M2.world("drawn", _seed(name)).history
    assert np.array_equal(drifted.spend, base.spend)
    assert np.array_equal(drifted.premedia, base.premedia)
    assert np.array_equal(drifted.media, base.media * drifted.multiplier)
    assert np.array_equal(drifted.sales, base.premedia + drifted.media.sum(axis=1))
    assert np.ptp(drifted.multiplier, axis=0).min() > 0.0  # every channel's moves


def _shocks(name, seed, role, shape):
    return np.random.default_rng((DRIFT.stream, role, seed)).standard_normal(shape)


@pytest.mark.parametrize("name", LAWS)
def test_each_law_draws_the_path_its_table_gives(worlds, name):
    seed = _seed(name)
    moving = law(name, seed, CHANNELS, WEEKS)
    x = np.log(worlds[name].history.multiplier)
    t = np.arange(1, WEEKS + 1)[:, None]
    shocks = _shocks(name, seed, 1, (WEEKS + 1, CHANNELS))
    if isinstance(moving, Walk):
        np.testing.assert_allclose(np.diff(x, axis=0, prepend=0.0), moving.sd * shocks[1:])
    elif isinstance(moving, Revert):
        phi, s = moving.persistence, moving.sd
        before = np.vstack([s * shocks[0], x[:-1]])
        np.testing.assert_allclose(x - phi * before, s * np.sqrt(1 - phi**2) * shocks[1:])
    elif isinstance(moving, Trend):
        np.testing.assert_allclose(x, moving.slope * (t - 1) / (WEEKS - 1), atol=1e-15)
        np.testing.assert_allclose(x[-1], moving.slope)
    elif isinstance(moving, Step):
        np.testing.assert_allclose(x, np.where(t >= moving.when, moving.size, 0.0), atol=1e-15)
    elif isinstance(moving, Season):
        wave = moving.amplitude * np.sin(2 * math.pi * t / YEAR + moving.phase)
        np.testing.assert_allclose(x, wave, atol=1e-15)
        np.testing.assert_allclose(x[YEAR:], x[:-YEAR], atol=1e-15)
    else:
        assert name == "static" and np.all(x == 0.0)


def test_each_laws_parameters_lie_in_their_ranges():
    drawn = {
        name: [law(name, s, CHANNELS, WEEKS) for s in DRIFT.scored[name][:200]] for name in LAWS
    }

    def stacked(name, part):
        return np.concatenate([getattr(m, part) for m in drawn[name]])

    sd = stacked("walk", "sd")
    assert 0.01 <= sd.min() <= sd.max() <= 0.04
    assert np.mean(sd < 0.02) == pytest.approx(0.5, abs=0.06)  # log-uniform: half below 0.02
    slope = stacked("trend", "slope")
    assert -0.7 <= slope.min() < -0.6 and 0.6 < slope.max() <= 0.7
    when, size = stacked("step", "when"), stacked("step", "size")
    assert (when.min(), when.max()) == (53, 143) and when.dtype.kind == "i"
    assert 0.2 <= np.abs(size).min() <= np.abs(size).max() <= 0.7
    assert np.mean(size > 0) == pytest.approx(0.5, abs=0.06)
    amplitude, phase = stacked("season", "amplitude"), stacked("season", "phase")
    assert 0.1 <= amplitude.min() <= amplitude.max() <= 0.4
    assert 0.0 <= phase.min() < 0.1 and 2 * math.pi - 0.1 < phase.max() < 2 * math.pi
    persistence, s = stacked("revert", "persistence"), stacked("revert", "sd")
    assert 0.8 <= persistence.min() <= persistence.max() <= 0.97
    assert 0.15 <= s.min() <= s.max() <= 0.4


def _formula(name, moving, last, h):
    """The design's expected multiplier ``h`` weeks past the history, written from its table."""
    if isinstance(moving, Walk):
        return np.exp(last) * np.exp(h * moving.sd**2 / 2)
    if isinstance(moving, Revert):
        phi, s = moving.persistence, moving.sd
        return np.exp(phi**h * last + s**2 * (1 - phi ** (2 * h)) / 2)
    if isinstance(moving, Trend):
        return np.exp(moving.slope * (WEEKS + h - 1) / (WEEKS - 1))
    if isinstance(moving, Step):
        return np.exp(last)
    if isinstance(moving, Season):
        return np.exp(moving.amplitude * np.sin(2 * math.pi * (WEEKS + h) / YEAR + moving.phase))
    assert name == "static"
    return np.ones(CHANNELS)


@pytest.mark.parametrize("name", LAWS)
def test_each_laws_expected_continuation_is_its_formula_and_the_mean_of_its_own_paths(worlds, name):
    world = worlds[name]
    moving = law(name, world.seed, CHANNELS, WEEKS)
    last = np.log(world.history.multiplier[-1])
    formula = np.stack([_formula(name, moving, last, h) for h in range(1, HORIZON + 1)])
    ours = expected(moving, last, WEEKS, HORIZON)
    np.testing.assert_allclose(ours, formula, rtol=1e-13)
    effect = np.asarray(world.history.effect)[:, None]
    np.testing.assert_allclose(world.expected, effect * ours.T, rtol=1e-13)
    paths = 200_000
    shocks = np.random.default_rng(20_261_003).standard_normal((HORIZON, paths, CHANNELS))
    multipliers = np.exp(onward(moving, last, WEEKS, shocks))
    mean = multipliers.mean(axis=1)
    error = multipliers.std(axis=1) / math.sqrt(paths)
    assert np.all(np.abs(mean - formula) <= 4.5 * error + 1e-12 * formula)


def _variance(moving, horizon):
    h = np.arange(1, horizon + 1)[:, None]
    if isinstance(moving, Walk):
        return h * moving.sd**2
    assert isinstance(moving, Revert)
    return moving.sd**2 * (1 - moving.persistence ** (2 * h))


def test_without_its_variance_term_a_laws_continuation_misses_its_own_paths(worlds):
    for name in ("walk", "revert"):
        world = worlds[name]
        moving = law(name, world.seed, CHANNELS, WEEKS)
        last = np.log(world.history.multiplier[-1])
        paths = 200_000
        shocks = np.random.default_rng(20_261_003).standard_normal((HORIZON, paths, CHANNELS))
        multipliers = np.exp(onward(moving, last, WEEKS, shocks))
        error = multipliers.std(axis=1) / math.sqrt(paths)
        naive = expected(moving, last, WEEKS, HORIZON) * np.exp(-_variance(moving, HORIZON) / 2)
        assert np.max(np.abs(multipliers.mean(axis=1) - naive) / error) > 10


def test_the_revert_law_holds_its_stationary_sd_and_persistence(worlds):
    moving = law("revert", worlds["revert"].seed, CHANNELS, WEEKS)
    assert isinstance(moving, Revert)
    paths = 20_000
    x = path(moving, np.random.default_rng(7).standard_normal((WEEKS + 1, paths, CHANNELS)))
    for week in (0, 77, WEEKS - 1):
        np.testing.assert_allclose(x[week].std(axis=0), moving.sd, rtol=0.03)
    lagged = np.array([np.corrcoef(x[99, :, c], x[100, :, c])[0, 1] for c in range(CHANNELS)])
    np.testing.assert_allclose(lagged, moving.persistence, atol=0.015)


@pytest.mark.parametrize("name", ["walk", "step"])
def test_a_lift_test_reads_the_effect_of_its_own_weeks(worlds, name):
    drifted = worlds[name].history
    base = TRACK_M2.world("drawn", _seed(name)).history
    for c, channel in enumerate(base.channels):
        ours = drifted.geo_test(channel, STARTS, seed=5).true_gap
        theirs = base.geo_test(channel, STARTS, seed=5).true_gap
        np.testing.assert_allclose(ours, theirs * drifted.multiplier[:, c], rtol=1e-12, atol=1e-9)
        assert np.any(ours != theirs)
    rows = DRIFT.observe(worlds[name], len(STARTS)).lift
    base_rows = TRACK_M2.observe(TRACK_M2.world("drawn", _seed(name)), len(STARTS)).lift
    assert not np.array_equal(rows.delta_y, base_rows.delta_y)


def test_the_best_plan_is_made_for_the_expected_path_and_the_quarter_realises_its_own(worlds):
    world = worlds["walk"]
    moving = law("walk", world.seed, CHANNELS, WEEKS)
    assert isinstance(moving, Walk)
    last = np.log(world.history.multiplier[-1])
    shocks = _shocks("walk", world.seed, 3, (HORIZON, CHANNELS))
    effect = np.asarray(world.history.effect)[:, None]
    np.testing.assert_allclose(
        world.realised, effect * np.exp(last + np.cumsum(moving.sd * shocks, axis=0)).T
    )
    truth = DRIFT.truth(world)
    assert truth.realised is not None and truth.hindsight is not None
    assert all(
        np.array_equal(c.effect, e) for c, e in zip(truth.cells, world.expected, strict=True)
    )
    assert all(
        np.array_equal(c.effect, e) for c, e in zip(truth.realised, world.realised, strict=True)
    )
    quarter = Quarter.after(world.history)
    media = [
        c.returns(float(w))[:PLANNED]
        for c, w in zip(
            cells(world.history, quarter, world.realised), quarter.status_quo, strict=True
        )
    ]
    np.testing.assert_array_equal(world.shadow.media, np.column_stack(media))
    for name in ("trend", "step", "season"):
        assert DRIFT.truth(worlds[name]).realised is None


def test_a_drifting_world_is_scored_on_both_paths(worlds):
    scored = score_world(DRIFT, "walk", _seed("walk"), 0, {})
    assert (scored.family, scored.environment, scored.index, scored.k) == (1, "walk", 0, 0)
    for arm in scored.arms.values():
        assert arm.regret >= 0.0 and arm.realised is not None and arm.realised >= 0.0
    static = score_world(DRIFT, "static", _seed("static"), 0, {})
    assert all(arm.realised is None for arm in static.arms.values())


def test_the_family_draws_its_own_blocks():
    assert number(DRIFT) == 1 and tuple(DRIFT.scored) == tuple(LAWS) == tuple(DRIFT.pilots)
    for e, name in enumerate(LAWS):
        assert DRIFT.scored[name] == range(221_000 + 1_000 * e, 222_000 + 1_000 * e)
        assert DRIFT.pilots[name] == range(220_000 + 100 * e, 220_100 + 100 * e)
    assert all(label.endswith(".") for label in DRIFT.labels)
    with pytest.raises(ValueError, match="no environment 'drawn'"):
        DRIFT.world("drawn", 221_000)
    with pytest.raises(ValueError, match="no law 'drawn'"):
        law("drawn", 221_000, CHANNELS, WEEKS)

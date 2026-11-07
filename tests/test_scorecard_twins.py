"""Family 17: the twins draw one history from their own block and part only in television's
response; along the ramp they return the same at the week's level, so neither a ridge's held-out
score nor a go-dark test tells them apart; on the bench's own truth each twin's best plan costs the
other more than 0.05 of a euro; and an arm reads the twins, their lift tests and their export as
every family's."""

import math
from statistics import NormalDist

import numpy as np
import pytest

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.observe import VERSION, export, read
from causaldyn_bench.scorecard.scoring import score_world
from causaldyn_bench.scorecard.seeds import pilot, scored
from causaldyn_bench.scorecard.truth import oracle, worth
from causaldyn_bench.scorecard.twins import (
    BEND,
    ENVIRONMENTS,
    HALF,
    HILL,
    LEVELS,
    RISE,
    SPREAD,
    TWINS,
    WEEKS,
    curve,
    drift,
    ramp,
)

SEEDS = TWINS.pilots["curve"]


def test_the_twins_draw_one_history_from_their_own_block():
    assert number(TWINS) == 17 and tuple(TWINS.pilots) == tuple(TWINS.scored) == ENVIRONMENTS
    for name in ENVIRONMENTS:
        assert TWINS.pilots[name] == pilot(17, 0) and TWINS.scored[name] == scored(17, 0)
    assert all(label.endswith(".") for label in TWINS.labels)
    seed = SEEDS[0]
    line, bent = TWINS.world("line", seed), TWINS.world("curve", seed)
    np.testing.assert_array_equal(line.spend, bent.spend)
    np.testing.assert_array_equal(line.media[:, 1], bent.media[:, 1])
    assert not np.array_equal(line.sales, bent.sales)
    np.testing.assert_allclose(
        line.sales - line.media.sum(axis=1), bent.sales - bent.media.sum(axis=1), rtol=1e-14
    )
    again, other = TWINS.world("line", seed), TWINS.world("line", seed + 1)
    for k in (0, 4):
        assert TWINS.observe(line, k).digest() == TWINS.observe(again, k).digest()
        assert TWINS.observe(line, k).digest() != TWINS.observe(other, k).digest()
        assert TWINS.observe(line, k).digest() != TWINS.observe(bent, k).digest()
    with pytest.raises(ValueError, match="no environment 'drawn'"):
        TWINS.world("drawn", seed)


def test_along_the_ramp_the_twins_return_the_same_at_the_weeks_level():
    """The line's slope is the curve's average return at the week's level: 1.30 in the first week,
    3.09 at most where the level passes 76 (``60 cbrt 2``, where ``x^2 / (x^3 + 60^3)`` peaks) and
    875 / 304, 2.88, once the ramp holds (scripts/twin_hill_bend.mac)."""
    t = np.arange(WEEKS)
    level = ramp(t)
    np.testing.assert_allclose(drift(t) * level, curve(level), rtol=1e-14)
    assert level[0] == LEVELS[0] and level[RISE - 1] == LEVELS[1]
    assert np.all(level[RISE:] == LEVELS[1])
    assert float(drift(0)) == pytest.approx(1.30, rel=0.0, abs=0.005)
    peak = int(np.argmax(drift(t)))
    assert float(drift(peak)) == pytest.approx(3.09, rel=0.0, abs=0.005)
    assert abs(float(ramp(peak)) - 60.0 * 2.0 ** (1.0 / 3.0)) < 70.0 / (RISE - 1)
    assert float(drift(WEEKS - 1)) == pytest.approx(875.0 / 304.0, rel=1e-15, abs=0.0)


def test_the_hills_bend_bounds_its_concavity():
    """BEND is the largest ``-f''`` of the Hill of slope 3 in its half-saturations (Maxima,
    scripts/twin_hill_bend.mac); a grid finer than its peak's width finds nothing above it."""
    assert pytest.approx(0.879475680097885, rel=1e-15, abs=0.0) == BEND
    a = np.linspace(1e-6, 20.0 * HALF, 400_001)
    slope = HILL.slope(a, HALF)
    bent = -np.gradient(slope, a)
    bend = HILL.bend
    assert bend is not None
    bound = bend(HALF)
    assert bound == BEND / HALF**2
    assert np.max(bent) <= bound * (1.0 + 1e-6)
    assert np.max(bent) >= bound * (1.0 - 1e-4)


def test_televisions_spend_holds_its_level_and_search_its_mean():
    """Over the 100 pilot worlds, 15 600 weeks, television's spend over its level and search's have
    means 1 and 50 and log-scales 0.05 and 0.5, each within five of its standard errors."""
    tv, paid = np.concatenate([TWINS.world("curve", seed).spend for seed in SEEDS]).T
    ratio = tv / np.tile(ramp(np.arange(WEEKS)), len(SEEDS))
    n = ratio.size
    assert abs(ratio.mean() - 1.0) < 5.0 * SPREAD / math.sqrt(n)
    assert abs(paid.mean() - 50.0) < 5.0 * 50.0 * math.sqrt(math.expm1(0.25) / n)
    assert abs(np.log(ratio).std(ddof=1) - SPREAD) < 5.0 * SPREAD / math.sqrt(2.0 * n)
    assert abs(np.log(paid).std(ddof=1) - 0.5) < 5.0 * 0.5 / math.sqrt(2.0 * n)


def _held_out(world) -> np.ndarray:
    """Each held-out week's log density under a ridge fitted to the ramp's weeks: television
    through a Hill of slope 3 half-saturated at 60, search through one of slope 1 half-saturated at
    50, neither carried over, the intercept free and the two coefficients penalised by a thousandth
    a week; its noise the fit's residuals' root mean square."""
    tv, paid = world.spend.T
    features = np.column_stack([tv**3 / (tv**3 + 60.0**3), paid / (paid + 50.0)])
    fit = slice(0, RISE)
    centre, level = features[fit].mean(axis=0), world.sales[fit].mean()
    centred = features[fit] - centre
    beta = np.linalg.solve(
        centred.T @ centred + 1e-3 * RISE * np.eye(2), centred.T @ (world.sales[fit] - level)
    )
    mean = level + (features - centre) @ beta
    sigma = math.sqrt(np.mean((world.sales[fit] - mean[fit]) ** 2))
    z = (world.sales[RISE:] - mean[RISE:]) / sigma
    return -0.5 * z**2 - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def test_a_ridge_scores_the_twins_held_out_weeks_alike_while_their_best_plans_part():
    """A ridge fitted to each twin's 117 weeks of the ramp scores the 39 weeks after alike: the 95 %
    interval of the summed weekly difference of the log scores, line less curve, holds 0 on 89 of
    the block's 100 pilot worlds (asserted: at least 85), and on its first. And on the bench's
    truth, the oracle of the scorecard, the line's best plan spends 110.81 a week on television and
    the curve's 94.69, and each costs the other twin more than 0.05 of a euro of the budget: 0.0948
    in the curve's and 0.0609 in the line's (scripts/twin_hill_bend.mac)."""
    quantile = NormalDist().inv_cdf(0.975)
    held = []
    for seed in SEEDS:
        gap = _held_out(TWINS.world("line", seed)) - _held_out(TWINS.world("curve", seed))
        held.append(abs(gap.sum()) < quantile * gap.std(ddof=1) * math.sqrt(gap.size))
    assert held[0] and sum(held) >= 85
    for seed in SEEDS[:3]:
        line, bent = (TWINS.truth(TWINS.world(name, seed)) for name in ("line", "curve"))
        assert line.best.weekly[0] == pytest.approx(110.807335646357, rel=1e-6, abs=0.0)
        assert bent.best.weekly[0] == pytest.approx(94.6934837826523, rel=1e-6, abs=0.0)
        budget = line.quarter.budget
        in_curve = (bent.best.worth - worth(bent.cells, line.best.weekly)) / budget
        in_line = (line.best.worth - worth(line.cells, bent.best.weekly)) / budget
        assert in_curve == pytest.approx(0.0947634154423046, rel=1e-5, abs=0.0)
        assert in_line == pytest.approx(0.0609361208479292, rel=1e-5, abs=0.0)
        assert min(in_curve, in_line) > 0.05


def test_swapping_the_twins_response_moves_the_oracle():
    """The line's cells, at the curve's twin's quarter, plan the line's 110.81 and not 94.69."""
    seed = SEEDS[0]
    line, bent = (TWINS.truth(TWINS.world(name, seed)) for name in ("line", "curve"))
    swapped = (line.cells[0], bent.cells[1])
    moved = oracle(swapped, bent.quarter)
    np.testing.assert_allclose(moved.weekly, line.best.weekly, rtol=1e-6)
    assert moved.weekly[0] - bent.best.weekly[0] > 16.0


def test_a_go_dark_test_reads_the_twins_alike():
    """At the ladder's top rung each of television's go-dark tests reads the twins within half of
    its own standard error on each of the first 20 pilot worlds: the tests read ``curve(x)`` in one
    twin and ``drift(t) x`` in the other at a spend near the week's level."""
    for seed in SEEDS[:20]:
        line, bent = (TWINS.observe(TWINS.world(name, seed), 4) for name in ("line", "curve"))
        rows = [r for r, channel in enumerate(line.lift.channel) if channel == "tv"]
        assert len(rows) == 4
        np.testing.assert_array_equal(line.lift.x, bent.lift.x)
        apart = np.abs(line.lift.delta_y[rows] - bent.lift.delta_y[rows])
        assert np.all(apart < 0.5 * line.lift.sigma[rows])


def test_an_arm_reads_the_twins_and_their_export_reads_back(tmp_path):
    seed = SEEDS[0]
    world = TWINS.world("curve", seed)
    for k in (0, 4):
        observation = TWINS.observe(world, k)
        assert observation.controls == {} and observation.future_controls == {}
        assert observation.kernel_length == 1 and observation.channels == ("tv", "search")
        assert len(observation.lift.channel) == 2 * k
        back = read(export(observation, tmp_path / str(k)))
        assert back.digest() == observation.digest() and back.version == VERSION
        np.testing.assert_array_equal(back.spend, world.spend)
        np.testing.assert_array_equal(back.sales, world.sales)


def test_each_twins_best_plan_scores_nought_and_the_others_more_than_0_05():
    seed = SEEDS[0]
    for name, other in (("curve", "line"), ("line", "curve")):
        world = TWINS.world(name, seed)
        observation = TWINS.observe(world, 4)

        def record(weekly, observation=observation):
            return {
                "digest": observation.digest(),
                "version": VERSION,
                "error": None,
                "weekly": [float(w) for w in weekly],
            }

        plans = {
            "own": record(TWINS.truth(world).best.weekly),
            "twin": record(TWINS.truth(TWINS.world(other, seed)).best.weekly),
        }
        scores = score_world(TWINS, name, seed, 4, plans, pilot=True)
        assert (scores.family, scores.environment, scores.channels) == (17, name, 2)
        assert abs(scores.arms["own"].regret) <= TIE
        assert scores.arms["twin"].regret > 0.05
        assert scores.arms["status quo"].regret > TIE

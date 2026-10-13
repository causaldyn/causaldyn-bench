"""Family 10, family 8 at three geos, at eight and at fifty small, noisy ones: each world its seed
alone, drawn from the family's blocks at double precision whatever the caller's setting; each
environment family 8's mix at its count of geos, its median population and its noise a head, its
spreads from the family's own stream; fifty small geos hold family 8's market and media share and
are several times noisier, geo by geo; every cell is concave, so the oracle is exact, checked
against a dense grid on three geos and a search written apart from it on eight and fifty; on three
geos a test takes two geos dark or one, and is exported either way; a test's error over its
standard error has a spread of 1 at every count of geos; and every world exports, reads back and
scores, the oracle's own plan at nought, the equal split and the status quo in cells."""

import dataclasses
import itertools
import json
import math

import jax
import numpy as np
import pytest
from chc.geo_world import GeoMediaWorld
from scipy.optimize import minimize

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.lift_calibration import COOLDOWN, PRE, TEST
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard import truth
from causaldyn_bench.scorecard.continuation import lifted
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.few_geos import FAMILY, FEW_GEOS, NOISY, SIZES, SMALL
from causaldyn_bench.scorecard.geo import GEO, NOISE, POPULATION, mix
from causaldyn_bench.scorecard.ladder import GEO_STARTS, dark_geos, geo_rung
from causaldyn_bench.scorecard.observe import VERSION, export, read
from causaldyn_bench.scorecard.scoring import WorldRecord, score_world
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored
from causaldyn_bench.scorecard.truth import allotted, geo_cells, geo_quarter, oracle

LENGTH = 8  # the kernel's
HORIZON = PLANNED + LENGTH - 1


def sloped_reading(ours: np.ndarray, theirs: np.ndarray, first: int) -> float:
    """A geo test's reading from week ``first``: each group's KPI less its mean over the ``PRE``
    weeks before, the others' times the least-squares slope of ``ours`` on ``theirs`` over the weeks
    clear of the kernel's first seven and of every test's window, summed over the readout."""
    clear = np.ones(ours.size, dtype=bool)
    clear[:7] = False
    for start in GEO_STARTS:
        clear[start - 1 - PRE : start - 1 + TEST + COOLDOWN] = False
    slope = np.polyfit(theirs[clear], ours[clear], 1)[0]
    before = slice(first - PRE, first)
    gap = ours - ours[before].mean() - slope * (theirs - theirs[before].mean())
    return float(gap[first : first + TEST + COOLDOWN].sum()) / TEST


def test_the_family_draws_its_three_environments_from_its_own_blocks():
    assert number(FEW_GEOS) == FAMILY == 10 and tuple(FEW_GEOS.pilots) == tuple(FEW_GEOS.scored)
    assert {name: geos.count for name, geos in SIZES.items()} == {
        "geos-3": 3,
        "geos-8": 8,
        "geos-50": 50,
    }
    for e, name in enumerate(SIZES):
        assert FEW_GEOS.pilots[name] == pilot(10, e) and FEW_GEOS.scored[name] == scored(10, e)
    assert FEW_GEOS.labels == GEO.labels
    with pytest.raises(ValueError, match="no environment 'policy-0'"):
        FEW_GEOS.world("policy-0", scored(10, 0)[0])


@pytest.mark.parametrize("name", ["geos-3", "geos-50"])
def test_a_world_is_its_seed_alone_at_double_precision_whatever_the_setting(name):
    seed = FEW_GEOS.scored[name][1]
    first, again, other = (FEW_GEOS.world(name, s) for s in (seed, seed, seed + 1))
    with jax.enable_x64(True):
        wide = FEW_GEOS.world(name, seed)
    assert first.history.media.dtype == np.float64
    for k in (0, 4):
        stamp = FEW_GEOS.observe(first, k).digest()
        assert stamp == FEW_GEOS.observe(again, k).digest() == FEW_GEOS.observe(wide, k).digest()
        assert stamp != FEW_GEOS.observe(other, k).digest()
    ours, theirs = FEW_GEOS.truth(first), FEW_GEOS.truth(again)
    assert ours.best.worth == theirs.best.worth
    np.testing.assert_array_equal(ours.best.weekly, theirs.best.weekly)
    np.testing.assert_array_equal(ours.target, theirs.target)


@pytest.mark.parametrize("name", list(SIZES))
def test_an_environment_is_family_8s_mix_at_its_geos_population_and_noise(name):
    """Every parameter but the count of geos, their median population and their noise a head is
    family 8's at policy 0; each channel's spread is drawn from the family's own stream."""
    seed = FEW_GEOS.pilots[name][3]
    world = FEW_GEOS.world(name, seed).history
    geos = SIZES[name]
    assert (world.mix.geos, world.mix.population, world.mix.noise) == geos
    expected = rng(10, Role.PARAMETERS, seed).uniform(0.2, 0.6, 3)
    np.testing.assert_array_equal([c.spread for c in world.mix.channels], expected)
    assert not np.array_equal(expected, rng(8, Role.PARAMETERS, seed).uniform(0.2, 0.6, 3))
    eight = mix(0.0, seed, family=10)
    for field in dataclasses.fields(world.mix):
        if field.name not in {"channels", "geos", "population", "noise"}:
            assert getattr(world.mix, field.name) == getattr(eight, field.name), field.name

    def stated(channel) -> tuple:
        curve = channel.curve
        return (
            channel.name,
            type(curve).__name__,
            float(curve.scale),
            channel.effect,
            channel.spread,
            channel.retention,
            channel.spend,
            channel.national,
        )

    assert [stated(c) for c in world.mix.channels] == [stated(c) for c in eight.channels]
    assert world.mix.policy == 0.0 and world.spend.shape == (geos.count, 104, 3)


def test_fifty_small_geos_hold_family_8s_market_and_media_share_and_are_several_times_noisier():
    """A fifth of the median population, so fifty geos hold ten of family 8's in expectation, and
    twice the noise a head, so a median geo's noise over its mean KPI is ``2 sqrt(5)`` times
    family 8's (Maxima). On the first 30 pilot worlds of each, the market's mean population, media's
    share of the KPI and a geo's noise over its mean KPI, the median over the geos, read so."""
    assert 50 * SMALL == 10 * POPULATION and NOISY == 2 * NOISE
    at_median = (NOISY / math.sqrt(SMALL)) / (NOISE / math.sqrt(POPULATION))
    assert at_median == pytest.approx(2 * math.sqrt(5), rel=1e-15, abs=0.0)

    def measured(family, name: str) -> tuple[float, float, float]:
        markets, shares, ratios = [], [], []
        for seed in family.pilots[name][:30]:
            world = family.world(name, seed).history
            markets.append(float(world.population.sum()))
            shares.append(float(world.media.sum() / world.kpi.sum()))
            mean = (world.kpi - world.noise).mean(axis=1)
            ratios += list(world.mix.noise * np.sqrt(world.population) / mean)
        return float(np.mean(markets)), float(np.mean(shares)), float(np.median(ratios))

    small, ten = measured(FEW_GEOS, "geos-50"), measured(GEO, "policy-0")
    assert small[0] == pytest.approx(ten[0], rel=0.05, abs=0.0)
    assert small[1] == pytest.approx(ten[1], abs=0.005)
    assert 4.0 < small[2] / ten[2] < 5.0


@pytest.mark.parametrize("name", ["geos-3", "geos-50"])
def test_a_cells_worth_is_what_chcs_world_lifts_over_the_quarter_and_its_tail(name):
    world = FEW_GEOS.world(name, FEW_GEOS.pilots[name][2]).history
    quarter = geo_quarter(world)
    on = geo_cells(world, quarter)
    assert len(on) == 2 * SIZES[name].count + 1 and all(cell.concave for cell in on)
    draw = np.random.default_rng(1)
    plans = [quarter.status_quo, quarter.lower, quarter.upper] + [
        quarter.project(draw.uniform(quarter.lower, quarter.upper)) for _ in range(3)
    ]
    for weekly in plans:
        kpi = lifted(world, allotted(world, weekly), HORIZON)
        assert truth.worth(on, weekly) == pytest.approx(float(kpi.sum()), rel=1e-12, abs=0.0)


def _apart(on, quarter: Quarter, starts: list[np.ndarray]) -> float:
    """The best worth SLSQP reaches from ``starts`` with finite-difference gradients, each cell's
    spend in units of its status quo: a search that shares nothing with the oracle's. The worth is
    a sum over the cells, so each cell's difference reads that cell alone."""
    unit = quarter.status_quo
    rate = quarter.budget / PLANNED
    step = 1e-7

    def gradient(u: np.ndarray) -> np.ndarray:
        return (
            np.array(
                [
                    (cell.worth((x + step) * w) - cell.worth((x - step) * w)) / (2 * step)
                    for cell, x, w in zip(on, u, unit, strict=True)
                ]
            )
            / -quarter.budget
        )

    best = -math.inf
    for start in starts:
        solution = minimize(
            lambda u: -truth.worth(on, u * unit) / quarter.budget,
            start / unit,
            jac=gradient,
            method="SLSQP",
            bounds=list(zip(quarter.lower / unit, quarter.upper / unit, strict=True)),
            constraints={"type": "eq", "fun": lambda u: float(u @ unit) / rate - 1.0},
            options={"ftol": 1e-15, "maxiter": 1000},
        )
        weekly = solution.x * unit
        if quarter.feasible(weekly, tolerance=1e-9):
            best = max(best, truth.worth(on, weekly))
    return best


def _dense(on, quarter: Quarter, points: int) -> tuple[float, np.ndarray]:
    """The best worth, and its plan, on a grid of ``points`` a side over every cell's box but the
    last's, the last spending the rest: a search written apart from the oracle."""
    rate = quarter.budget / PLANNED
    low, high = quarter.lower, quarter.upper
    axes = [np.linspace(low[c], high[c], points) for c in range(len(on) - 1)]
    values = [cell.worths(axis) for cell, axis in zip(on[:-1], axes, strict=True)]
    inner = np.meshgrid(*axes[1:], indexing="ij")
    inner_spent = sum(inner)
    inner_value = sum(np.meshgrid(*values[1:], indexing="ij"))
    best, plan = -math.inf, np.zeros(len(on))
    for first, first_value in zip(axes[0], values[0], strict=True):
        rest = rate - first - inner_spent
        inside = (rest >= low[-1]) & (rest <= high[-1])
        total = first_value + inner_value + on[-1].worths(np.clip(rest, low[-1], high[-1]))
        total = np.where(inside, total, -np.inf)
        at = np.unravel_index(int(np.argmax(total)), total.shape)
        if total[at] > best:
            best = float(total[at])
            plan = np.array([first, *(grid[at] for grid in inner), rest[at]])
    return best, plan


def test_on_three_geos_no_point_of_a_dense_grid_beats_the_oracle_and_its_best_refines_to_it():
    """Three geos: six own cells and TV's, a grid of 17 points a side over six of them, 24 million
    plans; a search from its best point, apart from the oracle, lands on the oracle's worth and not
    above it."""
    world = FEW_GEOS.world("geos-3", FEW_GEOS.pilots["geos-3"][5]).history
    quarter = geo_quarter(world)
    on = geo_cells(world, quarter)
    assert len(on) == 7
    plan = oracle(on, quarter)
    grid, at = _dense(on, quarter, 17)
    assert truth.worth(on, quarter.status_quo) <= grid <= plan.worth + TIE * quarter.budget
    refined = _apart(on, quarter, [quarter.project(at)])
    assert plan.worth * (1 - 1e-10) <= refined <= plan.worth + TIE * quarter.budget


@pytest.mark.parametrize("name", ["geos-8", "geos-50"])
def test_on_eight_and_fifty_geos_the_exact_plan_meets_its_conditions_and_no_search_beats_it(name):
    world = FEW_GEOS.world(name, FEW_GEOS.pilots[name][4])
    found = FEW_GEOS.truth(world)
    quarter, plan, on = found.quarter, found.best, found.cells
    assert quarter.feasible(plan.weekly) and math.isfinite(plan.price)
    free = 0
    for cell, weekly, low, high in zip(on, plan.weekly, quarter.lower, quarter.upper, strict=True):
        slope = cell.slope(float(weekly)) / PLANNED
        if np.isclose(weekly, low, rtol=1e-9):
            assert slope <= plan.price * (1 + 1e-9)
        elif np.isclose(weekly, high, rtol=1e-9):
            assert slope >= plan.price * (1 - 1e-9)
        else:
            free += 1
            assert slope == pytest.approx(plan.price, rel=1e-8, abs=0.0)
    assert free >= 1
    draw = np.random.default_rng(7)
    starts = [quarter.status_quo, quarter.project(draw.uniform(quarter.lower, quarter.upper))]
    apart = _apart(on, quarter, starts)
    assert plan.worth * (1 - 1e-8) <= apart <= plan.worth + TIE * quarter.budget


@pytest.mark.parametrize("name", ["geos-3", "geos-50"])
def test_the_equal_split_gives_each_channel_a_third_and_the_status_quo_each_cell_its_mean(name):
    """Family 8's rules at any count of geos: each channel a third of the budget, a per-geo
    channel's third split across the geos by population and TV's on its one cell, moved into the
    box; every cell at its mean over the last year, TV's its spend over the geos summed; both
    scored as the harness's built-in arms."""
    seed = FEW_GEOS.pilots[name][1]
    world = FEW_GEOS.world(name, seed)
    history, geos = world.history, SIZES[name].count
    wide = geo_quarter(history, box=(0.0, 1e3))
    rate = wide.budget / PLANNED
    weekly = wide.equal_split()
    local = weekly[: 2 * geos].reshape(geos, 2)
    np.testing.assert_allclose(local, np.outer(history.shares, [rate / 3.0] * 2), rtol=1e-9)
    assert weekly[2 * geos] == pytest.approx(rate / 3.0, rel=1e-9, abs=0.0)
    quarter = geo_quarter(history)
    split = quarter.equal_split()
    assert quarter.feasible(split)
    np.testing.assert_allclose(split, quarter.project(weekly), rtol=1e-9)
    own = (history.spend[g, :, c] for g in range(geos) for c in range(2))
    spend = np.column_stack([*own, history.spend[:, :, 2].sum(axis=0)])
    np.testing.assert_array_equal(quarter.status_quo, spend[-52:].mean(axis=0))
    on = FEW_GEOS.truth(world)
    scores = score_world(FEW_GEOS, name, seed, 0, {}, pilot=True)
    for arm, plan in (("equal split", split), ("status quo", quarter.status_quo)):
        regret = (on.best.worth - truth.worth(on.cells, plan)) / quarter.budget
        assert scores.arms[arm].regret == pytest.approx(regret, rel=1e-12, abs=0.0)
        assert regret > TIE


def _true_change(world: GeoMediaWorld, c: int, j: int) -> float:
    """What the ``j``-th test of channel ``c`` costs its tested geos over its readout, a dark
    week, from chc's world: its lift with the geos dark less its lift as spent."""
    start, geos = GEO_STARTS[j], list(dark_geos(world.mix.geos, j))
    dark = world.spend.copy()
    dark[geos, start - 1 : start - 1 + TEST, c] = 0.0
    with jax.enable_x64(True):
        gap = np.asarray(world.lift(dark)) - np.asarray(world.lift(world.spend))
    return float(gap[geos, start - 1 : start - 1 + TEST + COOLDOWN].sum()) / TEST


def test_on_three_geos_a_test_takes_two_geos_dark_or_one_and_every_test_is_exported():
    """Alternate geos of three: geos 0 and 2 in the first and third test, geo 1 alone in the second
    and fourth, each read against the others; every test is a row, its noise from the family's
    tests' stream, ``(110, 2, seed, c + 1)``."""
    seed = FEW_GEOS.pilots["geos-3"][6]
    world = FEW_GEOS.world("geos-3", seed).history
    rows = FEW_GEOS.observe(FEW_GEOS.world("geos-3", seed), 4).lift
    assert rows.geos is not None and rows.dropped == 0 and len(rows.channel) == 12
    names = [c.name for c in world.mix.channels]
    assert sorted(zip(rows.channel, rows.start, strict=True)) == sorted(
        itertools.product(names, GEO_STARTS)
    )
    for row, start in enumerate(rows.start):
        assert rows.geos[row] == ((0, 2), (1,))[GEO_STARTS.index(start) % 2]
    assert np.all(np.isfinite(rows.sigma)) and np.all(rows.sigma > 0.0)
    c = 1
    normals = np.random.default_rng((110, 2, seed, c + 1)).standard_normal((3, 104))
    dark = world.spend.copy()
    for j, start in enumerate(GEO_STARTS):
        dark[list(dark_geos(3, j)), start - 1 : start - 1 + TEST, c] = 0.0
    with jax.enable_x64(True):
        kpi = world.base + np.asarray(world.lift(dark))
    kpi = kpi + NOISE * np.sqrt(world.population)[:, None] * normals
    for j, start in enumerate(GEO_STARTS):
        ours = kpi[list(dark_geos(3, j))].sum(axis=0)
        theirs = kpi[list(dark_geos(3, j + 1))].sum(axis=0)
        reading = sloped_reading(ours, theirs, start - 1)
        held = [
            i
            for i, (n, s) in enumerate(zip(rows.channel, rows.start, strict=True))
            if (n, s) == (names[c], start)
        ]
        assert rows.delta_y[held[0]] == pytest.approx(reading, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("name", list(SIZES))
def test_on_the_familys_worlds_a_geo_tests_error_over_its_standard_error_has_a_spread_of_1(name):
    """On each environment's first 30 pilot worlds: a reading less the true change, over its
    standard error, has a standard deviation of 1 within four standard errors, on three geos as on
    fifty."""
    z = []
    for seed in FEW_GEOS.pilots[name][:30]:
        world = FEW_GEOS.world(name, seed).history
        rows = geo_rung(world, 10, seed, 4)
        names = [c.name for c in world.mix.channels]
        for row, (channel, start) in enumerate(zip(rows.channel, rows.start, strict=True)):
            true = _true_change(world, names.index(channel), GEO_STARTS.index(start))
            z.append((rows.delta_y[row] - true) / rows.sigma[row])
    z = np.array(z)
    assert z.size == 360
    assert abs(float(np.std(z, ddof=1)) - 1.0) <= 4.0 / math.sqrt(2 * (z.size - 1))


def test_the_quarter_at_the_status_quo_runs_on_from_the_familys_stream_at_the_worlds_noise():
    name = "geos-50"
    seed = FEW_GEOS.pilots[name][8]
    world = FEW_GEOS.world(name, seed)
    normals = rng(10, Role.CONTINUATION, seed).standard_normal((50, PLANNED))
    size = NOISY * np.sqrt(world.history.population)
    np.testing.assert_allclose(world.shadow.noise, (size[:, None] * normals).T, rtol=1e-15)
    np.testing.assert_array_equal(FEW_GEOS.truth(world).target, world.shadow.sales.sum(axis=1))


def test_every_world_exports_reads_back_and_scores_the_oracles_plan_at_nought(tmp_path):
    for e, name in enumerate(SIZES):
        seed = FEW_GEOS.scored[name][e]
        world = FEW_GEOS.world(name, seed)
        best = FEW_GEOS.truth(world).best
        geos = SIZES[name].count
        for k in (0, 4):
            observation = FEW_GEOS.observe(world, k)
            assert observation.sales.shape == (104, geos)
            assert observation.spend.shape == (104, geos, 3)
            assert observation.national == ("tv",) and observation.controls == {}
            assert observation.lower.shape == (2 * geos + 1,)
            assert observation.population is not None
            assert observation.population.sum() == pytest.approx(1.0, rel=1e-15)
            assert len(observation.lift.channel) == 3 * k
            back = read(export(observation, tmp_path / name / str(k)))
            assert back.digest() == observation.digest()
            np.testing.assert_array_equal(back.sales, world.history.kpi.T)
            assert back.lift.geos == observation.lift.geos and back.national == ("tv",)
            record = {
                "digest": observation.digest(),
                "version": VERSION,
                "error": None,
                "weekly": best.weekly.tolist(),
            }
            scores = score_world(FEW_GEOS, name, seed, k, {"oracle": record})
            assert (scores.family, scores.environment, scores.index, scores.k) == (10, name, e, k)
            assert scores.channels == 2 * geos + 1
            assert set(scores.returns) == {"search", "social", "tv"}
            assert abs(scores.arms["oracle"].regret) <= TIE
            for arm in scores.arms.values():
                assert arm.regret >= -TIE and arm.realised is None and arm.failure is None
            assert scores.arms["status quo"].regret > TIE
            again = WorldRecord.from_json(json.loads(json.dumps(scores.as_json())))
            assert again.arms["oracle"].regret == scores.arms["oracle"].regret

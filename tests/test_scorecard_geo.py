"""Family 8, geo worlds drawn by chc.geo_world: each world its seed alone, drawn from the family's
blocks at double precision whatever the caller's setting; the policy raises the spend with the
season by exactly its factor; every cell's worth is what chc's world lifts over the quarter, so the
oracle, exact on concave cells, is checked against chc's own world, a dense grid on two geos and a
search written apart from it; the geo tests read by difference-in-differences, exact on a world that
holds still outside them, their standard error read in time from placebo readings, the readings
and the standard errors nesting under common random numbers;
the quarter at the status quo and the returns are chc's world run on; and every world exports,
reads back and scores, the oracle's own plan at nought."""

import dataclasses
import itertools
import json
import math

import jax
import numpy as np
import pytest
from chc.geo_world import GeoMediaWorld
from chc.response import Hill
from scipy.optimize import minimize

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.lift_calibration import COOLDOWN, PRE, TEST
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard import truth
from causaldyn_bench.scorecard.continuation import geo_shadow, lifted
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.geo import FAMILY, GEO, POLICIES, mix, window
from causaldyn_bench.scorecard.ladder import GEO_STARTS, dark_geos, geo_rung, placebo_error
from causaldyn_bench.scorecard.observe import VERSION, export, read
from causaldyn_bench.scorecard.returns import drawn_returns, true_geo_returns
from causaldyn_bench.scorecard.scoring import WorldRecord, score_world
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored
from causaldyn_bench.scorecard.truth import allotted, geo_cells, geo_quarter, geo_spend, oracle

LENGTH = 8  # the kernel's
HORIZON = PLANNED + LENGTH - 1


def _world(seed: int, policy: float = 0.0, **change) -> GeoMediaWorld:
    """The family's mix for ``seed`` with ``change``, drawn at double precision."""
    generator = dataclasses.replace(mix(policy, seed, change.pop("geos", 10)), **change)
    with jax.enable_x64(True):
        return generator.draw(seed)


def test_the_family_draws_its_two_environments_from_its_own_blocks():
    assert number(GEO) == FAMILY == 8 and tuple(GEO.pilots) == tuple(GEO.scored)
    assert dict(POLICIES) == {"policy-0": 0.0, "policy-0.5": 0.5}
    for e, name in enumerate(POLICIES):
        assert name == f"policy-{POLICIES[name]:g}"
        assert GEO.pilots[name] == pilot(8, e) and GEO.scored[name] == scored(8, e)
    assert all(label.endswith(".") for label in GEO.labels)
    with pytest.raises(ValueError, match="no environment 'policy-1'"):
        GEO.world("policy-1", scored(8, 0)[0])


def test_a_world_is_its_seed_alone_at_double_precision_whatever_the_setting():
    name = "policy-0.5"
    seed = GEO.scored[name][1]
    first, again, other = (GEO.world(name, s) for s in (seed, seed, seed + 1))
    with jax.enable_x64(True):
        wide = GEO.world(name, seed)
    assert first.history.media.dtype == np.float64
    for k in (0, 4):
        stamp = GEO.observe(first, k).digest()
        assert stamp == GEO.observe(again, k).digest() == GEO.observe(wide, k).digest()
        assert stamp != GEO.observe(other, k).digest()
    ours, theirs = GEO.truth(first), GEO.truth(again)
    assert ours.best.worth == theirs.best.worth
    np.testing.assert_array_equal(ours.best.weekly, theirs.best.weekly)
    np.testing.assert_array_equal(ours.target, theirs.target)
    others = [GEO.observe(GEO.world(n, seed), 0).digest() for n in POLICIES]
    assert len(set(others)) == 2


def test_each_channels_spread_is_drawn_from_the_familys_stream():
    seed = GEO.pilots["policy-0"][3]
    spreads = [c.spread for c in GEO.world("policy-0", seed).history.mix.channels]
    expected = rng(8, Role.PARAMETERS, seed).uniform(0.2, 0.6, 3)
    np.testing.assert_array_equal(spreads, expected)
    drawn_spreads = np.array(
        [[c.spread for c in mix(0.0, s).channels] for s in GEO.pilots["policy-0"]]
    )
    assert drawn_spreads.min() >= 0.2 and drawn_spreads.max() <= 0.6
    assert drawn_spreads.std() > 0.1  # the uniform's 0.115 over a hundred draws


def test_the_policy_raises_the_spend_with_the_season_by_its_factor_and_moves_nothing_else():
    """At one seed both policies read the same variates, so the spend at policy 0.5 over the
    spend at 0 is 1 + 0.5 sin(2 pi t / 52) in every geo, week and channel (Maxima), the realised
    policy, that ratio's slope on the season, 0.5 to rounding; the populations, effects, bases and
    noise stay; the media move."""
    for seed in GEO.pilots["policy-0.5"][:3]:
        calm, raised = _world(seed, 0.0), _world(seed, 0.5)
        season = np.sin(2.0 * np.pi * np.arange(104) / 52.0)
        ratio = raised.spend / calm.spend
        np.testing.assert_allclose(
            ratio, np.broadcast_to((1.0 + 0.5 * season)[None, :, None], ratio.shape), rtol=1e-14
        )
        realised = float(np.sum((ratio - 1.0) * season[None, :, None]))
        realised /= ratio.shape[0] * ratio.shape[2] * float(season @ season)
        assert realised == pytest.approx(0.5, rel=1e-13, abs=0.0)
        for field in ("population", "effect", "base", "noise", "drift"):
            assert np.array_equal(getattr(raised, field), getattr(calm, field)), field
        assert not np.allclose(raised.media, calm.media)
        # the spend reads the season the KPI's base moves with, at policy 0.5 alone
        rises = [np.corrcoef(w.spend.sum(axis=(0, 2)), season)[0, 1] for w in (calm, raised)]
        assert rises[1] > 0.8 and abs(rises[0]) < 0.4


def test_national_media_is_allotted_by_population_and_its_cell_is_its_spend_summed():
    world = _world(GEO.pilots["policy-0"][1])
    per_head = world.spend[:, :, 2] / world.population[:, None]
    np.testing.assert_allclose(per_head, np.broadcast_to(per_head[:1], per_head.shape), rtol=1e-14)
    spend = geo_spend(world)
    assert spend.shape == (104, 21)
    for g, c in itertools.product(range(10), range(2)):
        np.testing.assert_array_equal(spend[:, 2 * g + c], world.spend[g, :, c])
    np.testing.assert_array_equal(spend[:, 20], world.spend[:, :, 2].sum(axis=0))
    quarter = geo_quarter(world)
    np.testing.assert_array_equal(quarter.status_quo, spend[-52:].mean(axis=0))
    np.testing.assert_array_equal(quarter.lower, 0.5 * quarter.status_quo)
    np.testing.assert_array_equal(quarter.upper, 2.0 * quarter.status_quo)
    assert quarter.budget == PLANNED * float(quarter.status_quo.sum())
    panel = allotted(world, quarter.status_quo)
    np.testing.assert_array_equal(panel[:, :2].ravel(), quarter.status_quo[:20])
    np.testing.assert_allclose(panel[:, 2], world.shares * quarter.status_quo[20], rtol=1e-15)
    with pytest.raises(ValueError, match="cells are not"):
        allotted(world, quarter.status_quo[:20])


def test_the_equal_split_gives_each_channel_a_third_a_per_geo_ones_split_by_population():
    """The equal split is the channels', not the cells': each of the three gets a third of the
    budget, a per-geo channel's third split across the geos by population and TV's on its one
    cell, then moved into the box; the score's equal split is it. The status quo needs no such
    rule: every cell is at its own mean, TV's its spend over the geos summed."""
    seed = GEO.pilots["policy-0"][1]
    world = GEO.world("policy-0", seed)
    history = world.history
    wide = geo_quarter(history, box=(0.0, 1e3))
    rate = wide.budget / PLANNED
    weekly = wide.equal_split()
    assert wide.feasible(weekly)
    local = weekly[:20].reshape(10, 2)
    np.testing.assert_allclose(local, np.outer(history.shares, [rate / 3.0] * 2), rtol=1e-9)
    assert weekly[20] == pytest.approx(rate / 3.0, rel=1e-9, abs=0.0)
    quarter = geo_quarter(history)
    split = quarter.equal_split()
    assert quarter.feasible(split)
    np.testing.assert_allclose(split, quarter.project(weekly), rtol=1e-9)
    assert quarter.status_quo[20] == pytest.approx(
        float(history.spend[:, -52:, 2].sum(axis=0).mean()), rel=1e-14, abs=0.0
    )
    on = GEO.truth(world)
    regret = (on.best.worth - truth.worth(on.cells, split)) / quarter.budget
    scores = score_world(GEO, "policy-0", seed, 0, {}, pilot=True)
    assert scores.arms["equal split"].regret == pytest.approx(regret, rel=1e-12, abs=0.0)


def _plans(quarter: Quarter, count: int, seed: int) -> list[np.ndarray]:
    draw = np.random.default_rng(seed)
    return [quarter.status_quo, quarter.lower, quarter.upper] + [
        quarter.project(draw.uniform(quarter.lower, quarter.upper)) for _ in range(count)
    ]


@pytest.mark.parametrize(
    "change",
    [{}, {"policy": 0.5}, {"geos": 2}, {"spillover": 0.4}],
    ids=["policy-0", "policy-0.5", "two geos", "spillover"],
)
def test_a_cells_worth_is_what_chcs_world_lifts_over_the_quarter_and_its_tail(change):
    """Every geo's KPI over the quarter and the kernel's tail, spent the plan and then nothing,
    from chc's own world run on: summed, the cells' worth to rounding, the history's carryover
    in. A world whose media spill reads it into the spending geo's cells."""
    world = _world(GEO.pilots["policy-0"][2], **change)
    quarter = geo_quarter(world)
    on = geo_cells(world, quarter)
    assert len(on) == world.mix.geos * 2 + 1 and all(cell.concave for cell in on)
    for weekly in _plans(quarter, 4, 1):
        kpi = lifted(world, allotted(world, weekly), HORIZON)
        assert truth.worth(on, weekly) == pytest.approx(float(kpi.sum()), rel=1e-12, abs=0.0)


def test_a_drifting_world_and_a_curve_the_cells_do_not_read_are_refused():
    world = _world(GEO.pilots["policy-0"][2], drift=0.01)
    with pytest.raises(ValueError, match="drift"):
        geo_spend(world)
    with pytest.raises(ValueError, match="drift"):
        geo_shadow(world, 8, 1)
    with pytest.raises(ValueError, match="Tanh and Hill of slope 1"):
        truth.geo_curve(Hill(1.0, 2.0), 1e5)


def _apart(on, quarter: Quarter, starts: list[np.ndarray]) -> float:
    """The best worth SLSQP reaches from ``starts`` with finite-difference gradients, each cell's
    spend in units of its status quo so the differences' step suits it: a search that shares
    nothing with the oracle's."""
    unit = quarter.status_quo
    rate = quarter.budget / PLANNED
    best = -math.inf
    for start in starts:
        solution = minimize(
            lambda u: -truth.worth(on, u * unit) / quarter.budget,
            start / unit,
            method="SLSQP",
            bounds=list(zip(quarter.lower / unit, quarter.upper / unit, strict=True)),
            constraints={"type": "eq", "fun": lambda u: float(u @ unit) / rate - 1.0},
            options={"ftol": 1e-15, "maxiter": 1000},
        )
        weekly = solution.x * unit
        if quarter.feasible(weekly, tolerance=1e-9):
            best = max(best, truth.worth(on, weekly))
    return best


@pytest.mark.parametrize("name", list(POLICIES))
def test_on_ten_geos_the_exact_plan_meets_its_conditions_and_no_search_beats_it(name):
    world = GEO.world(name, GEO.pilots[name][4])
    found = GEO.truth(world)
    quarter, plan, on = found.quarter, found.best, found.cells
    assert quarter.feasible(plan.weekly) and math.isfinite(plan.price)
    assert plan.worth == pytest.approx(truth.worth(on, plan.weekly), rel=1e-15, abs=0.0)
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
    apart = _apart(on, quarter, _plans(quarter, 3, 7))
    assert plan.worth * (1 - 1e-8) <= apart <= plan.worth + TIE * quarter.budget


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


@pytest.mark.parametrize("policy", [0.0, 0.5])
def test_on_two_geos_no_point_of_a_dense_grid_beats_the_oracle_and_its_best_refines_to_it(policy):
    """Two geos: four own cells and TV's, a grid of 41 points a side over four of them, 2.8
    million plans; a search from its best point, apart from the oracle, lands on the oracle's
    worth and not above it."""
    world = _world(GEO.pilots["policy-0"][5], policy, geos=2)
    quarter = geo_quarter(world)
    on = geo_cells(world, quarter)
    assert len(on) == 5
    plan = oracle(on, quarter)
    grid, at = _dense(on, quarter, 41)
    assert grid <= plan.worth + TIE * quarter.budget
    assert grid >= truth.worth(on, quarter.status_quo)
    refined = _apart(on, quarter, [quarter.project(at)])
    assert plan.worth * (1 - 1e-10) <= refined <= plan.worth + TIE * quarter.budget


def _true_change(world: GeoMediaWorld, c: int, j: int) -> float:
    """What the ``j``-th test of channel ``c`` costs its tested geos over its readout, a dark
    week, from chc's world: its lift with the geos dark less its lift as spent."""
    start, geos = GEO_STARTS[j], list(dark_geos(world.mix.geos, j))
    dark = world.spend.copy()
    dark[geos, start - 1 : start - 1 + TEST, c] = 0.0
    with jax.enable_x64(True):
        gap = np.asarray(world.lift(dark)) - np.asarray(world.lift(world.spend))
    return float(gap[geos, start - 1 : start - 1 + TEST + COOLDOWN].sum()) / TEST


def test_a_geo_test_takes_alternate_geos_dark_and_reads_their_planned_spend():
    world = GEO.world("policy-0", GEO.pilots["policy-0"][6]).history
    rows = geo_rung(world, 8, GEO.pilots["policy-0"][6], 4)
    assert rows.geos is not None
    assert len(rows.channel) == 12 and rows.dropped == 0
    assert dark_geos(10, 0) == dark_geos(10, 2) == (0, 2, 4, 6, 8)
    assert dark_geos(10, 1) == dark_geos(10, 3) == (1, 3, 5, 7, 9)
    assert dark_geos(3, 0) == (0, 2) and dark_geos(3, 1) == (1,)
    names = [c.name for c in world.mix.channels]
    for row, (channel, start) in enumerate(zip(rows.channel, rows.start, strict=True)):
        j = GEO_STARTS.index(start)
        assert rows.geos[row] == dark_geos(10, j)
        dark = world.spend[list(rows.geos[row]), start - 1 : start - 1 + TEST, names.index(channel)]
        assert rows.x[row] == pytest.approx(float(dark.sum(axis=0).mean()), rel=1e-14, abs=0.0)
        assert rows.delta_x[row] == -rows.x[row]
    assert set(rows.start) <= set(GEO_STARTS)
    assert GEO_STARTS[0] - PRE >= 1 + LENGTH  # every readout past the adstock's first weeks
    assert GEO_STARTS[-1] + TEST + COOLDOWN - 1 <= 104
    # each test's carryover is gone before the next one's readout opens
    assert all(b - PRE > a + TEST - 1 + LENGTH - 1 for a, b in itertools.pairwise(GEO_STARTS))


def test_on_a_world_that_holds_still_outside_its_tests_the_reading_is_the_true_change():
    """Without the spend's jitter, the season and the noise, each geo's KPI is flat but for its
    test, so the difference-in-differences reads the change chc's world lifts, every test kept."""
    seed = GEO.pilots["policy-0"][7]
    world = _world(seed, spend_noise=0.0, season=0.0, noise=0.0)
    rows = geo_rung(world, 8, seed, 4)
    assert rows.dropped == 0 and len(rows.channel) == 12
    names = [c.name for c in world.mix.channels]
    for row, (channel, start) in enumerate(zip(rows.channel, rows.start, strict=True)):
        true = _true_change(world, names.index(channel), GEO_STARTS.index(start))
        assert rows.delta_y[row] == pytest.approx(true, rel=1e-9, abs=0.0)


def test_a_geo_test_whose_kpi_rose_is_exported():
    """Dropping a test on its reading truncates on the outcome: every test of the rung is a row,
    whichever way its KPI moved, and nothing is dropped."""
    seed = GEO.pilots["policy-0"][3]
    world = GEO.world("policy-0", seed).history
    rows = geo_rung(world, 8, seed, 4)
    assert rows.dropped == 0 and len(rows.channel) == 12
    assert int(np.sum(rows.delta_y >= 0.0)) == 8
    names = [c.name for c in world.mix.channels]
    assert sorted(zip(rows.channel, rows.start, strict=True)) == sorted(
        itertools.product(names, GEO_STARTS)
    )


def test_each_channels_tests_run_in_a_universe_of_their_own():
    """A channel's universe is the world's base and its media with every test of the rung dark,
    and noise of the world's size from the family's tests' stream, ``(108, 2, seed, c + 1)``; each
    test reads its geos against the others, scaled over the four weeks before it."""
    seed = GEO.pilots["policy-0.5"][3]
    world = GEO.world("policy-0.5", seed).history
    rows = geo_rung(world, 8, seed, 4)
    for c, channel in enumerate(world.mix.channels):
        normals = np.random.default_rng((108, 2, seed, c + 1)).standard_normal((10, 104))
        dark = world.spend.copy()
        for j, start in enumerate(GEO_STARTS):
            dark[list(dark_geos(10, j)), start - 1 : start - 1 + TEST, c] = 0.0
        with jax.enable_x64(True):
            kpi = world.base + np.asarray(world.lift(dark))
        kpi = kpi + 230.0 * np.sqrt(world.population)[:, None] * normals
        for j, start in enumerate(GEO_STARTS):
            ours = kpi[list(dark_geos(10, j))].sum(axis=0)
            theirs = kpi[list(dark_geos(10, j + 1))].sum(axis=0)
            before = slice(start - 1 - PRE, start - 1)
            gap = ours - ours[before].mean() / theirs[before].mean() * theirs
            reading = gap[start - 1 : start - 1 + TEST + COOLDOWN].sum() / TEST
            held = [
                i
                for i, (n, s) in enumerate(zip(rows.channel, rows.start, strict=True))
                if n == channel.name and s == start
            ]
            assert len(held) == 1
            assert rows.delta_y[held[0]] == pytest.approx(reading, rel=1e-12, abs=0.0)


def test_on_worlds_still_but_for_their_noise_a_geo_tests_standard_error_is_its_groupings():
    """On worlds that hold still but for the noise, a tenth of the family's: the reading less the
    true change, over its noise's standard deviation, ``sigma_u sqrt(Q + Q^2 / P) / TEST`` with
    ``sigma_u`` the tested geos' noise beside the others' at the ratio, has a standard deviation of
    1 within four standard errors, and so has it over the test's standard error. The even and the
    odd geos' noises differ by their ratio, and a test's standard error follows its own: within a
    channel its log moves with the noise's with a slope of 1, where one weekly noise pooled over the
    channel's tests does not move at all."""
    z, read, noises, errors = [], [], [], []
    for seed in GEO.pilots["policy-0"][:40]:
        world = _world(seed, spend_noise=0.0, season=0.0, noise=23.0)
        quiet = dataclasses.replace(world, noise=np.zeros_like(world.noise))
        rows = geo_rung(world, 8, seed, 4)
        assert rows.dropped == 0
        sizes = 23.0**2 * world.population
        kpi = quiet.kpi
        names = [c.name for c in world.mix.channels]
        for c in range(3):
            spreads, sigmas = [], []
            for j, start in enumerate(GEO_STARTS):
                ours = list(dark_geos(10, j))
                theirs = [g for g in range(10) if g not in ours]
                pre = slice(start - 1 - PRE, start - 1)
                ratio = kpi[ours, pre].sum() / kpi[theirs, pre].sum()
                variance = float(sizes[ours].sum() + ratio**2 * sizes[theirs].sum())
                spread = math.sqrt(variance * (12 + 12**2 / PRE)) / TEST
                row = next(
                    i
                    for i, (n, s) in enumerate(zip(rows.channel, rows.start, strict=True))
                    if n == names[c] and s == start
                )
                error = rows.delta_y[row] - _true_change(world, c, j)
                z.append(error / spread)
                read.append(error / rows.sigma[row])
                spreads.append(math.log(spread))
                sigmas.append(math.log(rows.sigma[row]))
            noises += list(np.array(spreads) - np.mean(spreads))
            errors += list(np.array(sigmas) - np.mean(sigmas))
    z, read, noises, errors = np.array(z), np.array(read), np.array(noises), np.array(errors)
    assert z.size == read.size == 480
    for values in (z, read):
        assert abs(float(np.std(values, ddof=1)) - 1.0) <= 4.0 / math.sqrt(2 * (values.size - 1))
        assert abs(float(np.mean(values))) <= 4.0 / math.sqrt(values.size)
    assert float(np.std(noises)) > 0.2  # the groupings' noises do differ
    assert float(noises @ errors / (noises @ noises)) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("name", list(POLICIES))
def test_on_the_familys_worlds_a_geo_tests_error_over_its_standard_error_has_a_spread_of_1(name):
    """On the family's first 30 pilot worlds, whose spend jitters and whose season moves the base,
    at policy 0.5 the spend too: a reading less the true change, over its standard error, has a
    standard deviation of 1 within four standard errors. The noise alone misses the KPI's movement
    that is not proportional across the groups; the standard error, read in time, holds it."""
    z = []
    for seed in GEO.pilots[name][:30]:
        world = GEO.world(name, seed).history
        rows = geo_rung(world, 8, seed, 4)
        names = [c.name for c in world.mix.channels]
        for row, (channel, start) in enumerate(zip(rows.channel, rows.start, strict=True)):
            true = _true_change(world, names.index(channel), GEO_STARTS.index(start))
            z.append((rows.delta_y[row] - true) / rows.sigma[row])
    z = np.array(z)
    assert z.size == 360
    assert abs(float(np.std(z, ddof=1)) - 1.0) <= 4.0 / math.sqrt(2 * (z.size - 1))


def _placebo_kpi() -> np.ndarray:
    return 100.0 + np.random.default_rng(5).standard_normal((4, 104))


def test_a_placebo_reading_reads_no_week_of_the_tests_own_window_nor_the_adstocks_first():
    """A test's standard error is read from the windows clear of its own four weeks before and
    twelve of readout, and of the adstock's first ``length - 1`` weeks: a spike in any of those
    weeks leaves it as it was, and one in a week next to them moves it."""
    kpi, treated, first, length = _placebo_kpi(), (1, 3), 50, LENGTH
    after = TEST + COOLDOWN
    alone = placebo_error(kpi, treated, first, length)

    def spiked(week: int) -> float:
        changed = kpi.copy()
        changed[1, week] += 50.0
        return placebo_error(changed, treated, first, length)

    for week in (0, length - 2, first - PRE, first, first + after - 1):
        assert spiked(week) == alone, week
    for week in (length - 1, first - PRE - 1, first + after, 103):
        assert spiked(week) != alone, week


def test_a_tests_standard_error_widens_its_placebo_readings_by_a_ts_spread_at_their_count():
    """Over independent weeks two placebo readings ``h`` weeks apart correlate by ``rho(h)``,
    Maxima's exact values for four weeks before and twelve of readout and 0 from 16; the standard
    error is the readings' root mean square times ``sqrt(nu / (nu - 2))``,
    ``nu = n^2 / sum rho(|f - f'|)^2``. A universe with two effective readings or fewer is
    refused."""
    rho = np.array([1, 35 / 48, 11 / 24, 3 / 16, -1 / 12, -5 / 48, -1 / 8, -7 / 48, -1 / 6])
    rho = np.concatenate([rho, [-3 / 16, -5 / 24, -11 / 48, -1 / 4, -3 / 16, -1 / 8, -1 / 16, 0]])
    kpi, treated, first = _placebo_kpi(), (0, 2), 30
    ours, theirs = kpi[[0, 2]].sum(axis=0), kpi[[1, 3]].sum(axis=0)
    after = TEST + COOLDOWN
    firsts = np.array(
        [
            f
            for f in range(LENGTH - 1 + PRE, 104 - after + 1)
            if f + after <= first - PRE or f - PRE >= first + after
        ]
    )
    readings = []
    for f in firsts:
        gap = ours - ours[f - PRE : f].mean() / theirs[f - PRE : f].mean() * theirs
        readings.append(gap[f : f + after].sum() / TEST)
    lags = np.minimum(np.abs(firsts[:, None] - firsts[None, :]), 16)
    nu = firsts.size**2 / float(np.sum(rho[lags] ** 2))
    assert firsts.size == 51 and 17.0 < nu < 18.0
    expected = math.sqrt(float(np.mean(np.square(readings))) * nu / (nu - 2.0))
    assert placebo_error(kpi, treated, first, LENGTH) == pytest.approx(expected, rel=1e-12, abs=0.0)
    # one placebo window, at week 11, and none at all
    with pytest.raises(ValueError, match="too few"):
        placebo_error(kpi[:, :40], treated, 27, LENGTH)
    with pytest.raises(ValueError, match="too few"):
        placebo_error(kpi[:, :40], treated, 20, LENGTH)


def test_the_rungs_nest_under_common_random_numbers():
    seed = GEO.pilots["policy-0.5"][2]
    world = GEO.world("policy-0.5", seed).history
    top = geo_rung(world, 8, seed, 4)
    assert top.geos is not None
    for k in range(4):
        rows = geo_rung(world, 8, seed, k)
        assert rows.geos is not None
        assert set(rows.start) <= set(GEO_STARTS[4 - k :])
        for row, (channel, start) in enumerate(zip(rows.channel, rows.start, strict=True)):
            same = [
                i
                for i, (n, s) in enumerate(zip(top.channel, top.start, strict=True))
                if n == channel and s == start
            ]
            assert len(same) == 1
            for field in ("x", "delta_x", "delta_y", "sigma"):
                assert getattr(rows, field)[row] == getattr(top, field)[same[0]], field
            assert rows.geos[row] == top.geos[same[0]]
        kept = [s for s in top.start if s in GEO_STARTS[4 - k :]]
        assert len(rows.channel) == len(kept)
    nought = geo_rung(world, 8, seed, 0)
    assert nought.channel == () and nought.geos == () and nought.dropped == 0


def test_a_rung_off_the_ladder_one_geo_and_a_short_world_are_refused():
    seed = GEO.pilots["policy-0"][0]
    world = GEO.world("policy-0", seed).history
    with pytest.raises(ValueError, match="not 5"):
        geo_rung(world, 8, seed, 5)
    with pytest.raises(ValueError, match="has 1"):
        geo_rung(_world(seed, geos=1), 8, seed, 4)
    with pytest.raises(ValueError, match="run past"):
        geo_rung(_world(seed, weeks=102), 8, seed, 4)


def test_the_quarter_at_the_status_quo_is_chcs_world_run_on():
    """The base and its season are chc's own over the weeks after the history, read off a world
    drawn longer from the same seed; the media are every cell's return at the status quo; the
    noise is the family's continuation stream at the world's size; the target is the market's."""
    name = "policy-0.5"
    seed = GEO.pilots[name][8]
    world = GEO.world(name, seed)
    shadow, history = world.shadow, world.history
    np.testing.assert_array_equal(shadow.week, np.arange(105, 118))
    longer = _world(seed, 0.5, weeks=104 + PLANNED)
    np.testing.assert_array_equal(longer.population, history.population)
    np.testing.assert_allclose(shadow.base, longer.base[:, 104:].T, rtol=1e-15)
    quarter = geo_quarter(history)
    on = geo_cells(history, quarter)
    returns = sum(
        cell.returns(float(w))[:PLANNED] for cell, w in zip(on, quarter.status_quo, strict=True)
    )
    np.testing.assert_allclose(shadow.media.sum(axis=1), returns, rtol=1e-12)
    normals = rng(8, Role.CONTINUATION, seed).standard_normal((10, PLANNED))
    size = 230.0 * np.sqrt(history.population)
    np.testing.assert_allclose(shadow.noise, (size[:, None] * normals).T, rtol=1e-15)
    np.testing.assert_array_equal(shadow.sales, shadow.base + shadow.media + shadow.noise)
    found = GEO.truth(world)
    np.testing.assert_array_equal(found.target, shadow.sales.sum(axis=1))
    assert found.scale == pytest.approx(float(history.kpi.sum(axis=0).mean()), rel=1e-15)


def _returned(world: GeoMediaWorld, spend: np.ndarray) -> float:
    """What chc's world lifts over the history and the kernel's tail, every geo and channel, at
    ``spend`` over the history, nothing spent after it."""
    spend = np.concatenate([spend, np.zeros((world.mix.geos, LENGTH - 1, 3))], axis=1)
    longer = dataclasses.replace(world, spend=spend, drift=np.ones((spend.shape[1], 3)))
    with jax.enable_x64(True):
        return float(np.asarray(longer.lift(spend)).sum())


@pytest.mark.parametrize("change", [{}, {"spillover": 0.4}], ids=["drawn", "spillover"])
def test_a_channels_returns_are_the_markets_read_off_chcs_world(change):
    """The ROI: the whole market's return less its return with the window's spend of the channel
    removed, over that spend; the mROI: the return's derivative along it, by a central difference,
    over that spend; both through chc's own world, its carryover past the history counted."""
    world = _world(GEO.pilots["policy-0"][9], **change)
    found = true_geo_returns(world, window(world))
    assert found.window == (53, 104)
    inside = np.zeros(104, dtype=bool)
    inside[52:] = True
    every = _returned(world, world.spend)
    for c in range(3):
        spent = float(world.spend[:, inside, c].sum())
        without = world.spend.copy()
        without[:, inside, c] = 0.0
        roi = (every - _returned(world, without)) / spent
        assert found.roi[c] == pytest.approx(roi, rel=1e-10, abs=0.0)
        step = 1e-4
        along = np.zeros_like(world.spend)
        along[:, inside, c] = world.spend[:, inside, c]
        up = _returned(world, world.spend + step * along)
        down = _returned(world, world.spend - step * along)
        assert found.marginal[c] == pytest.approx((up - down) / (2 * step) / spent, rel=1e-6)


def test_every_world_exports_reads_back_and_scores_the_oracles_plan_at_nought(tmp_path):
    for e, name in enumerate(POLICIES):
        seed = GEO.scored[name][e]
        world = GEO.world(name, seed)
        best = GEO.truth(world).best
        for k in (0, 4):
            observation = GEO.observe(world, k)
            assert observation.sales.shape == (104, 10)
            assert observation.spend.shape == (104, 10, 3)
            assert observation.national == ("tv",) and observation.controls == {}
            assert observation.lower.shape == (21,) and observation.kernel_length == LENGTH
            assert observation.population is not None
            assert observation.population.sum() == pytest.approx(1.0, rel=1e-15)
            back = read(export(observation, tmp_path / name / str(k)))
            assert back.digest() == observation.digest()
            np.testing.assert_array_equal(back.sales, world.history.kpi.T)
            np.testing.assert_array_equal(back.spend, world.history.spend.transpose(1, 0, 2))
            assert back.lift.geos == observation.lift.geos and back.national == ("tv",)
            np.testing.assert_array_equal(back.lift.delta_y, observation.lift.delta_y)
            record = {
                "digest": observation.digest(),
                "version": VERSION,
                "error": None,
                "weekly": best.weekly.tolist(),
            }
            scores = score_world(GEO, name, seed, k, {"oracle": record})
            assert (scores.family, scores.environment, scores.index, scores.k) == (8, name, e, k)
            assert scores.channels == 21 and set(scores.returns) == {"search", "social", "tv"}
            assert abs(scores.arms["oracle"].regret) <= TIE
            for arm in scores.arms.values():
                assert arm.regret >= -TIE and arm.realised is None and arm.failure is None
            assert scores.arms["status quo"].regret > TIE
            again = WorldRecord.from_json(json.loads(json.dumps(scores.as_json())))
            assert again.arms["oracle"].regret == scores.arms["oracle"].regret


def test_a_national_response_on_a_geo_export_is_read_against_the_markets_returns():
    """TV's spend a head is the same in every geo, so its geos' lifts are one response of the
    market's spend: read on the spend summed over the geos, its ROI and mROI are the world's."""
    world = GEO.world("policy-0", GEO.pilots["policy-0"][10])
    history = world.history
    found = GEO.truth(world).returns
    population = float(history.population.sum())
    response = {
        "kernel": "GeometricAdstock",
        "length": LENGTH,
        "normalized": True,
        "curve": "Tanh",
        "parameters": {
            c.name: {
                "retention": [c.retention],
                "scale": [float(c.curve.scale) * population],
                "coefficient": [float(history.population @ history.effect[:, i])],
            }
            for i, c in enumerate(history.mix.channels)
        },
        "decomposition": {"total": {}, "held": [], "weekly": {}},
    }
    assert history.mix.channels[2].national
    returns = drawn_returns(response, GEO.observe(world, 0))
    assert returns.roi[0, 2] == pytest.approx(found.roi[2], rel=1e-12, abs=0.0)
    assert returns.marginal[0, 2] == pytest.approx(found.marginal[2], rel=1e-12, abs=0.0)
    assert returns.roi[0, 0] != pytest.approx(found.roi[0], rel=1e-3)  # search is not one curve


def test_a_drawn_world_holds_what_the_family_states():
    world = GEO.world("policy-0", GEO.pilots["policy-0"][11]).history
    names = [(c.name, type(c.curve).__name__, c.national) for c in world.mix.channels]
    assert names == [("search", "Tanh", False), ("social", "Hill", False), ("tv", "Tanh", True)]
    social = world.mix.channels[1].curve
    assert isinstance(social, Hill) and float(social.slope) == 1.0
    assert world.mix.geos == 10 and world.spend.shape == (10, 104, 3)
    assert world.mix.population == 1e5 and world.mix.policy == 0.0

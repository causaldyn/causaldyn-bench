"""A schedule scored against a world's truth: each cell's reach of a unit in each quarter week
alone, the kernel from that week, its rows the cell's reach, so a schedule held flat is worth its
plan; the best schedule in the box at the budget, the closed form where nothing carries and the
exponential curve's effect moves from week to week (``scripts/schedule_kkt.mac``), flat where the
effect holds still, beaten by no schedule on the families' worlds and certified by its Frank-Wolfe
gap; and a schedule's regret against it, on the expected and the realised path, a channel's
schedule allotted to its cells week by week and one off the budget moved into the box."""

import math

import numpy as np
import pytest

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.endogenous_mmm import CURVES, LN2, LN3
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard import schedule, truth
from causaldyn_bench.scorecard.drift import DRIFT
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.s_curves import S_CURVES
from causaldyn_bench.scorecard.scoring import built_in
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from causaldyn_bench.scorecard.twins import TWINS

WORLDS = {
    "track m2": (TRACK_M2, "drawn", 905),
    "geo": (GEO, "policy-0", GEO.pilots["policy-0"][1]),
    "cpm": (DRIFT, "cpm", DRIFT.pilots["cpm"][0]),
}


@pytest.fixture(scope="module", params=sorted(WORLDS))
def truths(request):
    family, environment, seed = WORLDS[request.param]
    return family.truth(family.world(environment, seed))


def test_a_cells_unit_is_its_kernel_from_each_quarter_week_and_its_rows_its_reach(truths):
    """Each column holds the kernel from its week, times what a euro buys that week where the world
    prices it, so each column is the first's moved down by its week; nothing reaches a week before
    it or past the kernel."""
    for cells in (truths.cells, truths.realised or ()):
        for cell in cells:
            unit = cell.unit
            assert unit is not None and unit.shape == (cell.reach.size, PLANNED)
            np.testing.assert_allclose(unit.sum(axis=1), cell.reach, rtol=1e-13, atol=0.0)
            length = cell.reach.size - PLANNED + 1
            kernel = unit[:length, 0] / unit[:length, 0].sum()
            for s in range(PLANNED):
                column = np.zeros(cell.reach.size)
                column[s : s + length] = kernel * unit[:, s].sum()
                np.testing.assert_allclose(unit[:, s], column, rtol=1e-14, atol=0.0)


def test_a_schedule_held_flat_is_worth_its_plan(truths):
    quarter = truths.quarter
    for plan in (truths.best.weekly, quarter.status_quo, quarter.equal_split()):
        flat = np.tile(plan, (PLANNED, 1))
        assert schedule.worth(truths.cells, flat) == pytest.approx(
            truth.worth(truths.cells, plan), rel=1e-13, abs=0.0
        )


def _exponential(effects: np.ndarray, rates: tuple[float, ...]) -> tuple[truth.Cell, ...]:
    """Cells without carryover on the exponential curve, ``1 - exp(-k a)`` at the rate ``k``, the
    effect a path over the quarter: each week's spend reaches that week alone."""
    curve = CURVES["exponential"]
    nothing, unit = np.zeros(PLANNED), np.eye(PLANNED)
    return tuple(
        truth.Cell(nothing, np.ones(PLANNED), curve, k * LN3 / LN2, effect, unit)
        for k, effect in zip(rates, effects, strict=True)
    )


def _quarter(cells: int, budget: float) -> Quarter:
    return Quarter(
        budget=budget,
        lower=np.full(cells, 0.01),
        upper=np.full(cells, 100.0),
        status_quo=np.full(cells, budget / PLANNED / cells),
        history=np.zeros((10, cells)),
    )


def test_where_nothing_carries_the_best_schedule_is_the_closed_form():
    """At a price ``mu`` each week's slope ``e k exp(-k w)`` meets it, so ``w = log(e k / mu) / k``,
    and the budget sets ``log mu`` (Maxima, ``scripts/schedule_kkt.mac``); every spend lies inside
    its box here."""
    rates = (0.5, 0.8)
    effects = np.random.default_rng(20261107).uniform(2.0, 6.0, (2, PLANNED))
    budget = 160.0
    cells = _exponential(effects, rates)
    best = schedule.oracle(cells, _quarter(2, budget))
    k = np.array(rates)[None, :]
    logged = np.log(effects.T * k)
    log_mu = (float((logged / k).sum()) - budget) / float(PLANNED * (1.0 / k).sum())
    exact = (logged - log_mu) / k
    assert exact.min() > 0.01 and exact.max() < 100.0
    np.testing.assert_allclose(best.weekly, exact, rtol=1e-10, atol=0.0)
    assert best.price == pytest.approx(math.exp(log_mu), rel=1e-10, abs=0.0)
    assert best.worth == pytest.approx(schedule.worth(cells, exact), rel=1e-14, abs=0.0)
    assert 0.0 <= best.gap <= schedule.GAP * budget


def test_where_the_effect_holds_still_and_nothing_carries_the_best_schedule_is_the_best_plan():
    effects = np.array([np.full(PLANNED, 3.0), np.full(PLANNED, 5.0)])
    cells = _exponential(effects, (0.5, 0.8))
    quarter = _quarter(2, 160.0)
    best, plan = schedule.oracle(cells, quarter), truth.oracle(cells, quarter)
    np.testing.assert_allclose(
        best.weekly, np.tile(plan.weekly, (PLANNED, 1)), rtol=1e-10, atol=0.0
    )
    assert best.worth == pytest.approx(plan.worth, rel=1e-13, abs=0.0)


def test_on_the_families_worlds_no_schedule_beats_the_best_and_its_gap_is_certified(truths):
    """The worth is concave and lies below its tangent plane, so the gap bounds what any schedule
    returns over the best; the best returns at least the best plan, its weeks within their box meet
    its price and those at an end lean the way the end holds them, and 200 schedules drawn in the
    box at the budget, and every weekly plan the scorecard plays, fall short of it. The bound holds
    for every schedule, so no draw fails a sound oracle."""
    quarter = truths.quarter
    best = schedule.oracle(truths.cells, quarter)
    assert 0.0 <= best.gap <= schedule.GAP * quarter.budget
    assert best.weekly.sum() == pytest.approx(quarter.budget, rel=1e-12, abs=0.0)
    assert np.all(best.weekly >= quarter.lower) and np.all(best.weekly <= quarter.upper)
    assert best.worth >= truths.best.worth - TIE * quarter.budget
    slopes = schedule._slopes(truths.cells, schedule._units(truths.cells), best.weekly)
    low, high = np.tile(quarter.lower, (PLANNED, 1)), np.tile(quarter.upper, (PLANNED, 1))
    inside = (best.weekly > low * (1 + 1e-9)) & (best.weekly < high * (1 - 1e-9))
    assert inside.any()
    np.testing.assert_allclose(slopes[inside], best.price, rtol=1e-9, atol=0.0)
    assert np.all(slopes[best.weekly <= low] <= best.price * (1 + 1e-9))
    assert np.all(slopes[best.weekly >= high] >= best.price * (1 - 1e-9))
    draws = np.random.default_rng(7).uniform(0.0, 1.0, (200, PLANNED, quarter.lower.size))
    for draw in draws:
        drawn = schedule._project(quarter, low + draw * (high - low))
        assert schedule.worth(truths.cells, drawn) <= best.worth + best.gap
    for plan in (truths.best.weekly, quarter.status_quo, quarter.equal_split()):
        assert schedule.worth(truths.cells, np.tile(plan, (PLANNED, 1))) <= best.worth + best.gap


def test_a_schedule_is_scored_against_the_best_schedule(truths):
    """The best schedule scores nought; a weekly plan held flat scores its plan's regret plus what
    the best schedule returns over the best plan; a schedule off the budget is moved into the box
    and scored where it was moved to; and where the quarter realised another path, its regret
    there is against the best schedule in hindsight."""
    quarter = truths.quarter
    best = schedule.oracle(truths.cells, quarter)
    own = schedule.score_schedule(truths, best.weekly)
    assert abs(own.regret) <= TIE and own.moved is None
    assert own.edge == pytest.approx((best.worth - truths.best.worth) / quarter.budget, abs=1e-15)
    assert own.edge >= -TIE
    played = built_in(truths)["status quo"]
    held = schedule.score_schedule(truths, np.tile(quarter.status_quo, (PLANNED, 1)))
    assert held.regret == pytest.approx(played.regret + own.edge, rel=1e-9, abs=0.0)
    flat = schedule.score_schedule(truths, np.tile(truths.best.weekly, (PLANNED, 1)))
    assert flat.regret == pytest.approx(own.edge, rel=1e-6, abs=1e-12)
    over = schedule.score_schedule(truths, 1.1 * best.weekly)
    inside = schedule._project(quarter, 1.1 * best.weekly)
    assert over.moved == pytest.approx(float(np.max(np.abs(inside - 1.1 * best.weekly))))
    assert over.regret == pytest.approx(
        (best.worth - schedule.worth(truths.cells, inside)) / quarter.budget, rel=1e-12, abs=0.0
    )
    if truths.realised is None:
        assert own.realised is None
    else:
        hindsight = schedule.oracle(truths.realised, quarter)
        assert own.realised == pytest.approx(
            (hindsight.worth - schedule.worth(truths.realised, best.weekly)) / quarter.budget,
            rel=1e-12,
            abs=0.0,
        )
        assert own.realised >= -TIE


def test_a_channels_schedule_on_a_geo_world_is_allotted_to_its_cells_week_by_week():
    world = GEO.world("policy-0", GEO.pilots["policy-0"][1])
    truths = GEO.truth(world)
    quarter = truths.quarter
    channels = np.bincount(quarter.channel, weights=quarter.status_quo)
    rising = np.linspace(0.8, 1.2, PLANNED)[:, None] * channels[None, :]
    by_channel = schedule.score_schedule(truths, rising)
    by_cell = schedule.score_schedule(truths, np.array([quarter.allot(row) for row in rising]))
    assert by_channel == by_cell
    assert by_channel.regret > 0.0


def test_a_malformed_schedule_and_cells_it_cannot_solve_are_refused():
    world = TRACK_M2.world("drawn", 905)
    truths = TRACK_M2.truth(world)
    with pytest.raises(ValueError, match="each of the quarter's 13 weeks"):
        schedule.score_schedule(truths, np.ones((12, 3)))
    with pytest.raises(ValueError, match="a week by a cell"):
        schedule.worth(truths.cells, np.ones((13, 2)))
    twins = TWINS.truth(TWINS.world("line", TWINS.pilots["line"][0]))
    with pytest.raises(ValueError, match="each quarter week's reach"):
        schedule.oracle(twins.cells, twins.quarter)
    curves = S_CURVES.truth(S_CURVES.world("hill-2", S_CURVES.pilots["hill-2"][0]))
    with pytest.raises(ValueError, match="concave cells alone"):
        schedule.oracle(curves.cells, curves.quarter)
    cell = truths.cells[0]
    with pytest.raises(ValueError, match="week by week"):
        truth.Cell(cell.carry, cell.reach, cell.curve, cell.scale, cell.effect, np.eye(PLANNED))

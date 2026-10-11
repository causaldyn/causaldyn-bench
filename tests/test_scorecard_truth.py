"""The scorecard's oracle against the committed Track M v2 records it must reproduce, and against
searches written apart from it where no record exists: effect paths, more than three cells, and
S-shaped cells, where its branch and bound is checked part by part: each envelope against the worth
it bounds, the relaxation against a linear programme, and the plan against dense grids and a corner
that a local search stops short of."""

import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog, minimize

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.endogenous_mmm import CURVES, LN3, EndogenousMediaMix
from causaldyn_bench.geo_selection import GeoPanel
from causaldyn_bench.mmm_decision import PLANNED, Quarter, drawn, worth
from causaldyn_bench.scorecard import truth
from causaldyn_bench.scorecard.truth import (
    GAP,
    ROUNDING,
    SAMPLES,
    Cell,
    _bounded,
    _envelope,
    _gridded,
    _priced,
    _programmed,
    _refined,
    _relaxed,
    _split,
    cells,
    oracle,
)

ROOT = Path(__file__).resolve().parents[1]
S_SHAPED = sorted(name for name, curve in CURVES.items() if not curve.concave)


def _record(name: str) -> dict:
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def _market(environment: str, seed: int):
    if environment == "drawn":
        return drawn(seed).simulate(seed)
    return EndogenousMediaMix().simulate(seed)


def _reproduces(world, best: float, status_quo: float | None, curves=None) -> None:
    """The oracle's worth, per euro of the budget, and the status quo's regret, each within
    ``TIE`` of what a committed run recorded."""
    quarter = Quarter.after(world)
    on = cells(world, quarter, curves=curves)
    plan = oracle(on, quarter)
    assert quarter.feasible(plan.weekly)
    assert abs(plan.worth - best) <= TIE * quarter.budget
    if status_quo is not None:
        regret = (plan.worth - truth.worth(on, quarter.status_quo)) / quarter.budget
        assert abs(regret - status_quo) <= TIE


@pytest.mark.parametrize(
    "name",
    [
        "track_m2_budgets",
        "track_m2_budgets_pilot",
        "track_m2_external",
        "track_m2_external_pilot_genre",
        "track_m2_external_pilot_union",
    ],
)
def test_every_committed_budgets_and_external_world_is_reproduced(name):
    for environment in _record(name)["environments"]:
        for world in environment["worlds"]:
            status_quo = world["committed" if "committed" in world else "regret"]["status quo"]
            market = _market(environment["environment"]["name"], world["seed"])
            _reproduces(market, world["best"], status_quo)


@pytest.mark.parametrize("name", ["track_m2_families", "track_m2_families_pilot"])
def test_every_committed_curve_family_world_is_reproduced(name):
    """The S-shaped families' plans are searched: the grid and its refinement land where the
    committed search did."""
    for family in _record(name)["curves"]:
        curve = family["environment"]["curve"]
        for world in family["worlds"]:
            seed = world["seed"]
            market = dataclasses.replace(drawn(seed), curve=curve).simulate(seed)
            _reproduces(market, world["best"], world["regret"]["status quo"])


@pytest.mark.parametrize("name", ["track_m2_geo", "track_m2_geo_pilot"])
def test_every_committed_geo_world_and_its_arms_regrets_are_reproduced(name):
    """The geo run's market reads each channel as its geos' mixture, a curve of its own; it kept no
    status quo, so every arm's recorded plan is scored again instead."""
    for world in _record(name)["worlds"]:
        panel = GeoPanel.draw(world["seed"])
        quarter = Quarter.after(panel.world)
        _reproduces(panel.world, world["best"], None, panel.curves())
        on = cells(panel.world, quarter, curves=panel.curves())
        best = oracle(on, quarter).worth
        for arm in world["arms"].values():
            regret = (best - truth.worth(on, arm["weekly"])) / quarter.budget
            assert abs(regret - arm["regret"]) <= TIE


@pytest.fixture(scope="module")
def worlds():
    return [drawn(seed).simulate(seed) for seed in range(4)]


@pytest.mark.parametrize("curve", sorted(CURVES))
def test_a_cells_worth_is_the_decisions_worth_on_the_worlds_channels(curve):
    world = dataclasses.replace(drawn(3), curve=curve).simulate(3)
    quarter = Quarter.after(world)
    on = cells(world, quarter)
    rng = np.random.default_rng(1)
    for _ in range(5):
        weekly = quarter.project(rng.uniform(quarter.lower, quarter.upper))
        assert truth.worth(on, weekly) == pytest.approx(worth(world, quarter, weekly), rel=1e-13)


def _paths(world, seed: int) -> np.ndarray:
    """A positive effect path a channel over the quarter and the tail, moving as a drift would."""
    rng = np.random.default_rng(seed)
    horizon = PLANNED + world.kernel_length - 1
    steps = rng.normal(0.0, 0.08, (len(world.channels), horizon)).cumsum(axis=1)
    return np.asarray(world.effect)[:, None] * np.exp(steps + rng.uniform(-0.5, 0.5, (3, 1)))


def test_a_cells_worth_is_linear_in_its_effect(worlds):
    """So the plan that is best for the expected effect is best in expectation: the expected path
    is all the oracle needs."""
    world = worlds[0]
    quarter = Quarter.after(world)
    first, second = _paths(world, 1), _paths(world, 2)
    weekly = quarter.equal_split()
    mixed = truth.worth(cells(world, quarter, 0.3 * first + 0.7 * second), weekly)
    apart = 0.3 * truth.worth(cells(world, quarter, first), weekly) + 0.7 * truth.worth(
        cells(world, quarter, second), weekly
    )
    assert mixed == pytest.approx(apart, rel=1e-13)


def _searched_apart(on, quarter, starts: int, seed: int) -> float:
    """The best worth a generic SLSQP reaches from random plans in the box, with finite-difference
    gradients: a search that shares nothing with the oracle's."""
    rng = np.random.default_rng(seed)
    rate = quarter.budget / PLANNED
    best = -np.inf
    for _ in range(starts):
        start = quarter.project(rng.uniform(quarter.lower, quarter.upper))
        solution = minimize(
            lambda w: -truth.worth(on, w) / quarter.budget,
            start,
            method="SLSQP",
            bounds=list(zip(quarter.lower, quarter.upper, strict=True)),
            constraints={"type": "eq", "fun": lambda w: float(np.sum(w)) / rate - 1.0},
            options={"ftol": 1e-14, "maxiter": 500},
        )
        if quarter.feasible(solution.x, tolerance=1e-9):
            best = max(best, truth.worth(on, solution.x))
    return best


def test_on_an_effect_path_the_exact_plan_meets_its_conditions_and_no_search_beats_it(worlds):
    for k, world in enumerate(worlds):
        quarter = Quarter.after(world)
        on = cells(world, quarter, _paths(world, 10 + k))
        plan = oracle(on, quarter)
        assert quarter.feasible(plan.weekly)
        for cell, weekly, low, high in zip(
            on, plan.weekly, quarter.lower, quarter.upper, strict=True
        ):
            slope = cell.slope(float(weekly)) / PLANNED
            if np.isclose(weekly, low, rtol=1e-9):
                assert slope <= plan.price * (1 + 1e-9)
            elif np.isclose(weekly, high, rtol=1e-9):
                assert slope >= plan.price * (1 - 1e-9)
            else:
                assert slope == pytest.approx(plan.price, rel=1e-8)
        apart = _searched_apart(on, quarter, starts=6, seed=k)
        assert apart <= plan.worth * (1 + 1e-12)
        assert apart >= plan.worth * (1 - 1e-8)


def _s_shaped(curve: str, seed: int):
    world = dataclasses.replace(drawn(seed), curve=curve).simulate(seed)
    quarter = Quarter.after(world)
    return cells(world, quarter), quarter


@pytest.mark.parametrize("curve", S_SHAPED)
def test_the_programme_finds_the_grids_plan_on_three_cells(curve):
    for seed in range(3):
        on, quarter = _s_shaped(curve, 40 + seed)
        gridded, programmed = _gridded(on, quarter), _programmed(on, quarter)
        assert programmed.worth == pytest.approx(gridded.worth, rel=1e-11)
        np.testing.assert_allclose(programmed.weekly, gridded.weekly, rtol=1e-5, atol=1e-5)


def test_the_programme_finds_the_exact_plan_on_concave_cells(worlds):
    for world in worlds:
        quarter = Quarter.after(world)
        on = cells(world, quarter)
        assert _programmed(on, quarter).worth == pytest.approx(
            _priced(on, quarter).worth, rel=1e-12
        )


def _six(curves: tuple[str, str], seed: int):
    """Two drawn worlds' channels as six cells of one box at the sum of their budgets."""
    on, lower, upper, status_quo = [], [], [], []
    budget = 0.0
    for k, curve in enumerate(curves):
        world = dataclasses.replace(drawn(seed + k), curve=curve).simulate(seed + k)
        quarter = Quarter.after(world)
        on += cells(world, quarter)
        lower.append(quarter.lower)
        upper.append(quarter.upper)
        status_quo.append(quarter.status_quo)
        budget += quarter.budget
    box = Quarter(
        budget,
        np.concatenate(lower),
        np.concatenate(upper),
        np.concatenate(status_quo),
        np.zeros(0),
    )
    return tuple(on), box


@pytest.mark.parametrize("curves", [("hill-2", "logistic-4"), ("weibull-2", "tanh")])
def test_past_three_cells_no_search_from_random_plans_beats_the_programme(curves):
    for seed in (50, 60):
        on, box = _six(curves, seed)
        plan = oracle(on, box)
        assert box.feasible(plan.weekly)
        apart = _searched_apart(on, box, starts=40, seed=seed)
        # by less than a tie: both end in SLSQP, whose stop leaves the worth's last digits to the
        # path it took, up to 2e-13 of it here and 1.6e-12 where numpy's exp is its AVX-512 kernel
        assert apart <= plan.worth + TIE * box.budget


def test_the_programmes_split_is_the_best_of_every_split_of_its_steps():
    """The max-plus recursion against every split of a coarse budget's steps over six cells."""
    on, box = _six(("logistic-4", "hill-2"), 80)
    steps = 12
    step = (box.budget / PLANNED - box.lower.sum()) / steps
    counts = np.minimum(np.floor((box.upper - box.lower) / step).astype(int), steps)
    values = [
        cell.worths(low + step * np.arange(n + 1))
        for cell, low, n in zip(on, box.lower, counts, strict=True)
    ]
    best, where = -np.inf, None
    for split in itertools.product(*(range(n + 1) for n in counts)):
        if sum(split) == steps:
            total = sum(v[k] for v, k in zip(values, split, strict=True))
            if total > best:
                best, where = total, split
    assert where is not None
    split = _split(on, box, steps)
    np.testing.assert_allclose(split, box.lower + step * np.array(where), rtol=1e-12)
    assert truth.worth(on, split) == pytest.approx(best, rel=1e-13)


def test_past_three_concave_cells_the_bisection_and_the_programme_agree():
    on, box = _six(("tanh", "michaelis-menten"), 70)
    assert oracle(on, box).worth == pytest.approx(_programmed(on, box).worth, rel=1e-12)


def test_a_budget_the_box_cannot_spend_and_a_misshapen_cell_are_refused(worlds):
    world = worlds[0]
    quarter = Quarter.after(world)
    on = cells(world, quarter)
    over = dataclasses.replace(quarter, budget=PLANNED * float(quarter.upper.sum()) * 1.01)
    with pytest.raises(ValueError, match="no plan in the box"):
        oracle(on, over)
    with pytest.raises(ValueError, match="cells against a box"):
        oracle(on[:2], quarter)
    with pytest.raises(ValueError, match="same weeks"):
        Cell(np.zeros(18), np.zeros(17), CURVES["tanh"], 0.01, np.ones(18))
    with pytest.raises(ValueError, match="effect path"):
        cells(world, quarter, np.ones((3, 5)))


@pytest.mark.parametrize("curve", sorted(CURVES))
def test_an_envelope_lies_above_its_cells_worth_and_touches_it_at_its_lift(curve):
    """On the whole box and on a random interval of it, the envelope is concave, never below the
    worth at sixty-four points between two of its samples, and above it by its lift, ``bend h^2 /
    8`` and the samples' rounding, where it touches."""
    on, quarter = _s_shaped(curve, 41)
    rng = np.random.default_rng(3)
    for cell, low, high in zip(on, quarter.lower, quarter.upper, strict=True):
        for a, b in ((low, high), tuple(np.sort(rng.uniform(low, high, 2)))):
            envelope = _envelope(cell, a, b)
            dense = np.linspace(a, b, 64 * SAMPLES + 1)
            assert np.min(np.interp(dense, envelope.x, envelope.y) - cell.worths(dense)) >= 0.0
            assert envelope.x[0] == a and envelope.x[-1] == b
            assert np.all(np.diff(np.diff(envelope.y) / np.diff(envelope.x)) <= 1e-12)
            assert cell.bend is not None
            samples = cell.worths(np.linspace(a, b, SAMPLES + 1))
            lift = cell.bend * ((b - a) / SAMPLES) ** 2 / 8 + ROUNDING * np.max(np.abs(samples))
            np.testing.assert_allclose(
                envelope.y - cell.worths(envelope.x), lift, rtol=1e-9, atol=0.0
            )


def test_a_cells_bend_is_its_worths_greatest_concavity_bound():
    """``-worth''`` over the box, read off the slope's central differences, stays below the cell's
    bend; its effect's sign and a curve's stated bend decide whether it has one."""
    on, quarter = _s_shaped("weibull-2", 43)
    for cell, low, high in zip(on, quarter.lower, quarter.upper, strict=True):
        weekly = np.linspace(low, high, 2001)
        step = 1e-6 * (high - low)
        concavity = [-(cell.slope(w + step) - cell.slope(w - step)) / (2 * step) for w in weekly]
        assert cell.bend is not None and max(concavity) <= cell.bend
    cell = on[0]
    assert dataclasses.replace(cell, effect=-cell.effect).bend is None
    assert dataclasses.replace(cell, curve=dataclasses.replace(cell.curve, bend=None)).bend is None


def test_the_relaxation_is_the_linear_programme_over_the_envelopes():
    """The steepest pieces first, against HiGHS on the pieces' fills, on random boxes inside the
    budget's reach."""
    on, quarter = _s_shaped("logistic-4", 42)
    rate = quarter.budget / PLANNED
    rng = np.random.default_rng(5)
    for _ in range(6):
        lows = rng.uniform(quarter.lower, (quarter.lower + quarter.upper) / 2)
        highs = rng.uniform((quarter.lower + quarter.upper) / 2, quarter.upper)
        if not lows.sum() <= rate <= highs.sum():
            continue
        envelopes = [_envelope(c, lo, hi) for c, lo, hi in zip(on, lows, highs, strict=True)]
        bound, weekly = _relaxed(envelopes, rate)
        slopes = np.concatenate([np.diff(e.y) / np.diff(e.x) for e in envelopes])
        lengths = np.concatenate([np.diff(e.x) for e in envelopes])
        programme = linprog(
            -slopes,
            A_eq=np.ones((1, slopes.size)),
            b_eq=[rate - lows.sum()],
            bounds=list(zip(np.zeros(slopes.size), lengths, strict=True)),
            method="highs",
        )
        assert programme.status == 0
        start = sum(float(e.y[0]) for e in envelopes)
        assert bound == pytest.approx(start - programme.fun, rel=1e-12, abs=0.0)
        assert float(np.sum(weekly)) == pytest.approx(rate, rel=1e-13, abs=0.0)
        assert np.all(weekly >= lows) and np.all(weekly <= highs * (1 + 1e-15))
        assert sum(
            e.at(w) for e, w in zip(envelopes, weekly.tolist(), strict=True)
        ) == pytest.approx(bound, rel=1e-12, abs=0.0)


def _dense(on, quarter, points: int) -> float:
    """The best worth on a grid of ``points`` a side over every box but the last's, the last cell
    spending the rest: a search written apart from the oracle, on two or three cells."""
    if len(on) not in (2, 3):
        raise ValueError(f"a dense grid takes two or three cells, not {len(on)}")
    rate = quarter.budget / PLANNED
    low, high = quarter.lower, quarter.upper
    middle = np.linspace(low[-2], high[-2], points)
    middles = on[-2].worths(middle)
    firsts = [(0.0, 0.0)]
    if len(on) == 3:
        first = np.linspace(low[0], high[0], points)
        firsts = list(zip(first, on[0].worths(first), strict=True))
    best = -np.inf
    for spent, value in firsts:
        rest = rate - spent - middle
        inside = (rest >= low[-1]) & (rest <= high[-1])
        total = value + middles + on[-1].worths(np.clip(rest, low[-1], high[-1]))
        best = max(best, float(np.max(np.where(inside, total, -np.inf))))
    return best


@pytest.mark.parametrize("curve", S_SHAPED)
def test_on_s_shaped_cells_no_grid_beats_the_bound(curve):
    """Three cells: a grid of 1 201 points a side and the searched plan; two: 200 001 points."""
    for seed in (44, 45):
        on, quarter = _s_shaped(curve, seed)
        plan = oracle(on, quarter)
        np.testing.assert_array_equal(plan.weekly, _bounded(on, quarter).weekly)
        assert quarter.feasible(plan.weekly)
        assert plan.worth == pytest.approx(truth.worth(on, plan.weekly), rel=1e-15, abs=0.0)
        allowance = GAP * quarter.budget
        assert _dense(on, quarter, 1201) <= plan.worth + allowance
        assert _gridded(on, quarter).worth <= plan.worth + allowance
        pair = Quarter(
            PLANNED * float(quarter.status_quo[:2].sum()),
            quarter.lower[:2],
            quarter.upper[:2],
            quarter.status_quo[:2],
            quarter.history[:, :2],
        )
        two = oracle(on[:2], pair)
        assert _dense(on[:2], pair, 200_001) <= two.worth + GAP * pair.budget


def test_the_bound_finds_the_corner_a_local_search_stops_short_of():
    """Two equal logistic cells at a budget inside their convex stretch: the even split meets the
    conditions of a plan, both slopes equal, and SLSQP from it stays there, but it is the worst
    plan; the best spends the budget on one cell."""
    lam = 0.01
    weeks = 4
    cell = Cell(np.zeros(weeks), np.ones(weeks), CURVES["logistic-4"], lam, np.full(weeks, 100.0))
    rate = LN3 / lam
    quarter = Quarter(
        PLANNED * rate, np.zeros(2), np.full(2, rate), np.full(2, rate / 2), np.zeros((0, 2))
    )
    best = oracle((cell, cell), quarter)
    stuck = _refined((cell, cell), quarter, [quarter.status_quo])
    np.testing.assert_allclose(stuck.weekly, quarter.status_quo, rtol=1e-12)
    np.testing.assert_allclose(np.sort(best.weekly), [0.0, rate], atol=1e-9 * rate)
    assert best.worth > 2.0 * stuck.worth
    assert _dense((cell, cell), quarter, 200_001) <= best.worth + GAP * quarter.budget


def test_a_bound_that_cannot_close_its_gap_is_refused(monkeypatch):
    on, quarter = _s_shaped("hill-2", 46)
    monkeypatch.setattr(truth, "NODES", 1)
    with pytest.raises(RuntimeError, match="boxes split"):
        _bounded(on, quarter)


def test_cells_without_a_bend_are_searched():
    """A curve that states no bend leaves the plan to the grid, and to the programme past three
    cells, as before the bound."""
    on, quarter = _s_shaped("hill-2", 47)
    unbent = tuple(
        dataclasses.replace(c, curve=dataclasses.replace(c.curve, bend=None)) for c in on
    )
    np.testing.assert_array_equal(oracle(unbent, quarter).weekly, _gridded(unbent, quarter).weekly)
    with pytest.raises(ValueError, match="states its bend"):
        _envelope(unbent[0], quarter.lower[0], quarter.upper[0])

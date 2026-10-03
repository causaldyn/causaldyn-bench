"""The scorecard's oracle against the committed Track M v2 records it must reproduce, and against
searches written apart from it where no record exists: effect paths, and more than three cells."""

import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.endogenous_mmm import CURVES, EndogenousMediaMix
from causaldyn_bench.geo_selection import GeoPanel
from causaldyn_bench.mmm_decision import PLANNED, Quarter, drawn, worth
from causaldyn_bench.scorecard import truth
from causaldyn_bench.scorecard.truth import (
    Cell,
    _gridded,
    _priced,
    _programmed,
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
        assert apart <= plan.worth * (1 + 1e-12)


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

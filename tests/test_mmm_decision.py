"""Track M v2, decisions: the quarter, a plan's worth and the best plan, checked apart from each
other."""

import jax
import numpy as np
import pytest
from chc.allocation import allocate
from chc.response import Channel, GeometricAdstock, Tanh
from scipy.optimize import minimize

from causaldyn_bench.endogenous_mmm import EndogenousMediaMix
from causaldyn_bench.mmm_decision import PLANNED, Quarter, drawn, oracle, regret, worth


@pytest.fixture(scope="module")
def worlds():
    reference = [EndogenousMediaMix().simulate(seed) for seed in range(5)]
    return reference + [drawn(seed).simulate(seed) for seed in range(5)]


@pytest.fixture
def x64():
    """The library's channels run in JAX; the agreement below is a float64 claim."""
    with jax.enable_x64(True):
        yield


def test_a_plans_worth_is_the_library_channels_return_over_the_quarter_and_its_tail(worlds, x64):
    world = worlds[5]
    quarter = Quarter.after(world)
    weekly = quarter.equal_split()
    tail = np.zeros(world.kernel_length - 1)
    expected = 0.0
    for c, (alpha, lam, beta) in enumerate(
        zip(world.retention, world.saturation, world.effect, strict=True)
    ):
        channel = Channel(GeometricAdstock(alpha, length=6, normalized=True), Tanh(2 / lam), beta)
        spend = np.concatenate([world.spend[:, c], np.full(PLANNED, weekly[c]), tail])
        expected += float(np.sum(np.asarray(channel(spend))[world.week.size :]))
    assert worth(world, quarter, weekly) == pytest.approx(expected, rel=1e-12)


def test_the_status_quo_and_the_equal_split_spend_the_budget_inside_the_box(worlds):
    for world in worlds:
        quarter = Quarter.after(world)
        assert quarter.feasible(quarter.status_quo)
        assert quarter.feasible(quarter.equal_split())
        np.testing.assert_allclose(quarter.project(quarter.status_quo), quarter.status_quo)


def test_the_oracle_meets_its_optimality_conditions_and_no_solver_beats_it(worlds):
    for world in worlds:
        quarter = Quarter.after(world)
        best = oracle(world, quarter)
        assert quarter.feasible(best.weekly)
        step = 1e-5
        for c, weekly in enumerate(best.weekly):
            nudged = best.weekly.copy()
            nudged[c] += step
            slope = (worth(world, quarter, nudged) - best.worth) / step / PLANNED
            if np.isclose(weekly, quarter.lower[c], rtol=1e-9):
                assert slope <= best.price + 1e-4
            elif np.isclose(weekly, quarter.upper[c], rtol=1e-9):
                assert slope >= best.price - 1e-4
            else:
                assert slope == pytest.approx(best.price, abs=1e-4)

        def loss(w, world=world, quarter=quarter):
            return -worth(world, quarter, w) / quarter.budget

        def gradient(w, loss=loss, step=1e-6):
            return np.array(
                [(loss(w + e) - loss(w - e)) / (2 * step) for e in step * np.eye(w.size)]
            )

        general = minimize(
            loss,
            quarter.status_quo,
            jac=gradient,
            method="SLSQP",
            bounds=list(zip(quarter.lower, quarter.upper, strict=True)),
            constraints={
                "type": "eq",
                "fun": lambda w, q=quarter: PLANNED * w.sum() / q.budget - 1,
            },
            options={"ftol": 1e-13, "maxiter": 500},
        )
        assert general.success
        assert -general.fun * quarter.budget <= best.worth * (1 + 1e-9)
        assert -general.fun * quarter.budget == pytest.approx(best.worth, rel=1e-7)


def test_no_plan_in_the_box_scores_a_negative_regret(worlds):
    rng = np.random.default_rng(0)
    for world in worlds:
        quarter = Quarter.after(world)
        best = oracle(world, quarter)
        for _ in range(20):
            weekly = quarter.project(rng.uniform(quarter.lower, quarter.upper))
            assert regret(world, quarter, weekly, best) >= -1e-9 * best.worth


def test_on_his_reference_instance_the_best_plan_is_a_corner(worlds):
    """PLA's return at twice its spend tops the others' at their floors: they sit there, and PLA
    takes what is left, which is why the track draws its channels."""
    for world in worlds[:5]:
        quarter = Quarter.after(world)
        best = oracle(world, quarter)
        np.testing.assert_allclose(best.weekly[1:], quarter.lower[1:], rtol=1e-9)
        assert best.weekly[0] < quarter.upper[0]


def test_drawn_channels_span_his_table_and_move_the_best_plan():
    reference = EndogenousMediaMix()
    generators = [drawn(seed) for seed in range(200)]
    for name in ("retention", "saturation", "effect"):
        values = np.array([getattr(g, name) for g in generators])
        assert values.min() >= min(getattr(reference, name))
        assert values.max() <= max(getattr(reference, name))
    corners = set()
    for seed in range(12):
        world = drawn(seed).simulate(seed)
        quarter = Quarter.after(world)
        best = oracle(world, quarter)
        corners.add(
            tuple(np.round((best.weekly - quarter.lower) / (quarter.upper - quarter.lower), 1))
        )
    assert len(corners) >= 6


def test_regret_refuses_a_plan_that_misses_the_budget(worlds):
    world = worlds[0]
    quarter = Quarter.after(world)
    best = oracle(world, quarter)
    with pytest.raises(ValueError, match="misses the budget"):
        regret(world, quarter, 1.01 * quarter.status_quo, best)


def test_the_library_allocation_on_the_worlds_channels_is_the_oracle(worlds, x64):
    """``chc.allocation.allocate``, which every planning arm calls, against the oracle written
    here without it."""
    for world in worlds:
        quarter = Quarter.after(world)
        channels = tuple(
            Channel(GeometricAdstock(alpha, length=6, normalized=True), Tanh(2 / lam), beta)
            for alpha, lam, beta in zip(
                world.retention, world.saturation, world.effect, strict=True
            )
        )
        plan = allocate(
            channels,
            quarter.budget,
            PLANNED,
            lower=quarter.lower,
            upper=quarter.upper,
            history=quarter.history,
        )
        best = oracle(world, quarter)
        np.testing.assert_allclose(plan.spend, best.weekly, rtol=1e-7, atol=1e-7)
        assert plan.worth == pytest.approx(best.worth, rel=1e-12)
        assert plan.price == pytest.approx(best.price, rel=1e-7)

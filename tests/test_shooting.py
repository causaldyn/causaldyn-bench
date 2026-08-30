"""The derivative-free planner: does CHC's adjoint machinery earn its complexity on this task?"""

from __future__ import annotations

import jax.numpy as jnp
from chc import QuadraticCost, total_cost

from causaldyn_bench.shooting import cross_entropy_control, planner_gap, track_planner
from causaldyn_bench.tracks import DT, fit_dynamics_models


def test_cross_entropy_control_descends_and_respects_the_box() -> None:
    models = fit_dynamics_models(steps=200)
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.1])),
        R=jnp.array([[0.05]]),
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.5, 0.0])
    us, history = cross_entropy_control(
        models["plant"], x0, DT, cost, horizon=20, u_lo=-3.0, u_hi=3.0, iterations=15
    )
    assert bool((jnp.abs(us) <= 3.0 + 1e-9).all())
    assert float(history[-1]) < float(history[0])
    assert float(total_cost(models["plant"], x0, us, DT, cost)) < float(
        total_cost(models["plant"], x0, jnp.zeros_like(us), DT, cost)
    )


def test_the_model_axis_dominates_the_planner_axis() -> None:
    gaps = planner_gap(track_planner())
    planner_spread = abs(gaps["plant/cem"] - gaps["plant/gradient"])
    model_spread = abs(gaps["known_only/gradient"] - gaps["plant/gradient"])
    # Learning the residual is worth far more here than the choice of planner. Asserting the
    # ordering rather than either number keeps the claim about the task, not about a tolerance.
    assert model_spread > 100 * planner_spread
    # And the sampling planner is a real competitor, not a strawman that fails to plan at all.
    assert gaps["plant/cem"] < 0.1 * gaps["known_only/gradient"]

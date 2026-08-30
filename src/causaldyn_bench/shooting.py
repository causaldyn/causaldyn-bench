"""A derivative-free planner, so the benchmark can ask what CHC's gradients are actually worth.

Cross-entropy-method planning samples control sequences, keeps the cheapest elite fraction and
refits a diagonal Gaussian to them. It needs no adjoint, no Jacobian and no differentiable model --
only the ability to evaluate a rollout -- which makes it the honest control against the claim that
the library's discrete-adjoint machinery earns its complexity.

It minimises *exactly* the objective :func:`chc.control.projected_gradient_control` minimises, on
exactly the same models, so the comparison isolates the planner. Two axes cross here: the planner
(gradient vs sampling) and the model (true plant vs learned hybrid vs physics-only). Reading them
apart is the point -- a sampling planner that matches the gradient one on the true plant says the
adjoint buys nothing *here*, which is a result the benchmark should be able to report.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from chc import QuadraticCost, projected_gradient_control, total_cost
from chc.dynamics import Dynamics

from causaldyn_bench.tracks import DT, TrackResult, fit_dynamics_models


def cross_entropy_control(
    model: Dynamics,
    x0: jax.Array,
    dt: float,
    cost: QuadraticCost,
    horizon: int,
    u_lo: float,
    u_hi: float,
    *,
    population: int = 256,
    elite_fraction: float = 0.125,
    iterations: int = 30,
    initial_std: float = 1.0,
    std_floor: float = 1e-3,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Plan a control sequence by the cross-entropy method; return it and the elite-cost history.

    The returned sequence is the final distribution mean clipped to the box, which is the standard
    CEM readout. ``std_floor`` keeps the sampler from collapsing to a point and mistaking a narrow
    distribution for a converged one.
    """
    control_dim = cost.R.shape[0]
    n_elite = max(2, round(elite_fraction * population))
    evaluate = jax.jit(
        jax.vmap(lambda us: total_cost(model, x0, jnp.clip(us, u_lo, u_hi), dt, cost))
    )

    mean = jnp.zeros((horizon, control_dim))
    std = jnp.full((horizon, control_dim), initial_std)
    history = []
    key = jax.random.key(seed)
    for _ in range(iterations):
        key, subkey = jax.random.split(key)
        samples = mean + std * jax.random.normal(subkey, (population, horizon, control_dim))
        costs = evaluate(samples)
        elite = samples[jnp.argsort(costs)[:n_elite]]
        mean = jnp.mean(elite, axis=0)
        std = jnp.maximum(jnp.std(elite, axis=0), std_floor)
        history.append(float(jnp.mean(jnp.sort(costs)[:n_elite])))

    return jnp.clip(mean, u_lo, u_hi), jnp.asarray(history)


def track_planner(
    models: dict[str, Any] | None = None, horizon: int = 30, seed: int = 0
) -> list[TrackResult]:
    """Cross the planner axis with the model axis; score every cell on the true plant.

    Regret is measured against the best cost any cell achieves, not against the gradient planner on
    the plant -- pinning the reference to one arm would hide the case where the other arm wins.
    """
    models = fit_dynamics_models(seed=seed) if models is None else models
    plant = models["plant"]
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.1])),
        R=jnp.array([[0.05]]),
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.5, 0.0])
    u_lo, u_hi = -3.0, 3.0
    us0 = jnp.zeros((horizon, 1))

    plans: dict[str, jax.Array] = {}
    for model_name in ("plant", "hybrid", "known_only"):
        model = models[model_name]
        gradient_plan, _ = projected_gradient_control(
            model, x0, us0, DT, cost, u_lo, u_hi, steps=300
        )
        sampled_plan, _ = cross_entropy_control(model, x0, DT, cost, horizon, u_lo, u_hi, seed=seed)
        plans[f"{model_name}/gradient"] = gradient_plan
        plans[f"{model_name}/cem"] = sampled_plan

    realised = {name: float(total_cost(plant, x0, plan, DT, cost)) for name, plan in plans.items()}
    best = min(realised.values())
    return [
        TrackResult("D-planner", name, "regret", value - best)
        for name, value in sorted(realised.items(), key=lambda item: item[1])
    ]


def planner_gap(results: list[TrackResult]) -> dict[str, float]:
    """Regret by cell, for asserting on the two axes separately."""
    return {result.method: result.value for result in results}


__all__ = ["cross_entropy_control", "planner_gap", "track_planner"]

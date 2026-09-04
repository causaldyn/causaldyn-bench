"""The five benchmark tracks. Each returns ``list[TrackResult]``; competitors are scored on the axis
each deserves, so a tree wins Track A while the hybrid/causal method wins the decision tracks.

    A one-step prediction  |  B long-horizon rollout  |  C counterfactual effect
    D closed-loop control  |  E systems performance

Everything is built on ``chc`` so the benchmark and the library evolve together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from chc import (
    BackdoorOLS,
    ConfoundedLinearSystem,
    DampedOscillator,
    DoubleML,
    HybridDynamics,
    MLPResidual,
    QuadraticCost,
    ZeroResidual,
    fit_residual,
    one_step_mse,
    projected_gradient_control,
    rk4_step,
    rollout,
)
from chc.benchmark import InventoryTask, PricingTask, SupportShiftTask

from causaldyn_bench.baselines import LinearFitDynamics

DT = 0.05


@dataclass(frozen=True)
class TrackResult:
    """One method's score on one track."""

    track: str
    method: str
    metric: str
    value: float
    lower_is_better: bool = True


class _Cubic(eqx.Module):
    """The hidden physics the known model omits (a cubic stiffening term)."""

    beta: float

    def __call__(self, t: float, x: jax.Array, u: jax.Array) -> jax.Array:
        return jnp.array([0.0, -self.beta * x[0] ** 3])


def fit_dynamics_models(seed: int = 0, steps: int = 1500, n: int = 2000) -> dict[str, Any]:
    """Fit the shared dynamics competitors once (Tracks A/B/E reuse them)."""
    known = DampedOscillator(omega=1.0, zeta=0.1)
    plant = HybridDynamics(known=known, residual=_Cubic(beta=0.5))
    k_x, k_u = jax.random.split(jax.random.key(seed))
    xs = jax.random.normal(k_x, (n, 2))
    us = 0.5 * jax.random.normal(k_u, (n, 1))
    x_next = jax.vmap(lambda x, u: rk4_step(plant, 0.0, x, u, DT))(xs, us)
    data = {"x": xs, "u": us, "x_next": x_next}

    known_only = HybridDynamics(known=known, residual=ZeroResidual(2))
    hybrid, _ = fit_residual(
        HybridDynamics(known=known, residual=MLPResidual(2, 1, 2, width=32, key=jax.random.key(1))),
        data,
        DT,
        steps=steps,
    )
    dlm = LinearFitDynamics().fit(np.asarray(xs), np.asarray(us), np.asarray(x_next))
    tree = None
    try:
        from chc.surrogate import GradientBoostedDynamics

        tree = GradientBoostedDynamics().fit(np.asarray(xs), np.asarray(us), np.asarray(x_next))
    except ImportError:
        pass
    return {
        "plant": plant,
        "known_only": known_only,
        "hybrid": hybrid,
        "dlm": dlm,
        "tree": tree,
        "data": data,
    }


def track_a_onestep(models: dict[str, Any]) -> list[TrackResult]:
    """A: one-step prediction RMSE. Trees are expected to win here."""
    xs, us, x_next = models["data"]["x"], models["data"]["u"], models["data"]["x_next"]
    out = [
        TrackResult(
            "A-onestep",
            "known-only",
            "rmse",
            float(one_step_mse(models["known_only"], xs, us, x_next, DT)) ** 0.5,
        ),
        TrackResult(
            "A-onestep",
            "hybrid-CHC",
            "rmse",
            float(one_step_mse(models["hybrid"], xs, us, x_next, DT)) ** 0.5,
        ),
    ]
    dlm_pred = models["dlm"].predict(np.asarray(xs), np.asarray(us))
    dlm_rmse = float(np.sqrt(np.mean((dlm_pred - np.asarray(x_next)) ** 2)))
    out.append(TrackResult("A-onestep", "dlm", "rmse", dlm_rmse))
    if models["tree"] is not None:
        pred = models["tree"].predict(np.asarray(xs), np.asarray(us))
        rmse = float(np.sqrt(np.mean((pred - np.asarray(x_next)) ** 2)))
        out.append(TrackResult("A-onestep", "tree-surrogate", "rmse", rmse))
    return out


def track_b_rollout(
    models: dict[str, Any], seed: int = 1, horizon: int = 40, n_eval: int = 40
) -> list[TrackResult]:
    """B: long-horizon rollout RMSE. Known physics anchors the hybrid over the horizon."""
    plant = models["plant"]
    k0, k1 = jax.random.split(jax.random.key(seed))
    x0s = 0.8 * jax.random.normal(k0, (n_eval, 2))
    u_seq = 0.3 * jax.random.normal(k1, (n_eval, horizon, 1))
    truth = jax.vmap(lambda x0, u: rollout(plant, x0, u, DT))(x0s, u_seq)

    def jax_rmse(model: Any) -> float:
        pred = jax.vmap(lambda x0, u: rollout(model, x0, u, DT))(x0s, u_seq)
        return float(jnp.sqrt(jnp.mean((pred - truth) ** 2)))

    def np_rollout_rmse(model: Any) -> float:
        errs = [
            np.mean(
                (model.rollout(np.asarray(x0s[i]), np.asarray(u_seq[i])) - np.asarray(truth[i]))
                ** 2
            )
            for i in range(n_eval)
        ]
        return float(np.sqrt(np.mean(errs)))

    out = [
        TrackResult("B-rollout", "known-only", "rmse", jax_rmse(models["known_only"])),
        TrackResult("B-rollout", "hybrid-CHC", "rmse", jax_rmse(models["hybrid"])),
        TrackResult("B-rollout", "dlm", "rmse", np_rollout_rmse(models["dlm"])),
    ]
    if models["tree"] is not None:
        out.append(
            TrackResult("B-rollout", "tree-surrogate", "rmse", np_rollout_rmse(models["tree"]))
        )
    return out


def track_c_effect(seed: int = 0, n: int = 20_000) -> list[TrackResult]:
    """C: interventional-effect error under confounding. Causal methods win; naive does not."""
    system = ConfoundedLinearSystem()
    data = system.sample(n, jax.random.key(seed))
    b_true = system.b_true

    def err(effect: float) -> float:
        return abs(effect - b_true)

    return [
        TrackResult(
            "C-effect",
            "naive",
            "ate_error",
            err(float(BackdoorOLS().estimate(data, covariates=("x",)).effect)),
        ),
        TrackResult(
            "C-effect",
            "backdoor-CHC",
            "ate_error",
            err(float(BackdoorOLS().estimate(data, covariates=("x", "z")).effect)),
        ),
        TrackResult(
            "C-effect",
            "double-ml-CHC",
            "ate_error",
            err(float(DoubleML().estimate(data, covariates=("x", "z")).effect)),
        ),
    ]


def track_d_control() -> list[TrackResult]:
    """D: closed-loop regret vs oracle across the CHC benchmark tasks. The decision track."""
    out = []
    for task_name, task in (
        ("pricing", PricingTask()),
        ("inventory", InventoryTask()),
        ("support-shift", SupportShiftTask()),
    ):
        for r in task.run():
            out.append(TrackResult("D-control", f"{task_name}/{r.controller}", "regret", r.regret))
    return out


def track_e_systems(
    models: dict[str, Any], horizon: int = 20, steps: int = 40, repeats: int = 7
) -> list[TrackResult]:
    """E: control-solve latency on CPU (JIT-warmed). A compiled runtime would compete here later.

    Reports the MINIMUM over ``repeats`` timed solves, not one sample. A single sample after a
    single warm-up is not reproducible: measured over 12 fresh calls on an idle machine, the
    known-only solve ranged 0.586-0.823 ms (max/min 1.40x, CV 9.8%) and the FIRST timed sample was
    the outlier every time -- one warm-up does not finish the frequency ramp. That is what moved
    this track from 0.67 to 0.92 ms between two idle-machine snapshots, and it was misattributed
    to machine load before being measured. Latency noise is one-sided, so the minimum estimates
    the uncontended cost and the mean does not.
    """
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.1])),
        R=jnp.array([[0.05]]),
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.0, 0.0])
    us0 = jnp.zeros((horizon, 1))

    def solve(model: Any) -> Any:
        us, _ = projected_gradient_control(model, x0, us0, DT, cost, -5.0, 5.0, steps=steps)
        return us

    out = []
    for name, model in (("known-only", models["known_only"]), ("hybrid-CHC", models["hybrid"])):
        for _ in range(2):  # two warm-ups: one compiles, the second rides out the frequency ramp
            solve(model).block_until_ready()
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            solve(model).block_until_ready()
            best = min(best, (time.perf_counter() - t0) * 1e3)
        out.append(TrackResult("E-systems", name, "solve_ms", best))
    return out

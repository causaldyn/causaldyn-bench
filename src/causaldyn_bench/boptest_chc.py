"""A first CHC controller for BOPTEST: identify a thermal model, then certainty-equivalent control.

Same shape as the CHC pricing flagship (estimate the action's effect, then act on it), here on the
live ``bestest_hydronic_heat_pump``: from a short exploration episode fit ``T_next ~ a*T + b*u + d``
(modulation ``u`` -> next zone temp ``T``); then set the modulation that steers ``T`` to setpoint.
Requires a running BOPTEST-Service; the first controller-in-the-loop step toward an MPC.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from causaldyn_bench.boptest import (
    DEFAULT_TESTCASE,
    BOPTestClient,
    baseline_controller,
    run_episode,
)

# A step function maps (measurements, comfort-bound forecast) to a BOPTEST overwrite dict.
StepFn = Callable[[Mapping[str, float], Mapping[str, list[float]]], Mapping[str, float]]

T_ZONE = "reaTZon_y"  # zone operative temperature (K)
T_SET_HEAT = "reaTSetHea_y"  # heating setpoint (K)
U_HP = "oveHeaPumY_u"  # heat-pump modulation overwrite [0, 1]
U_ACT = "oveHeaPumY_activate"


def identify_thermal_model(
    client: BOPTestClient,
    testcase: str,
    n: int = 96,
    step_s: float = 1800.0,
    seed: int = 0,
    hold: int = 4,
    a_max: float = 0.98,
) -> np.ndarray:
    """Fit ``T_next ~ a*T + b*u + d`` by least squares from a *slow-PRBS* exploration episode.

    I.i.d. per-step modulation poorly excites a building's slow thermal mode: the zone low-passes it
    to its mean and the fitted DC gain ``(b+d)/(1-a)`` collapses far below the true one, so any
    controller then thinks the heat pump can't reach setpoint. Holding each level for ``hold`` steps
    (a slow pseudo-random level sequence) excites the slow mode and recovers the real gain --
    persistent excitation, not a controller tweak.
    """
    rng = np.random.default_rng(seed)
    testid = client.select(testcase)
    try:
        client.set_step(testid, step_s)
        measurements = client.initialize(testid, 0.0, 0.0)
        temps, controls, next_temps = [], [], []
        control = 0.0
        for i in range(n):
            if i % hold == 0:  # slow PRBS: resample the level only every `hold` steps
                control = float(rng.uniform(0.0, 1.0))
            temp = measurements[T_ZONE]
            measurements = client.advance(testid, {U_HP: control, U_ACT: 1})
            temps.append(temp)
            controls.append(control)
            next_temps.append(measurements[T_ZONE])
    finally:
        client.stop(testid)
    temps, controls, next_temps = map(np.asarray, (temps, controls, next_temps))
    features = np.column_stack([temps, controls, np.ones(len(temps))])
    coef, *_ = np.linalg.lstsq(features, next_temps, rcond=None)
    if coef[0] >= a_max:  # OLS on a short slow series drifts to a near-unit-root / unstable pole;
        design = np.column_stack([controls, np.ones(len(controls))])  # impose BIBO stability (a
        rest, *_ = np.linalg.lstsq(design, next_temps - a_max * temps, rcond=None)  # physics prior)
        coef = np.array([a_max, rest[0], rest[1]])
    return coef  # [a, b, d]


def chc_controller(coef: np.ndarray, offset: float = 0.5):
    """Certainty-equivalent control: pick the modulation that drives ``T`` to setpoint+offset."""
    a, b, d = float(coef[0]), float(coef[1]), float(coef[2])

    def control(measurements: Any) -> dict[str, float]:
        temp, setpoint = measurements.get(T_ZONE), measurements.get(T_SET_HEAT)
        if temp is None or setpoint is None or abs(b) < 1e-9:
            return {}
        u = (setpoint + offset - a * temp - d) / b
        return {U_HP: float(np.clip(u, 0.0, 1.0)), U_ACT: 1}

    return control


LOWER_SETP = "LowerSetp[1]"  # forecast of the lower comfort bound (what `tdis` is scored against)
UPPER_SETP = "UpperSetp[1]"  # forecast of the upper comfort bound (for plotting the comfort band)


def _make_mpc_solver(
    coef: np.ndarray,
    w_comfort: float = 800.0,
    margin: float = 1.5,
    steps: int = 60,
    lr: float = 0.1,
):
    """Projected-gradient MPC over a *per-step* comfort-bound trajectory; returns the first action.

    Minimises ``sum_t [w_comfort * relu(sp_t + margin - T_{t+1})^2 + u_t^2]`` on the fitted model
    with ``u in [0, 1]``. Comfort >> energy so it holds the band; feeding the *forecast* of the
    comfort bound (not the current setpoint) is what lets it pre-heat before the occupancy ramp.
    """
    a, b, d = float(coef[0]), float(coef[1]), float(coef[2])

    def cost(u_seq: jax.Array, setpoints: jax.Array, temp0: jax.Array) -> jax.Array:
        def step(temp: jax.Array, u_sp: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            u, setpoint = u_sp
            temp_next = a * temp + b * u + d
            shortfall = jnp.maximum(setpoint + margin - temp_next, 0.0)
            return temp_next, w_comfort * shortfall**2 + u**2

        _, costs = jax.lax.scan(step, temp0, (u_seq, setpoints))
        return jnp.sum(costs)

    grad = jax.jit(jax.grad(cost))

    def solve(temp0: float, setpoints: jax.Array) -> float:
        u = jnp.full(setpoints.shape[0], 0.5)
        for _ in range(steps):  # projected gradient on the box [0, 1]
            u = jnp.clip(u - lr * grad(u, setpoints, temp0), 0.0, 1.0)
        return float(u[0])

    return solve


def mpc_controller(coef: np.ndarray, horizon: int = 8, **solver: Any):
    """A generic :class:`Controller` MPC that tracks the current heating setpoint (no forecast).

    Kept for the plain :func:`run_episode` path; :func:`run_mpc_episode` is the forecast-aware
    version that actually anticipates the occupancy ramp.
    """
    solve = _make_mpc_solver(coef, **solver)

    def control(measurements: Any) -> dict[str, float]:
        temp, setpoint = measurements.get(T_ZONE), measurements.get(T_SET_HEAT)
        if temp is None or setpoint is None:
            return {}
        return {U_HP: solve(temp, jnp.full(horizon, setpoint)), U_ACT: 1}

    return control


def run_mpc_episode(
    client: BOPTestClient,
    testcase: str,
    coef: np.ndarray,
    horizon: int = 16,
    step_s: float = 1800.0,
    horizon_steps: int = 48,
    **solver: Any,
) -> dict[str, float]:
    """Run a forecast-driven comfort MPC in the loop and return the KPIs.

    Each step fetches the ``LowerSetp[1]`` forecast over the horizon and plans against it, so the
    controller pre-heats ahead of the morning comfort ramp instead of tracking the night setback
    down (the failure mode of the setpoint-tracking controllers).
    """
    solve = _make_mpc_solver(coef, **solver)
    testid = client.select(testcase)
    try:
        client.set_step(testid, step_s)
        measurements = client.initialize(testid, 0.0, 0.0)
        for _ in range(horizon_steps):
            forecast = client.forecast(testid, [LOWER_SETP], horizon * step_s, step_s)
            setpoints = jnp.asarray(forecast[LOWER_SETP][1 : horizon + 1])  # next `horizon` bounds
            action = solve(measurements[T_ZONE], setpoints)
            measurements = client.advance(testid, {U_HP: action, U_ACT: 1})
        return client.kpi(testid)
    finally:
        client.stop(testid)


def naive_step(coef: np.ndarray) -> StepFn:
    """Wrap the certainty-equivalent controller as a forecast-ignoring :data:`StepFn`."""
    controller = chc_controller(coef)
    return lambda measurements, _forecast: controller(measurements)


def mpc_step(coef: np.ndarray, horizon: int = 16, **solver: Any) -> StepFn:
    """A forecast-driven comfort-MPC :data:`StepFn`: plan against the comfort-bound look-ahead."""
    solve = _make_mpc_solver(coef, **solver)

    def step(
        measurements: Mapping[str, float], forecast: Mapping[str, list[float]]
    ) -> dict[str, Any]:
        setpoints = jnp.asarray(forecast[LOWER_SETP][1 : horizon + 1])
        return {U_HP: solve(measurements[T_ZONE], setpoints), U_ACT: 1}

    return step


def trace_episode(
    client: BOPTestClient,
    testcase: str,
    step_fn: StepFn,
    horizon: int = 16,
    step_s: float = 1800.0,
    horizon_steps: int = 48,
) -> dict[str, list[float]]:
    """Run ``step_fn`` in the loop, recording the per-step trajectory for the comfort plot.

    Returns parallel series ``hour``/``tzon``/``lower``/``upper``/``action`` (temps in Celsius): the
    zone temperature against the time-varying comfort band, plus the applied modulation. Both bounds
    are fetched every step -- the band for the plot, the look-ahead for the controller.
    """
    testid = client.select(testcase)
    trace: dict[str, list[float]] = {k: [] for k in ("hour", "tzon", "lower", "upper", "action")}
    try:
        client.set_step(testid, step_s)
        measurements = client.initialize(testid, 0.0, 0.0)
        for i in range(horizon_steps):
            forecast = client.forecast(testid, [LOWER_SETP, UPPER_SETP], horizon * step_s, step_s)
            action = step_fn(measurements, forecast)
            trace["hour"].append(i * step_s / 3600.0)
            trace["tzon"].append(measurements[T_ZONE] - 273.15)
            trace["lower"].append(forecast[LOWER_SETP][0] - 273.15)
            trace["upper"].append(forecast[UPPER_SETP][0] - 273.15)
            trace["action"].append(float(action.get(U_HP, 0.0)))
            measurements = client.advance(testid, action)
    finally:
        client.stop(testid)
    return trace


def compare_controllers(
    base_url: str, testcase: str = DEFAULT_TESTCASE, horizon_steps: int = 48, step_s: float = 1800.0
) -> dict[str, Any]:
    """Identify the model, then score baseline, naive CHC, and the forecast-driven comfort MPC."""
    client = BOPTestClient(base_url)
    coef = identify_thermal_model(client, testcase, n=2 * horizon_steps, step_s=step_s)

    def episode(controller: Any) -> dict[str, float]:
        return run_episode(client, testcase, controller, horizon_steps=horizon_steps, step_s=step_s)

    return {
        "model": coef.tolist(),
        "baseline": episode(baseline_controller()),
        "chc-naive": episode(chc_controller(coef)),
        "chc-mpc": run_mpc_episode(
            client, testcase, coef, horizon_steps=horizon_steps, step_s=step_s
        ),
    }

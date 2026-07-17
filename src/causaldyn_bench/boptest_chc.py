"""A first CHC controller for BOPTEST: identify a thermal model, then certainty-equivalent control.

Same shape as the CHC pricing flagship (estimate the action's effect, then act on it), here on the
live ``bestest_hydronic_heat_pump``: from a short exploration episode fit ``T_next ~ a*T + b*u + d``
(modulation ``u`` -> next zone temp ``T``); then set the modulation that steers ``T`` to setpoint.
Requires a running BOPTEST-Service; the first controller-in-the-loop step toward an MPC.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causaldyn_bench.boptest import (
    DEFAULT_TESTCASE,
    BOPTestClient,
    baseline_controller,
    run_episode,
)

T_ZONE = "reaTZon_y"  # zone operative temperature (K)
T_SET_HEAT = "reaTSetHea_y"  # heating setpoint (K)
U_HP = "oveHeaPumY_u"  # heat-pump modulation overwrite [0, 1]
U_ACT = "oveHeaPumY_activate"


def identify_thermal_model(
    client: BOPTestClient, testcase: str, n: int = 48, step_s: float = 1800.0, seed: int = 0
) -> np.ndarray:
    """Run an exploration episode with random modulation and fit ``[a, b, d]`` by least squares."""
    rng = np.random.default_rng(seed)
    testid = client.select(testcase)
    try:
        client.set_step(testid, step_s)
        measurements = client.initialize(testid, 0.0, 0.0)
        temps, controls, next_temps = [], [], []
        for _ in range(n):
            temp = measurements[T_ZONE]
            control = float(rng.uniform(0.0, 1.0))
            measurements = client.advance(testid, {U_HP: control, U_ACT: 1})
            temps.append(temp)
            controls.append(control)
            next_temps.append(measurements[T_ZONE])
    finally:
        client.stop(testid)
    features = np.column_stack([temps, controls, np.ones(len(temps))])
    coef, *_ = np.linalg.lstsq(features, np.asarray(next_temps), rcond=None)
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


def compare_baseline_vs_chc(
    base_url: str, testcase: str = DEFAULT_TESTCASE, horizon_steps: int = 48, step_s: float = 1800.0
) -> dict[str, Any]:
    """Identify the model, then score the baseline and the CHC controller on one episode."""
    client = BOPTestClient(base_url)
    coef = identify_thermal_model(client, testcase, n=horizon_steps, step_s=step_s)
    baseline_kpis = run_episode(
        client, testcase, baseline_controller(), horizon_steps=horizon_steps, step_s=step_s
    )
    chc_kpis = run_episode(
        client, testcase, chc_controller(coef), horizon_steps=horizon_steps, step_s=step_s
    )
    return {"model": coef.tolist(), "baseline": baseline_kpis, "chc": chc_kpis}

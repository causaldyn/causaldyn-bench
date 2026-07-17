"""Track G: dynamic causal-effect (impulse-response) recovery + the control payoff it enables.

Two axes, mirroring Track F. **G-effect** -- how well each method recovers the true IRF
``d x_{t+h}/d u_t`` on a confounded ARX: a static one-step read that ignores carryover, ``chc.irf``
local projections vs the structured Toeplitz-Levinson route. **G-payoff** -- the decision the IRF
enables: on a distributed-lag plant a one-step controller (inverts only ``g_1``) overshoots to the
full gain, while deconvolving the whole response tracks. Scoring the effect on what it enables, not
just its accuracy. Both DGPs match the gated ``chc`` tests. See plans/18.
"""

from __future__ import annotations

import numpy as np
from chc import irf_control_sequence, local_projection_irf, structured_irf

from causaldyn_bench.tracks import TrackResult

# x_{t+1} = a x_t + b u_t + c z_t + noise, with the policy u_t = kappa z_t + eta (confounded by z).
_A, _B, _C, _KAPPA = 0.6, 1.0, 1.5, -1.2
_KERNEL = np.array([1.0, 0.6, 0.3, 0.1])  # distributed-lag plant: gain 2.0 spread over 4 lags


def _confounded_arx(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    u = _KAPPA * z + 0.5 * rng.standard_normal(n)
    x = np.zeros(n)
    noise = 0.1 * rng.standard_normal(n)
    for t in range(1, n):
        x[t] = _A * x[t - 1] + _B * u[t - 1] + _C * z[t - 1] + noise[t]
    return {"x": x, "u": u, "z": z}


def _analytic_irf(horizon: int) -> np.ndarray:
    return np.array([0.0] + [_B * _A ** (h - 1) for h in range(1, horizon + 1)])


def _distributed_lag_state(kernel: np.ndarray, u: np.ndarray) -> np.ndarray:
    n, p = u.shape[0], kernel.shape[0]
    x = np.zeros(n)
    for t in range(n):
        x[t] = sum(kernel[k] * u[t - 1 - k] for k in range(p) if t - 1 - k >= 0)
    return x


def _tracking_error(control: np.ndarray, target: np.ndarray) -> float:
    padded = np.concatenate([control, np.zeros(_KERNEL.shape[0])])
    achieved = _distributed_lag_state(_KERNEL, padded)[1 : target.shape[0] + 1]
    return float(np.max(np.abs(achieved - target)))


def track_dynamic_effect(seed: int = 0) -> list[TrackResult]:
    """Score IRF recovery (error vs truth) and the control payoff of the recovered effect."""
    horizon = 6
    data = _confounded_arx(8000, seed)
    truth = _analytic_irf(horizon)

    one_step = float(local_projection_irf(data, 1, adjust_for=("x", "z"))[1])
    static = np.concatenate([[0.0, one_step], np.zeros(horizon - 1)])  # ignores the carryover tail
    projections = np.asarray(local_projection_irf(data, horizon, adjust_for=("x", "z")))
    structured = structured_irf(data, horizon, order=4, adjust_for=("x", "z"))

    def irf_error(estimate: np.ndarray) -> float:
        return float(np.max(np.abs(estimate - truth)))

    results = [
        TrackResult("G-effect", "naive-static", "irf_error", irf_error(static)),
        TrackResult("G-effect", "local-projections", "irf_error", irf_error(projections)),
        TrackResult("G-effect", "structured-toeplitz", "irf_error", irf_error(structured)),
    ]

    rng = np.random.default_rng(seed)
    u = rng.standard_normal(6000)
    lag_data = {"x": _distributed_lag_state(_KERNEL, u) + 0.02 * rng.standard_normal(6000), "u": u}
    irf = np.asarray(local_projection_irf(lag_data, horizon, adjust_for=()))
    target = np.ones(30)
    one_step_error = _tracking_error(target / irf[1], target)
    chc_error = _tracking_error(irf_control_sequence(irf, target), target)
    results += [
        TrackResult("G-payoff", "one-step", "track_error", one_step_error),
        TrackResult("G-payoff", "chc-irf", "track_error", chc_error),
    ]
    return results

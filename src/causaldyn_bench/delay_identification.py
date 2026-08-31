"""Track K: recovering an actuation delay, where the confounder moves the *peak* and the payoff
is a **bifurcation**.

Two axes, mirroring Track G. **K-delay** -- how well each method recovers `tau` from a log whose
policy is confounded, on a plant where the confounder acts at a *different* lag than the incentive.
The failure this track exists to expose is not amplitude bias: the unadjusted impulse response peaks
at the **confounder's** lag, so the estimate is wrong about *when*, not about *how much*. No other
track scores the argmax of an effect. **K-payoff** -- the decision each estimate implies. Under
proportional feedback the closed loop is `x' = -channel*K*x(t - tau)` with an exact boundary at
`channel*K*tau = pi/2` (`chc.delay.delay_margin`), so the score is *discontinuous* in the estimate:
past the boundary the loop Hopf-bifurcates and the cost is a divergence, not a gap.

The three-tier outcome is the content. Ignoring the delay diverges; estimating it badly still
stabilises, because the stabilising set in delay space is a **half-line** (`chc.delay.delay_ball`:
under-estimating by up to 76% is survivable, over-estimating never destabilises); and only
adjustment reaches the oracle. "Estimate something" buys the bifurcation; "estimate it causally"
buys the regret. See plans/24.

The bias has an exploration threshold, and it is exact rather than empirical. Aggregating the plant
over one observation stride, the incentive's window straddles a block boundary and splits **2:1**
across lags 3 and 4, while the confounder's lands wholly in lag 2. So the unadjusted regression
coefficients are `20*b*Var(u)` at lag 3 against `30*c*kappa*Var(z)` at lag 2, and the peak relocates
iff

    sigma_eta^2  <  1.5 * |c*kappa| * sigma_z^2 / |b|  -  kappa^2 * sigma_z^2

-- `sigma_eta = 1.4697` on the shipped constants, bracketed by measurement in `[1.45, 1.50]`. Two
consequences worth stating. The 2:1 split is visible in the measured impulse response (`0.168` at
lag 3, `0.085` at lag 4), which is what makes the derivation checkable rather than decorative. And
**enough exploration in the logging policy restores the correct delay with no adjustment at all**:
this track's failure is a property of a thin log, not of cross-correlation, and the shipped
`sigma_eta = 0.5` sits deliberately below the threshold.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from chc import exact_delayed_rollout
from chc.irf import delay_estimate

from causaldyn_bench.tracks import TrackResult

_DT = 0.01  # plant integration step
_EVERY = 30  # observation stride: tau is 3.33 samples, deliberately off the grid
_N_OBS = 1500
_TAU_U = 1.0  # the incentive's delay -- the estimand
_TAU_Z = 0.6  # the confounder's delay -- exactly 2 samples, so a relocated peak is legible
_CHANNEL, _CONFOUND, _KAPPA = 1.0, -2.0, -1.2  # x' = channel*u(t-tau_u) + confound*z(t-tau_z)
_ETA, _NOISE = 0.5, 0.4  # exploration in the logging policy; plant noise
_STATE_WEIGHT, _CONTROL_WEIGHT = 1.0, 0.1
_SCORE_HORIZON, _STATE_CAP, _GAIN_GRID = 30.0, 20.0, 400


def _confounded_delay_log(seed: int, eta: float = _ETA) -> dict[str, np.ndarray]:
    """Open-loop log: a confounded incentive drives the state `tau_u` later, `z` acts `tau_z` later.

    The incentive is zero-order held across the observation stride, so the fine-grid plant and the
    coarse log describe the same experiment at two resolutions.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(_N_OBS)
    u = _KAPPA * z + eta * rng.standard_normal(_N_OBS)
    lag_u, lag_z = round(_TAU_U / _DT), round(_TAU_Z / _DT)
    steps = _N_OBS * _EVERY
    held_u, held_z = np.repeat(u, _EVERY), np.repeat(z, _EVERY)
    x = np.zeros(steps + 1)
    noise = _NOISE * np.sqrt(_DT) * rng.standard_normal(steps)
    for t in range(steps):
        drive = (_CHANNEL * held_u[t - lag_u] if t >= lag_u else 0.0) + (
            _CONFOUND * held_z[t - lag_z] if t >= lag_z else 0.0
        )
        x[t + 1] = x[t] + _DT * drive + noise[t]
    return {"z": z[:-1], "u": u[:-1], "x": np.diff(x[::_EVERY][:_N_OBS])}


def crosscorrelation_delay(data: dict[str, np.ndarray], horizon: int = 12) -> float:
    """The classical time-delay estimate: the lag maximising `|corr(dx_t, u_{t-h})|`.

    Carried because it is what a practitioner reaches for, and because it is *not* a strawman -- an
    unadjusted local projection is the same estimator up to normalisation, which the tests assert.
    """
    dx, u = data["x"], data["u"]
    scores = [abs(np.corrcoef(dx[h:], u[: dx.shape[0] - h])[0, 1]) for h in range(horizon + 1)]
    return float(np.argmax(scores)) * _DT * _EVERY


def _closed_loop(gains: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """States and issued actions for every gain, on a plant whose delay is `tau`."""
    lag, steps = round(tau / _DT), int(_SCORE_HORIZON / _DT)

    def core(
        t: float | jnp.ndarray, x: jnp.ndarray, x_delayed: jnp.ndarray, u: jnp.ndarray
    ) -> jnp.ndarray:
        return -_CHANNEL * u[0] * x_delayed  # the gain rides in as data, so one trace serves

    def one(gain: jnp.ndarray) -> jnp.ndarray:
        return exact_delayed_rollout(core, jnp.array([1.0]), jnp.full((steps, 1), gain), _DT, lag)[
            :, 0
        ]

    states = np.asarray(jax.vmap(one)(jnp.asarray(gains)))
    states = np.clip(np.nan_to_num(states, nan=_STATE_CAP), -_STATE_CAP, _STATE_CAP)
    return states, -gains[:, None] * states[:, :steps]


def _cost(gains: np.ndarray, tau: float) -> np.ndarray:
    states, actions = _closed_loop(gains, tau)
    return _DT * (
        _STATE_WEIGHT * np.sum(states[:, :-1] ** 2, axis=1)
        + _CONTROL_WEIGHT * np.sum(actions**2, axis=1)
    )


def _best_gain(assumed_tau: float) -> float:
    """The cost-minimising gain, computed on a plant carrying the ASSUMED delay."""
    grid = np.geomspace(0.05, 6.0, _GAIN_GRID)
    return float(grid[int(np.argmin(_cost(grid, assumed_tau)))])


def track_delay_identification(seed: int = 0) -> list[TrackResult]:
    """Score delay recovery under confounding, and the closed loop each estimate decides."""
    data = _confounded_delay_log(seed)
    coarse_dt = _DT * _EVERY
    estimates = {"cross-correlation": crosscorrelation_delay(data)}
    for name, adjust, refine in (
        ("local-projection", (), False),
        ("adjusted-LP", ("z",), False),
        ("adjusted-LP-refined", ("z",), True),
    ):
        estimates[name] = delay_estimate(
            data, horizon=12, dt=coarse_dt, adjust_for=adjust, refine=refine, seed=seed
        ).delay

    gains = {name: _best_gain(tau) for name, tau in estimates.items()}
    gains["delay-blind"] = _best_gain(0.0)
    gains["oracle"] = _best_gain(_TAU_U)
    ordered = list(gains)
    costs = _cost(np.array([gains[name] for name in ordered]), _TAU_U)
    oracle_cost = float(costs[ordered.index("oracle")])

    delay_rows = [
        TrackResult("K-delay", name, "|tau_hat - tau|", abs(tau - _TAU_U))
        for name, tau in estimates.items()
    ]
    payoff_rows = [
        TrackResult("K-payoff", name, "closed-loop regret", float(costs[row]) - oracle_cost)
        for row, name in enumerate(ordered)
    ]
    return delay_rows + payoff_rows

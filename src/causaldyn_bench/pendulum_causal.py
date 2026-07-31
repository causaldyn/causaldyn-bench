"""Track J: the same identification question on a plant that is not a building.

Every number in the BOPTEST track (:mod:`causaldyn_bench.boptest_causal`) comes from one plant, and
a thermal zone is a forgiving one: slow, over-damped, and open-loop stable. This track re-asks the
question on ``Pendulum-v1`` from Gymnasium -- fast, open-loop *unstable* about the operating point
the controller cares about, and mechanical rather than thermal. The environment is third-party code
at a pinned version; the constants below are read off the live object rather than transcribed.

Three things this plant has that the emulator does not:

* **A ground truth.** The environment integrates
  ``thdot' = thdot + (3g/(2l) sin(th) + 3/(m l^2) u) dt``, so the control channel is exactly
  ``3/(m l^2) = 3.0`` and the gravity coefficient exactly ``3g/(2l) = 15.0``. BOPTEST could only
  score arms against a randomised-design *reference*; here an arm can be scored against the answer.
* **A physics prior worth having.** ``sin(th)`` is known from first principles while the actuator
  gain -- the inverse inertia -- is not. That is the exact shape the library claims to serve:
  *causal identification for physics-structured residual dynamics*, with the structure carrying the
  part that is known and the estimator carrying the part that is confounded.
* **A separation between where you can excite and where you must control.** The pendulum can be
  swung freely about the downward equilibrium at any amplitude, and has to be stabilised about the
  *upward* one, which the log never visits. Extrapolating that far is what physics structure buys
  and what a flexible drift cannot fake.

**The confounder is a wind torque.** An exogenous AR(1) disturbance ``w`` is added to the commanded
torque, so the applied torque is ``u + w`` and ``w`` enters the state rate directly. The logging
policy is an operator who partially compensates it, ``u = -k w + e``: the textbook confounder, a
common cause of the action and of the rate. ``w`` is *logged* -- withholding it is the causality-off
ablation, exactly as withholding the weather channel is on BOPTEST -- so nothing here relies on an
unobservable.

The 2x2 is the same as the emulator's, with the logging policy in place of the outdoor reset:

===================  ================  =========================================
logging policy       adjust for wind   expectation
===================  ================  =========================================
wind-reactive        no                biased channel
wind-reactive        yes               recovered
randomised           no                unbiased already
randomised           yes               unchanged
===================  ================  =========================================

What it measured, so that reading the code does not require reading the results file. The plant is
deterministic and the fitted class contains it, so the adjusted arm recovers ``3.000000`` on every
seed of **both** policies -- adjustment neither helps nor harms a randomised design, where the
unadjusted fit already sits at ``+2.985 +- 0.037``. The confounded-and-unadjusted arm returns
``-1.053 +- 0.025`` over ``[-1.107, -0.969]``: the estimator concludes the actuator pushes the
pendulum backwards, on all five seeds. That is the sign flip of ``chc.benchmark.CausalDynamicsTask``
reproduced on third-party physics rather than on a self-authored DGP, and a controller built on it
does exactly what the sign says: scored by Gymnasium's own reward over a 200-step upright-regulation
episode, the de-confounded arm lands **3.1% off an oracle built from the environment's own
dynamics** (0.3289 against 0.3189) while the confounded one costs **1537.04** -- the actuator
saturated at its bound on every step of every seed, pushing the wrong way, and the pendulum on the
floor 5 times out of 5.

Three findings that were not designed in and are worth stating plainly:

* **Partial compensation is worse than full compensation.** The bias is not monotone in the
  operator's gain ``k``. Omitted-variable algebra gives, for white wind,
  ``b_hat = b (k(k-1) s_w^2 + s_e^2) / (k^2 s_w^2 + s_e^2)`` (:func:`predicted_naive_gain`), which
  is negative exactly on ``k`` between the roots of ``k^2 - k + s_e^2/s_w^2`` -- an interval
  strictly inside ``(0, 1)``, and ``(0.132, 0.868)`` at the shipped scales. An operator who
  cancels the wind *perfectly* leaves ``+0.317``, attenuated but correctly signed; one who cancels
  a quarter of it gets ``-1.220``. Over a seven-point sweep at ``rho = 0.6`` the largest gap to
  the closed form is 0.083 in gain units, and that one is at ``k = 0``, where the prediction is
  exactly the truth and the gap is the estimator's own sampling error rather than the formula's;
  wherever the sign flips the gap is at most 0.020.
* **The two axes do not substitute for each other, and neither is bought by prediction.** Dropping
  the physics prior leaves the channel unbiased in the *mean* -- ``+2.872 +- 0.108`` against an
  exact ``+3.000000`` -- because the logged action is exogenous to the angle, so the polynomial's
  misfit of gravity lands in the error term. What it costs is precision on each individual seed
  (a spread of 0.24 where the structured arm has none) and a factor of **829** in the upright
  region the controller works in: 22.47 against 0.0271 rad/s^2. Dropping the *adjustment* instead
  **lowers** held-out one-step error, 0.297 against 0.514, while flipping the channel's sign. Each
  axis is invisible to the other's metric and both are invisible to prediction.
* **Zero exploration returns minus the truth, exactly.** With ``explore = 0`` the operator's action
  is ``-k w``, so at ``k = 0.5`` the applied torque is ``-u`` identically and the unadjusted fit
  returns ``-3.000``: not a large error but the negation of the answer. The adjusted arm returns
  ``0.000`` instead -- with the action residual at ``2e-21`` there is nothing to regress on -- which
  is the honest failure of the two. Identification needs **overlap** here as much as on the emulator
  (:func:`run_overlap_ablation`).

Precision: run with ``JAX_ENABLE_X64=1``. This module does **not** set it at import scope, because
that is a process-global mutation and importing a benchmark module should not change the arithmetic
of every other one in the same session. Unlike the BOPTEST black-box arm the fits here are linear
solves and float32 costs only a few digits, but the recorded numbers are float64 and the MPC's
curvature estimate is not something to run at reduced precision on a plant this stiff.

Requires the ``gym`` extra (``pip install causaldyn-bench[gym]``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from causaldyn_bench.tracks import TrackResult

TRACK = "J-causal-identification-pendulum"
METRIC = "control_gain_error"
ENV_ID = "Pendulum-v1"
HOLDOUT = 0.2  # chronological tail withheld from every arm and scored by `holdout_rate_error`


class ActuatorClipError(RuntimeError):
    """Raised when a step would hit one of the environment's own clips.

    Both clips break the model class the estimator assumes -- the torque clip makes the applied
    action a nonlinear function of the commanded one, and the speed clip truncates the state rate --
    so a log containing either is not a log of a control-affine plant. The designs below are scaled
    to stay clear of both; this exists so that a change of scale fails loudly instead of quietly
    fitting a different plant.
    """


class UnusablePlanError(RuntimeError):
    """Raised when a fitted model cannot produce a finite plan at the states the task visits.

    The counterpart of :class:`causaldyn_bench.boptest_causal.RunawayDriftError`, and it fires for
    the same reason: a model can be a perfectly good *fit* and still be an unusable *plant* for the
    optimiser. Here it is the flexible-drift arm, whose polynomial stands in for gravity across the
    swing it was logged on and diverges by the time the horizon reaches upright. Recorded as a
    refusal rather than as a large cost, because a refusal and a bad plan are different failures and
    averaging them together would report neither.
    """


@dataclass(frozen=True)
class PendulumSpec:
    """The environment's own constants, read off a live instance rather than transcribed."""

    gravity: float  # 3g/(2l), the coefficient on sin(theta)
    gain: float  # 3/(m l^2), the control channel and the estimand
    dt: float
    max_torque: float
    max_speed: float

    @classmethod
    def from_env(cls, env: Any) -> PendulumSpec:
        return cls(
            gravity=3.0 * env.g / (2.0 * env.l),
            gain=3.0 / (env.m * env.l**2),
            dt=env.dt,
            max_torque=float(env.max_torque),
            max_speed=float(env.max_speed),
        )


def make_env() -> Any:
    """The unwrapped ``Pendulum-v1``: the time-limit wrapper would cut an identification log."""
    return gym.make(ENV_ID).unwrapped


def default_spec() -> PendulumSpec:
    """The environment's constants, without leaving an environment open to read them."""
    with closing(make_env()) as env:
        return PendulumSpec.from_env(env)


@dataclass(frozen=True)
class PendulumLog:
    """One identification log: transitions, the logged wind, and the design that produced them."""

    theta: np.ndarray
    theta_dot: np.ndarray
    action: np.ndarray
    wind: np.ndarray
    theta_next: np.ndarray
    theta_dot_next: np.ndarray
    policy: str
    compensation: float
    explore: float
    spec: PendulumSpec

    def as_data(self) -> dict[str, Array]:
        """Columns in the layout :func:`chc.dynamics_id.fit_causal_residual` expects."""
        return {
            "x": jnp.asarray(np.column_stack([self.theta, self.theta_dot])),
            "u": jnp.asarray(self.action[:, None]),
            "wind": jnp.asarray(self.wind[:, None]),
            "x_next": jnp.asarray(np.column_stack([self.theta_next, self.theta_dot_next])),
        }

    def split(self, fraction: float = HOLDOUT) -> tuple[PendulumLog, PendulumLog]:
        """Chronological train/holdout split -- the tail is never seen by any arm's fit."""
        cut = round(len(self.theta) * (1.0 - fraction))
        return self._slice(slice(None, cut)), self._slice(slice(cut, None))

    def _slice(self, window: slice) -> PendulumLog:
        return PendulumLog(
            theta=self.theta[window],
            theta_dot=self.theta_dot[window],
            action=self.action[window],
            wind=self.wind[window],
            theta_next=self.theta_next[window],
            theta_dot_next=self.theta_dot_next[window],
            policy=self.policy,
            compensation=self.compensation,
            explore=self.explore,
            spec=self.spec,
        )

    @property
    def swing(self) -> float:
        """Half-width of the visited angle range, in radians about the log's own centre."""
        return 0.5 * float(self.theta.max() - self.theta.min())


def log_episode(
    spec: PendulumSpec,
    *,
    policy: str,
    seed: int,
    steps: int = 4000,
    compensation: float = 0.5,
    explore: float = 0.06,
    wind_scale: float = 0.175,
    wind_rho: float = 0.6,
    wind_cap: float = 0.5,
    segment: int = 200,
    swing: float = 1.6,
) -> PendulumLog:
    """Excite the pendulum about its *downward* equilibrium and record the transitions.

    Why downward, and why in segments. The upward equilibrium is where the controller works and the
    place a stabilising operator would log; it is also unreachable as an excitation design, because
    the torque needed to hold ``|sin(th)| > u_max * gain / gravity = 0.4`` does not exist on this
    actuator. Downward the plant is conservative -- Gymnasium's pendulum has no friction -- so an
    amplitude set by the initial condition persists, and the whole torque budget is left for the
    identification design instead of being spent holding the plant up. Each segment starts from a
    fresh draw so the log covers the swing amplitudes rather than one orbit.

    The two policies command the **same action variance**, which is what "equal excitation budget"
    can mean when one of them is defined by cancelling a disturbance:

    * ``reactive``: ``u = -k w + e`` -- an operator compensating the measured wind, exploring with
      ``e``. This is the confounded design.
    * ``random``: ``u = e'`` with ``var(e') = k^2 var(w) + var(e)`` -- the randomised reference.

    They do *not* visit the same states, and that is a property of the design rather than an
    oversight: a policy that cancels half the wind leaves a quieter plant. :attr:`PendulumLog.swing`
    reports the difference instead of hiding it.
    """
    if policy not in {"reactive", "random"}:
        raise ValueError(f"unknown logging policy {policy!r}; expected 'reactive' or 'random'")
    env = make_env()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    innovation = wind_scale * math.sqrt(1.0 - wind_rho**2)
    matched = math.hypot(explore, compensation * wind_scale)
    wind = 0.0
    rows = []
    for step in range(steps):
        if step % segment == 0:
            env.state = np.array([np.pi + rng.uniform(-swing, swing), rng.uniform(-1.0, 1.0)])
        wind = float(np.clip(wind_rho * wind + innovation * rng.normal(), -wind_cap, wind_cap))
        theta, theta_dot = (float(value) for value in env.state)
        noise = rng.normal()
        action = -compensation * wind + explore * noise if policy == "reactive" else matched * noise
        applied = action + wind
        if abs(applied) >= spec.max_torque:
            raise ActuatorClipError(
                f"applied torque {applied:+.3f} reaches the environment's clip at "
                f"{spec.max_torque}: the logged action would stop being the applied one"
            )
        env.step(np.array([applied]))
        theta_next, theta_dot_next = (float(value) for value in env.state)
        if abs(theta_dot_next) >= spec.max_speed:
            raise ActuatorClipError(
                f"angular velocity {theta_dot_next:+.3f} reaches the environment's speed clip at "
                f"{spec.max_speed}: the transition is no longer a transition of the fitted class"
            )
        rows.append((theta, theta_dot, action, wind, theta_next, theta_dot_next))
    env.close()
    columns = np.asarray(rows, dtype=np.float64)
    return PendulumLog(
        theta=columns[:, 0],
        theta_dot=columns[:, 1],
        action=columns[:, 2],
        wind=columns[:, 3],
        theta_next=columns[:, 4],
        theta_dot_next=columns[:, 5],
        policy=policy,
        compensation=compensation,
        explore=explore,
        spec=spec,
    )


class _KnownPendulum:
    """``f_known`` for :func:`chc.dynamics_id.fit_causal_residual`, or zero with the prior off.

    The angle row carries ``dt`` times the acceleration because the environment integrates
    semi-implicitly (``th' = th + dt * thdot'``, using the *updated* velocity). Writing the known
    part any other way would charge the integrator's coupling to the residual and make the fitted
    channel disagree with the environment by a factor of ``dt`` on one row -- an error that reads as
    a modelling failure and is arithmetic.
    """

    def __init__(self, spec: PendulumSpec, *, physics: bool) -> None:
        self.spec, self.physics = spec, physics

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        if not self.physics:
            return jnp.zeros(2, dtype=x.dtype)
        acceleration = self.spec.gravity * jnp.sin(x[0])
        return jnp.stack([x[1] + self.spec.dt * acceleration, acceleration])


def _drift_features(
    theta: Array, theta_dot: Array, degree: int, centre: tuple[float, float]
) -> Array:
    """Centred monomials up to ``degree``, bias first -- one basis for the fit and the rollout."""
    angle = theta - centre[0]
    speed = theta_dot - centre[1]
    terms = [jnp.ones_like(angle)]
    for total in range(1, degree + 1):
        terms.extend(angle ** (total - power) * speed**power for power in range(total + 1))
    return jnp.stack(terms, axis=-1)


@dataclass(frozen=True)
class PendulumFit:
    """A fitted plant plus what is and is not known about it.

    ``gain`` is the estimand. ``coupling`` is the same channel seen on the angle row, whose truth is
    ``gain * dt``; it is reported because it is a free consistency check that costs nothing -- a fit
    whose two rows disagree about the actuator by more than the integrator's factor has a bug, not a
    bias.
    """

    gain: float
    coupling: float
    drift: tuple[tuple[float, ...], ...]  # (2, n_features) on `_drift_features`
    drift_degree: int
    centre: tuple[float, float]
    physics: bool
    adjusted: bool
    policy: str
    identified: bool
    method: str
    channel_error: float | None
    action_residual_variance: float
    nuisance_r2_action: float
    spec: PendulumSpec

    @property
    def gain_error(self) -> float:
        """Absolute distance from the environment's own ``3/(m l^2)``."""
        return abs(self.gain - self.spec.gain)

    @property
    def sign_agrees(self) -> bool:
        """Does the fit agree with the plant about which way the actuator pushes?"""
        return self.gain * self.spec.gain > 0.0

    def rate(self, theta: float | Array, theta_dot: float | Array, action: float | Array) -> Array:
        """``(d theta/dt, d theta_dot/dt)`` at one state -- the only thing the MPC reads."""
        state = jnp.stack([jnp.asarray(theta), jnp.asarray(theta_dot)])
        known = _KnownPendulum(self.spec, physics=self.physics)(0.0, state, jnp.asarray(action))
        features = _drift_features(state[0], state[1], self.drift_degree, self.centre)
        drift = jnp.asarray(self.drift) @ features
        channel = jnp.asarray([self.coupling, self.gain])
        return known + drift + channel * action

    def next_state(
        self, theta: float | Array, theta_dot: float | Array, action: float | Array
    ) -> Array:
        """One Euler step of :meth:`rate` -- exactly the discretisation the estimator inverts."""
        state = jnp.stack([jnp.asarray(theta), jnp.asarray(theta_dot)])
        return state + self.spec.dt * self.rate(theta, theta_dot, action)


def oracle_fit(spec: PendulumSpec) -> PendulumFit:
    """The environment's own dynamics wearing the :class:`PendulumFit` interface.

    Not a competitor: the reference an arm's closed-loop cost is regret against. Its existence is
    the whole reason this plant was added -- on an emulator there is nothing to put here.
    """
    return PendulumFit(
        gain=spec.gain,
        coupling=spec.gain * spec.dt,
        drift=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        drift_degree=1,
        centre=(0.0, 0.0),
        physics=True,
        adjusted=True,
        policy="oracle",
        identified=True,
        method="known",
        channel_error=0.0,
        action_residual_variance=float("nan"),
        nuisance_r2_action=float("nan"),
        spec=spec,
    )


def fit_pendulum(
    log: PendulumLog,
    *,
    adjusted: bool,
    physics: bool = True,
    folds: int = 2,
    ridge: float = 1e-6,
    drift_degree: int | None = None,
) -> PendulumFit:
    """Channel from the orthogonal moment, drift from least squares on what the channel leaves.

    Two stages, and the order is the point -- the same split :func:`causaldyn_bench.boptest_causal
    .fit_thermal` uses. ``chc.dynamics_id.fit_causal_residual`` runs at ``degree=0`` because the
    truth *is* a constant channel here, so a state-dependent one would only add variance to the
    quantity being scored. The drift then absorbs whatever is left, on a basis whose flexibility is
    the physics ablation's knob: with the prior on, ``a_theta`` has nothing to explain and a linear
    basis suffices; with it off, the same basis has to stand in for ``15 sin(theta)``.
    """
    from chc.dynamics_id import fit_causal_residual

    degree = drift_degree if drift_degree is not None else (1 if physics else 5)
    known = _KnownPendulum(log.spec, physics=physics)
    fit = fit_causal_residual(
        known,
        log.as_data(),
        log.spec.dt,
        adjust_for=("wind",) if adjusted else (),
        degree=0,
        folds=folds,
        ridge=ridge,
    )
    coupling = float(fit.residual.channel[0, 0, 0])
    gain = float(fit.residual.channel[1, 0, 0])
    centre = (float(log.theta.mean()), float(log.theta_dot.mean()))
    drift = _fit_drift(log, known, (coupling, gain), degree, centre)
    return PendulumFit(
        gain=gain,
        coupling=coupling,
        drift=drift,
        drift_degree=degree,
        centre=centre,
        physics=physics,
        adjusted=adjusted,
        policy=log.policy,
        identified=fit.identified,
        method=fit.method,
        channel_error=fit.channel_error,
        action_residual_variance=fit.action_residual_variance,
        nuisance_r2_action=fit.nuisance_r2_action,
        spec=log.spec,
    )


def _fit_drift(
    log: PendulumLog,
    known: _KnownPendulum,
    channel: tuple[float, float],
    degree: int,
    centre: tuple[float, float],
) -> tuple[tuple[float, ...], ...]:
    """OLS of the rate net of the identified channel on the centred monomial basis."""
    state = jnp.asarray(np.column_stack([log.theta, log.theta_dot]))
    action = jnp.asarray(log.action)
    rate = jnp.asarray(
        np.column_stack([log.theta_next - log.theta, log.theta_dot_next - log.theta_dot])
        / log.spec.dt
    )
    known_part = jax.vmap(lambda x, u: known(0.0, x, u))(state, action[:, None])
    target = rate - known_part - jnp.asarray(channel) * action[:, None]
    design = _drift_features(state[:, 0], state[:, 1], degree, centre)
    coefficients = jnp.linalg.lstsq(design, target, rcond=None)[0]  # (F, 2)
    return tuple(tuple(float(value) for value in row) for row in np.asarray(coefficients).T)


def predicted_naive_gain(log: PendulumLog) -> float:
    """Closed-form omitted-variable bias of the unadjusted fit, for **white** wind.

    With applied torque ``u + w``, ``u = -k w + e`` and ``w`` independent of ``e`` and of the state,
    the unadjusted least-squares channel is ``b Cov(u+w, u) / Var(u)``, that is::

        b_hat = b * (k(k-1) s_w^2 + s_e^2) / (k^2 s_w^2 + s_e^2)

    Negative exactly when ``k(1-k) s_w^2 > s_e^2``, which is an interval strictly inside ``(0, 1)``:
    an operator who cancels the wind completely leaves an attenuated gain with the right sign, one
    who cancels part of it can flip it. Autocorrelated wind breaks the independence from the state
    -- the state carries information about ``w`` through its own past -- so this is a prediction to
    be checked rather than an identity; :func:`run_compensation_sweep` reports the gap.
    """
    wind_variance = float(np.var(log.wind))
    explore_variance = log.explore**2
    numerator = log.compensation * (log.compensation - 1.0) * wind_variance + explore_variance
    denominator = log.compensation**2 * wind_variance + explore_variance
    return log.spec.gain * numerator / denominator


def holdout_rate_error(fit: PendulumFit, log: PendulumLog) -> float:
    """RMSE of the predicted angular acceleration on rows no arm's fit has seen."""
    predicted = jax.vmap(fit.rate)(
        jnp.asarray(log.theta), jnp.asarray(log.theta_dot), jnp.asarray(log.action)
    )
    truth = jnp.asarray((log.theta_dot_next - log.theta_dot) / log.spec.dt)
    return float(jnp.sqrt(jnp.mean((predicted[:, 1] - truth) ** 2)))


def extrapolation_error(
    fit: PendulumFit,
    *,
    seed: int = 0,
    samples: int = 4000,
    angle: float = 0.5,
    speed: float = 2.0,
    torque: float = 1.0,
) -> float:
    """RMSE of the predicted acceleration in the *upright* region the log never visits.

    The control task lives here and the identification log does not, which is the point: an arm
    that carries the known ``sin`` term is being asked to interpolate a constant, and an arm that
    fitted a polynomial to gravity is being asked to extrapolate three radians.
    """
    rng = np.random.default_rng(seed + 777)
    theta = jnp.asarray(rng.uniform(-angle, angle, samples))
    theta_dot = jnp.asarray(rng.uniform(-speed, speed, samples))
    action = jnp.asarray(rng.uniform(-torque, torque, samples))
    predicted = jax.vmap(fit.rate)(theta, theta_dot, action)
    truth = fit.spec.gravity * jnp.sin(theta) + fit.spec.gain * action
    return float(jnp.sqrt(jnp.mean((predicted[:, 1] - truth) ** 2)))


def _mpc_solver(
    fit: PendulumFit,
    *,
    horizon: int,
    iterations: int,
    bound: float,
    weight_speed: float,
    weight_effort: float,
    learning_rate: float,
):
    """Projected-Adam MPC on the fitted model, scoring the environment's own cost.

    The objective is Gymnasium's: ``th^2 + 0.1 thdot^2 + 0.001 u^2``, summed over the horizon. It is
    charged on the states the plan *reaches* rather than on the state it starts from, which differs
    from the environment's bookkeeping by the current state's contribution -- a constant in the
    actions, so the minimiser is the same and the gradient is not dominated by a term nothing can
    move.

    ``fit`` enters only through :meth:`PendulumFit.next_state`, so every arm gets the same horizon,
    the same iteration count and the same projection.

    **This is not the BOPTEST solver, and the reason is measured rather than stylistic.** That one
    takes a Nesterov step of ``1/L`` with ``L`` the largest eigenvalue of the objective's Hessian.
    Both halves of that fail here. On an inverted pendulum the horizon rollout is nonconvex: at the
    zero-action initialisation from ``th = 0.15`` the Hessian's spectrum runs ``[-15.14, +0.45]``,
    so the largest *algebraic* eigenvalue is not a bound on the gradient's Lipschitz constant and
    ``1/L`` overshoots by a factor of 33. And with the correct ``max|lambda|`` the accelerated
    method still does not converge, because momentum tuned for a convex landscape oscillates on this
    one: measured on the oracle model the objective went 290.7 at 60 iterations, 7.85 at 600 and
    back to 206.1 at 2000. A method whose answer is non-monotone in its budget cannot be given a
    fixed budget and called equal compute.

    Projected Adam is used instead. It is the same optimiser for every arm at the same fixed
    iteration count, and its per-coordinate normalisation is what makes that fair: Adam's step is
    invariant to rescaling the objective, so an arm that identifies a larger channel -- and so a
    stiffer problem -- is not handed a worse controller by a shared constant. That was exactly the
    defect the BOPTEST solver documents; the fix there was to derive the step from the model, and
    the fix here is to use a method that does not need to. Against ``scipy.optimize`` L-BFGS-B on
    the same box-constrained problem it lands within 14% of the optimum on the oracle model and hits
    it exactly on the confounded one; a test pins the gap. L-BFGS-B itself was rejected as the
    shipped solver because it terminates adaptively, so its gradient count -- the compute being
    equalised -- would differ per arm.

    The wind is not in the plan. It is a disturbance no arm observes ahead of time, and giving the
    adjusted arm its coefficient would be comparing a controller with a forecast against controllers
    without one.
    """

    def rollout(actions: Array, theta0: Array, theta_dot0: Array) -> Array:
        def step(state: Array, action: Array) -> tuple[Array, Array]:
            nxt = fit.next_state(state[0], state[1], action)
            return nxt, nxt[0] ** 2 + weight_speed * nxt[1] ** 2 + weight_effort * action**2

        _, costs = jax.lax.scan(step, jnp.stack([theta0, theta_dot0]), actions)
        return jnp.sum(costs)

    gradient = jax.jit(jax.grad(rollout))
    optimiser = optax.adam(learning_rate)

    def solve(theta0: float, theta_dot0: float) -> Array:
        """The whole plan, not just its first action -- so its objective value is testable."""
        origin = (jnp.asarray(theta0), jnp.asarray(theta_dot0))
        actions = jnp.zeros(horizon)
        state = optimiser.init(actions)
        for _ in range(iterations):
            updates, state = optimiser.update(gradient(actions, *origin), state, actions)
            actions = jnp.clip(actions + jnp.asarray(updates), -bound, bound)
        return actions

    return solve


def run_control_episode(
    fit: PendulumFit,
    *,
    seed: int,
    steps: int = 200,
    theta0: float = 0.15,
    horizon: int = 25,
    iterations: int = 400,
    learning_rate: float = 0.3,
    wind_scale: float = 0.175,
    wind_rho: float = 0.6,
    wind_cap: float = 0.5,
    weight_speed: float = 0.1,
    weight_effort: float = 0.001,
) -> dict[str, Any]:
    """Stabilise the pendulum upright against the same wind, scoring the environment's own reward.

    The commanded torque is bounded by ``max_torque - wind_cap`` so that ``u + w`` never reaches the
    environment's clip: with the clip active the arms would be compared through a saturation
    nonlinearity none of them models, and the arm commanding hardest would be flattered most. What
    is left, ``1.5``, buys an equilibrium out to ``|th| = asin(1.5 * 3 / 15) = 0.305`` rad, so
    ``theta0`` sits at half the feasible displacement and the wind is worth at most ``1.5`` of the
    ``4.5 rad/s^2`` the actuator can produce. Those margins are why the wind is the *same* process
    as the identification log's rather than a second one: the confounding strength is set by
    ``wind_scale / explore`` and the controller's authority by ``wind_scale`` alone, so both can be
    halved together without moving a single identification number.

    Raises:
        UnusablePlanError: if the solver returns a non-finite action, which a model that
            extrapolates a polynomial drift three radians outside its log will.
    """
    spec = fit.spec
    env = make_env()
    env.reset(seed=seed)
    env.state = np.array([theta0, 0.0])
    rng = np.random.default_rng(seed + 20_000)
    innovation = wind_scale * math.sqrt(1.0 - wind_rho**2)
    solve = _mpc_solver(
        fit,
        horizon=horizon,
        iterations=iterations,
        bound=spec.max_torque - wind_cap,
        weight_speed=weight_speed,
        weight_effort=weight_effort,
        learning_rate=learning_rate,
    )
    wind = 0.0
    cost = 0.0
    worst = 0.0
    effort = 0.0
    for _ in range(steps):
        wind = float(np.clip(wind_rho * wind + innovation * rng.normal(), -wind_cap, wind_cap))
        theta, theta_dot = (float(value) for value in env.state)
        action = float(solve(theta, theta_dot)[0])
        if not math.isfinite(action):
            env.close()
            raise UnusablePlanError(
                f"the plan is non-finite at theta={theta:+.3f}, theta_dot={theta_dot:+.3f} for a "
                f"fit with gain {fit.gain:+.4f} and a degree-{fit.drift_degree} drift"
            )
        _, reward, _, _, _ = env.step(np.array([action + wind]))
        cost += -float(reward)
        effort += action**2
        worst = max(worst, abs(_wrap(float(env.state[0]))))
    env.close()
    return {
        "cost": cost,
        "worst_angle": worst,
        "effort": effort,
        "fell": bool(worst > 0.5 * math.pi),
        "gain": fit.gain,
        "sign_agrees": fit.sign_agrees,
        "policy": fit.policy,
        "adjusted": fit.adjusted,
        "physics": fit.physics,
    }


def _wrap(angle: float) -> float:
    """Gymnasium's ``angle_normalize``: the environment's own notion of how far from upright."""
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


def run_case(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    steps: int = 4000,
    compensation: float = 0.5,
    explore: float = 0.06,
) -> dict[str, Any]:
    """The 2x2: two logging policies by two adjustment sets, on a shared step budget."""
    arms: dict[str, list[dict[str, float]]] = {}
    for seed in seeds:
        for policy in ("reactive", "random"):
            spec = default_spec()
            log = log_episode(
                spec,
                policy=policy,
                seed=seed,
                steps=steps,
                compensation=compensation,
                explore=explore,
            )
            train, holdout = log.split()
            for adjusted in (False, True):
                fit = fit_pendulum(train, adjusted=adjusted)
                name = f"{policy}-{'adjusted' if adjusted else 'naive'}"
                arms.setdefault(name, []).append(
                    {
                        "gain": fit.gain,
                        "gain_error": fit.gain_error,
                        "coupling": fit.coupling,
                        "holdout": holdout_rate_error(fit, holdout),
                        "overlap": fit.action_residual_variance,
                        "channel_error": float("nan")
                        if fit.channel_error is None
                        else fit.channel_error,
                        "swing": log.swing,
                    }
                )
    return {"truth": default_spec().gain, "arms": arms}


def run_structure_ablation(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    steps: int = 4000,
    policy: str = "reactive",
) -> dict[str, list[dict[str, float]]]:
    """Physics on/off crossed with adjustment on/off: are the two axes really independent?

    Scored on three metrics at once because no single one ranks both axes -- the channel, held-out
    one-step error inside the log's own swing, and the acceleration error in the upright region the
    log never reaches.
    """
    arms: dict[str, list[dict[str, float]]] = {}
    for seed in seeds:
        spec = default_spec()
        log = log_episode(spec, policy=policy, seed=seed, steps=steps)
        train, holdout = log.split()
        for physics in (True, False):
            for adjusted in (False, True):
                fit = fit_pendulum(train, adjusted=adjusted, physics=physics)
                name = (
                    f"{'physics' if physics else 'flexible'}-{'adjusted' if adjusted else 'naive'}"
                )
                arms.setdefault(name, []).append(
                    {
                        "gain": fit.gain,
                        "gain_error": fit.gain_error,
                        "holdout": holdout_rate_error(fit, holdout),
                        "extrapolation": extrapolation_error(fit, seed=seed),
                    }
                )
    return arms


def run_overlap_ablation(
    *,
    seed: int = 0,
    steps: int = 4000,
    explores: Sequence[float] = (0.25, 0.12, 0.06, 0.025, 0.0),
) -> list[dict[str, float]]:
    """Drive the operator's exploration to zero and watch identification collapse.

    At ``explore = 0`` the action is a deterministic function of the logged wind: the Robinson
    action residual is zero, every arm's moment is ``0 = 0``, and no sample size helps. The adjusted
    arm is reported alongside the naive one because the collapse is a property of the *log*, not of
    the estimator -- adjustment cannot manufacture a comparison the design did not make.
    """
    rows = []
    for explore in explores:
        spec = default_spec()
        log = log_episode(spec, policy="reactive", seed=seed, steps=steps, explore=explore)
        naive = fit_pendulum(log, adjusted=False)
        adjusted = fit_pendulum(log, adjusted=True)
        rows.append(
            {
                "explore": explore,
                "overlap": adjusted.action_residual_variance,
                "naive_gain": naive.gain,
                "adjusted_gain": adjusted.gain,
                "adjusted_error": adjusted.gain_error,
            }
        )
    return rows


def run_compensation_sweep(
    *,
    seed: int = 0,
    steps: int = 4000,
    compensations: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
) -> list[dict[str, float]]:
    """Sweep the operator's compensation gain against :func:`predicted_naive_gain`.

    The falsifiable part of this track: the bias is not a direction, it is a formula, and a formula
    can be wrong. Both the non-monotonicity and the sign-flip interval are predictions made before
    the sweep runs.
    """
    rows = []
    for compensation in compensations:
        spec = default_spec()
        log = log_episode(
            spec, policy="reactive", seed=seed, steps=steps, compensation=compensation
        )
        naive = fit_pendulum(log, adjusted=False)
        adjusted = fit_pendulum(log, adjusted=True)
        predicted = predicted_naive_gain(log)
        rows.append(
            {
                "compensation": compensation,
                "naive_gain": naive.gain,
                "predicted": predicted,
                "gap": abs(naive.gain - predicted),
                "adjusted_gain": adjusted.gain,
            }
        )
    return rows


def run_closed_loop(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    steps: int = 4000,
    control_steps: int = 200,
    policy: str = "reactive",
) -> dict[str, list[dict[str, Any]]]:
    """Regret against the oracle: what a wrong channel costs a controller that believes it.

    The oracle is the environment's own dynamics, so this is regret against the answer rather than
    against the best arm -- the comparison an emulator cannot offer. A refusal
    (:class:`UnusablePlanError`) is recorded as ``refused`` and carries no cost: an arm that cannot
    produce a plan has not produced a bad one, and averaging a refusal in as a large number would
    make the arm look merely expensive.
    """
    arms: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        spec = default_spec()
        log = log_episode(spec, policy=policy, seed=seed, steps=steps)
        train, _ = log.split()
        candidates = {
            "oracle": oracle_fit(spec),
            "naive": fit_pendulum(train, adjusted=False),
            "adjusted": fit_pendulum(train, adjusted=True),
            "flexible-adjusted": fit_pendulum(train, adjusted=True, physics=False),
        }
        for name, fit in candidates.items():
            try:
                episode = run_control_episode(fit, seed=seed, steps=control_steps)
            except UnusablePlanError as refusal:
                episode = {"refused": True, "reason": str(refusal), "gain": fit.gain}
            else:
                episode["refused"] = False
            arms.setdefault(name, []).append(episode)
    return arms


def summarise(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    """Mean, standard error of the mean and range of one column across seeds."""
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    spread = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "mean": float(values.mean()),
        "sem": spread,
        "lo": float(values.min()),
        "hi": float(values.max()),
    }


def track_pendulum_causal(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    steps: int = 4000,
    compensation: float = 0.5,
    explore: float = 0.06,
) -> list[TrackResult]:
    """Score every arm by how far its fitted channel sits from the environment's own constant.

    Unlike the emulator track, the yardstick is not a randomised *design* but the answer, so the
    randomised arms are competitors here rather than the reference. Three things stay falsifiable:
    the confounded arm must land far away, adjustment must bring it back, and adjustment must leave
    the randomised arm where it was.
    """
    case = run_case(seeds=seeds, steps=steps, compensation=compensation, explore=explore)
    return [
        TrackResult(TRACK, name, METRIC, summarise(rows, "gain_error")["mean"])
        for name, rows in sorted(case["arms"].items())
    ]

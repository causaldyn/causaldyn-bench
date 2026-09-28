"""Track O: does an off-policy interval cover a plan's online cost on plants CHC did not write?

:func:`chc.evaluate_plan` estimates what deploying a feedback plan would cost, from another policy's
logs, and returns a nominal 95% interval. The library measures that interval on loops it wrote
itself. This track measures it on two Gymnasium environments, stepped through their own ``step``:
``Pendulum-v1`` about its hanging equilibrium and ``MountainCarContinuous-v0`` about its valley
floor. Neither is linear, both clip, and the evaluator sees them only through logs.

The environments are deterministic, so a white Gaussian disturbance is added to every command
before the step -- a torque on the pendulum, a force on the car -- and a plan's value is an
expectation. **The truth is online**: the plan's average cost over a long run through the same
``step`` with the same disturbance. **The estimate is offline**: each replicate draws a fresh log of
the logging policy, fits a linear plant to it by least squares, and asks ``evaluate_plan``. The
score is the share of replicates whose interval contains the truth.

The design is fixed by rule, the same for both environments, and was chosen through
:func:`chc.certify_evaluation`, which reads no data:

* **disturbance**: sd ``0.15`` of the actuator bound;
* **logger**: velocity damping to a damping ratio of ``0.15`` about the linearisation, a lightly
  damped legacy operator, plus a Gaussian dither of a quarter of the bound. Command plus
  disturbance then has an sd of ``0.29-0.30`` of the bound, and the actuator clips on under 0.1% of
  steps;
* **plans**: LQR on the linearisation, with the state cost in units of the logger's stationary
  spread, ``Q = diag(1 / sd_i^2)``, ``R = rho / bound^2``, and an offset of ``0.1`` of the bound.
  ``rho = 10`` is the *moderate* plan and ``rho = 1`` the *aggressive* one, whose states are much
  narrower than the logs'. Both are deterministic, so the weighted methods evaluate them smoothed
  and subtract the model's correction;
* **stress**: the moderate plan against a logger dithering at ``0.4`` of the bound, which clips on
  about 2% of steps. A clip is outside the class every number of the evaluator is exact for, and
  its certificate cannot see it; this row says what that costs.

Every arm is ``(model, method, model_error)``: the model fitted to the replicate's own log, or the
environment's exact linearisation as a control; ``"mis"``, ``"dr"`` and ``"fqe"``; and
``model_error`` 0 (the model's smoothing correction trusted) or 1 (the default, which carries the
whole correction in the interval and so over-covers by design wherever the model is right).

**The gate** (plans/26, 0.8.0) is read on the nominal design, the fitted model and
``model_error = 0``: coverage within two points of 0.95 over 500 replicates. The default is
reported beside it and is required not to undercover.

Precision: run with ``JAX_ENABLE_X64=1`` (``just track-o``). The cost matrices pass through JAX,
and the truth is scored with the matrices JAX holds, so both sides score the same cost.

Requires the ``gym`` extra (``pip install causaldyn-bench[gym]``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from chc import (
    AffinePolicy,
    InfeasibleEvaluation,
    LinearGaussianPlant,
    QuadraticCost,
    evaluate_plan,
)
from chc.evaluation import EvaluationMethod
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.stats import beta

TRACK = "O-offline-evaluation"
METRIC = "interval_coverage"
NOMINAL = 0.95
TOLERANCE = 0.02  # the gate: within two points of nominal
FLOOR = NOMINAL - TOLERANCE  # the default may over-cover, never under
SEED = 20260928
REPLICATES = 500
TRANSITIONS = 4000
BURN_IN = 500
TRUTH_STEPS = 400_000
TRUTH_BATCHES = 400
DISTURBANCE = 0.15  # of the actuator bound
DAMPING_RATIO = 0.15
DITHER = 0.25  # of the actuator bound
STRESS_DITHER = 0.4
OFFSET = 0.1  # of the actuator bound
PLANS = {"moderate": 10.0, "aggressive": 1.0}  # rho in R = rho / bound^2
DESIGNS = {"nominal": (DITHER, tuple(PLANS)), "stress": (STRESS_DITHER, ("moderate",))}
MODELS = ("fitted", "linearisation")
ARMS: tuple[tuple[EvaluationMethod, float], ...] = (
    ("mis", 0.0),
    ("mis", 1.0),
    ("dr", 0.0),
    ("dr", 1.0),
    ("fqe", 0.0),
)
MOUNTAIN_CAR_GRAVITY = 0.0025  # a literal in the step's source, not an attribute; see the tests


@dataclass(frozen=True)
class Operating:
    """An environment about an equilibrium: its exact linearisation and its state in deviations."""

    name: str
    a: np.ndarray
    b: np.ndarray
    bound: float
    damping: float  # the velocity gain that damps the linearisation to DAMPING_RATIO
    read: Callable[[Any], np.ndarray]
    place: Callable[[Any, np.ndarray], None]
    clipped: Callable[[Any, float], bool]

    def make(self) -> Any:
        env = make_env(self.name)
        env.reset(seed=0)
        return env


def make_env(name: str) -> Any:
    """The unwrapped environment: a time limit would cut the logs."""
    return gym.make(name).unwrapped


def pendulum() -> Operating:
    """``Pendulum-v1`` hanging, ``x = (theta - pi, theta_dot)``, through its semi-implicit Euler
    step ``theta_dot' = theta_dot + (3 g / (2 l) sin(theta) + 3 / (m l^2) u) dt``, ``theta' =
    theta + theta_dot' dt``."""
    with closing(make_env("Pendulum-v1")) as env:
        gravity = 3.0 * env.g / (2.0 * env.l)
        gain = 3.0 / (env.m * env.l**2)
        dt, bound, speed = env.dt, float(env.max_torque), float(env.max_speed)
    a = np.array([[1.0 - gravity * dt * dt, dt], [-gravity * dt, 1.0]])
    b = np.array([[gain * dt * dt], [gain * dt]])

    def read(env: Any) -> np.ndarray:
        theta, theta_dot = env.state
        return np.array([theta - math.pi, theta_dot])

    def place(env: Any, x: np.ndarray) -> None:
        env.state = np.array([math.pi + x[0], x[1]])

    def clipped(env: Any, applied: float) -> bool:
        return abs(applied) > bound or abs(env.state[1]) >= speed

    # continuous time, theta_ddot = -gravity phi - gain kd theta_dot: 2 zeta omega = gain kd
    damping = 2.0 * DAMPING_RATIO * math.sqrt(gravity) / gain
    return Operating("Pendulum-v1", a, b, bound, damping, read, place, clipped)


def mountain_car() -> Operating:
    """``MountainCarContinuous-v0`` at its valley floor ``p* = -pi/6``, ``x = (p - p*, v)``, through
    ``v' = v + power a - 0.0025 cos(3 p)``, ``p' = p + v'``: at ``p*`` the gravity term is zero and
    its slope ``-3 * 0.0025``. The environment keeps its state in float32, and so does the log."""
    with closing(make_env("MountainCarContinuous-v0")) as env:
        power, bound = float(env.power), float(env.max_action)
        speed, low, high = float(env.max_speed), float(env.min_position), float(env.max_position)
    slope = 3.0 * MOUNTAIN_CAR_GRAVITY
    a = np.array([[1.0 - slope, 1.0], [-slope, 1.0]])
    b = np.array([[power], [power]])
    floor = -math.pi / 6.0

    def read(env: Any) -> np.ndarray:
        position, velocity = env.state
        return np.array([float(position) - floor, float(velocity)])

    def place(env: Any, x: np.ndarray) -> None:
        env.state = np.array([floor + x[0], x[1]], dtype=np.float32)

    def clipped(env: Any, applied: float) -> bool:
        position, velocity = float(env.state[0]), float(env.state[1])
        return abs(applied) > bound or abs(velocity) >= speed or not low < position < high

    # per step, omega^2 = slope and the damping adds power kd: 2 zeta omega = power kd
    damping = 2.0 * DAMPING_RATIO * math.sqrt(slope) / power
    return Operating("MountainCarContinuous-v0", a, b, bound, damping, read, place, clipped)


ENVIRONMENTS: dict[str, Callable[[], Operating]] = {
    "pendulum": pendulum,
    "mountain_car": mountain_car,
}


@dataclass(frozen=True)
class Design:
    """One environment's logger, disturbance, linearisation and plans, all set by the module's
    rule."""

    operating: Operating
    disturbance: float  # sd, in actuator units
    logger: AffinePolicy
    plant: LinearGaussianPlant  # the exact linearisation, and the control arm's model
    plans: dict[str, AffinePolicy]
    costs: dict[str, QuadraticCost]


def design(operating: Operating, dither: float = DITHER) -> Design:
    """The logger and the plans for ``operating``. The plans' costs are scaled by the *nominal*
    logger's spread, so the stress design evaluates the same plans as the nominal one."""
    a, b, bound = operating.a, operating.b, operating.bound
    disturbance = DISTURBANCE * bound
    plant = LinearGaussianPlant(a, b, np.zeros(2), disturbance**2 * b @ b.T)
    gain = np.array([[0.0, -operating.damping]])
    nominal = AffinePolicy(gain, np.zeros(1), np.array([[(DITHER * bound) ** 2]]))
    spread = solve_discrete_lyapunov(
        a + b @ nominal.gain, b @ nominal.covariance @ b.T + plant.noise
    )
    q = np.diag(1.0 / np.diag(spread))
    plans, costs = {}, {}
    for name, rho in PLANS.items():
        cost = QuadraticCost(
            Q=jnp.asarray(q),
            R=jnp.asarray([[rho / bound**2]]),
            Qf=jnp.asarray(q),
            x_target=jnp.zeros(2),
        )
        q_held, r_held = np.asarray(cost.Q, np.float64), np.asarray(cost.R, np.float64)
        p = solve_discrete_are(a, b, q_held, r_held)
        k = -np.linalg.solve(r_held + b.T @ p @ b, b.T @ p @ a)
        plans[name] = AffinePolicy(k, np.array([OFFSET * bound]), np.zeros((1, 1)))
        costs[name] = cost
    logger = AffinePolicy(gain, np.zeros(1), np.array([[(dither * bound) ** 2]]))
    return Design(operating, disturbance, logger, plant, plans, costs)


def run_policy(
    operating: Operating,
    env: Any,
    policy: AffinePolicy,
    disturbance: float,
    steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """``steps`` transitions of ``policy`` through the environment's own ``step``, from the
    equilibrium: states ``(steps + 1, 2)`` in deviations, the commands ``(steps, 1)`` the policy
    chose, and the number of steps on which the environment clipped something."""
    operating.place(env, np.zeros(2))
    x, u = np.empty((steps + 1, 2)), np.empty((steps, 1))
    x[0] = operating.read(env)
    gain, offset = policy.gain[0], float(policy.offset[0])
    dither = rng.standard_normal(steps) * math.sqrt(float(policy.covariance[0, 0]))
    shock = rng.standard_normal(steps) * disturbance
    clips = 0
    for t in range(steps):
        command = float(gain @ x[t]) + offset + dither[t]
        applied = command + shock[t]
        env.step(np.array([applied]))
        clips += operating.clipped(env, applied)
        x[t + 1] = operating.read(env)
        u[t, 0] = command
    return x, u, clips


def stage_cost(x: np.ndarray, u: np.ndarray, cost: QuadraticCost) -> np.ndarray:
    """``cost.running`` against its zero target, ``(x' Q x + u' R u) / 2``, one per row."""
    q, r = np.asarray(cost.Q, np.float64), np.asarray(cost.R, np.float64)
    return 0.5 * (np.einsum("ti,ij,tj->t", x, q, x) + np.einsum("ti,ij,tj->t", u, r, u))


def linearised_value(spec: Design, plan: str) -> float:
    """The plan's average cost on the linearisation, in closed form: what the online truth would be
    if the environment were its linearisation and never clipped."""
    policy, plant = spec.plans[plan], spec.plant
    closed = plant.a + plant.b @ policy.gain
    sx = solve_discrete_lyapunov(closed, plant.noise)
    mx = np.linalg.solve(np.eye(2) - closed, plant.b @ policy.offset)
    mu, su = policy.gain @ mx + policy.offset, policy.gain @ sx @ policy.gain.T
    q = np.asarray(spec.costs[plan].Q, np.float64)
    r = np.asarray(spec.costs[plan].R, np.float64)
    return 0.5 * float(np.trace(q @ sx) + mx @ q @ mx + np.trace(r @ su) + mu @ r @ mu)


@dataclass(frozen=True)
class Truth:
    """A plan's online average cost, the batch-means standard error of that run, and its clips."""

    value: float
    se: float
    clip_share: float
    linearised: float


def online_truth(spec: Design, plan: str, seed: Sequence[int], steps: int = TRUTH_STEPS) -> Truth:
    rng = np.random.default_rng(list(seed))
    operating = spec.operating
    with closing(operating.make()) as env:
        x, u, clips = run_policy(
            operating, env, spec.plans[plan], spec.disturbance, BURN_IN + steps, rng
        )
    c = stage_cost(x[BURN_IN:-1], u[BURN_IN:], spec.costs[plan])
    batches = c[: c.size - c.size % TRUTH_BATCHES].reshape(TRUTH_BATCHES, -1).mean(axis=1)
    return Truth(
        float(c.mean()),
        float(batches.std(ddof=1) / math.sqrt(TRUTH_BATCHES)),
        clips / (BURN_IN + steps),
        linearised_value(spec, plan),
    )


def fitted_plant(x: np.ndarray, u: np.ndarray) -> LinearGaussianPlant:
    """Least squares of ``x'`` on ``(x, u, 1)``, with the residual covariance as the noise."""
    features = np.hstack([x[:-1], u, np.ones((u.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(features, x[1:], rcond=None)
    residual = x[1:] - features @ coef
    noise = residual.T @ residual / (u.shape[0] - features.shape[1])
    return LinearGaussianPlant(coef[:2].T, coef[2:3].T, coef[3], 0.5 * (noise + noise.T))


@dataclass(frozen=True)
class Interval:
    value: float
    low: float
    high: float
    model_share: float
    degrees_of_freedom: float | None


Estimate = Interval | str  # the interval, or the certificate's reason for refusing
ArmKey = tuple[str, str, EvaluationMethod, float]  # plan, model, method, model_error


def evaluate_log(
    x: np.ndarray, u: np.ndarray, spec: Design, plans: Sequence[str]
) -> dict[ArmKey, Estimate]:
    """Every arm of every plan in ``plans`` on one log."""
    out: dict[ArmKey, Estimate] = {}
    models = {"fitted": fitted_plant(x, u), "linearisation": spec.plant}
    for plan in plans:
        for model in MODELS:
            for method, model_error in ARMS:
                try:
                    result = evaluate_plan(
                        {"x": x, "u": u},
                        spec.plans[plan],
                        method,
                        plant=models[model],
                        cost=spec.costs[plan],
                        model_error=model_error,
                    )
                except InfeasibleEvaluation as refusal:
                    out[plan, model, method, model_error] = refusal.certificate.reason
                    continue
                out[plan, model, method, model_error] = Interval(
                    result.value,
                    result.interval[0],
                    result.interval[1],
                    result.model_share,
                    result.degrees_of_freedom,
                )
    return out


def clopper_pearson(covered: int, n: int, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    low = float(beta.ppf(tail, covered, n - covered + 1)) if covered > 0 else 0.0
    high = float(beta.ppf(1.0 - tail, covered + 1, n - covered)) if covered < n else 1.0
    return low, high


@dataclass(frozen=True)
class Scores:
    """What the intervals of one arm did, over the replicates the certificate passed."""

    coverage: float
    coverage_interval: tuple[float, float]  # Clopper-Pearson, 95%
    above_truth: float  # share of intervals wholly above the truth
    below_truth: float
    mean_error: float
    mean_error_se: float
    error_sd: float
    mean_half_width: float
    median_model_share: float
    median_degrees_of_freedom: float | None


@dataclass(frozen=True)
class Arm:
    design: str
    plan: str
    model: str
    method: EvaluationMethod
    model_error: float
    scored: int
    refusals: dict[str, int]
    scores: Scores | None  # None when the certificate passed fewer than two replicates


def score(estimates: Sequence[Estimate], truth: float) -> tuple[int, dict[str, int], Scores | None]:
    intervals = [e for e in estimates if isinstance(e, Interval)]
    refusals = dict(Counter(e for e in estimates if isinstance(e, str)))
    n = len(intervals)
    if n < 2:
        return n, refusals, None
    value = np.array([e.value for e in intervals])
    low = np.array([e.low for e in intervals])
    high = np.array([e.high for e in intervals])
    above, below = low > truth, high < truth
    covered = int(np.sum(~above & ~below))
    errors = value - truth
    dofs = [e.degrees_of_freedom for e in intervals if e.degrees_of_freedom is not None]
    return (
        n,
        refusals,
        Scores(
            coverage=covered / n,
            coverage_interval=clopper_pearson(covered, n),
            above_truth=float(above.mean()),
            below_truth=float(below.mean()),
            mean_error=float(errors.mean()),
            mean_error_se=float(errors.std(ddof=1) / math.sqrt(n)),
            error_sd=float(errors.std(ddof=1)),
            mean_half_width=float(np.mean(high - low) / 2.0),
            median_model_share=float(np.median([e.model_share for e in intervals])),
            median_degrees_of_freedom=float(np.median(dofs)) if dofs else None,
        ),
    )


def gate(arm: Arm) -> bool | None:
    """The 0.8.0 reading: the trusted correction within two points of nominal, the default never
    below that. ``None`` for the arms the gate does not read; a refused arm fails it."""
    if arm.design != "nominal" or arm.model != "fitted":
        return None
    if arm.scores is None:
        return False
    if arm.model_error == 0.0:
        return abs(arm.scores.coverage - NOMINAL) <= TOLERANCE
    return arm.scores.coverage >= FLOOR


@dataclass(frozen=True)
class EnvironmentRun:
    name: str
    damping: float
    bound: float
    disturbance: float
    plans: dict[str, dict[str, list]]  # gain and offset, per plan
    truths: dict[str, Truth]
    log_clip_share: dict[str, tuple[float, float]]  # per design: mean and max over replicates
    arms: list[Arm]
    seconds: float


def run_environment(
    key: str,
    replicates: int = REPLICATES,
    transitions: int = TRANSITIONS,
    truth_steps: int = TRUTH_STEPS,
    progress: Callable[[str], None] | None = None,
) -> EnvironmentRun:
    """Track O on one environment: the online truths, then ``replicates`` logs per design with
    every arm on each. Replicate ``r`` of a design draws from ``(SEED, env, design, r)``, and the
    plans of a design are evaluated on the same logs."""
    start = time.perf_counter()
    operating = ENVIRONMENTS[key]()
    env_index = list(ENVIRONMENTS).index(key)
    specs = {name: design(operating, dither) for name, (dither, _) in DESIGNS.items()}
    nominal = specs["nominal"]
    truths = {
        plan: online_truth(nominal, plan, (SEED, env_index, 99, i), truth_steps)
        for i, plan in enumerate(PLANS)
    }
    arms, clip_share = [], {}
    with closing(operating.make()) as env:
        for d, (design_name, (_, plans)) in enumerate(DESIGNS.items()):
            spec = specs[design_name]
            estimates: dict[ArmKey, list[Estimate]] = {}
            shares = []
            for r in range(replicates):
                rng = np.random.default_rng([SEED, env_index, d, r])
                x, u, clips = run_policy(
                    operating, env, spec.logger, spec.disturbance, BURN_IN + transitions, rng
                )
                shares.append(clips / (BURN_IN + transitions))
                for arm_key, estimate in evaluate_log(
                    x[BURN_IN:], u[BURN_IN:], spec, plans
                ).items():
                    estimates.setdefault(arm_key, []).append(estimate)
                if progress is not None and (r + 1) % 50 == 0:
                    progress(f"{key} {design_name} {r + 1}/{replicates}")
            clip_share[design_name] = (float(np.mean(shares)), float(np.max(shares)))
            for (plan, model, method, model_error), found in estimates.items():
                scored, refusals, scores = score(found, truths[plan].value)
                arms.append(
                    Arm(design_name, plan, model, method, model_error, scored, refusals, scores)
                )
    return EnvironmentRun(
        name=operating.name,
        damping=operating.damping,
        bound=operating.bound,
        disturbance=nominal.disturbance,
        plans={
            name: {"gain": policy.gain.tolist(), "offset": policy.offset.tolist()}
            for name, policy in nominal.plans.items()
        },
        truths=truths,
        log_clip_share=clip_share,
        arms=arms,
        seconds=time.perf_counter() - start,
    )


def _markdown(runs: Sequence[EnvironmentRun], replicates: int, x64: bool) -> str:
    lines = [
        "# Track O — off-policy interval coverage on plants CHC did not write",
        "",
        f"{replicates} replicates of {TRANSITIONS} logged transitions after a {BURN_IN}-step "
        f"burn-in, nominal {NOMINAL}, each environment stepped through its own `step` "
        f"(`causaldyn_bench.ope_calibration`, x64 {x64}). The truth is each plan's online average "
        f"cost over {TRUTH_STEPS} steps. Coverage intervals are Clopper-Pearson's.",
        "",
        "## The gate",
        "",
        "Nominal design, fitted model. `model_error = 0` must be within two points of "
        f"{NOMINAL}; the default `model_error = 1` must not fall below {FLOOR:.2f}.",
        "",
        "| environment | plan | method | `model_error` | coverage | 95% interval | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        for arm in run.arms:
            verdict = gate(arm)
            if verdict is None:
                continue
            if arm.scores is None:
                lines.append(
                    f"| {run.name} | {arm.plan} | `{arm.method}` | {arm.model_error:g} | "
                    "refused | | **no** |"
                )
                continue
            low, high = arm.scores.coverage_interval
            lines.append(
                f"| {run.name} | {arm.plan} | `{arm.method}` | {arm.model_error:g} | "
                f"{arm.scores.coverage:.3f} | [{low:.3f}, {high:.3f}] | "
                f"{'yes' if verdict else '**no**'} |"
            )
    for run in runs:
        clips = ", ".join(
            f"{name} {100 * mean:.2f}% / {100 * peak:.2f}%"
            for name, (mean, peak) in run.log_clip_share.items()
        )
        lines += [
            "",
            f"## {run.name}",
            "",
            f"Actuator bound {run.bound:g}, disturbance sd {run.disturbance:g}, logger velocity "
            f"gain {run.damping:.4g}. Steps clipped in the logs, mean / max over replicates: "
            f"{clips}.",
            "",
            "| plan | online truth | its SE | linearised | steps clipped |",
            "|---|---|---|---|---|",
        ]
        for plan, truth in run.truths.items():
            lines.append(
                f"| {plan} | {truth.value:.6g} | {truth.se:.2g} | {truth.linearised:.6g} | "
                f"{100 * truth.clip_share:.3f}% |"
            )
        lines += [
            "",
            "| design | plan | model | method | `model_error` | scored | coverage | above / below "
            "| mean error ± SE | half-width | model share | dof |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for arm in run.arms:
            head = (
                f"| {arm.design} | {arm.plan} | {arm.model} | `{arm.method}` | "
                f"{arm.model_error:g} | {arm.scored}"
            )
            if arm.refusals:
                head += f" ({sum(arm.refusals.values())} refused)"
            s = arm.scores
            if s is None:
                lines.append(head + " | — | — | — | — | — | — |")
                continue
            dof = (
                "—" if s.median_degrees_of_freedom is None else f"{s.median_degrees_of_freedom:.1f}"
            )
            lines.append(
                f"{head} | {s.coverage:.3f} | {s.above_truth:.3f} / {s.below_truth:.3f} | "
                f"{s.mean_error:.3g} ± {s.mean_error_se:.2g} | {s.mean_half_width:.3g} | "
                f"{s.median_model_share:.2f} | {dof} |"
            )
    lines += [
        "",
        "Wall-clock per environment is recorded in `track_o.json` and quoted nowhere: it was not "
        "measured on a clean machine.",
    ]
    return "\n".join(lines) + "\n"


def _record(runs: Sequence[EnvironmentRun], replicates: int, x64: bool) -> dict[str, Any]:
    return {
        "track": TRACK,
        "metric": METRIC,
        "replicates": replicates,
        "transitions": TRANSITIONS,
        "burn_in": BURN_IN,
        "truth_steps": TRUTH_STEPS,
        "seed": SEED,
        "x64": x64,
        "gymnasium": gym.__version__,
        "rule": {
            "disturbance": DISTURBANCE,
            "damping_ratio": DAMPING_RATIO,
            "dither": DITHER,
            "stress_dither": STRESS_DITHER,
            "offset": OFFSET,
            "rho": PLANS,
        },
        "wall_clock": "environments[*].seconds; recorded, not quoted: not measured on a clean "
        "machine",
        "environments": [
            {**asdict(run), "arms": [{**asdict(arm), "gate": gate(arm)} for arm in run.arms]}
            for run in runs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument(
        "--environments", nargs="+", choices=list(ENVIRONMENTS), default=list(ENVIRONMENTS)
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    x64 = bool(jnp.zeros(()).dtype == jnp.float64)
    runs = [
        run_environment(key, args.replicates, progress=lambda line: print(line, flush=True))
        for key in args.environments
    ]
    text = _markdown(runs, args.replicates, x64)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "track_o.md").write_text(text)
    (args.out / "track_o.json").write_text(
        json.dumps(_record(runs, args.replicates, x64), indent=1) + "\n"
    )
    print(text)
    print(f"written to {args.out}/track_o.md and {args.out}/track_o.json")


if __name__ == "__main__":
    main()

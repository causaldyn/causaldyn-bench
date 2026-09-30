"""Track Q: the analyst's graph is wrong. Does the logger check turn the error into a flag before a
plan or its evaluation fails silently?

:func:`chc.prescribe` derives its adjustment set from the caller's graph, and
:meth:`chc.Prescription.evaluate` weights a later panel on the premise that the levers read the
state and their recorded parents alone. Neither can tell a wrong graph from a right one. The logger
check (ADR 0028, *experimental*) tests the one implication of the graph both rest on that a panel
can falsify, the levers' local Markov condition, and warns. Its gate (plans/27, GR2): **a flag that
does not lower the silent-failure rate is not shipped.**

The worlds are the lifecycle market of the library's ``tests/test_lifecycle_loop.py``: 400 units,
twelve periods, supply and wait moved by an incentive and by demand, the incentive drawn afresh at
sd 1.5, and the decision the test makes (supply to 0.3 over three periods, the incentive in
``[-2, 2]`` at 0.05 a unit, wait at most 0.5), with a tolerance of 0.05, a sixth of the target,
where the test's 0.5 accepts any error these worlds can make. Each world adds one thing the graph
handed to ``prescribe`` gets wrong:

* **none**: the control; the graph is right.
* **chase**: the logger chases demand, and the graph omits that edge. Demand moves supply, so the
  adjustment set holds it anyway: an omitted logger input that costs the fit nothing.
* **sticky**: the logger keeps half its last incentive. No graph over the columns states that, so
  this world has no oracle; the fit is unbiased, the evaluation's weights are not.
* **confounder**, **confounder AR**: a promotion the logger reads and that moves supply, a column
  of the panel the graph leaves out, drawn afresh each period or with persistence 0.8.
* **mediator**: orders carry most of the incentive's push, and a latent moves both orders and
  supply; the graph turns the edge around, so the fit adjusts for orders. An orientation inside
  one Markov equivalence class.
* **collider observed**, **collider latent**: sessions are moved by the incentive and by demand, or
  by a latent that also moves supply, and the graph makes them a parent of the incentive.
* **non-ancestor**: the graph says the incentive does not move supply at all.

Three arms plan on the same panel: the true graph (the **oracle**), the **wrong** one, and an
adjustment set **chosen by fit**, the subset of the panel's other columns with the best adjusted
R^2 for next period's supply (Lopez de Prado's warning: a confounded mediator fits better). Each is
read as it is, and with the logger check read as a **stop**: a flag on the fitted panel refuses the
plan, a flag on the later panel refuses its evaluation.

A **silent failure** is a claim the library made that the true plant breaks, with nothing refused.
A plan the certificate trusts for at least one step claims two things:

* that its true mean path stays within the tolerance of the planned trajectory over the steps the
  certificate trusts (``trustworthy_steps``); it fails when the path **leaves** it;
* that it is within its regret bound of the best schedule in the box. The bound prices the planning
  objective only (Result 69), so the plan fails when it **loses** more on the true plant than the
  bound plus ``MATERIAL`` of the stakes, doing nothing's regret. A plan that believes its lever
  does nothing can keep the first claim, having predicted that nothing would happen, and fail the
  second.

An evaluation interval from a later panel of the same world claims the schedule's expected cost,
and fails when it **misses** it. All three are computed exactly on the linear plant, the plan's
from its own start. Reported beside them: the **headroom**, the share of the best schedule's
improvement over doing nothing that a plan buys on the true plant (Track H's measure), and how often
the check flags under each graph, which on the oracle is its size on these worlds (the 0.9.0 gate:
within two points of 5%).

By construction, the worlds are CHC's (R19): the track measures whether the check sees what these
graph errors leave in a panel, not how common the errors are.

Precision: run with ``JAX_ENABLE_X64=1`` (``just track-q``), the lifecycle test's precision.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import multiprocessing
import os
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chc
import jax
import jax.numpy as jnp
import numpy as np
from chc.dynamics import LinearDynamics
from chc.integrate import rollout
from numpy.typing import NDArray
from scipy.optimize import lsq_linear
from scipy.stats import fisher_exact

from causaldyn_bench.ope_calibration import clopper_pearson

TRACK = "Q-graph-errors"
METRIC = "silent_failure_share"
SEED = 20260930
REPLICATES = 200
UNITS = 400
PERIODS = 12
DT = 0.1
HORIZON = 3
TARGET = 0.3
UNIT_COST = 0.05
TOLERANCE = 0.05
ALPHA = 0.05  # the logger check's level, as the library reads it
MATERIAL = 0.1  # a plan whose regret passes its bound by this share of the stakes has failed
STEP = np.array([[0.94, 0.03], [0.0, 1.025]])  # (supply, wait), one period
DIRECT = np.array([0.08, -0.04])  # the incentive's push on (supply, wait) per period, in all
DEMAND_PUSH = 0.15
SHOCK = 0.01
SPREAD = 1.5  # the incentive's own sd
STATES = ("supply", "wait")
LEVER = "incentive"
EDGES = (("demand", "supply"), ("incentive", "supply"), ("incentive", "wait"), ("supply", "wait"))
ARMS = ("oracle", "wrong", "by fit")
MODES = ("as is", "stop")


@dataclass(frozen=True)
class Confounder:
    """A column the logger reads and that pushes supply, AR(1) at ``persistence``, unit variance."""

    read: float
    push: float
    persistence: float
    column: str = "promo"


@dataclass(frozen=True)
class Mediator:
    """Orders are the incentive plus a latent plus noise of sd ``noise``; they push supply by
    ``through`` and the latent by ``shared`` more. The incentive's direct push on supply is what is
    left of ``DIRECT``, so its total effect is the same as in every other world."""

    through: float
    shared: float
    noise: float
    column: str = "orders"


@dataclass(frozen=True)
class Collider:
    """Sessions are the incentive plus ``other`` plus noise of sd ``noise``. They push nothing. The
    other is demand, or a latent of unit variance that pushes supply by ``push``."""

    other: str  # "demand" or "latent"
    push: float
    noise: float
    column: str = "sessions"


@dataclass(frozen=True)
class World:
    name: str
    error: str  # what the wrong graph gets wrong
    true_edges: tuple[tuple[str, str], ...]
    wrong_edges: tuple[tuple[str, str], ...]
    true_latent: tuple[str, ...] = ()
    chase: float = 0.0
    sticky: float = 0.0
    confounder: Confounder | None = None
    mediator: Mediator | None = None
    collider: Collider | None = None
    oracle: bool = True  # False where no graph over the columns states the logger


def _worlds() -> dict[str, World]:
    edges = list(EDGES)
    promo = [("promo", "incentive"), ("promo", "supply")]
    worlds = [
        World("none", "is the true graph", EDGES, EDGES),
        World(
            "chase",
            "omits the logger's reading of demand",
            (*EDGES, ("demand", "incentive")),
            EDGES,
            chase=1.0,
        ),
        World(
            "sticky",
            "cannot state that the logger keeps half its last incentive",
            EDGES,
            EDGES,
            sticky=0.5,
            oracle=False,
        ),
        World(
            "confounder",
            "omits a promotion the logger reads and that moves supply",
            (*EDGES, *promo),
            EDGES,
            confounder=Confounder(read=1.0, push=0.2, persistence=0.0),
        ),
        World(
            "confounder AR",
            "omits the same promotion, persistent",
            (*EDGES, *promo),
            EDGES,
            confounder=Confounder(read=1.0, push=0.2, persistence=0.8),
        ),
        World(
            "mediator",
            "turns incentive -> orders around, so the fit adjusts for a confounded mediator",
            (
                *EDGES,
                ("incentive", "orders"),
                ("orders", "supply"),
                ("h", "orders"),
                ("h", "supply"),
            ),
            (*EDGES, ("orders", "incentive"), ("orders", "supply")),
            true_latent=("h",),
            mediator=Mediator(through=0.06, shared=0.1, noise=1.0),
        ),
        World(
            "collider observed",
            "makes sessions, moved by the incentive and demand, a parent of the incentive",
            (*EDGES, ("incentive", "sessions"), ("demand", "sessions")),
            (*EDGES, ("sessions", "incentive"), ("demand", "sessions")),
            collider=Collider(other="demand", push=0.0, noise=0.5),
        ),
        World(
            "collider latent",
            "makes sessions, moved by the incentive and a latent that moves supply, a parent",
            (*EDGES, ("incentive", "sessions"), ("v", "sessions"), ("v", "supply")),
            (*EDGES, ("sessions", "incentive")),
            true_latent=("v",),
            collider=Collider(other="latent", push=0.1, noise=0.5),
        ),
        World(
            "non-ancestor",
            "says the incentive does not move supply",
            EDGES,
            tuple(edge for edge in edges if edge != ("incentive", "supply")),
        ),
    ]
    return {world.name: world for world in worlds}


WORLDS = _worlds()


# ---------------------------------------------------------------------------------------- panels


@dataclass(frozen=True)
class Draw:
    columns: dict[str, NDArray[np.float64]]  # the panel, long format
    confounder_last: float | None  # the confounder's mean over units at the last period


def draw(world: World, seed: Sequence[int], units: int = UNITS) -> Draw:
    """``units`` units of ``PERIODS`` periods, period-major: every column the panel holds, and none
    of the latents."""
    rng = np.random.default_rng(list(seed))
    direct = DIRECT.copy()
    if world.mediator is not None:
        direct[0] -= world.mediator.through
    x = rng.normal(0.0, 0.2, (units, 2))
    incentive = rng.normal(0.0, SPREAD, units)
    z = rng.normal(size=units)
    names = ["unit", "time", *STATES, LEVER, "demand"]
    for extra in (world.confounder, world.mediator, world.collider):
        if extra is not None:
            names.append(extra.column)
    rows: dict[str, list[NDArray[np.float64]]] = {name: [] for name in names}
    for period in range(PERIODS):
        demand = rng.normal(size=units)
        fresh = rng.normal(0.0, SPREAD, units)
        push = DEMAND_PUSH * demand
        incentive = (
            world.sticky * incentive
            + math.sqrt(1.0 - world.sticky**2) * fresh
            + world.chase * demand
        )
        now: dict[str, NDArray[np.float64]] = {}
        if (confounder := world.confounder) is not None:
            shock = rng.normal(size=units)
            z = confounder.persistence * z + math.sqrt(1.0 - confounder.persistence**2) * shock
            incentive = incentive + confounder.read * z
            push = push + confounder.push * z
            now[confounder.column] = z
        if (mediator := world.mediator) is not None:
            latent = rng.normal(size=units)
            orders = incentive + latent + mediator.noise * rng.normal(size=units)
            push = push + mediator.through * orders + mediator.shared * latent
            now[mediator.column] = orders
        if (collider := world.collider) is not None:
            if collider.other == "demand":
                other = demand
            else:
                other = rng.normal(size=units)
                push = push + collider.push * other
            now[collider.column] = incentive + other + collider.noise * rng.normal(size=units)
        now |= {
            "unit": np.arange(units, dtype=float),
            "time": np.full(units, float(period)),
            "supply": x[:, 0],
            "wait": x[:, 1],
            LEVER: incentive,
            "demand": demand,
        }
        for name in names:
            rows[name].append(now[name])
        x = (
            x @ STEP.T
            + np.outer(incentive, direct)
            + np.outer(push, [1.0, 0.0])
            + SHOCK * rng.normal(size=(units, 2))
        )
    return Draw(
        columns={name: np.concatenate(values) for name, values in rows.items()},
        confounder_last=None if world.confounder is None else float(z.mean()),
    )


# ----------------------------------------------------------------------------------- the plant


@dataclass(frozen=True)
class Plant:
    """The world under an open-loop schedule, which reads nothing: ``s' = a s + b u + noise``, where
    ``s`` is (supply, wait), with the confounder appended when the world has one."""

    a: NDArray[np.float64]
    b: NDArray[np.float64]
    noise: NDArray[np.float64]


def plant(world: World) -> Plant:
    noise = SHOCK**2 * np.eye(2)
    noise[0, 0] += DEMAND_PUSH**2
    if (mediator := world.mediator) is not None:
        noise[0, 0] += (mediator.through + mediator.shared) ** 2 + (
            mediator.through * mediator.noise
        ) ** 2
    if (collider := world.collider) is not None and collider.other == "latent":
        noise[0, 0] += collider.push**2
    if (confounder := world.confounder) is None:
        return Plant(STEP.copy(), DIRECT.copy(), noise)
    a = np.zeros((3, 3))
    a[:2, :2] = STEP
    a[0, 2] = confounder.push
    a[2, 2] = confounder.persistence
    grown = np.zeros((3, 3))
    grown[:2, :2] = noise
    grown[2, 2] = 1.0 - confounder.persistence**2
    return Plant(a, np.append(DIRECT, 0.0), grown)


def mean_path(
    system: Plant, start: NDArray[np.float64], actions: NDArray[np.float64]
) -> NDArray[np.float64]:
    path = [np.asarray(start, dtype=float)]
    for action in actions:
        path.append(system.a @ path[-1] + system.b * action)
    return np.array(path)


def planning_cost(system: Plant, start: NDArray[np.float64], actions: NDArray[np.float64]) -> float:
    """The decision's own cost, running and terminal, on the plant's mean path. Its variance terms
    do not depend on the schedule, so they cancel in every difference taken of it."""
    supply = mean_path(system, start, actions)[:, 0]
    running = 0.5 * np.sum((supply[:-1] - TARGET) ** 2) + 0.5 * UNIT_COST * np.sum(actions**2)
    return float(running + 0.5 * (supply[-1] - TARGET) ** 2)


def best_cost(system: Plant, start: NDArray[np.float64]) -> float:
    """The best schedule in the box, exactly: the supply path is affine in the actions, so the cost
    is a bounded least-squares problem."""
    free = mean_path(system, start, np.zeros(HORIZON))[:, 0]
    reach = np.column_stack(
        [mean_path(system, np.zeros_like(start), np.eye(HORIZON)[j])[:, 0] for j in range(HORIZON)]
    )
    design = np.vstack([reach[1:], math.sqrt(UNIT_COST) * np.eye(HORIZON)])
    wanted = np.concatenate([TARGET - free[1:], np.zeros(HORIZON)])
    solved = lsq_linear(design, wanted, bounds=(-2.0, 2.0), tol=1e-12)
    return planning_cost(system, start, solved.x)


def regret(
    system: Plant, start: NDArray[np.float64], actions: NDArray[np.float64]
) -> tuple[float, float]:
    """``actions``' cost on the true plant less the best schedule's, and the stakes: doing nothing's
    cost less the best schedule's. The headroom a plan buys is one less their ratio."""
    best = best_cost(system, start)
    return (
        planning_cost(system, start, actions) - best,
        planning_cost(system, start, np.zeros(HORIZON)) - best,
    )


def expected_running_cost(
    system: Plant, starts: NDArray[np.float64], actions: NDArray[np.float64]
) -> float:
    """What :meth:`chc.Prescription.evaluate` estimates: the schedule's running cost, the terminal
    term left out, from the law of ``starts``, by mean and covariance propagation."""
    mean, covariance, total = starts.mean(axis=0), np.cov(starts, rowvar=False), 0.0
    for action in actions:
        total += 0.5 * (covariance[0, 0] + (mean[0] - TARGET) ** 2 + UNIT_COST * action**2)
        mean = system.a @ mean + system.b * action
        covariance = system.a @ covariance @ system.a.T + system.noise
    return float(total)


def episode_starts(world: World, later: Draw) -> NDArray[np.float64]:
    """The first state of every window ``evaluate`` cuts from a unit, with the confounder at it."""
    columns = later.columns
    names = [*STATES] + ([world.confounder.column] if world.confounder is not None else [])
    period = columns["time"]
    first = np.isin(period, np.arange(PERIODS - 1 - HORIZON, -1, -HORIZON))
    return np.column_stack([columns[name][first] for name in names])


# ------------------------------------------------------------------------------------ the arms


def prescribe(panel: chc.Panel, adjustment: chc.CausalGraph | Sequence[str]) -> chc.Prescription:
    return chc.prescribe(
        panel,
        levers=[chc.Lever(LEVER, lo=-2.0, hi=2.0, unit_cost=UNIT_COST)],
        target=chc.Target("supply", value=TARGET),
        constraints=[chc.Constraint("wait", hi=0.5)],
        adjustment=adjustment,
        horizon=HORIZON,
        dt=DT,
        tolerance=TOLERANCE,
    )


def by_fit(columns: dict[str, NDArray[np.float64]]) -> tuple[str, ...]:
    """The subset of the panel's other columns whose one-period regression of supply on the state,
    the incentive and the subset has the best adjusted R^2."""
    candidates = [name for name in columns if name not in ("unit", "time", *STATES, LEVER)]
    order = np.lexsort((columns["time"], columns["unit"]))
    same = columns["unit"][order][1:] == columns["unit"][order][:-1]
    now, then = order[:-1][same], order[1:][same]
    outcome = columns["supply"][then]
    spread = np.sum((outcome - outcome.mean()) ** 2)
    best: tuple[tuple[str, ...], float] = ((), -math.inf)
    for size in range(len(candidates) + 1):
        for subset in itertools.combinations(candidates, size):
            design = np.column_stack(
                [np.ones(now.size), *(columns[name][now] for name in (*STATES, LEVER, *subset))]
            )
            coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
            residual = outcome - design @ coefficients
            rows, width = design.shape
            adjusted = 1.0 - (residual @ residual / spread) * (rows - 1) / (rows - width)
            if adjusted > best[1]:
                best = (subset, adjusted)
    return best[0]


@dataclass(frozen=True)
class ArmRun:
    """One arm on one replicate. The flags are None where the check was not run."""

    identified: bool
    covariates: tuple[str, ...]
    trustworthy_steps: int
    fit_flag: bool | None
    channel: float | None  # the plan's model: one period's push on supply per unit of incentive
    # per step of the plan, how far the true mean path of the state is from the planned trajectory
    deviation: tuple[float, ...] | None
    regret: float | None  # on the true plant, from the plan's start
    stakes: float | None  # doing nothing's regret
    regret_bound: float | None  # the certificate's, on the planning objective
    evaluation: str  # "evaluated", "out of scope", "refused" or "no plan"
    covers: bool | None
    later_flag: bool | None


def _flag(check: chc.evaluation.LoggerCheck | None) -> bool | None:
    return None if check is None else bool(check.test.p_value <= ALPHA)


def _one_period_push(prescription: chc.Prescription) -> float:
    plan = prescription.plan
    assert plan is not None
    model = chc.HybridDynamics(
        known=LinearDynamics(jnp.zeros((2, 2)), jnp.zeros((2, 1))),
        residual=prescription.model_fit.residual,
    )
    start = plan.trajectory[0]
    # Eagerly: every fit is a new static node of the model's pytree, so a jitted rollout compiles,
    # and keeps, a program per fit, and a worker's memory grew without bound.
    with jax.disable_jit():
        pushed = rollout(model, start, jnp.ones((1, 1)), DT)[1, 0]
        idle = rollout(model, start, jnp.zeros((1, 1)), DT)[1, 0]
    return float(pushed - idle)


def run_arm(
    world: World,
    adjustment: chc.CausalGraph | Sequence[str],
    fitted: Draw,
    later: Draw,
) -> ArmRun:
    panel = chc.Panel.from_frame(fitted.columns, unit="unit", time="time", seed=0)
    prescription = prescribe(panel, adjustment)
    certificate = prescription.certificate
    covariates = tuple(certificate.adjustment.covariates)
    fit_flag = _flag(prescription.logger_check)
    plan = prescription.plan
    if plan is None:
        return ArmRun(
            False, covariates, 0, fit_flag, None, None, None, None, None, "no plan", None, None
        )
    system = plant(world)
    actions = np.asarray(plan.actions, dtype=float)[:, 0]
    planned = np.asarray(plan.trajectory, dtype=float)
    start = planned[0]
    if fitted.confounder_last is not None:
        start = np.append(start, fitted.confounder_last)
    true_path = mean_path(system, start, actions)[:, : len(STATES)]
    lost, stakes = regret(system, start, actions)
    run = {
        "identified": True,
        "covariates": covariates,
        "trustworthy_steps": certificate.trustworthy_steps,
        "fit_flag": fit_flag,
        "channel": _one_period_push(prescription),
        "deviation": tuple(float(d) for d in np.linalg.norm(true_path - planned, axis=1)),
        "regret": lost,
        "stakes": stakes,
        "regret_bound": certificate.regret_bound,
    }
    later_panel = chc.Panel.from_frame(later.columns, unit="unit", time="time", seed=0)
    try:
        evaluation = prescription.evaluate(later_panel)
    except chc.InfeasibleEvaluation:
        return ArmRun(**run, evaluation="refused", covers=None, later_flag=None)
    except chc.DecisionError:
        return ArmRun(**run, evaluation="out of scope", covers=None, later_flag=None)
    truth = expected_running_cost(system, episode_starts(world, later), actions)
    low, high = evaluation.interval
    return ArmRun(
        **run,
        evaluation="evaluated",
        covers=bool(low <= truth <= high),
        later_flag=_flag(evaluation.logger_check),
    )


def replicate(job: tuple[str, int]) -> dict[str, dict[str, Any]]:
    """Every arm on replicate ``index`` of ``world``: the panel it plans on draws from ``(SEED,
    world, index, 0)``, the later panel it is evaluated on from ``(SEED, world, index, 1)``."""
    logging.getLogger("chc").setLevel(logging.ERROR)  # a warning per flagged panel is the data here
    name, index = job
    world = WORLDS[name]
    number = list(WORLDS).index(name)
    fitted = draw(world, (SEED, number, index, 0))
    later = draw(world, (SEED, number, index, 1))
    adjustments: dict[str, chc.CausalGraph | Sequence[str]] = {
        "oracle": chc.CausalGraph.from_edges(world.true_edges, latent=world.true_latent),
        "wrong": chc.CausalGraph.from_edges(world.wrong_edges),
        "by fit": by_fit(fitted.columns),
    }
    return {
        arm: asdict(run_arm(world, adjustment, fitted, later))
        for arm, adjustment in adjustments.items()
    }


# ---------------------------------------------------------------------------------- the scores


@dataclass(frozen=True)
class Rate:
    count: int
    n: int
    share: float | None
    interval: tuple[float, float] | None  # Clopper-Pearson, 95%


def rate(count: int, n: int) -> Rate:
    if n == 0:
        return Rate(0, 0, None, None)
    return Rate(count, n, count / n, clopper_pearson(count, n))


@dataclass(frozen=True)
class Claims:
    """What one arm claimed on one replicate, read as it is or with the flag as a stop, and which
    claims the true plant broke."""

    acted: bool  # a plan the certificate trusts for at least one step
    evaluated: bool  # an evaluation interval
    left: bool  # the plan's true mean path left the tolerance within the trusted steps
    lost: bool  # its true regret passed its regret bound by more than MATERIAL of the stakes
    missed: bool  # the interval missed the schedule's true expected cost

    @property
    def failed(self) -> bool:
        return self.left or self.lost or self.missed


def claims(run: dict[str, Any], mode: str) -> Claims:
    stopped = mode == "stop" and bool(run["fit_flag"])
    acted = run["identified"] and run["trustworthy_steps"] >= 1 and not stopped
    trusted = run["deviation"][1 : run["trustworthy_steps"] + 1] if acted else ()
    evaluated = run["evaluation"] == "evaluated" and not stopped
    evaluated = evaluated and not (mode == "stop" and bool(run["later_flag"]))
    return Claims(
        acted=acted,
        evaluated=evaluated,
        left=any(d > TOLERANCE for d in trusted),
        lost=acted and run["regret"] > run["regret_bound"] + MATERIAL * run["stakes"],
        missed=evaluated and not run["covers"],
    )


@dataclass(frozen=True)
class ArmScore:
    arm: str
    mode: str
    covariates: str  # the adjustment set the arm used most often
    acted: Rate
    evaluated: Rate
    fit_flagged: Rate  # over the panels the check ran on
    later_flagged: Rate  # over the later panels it ran on
    left: Rate
    lost: Rate
    missed: Rate
    silent: Rate  # any of the three
    headroom_mean: float | None  # over every plan the arm made, acted on or not
    headroom_q05: float | None
    channel_mean: float | None


def score(runs: Sequence[dict[str, Any]], arm: str, mode: str) -> ArmScore:
    n = len(runs)
    read = [claims(run, mode) for run in runs]
    checked = [run["fit_flag"] for run in runs if run["fit_flag"] is not None]
    later = [run["later_flag"] for run in runs if run["later_flag"] is not None]
    headrooms = np.array(
        [1.0 - run["regret"] / run["stakes"] for run in runs if run["regret"] is not None]
    )
    channels = [run["channel"] for run in runs if run["channel"] is not None]
    sets = Counter(", ".join(run["covariates"]) or "none" for run in runs)
    return ArmScore(
        arm=arm,
        mode=mode,
        covariates=sets.most_common(1)[0][0],
        acted=rate(sum(c.acted for c in read), n),
        evaluated=rate(sum(c.evaluated for c in read), n),
        fit_flagged=rate(sum(checked), len(checked)),
        later_flagged=rate(sum(later), len(later)),
        left=rate(sum(c.left for c in read), n),
        lost=rate(sum(c.lost for c in read), n),
        missed=rate(sum(c.missed for c in read), n),
        silent=rate(sum(c.failed for c in read), n),
        headroom_mean=float(headrooms.mean()) if headrooms.size else None,
        headroom_q05=float(np.quantile(headrooms, 0.05)) if headrooms.size else None,
        channel_mean=float(np.mean(channels)) if channels else None,
    )


@dataclass(frozen=True)
class Stop:
    """What reading the flag as a stop did to a set of replicates: of those whose claims failed, on
    how many it removed every failure; of those whose claims held, on how many it refused one; and
    the one-sided Fisher exact p-value that it removed failures more often than it refused sound
    claims. A stop at the same rate that ignored the failures would not, and a stop only withdraws
    claims, so it cannot add a failure."""

    failed: int
    removed: int
    sound: int
    refused_sound: int
    p_value: float | None  # None where no claim failed or none held


def stop_effect(runs: Sequence[dict[str, Any]]) -> Stop:
    failed = removed = sound = refused_sound = 0
    for run in runs:
        before, after = claims(run, "as is"), claims(run, "stop")
        withdrawn = (before.acted and not after.acted) or (before.evaluated and not after.evaluated)
        failed += before.failed
        removed += before.failed and not after.failed
        sound += (before.acted or before.evaluated) and not before.failed
        refused_sound += withdrawn and not before.failed
    if failed == 0 or sound == 0:
        return Stop(failed, removed, sound, refused_sound, None)
    table = [[removed, failed - removed], [refused_sound, sound - refused_sound]]
    p_value = float(fisher_exact(table, alternative="greater").pvalue)
    return Stop(failed, removed, sound, refused_sound, p_value)


@dataclass(frozen=True)
class WorldRun:
    name: str
    error: str
    oracle: bool
    replicates: int
    scores: list[ArmScore]
    stop: (
        Stop  # over the true and the wrong graph's replicates, or the wrong one's where they agree
    )
    seconds: float
    runs: dict[str, list[dict[str, Any]]] = field(repr=False)


def run_world(name: str, replicates: int, pool: Executor) -> WorldRun:
    """The analyst holds the true graph or the wrong one, so the flag earns its stop by telling
    them apart: within the wrong graph alone, a flag that fires on every panel removes each failure
    and refuses each sound claim, and is no better than chance."""
    started = time.perf_counter()
    world = WORLDS[name]
    results = list(pool.map(replicate, [(name, i) for i in range(replicates)], chunksize=4))
    runs = {arm: [result[arm] for result in results] for arm in ARMS}
    distinct = world.oracle and world.true_edges != world.wrong_edges
    return WorldRun(
        name=name,
        error=world.error,
        oracle=world.oracle,
        replicates=replicates,
        scores=[score(runs[arm], arm, mode) for arm in ARMS for mode in MODES],
        stop=stop_effect(runs["oracle"] + runs["wrong"] if distinct else runs["wrong"]),
        seconds=time.perf_counter() - started,
        runs=runs,
    )


# ------------------------------------------------------------------------------------ the gate


@dataclass(frozen=True)
class Gate:
    """GR2's reading. ``lowered_in``: the worlds where reading the flag as a stop removed failures
    more often than it refused sound claims, at 5% (:class:`Stop`); the flag is shipped if there is
    one. ``size``: on every world with an oracle, the flag rate on the panels the true graph planned
    on, pooled, and whether it is within two points of ``ALPHA``, the 0.9.0 gate."""

    lowered_in: tuple[str, ...]
    removed: int  # replicates whose failures the stop removed, over every world
    refused_sound: int  # replicates on which it refused a claim that had not failed
    size: Rate
    size_within: bool


def gate(runs: Sequence[WorldRun]) -> Gate:
    flagged = [
        bool(arm_run["fit_flag"])
        for run in runs
        if run.oracle
        for arm_run in run.runs["oracle"]
        if arm_run["fit_flag"] is not None
    ]
    size = rate(sum(flagged), len(flagged))
    return Gate(
        lowered_in=tuple(
            run.name for run in runs if run.stop.p_value is not None and run.stop.p_value <= 0.05
        ),
        removed=sum(run.stop.removed for run in runs),
        refused_sound=sum(run.stop.refused_sound for run in runs),
        size=size,
        size_within=size.share is not None and abs(size.share - ALPHA) <= 0.02,
    )


def _share(value: Rate) -> str:
    return "—" if value.share is None else f"{value.share:.3f}"


def _interval(value: Rate) -> str:
    if value.interval is None:
        return "—"
    return f"[{value.interval[0]:.3f}, {value.interval[1]:.3f}]"


def _score(run: WorldRun, arm: str, mode: str) -> ArmScore:
    return next(s for s in run.scores if s.arm == arm and s.mode == mode)


def _markdown(runs: Sequence[WorldRun], replicates: int, x64: bool) -> str:
    verdict = gate(runs)
    lines = [
        "# Track Q — the analyst's graph is wrong",
        "",
        f"{replicates} replicates a world (`causaldyn_bench.graph_errors`, x64 {x64}): on each, "
        f"one panel of {UNITS} units x {PERIODS} periods to plan on and a later one to evaluate "
        "on. "
        f"Tolerance {TOLERANCE}; a plan loses when its true regret passes its bound by "
        f"{MATERIAL:g} of the stakes. Rate intervals are Clopper-Pearson's.",
        "",
        "## The gate",
        "",
        "A flag that does not lower the silent-failure rate is not shipped. Any stop lowers it, so "
        "the flag must do better than a stop at the same rate that ignored the failures: remove "
        "failures more often than it refuses sound claims (one-sided Fisher exact p). The analyst "
        "holds the true graph or the wrong one, so both count, and the wrong one alone where the "
        "two are the same. The wrong graph's silent-failure rate, read as it is and with the stop, "
        "then what the stop did over both graphs:",
        "",
        "| world | the wrong graph | silent | with the stop | failed, removed | sound, refused "
        "| p |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        stop = run.stop
        p_value = "—" if stop.p_value is None else f"{stop.p_value:.2g}"
        lines.append(
            f"| {run.name} | {run.error} | {_share(_score(run, 'wrong', 'as is').silent)} | "
            f"{_share(_score(run, 'wrong', 'stop').silent)} | {stop.failed}, {stop.removed} | "
            f"{stop.sound}, {stop.refused_sound} | {p_value} |"
        )
    low, high = verdict.size.interval or (math.nan, math.nan)
    where = ", ".join(verdict.lowered_in) or "no world"
    lines += [
        "",
        f"**The flag lowers the silent-failure rate beyond chance: "
        f"{'yes' if verdict.lowered_in else 'no'}**, in {where}. Over every world the stop removed "
        f"the failures of {verdict.removed} replicates and refused a sound claim on "
        f"{verdict.refused_sound}.",
        "",
        f"**Its size on these worlds**, the flag rate on the panels the true graph planned on, "
        f"pooled over the worlds with an oracle: {_share(verdict.size)} over {verdict.size.n} "
        f"panels, 95% [{low:.3f}, {high:.3f}]; within two points of {ALPHA:g}: "
        f"{'yes' if verdict.size_within else '**no**'}.",
    ]
    for run in runs:
        lines += [
            "",
            f"## {run.name}",
            "",
            f"The wrong graph {run.error}."
            + (
                ""
                if run.oracle
                else " No graph over the columns states this world's logger, so "
                "the oracle arm plans on the same graph as the wrong one."
            ),
            "",
            "| arm | reading | adjustment | acted | flagged | evaluated | flagged later | left | "
            "lost | missed | silent | headroom (5%) | channel |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for score_ in run.scores:
            headroom = (
                "—"
                if score_.headroom_mean is None
                else f"{score_.headroom_mean:.3f} ({score_.headroom_q05:.3f})"
            )
            channel = "—" if score_.channel_mean is None else f"{score_.channel_mean:.4f}"
            lines.append(
                f"| {score_.arm} | {score_.mode} | {score_.covariates} | {_share(score_.acted)} | "
                f"{_share(score_.fit_flagged)} | {_share(score_.evaluated)} | "
                f"{_share(score_.later_flagged)} | {_share(score_.left)} | {_share(score_.lost)} | "
                f"{_share(score_.missed)} | {_share(score_.silent)} {_interval(score_.silent)} | "
                f"{headroom} | {channel} |"
            )
    lines += [
        "",
        f"The true channel's push on supply is {DIRECT[0]:g} a period in every world. Wall-clock "
        "per world is recorded in `track_q.json` and quoted nowhere: it was not measured on a "
        "clean machine.",
    ]
    return "\n".join(lines) + "\n"


def _compact(runs: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    """Each replicate's reading, column by column, so another threshold can be read without a
    rerun: the flags, the evaluation and its coverage, the farthest the path went within the
    trusted steps, the headroom and the regret's excess over its bound as a share of the stakes."""

    def rounded(value: float | None) -> float | None:
        return None if value is None else float(f"{value:.4g}")

    return {
        "fit_flag": [run["fit_flag"] for run in runs],
        "later_flag": [run["later_flag"] for run in runs],
        "evaluation": [run["evaluation"] for run in runs],
        "covers": [run["covers"] for run in runs],
        "trustworthy_steps": [run["trustworthy_steps"] for run in runs],
        "deviation": [
            None if run["deviation"] is None else [rounded(d) for d in run["deviation"]]
            for run in runs
        ],
        "headroom": [
            None if run["regret"] is None else rounded(1.0 - run["regret"] / run["stakes"])
            for run in runs
        ],
        "excess": [
            None
            if run["regret"] is None
            else rounded((run["regret"] - run["regret_bound"]) / run["stakes"])
            for run in runs
        ],
    }


def _record(runs: Sequence[WorldRun], replicates: int, x64: bool) -> dict[str, Any]:
    return {
        "track": TRACK,
        "metric": METRIC,
        "seed": SEED,
        "replicates": replicates,
        "x64": x64,
        "rule": {
            "units": UNITS,
            "periods": PERIODS,
            "horizon": HORIZON,
            "target": TARGET,
            "unit_cost": UNIT_COST,
            "tolerance": TOLERANCE,
            "material": MATERIAL,
            "alpha": ALPHA,
        },
        "gate": asdict(gate(runs)),
        "wall_clock": "worlds[*].seconds; recorded, not quoted: not measured on a clean machine",
        "worlds": [
            {
                "world": asdict(WORLDS[run.name]),
                "scores": [asdict(s) for s in run.scores],
                "stop": asdict(run.stop),
                "seconds": run.seconds,
                "replicates": {arm: _compact(arm_runs) for arm, arm_runs in run.runs.items()},
            }
            for run in runs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--worlds", nargs="+", choices=list(WORLDS), default=list(WORLDS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    # the pool is the parallelism: the workers inherit one BLAS thread each, whatever the shell set
    os.environ["OMP_NUM_THREADS"] = "1"

    x64 = bool(jnp.zeros(()).dtype == jnp.float64)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        runs = []
        for name in args.worlds:
            runs.append(run_world(name, args.replicates, pool))
            print(f"{name}: {runs[-1].seconds:.0f} s", flush=True)
    text = _markdown(runs, args.replicates, x64)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "track_q.md").write_text(text)
    (args.out / "track_q.json").write_text(
        json.dumps(_record(runs, args.replicates, x64), separators=(",", ":")) + "\n"
    )
    print(text)
    print(f"written to {args.out}/track_q.md and {args.out}/track_q.json")


if __name__ == "__main__":
    main()

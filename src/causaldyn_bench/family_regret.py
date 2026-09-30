"""Track M v2, families: does a plan robust to the curve's family beat the family AIC picks?

**Pre-registered 2026-10-01, before any scored world ran.** The arms were written and piloted on
seeds no score reads (``PILOTS``: ``just track-m2-families-pilot``, into
``results/track_m2_families_pilot.md``). This docstring, the code and the pilot's results are
committed together, and only then does ``just track-m2-families`` run.

**The question** (the library's ADR 0038). A geo test that never bent a channel's curve reads its
slope and little of its shape, so several saturation families fit the tests alike and part where a
plan goes, past the spend the tests covered. Choosing one by AIC and planning on it reads that
family's shape as if the tests had. The alternative keeps every family the tests cannot tell from
the best and plans the split with the least worst regret over them,
:func:`chc.allocation.minimax_allocate`.

**The decision** is Track M v2's (:mod:`causaldyn_bench.mmm_decision`): the quarter after the
world's 156 weeks, one weekly spend a channel held for 13 weeks, at last year's rate, each channel
within half and twice its own mean. A plan's regret is what the best plan in that box returns over
it on the world's own channels, carryover in and out, per euro of the budget.

**The worlds** are Track M v2's drawn worlds (:func:`causaldyn_bench.mmm_decision.drawn`) with
every channel's curve one family in turn (:data:`causaldyn_bench.endogenous_mmm.CURVES`), each at
half its ceiling where his ``tanh(lambda x / 2)`` is: tanh, the exponential and Michaelis-Menten,
concave from zero; Hill of slope 2, Weibull of shape 2 and the logistic of steepness 4, S-shaped.
Every family runs on the same seeds, and a seed's spend, baseline and parameters do not read the
curve, so a seed's six worlds differ in the curve alone. Every channel carries his four go-dark geo
tests (:func:`causaldyn_bench.budget_regret.experiments`).

**The candidates**: each channel's tests fitted by :func:`chc.lift.fit_lift` under six families,
tanh, the exponential and Michaelis-Menten, and Hill, Weibull and the logistic with their shapes
free, so the world's own family is always among them. A fit that raises is no candidate, and is
counted. Two kept candidates whose returns over the box agree within a millionth at nine spends
are one reading: every concave family reaches the same line when the tests read one.

**The rules**:

* *AIC*: ``n log(SSE / n) + 2 p`` over the tests' ``n`` periods; the least wins, a tie going to the
  first in :data:`CANDIDATES`;
* *kept*: every candidate whose least squares lies within ``s^2 F_{1, dof}(0.95)`` of the least
  one's, ``s^2`` and ``dof`` that one's: one parameter's profile cutoff, the rule
  :func:`chc.lift.fit_lift`'s intervals use.

**The arms**, each planned by :mod:`chc.allocation` with the carryover counted both ways:

* *status quo*, last year's mix;
* *tanh*, his family whatever the world's: :mod:`causaldyn_bench.budget_regret`'s CHC arm;
* *AIC*, each channel's AIC family;
* *robust*, :func:`chc.allocation.minimax_allocate` over every combination of the channels' kept
  readings;
* *true family*, each channel's candidate of the world's own family: what knowing the family buys.

A fit whose coefficient is negative, a channel its tests read as losing sales, enters every arm
as one of nought, and is counted: :mod:`chc.allocation` plans increasing channels, and at any
positive price both put the channel at its floor. An arm whose fits raise plays the status quo,
and is counted.

**The oracle**, written apart from :mod:`chc.allocation`: exact bisection on the budget's price for
the concave families, and for the S-shaped every split of a grid refined by SLSQP
(:func:`causaldyn_bench.mmm_decision.oracle`).

**The gate** (the library's MM3 verification): the worst over the six families of each arm's mean
regret per euro, the robust plan's below AIC's, the difference's 95 % bootstrap interval under
nought, the worlds resampled by seed with one draw for every family. It fails if AIC's choice
already does as well.

**The sample**: 100 seeds from 30 000, each run under all six families, 600 worlds. On the pilot's
10 seeds both arms' worst family was the logistic, where the paired difference had a standard
deviation of 0.053 per euro, so 100 seeds put the gate's interval half-width near 0.010.

**Predictions**, from the pilot, before the run:

* **the gate is met, narrowly.** On the pilot the robust plan's worst mean less AIC's was -0.017
  per euro [-0.050, +0.000], both on the logistic worlds. The difference came from one world in
  ten, the other nine tied, so the prediction is weak: a gain in fewer worlds than that fails it;
* the logistic is the worst family for both arms (pilot: 0.356 and 0.373, the Weibull next, 0.208
  and 0.200);
* the hedge costs where AIC's reading is close: on the tanh and exponential worlds the robust
  plan's mean is above AIC's (pilot: +0.012, +0.018), on the Michaelis-Menten and Hill worlds
  below it (-0.016, -0.022), reported and not tested;
* his tanh, planned on S-shaped worlds, costs most: on the Weibull and the logistic its mean is
  about twice AIC's (pilot: 0.404 against 0.200, 0.694 against 0.373);
* every arm that reads the tests has a lower mean than the status quo on every family but the
  true family on the logistic, which plays the status quo there (pilot: in 10 worlds of 10).

**By construction** (R19). A test that never bent a channel's curve reads a line, and every
family reaches one: the concave families with the coefficient past a million (pilot: 23 to 30
channels of 30 on each S-shaped family's worlds), the S-shaped with the inflection past the tested
spend and the coefficient without bound, where the fit's slope is not finite and
:func:`chc.lift.fit_lift` raises. Of the pilot's 1 080 fits 158 raised, 95 of them the
logistic's, 47 the Weibull's, 16 Hill's; on its own worlds the logistic raised on 26 channels of
30. The true-family arm needs the world's family on all three channels and otherwise plays the
status quo (pilot: in 2, 6 and 10 worlds of 10 on Hill's, the Weibull's and the logistic's), so it
prices knowing the family only where the family can be fitted. The rule for a negative
coefficient was written after the pilot's first run stopped on one (AIC's pick on a logistic
world, a coefficient near -2.6e17), and the pilot run again under it. Not in this run: a hedge
over the bend's interval within one family, weights over the families (averaging or stacking),
and the check of an observational channel against the tests.

Precision: ``JAX_ENABLE_X64=1``. The worlds are dealt to ``--workers`` processes; each is scored
apart from the others, so the split moves no number.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import multiprocessing
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from chc.allocation import minimax_allocate
from chc.lift import LiftFit, LiftTest, fit_lift
from chc.response import (
    Channel,
    Exponential,
    GeometricAdstock,
    Hill,
    Logistic,
    MichaelisMenten,
    Saturation,
    Tanh,
    Weibull,
)
from scipy import stats

from causaldyn_bench.budget_regret import (
    OUT_OF_PLAN,
    TIE,
    Comparison,
    Mean,
    experiments,
    planned,
)
from causaldyn_bench.endogenous_mmm import CURVES, EndogenousMediaMix, MediaMixWorld, Series
from causaldyn_bench.mmm_decision import PLANNED, Quarter, drawn, oracle, regret

TRACK = "M2-families"
LEVEL = 0.95
KEEP = 0.95  # the keep rule's F quantile
SAME = 1e-6  # kept readings of a channel whose returns agree within this share are one
SPENDS = 9  # the spends over the box two readings are compared at
DRAWS = 10_000  # the gate's bootstrap
ARMS = ("status quo", "tanh", "AIC", "robust", "true family")
# every candidate family, and where its fit starts beyond the scale: Hill and Weibull at a shape of
# 1.5, between the concave families and the worlds' S-shaped 2, and the logistic at a steepness of 2
CANDIDATES: dict[str, Callable[[float], Saturation]] = {
    "tanh": Tanh,
    "exponential": Exponential,
    "michaelis-menten": MichaelisMenten,
    "hill": lambda scale: Hill(scale, 1.5),
    "weibull": lambda scale: Weibull(scale, 1.5),
    "logistic": lambda scale: Logistic(scale, 2.0),
}
# the candidate of each world's curve
TRUE_CANDIDATE = {
    "tanh": "tanh",
    "exponential": "exponential",
    "michaelis-menten": "michaelis-menten",
    "hill-2": "hill",
    "weibull-2": "weibull",
    "logistic-4": "logistic",
}


@dataclass(frozen=True)
class Environment:
    """A row of the table: the worlds' curve, and the seeds they run from."""

    curve: str
    first: int  # the first seed scored
    worlds: int

    def generator(self, seed: int) -> EndogenousMediaMix:
        return dataclasses.replace(drawn(seed), curve=self.curve)

    @property
    def seeds(self) -> range:
        return range(self.first, self.first + self.worlds)


ENVIRONMENTS = tuple(Environment(curve, 30_000, 100) for curve in CURVES)
PILOTS = tuple(Environment(curve, 900, 10) for curve in CURVES)  # the design's seeds


def template(family: str, tests: Sequence[LiftTest], length: int) -> Channel:
    """Where a candidate's fit starts: a retention of 0.5 and the scale at the control readouts'
    mean spend, as Track M v2's lift fit starts, and the family's shape as :data:`CANDIDATES`
    sets it."""
    spend = np.concatenate([test.control.spend[test.control.history :] for test in tests])
    return Channel(
        GeometricAdstock(0.5, length=length, normalized=True),
        CANDIDATES[family](float(np.mean(spend))),
        1.0,
    )


def candidates(tests: Sequence[LiftTest], length: int) -> dict[str, LiftFit | str]:
    """Every candidate family's fit to one channel's tests, or the error that stopped it."""
    fits: dict[str, LiftFit | str] = {}
    for family in CANDIDATES:
        try:
            fits[family] = fit_lift(tests, template(family, tests, length))
        except (RuntimeError, ValueError) as error:  # a family the tests cannot fit is none
            fits[family] = f"{type(error).__name__}: {error}"
    return fits


@dataclass(frozen=True)
class Selection:
    """What the rules make of one channel's candidates."""

    least: str  # the family with the least cost
    aic: str  # the family AIC picks
    kept: tuple[str, ...]  # every family within the keep rule's cutoff of the least cost


def select(fits: Mapping[str, LiftFit | str]) -> Selection:
    """AIC's family and the kept ones, from the candidates that fitted.

    Raises:
        RuntimeError: when no candidate fitted.
    """
    fitted = {family: fit for family, fit in fits.items() if isinstance(fit, LiftFit)}
    if not fitted:
        raise RuntimeError("no candidate family fitted the tests")
    cost = {family: fit.noise_sd**2 * fit.dof for family, fit in fitted.items()}
    least = min(cost, key=cost.__getitem__)
    dof = fitted[least].dof
    cutoff = cost[least] / dof * float(stats.f.ppf(KEEP, 1, dof))
    periods = dof + len(fitted[least].parameters)
    aic = {
        family: periods * math.log(cost[family] / periods) + 2 * len(fit.parameters)
        for family, fit in fitted.items()
    }
    kept = tuple(family for family in fitted if cost[family] - cost[least] <= cutoff)
    return Selection(least, min(aic, key=aic.__getitem__), kept)


def returns(channel: Channel, column: int, quarter: Quarter, weekly: Series) -> Series:
    """``channel``'s return over the quarter and its tail at each weekly spend in ``weekly``, the
    history's carryover in."""
    history = jnp.asarray(quarter.history[:, column])
    tail = jnp.zeros(channel.kernel.length - 1)

    def one(rate: jax.Array) -> jax.Array:
        series = jnp.concatenate([history, jnp.full(PLANNED, rate), tail])
        return jnp.sum(channel(series)[history.shape[0] :])

    return np.asarray(jax.vmap(one)(jnp.asarray(weekly, dtype=float)))


def distinct(channels: Sequence[Channel], column: int, quarter: Quarter) -> tuple[int, ...]:
    """The positions of the channels that are new readings: one whose returns at every one of
    :data:`SPENDS` spends across the box lie within :data:`SAME` of an earlier one's is that one."""
    weekly = np.linspace(quarter.lower[column], quarter.upper[column], SPENDS)
    seen: list[tuple[int, Series]] = []
    for position, channel in enumerate(channels):
        values = returns(channel, column, quarter, weekly)
        scale = float(np.max(np.abs(values))) or 1.0
        if all(np.max(np.abs(values - other)) > SAME * scale for _, other in seen):
            seen.append((position, values))
    return tuple(position for position, _ in seen)


def plans(
    world: MediaMixWorld, tests: Mapping[str, Sequence[LiftTest]]
) -> tuple[dict[str, Series | str], tuple[dict[str, Any], ...], dict[str, float]]:
    """Every arm's plan, or the error that stopped it; each channel's candidates and what the
    rules made of them; and the robust plan's own account of its hedge."""
    quarter = Quarter.after(world)
    fits = [candidates(channel_tests, world.kernel_length) for channel_tests in tests.values()]
    out: dict[str, Series | str] = {"status quo": quarter.status_quo}
    for arm, family in (("tanh", "tanh"), ("true family", TRUE_CANDIDATE[world.curve])):
        missing = [fit[family] for fit in fits if not isinstance(fit[family], LiftFit)]
        out[arm] = (
            str(missing[0])
            if missing
            else planned([_channel(fit[family]) for fit in fits], quarter)
        )
    records = tuple(_candidate_record(fit) for fit in fits)
    hedge: dict[str, float] = {}
    try:
        selections = [select(fit) for fit in fits]
    except RuntimeError as error:
        out["AIC"] = out["robust"] = str(error)
        return out, records, hedge
    out["AIC"] = planned(
        [_channel(fit[s.aic]) for fit, s in zip(fits, selections, strict=True)], quarter
    )
    options = []
    for column, (fit, selection) in enumerate(zip(fits, selections, strict=True)):
        kept = [_channel(fit[family]) for family in selection.kept]
        new = distinct(kept, column, quarter)
        options.append([kept[position] for position in new])
        records[column].update(
            least=selection.least,
            aic=selection.aic,
            kept=list(selection.kept),
            distinct=[selection.kept[position] for position in new],
        )
    hedged = minimax_allocate(
        list(itertools.product(*options)),
        quarter.budget,
        PLANNED,
        lower=quarter.lower,
        upper=quarter.upper,
        history=quarter.history,
    )
    out["robust"] = hedged.spend
    hedge = {
        "readings": float(math.prod(len(o) for o in options)),
        "worst": hedged.worst,
        "bound": hedged.bound,
    }
    return out, records, hedge


def _channel(fit: LiftFit | str) -> Channel:
    """The channel a plan reads from ``fit``: a negative coefficient as nought."""
    if not isinstance(fit, LiftFit):
        raise TypeError(f"no fit to plan on: {fit}")
    channel = fit.channel
    if float(channel.coefficient) < 0.0:
        return Channel(channel.kernel, channel.curve, 0.0)
    return channel


def _candidate_record(fits: Mapping[str, LiftFit | str]) -> dict[str, Any]:
    """A channel's candidates as a score keeps them: each family's cost, degrees of freedom,
    parameter count and coefficient, or why it raised."""
    return {
        "fits": {
            family: (
                {
                    "cost": fit.noise_sd**2 * fit.dof,
                    "dof": fit.dof,
                    "parameters": len(fit.parameters),
                    "coefficient": float(fit.channel.coefficient),
                }
                if isinstance(fit, LiftFit)
                else {"failure": fit}
            )
            for family, fit in fits.items()
        }
    }


@dataclass(frozen=True)
class WorldScore:
    """One world's plans, scored: each arm's regret per euro of the quarter's budget."""

    seed: int
    curve: str
    budget: float
    best: float  # the oracle's worth
    regret: dict[str, float]  # an arm that failed is scored on the status quo it then plays
    failure: dict[str, str]
    moved: dict[str, float]  # the arms whose plan left the box or missed the budget: how far
    channels: tuple[dict[str, Any], ...]  # each channel's candidates and what the rules made
    hedge: dict[str, float]  # the robust plan's readings, worst regret and bound; empty if none


def score(
    world: MediaMixWorld,
    seed: int,
    arms: Mapping[str, Series | str],
    channels: tuple[dict[str, Any], ...],
    hedge: Mapping[str, float],
) -> WorldScore:
    """Every arm's plan on ``world`` against the oracle.

    Raises:
        RuntimeError: a plan beat the oracle, which only a broken oracle allows.
    """
    quarter = Quarter.after(world)
    best = oracle(world, quarter)
    losses: dict[str, float] = {}
    failure: dict[str, str] = {}
    moved: dict[str, float] = {}
    for arm in ARMS:
        plan = arms[arm]
        if isinstance(plan, str):
            failure[arm] = plan
            plan = quarter.status_quo
        elif not quarter.feasible(plan, tolerance=OUT_OF_PLAN):
            inside = quarter.project(plan)
            moved[arm] = float(np.max(np.abs(inside - plan)))
            plan = inside
        losses[arm] = regret(world, quarter, plan, best) / quarter.budget
        if losses[arm] < -TIE:
            raise RuntimeError(f"world {seed}: {arm} beat the oracle by {-losses[arm]:.3g}")
    return WorldScore(
        seed, world.curve, quarter.budget, best.worth, losses, failure, moved, channels, dict(hedge)
    )


def score_world(job: tuple[str, int]) -> WorldScore:
    """One world of curve ``job[0]`` from seed ``job[1]``, its arms planned and scored."""
    curve, seed = job
    world = Environment(curve, seed, 1).generator(seed).simulate(seed)
    arms, channels, hedge = plans(world, experiments(world, seed))
    return score(world, seed, arms, channels, hedge)


@dataclass(frozen=True)
class CurveRun:
    environment: Environment
    scores: tuple[WorldScore, ...]
    seconds: float

    def regrets(self, arm: str) -> Series:
        return np.array([s.regret[arm] for s in self.scores])

    @property
    def means(self) -> dict[str, Mean]:
        return {arm: Mean.of(self.regrets(arm)) for arm in ARMS}

    @property
    def comparisons(self) -> tuple[Comparison, ...]:
        """The robust plan against each other arm: its regret less theirs, world by world."""
        robust = self.regrets("robust")
        return tuple(Comparison.of(a, robust, self.regrets(a)) for a in ARMS if a != "robust")


def run_environment(environment: Environment, pool: Executor) -> CurveRun:
    began = time.perf_counter()
    jobs = [(environment.curve, seed) for seed in environment.seeds]
    scores = tuple(pool.map(score_world, jobs, chunksize=1))
    return CurveRun(environment, scores, time.perf_counter() - began)


@dataclass(frozen=True)
class Gate:
    """The worst family's mean regret per euro, the robust plan's against AIC's."""

    robust: float
    robust_family: str
    aic: float
    aic_family: str
    difference: float  # robust less AIC
    interval: tuple[float, float]  # bootstrap percentile, the worlds resampled by seed

    @property
    def passes(self) -> bool:
        return self.interval[1] < 0.0


def gate(runs: Sequence[CurveRun], draws: int = DRAWS, seed: int = 0) -> Gate:
    """The gate's statistic and its interval: one resample of the seeds serves every family, so
    the families' pairing by seed is kept.

    Raises:
        ValueError: when the runs do not share their seeds.
    """
    seeds = [s.seed for s in runs[0].scores]
    if any([s.seed for s in run.scores] != seeds for run in runs):
        raise ValueError("the families' runs do not share their seeds")
    robust = np.stack([run.regrets("robust") for run in runs])
    aic = np.stack([run.regrets("AIC") for run in runs])
    pick = np.random.default_rng(seed).integers(0, len(seeds), size=(draws, len(seeds)))
    spread = robust[:, pick].mean(axis=2).max(axis=0) - aic[:, pick].mean(axis=2).max(axis=0)
    low, high = np.quantile(spread, [(1.0 - LEVEL) / 2.0, (1.0 + LEVEL) / 2.0])
    worst_robust, worst_aic = robust.mean(axis=1), aic.mean(axis=1)
    names = [run.environment.curve for run in runs]
    return Gate(
        float(worst_robust.max()),
        names[int(worst_robust.argmax())],
        float(worst_aic.max()),
        names[int(worst_aic.argmax())],
        float(worst_robust.max() - worst_aic.max()),
        (float(low), float(high)),
    )


def _selection_line(run: CurveRun) -> str:
    channels = [c for s in run.scores for c in s.channels]
    read = [c for c in channels if "aic" in c]
    own = TRUE_CANDIDATE[run.environment.curve]
    raised = {
        family: sum("failure" in c["fits"][family] for c in channels) for family in CANDIDATES
    }
    negative = {
        family: sum(c["fits"][family].get("coefficient", 0.0) < 0.0 for c in channels)
        for family in CANDIDATES
    }
    readings = [int(s.hedge["readings"]) for s in run.scores if s.hedge]
    kept = [len(c["kept"]) for c in read]
    return (
        f"Of {len(channels)} channels, AIC picked the world's family ({own}) in "
        f"{sum(c['aic'] == own for c in read)}, and the keep rule kept it in "
        f"{sum(own in c['kept'] for c in read)}; families kept a channel: mean "
        f"{np.mean(kept):.2f}, distinct readings "
        f"{np.mean([len(c['distinct']) for c in read]):.2f}; the robust plan's readings, median "
        f"{np.median(readings) if readings else math.nan:.0f}, most {max(readings, default=0)}. "
        "Fits that raised, by family: "
        + ", ".join(f"{family} {count}" for family, count in raised.items())
        + "; fits with a negative coefficient, planned as nought: "
        + ", ".join(f"{family} {count}" for family, count in negative.items())
        + "."
    )


def _markdown(runs: Sequence[CurveRun], x64: bool, *, pilot: bool = False) -> str:
    verdict = gate(runs)
    lines = [
        f"# Track M v2, families{', pilot' if pilot else ''} ({'float64' if x64 else 'float32'})",
        "",
        "Regret per euro of the quarter's budget, on worlds whose channels' curve is each family "
        "in turn, every family on the same seeds. Mean with its 95 % Student-t interval.",
    ]
    if pilot:
        lines += [
            "",
            "The pilot: the seeds the design was set on, apart from every scored one. No gate "
            "reads it.",
        ]
    lines += ["", "| curve | " + " | ".join(ARMS) + " |", "|---" * (len(ARMS) + 1) + "|"]
    for run in runs:
        cells = [
            f"{m.mean:.4f} [{m.interval[0]:.4f}, {m.interval[1]:.4f}]" for m in run.means.values()
        ]
        lines.append(f"| {run.environment.curve} | " + " | ".join(cells) + " |")
    (low, high) = verdict.interval
    lines += [
        "",
        f"The worst family's mean: robust {verdict.robust:.4f} ({verdict.robust_family}), AIC "
        f"{verdict.aic:.4f} ({verdict.aic_family}); robust less AIC {verdict.difference:+.4f} "
        f"[{low:+.4f}, {high:+.4f}] (bootstrap, {DRAWS} draws of the seeds).",
    ]
    if not pilot:
        lines += [
            "",
            f"**Gate** (the robust plan's worst family mean regret below AIC's, the interval under "
            f"nought): {'met' if verdict.passes else 'NOT met'}.",
        ]
    for run in runs:
        env = run.environment
        lines += [
            "",
            f"## {env.curve}: {env.worlds} worlds from seed {env.first}",
            "",
            "| arm | mean [95 %] | median | failed |",
            "|---|---|---|---|",
        ]
        for arm, m in run.means.items():
            failed = sum(arm in s.failure for s in run.scores)
            lines.append(
                f"| {arm} | {m.mean:.4f} [{m.interval[0]:.4f}, {m.interval[1]:.4f}] | "
                f"{m.median:.4f} | {failed} |"
            )
        lines += [
            "",
            "The robust plan against each arm: its regret less the arm's over the same worlds, and "
            "the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower "
            "share).",
            "",
            "| against | difference [95 %] | lower | tied | higher | lower share [95 %] |",
            "|---|---|---|---|---|---|",
        ]
        for c in run.comparisons:
            d, (share_low, share_high) = c.difference, c.lower_share
            lines.append(
                f"| {c.arm} | {d.mean:+.4f} [{d.interval[0]:+.4f}, {d.interval[1]:+.4f}] | "
                f"{c.lower} | {c.tied} | {c.higher} | "
                f"{c.lower / len(run.scores):.2f} [{share_low:.2f}, {share_high:.2f}] |"
            )
        lines += ["", _selection_line(run)]
    return "\n".join(lines) + "\n"


def _jsonable(value: Any) -> Any:
    """Non-finite floats as ``null``, so a strict parser reads the artefact."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _record(runs: Sequence[CurveRun], x64: bool, *, pilot: bool = False) -> dict[str, Any]:
    verdict = gate(runs)
    return _jsonable(
        {
            "track": TRACK,
            "pilot": pilot,
            "x64": x64,
            "design": {
                "planned": PLANNED,
                "arms": ARMS,
                "candidates": list(CANDIDATES),
                "keep": KEEP,
                "same": SAME,
                "level": LEVEL,
                "draws": DRAWS,
            },
            "gate": {**asdict(verdict), "passes": None if pilot else verdict.passes},
            "wall_clock": "curves[*].seconds; recorded, not quoted: not measured on a clean "
            "machine",
            "curves": [
                {
                    "environment": asdict(run.environment),
                    "seconds": run.seconds,
                    "means": {arm: asdict(m) for arm, m in run.means.items()},
                    "comparisons": [asdict(c) for c in run.comparisons],
                    "worlds": [asdict(s) for s in run.scores],
                }
                for run in runs
            ],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="the design's seeds, not the scored")
    parser.add_argument("--curves", nargs="+", default=list(CURVES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--name", help="default track_m2_families, with _pilot for the pilot")
    args = parser.parse_args()
    name = args.name or ("track_m2_families_pilot" if args.pilot else "track_m2_families")
    if not bool(jnp.zeros(()).dtype == jnp.float64):
        raise SystemExit("JAX_ENABLE_X64=1 is required: the arms were piloted at float64")
    # the pool is the parallelism: the workers inherit one BLAS thread each, whatever the shell set
    os.environ["OMP_NUM_THREADS"] = "1"
    environments = [e for e in (PILOTS if args.pilot else ENVIRONMENTS) if e.curve in args.curves]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        runs = []
        for environment in environments:
            runs.append(run_environment(environment, pool))
            print(f"{environment.curve}: {runs[-1].seconds:.0f} s", flush=True)
    text = _markdown(runs, True, pilot=args.pilot)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.md").write_text(text)
    (args.out / f"{name}.json").write_text(
        json.dumps(_record(runs, True, pilot=args.pilot), separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    print(text)
    print(f"written to {args.out}/{name}.md and {args.out}/{name}.json")


if __name__ == "__main__":
    main()

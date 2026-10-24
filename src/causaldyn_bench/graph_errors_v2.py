"""Track Q, version 2: the analyst's graph is wrong, and a plan is scored on the claim its error
tube makes. Pre-registered in ``results/track_q_v2_preregistration.md`` before any scored replicate
ran.

Version 1 (:mod:`causaldyn_bench.graph_errors`) scored a plan on its true mean path, within a
tolerance of 0.05 over the steps its certificate trusts. From causal-hybrid-control 0.17.0 the tube
carries the plant's own one-step noise (the library's ADR 0080), 0.30 or more at the first step of
these worlds, so at 0.05 no plan is trusted for a step and that leg is not armed. Version 2 keeps
the worlds, the arms, the flags, the regret and evaluation legs and the gate, and changes three
things:

* **The tolerance** is set a priori from the generator: ``WIDEN`` times the largest, over the
  worlds, of the true plant's 0.95 radius at the plan's last step, the radius within which the
  realised (supply, wait) of a unit planned from a known start stays with probability 0.95. One
  value for every world: the mediator world's, the widest, so that a plan with the true model's
  own spread is trusted for all its steps in each.
* **A plan leaves** when, at a step its certificate trusts, the true plant's realised path is
  outside the tube with probability above ``1 - LEVEL + MARGIN``: the claim the tube makes, read
  exactly from the Gaussian (:func:`exceedance`). ``MARGIN`` is the estimation slack of a 0.95
  tube whose noise and slope are read off one log of ``UNITS x (PERIODS - 1)`` transitions: the
  95th percentile, over logs, of a correctly specified tube's exceedance, less 0.05, at the plan's
  last step, where it is largest.
* **The seed**, new and fixed in the pre-registration.

Precision: run with ``JAX_ENABLE_X64=1`` (``just track-q-v2``), version 1's.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import chi2

from causaldyn_bench import graph_errors as v1
from causaldyn_bench.graph_errors import (
    ARMS,
    HORIZON,
    MATERIAL,
    PERIODS,
    STATES,
    UNITS,
    WORLDS,
    Claims,
    Plant,
    Rate,
    WorldRun,
    gate,
    plant,
    rate,
)

TRACK = "Q-graph-errors-v2"
SEED = 20261022
LEVEL = 0.95  # the probability with which the plan's tube holds the plant at each step
WIDEN = 1.1  # the tolerance over the widest world's own 0.95 radius at the plan's last step
READ = UNITS * (PERIODS - 1)  # the transitions the tube's noise is read off
# The estimation slack of a 0.95 tube read off one log of nu = READ = 4400 transitions, on the one
# state that carries the noise. Its radius at step k scales with the read noise's root, of relative
# sd 1 / sqrt(2 nu), and with supply's fitted persistence a through sqrt(sum_{j<k} a^(2j)), of
# relative sd g_k s_a, g_k its log-derivative in a and s_a = 0.005961 the persistence's standard
# error, sigma / sqrt(nu v), v = 0.14454 supply's variance given wait in the base worlds' panel,
# averaged over the transitions. At the 5% quantile of the radius, 1 - 1.645 rho_k, rho_k the root
# of the two variances' sum, a correctly specified tube is crossed with probability
# 2 (1 - Phi(1.95996 (1 - 1.645 rho_k))): 0.05 + 0.00415, 0.00432 and 0.00476 at steps 1 to 3.
# The margin is the largest, rounded to two figures.
MARGIN = 0.0048
THRESHOLD = 1.0 - LEVEL + MARGIN
BONFERRONI = 0.05 / len(WORLDS)  # a secondary reading of the gate, over its nine worlds
_ROOT_TAU = math.sqrt(2.0 * math.pi)


def predictive_covariances(system: Plant, steps: int = HORIZON) -> list[NDArray[np.float64]]:
    """``Sigma_0 .. Sigma_steps``: the covariance of the realised (supply, wait) about the true mean
    path, from a known start and under an open-loop schedule, which moves the mean alone:
    ``Sigma_{k+1} = A Sigma_k A' + Q`` over the plant's whole state, then its states' block."""
    states = len(STATES)
    covariance = np.zeros_like(system.a)
    covariances = [covariance[:states, :states].copy()]
    for _ in range(steps):
        covariance = system.a @ covariance @ system.a.T + system.noise
        covariances.append(covariance[:states, :states].copy())
    return covariances


def exceedance(offset: ArrayLike, covariance: ArrayLike, radius: float) -> float:
    """``P(||offset + e|| > radius)``, ``e`` Gaussian of the two states' ``covariance``, exactly.

    The norm does not see a rotation, so the gap is read along the covariance's axes: ``e`` is
    ``sigma_s t`` along the narrow one and ``sigma_l w`` along the wide one, ``t`` and ``w``
    standard normals. Past ``|offset_s + sigma_s t| = radius`` the path is outside whatever ``w``
    is; within it, outside where ``|offset_l + sigma_l w|`` passes ``sqrt(radius^2 - (offset_s +
    sigma_s t)^2)``, two normal tails, integrated over ``t`` by QUADPACK. ``t`` past 12 holds less
    than 1e-32 of the mass and is counted as outside."""
    values, vectors = np.linalg.eigh(np.asarray(covariance, dtype=np.float64))
    if not values[0] > 0.0:
        raise ValueError(f"the covariance is not positive definite: eigenvalues {values}")
    narrow, wide = np.sqrt(values)
    across, along = (float(x) for x in vectors.T @ np.asarray(offset, dtype=np.float64))
    low, high = (-radius - across) / narrow, (radius - across) / narrow
    outside = float(ndtr(low) + ndtr(-high))
    start, stop = max(low, -12.0), min(high, 12.0)
    if start >= stop:
        return outside

    def beyond(t: float) -> float:
        reach = math.sqrt(max(radius * radius - (across + narrow * t) ** 2, 0.0))
        tails = ndtr((along - reach) / wide) + ndtr((-reach - along) / wide)
        return math.exp(-0.5 * t * t) / _ROOT_TAU * float(tails)

    inside, _ = integrate.quad(beyond, start, stop, epsabs=1e-13, epsrel=1e-11, limit=200)
    return outside + inside


def predictive_radius(covariance: ArrayLike, level: float = LEVEL) -> float:
    """The radius the norm of a Gaussian of the two states' ``covariance`` stays within with
    probability ``level``: the root of :func:`exceedance` at ``1 - level``, between
    ``sqrt(chi2_1(level) lambda_max)``, where the narrow axis holds nothing, and
    ``sqrt(chi2_2(level) lambda_max)``, where it is as wide as the other."""
    covariance = np.asarray(covariance, dtype=np.float64)
    largest = float(np.linalg.eigvalsh(covariance)[-1])
    centre = np.zeros(covariance.shape[0])

    def excess(radius: float) -> float:
        return exceedance(centre, covariance, radius) - (1.0 - level)

    low = math.sqrt(chi2.ppf(level, 1) * largest)
    high = math.sqrt(chi2.ppf(level, covariance.shape[0]) * largest)
    if excess(high) >= 0.0:
        return high
    return float(brentq(excess, low, high, xtol=1e-15, rtol=1e-13))


def a_priori_tolerance() -> tuple[float, str, dict[str, float]]:
    """``WIDEN`` times the largest of the worlds' 0.95 radii at the plan's last step under their
    true plants; the world it is read on; and every world's radius."""
    radii = {
        name: predictive_radius(predictive_covariances(plant(world))[HORIZON])
        for name, world in WORLDS.items()
    }
    widest = max(radii, key=radii.__getitem__)
    return WIDEN * radii[widest], widest, radii


TOLERANCE, WIDEST, RADII = a_priori_tolerance()


def replicate(job: tuple[str, int], seed: int = SEED) -> dict[str, dict[str, Any]]:
    """Version 1's replicate at this tolerance and ``seed``, each arm's plan with its exceedance
    at steps 1 to ``HORIZON`` under the true plant from the plan's own start, the confounder at its
    mean over the panel's units at the last period, as version 1 reads the mean path."""
    runs = v1.replicate(job, seed=seed, tolerance=TOLERANCE)
    covariances = predictive_covariances(plant(WORLDS[job[0]]))
    for run in runs.values():
        tube, offset = run["tube"], run["offset"]
        run["exceedance"] = (
            None
            if tube is None
            else tuple(
                exceedance(offset[step], covariances[step], tube[step])
                for step in range(1, HORIZON + 1)
            )
        )
    return runs


def claims(run: dict[str, Any], mode: str) -> Claims:
    """Version 1's claims, the plan's own read on its tube: it leaves where the true plant is
    outside the tube with probability above ``THRESHOLD`` at a step the certificate trusts."""
    read = v1.claims(run, mode)
    trusted = run["exceedance"][: run["trustworthy_steps"]] if read.acted else ()
    return replace(read, left=any(e > THRESHOLD for e in trusted))


@dataclass(frozen=True)
class Calibration:
    """On the true graph, how often the tube leaves for a reason that is not the graph: of the
    trusted steps of the plans acted on as they are, the share whose exceedance is above
    ``THRESHOLD``; and of those plans, the share with such a step, the plans' "left". One read
    scale serves every step of a plan, so its steps move together and the steps' interval, which
    counts them as independent, is too narrow; the plans' is the one to read."""

    steps: Rate
    plans: Rate


def calibration(runs: Sequence[dict[str, Any]]) -> Calibration:
    acted = _acted(runs)
    steps = [e for run in acted for e in run["exceedance"][: run["trustworthy_steps"]]]
    left = sum(claims(run, "as is").left for run in acted)
    return Calibration(
        steps=rate(sum(e > THRESHOLD for e in steps), len(steps)), plans=rate(left, len(acted))
    )


def run_world(name: str, replicates: int, pool: Executor, seed: int = SEED) -> WorldRun:
    return v1.run_world(name, replicates, pool, job=partial(replicate, seed=seed), reading=claims)


# --------------------------------------------------------------------------------- the report


def _share(value: Rate) -> str:
    return "—" if value.share is None else f"{value.share:.3f}"


def _interval(value: Rate) -> str:
    if value.interval is None:
        return "—"
    return f"[{value.interval[0]:.3f}, {value.interval[1]:.3f}]"


def _acted(runs: Sequence[dict[str, Any]], mode: str = "as is") -> list[dict[str, Any]]:
    return [run for run in runs if claims(run, mode).acted]


def _largest(runs: Sequence[dict[str, Any]]) -> NDArray[np.float64]:
    """Per plan acted on, its largest exceedance over the steps its certificate trusts."""
    return np.array([max(run["exceedance"][: run["trustworthy_steps"]]) for run in _acted(runs)])


def _by_step(runs: Sequence[dict[str, Any]]) -> list[float | None]:
    """Per step, the mean exceedance over the plans acted on that trust it."""
    acted = _acted(runs)
    means: list[float | None] = []
    for step in range(HORIZON):
        values = [run["exceedance"][step] for run in acted if run["trustworthy_steps"] > step]
        means.append(float(np.mean(values)) if values else None)
    return means


def _show(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _markdown(
    runs: Sequence[WorldRun],
    replicates: int,
    seed: int,
    x64: bool,
    preregistration: str | None,
    chc_commit: str | None,
) -> str:
    verdict = gate(runs)
    oracle = [arm_run for run in runs if run.oracle for arm_run in run.runs["oracle"]]
    pooled = calibration(oracle)
    lines = [
        "# Track Q, version 2 — the analyst's graph is wrong, scored on the tube's claim",
        "",
        f"{replicates} replicates a world (`causaldyn_bench.graph_errors_v2`, seed {seed}, x64 "
        f"{x64}), on causal-hybrid-control "
        + (f"`{chc_commit}`" if chc_commit else "at a commit not recorded")
        + ", pre-registered in `results/track_q_v2_preregistration.md` "
        + (f"at `{preregistration}`" if preregistration else "(commit not recorded)")
        + f". On each replicate, one panel of {UNITS} units x {PERIODS} periods to plan on and a "
        "later one to evaluate on. "
        f"Tolerance {TOLERANCE:.4f}, {WIDEN:g} times the {WIDEST} world's 0.95 radius at step "
        f"{HORIZON} under its true plant, {RADII[WIDEST]:.4f}. A plan leaves when the true plant "
        f"is outside its tube with probability above {1.0 - LEVEL:g} + {MARGIN:g} = "
        f"{THRESHOLD:.4f} at a step its certificate trusts, read exactly from the Gaussian; it "
        f"loses when its true regret passes its bound by {MATERIAL:g} of the stakes. Rate "
        "intervals are Clopper-Pearson's.",
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
    strict = [
        run.name for run in runs if run.stop.p_value is not None and run.stop.p_value <= BONFERRONI
    ]
    lines += [
        "",
        f"**The flag lowers the silent-failure rate beyond chance: "
        f"{'yes' if verdict.lowered_in else 'no'}**, in {where}. Over every world the stop removed "
        f"the failures of {verdict.removed} replicates and refused a sound claim on "
        f"{verdict.refused_sound}. At {BONFERRONI:.4f}, 0.05 over the {len(WORLDS)} worlds: "
        f"{', '.join(strict) or 'no world'}.",
        "",
        f"**Its size on these worlds**, the flag rate on the panels the true graph planned on, "
        f"pooled over the worlds with an oracle: {_share(verdict.size)} over {verdict.size.n} "
        f"panels, 95% [{low:.3f}, {high:.3f}]; within two points of {v1.ALPHA:g}: "
        f"{'yes' if verdict.size_within else '**no**'}.",
        "",
        "## The calibration check",
        "",
        f"On the true graph, the steps its certificate trusts whose exceedance is above "
        f"{THRESHOLD:.4f}: how often the tube leaves for a reason that is not the graph. Pooled "
        f"over the worlds with an oracle: **{_share(pooled.steps)}** of {pooled.steps.n} trusted "
        f"steps, {_interval(pooled.steps)}, and **{_share(pooled.plans)}** of {pooled.plans.n} "
        f"plans, {_interval(pooled.plans)}. One read scale serves every step of a plan, so the "
        "steps' interval, which counts them as independent, is too narrow, and the plans' is the "
        "one to read. Per world, with the oracle's mean exceedance at each step over the plans "
        "that trust it, and the largest over every plan:",
        "",
        "| world | plans acted on | trusted steps | steps above | share of steps | plans that "
        "leave | share of plans | mean exceedance, steps "
        + ", ".join(str(step) for step in range(1, HORIZON + 1))
        + " | largest |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        runs_ = run.runs["oracle"]
        check = calibration(runs_)
        largest = _largest(runs_)
        lines.append(
            f"| {run.name}{'' if run.oracle else ' (no oracle)'} | {check.plans.n} | "
            f"{check.steps.n} | {check.steps.count} | {_share(check.steps)} "
            f"{_interval(check.steps)} | {check.plans.count} | {_share(check.plans)} "
            f"{_interval(check.plans)} | "
            + ", ".join(_show(mean) for mean in _by_step(runs_))
            + f" | {_show(float(largest.max()) if largest.size else None)} |"
        )
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
            "| arm | reading | adjustment | acted | trusted | flagged | evaluated | flagged later "
            "| left | lost | missed | silent | exceedance (95%) | headroom (5%) | channel |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for score_ in run.scores:
            arm_runs = run.runs[score_.arm]
            acted = _acted(arm_runs, score_.mode)
            trusted = float(np.mean([r["trustworthy_steps"] for r in acted])) if acted else None
            largest = np.array(
                [max(r["exceedance"][: r["trustworthy_steps"]]) for r in acted], dtype=float
            )
            exceeds = (
                "—"
                if not largest.size
                else f"{largest.mean():.4f} ({np.quantile(largest, 0.95):.4f})"
            )
            headroom = (
                "—"
                if score_.headroom_mean is None
                else f"{score_.headroom_mean:.3f} ({score_.headroom_q05:.3f})"
            )
            channel = "—" if score_.channel_mean is None else f"{score_.channel_mean:.4f}"
            lines.append(
                f"| {score_.arm} | {score_.mode} | {score_.covariates} | {_share(score_.acted)} | "
                f"{_show(trusted, 2)} | {_share(score_.fit_flagged)} | "
                f"{_share(score_.evaluated)} | {_share(score_.later_flagged)} | "
                f"{_share(score_.left)} | {_share(score_.lost)} | {_share(score_.missed)} | "
                f"{_share(score_.silent)} {_interval(score_.silent)} | {exceeds} | {headroom} | "
                f"{channel} |"
            )
    lines += [
        "",
        "In each world's table, *trusted* is the mean number of steps the certificate trusts over "
        "the plans acted on, and *exceedance* the mean over them of the largest exceedance at a "
        f"trusted step, with its 95th percentile. The true channel's push on supply is "
        f"{v1.DIRECT[0]:g} a period in every world. Wall-clock per world is recorded in "
        "`track_q_v2.json` and quoted nowhere: it was not measured on a clean machine.",
    ]
    return "\n".join(lines) + "\n"


def _score(run: WorldRun, arm: str, mode: str) -> v1.ArmScore:
    return next(s for s in run.scores if s.arm == arm and s.mode == mode)


def _compact(runs: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    """Version 1's columns, with each plan's tube and its exceedance at steps 1 to ``HORIZON``, so
    another threshold can be read without a rerun."""

    def rounded(values: Sequence[float] | None) -> list[float] | None:
        return None if values is None else [float(f"{value:.6g}") for value in values]

    return v1._compact(runs) | {
        "tube": [rounded(run["tube"]) for run in runs],
        "exceedance": [rounded(run["exceedance"]) for run in runs],
    }


def _record(
    runs: Sequence[WorldRun],
    replicates: int,
    seed: int,
    x64: bool,
    preregistration: str | None,
    chc_commit: str | None,
) -> dict[str, Any]:
    oracle = [arm_run for run in runs if run.oracle for arm_run in run.runs["oracle"]]
    return {
        "track": TRACK,
        "metric": v1.METRIC,
        "seed": seed,
        "replicates": replicates,
        "x64": x64,
        "preregistration": preregistration,
        "chc_commit": chc_commit,
        "rule": {
            "units": UNITS,
            "periods": PERIODS,
            "horizon": HORIZON,
            "target": v1.TARGET,
            "unit_cost": v1.UNIT_COST,
            "tolerance": TOLERANCE,
            "widen": WIDEN,
            "widest_world": WIDEST,
            "radii": RADII,
            "level": LEVEL,
            "margin": MARGIN,
            "threshold": THRESHOLD,
            "material": MATERIAL,
            "alpha": v1.ALPHA,
            "bonferroni": BONFERRONI,
        },
        "gate": asdict(gate(runs)),
        "calibration": {
            "pooled": asdict(calibration(oracle)),
            "worlds": {run.name: asdict(calibration(run.runs["oracle"])) for run in runs},
        },
        "wall_clock": "worlds[*].seconds; recorded, not quoted: not measured on a clean machine",
        "worlds": [
            {
                "world": asdict(WORLDS[run.name]),
                "scores": [asdict(s) for s in run.scores],
                "stop": asdict(run.stop),
                "seconds": run.seconds,
                "replicates": {arm: _compact(run.runs[arm]) for arm in ARMS},
            }
            for run in runs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=v1.REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--worlds", nargs="+", choices=list(WORLDS), default=list(WORLDS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--name", default="track_q_v2", help="the report's file name, unsuffixed")
    parser.add_argument("--preregistration", help="the pre-registration's commit, to record")
    parser.add_argument("--chc-commit", help="the library commit this run imports, to record")
    args = parser.parse_args()
    # the pool is the parallelism: the workers inherit one BLAS thread each, whatever the shell set
    os.environ["OMP_NUM_THREADS"] = "1"

    x64 = bool(jnp.zeros(()).dtype == jnp.float64)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        runs = []
        for name in args.worlds:
            runs.append(run_world(name, args.replicates, pool, args.seed))
            print(f"{name}: done", flush=True)
    provenance = (args.replicates, args.seed, x64, args.preregistration, args.chc_commit)
    text = _markdown(runs, *provenance)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.name}.md").write_text(text)
    (args.out / f"{args.name}.json").write_text(
        json.dumps(_record(runs, *provenance), separators=(",", ":")) + "\n"
    )
    print(text)
    print(f"written to {args.out}/{args.name}.md and {args.out}/{args.name}.json")


if __name__ == "__main__":
    main()

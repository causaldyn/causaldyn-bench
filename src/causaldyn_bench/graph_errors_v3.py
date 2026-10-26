"""Track Q, version 3: the analyst's graph is wrong, and the stop reads the logger's own draw.
Pre-registered in ``results/track_q_v3_preregistration.md`` before any scored replicate ran.

Version 2 (:mod:`causaldyn_bench.graph_errors_v2`) read the logger check's flag as a stop and found
it no better than a stop at the same rate that ignored the failures, in every world. The library
now states why (its ``validation/confounded_log_equivalence.mac``): a confounder drawn afresh and
Gaussian leaves the logged states and levers with the law of a loop without it, so no test that
reads only those can see it. A test that reads the logger's own draw can. Version 3 logs the draw,
``fresh``, and reads the stop off :class:`chc.decision.DitherCheck`: the plan's channel against
the one the draw instruments, a Durbin-Wu-Hausman contrast priced in the plan's regret. It keeps
version 2's worlds, arms, tolerance, margin, claims, scoring and gate, and changes the flag:

* **The stop** refuses a plan, and its evaluation, where the check was tested, rejects at
  ``ALPHA`` and prices the contrast above ``MATERIAL`` times the gain the plan claims on its own
  model: what version 1 counts a material loss, in the plan's units. The check reads the panel the
  plan is fitted on; no flag is read on the later panel.
* **The seed**, new and fixed in the pre-registration.

The arm chosen by fit reads no draw, as in version 2, and is not stopped.

Precision: run with ``JAX_ENABLE_X64=1`` (``just track-q-v3``), version 1's.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import chc
import jax.numpy as jnp
import numpy as np

from causaldyn_bench import graph_errors as v1
from causaldyn_bench import graph_errors_v2 as v2
from causaldyn_bench.graph_errors import (
    ARMS,
    HORIZON,
    MATERIAL,
    PERIODS,
    UNITS,
    WORLDS,
    Rate,
    Stop,
    WorldRun,
    gate,
    plant,
    rate,
    stop_effect,
)
from causaldyn_bench.graph_errors_v2 import (
    LEVEL,
    MARGIN,
    THRESHOLD,
    TOLERANCE,
    claims,
    exceedance,
    predictive_covariances,
)

TRACK = "Q-graph-errors-v3"
SEED = 20261026
DITHER = "fresh"  # the column the panel logs the logger's draw in
ALPHA = 0.05  # the contrast's level
SIZE_LIMIT = 0.07  # the test's rejection rate on the true graph may be at most this, pooled
BONFERRONI = 0.05 / len(WORLDS)
# the stop's margin, as a share of the gain the plan claims on its own model: version 1's material
# loss
COST_SHARE = MATERIAL


def has_dither() -> bool:
    """Whether the installed library's ``prescribe`` reads a logged draw."""
    return "dither" in inspect.signature(chc.prescribe).parameters


def stop(check: dict[str, Any] | None, share: float = COST_SHARE) -> bool | None:
    """The stop on one arm's check: tested, rejected at ``ALPHA``, and priced above ``share`` of
    the gain the plan claims. None where the arm read no draw."""
    if check is None:
        return None
    return bool(
        check["status"] == "tested"
        and check["p_value"] <= ALPHA
        and check["cost"] > share * check["claimed_gain"]
    )


def rejected(check: dict[str, Any] | None) -> bool | None:
    """The contrast alone: tested and rejected at ``ALPHA``, at any cost."""
    if check is None:
        return None
    return bool(check["status"] == "tested" and check["p_value"] <= ALPHA)


def replicate(job: tuple[str, int], seed: int = SEED) -> dict[str, dict[str, Any]]:
    """Version 2's replicate on panels that log the draw, each arm's flag the stop on its check.
    The logger check's flag on the fitted panel is kept as ``logger_flag``, for the secondary
    reading; no flag is read on the later panel."""
    runs = v1.replicate(job, seed=seed, tolerance=TOLERANCE, dither=DITHER)
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
        run["logger_flag"] = run["fit_flag"]
        run["fit_flag"] = stop(run["dither"])
        run["later_flag"] = None
    return runs


def run_world(name: str, replicates: int, pool: Executor, seed: int = SEED) -> WorldRun:
    return v1.run_world(name, replicates, pool, job=partial(replicate, seed=seed), reading=claims)


def reflagged(run: WorldRun, flag: Callable[[dict[str, Any]], bool | None]) -> Stop:
    """The world's stop table with each arm's flag read by ``flag`` instead: over the true and the
    wrong graph's replicates, or the wrong one's where they agree, as :func:`v1.run_world` reads
    it."""
    world = WORLDS[run.name]
    arms = (
        ("oracle", "wrong")
        if world.oracle and world.true_edges != world.wrong_edges
        else ("wrong",)
    )
    runs = [{**arm_run, "fit_flag": flag(arm_run)} for arm in arms for arm_run in run.runs[arm]]
    return stop_effect(runs, claims)


@dataclass(frozen=True)
class Size:
    """The contrast's rejection rate at ``ALPHA`` on the panels the true graph planned on, pooled
    over the worlds with an oracle, and the panels it read as void."""

    rejected: Rate
    void: int


def size(runs: Sequence[WorldRun]) -> Size:
    checks = [
        arm_run["dither"]
        for run in runs
        if run.oracle
        for arm_run in run.runs["oracle"]
        if arm_run["dither"] is not None
    ]
    return Size(
        rejected=rate(sum(bool(rejected(check)) for check in checks), len(checks)),
        void=sum(check["status"] == "void" for check in checks),
    )


def _checks(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run["dither"] for run in runs if run["dither"] is not None]


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


@dataclass(frozen=True)
class ArmTest:
    """One arm's checks in one world: how often the contrast rejected, how often the stop fired,
    and the mean priced cost and margin over the tested panels."""

    tested: int
    void: int
    rejected: Rate
    stopped: Rate
    cost: float | None
    margin: float | None


def arm_test(runs: Sequence[dict[str, Any]]) -> ArmTest:
    checks = _checks(runs)
    tested = [check for check in checks if check["status"] == "tested"]
    return ArmTest(
        tested=len(tested),
        void=len(checks) - len(tested),
        rejected=rate(sum(bool(rejected(check)) for check in checks), len(checks)),
        stopped=rate(sum(bool(stop(check)) for check in checks), len(checks)),
        cost=_mean([check["cost"] for check in tested]),
        margin=_mean([COST_SHARE * check["claimed_gain"] for check in tested]),
    )


# --------------------------------------------------------------------------------- the report


def _share(value: Rate) -> str:
    return "—" if value.share is None else f"{value.share:.3f}"


def _interval(value: Rate) -> str:
    if value.interval is None:
        return "—"
    return f"[{value.interval[0]:.3f}, {value.interval[1]:.3f}]"


def _show(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def _p(stop_: Stop) -> str:
    return "—" if stop_.p_value is None else f"{stop_.p_value:.2g}"


def _gate_rows(runs: Sequence[WorldRun], stops: Sequence[Stop]) -> list[str]:
    lines = [
        "| world | the wrong graph | failed, removed | sound, refused | p |",
        "|---|---|---|---|---|",
    ]
    for run, stop_ in zip(runs, stops, strict=True):
        lines.append(
            f"| {run.name} | {run.error} | {stop_.failed}, {stop_.removed} | "
            f"{stop_.sound}, {stop_.refused_sound} | {_p(stop_)} |"
        )
    return lines


def _passes(runs: Sequence[WorldRun], stops: Sequence[Stop], level: float) -> list[str]:
    return [
        run.name
        for run, stop_ in zip(runs, stops, strict=True)
        if stop_.p_value is not None and stop_.p_value <= level
    ]


def _markdown(
    runs: Sequence[WorldRun],
    replicates: int,
    seed: int,
    x64: bool,
    preregistration: str | None,
    chc_commit: str | None,
) -> str:
    verdict = gate(runs)
    strict = _passes(runs, [run.stop for run in runs], BONFERRONI)
    tests = size(runs)
    low, high = tests.rejected.interval or (math.nan, math.nan)
    lines = [
        "# Track Q, version 3 — the analyst's graph is wrong, and the stop reads the logger's draw",
        "",
        f"{replicates} replicates a world (`causaldyn_bench.graph_errors_v3`, seed {seed}, x64 "
        f"{x64}), on causal-hybrid-control "
        + (f"`{chc_commit}`" if chc_commit else "at a commit not recorded")
        + ", pre-registered in `results/track_q_v3_preregistration.md` "
        + (f"at `{preregistration}`" if preregistration else "(commit not recorded)")
        + f". On each replicate, one panel of {UNITS} units x {PERIODS} periods to plan on, which "
        f"logs the logger's draw as `{DITHER}`, and a later one to evaluate on. Version 2's "
        f"tolerance {TOLERANCE:.4f} and threshold {1.0 - LEVEL:g} + {MARGIN:g} = "
        f"{THRESHOLD:.4f}. The stop refuses a plan and its evaluation where `prescribe(dither=)`'s "
        f"check was tested, rejects at {ALPHA:g}, and prices the contrast above {COST_SHARE:g} of "
        "the gain the plan claims on its own model. Rate intervals are Clopper-Pearson's.",
        "",
        "## The gate",
        "",
        "The stop must remove failures more often than it refuses sound claims, as a stop at the "
        "same rate that ignored the failures would not (one-sided Fisher exact p), over the true "
        "and the wrong graph, or the wrong one alone where the two are the same:",
        "",
        *_gate_rows(runs, [run.stop for run in runs]),
        "",
        f"**The stop lowers the silent-failure rate beyond chance: "
        f"{'yes' if verdict.lowered_in else 'no'}**, in "
        f"{', '.join(verdict.lowered_in) or 'no world'}. At {BONFERRONI:.4f}, 0.05 over the "
        f"{len(WORLDS)} worlds: {', '.join(strict) or 'no world'}.",
        "",
        f"**The contrast's size**, its rejection rate at {ALPHA:g} on the panels the true graph "
        f"planned on, pooled over the worlds with an oracle: {_share(tests.rejected)} over "
        f"{tests.rejected.n} panels, 95% [{low:.3f}, {high:.3f}], {tests.void} of them void; at "
        f"most {SIZE_LIMIT:g}: "
        + (
            "—"
            if tests.rejected.share is None
            else "yes"
            if tests.rejected.share <= SIZE_LIMIT
            else "**no**"
        )
        + ".",
        "",
        "## The contrast by world",
        "",
        "Per arm that read the draw: the panels tested and void, the contrast's rejection rate, "
        "the stop's rate, and over the tested panels the mean priced cost and the mean margin it "
        "is held to.",
        "",
        "| world | arm | tested | void | rejected | stopped | cost | margin |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        for arm in ("oracle", "wrong"):
            test = arm_test(run.runs[arm])
            lines.append(
                f"| {run.name} | {arm} | {test.tested} | {test.void} | "
                f"{_share(test.rejected)} {_interval(test.rejected)} | {_share(test.stopped)} | "
                f"{_show(test.cost)} | {_show(test.margin)} |"
            )
    loose = [reflagged(run, lambda arm_run: rejected(arm_run["dither"])) for run in runs]
    logger = [reflagged(run, lambda arm_run: arm_run["logger_flag"]) for run in runs]
    lines += [
        "",
        "## Secondary readings, not criteria",
        "",
        "**The contrast alone as the stop**, at any priced cost:",
        "",
        *_gate_rows(runs, loose),
        "",
        f"In {', '.join(_passes(runs, loose, 0.05)) or 'no world'} at 0.05.",
        "",
        "**The logger check's flag as the stop**, version 2's, on these panels:",
        "",
        *_gate_rows(runs, logger),
        "",
        f"In {', '.join(_passes(runs, logger, 0.05)) or 'no world'} at 0.05.",
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
            "| arm | reading | adjustment | acted | stopped | evaluated | left | lost | missed | "
            "silent | headroom (5%) | channel |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
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
                f"{_share(score_.left)} | {_share(score_.lost)} | {_share(score_.missed)} | "
                f"{_share(score_.silent)} {_interval(score_.silent)} | {headroom} | {channel} |"
            )
    lines += [
        "",
        "In each world's table, *stopped* is the stop's rate over the panels whose arm read the "
        "draw; the arm chosen by fit reads none. The true channel's push on supply is "
        f"{v1.DIRECT[0]:g} a period in every world. Wall-clock per world is recorded in "
        "`track_q_v3.json` and quoted nowhere: it was not measured on a clean machine.",
    ]
    return "\n".join(lines) + "\n"


def _compact(runs: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    """Version 2's columns, with each arm's check and the logger check's flag, so another stop
    can be read without a rerun."""

    def rounded(value: float | None) -> float | None:
        return None if value is None else float(f"{value:.6g}")

    def check(value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            key: (
                [rounded(entry) for entry in item]
                if isinstance(item, list)
                else rounded(item)
                if isinstance(item, float)
                else item
            )
            for key, item in value.items()
        }

    return v2._compact(runs) | {
        "dither": [check(run["dither"]) for run in runs],
        "logger_flag": [run["logger_flag"] for run in runs],
    }


def _record(
    runs: Sequence[WorldRun],
    replicates: int,
    seed: int,
    x64: bool,
    preregistration: str | None,
    chc_commit: str | None,
) -> dict[str, Any]:
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
            "dither": DITHER,
            "tolerance": TOLERANCE,
            "level": LEVEL,
            "margin": MARGIN,
            "threshold": THRESHOLD,
            "material": MATERIAL,
            "alpha": ALPHA,
            "cost_share": COST_SHARE,
            "size_limit": SIZE_LIMIT,
            "bonferroni": BONFERRONI,
        },
        "gate": asdict(gate(runs)),
        "size": asdict(size(runs)),
        "secondary": {
            "contrast_alone": {
                run.name: asdict(reflagged(run, lambda r: rejected(r["dither"]))) for run in runs
            },
            "logger_check": {
                run.name: asdict(reflagged(run, lambda r: r["logger_flag"])) for run in runs
            },
        },
        "wall_clock": "worlds[*].seconds; recorded, not quoted: not measured on a clean machine",
        "worlds": [
            {
                "world": asdict(WORLDS[run.name]),
                "scores": [asdict(s) for s in run.scores],
                "stop": asdict(run.stop),
                "tests": {arm: asdict(arm_test(run.runs[arm])) for arm in ("oracle", "wrong")},
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
    parser.add_argument("--name", default="track_q_v3", help="the report's file name, unsuffixed")
    parser.add_argument("--preregistration", help="the pre-registration's commit, to record")
    parser.add_argument("--chc-commit", help="the library commit this run imports, to record")
    args = parser.parse_args()
    if not has_dither():
        raise SystemExit(
            "this causal-hybrid-control's prescribe reads no logged draw (dither=); version 3 "
            "needs the release that adds it"
        )
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

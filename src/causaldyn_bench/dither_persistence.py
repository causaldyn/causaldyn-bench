"""Track Q v3's caveat: the dither contrast's size on the true graph under a persistent covariate.

Track Q v3 (``results/track_q_v3.md``) read ``prescribe(dither=)``'s contrast on the true graph at
a rate of 0.000 in every world but one: in **confounder AR**, whose promotion persists at 0.8, it
rejected 0.155 of 200 panels at 0.05. This study measures that rate per persistence level, on
version 1's confounder world (:mod:`causaldyn_bench.graph_errors`) with the promotion's
persistence set: 400 units x 12 periods, the promotion read by the logger and pushing supply, the
logger's draw logged as ``fresh``, the true graph handed to ``prescribe`` as Track Q's oracle arm
hands it. The library is whichever is installed, and the run records it: run once on a library and
once on another, the panels are the same.

**Why the rate moves with the persistence.** The plan's fit holds the adjustment set's push in its
drift. Up to causal-hybrid-control 1.3.1 the check's instrumented fit conditioned on the states
alone, its drift included, so its drift read the promotion's lean on the state into its slope: a
promotion that persists leans on the state, since its past pushed it. The two fits then estimate
two drifts under the true graph, and under ``rk4`` the drift reaches the channel through RK4's
growth factor. ``scripts/dither_persistence_lean.mac`` computes that lean exactly over the panel's
11 transitions; :func:`lean` reads it in floats.

Beside the size, the **power** where Track Q v3's gate passed, on the wrong graph: mediator,
collider latent and confounder AR, each world as Track Q builds it.

Precision: run with ``JAX_ENABLE_X64=1`` (``just dither-persistence``), Track Q's.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import multiprocessing
import os
import subprocess
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chc
import jax.numpy as jnp
import numpy as np

from causaldyn_bench import graph_errors as v1
from causaldyn_bench.graph_errors import PERIODS, STEP, UNITS, WORLDS, World, draw, prescribe
from causaldyn_bench.graph_errors_v2 import TOLERANCE
from causaldyn_bench.graph_errors_v3 import DITHER, has_dither
from causaldyn_bench.ope_calibration import clopper_pearson

STUDY = "dither-persistence"
SEED = 20261121
ALPHA = 0.05
REPLICATES = 200
POWER_REPLICATES = 50
PERSISTENCE = (0.0, 0.5, 0.8, 0.95)
BASE = "confounder AR"  # the world whose promotion's persistence each setting sets
POWER_WORLDS = ("mediator", "collider latent", "confounder AR")


@dataclass(frozen=True)
class Setting:
    """One cell of the study: a world, the graph handed to ``prescribe``, and the promotion's
    persistence where the cell sets it."""

    name: str
    world: str
    arm: str  # "oracle", the true graph, or "wrong"
    persistence: float | None = None  # None: the world's own


def settings() -> tuple[Setting, ...]:
    size = tuple(Setting(f"size {p:g}", BASE, "oracle", p) for p in PERSISTENCE)
    power = tuple(Setting(f"power {name}", name, "wrong") for name in POWER_WORLDS)
    return size + power


SETTINGS = {setting.name: setting for setting in settings()}


def world_of(setting: Setting) -> World:
    world = WORLDS[setting.world]
    if setting.persistence is None:
        return world
    assert world.confounder is not None
    confounder = dataclasses.replace(world.confounder, persistence=setting.persistence)
    return dataclasses.replace(world, confounder=confounder)


def lean(persistence: float, read: float = 1.0, push: float = 0.2) -> tuple[float, float]:
    """The pooled slope of the promotion's push, per unit of time, on (supply, wait) over a
    panel's ``PERIODS - 1`` transitions: what a drift on the states alone reads into its supply
    row beside a drift that holds the promotion. The covariance recursion of
    ``scripts/dither_persistence_lean.mac`` in floats."""
    gain = read * v1.DIRECT + np.array([push, 0.0])
    step = np.zeros((3, 3))
    step[:2, :2], step[:2, 2], step[2, 2] = STEP, gain, persistence
    spread = v1.SPREAD**2 * np.outer(v1.DIRECT, v1.DIRECT) + v1.SHOCK**2 * np.eye(2)
    spread[0, 0] += v1.DEMAND_PUSH**2
    noise = np.zeros((3, 3))
    noise[:2, :2], noise[2, 2] = spread, 1.0 - persistence**2
    covariance = np.diag([0.2**2, 0.2**2, 1.0])
    states, cross = np.zeros((2, 2)), np.zeros(2)
    for _ in range(PERIODS - 1):
        states += covariance[:2, :2]
        cross += covariance[:2, 2]
        covariance = step @ covariance @ step.T + noise
    slope = np.linalg.solve(states, cross) * push / v1.DT
    return float(slope[0]), float(slope[1])


def replicate(job: tuple[str, int], seed: int = SEED) -> dict[str, Any]:
    """One panel of ``setting``, drawn from ``(seed, setting's index, index, 0)``, and the check
    ``prescribe`` reads on it."""
    logging.getLogger("chc").setLevel(logging.ERROR)  # a warning per rejected panel is the data
    name, index = job
    setting = SETTINGS[name]
    world = world_of(setting)
    fitted = draw(world, (seed, list(SETTINGS).index(name), index, 0), dither=DITHER)
    panel = chc.Panel.from_frame(fitted.columns, unit="unit", time="time", seed=0)
    edges = world.true_edges if setting.arm == "oracle" else world.wrong_edges
    latent = world.true_latent if setting.arm == "oracle" else ()
    graph = chc.CausalGraph.from_edges(edges, latent=latent)
    check = prescribe(panel, graph, TOLERANCE, dither=DITHER).dither_check
    if check is None:
        raise RuntimeError("prescribe read no dither check on a panel that logs the draw")
    contrast = check.contrast
    return {
        "index": index,
        "status": check.status,
        "p_value": None if contrast is None else contrast.p_value,
        "cost": None if contrast is None else contrast.cost,
        "noise": None if contrast is None else contrast.noise,
        "share": check.share,
    }


@dataclass(frozen=True)
class Cell:
    setting: Setting
    tested: int
    void: int
    rejected: int
    rate: float | None
    mc_error: float | None  # the binomial standard error at the measured rate
    interval: tuple[float, float] | None  # Clopper-Pearson, 95%
    median_p: float | None
    # the rates at 0.1 and 0.2: where the p-values' law sits beside the level's tail
    rate_10: float | None
    rate_20: float | None
    mean_cost: float | None  # the contrast's priced cost, (d' W d - tr(W S)) / 2, on average
    mean_noise: float | None  # tr(W S) / 2 on average: the variance the contrast reads
    lean: tuple[float, float] | None  # :func:`lean` at the cell's persistence, size cells only


def cell(setting: Setting, rows: Sequence[dict[str, Any]]) -> Cell:
    tested = [row for row in rows if row["status"] == "tested"]
    n = len(tested)
    p_values = np.array([row["p_value"] for row in tested], dtype=float)
    count = int(np.sum(p_values <= ALPHA))
    rate = count / n if n else None

    def mean(key: str) -> float | None:
        return float(np.mean([row[key] for row in tested])) if n else None

    return Cell(
        setting=setting,
        tested=n,
        void=len(rows) - n,
        rejected=count,
        rate=rate,
        mc_error=None if rate is None else math.sqrt(rate * (1.0 - rate) / n),
        interval=clopper_pearson(count, n) if n else None,
        median_p=float(np.median(p_values)) if n else None,
        rate_10=float(np.mean(p_values <= 0.1)) if n else None,
        rate_20=float(np.mean(p_values <= 0.2)) if n else None,
        mean_cost=mean("cost"),
        mean_noise=mean("noise"),
        lean=None if setting.persistence is None else lean(setting.persistence),
    )


def library_commit() -> str | None:
    """The installed library's commit, where it is imported from a git checkout."""
    root = Path(chc.__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _rate(c: Cell) -> str:
    if c.rate is None or c.interval is None or c.mc_error is None:
        return "—"
    low, high = c.interval
    return f"{c.rate:.3f} ± {c.mc_error:.3f} [{low:.3f}, {high:.3f}]"


def markdown(records: Sequence[dict[str, Any]]) -> str:
    """The size and power tables, one column per run."""
    labels = list(dict.fromkeys(record["label"] for record in records))
    commits = {
        label: {
            (r["chc_version"], r["chc_commit"], r["x64"]) for r in records if r["label"] == label
        }
        for label in labels
    }
    for label, seen in commits.items():
        if len(seen) != 1:
            raise ValueError(f"the records of {label!r} come from more than one library: {seen}")
    # each cell read again from the record's rows, which are what the run measured
    by_label: dict[str, dict[str, Cell]] = {label: {} for label in labels}
    for record in records:
        for name, rows in record["rows"].items():
            by_label[record["label"]][name] = cell(SETTINGS[name], rows)
    head = (
        "| setting | world | persistence | lean on (supply, wait) | "
        + " | ".join(f"{label}: rejected at 0.05 (± MC error) [95%]" for label in labels)
        + " |"
    )
    rule = "|---|---|---|---|" + "---|" * len(labels)
    lines = [
        "# The dither contrast under a persistent covariate",
        "",
        f"Track Q v3's caveat (`results/track_q_v3.md`): on the true graph, `prescribe(dither=)`'s "
        f"contrast rejected 0.155 of 200 panels at 0.05 in confounder AR and 0.000 in every other "
        f"world. `causaldyn_bench.dither_persistence`, seed {SEED}, panels of {UNITS} units x "
        f"{PERIODS} periods, the logger's draw logged as `{DITHER}`, `JAX_ENABLE_X64` "
        f"{records[0]['x64']}. Runs: "
        + "; ".join(
            f"**{label}** on causal-hybrid-control {version} at `{commit}`"
            for label in labels
            for version, commit, _ in commits[label]
        )
        + ". Each run draws the same panels.",
        "",
        "## Size: the true graph, the promotion's persistence set",
        "",
        "Confounder AR's world (a promotion the logger reads, unit variance, pushing supply by "
        "0.2 a period) at each persistence; the oracle arm's graph. The lean is the slope of the "
        "promotion's push on the state that a drift on the states alone reads, per unit of time, "
        "over the panel's 11 transitions (`scripts/dither_persistence_lean.mac`).",
        "",
        head,
        rule,
    ]
    for setting in settings():
        if setting.arm != "oracle":
            continue
        cells = [by_label[label].get(setting.name) for label in labels]
        if all(c is None for c in cells):
            continue
        first = next(c for c in cells if c is not None)
        lean_text = "—" if first.lean is None else f"({first.lean[0]:.3f}, {first.lean[1]:.3f})"
        lines.append(
            f"| {setting.name} | {setting.world} | {setting.persistence:g} | {lean_text} | "
            + " | ".join("—" if c is None else _rate(c) for c in cells)
            + " |"
        )
    lines += [
        "",
        "## Power: the wrong graph, where Track Q v3's gate passed",
        "",
        "| setting | world | the wrong graph | "
        + " | ".join(f"{label}: rejected at 0.05 (± MC error) [95%]" for label in labels)
        + " |",
        "|---|---|---|" + "---|" * len(labels),
    ]
    for setting in settings():
        if setting.arm != "wrong":
            continue
        cells = [by_label[label].get(setting.name) for label in labels]
        if all(c is None for c in cells):
            continue
        lines.append(
            f"| {setting.name} | {setting.world} | {WORLDS[setting.world].error} | "
            + " | ".join("—" if c is None else _rate(c) for c in cells)
            + " |"
        )
    lines += [
        "",
        "## The p-values' law, every cell",
        "",
        "The rates at 0.1 and 0.2, the median p-value, and on average the priced cost "
        "`(d' W d - tr(W S)) / 2` and the noise `tr(W S) / 2`, the variance the contrast reads: on "
        "the true graph the cost averages 0 where `S` is the difference's variance and the two "
        "fits estimate one thing.",
        "",
        "| setting | run | rejected at 0.1 | at 0.2 | median p | mean cost | mean noise |",
        "|---|---|---|---|---|---|---|",
    ]
    for setting in settings():
        for label in labels:
            c = by_label[label].get(setting.name)
            if c is None or c.rate_10 is None or c.rate_20 is None or c.median_p is None:
                continue
            lines.append(
                f"| {setting.name} | {label} | {c.rate_10:.3f} | {c.rate_20:.3f} | "
                f"{c.median_p:.3f} | {c.mean_cost:.3g} | {c.mean_noise:.3g} |"
            )
    lines += [
        "",
        "Void panels: "
        + "; ".join(f"{label} {sum(c.void for c in by_label[label].values())}" for label in labels)
        + ". The MC error is the binomial standard error at the measured rate; the interval is "
        "Clopper-Pearson's. Wall-clock is recorded in the JSON and quoted nowhere: it was not "
        "measured on a clean machine.",
        "",
    ]
    return "\n".join(lines)


def run(
    names: Sequence[str], replicates: dict[str, int], workers: int, seed: int
) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"  # the pool is the parallelism
    context = multiprocessing.get_context("spawn")
    cells, rows, seconds = [], {}, {}
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        for name in names:
            started = time.perf_counter()
            jobs = [(name, index) for index in range(replicates[name])]
            rows[name] = list(pool.map(replicate, jobs, [seed] * len(jobs)))
            seconds[name] = time.perf_counter() - started
            cells.append(cell(SETTINGS[name], rows[name]))
            print(f"{name}: {_rate(cells[-1])}", flush=True)
    return {"cells": [asdict(c) for c in cells], "rows": rows, "seconds": seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run", help="measure the settings on the installed library")
    go.add_argument("--label", required=True, help="the run's name in the report")
    go.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    go.add_argument("--replicates", type=int, default=REPLICATES)
    go.add_argument("--power-replicates", type=int, default=POWER_REPLICATES)
    go.add_argument("--seed", type=int, default=SEED)
    go.add_argument("--workers", type=int, default=4)
    go.add_argument("--out", type=Path, default=Path("results/dither_persistence"))
    report = sub.add_parser("report", help="write the tables from the runs' records")
    report.add_argument("records", nargs="+", type=Path)
    report.add_argument("--out", type=Path, default=Path("results/dither_persistence.md"))
    args = parser.parse_args()
    if args.command == "report":
        records = [json.loads(path.read_text()) for path in args.records]
        args.out.write_text(markdown(records))
        print(f"written to {args.out}")
        return
    if not has_dither():
        raise SystemExit("this causal-hybrid-control's prescribe reads no logged draw (dither=)")
    counts = {
        name: args.power_replicates if SETTINGS[name].arm == "wrong" else args.replicates
        for name in args.settings
    }
    measured = run(args.settings, counts, args.workers, args.seed)
    record = {
        "study": STUDY,
        "label": args.label,
        "seed": args.seed,
        "x64": bool(jnp.zeros(()).dtype == jnp.float64),
        "chc_version": chc.__version__,
        "chc_commit": library_commit(),
        "alpha": ALPHA,
        "units": UNITS,
        "periods": PERIODS,
        "wall_clock": "seconds; recorded, not quoted: not measured on a clean machine",
        **measured,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    slug = "+".join(name.replace(" ", "-") for name in args.settings)
    path = args.out / f"{args.label}--{slug}.json"
    path.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"written to {path}")


if __name__ == "__main__":
    main()

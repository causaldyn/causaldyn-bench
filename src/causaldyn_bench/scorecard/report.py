"""A family's report: each rung's cells, one an environment and one pooled over them, every arm's
columns over the look each environment holds whole; the family's compute as its records hold it;
and, for a scored run, the pre-registration it ran under.

A look at ``n`` reads the first ``n`` worlds of each environment's seeded order
(:func:`~causaldyn_bench.scorecard.scoring.look`), so a report of a run still under way reads a
prefix of it. Each cell gives every arm's columns (:func:`columns`):

* its regret per euro, the mean with its 95 % Student-t interval; its failures; the plans moved
  into the box;
* where it forecasts, the CRPS of its draws of the quarter's sales at the status quo, in units of
  the history's mean weekly sales, and beside it the field's MAPE on that holdout, the draws' mean
  against the sales, as a share;
* where its response was read, the share of the channels whose true ROI and mROI the draws' central
  80 %, 90 % and 95 % hold, pooled over the look's channels;
* and its effect shares against the spend shares (:func:`shares`): a channel's effect share is its
  ROI times its spend over the window its returns are read on, over the channels' sum. The arm's,
  from its draws' mean ROI, less the truth's is regressed through the origin on the spend share
  less the truth's effect share, pooled over the channels of the look. Its slope, the *pull*, is 0
  where the arm's shares miss the truth's in no direction of the spend's and 1 where they sit on
  the spend shares; the mean absolute miss is beside it. An arm that reads two channels whose spend
  moves together (family 3) and splits their effect by their spend has a pull near 1.

The **compute** (:func:`compute`) is each arm's, over every record of the report, from the fields
its records' ``cost`` holds: ``cpu_model`` and ``threads``, as recorded; ``peak_rss_bytes``, its
median and largest; ``wall_seconds``, its median a fit; and ``cpu_seconds``, summed into
core-hours. A record that holds none, as one whose rung shares another rung's fit, counts in no
column, and a column no record holds reads ``not recorded``. The records were made under whatever
load the run's machine carried: the table says what was spent, not how fast anything is.

A **scored** report, one that holds a world that is not a pilot, prints the pre-registration its
run declares (:class:`~causaldyn_bench.scorecard.preregistration.Stamp`) in its header and its
JSON, and :func:`report` refuses to write one without it. The markdown is written from the JSON
(:func:`markdown`), so the two cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from causaldyn_bench.budget_regret import Mean
from causaldyn_bench.scorecard.family import Family, number
from causaldyn_bench.scorecard.preregistration import Stamp
from causaldyn_bench.scorecard.scoring import LEVELS, ArmScore, WorldRecord, look, percent

POOLED = "pooled"
GIB = 2.0**30
# the fields of a record's ``cost`` the compute table reads
COST = ("cpu_model", "threads", "peak_rss_bytes", "wall_seconds", "cpu_seconds")


def whole(records: Sequence[WorldRecord]) -> int:
    """The largest look every environment of ``records`` holds whole: its first worlds in order."""
    held: dict[str, set[int]] = {}
    for r in records:
        held.setdefault(r.environment, set()).add(r.index)
    if not held:
        return 0
    return min(
        next(i for i in range(len(places) + 1) if i not in places) for places in held.values()
    )


def _mean(values: Sequence[float]) -> dict[str, Any]:
    """The values' mean, with its 95 % Student-t interval where there are two or more."""
    if len(values) < 2:
        return {"mean": float(values[0]), "low": None, "high": None, "n": len(values)}
    m = Mean.of(np.asarray(values, dtype=float))
    return {"mean": m.mean, "low": m.interval[0], "high": m.interval[1], "n": len(values)}


def _coverage(scores: Sequence[ArmScore]) -> dict[str, Any] | None:
    """The share of the read channels whose truth each central share held, pooled over the look;
    every world of a family reads the same channels, so the mean of the worlds' shares is it."""
    read = [s for s in scores if s.roi_coverage is not None and s.mroi_coverage is not None]
    if not read:
        return None
    return {
        key: {
            percent(level): float(np.mean([getattr(s, key)[percent(level)] for s in read]))
            for level in LEVELS
        }
        for key in ("roi_coverage", "mroi_coverage")
    } | {"worlds": len(read)}


def _effect_shares(record: WorldRecord, score: ArmScore) -> tuple[list[float], ...] | None:
    """The arm's, the truth's and the spend's shares of ``record``'s channels, or None where the
    arm's response was not read, the world holds no spend, or the arm's returns sum to no more
    than 0."""
    if score.returns is None or not record.spend:
        return None
    names = list(record.returns)
    spend = np.array([record.spend[n] for n in names])
    truth = np.array([record.returns[n]["roi"] for n in names]) * spend
    arm = np.array([score.returns[n]["roi"][0] for n in names]) * spend
    if not arm.sum() > 0.0:
        return None
    return tuple((v / v.sum()).tolist() for v in (arm, truth, spend))


def shares(records: Sequence[WorldRecord], arm: str) -> dict[str, Any] | None:
    """``arm``'s effect shares against the spend shares over ``records``: the pull and the mean
    absolute miss, or None where no world read its response."""
    read = [
        found
        for r in records
        if arm in r.arms and (found := _effect_shares(r, r.arms[arm])) is not None
    ]
    if not read:
        return None
    ours, truth, spend = (np.concatenate([np.asarray(f[i]) for f in read]) for i in range(3))
    miss, lean = ours - truth, spend - truth
    scale = float(lean @ lean)
    return {
        "pull": float(miss @ lean) / scale if scale > 0.0 else None,
        "miss": float(np.mean(np.abs(miss))),
        "channels": int(miss.size),
    }


def columns(seen: Sequence[WorldRecord], arm: str) -> dict[str, Any]:
    """``arm``'s columns over the look ``seen``."""
    scores = [r.arms[arm] for r in seen]
    crps = [s.crps for s in scores if s.crps is not None]
    mape = [s.mape for s in scores if s.mape is not None]
    return {
        "regret": _mean([s.regret for s in scores]),
        "failed": sum(s.failure is not None for s in scores),
        "moved": sum(s.moved is not None for s in scores),
        "crps": _mean(crps) if crps else None,
        "mape": _mean(mape) if mape else None,
        "coverage": _coverage(scores),
        "shares": shares(seen, arm),
    }


def compute(records: Sequence[WorldRecord]) -> dict[str, dict[str, Any]]:
    """Each arm's compute over ``records``, as their ``cost`` fields hold it."""
    arms = list(dict.fromkeys(arm for r in records for arm in r.arms))
    table = {}
    for arm in arms:
        costs = [r.arms[arm].record.get("cost") or {} for r in records if arm in r.arms]
        held = {key: [c[key] for c in costs if c.get(key) is not None] for key in COST}
        peak = np.asarray(held["peak_rss_bytes"], dtype=float) / GIB
        table[arm] = {
            "records": len(costs),
            "cpu_model": sorted(set(map(str, held["cpu_model"]))) or None,
            "threads": sorted(set(map(str, held["threads"]))) or None,
            "peak_rss_gib": {
                "median": float(np.median(peak)),
                "most": float(peak.max()),
                "n": int(peak.size),
            }
            if peak.size
            else None,
            "wall_seconds_a_fit": {
                "median": float(np.median(held["wall_seconds"])),
                "n": len(held["wall_seconds"]),
            }
            if held["wall_seconds"]
            else None,
            "core_hours": {
                "total": float(np.sum(held["cpu_seconds"])) / 3600.0,
                "n": len(held["cpu_seconds"]),
            }
            if held["cpu_seconds"]
            else None,
        }
    return table


def report(
    family: Family[Any],
    records: Sequence[WorldRecord],
    *,
    preregistration: Path | str | None = None,
) -> dict[str, Any]:
    """The report of ``family``'s ``records``, as JSON holds it; ``preregistration`` the file the
    run declares, required where a record is scored.

    Raises:
        ValueError: on records of another family or none, pilots beside scored worlds, or
            scored worlds without a pre-registration.
        FileNotFoundError: no file is at ``preregistration``.
    """
    if not records:
        raise ValueError("a report reads at least one world's record")
    families = {r.family for r in records}
    if families != {number(family)}:
        raise ValueError(f"{family.name}'s report reads family {number(family)}, not {families}")
    if len({r.pilot for r in records}) > 1:
        raise ValueError("a report reads a run's pilots or its scored worlds, not both")
    scored = not records[0].pilot
    if scored and preregistration is None:
        raise ValueError("a scored run's report prints the pre-registration the run declares")
    stamp = None if preregistration is None else Stamp.of(preregistration)
    cells = []
    for k in sorted({r.k for r in records}):
        at = [r for r in records if r.k == k]
        environments = list(dict.fromkeys(r.environment for r in at))
        for environment in (*environments, POOLED):
            group = at if environment == POOLED else [r for r in at if r.environment == environment]
            n = whole(group)
            seen = look(group, n) if n else ()
            cells.append(
                {
                    "k": k,
                    "environment": environment,
                    "worlds": n,
                    "look": len(seen),
                    "arms": {arm: columns(seen, arm) for arm in seen[0].arms} if seen else {},
                }
            )
    return {
        "family": number(family),
        "name": family.name,
        "kind": "scored" if scored else "pilot",
        "preregistration": None if stamp is None else stamp.as_json(),
        "labels": list(family.labels),
        "cells": cells,
        "compute": compute(records),
    }


def _number(value: Mapping[str, Any] | None, digits: int = 4, scale: float = 1.0) -> str:
    if value is None:
        return "n/a"
    mean = f"{scale * value['mean']:+.{digits}f}"
    if value.get("low") is None:
        return mean
    return f"{mean} [{scale * value['low']:+.{digits}f}, {scale * value['high']:+.{digits}f}]"


def _coverage_text(coverage: Mapping[str, Any] | None) -> str:
    if coverage is None:
        return "n/a"
    return "; ".join(
        f"{key.split('_')[0]} "
        + "/".join(f"{coverage[key][percent(level)]:.3f}" for level in LEVELS)
        for key in ("roi_coverage", "mroi_coverage")
    )


def _shares_text(shares: Mapping[str, Any] | None) -> str:
    if shares is None:
        return "n/a"
    pull = "n/a" if shares["pull"] is None else f"{shares['pull']:+.3f}"
    return f"pull {pull}, miss {shares['miss']:.3f}"


def _cost_text(row: Mapping[str, Any]) -> list[str]:
    def listed(values: Sequence[str] | None) -> str:
        return "not recorded" if values is None else ", ".join(values)

    peak, wall, core = row["peak_rss_gib"], row["wall_seconds_a_fit"], row["core_hours"]
    return [
        listed(row["cpu_model"]),
        listed(row["threads"]),
        "not recorded" if peak is None else f"{peak['median']:.2f} / {peak['most']:.2f}",
        "not recorded" if wall is None else f"{wall['median']:.1f}",
        "not recorded" if core is None else f"{core['total']:.3f}",
    ]


def markdown(data: Mapping[str, Any]) -> str:
    """The report's markdown, written from its JSON (:func:`report`)."""
    levels = "/".join(percent(level) for level in LEVELS)
    lines = [f"# Family {data['family']}: {data['name']}, {data['kind']}", ""]
    stamp = data["preregistration"]
    if stamp is not None:
        lines += [Stamp.from_json(stamp).line, ""]
    lines += [f"By construction: {label}" for label in data["labels"]]
    for cell in data["cells"]:
        lines += ["", f"## k = {cell['k']}, {cell['environment']}", ""]
        if not cell["arms"]:
            lines.append("No environment holds its first world yet.")
            continue
        lines += [
            f"{cell['look']} worlds, the first {cell['worlds']} of each environment.",
            "",
            f"| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage {levels} "
            "| effect against spend share |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for arm, c in cell["arms"].items():
            lines.append(
                f"| {arm} | {_number(c['regret'])} | {c['failed']} | {c['moved']} "
                f"| {_number(c['crps'])} | {_number(c['mape'], 2, 100.0)} "
                f"| {_coverage_text(c['coverage'])} | {_shares_text(c['shares'])} |"
            )
    lines += [
        "",
        "## Compute, as recorded",
        "",
        "Recorded under the load the run's machine carried: what was spent, not a speed.",
        "",
        "| arm | records | CPU model | threads | peak RSS, GiB, median / most "
        "| wall a fit, s, median | core-hours |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, row in data["compute"].items():
        lines.append(f"| {arm} | {row['records']} | " + " | ".join(_cost_text(row)) + " |")
    return "\n".join(lines) + "\n"

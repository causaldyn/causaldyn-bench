"""An arm fitted again as weeks arrive: how far its returns and its plan swing between refits, and
how its forecasts read from a rolling origin.

**The refresh swing** (:func:`windows`, :func:`swing`). A world's export is cut to its first
``T - 4 (6 - i)`` weeks for ``i = 0, ..., 6``: a first fit and six refits, each on four more
weeks than the last, the last on the whole history. Every cut plans the world's one quarter, its
budget, box and status quo, so every plan is scored on the world's one truth; its controls over
the quarter are the 13 weeks after its cut, the history's and then the quarter's, as a planner
who refits on those weeks would read them; and its lift tests are those whose readout ends inside
it. Each fit's returns are read on one window, the year that ends at the first cut's last week,
inside every cut, from the draws of its mapped response (:mod:`.mapping`, :mod:`.returns`), so a
return that moves between refits moves on the same spend. Read for each channel: the largest
move of the draws' mean between consecutive fits, and the refits whose mean leaves the previous
fit's central 80 %; and for the plans, each refit's regret per euro on the truth less the
previous fit's, the regret of plan ``t`` against plan ``t - 1``.

**The rolling origin** (:func:`origins`, :func:`rolling`). A world whose effect holds still is cut
to the same seven lengths of history, and each cut is a world of its own: its quarter, the 13
weeks after the cut at its status quo, and its truth. An arm reads what its family hands it of
each cut, so on family 7 (:mod:`.short`) it reads the last ``L`` weeks before each origin, a
window that rolls. Read beside the swing's columns: each origin's CRPS of the arm's forecast of
its quarter at the status quo, in units of the history's mean weekly sales
(:mod:`.scoring`), and each fit's returns read on the weeks every origin's window holds.

A fit that raised is counted and plays the status quo, as the scorer plays it; a response that is
missing or off its tool's decomposition is not read, and the reading says how many were not.
:func:`report` lays the readings out as JSON and :func:`markdown` writes them from it.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from causaldyn_bench.endogenous_mmm import YEAR, MediaMixWorld
from causaldyn_bench.lift_calibration import COOLDOWN, TEST
from causaldyn_bench.scorecard.family import Family, Truth
from causaldyn_bench.scorecard.mapping import TOLERANCE, residual
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.returns import drawn_returns
from causaldyn_bench.scorecard.scoring import _score, percent
from causaldyn_bench.scorecard.track_m2 import World, settled, still

STEP = 4  # the weeks each refit reads beyond the fit before it
REFITS = 6  # the refits after the first fit
LEVEL = 0.8  # the central share of a fit's draws a refit's mean is read against
# a world's history's fields that run over its weeks, which a cut shortens
WEEKLY = (
    "week",
    "sales",
    "baseline",
    "promotion",
    "premedia",
    "media",
    "spend",
    "seasonality",
    "quality",
    "price_position",
    "sentiment",
    "discount",
    "budget",
    "bidding",
    "plan_noise",
    "observed_promotion",
    "observed_price",
)


def lengths(weeks: int, step: int = STEP, refits: int = REFITS) -> tuple[int, ...]:
    """The weeks of history each fit reads, first to last: ``weeks - step (refits - i)``.

    Raises:
        ValueError: on a step or a count of refits below 1, or a first fit of fewer than a year.
    """
    if step < 1 or refits < 1:
        raise ValueError(f"refits add at least one week, at least once: {step}, {refits}")
    first = weeks - step * refits
    if first < YEAR:
        raise ValueError(f"the first fit reads {first} weeks, fewer than a year")
    return tuple(first + step * i for i in range(refits + 1))


def _cut(observation: Observation, weeks: int, window: tuple[int, int]) -> Observation:
    """``observation`` cut to its first ``weeks`` weeks, planning its quarter; see the module."""
    rows = observation.lift
    kept = [i for i, start in enumerate(rows.start) if start + TEST + COOLDOWN - 1 <= weeks]
    index = np.asarray(kept, dtype=np.int64)
    lift = dataclasses.replace(
        rows,
        channel=tuple(rows.channel[i] for i in kept),
        start=tuple(rows.start[i] for i in kept),
        x=np.asarray(rows.x)[index],
        delta_x=np.asarray(rows.delta_x)[index],
        delta_y=np.asarray(rows.delta_y)[index],
        sigma=np.asarray(rows.sigma)[index],
        gaps=None if rows.gaps is None else np.asarray(rows.gaps)[index],
    )
    future = observation.future_controls or {}
    planned = observation.planned
    return dataclasses.replace(
        observation,
        sales=observation.sales[:weeks],
        spend=observation.spend[:weeks],
        controls={name: series[:weeks] for name, series in observation.controls.items()},
        future_controls={
            name: np.concatenate([series[weeks:], future[name]])[:planned]
            for name, series in observation.controls.items()
        },
        roi_window=window,
        lift=lift,
        impressions=None if observation.impressions is None else observation.impressions[:weeks],
    )


def windows(
    observation: Observation, step: int = STEP, refits: int = REFITS
) -> tuple[Observation, ...]:
    """The first fit's export and each refit's, in order, the last the whole history: see the
    module.

    Raises:
        ValueError: on a geo export, an export whose history holds its tests, one that gives no
            controls over the quarter, and what :func:`lengths` refuses.
    """
    if observation.population is not None:
        raise ValueError("a refresh swing cuts a national export")
    if observation.held is not None:
        raise ValueError("an export whose history holds its tests is not cut")
    if observation.future_controls is None:
        raise ValueError("a refresh swing reads an export that gives the quarter's controls")
    cuts = lengths(observation.sales.shape[0], step, refits)
    window = (cuts[0] - YEAR + 1, cuts[0])
    return tuple(_cut(observation, weeks, window) for weeks in cuts)


def cut(world: World, weeks: int) -> World:
    """``world`` with its history's first ``weeks`` weeks alone, settled again: the quarter after
    them at its status quo, each channel's effect holding still.

    Raises:
        ValueError: on a world whose effect moves, that prices its impressions or measures a
            control of its own, or a cut outside its history.
    """
    history = world.history
    if (
        world.price is not None
        or world.measured
        or not np.array_equal(world.expected, still(history))
        or not np.array_equal(world.realised, world.expected)
    ):
        raise ValueError("a rolling origin cuts a world whose effect holds still, and no more")
    if not 1 <= weeks <= history.week.size:
        raise ValueError(f"a cut keeps 1 to {history.week.size} weeks, not {weeks}")
    shorter: MediaMixWorld = dataclasses.replace(
        history, **{name: getattr(history, name)[:weeks] for name in WEEKLY}
    )
    return settled(world.environment, world.seed, world.generator, shorter)


def origins(world: World, step: int = STEP, refits: int = REFITS) -> tuple[World, ...]:
    """The world cut at each origin, first to last, the last the world's whole history.

    Raises:
        ValueError: what :func:`cut` and :func:`lengths` refuse.
    """
    return tuple(cut(world, weeks) for weeks in lengths(world.history.week.size, step, refits))


@dataclass(frozen=True)
class Fit:
    """One fit of a refresh or an origin, read: its plan's regret per euro on its truth, why it
    failed where it did, each channel's returns on the reading's window, as the draws' mean and the
    ends of their central :data:`LEVEL`, and why they were not read where they were not, and its
    forecast's CRPS where the reading scores one."""

    regret: float
    failure: str | None
    returns: dict[str, tuple[float, float, float]] | None
    unread: str | None
    crps: float | None


def _fit(
    truth: Truth,
    observation: Observation,
    reading: Observation,
    record: Mapping[str, Any],
    forecasts: bool,
) -> Fit:
    """``record`` of ``observation`` read: its plan on ``truth``, its returns on ``reading``'s
    window, its forecast where ``forecasts``.

    Raises:
        RuntimeError: the record was fitted to other data.
    """
    if record["digest"] != observation.digest(int(record.get("version", observation.version))):
        raise RuntimeError("the record was fitted to other data")
    scored = _score(truth, record)
    returns = unread = None
    response = record.get("response")
    if scored.failure:
        unread = "the fit failed"
    elif response is None:
        unread = "no response"
    elif "unmapped" in response:
        unread = f"unmapped: {response['unmapped']}"
    elif (gap := residual(response, observation)) > TOLERANCE:
        unread = f"{gap:.3g} off its tool's decomposition, beyond {TOLERANCE:g}"
    else:
        drawn = drawn_returns(response, reading).roi
        low, high = np.quantile(drawn, [(1.0 - LEVEL) / 2.0, (1.0 + LEVEL) / 2.0], axis=0)
        returns = {
            name: (float(drawn[:, c].mean()), float(low[c]), float(high[c]))
            for c, name in enumerate(observation.channels)
        }
    return Fit(scored.regret, scored.failure, returns, unread, scored.crps if forecasts else None)


def _read(fits: Sequence[Fit], channels: Sequence[str], forecasts: bool) -> dict[str, Any]:
    """The columns of one arm's fits of one world, first to last."""
    regrets = [f.regret for f in fits]
    moves = np.diff(regrets)
    channel = {}
    for name in channels:
        leaves, pairs, largest = 0, 0, None
        for before, after in itertools.pairwise(fits):
            if before.returns is None or after.returns is None:
                continue
            mean, (_, low, high) = after.returns[name][0], before.returns[name]
            move = abs(mean - before.returns[name][0])
            largest = move if largest is None else max(largest, move)
            leaves += not low <= mean <= high
            pairs += 1
        means = [None if f.returns is None else f.returns[name][0] for f in fits]
        channel[name] = {"means": means, "swing": largest, "leaves": leaves, "pairs": pairs}
    out: dict[str, Any] = {
        "fits": len(fits),
        "failed": sum(f.failure is not None for f in fits),
        "unread": sum(f.returns is None for f in fits),
        "regret": regrets,
        "moves": moves.tolist(),
        "largest move": float(np.max(np.abs(moves))),
        "channels": channel,
    }
    if forecasts:
        crps = [f.crps for f in fits if f.crps is not None]
        out["crps"] = [f.crps for f in fits]
        out["mean crps"] = float(np.mean(crps)) if crps else None
    return out


def _records(records: Sequence[Mapping[str, Any]], count: int) -> None:
    if len(records) != count:
        raise ValueError(f"a reading takes one record a fit, {count}, not {len(records)}")


def swing(
    family: Family[World],
    world: World,
    k: int,
    records: Sequence[Mapping[str, Any]],
    step: int = STEP,
    refits: int = REFITS,
) -> dict[str, Any]:
    """One arm's refresh swing on ``world`` at rung ``k``: its ``records``, one a cut of
    :func:`windows` in order, read on the world's truth.

    Raises:
        ValueError: on a count of records that is not the cuts', and what :func:`windows` refuses.
        RuntimeError: a record was fitted to another cut.
    """
    cuts = windows(family.observe(world, k), step, refits)
    _records(records, len(cuts))
    truth = family.truth(world)
    fits = [_fit(truth, c, c, r, forecasts=False) for c, r in zip(cuts, records, strict=True)]
    return {"weeks": [c.sales.shape[0] for c in cuts], "window": list(cuts[0].roi_window)} | _read(
        fits, cuts[0].channels, forecasts=False
    )


def rolling(
    family: Family[World],
    world: World,
    k: int,
    records: Sequence[Mapping[str, Any]],
    step: int = STEP,
    refits: int = REFITS,
) -> dict[str, Any]:
    """One arm's rolling origin on ``world`` at rung ``k``: its ``records``, one an origin of
    :func:`origins` in order, each read on its origin's truth, its returns on the weeks every
    origin's window holds, numbered from the world's first week.

    Raises:
        ValueError: on a count of records that is not the origins', windows that share no week,
            and what :func:`origins` refuses.
        RuntimeError: a record was fitted to another origin.
    """
    cuts = origins(world, step, refits)
    _records(records, len(cuts))
    observed = [family.observe(c, k) for c in cuts]
    ends = [c.history.week.size for c in cuts]
    starts = [end - o.sales.shape[0] for end, o in zip(ends, observed, strict=True)]
    first, last = max(starts) + 1, min(ends)
    if first > last:
        raise ValueError(f"the origins' windows share no week: {first} to {last}")
    fits = [
        _fit(
            family.truth(c),
            o,
            dataclasses.replace(o, roi_window=(first - start, last - start)),
            r,
            forecasts=True,
        )
        for c, o, start, r in zip(cuts, observed, starts, records, strict=True)
    ]
    return {"weeks": ends, "window": [first, last]} | _read(fits, observed[0].channels, True)


def report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The readings ``rows`` as JSON holds them, each with its ``kind``, ``swing`` or
    ``rolling``, its family, environment, world, rung and arm, and its reading."""
    keys = ("kind", "family", "environment", "seed", "k", "arm", "reading")
    for row in rows:
        missing = [key for key in keys if key not in row]
        if missing or row["kind"] not in ("swing", "rolling"):
            raise ValueError(f"a row names {keys}, its kind swing or rolling: {sorted(row)}")
    return {"step": STEP, "refits": REFITS, "level": LEVEL, "rows": [dict(r) for r in rows]}


def _number(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def markdown(data: Mapping[str, Any]) -> str:
    """The section's markdown, written from its JSON (:func:`report`)."""
    level = percent(data["level"])
    lines = [
        "## Refresh swing and rolling origin",
        "",
        f"Each arm fitted on a history and refitted {data['refits']} times, {data['step']} weeks "
        "more each time. A channel's swing is the largest move of its return's mean between "
        f"consecutive fits, read on one window; leaves counts the refits whose mean lies outside "
        f"the fit before's central {level} %. The regret's move is a refit's regret per euro on "
        "the truth less the fit before's.",
    ]
    for kind, title in (("swing", "Refresh swing"), ("rolling", "Rolling origin")):
        rows = [r for r in data["rows"] if r["kind"] == kind]
        if not rows:
            continue
        crps = " | mean CRPS" if kind == "rolling" else ""
        lines += [
            "",
            f"### {title}",
            "",
            "| family | environment | world | k | arm | fits, failed, unread | channel: swing, "
            f"leaves | regret, first to last | largest regret move{crps} |",
            "|---|---|---|---|---|---|---|---|---|" + ("---|" if crps else ""),
        ]
        for r in rows:
            read = r["reading"]
            channels = "; ".join(
                f"{name}: {_number(c['swing'])}, {c['leaves']} of {c['pairs']}"
                for name, c in read["channels"].items()
            )
            regret = f"{read['regret'][0]:+.4f} to {read['regret'][-1]:+.4f}"
            tail = f" | {_number(read['mean crps'])}" if kind == "rolling" else ""
            lines.append(
                f"| {r['family']} | {r['environment']} | {r['seed']} | {r['k']} | {r['arm']} "
                f"| {read['fits']}, {read['failed']}, {read['unread']} | {channels} | {regret} "
                f"| {read['largest move']:.4f}{tail} |"
            )
    return "\n".join(lines) + "\n"

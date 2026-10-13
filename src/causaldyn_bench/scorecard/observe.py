"""What an arm reads of a world, as one export a fit outside the bench runs on, and the digest that
ties the arm's plan to it.

An :class:`Observation` is all an arm may read of one world at one rung of the ladder: the history
(sales, spend and the named controls the measurement layer reads), the quarter it plans (the
budget, each cell's box, the status quo, the controls' values over the quarter), the window a
return is read over, and the lift tests, every one whichever way its sales moved, reduced as
:func:`causaldyn_bench.budget_regret.lift_rows` reduces them, beside the weeks each test is dark
and the weeks of cooldown its readout counts after them, which a model of the carryover reads a
test's change in spend over. A tool whose likelihood cannot take a test drops it itself. A geo
family's sales and spend carry a geo axis, with each geo's share of the population beside them,
and the channels it buys nationally named. Its cells are each geo's own channels, geo by geo in
the channels' order, then each national channel once, its spend summed over the geos it is
allotted to; its tests name the geos each took dark.

The ladder's tests run in universes of their own, so the history holds none of them. Where it does
hold them, at an overlapping rung (:func:`causaldyn_bench.scorecard.ladder.overlap`), the
observation says so (:class:`Held`): the share of the market that ran them, and each test's
channel's spend as planned over its dark weeks, which the history no longer shows.

**Version 2**, what :func:`export` writes by default, names the family, the environment and the
rung, and its digest reads every array the export holds, so nothing an arm reads lies outside it.
**Version 3** is version 2 with each test's gap in each week of its readout (``lift_gaps``), for a
tool that reads a test's carryover from its shape rather than its sum; its digest reads the gaps
too. An observation is exported at its ``version``, so a run asks for the gaps by observing at
:data:`SERIES`; every number of version 2 is the same in version 3, and a version-3 observation
gives version 2's digest when a record fitted to version 2 asks for it. **Version 1**
is what Track M v2's budgets run exported; :func:`read` reads it as an observation of family 0 at
the top rung, its environment the directory it was written to, and :meth:`Observation.digest`
computes its digest by that run's rule, so a record fitted to a version-1 export is still scored
on its world.

A world is exported once: a second export of it must hold the same digest, or it is refused, and
the arms run on the archived file's bits. numpy's float64 kernels differ in their last bits from
one machine class to another, so a world re-simulated elsewhere need not hash the same, and an
archive keeps every arm on one set of bits.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from causaldyn_bench.budget_regret import LiftRows, falling
from causaldyn_bench.endogenous_mmm import YEAR, Series
from causaldyn_bench.lift_calibration import COOLDOWN, STARTS, TEST

VERSION = 2
FIRST = 1  # the version Track M v2's budgets run exported, read with family 0 at the top rung
SERIES = 3  # version 2 with each test's gap week by week over its readout
LEGACY_CONTROLS = ("promotion", "price")  # the controls version 1 exported, under these names


@dataclass(frozen=True, eq=False)
class Held:
    """The lift tests the history holds: ``share``, the share of the market that ran them, and
    ``planned``, each lift row's channel's spend as planned over the row's dark weeks, ``(rows,
    weeks dark)``. The history's sales and spend are the market's as it ran the tests, so its
    spend over a row's dark weeks is the share's spend as tested beside the rest's as planned."""

    share: float
    planned: Series


@dataclass(frozen=True, eq=False)
class Observation:
    """One world at one rung, as an arm reads it. Weeks are numbered from 1; ``sales`` is
    ``(weeks,)`` and ``spend`` ``(weeks, channels)``, each with a geo axis before the channels where
    ``population`` is given; the box and the status quo hold a weekly spend for each cell.
    ``held`` is given where the history holds the lift tests, and None where they ran apart.
    ``version`` is the export's; version 3 needs the tests' weekly gaps."""

    family: int
    environment: str
    seed: int
    k: int  # each channel's lift tests
    channels: tuple[str, ...]
    sales: Series
    spend: Series
    controls: Mapping[str, Series]  # each over the history's weeks
    future_controls: Mapping[str, Series] | None  # each over the quarter; None in version 1 alone
    kernel_length: int
    planned: int  # the quarter's weeks
    budget: float  # over the quarter
    lower: Series
    upper: Series
    status_quo: Series
    roi_window: tuple[int, int]  # the first and the last week a return is read over
    lift: LiftRows
    population: Series | None = None  # each geo's share, where the world has a geo axis
    held: Held | None = None
    version: int = VERSION
    national: tuple[str, ...] = ()  # the channels bought nationally, where there is a geo axis

    def __post_init__(self) -> None:
        if self.version not in (FIRST, VERSION, SERIES):
            raise ValueError(f"no export has version {self.version}")
        if self.version == SERIES:
            gaps = self.lift.gaps
            rows = (len(self.lift.channel), TEST + COOLDOWN)
            if gaps is None or np.shape(gaps) != rows or not np.isfinite(gaps).all():
                raise ValueError(f"version {SERIES} holds each test's weekly gaps, finite, {rows}")
        if (self.future_controls is None) != (self.version == FIRST):
            raise ValueError(
                "the quarter's controls are given in version 2 and later, and only there"
            )
        weeks = self.sales.shape[0]
        geos = () if self.population is None else (self.population.size,)
        if self.sales.shape != (weeks, *geos):
            raise ValueError(f"sales are {(weeks, *geos)}, not {self.sales.shape}")
        if self.spend.shape != (weeks, *geos, len(self.channels)):
            raise ValueError(
                f"spend is {(weeks, *geos, len(self.channels))}, not {self.spend.shape}"
            )
        if any(np.shape(series) != (weeks,) for series in self.controls.values()):
            raise ValueError(f"every control runs over the history's {weeks} weeks")
        future = self.future_controls
        if future is not None and (
            tuple(future) != tuple(self.controls)
            or any(np.shape(series) != (self.planned,) for series in future.values())
        ):
            raise ValueError(f"the quarter's controls are the history's, over {self.planned} weeks")
        if not self.lower.shape == self.upper.shape == self.status_quo.shape:
            raise ValueError("the box and the status quo hold one weekly spend a cell")
        if not set(self.national) <= set(self.channels):
            raise ValueError(f"the national channels {self.national} are not all {self.channels}")
        if self.national and self.population is None:
            raise ValueError("a channel is bought nationally only where there is a geo axis")
        geos = self.lift.geos
        if (geos is None) != (self.population is None):
            raise ValueError("a geo axis's tests name their geos, and only its tests do")
        if self.population is not None and geos is not None:
            local = len(self.channels) - len(self.national)
            cells = self.population.size * local + len(self.national)
            if self.lower.shape != (cells,):
                raise ValueError(
                    f"{self.population.size} geos of {local} own channels and "
                    f"{len(self.national)} national ones are {cells} cells, not {self.lower.shape}"
                )
            axis = set(range(self.population.size))
            if len(geos) != len(self.lift.channel) or any(
                not row or len(set(row)) < len(row) or not set(row) <= axis for row in geos
            ):
                raise ValueError("each test names distinct geos of the geo axis, one row a test")
        first, last = self.roi_window
        if not 1 <= first <= last <= weeks:
            raise ValueError(f"the window {self.roi_window} lies outside weeks 1 to {weeks}")
        held = self.held
        if held is not None:
            rows = (len(self.lift.channel), TEST)
            if not 0.0 < held.share <= 1.0:
                raise ValueError(f"the tests ran in a share {held.share} of the market, not (0, 1]")
            if np.shape(held.planned) != rows:
                raise ValueError(f"the planned dark weeks are {rows}, not {np.shape(held.planned)}")
            if not (np.isfinite(held.planned).all() and (np.asarray(held.planned) >= 0.0).all()):
                raise ValueError("the planned spend over the dark weeks is finite and at least 0")

    def digest(self, version: int | None = None) -> str:
        """The hash an arm's record must echo: by ``version``'s rule, this observation's own by
        default. Version 1's reads Track M v2's national export alone; version 3's needs the
        tests' weekly gaps, and version 2's is read off a version-3 observation as well."""
        version = self.version if version is None else version
        if version in (VERSION, SERIES):
            return _hash(_fields(self, version))
        if version == FIRST:
            return _first_digest(self)
        raise ValueError(f"no export has version {version}")

    @property
    def market_spend(self) -> Series:
        """``(weeks, channels)``: the spend summed over the geos, where there is a geo axis."""
        return self.spend if self.population is None else self.spend.sum(axis=1)


def _fields(observation: Observation, version: int = VERSION) -> dict[str, np.ndarray]:
    """Every array an export of ``version``, 2 or 3, holds but its digest."""
    o = observation
    if o.future_controls is None:
        raise ValueError("a version-1 observation is read, never exported")
    if version == SERIES and o.lift.gaps is None:
        raise ValueError(
            f"a version-{SERIES} export holds the tests' weekly gaps, and these have none"
        )
    names = tuple(o.controls)
    weeks = o.sales.shape[0]

    def stacked(series: Mapping[str, Series], length: int) -> np.ndarray:
        columns = [np.asarray(series[name], dtype=float) for name in names]
        return np.column_stack(columns) if columns else np.zeros((length, 0))

    fields = {
        "version": np.array(version),
        "family": np.array(o.family),
        "environment": np.array(o.environment),
        "seed": np.array(o.seed),
        "k": np.array(o.k),
        "channels": np.array(o.channels, dtype=str),
        "sales": np.asarray(o.sales, dtype=float),
        "spend": np.asarray(o.spend, dtype=float),
        "control_names": np.array(names, dtype=str),
        "controls": stacked(o.controls, weeks),
        "future_controls": stacked(o.future_controls, o.planned),
        "kernel_length": np.array(o.kernel_length),
        "planned": np.array(o.planned),
        "budget": np.array(float(o.budget)),
        "lower": np.asarray(o.lower, dtype=float),
        "upper": np.asarray(o.upper, dtype=float),
        "status_quo": np.asarray(o.status_quo, dtype=float),
        "roi_window": np.array(o.roi_window, dtype=np.int64),
        "lift_channel": np.array(o.lift.channel, dtype=str),
        "lift_start": np.array(o.lift.start, dtype=np.int64),
        "lift_x": np.asarray(o.lift.x, dtype=float),
        "lift_delta_x": np.asarray(o.lift.delta_x, dtype=float),
        "lift_delta_y": np.asarray(o.lift.delta_y, dtype=float),
        "lift_sigma": np.asarray(o.lift.sigma, dtype=float),
        "lift_dropped": np.array(o.lift.dropped),
        "lift_weeks": np.array(TEST),
        "lift_cooldown": np.array(COOLDOWN),
    }
    if o.population is not None:
        fields["population"] = np.asarray(o.population, dtype=float)
        fields["national"] = np.array(o.national, dtype=str)
        # a geo axis's tests name their geos (Observation's check), one row a test
        taken = np.zeros((len(o.lift.geos or ()), o.population.size), dtype=bool)
        for row, geos in enumerate(o.lift.geos or ()):
            taken[row, list(geos)] = True
        fields["lift_geos"] = taken
    if version == SERIES:
        fields["lift_gaps"] = np.asarray(o.lift.gaps, dtype=float).reshape(-1, TEST + COOLDOWN)
    if o.held is not None:
        fields["held_share"] = np.array(float(o.held.share))
        fields["held_planned"] = np.asarray(o.held.planned, dtype=float)
    return fields


def _hash(fields: Mapping[str, np.ndarray]) -> str:
    """Every field's name, type, shape and bits, in the order of the names."""
    blob = hashlib.sha256()
    for name in sorted(fields):
        value = np.asarray(fields[name])
        blob.update(name.encode())
        if value.dtype.kind == "U":
            blob.update(json.dumps(value.tolist()).encode())
        else:
            value = np.ascontiguousarray(value)
            blob.update(f"{value.dtype.str}{value.shape}".encode())
            blob.update(value.tobytes())
    return blob.hexdigest()[:16]


def _first_digest(observation: Observation) -> str:
    """Track M v2's budgets run's digest (:func:`causaldyn_bench.budget_regret.digest`), read off
    the observation's arrays: over the tests whose sales fell, the rows that run exported
    (:func:`causaldyn_bench.budget_regret.falling`)."""
    o = observation
    if tuple(o.controls) != LEGACY_CONTROLS or o.population is not None:
        raise ValueError("version 1 exported Track M v2's national worlds alone")
    lift = falling(o.lift)
    blob = hashlib.sha256()
    for array in (
        o.sales,
        o.spend,
        o.controls["promotion"],
        o.controls["price"],
        lift.x,
        lift.delta_y,
        lift.sigma,
    ):
        blob.update(np.ascontiguousarray(array, dtype=float).tobytes())
    blob.update("|".join(o.channels + lift.channel).encode())
    return blob.hexdigest()[:16]


def export(observation: Observation, directory: Path) -> Path:
    """``observation`` as ``directory/world_<seed>.npz`` at its version, written once. An archive
    of the world at another version is kept where it holds the same world by its own version's
    rule: the arms run on the archive.

    Raises:
        RuntimeError: the directory already holds an export of the world with other bits.
    """
    fields = _fields(observation, observation.version)
    stamp = _hash(fields)
    path = directory / f"world_{observation.seed}.npz"
    if path.exists():
        saved = read(path)
        archived, ours = saved.digest(), observation.digest(saved.version)
        if archived != ours:
            raise RuntimeError(
                f"{path} holds the world with digest {archived}, not {ours}: a world is exported "
                "once, and its arms run on the archive"
            )
        return path
    directory.mkdir(parents=True, exist_ok=True)
    # written aside and moved into place, so a reader sees the whole file or none
    aside = directory / f".world_{observation.seed}.{os.getpid()}.npz"
    np.savez(aside, allow_pickle=False, digest=np.array(stamp), **fields)
    os.replace(aside, path)
    return path


def read(path: Path) -> Observation:
    """The observation an export holds, of either version.

    Raises:
        RuntimeError: the export's digest is not that of what it holds.
    """
    with np.load(path) as saved:
        if "version" in saved.files:
            observation = _second(saved)
        else:
            observation = _first(saved, path.parent.name)
        stored = str(saved["digest"])
    if observation.digest() != stored:
        raise RuntimeError(f"{path}: its digest {stored} is not that of what it holds")
    return observation


def _lift(saved: Mapping[str, np.ndarray]) -> LiftRows:
    return LiftRows(
        channel=tuple(str(c) for c in saved["lift_channel"]),
        start=tuple(int(s) for s in saved["lift_start"]),
        x=saved["lift_x"],
        delta_x=saved["lift_delta_x"],
        delta_y=saved["lift_delta_y"],
        sigma=saved["lift_sigma"],
        dropped=int(saved["lift_dropped"]),
        geos=(
            tuple(tuple(int(g) for g in np.flatnonzero(row)) for row in saved["lift_geos"])
            if "lift_geos" in saved
            else None
        ),
        gaps=saved.get("lift_gaps"),
    )


def _second(saved: Mapping[str, np.ndarray]) -> Observation:
    version = int(saved["version"])
    if version not in (VERSION, SERIES):
        raise ValueError(f"this bench reads exports of versions {FIRST} to {SERIES} alone")
    names = tuple(str(name) for name in saved["control_names"])
    return Observation(
        family=int(saved["family"]),
        environment=str(saved["environment"]),
        seed=int(saved["seed"]),
        k=int(saved["k"]),
        channels=tuple(str(c) for c in saved["channels"]),
        sales=saved["sales"],
        spend=saved["spend"],
        controls={name: saved["controls"][:, j] for j, name in enumerate(names)},
        future_controls={name: saved["future_controls"][:, j] for j, name in enumerate(names)},
        kernel_length=int(saved["kernel_length"]),
        planned=int(saved["planned"]),
        budget=float(saved["budget"]),
        lower=saved["lower"],
        upper=saved["upper"],
        status_quo=saved["status_quo"],
        roi_window=(int(saved["roi_window"][0]), int(saved["roi_window"][1])),
        lift=_lift(saved),
        population=saved.get("population"),
        held=(
            Held(float(saved["held_share"]), saved["held_planned"])
            if "held_share" in saved
            else None
        ),
        national=tuple(str(c) for c in saved["national"]) if "national" in saved else (),
        version=version,
    )


def _first(saved: Mapping[str, np.ndarray], environment: str) -> Observation:
    """A version-1 export: every channel's four tests, the status quo and the window what the
    budgets run read off the history, last year's mean spend and last year's weeks."""
    spend = saved["spend"]
    weeks = saved["sales"].size
    return Observation(
        family=0,
        environment=environment,
        seed=int(saved["seed"]),
        k=len(STARTS),
        channels=tuple(str(c) for c in saved["channels"]),
        sales=saved["sales"],
        spend=spend,
        controls={name: saved[name] for name in LEGACY_CONTROLS},
        future_controls=None,
        kernel_length=int(saved["kernel_length"]),
        planned=int(saved["planned"]),
        budget=float(saved["budget"]),
        lower=saved["lower"],
        upper=saved["upper"],
        status_quo=spend[-YEAR:].mean(axis=0),
        roi_window=(weeks - YEAR + 1, weeks),
        lift=_lift(saved),
        version=FIRST,
    )

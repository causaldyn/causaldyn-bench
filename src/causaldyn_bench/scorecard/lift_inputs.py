"""A world's lift tests as the files a tool that calibrates on them reads, written beside its
export.

An export (:mod:`.observe`) holds each lift test as
:func:`~causaldyn_bench.budget_regret.lift_rows` reduces it, with its first dark week numbered from
1 (``lift_start``), the dark weeks and the cooldown each readout counts (``lift_weeks``,
``lift_cooldown``), and, on a geo axis, the geos each test took dark (``lift_geos``). That is all a
tool's file of tests needs, and :func:`write` writes it out from the observation an export holds,
leaving the export and its digest as they are:

* ``lift_<seed>.csv``, PyMC-Marketing's lift table, a row for each test that one row holds, national
  or of one geo: ``channel``, ``x``, ``delta_x``, ``delta_y`` and ``sigma`` as the export holds
  them, ``date``, the first dark week's Monday, and ``geo`` where the world has a geo axis. Told the
  dark weeks and the cooldown, a reader reads a row's lift as ``delta_y`` times the dark weeks, the
  spend ``x + delta_x`` over them and ``x`` over the cooldown;
* ``experiments_<seed>.csv``, each test of several geos, which no row of that table holds, as an
  experiment's record: ``channel``; ``estimate`` and ``standard_error``, the lift over the readout
  and its error, ``delta_y`` and ``sigma`` times the dark weeks; ``start`` and ``end``, the first
  and the last dark week's Mondays; ``spend`` and ``planned``, the tested geos' weekly spend over
  the dark weeks, ``x + delta_x``, and as planned, ``x``; ``cooldown``; ``share``, the tested geos'
  share of the population, the share of the market the record is of; and ``geos``, their names;
* ``lift_<seed>.json`` beside them: the dark weeks and the cooldown, the calendar, which file holds
  how many tests, the tests the export left out, and the export's version and digest, which tie the
  files to the bits the arms ran on.

Weeks are dated Mondays from 2023-01-02, from the first week the export holds, the calendar every
arm dates them on; a geo is named ``geo_<g>``, its place on the geo axis in two digits, as the arms
name it. Numbers are written as Python's shortest repr of each float, so they read back to the bit.
A table with no test to hold is not written.

``python -m causaldyn_bench.scorecard.lift_inputs DIRECTORY`` writes them for every export in
``DIRECTORY``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from causaldyn_bench.lift_calibration import COOLDOWN, TEST
from causaldyn_bench.scorecard.observe import Observation, read

FIRST_MONDAY = np.datetime64("2023-01-02", "D")  # week 1's Monday on the arms' calendar
TABLE = ("channel", "x", "delta_x", "delta_y", "sigma", "date")
RECORD = ("channel", "estimate", "standard_error", "start", "end", "spend", "planned", "cooldown")
GEO_RECORD = ("share", "geos")
VERSION = 1  # of the files' layout, which the sidecar states


def monday(week: int) -> str:
    """The Monday of ``week``, numbered from 1, as ``YYYY-MM-DD``."""
    return str(FIRST_MONDAY + np.timedelta64(7 * (week - 1), "D"))


def geo_name(g: int) -> str:
    """The geo at place ``g`` of the geo axis, as the arms name it."""
    return f"geo_{g:02d}"


def tables(observation: Observation) -> tuple[dict[str, list[object]], dict[str, list[object]]]:
    """The lift table's columns and the experiments', each a list a column, every test in one."""
    lift = observation.lift
    population = observation.population
    table: dict[str, list[object]] = {name: [] for name in TABLE}
    if population is not None:
        table["geo"] = []
    records: dict[str, list[object]] = {name: [] for name in (*RECORD, *GEO_RECORD)}
    for row, channel in enumerate(lift.channel):
        x, change, lift_y, sigma = (
            float(np.asarray(column)[row])
            for column in (lift.x, lift.delta_x, lift.delta_y, lift.sigma)
        )
        start = lift.start[row]
        geos = () if lift.geos is None else lift.geos[row]
        if len(geos) <= 1:
            for name, value in zip(
                TABLE, (channel, x, change, lift_y, sigma, monday(start)), strict=True
            ):
                table[name].append(value)
            if population is not None:
                table["geo"].append(geo_name(geos[0]))
            continue
        values = (
            channel,
            lift_y * TEST,
            sigma * TEST,
            monday(start),
            monday(start + TEST - 1),
            x + change,
            x,
            COOLDOWN,
            float(np.sum(np.asarray(population)[list(geos)])),
            ";".join(geo_name(g) for g in geos),
        )
        for name, value in zip((*RECORD, *GEO_RECORD), values, strict=True):
            records[name].append(value)
    return table, records


def _csv(columns: dict[str, list[object]], path: Path) -> None:
    with path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(columns)
        for row in zip(*columns.values(), strict=True):
            writer.writerow([repr(value) if isinstance(value, float) else value for value in row])


def write(observation: Observation, directory: Path) -> Path:
    """``observation``'s lift tests as the files the module names, in ``directory``; the sidecar's
    path."""
    directory.mkdir(parents=True, exist_ok=True)
    seed = observation.seed
    names = {"lift_table": f"lift_{seed}.csv", "experiments": f"experiments_{seed}.csv"}
    written: dict[str, str | None] = {}
    rows: dict[str, int] = {}
    for key, columns in zip(names, tables(observation), strict=True):
        rows[key] = len(columns["channel"])
        written[key] = names[key] if rows[key] else None
        if rows[key]:
            _csv(columns, directory / names[key])
    sidecar = {
        "layout": VERSION,
        "family": observation.family,
        "environment": observation.environment,
        "seed": seed,
        "k": observation.k,
        "export_version": observation.version,
        "export_digest": observation.digest(),
        "weeks": TEST,
        "cooldown": COOLDOWN,
        "first_monday": str(FIRST_MONDAY),
        "files": written,
        "rows": rows,
        "dropped": observation.lift.dropped,
    }
    path = directory / f"lift_{seed}.json"
    path.write_text(json.dumps(sidecar, indent=2) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    """The files for every export, ``world_<seed>.npz``, in the directory ``argv`` names."""
    (directory,) = sys.argv[1:] if argv is None else argv
    for path in sorted(Path(directory).glob("world_*.npz")):
        write(read(path), path.parent)


if __name__ == "__main__":
    main()

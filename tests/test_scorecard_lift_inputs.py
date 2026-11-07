"""A world's lift tests written beside its export as one file a tool reads: PyMC-Marketing's lift
table with each test's date, the experiments' records of the tests of several geos, and the
sidecar with the dark weeks, the cooldown and the export's digest. Every number reads back to the
bit, every date to the week the export numbers, and the export is left as it was."""

import csv
import dataclasses
import json

import numpy as np
import pytest

from causaldyn_bench.lift_calibration import COOLDOWN, TEST
from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.lift_inputs import FIRST_MONDAY, geo_name, main, monday, write
from causaldyn_bench.scorecard.observe import SERIES, export, read
from causaldyn_bench.scorecard.short import SHORT
from causaldyn_bench.scorecard.track_m2 import TRACK_M2


def _rows(path):
    with path.open(newline="") as table:
        return list(csv.DictReader(table))


def _week(date: str) -> int:
    """The week a Monday dates, numbered from 1."""
    days = (np.datetime64(date, "D") - FIRST_MONDAY).astype(int)
    assert days % 7 == 0
    return int(days) // 7 + 1


def test_mondays_count_weeks_from_the_arms_first():
    assert monday(1) == "2023-01-02" and monday(2) == "2023-01-09" and monday(53) == "2024-01-01"
    assert [_week(monday(w)) for w in (1, 20, 156)] == [1, 20, 156]
    assert geo_name(3) == "geo_03" and geo_name(49) == "geo_49"


def test_a_national_worlds_tests_are_one_lift_table_with_their_dates(tmp_path):
    world = TRACK_M2.world("drawn", TRACK_M2.pilots["drawn"][0])
    observation = TRACK_M2.observe(world, 4)
    archive = export(observation, tmp_path)
    before = archive.read_bytes()
    sidecar = json.loads(write(read(archive), tmp_path).read_text())
    assert archive.read_bytes() == before
    seed = observation.seed
    assert sidecar["files"] == {"lift_table": f"lift_{seed}.csv", "experiments": None}
    assert sidecar["rows"] == {"lift_table": 12, "experiments": 0}
    assert (sidecar["weeks"], sidecar["cooldown"]) == (TEST, COOLDOWN) == (4, 8)
    assert sidecar["first_monday"] == "2023-01-02"
    assert sidecar["export_digest"] == observation.digest() == str(np.load(archive)["digest"])
    assert (sidecar["family"], sidecar["environment"], sidecar["seed"], sidecar["k"]) == (
        0,
        "drawn",
        seed,
        4,
    )
    with np.load(archive) as saved:
        assert (int(saved["lift_weeks"]), int(saved["lift_cooldown"])) == (TEST, COOLDOWN)
    rows = _rows(tmp_path / f"lift_{seed}.csv")
    lift = observation.lift
    assert list(rows[0]) == ["channel", "x", "delta_x", "delta_y", "sigma", "date"]
    assert tuple(row["channel"] for row in rows) == lift.channel
    for name in ("x", "delta_x", "delta_y", "sigma"):
        read_back = np.array([float(row[name]) for row in rows])
        np.testing.assert_array_equal(read_back, getattr(lift, name))
    assert tuple(_week(row["date"]) for row in rows) == lift.start
    assert not (tmp_path / f"experiments_{seed}.csv").exists()


def test_a_rows_lift_over_its_dark_weeks_is_its_readouts_summed_gap(tmp_path):
    """A reader told the dark weeks reads a row's lift as ``delta_y`` times them: the gap the test's
    readout sums, over its dark weeks and the cooldown, to rounding."""
    world = TRACK_M2.world("drawn", TRACK_M2.pilots["drawn"][1])
    observation = TRACK_M2.observe(world, 2)
    write(observation, tmp_path)
    rows = _rows(tmp_path / f"lift_{observation.seed}.csv")
    gaps = np.asarray(observation.lift.gaps)
    assert gaps.shape == (len(rows), TEST + COOLDOWN)
    for row, gap in zip(rows, gaps, strict=True):
        assert float(row["delta_y"]) * TEST == pytest.approx(gap.sum(), rel=1e-12, abs=1e-9)
        assert float(row["x"]) + float(row["delta_x"]) == 0.0


def test_a_short_series_dates_its_tests_from_the_first_week_it_holds(tmp_path):
    name = "L-52"
    world = SHORT.world(name, SHORT.pilots[name][0])
    observation = SHORT.observe(world, 1)
    write(observation, tmp_path)
    rows = _rows(tmp_path / f"lift_{observation.seed}.csv")
    weeks = observation.sales.shape[0]
    assert rows and all(1 <= _week(row["date"]) <= weeks for row in rows)
    assert tuple(_week(row["date"]) for row in rows) == observation.lift.start


def test_a_geo_worlds_tests_of_several_geos_are_experiments_records(tmp_path):
    """Three geos: a channel's tests alternate between geos 0 and 2 together and geo 1 alone. The
    tests of one geo are lift-table rows naming it; the others are records of the tested geos'
    share of the market, their lift and error the row's times the dark weeks, their window the dark
    weeks and their spend and planned spend the row's."""
    name = "geos-3"
    observation = FEW_GEOS.observe(FEW_GEOS.world(name, FEW_GEOS.pilots[name][0]), 2)
    sidecar = json.loads(write(observation, tmp_path).read_text())
    lift = observation.lift
    assert lift.geos is not None and observation.population is not None
    one = [r for r, geos in enumerate(lift.geos) if len(geos) == 1]
    several = [r for r, geos in enumerate(lift.geos) if len(geos) > 1]
    assert one and several
    assert sidecar["rows"] == {"lift_table": len(one), "experiments": len(several)}
    table = _rows(tmp_path / sidecar["files"]["lift_table"])
    assert [row["geo"] for row in table] == [geo_name(lift.geos[r][0]) for r in one]
    np.testing.assert_array_equal([float(row["x"]) for row in table], np.asarray(lift.x)[one])
    records = _rows(tmp_path / sidecar["files"]["experiments"])
    assert list(records[0]) == [
        "channel",
        "estimate",
        "standard_error",
        "start",
        "end",
        "spend",
        "planned",
        "cooldown",
        "share",
        "geos",
    ]
    for record, r in zip(records, several, strict=True):
        assert record["channel"] == lift.channel[r]
        assert float(record["estimate"]) / TEST == float(np.asarray(lift.delta_y)[r])
        assert float(record["standard_error"]) / TEST == float(np.asarray(lift.sigma)[r])
        assert _week(record["start"]) == lift.start[r]
        assert _week(record["end"]) == lift.start[r] + TEST - 1
        assert float(record["planned"]) == float(np.asarray(lift.x)[r])
        assert float(record["spend"]) == 0.0
        assert int(record["cooldown"]) == COOLDOWN
        share = float(np.sum(np.asarray(observation.population)[list(lift.geos[r])]))
        assert float(record["share"]) == share and 0.0 < share < 1.0
        assert record["geos"].split(";") == [geo_name(g) for g in lift.geos[r]]


def test_every_export_in_a_directory_gains_its_files(tmp_path):
    for seed in TRACK_M2.pilots["drawn"][:2]:
        export(TRACK_M2.observe(TRACK_M2.world("drawn", seed), 1), tmp_path)
    empty = TRACK_M2.pilots["drawn"][2]
    export(TRACK_M2.observe(TRACK_M2.world("drawn", empty), 0), tmp_path)
    main([str(tmp_path)])
    for seed in TRACK_M2.pilots["drawn"][:2]:
        sidecar = json.loads((tmp_path / f"lift_{seed}.json").read_text())
        assert sidecar["rows"]["lift_table"] == 3 and (tmp_path / f"lift_{seed}.csv").exists()
    sidecar = json.loads((tmp_path / f"lift_{empty}.json").read_text())
    assert sidecar["files"] == {"lift_table": None, "experiments": None}
    assert not (tmp_path / f"lift_{empty}.csv").exists()


def test_a_version_3_export_writes_the_same_files_as_version_2(tmp_path):
    world = TRACK_M2.world("drawn", TRACK_M2.pilots["drawn"][0])
    observation = TRACK_M2.observe(world, 2)
    series = dataclasses.replace(observation, version=SERIES)
    write(observation, tmp_path / "2")
    write(series, tmp_path / "3")
    name = f"lift_{observation.seed}.csv"
    assert (tmp_path / "2" / name).read_bytes() == (tmp_path / "3" / name).read_bytes()
    assert (
        json.loads((tmp_path / "3" / f"lift_{observation.seed}.json").read_text())["export_version"]
        == SERIES
    )

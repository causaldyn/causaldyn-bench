"""The PyMC-Marketing arm's own rule for the lift rows its likelihood takes, read from
``scripts/pymc_marketing_lift.py`` without PyMC-Marketing: an export carries every test, and the arm
drops the tests whose sales did not fall, and counts them; on a geo export it apportions each test
across its tested geos at one ROI, rows that sum back to the test."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.observe import export
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pymc_marketing_lift.py"
_SPEC = importlib.util.spec_from_file_location("pymc_marketing_lift", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
lift = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lift)


def test_the_arm_takes_a_row_whose_sales_moved_with_its_spend_and_counts_the_rest():
    world = {
        "lift_delta_x": np.array([-2.0, -2.0, -2.0, 3.0, 3.0, -1.0]),
        "lift_delta_y": np.array([-5.0, 0.0, 4.0, 1.0, -1.0, -1e-300]),
    }
    keep, dropped = lift.taken(world)
    np.testing.assert_array_equal(keep, [True, False, False, True, False, True])
    assert dropped == 3


def test_on_a_geo_export_whose_tests_rose_the_arm_drops_them_itself(tmp_path):
    seed = GEO.pilots["policy-0"][3]
    path = export(GEO.observe(GEO.world("policy-0", seed), 4), tmp_path)
    with np.load(path) as world:
        assert world["lift_channel"].size == 12 and int(world["lift_dropped"]) == 0
        keep, dropped = lift.taken(world)
        np.testing.assert_array_equal(keep, world["lift_delta_y"] < 0.0)
    assert (int(keep.sum()), dropped) == (4, 8)


def test_on_a_national_export_the_rows_are_the_ladders_and_the_arm_drops_the_one_that_rose(
    tmp_path,
):
    # drawn world 900's top rung holds one test whose sales rose, as the budgets run's export did
    path = export(TRACK_M2.observe(TRACK_M2.world("drawn", 900), 4), tmp_path)
    with np.load(path) as world:
        assert world["lift_channel"].size == 12 and int(world["lift_dropped"]) == 0
        keep, dropped = lift.taken(world)
        np.testing.assert_array_equal(keep, world["lift_delta_y"] < 0.0)
    assert dropped == 1


def _two_tests():
    """Six geos and two channels over six weeks of spend: a test of geos 0 to 4 dark on ``b`` over
    weeks 2 and 3, whose sales fell, and a test of geo 5 alone on ``a`` over weeks 4 and 5, whose
    sales rose."""
    spend = np.random.default_rng(7).uniform(1.0, 9.0, (6, 6, 2))
    x = [spend[1:3, :5, 1].sum(axis=1).mean(), spend[3:5, 5, 0].mean()]
    return {
        "channels": np.array(["a", "b"]),
        "spend": spend,
        "lift_weeks": np.array(2),
        "lift_channel": np.array(["b", "a"]),
        "lift_start": np.array([2, 4]),
        "lift_geos": np.array([[1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1]], dtype=bool),
        "lift_x": np.array(x),
        "lift_delta_x": -np.array(x),
        "lift_delta_y": np.array([-10.0, 3.0]),
        "lift_sigma": np.array([4.0, 1.5]),
    }


def _sums_back(world, rows, test):
    """A test's rows sum back to it at its ROI and hold its information about a common return."""
    at = rows["test"] == test
    dx, dy, sigma = (float(world[f"lift_{key}"][test]) for key in ("delta_x", "delta_y", "sigma"))
    assert rows["delta_x"][at].sum() == pytest.approx(dx, rel=1e-14, abs=0.0)
    assert rows["delta_y"][at].sum() == pytest.approx(dy, rel=1e-14, abs=0.0)
    np.testing.assert_allclose(rows["delta_y"][at] / rows["delta_x"][at], dy / dx, rtol=1e-14)
    information = np.sum(rows["delta_x"][at] ** 2 / rows["sigma"][at] ** 2)
    assert information == pytest.approx(dx**2 / sigma**2, rel=1e-13, abs=0.0)
    estimate = np.sum(rows["delta_x"][at] * rows["delta_y"][at] / rows["sigma"][at] ** 2)
    assert estimate / information == pytest.approx(dy / dx, rel=1e-13, abs=0.0)


def test_a_test_of_five_geos_splits_into_five_rows_that_sum_back_to_it():
    """Each tested geo's row withholds its own mean spend over the dark weeks and reads its share of
    the change in sales, at an error ``s_g * sigma * sqrt(m)``; a test of one geo is its own row,
    and a test whose sales rose drops its every row."""
    world = _two_tests()
    rows, keep = lift.apportioned(world)
    np.testing.assert_array_equal(rows["test"], [0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(rows["geo"], [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(rows["channel"], ["b"] * 5 + ["a"])
    np.testing.assert_array_equal(keep, [True] * 5 + [False])
    np.testing.assert_allclose(rows["x"][:5], world["spend"][1:3, :5, 1].mean(axis=0), rtol=1e-15)
    np.testing.assert_array_equal(rows["delta_x"], -rows["x"])
    share = rows["x"][:5] / rows["x"][:5].sum()
    np.testing.assert_allclose(rows["sigma"][:5], share * 4.0 * np.sqrt(5.0), rtol=1e-14)
    _sums_back(world, rows, 0)
    assert rows["x"][5] == pytest.approx(world["lift_x"][1], rel=1e-15, abs=0.0)
    assert (rows["delta_y"][5], rows["sigma"][5]) == (3.0, 1.5)


def test_on_the_familys_geo_export_each_test_splits_into_its_five_geos(tmp_path):
    seed = GEO.pilots["policy-0"][3]
    path = export(GEO.observe(GEO.world("policy-0", seed), 4), tmp_path)
    with np.load(path) as saved:
        world = {key: saved[key] for key in saved.files}
    rows, keep = lift.apportioned(world)
    assert rows["test"].size == 60
    for test in range(12):
        at = rows["test"] == test
        np.testing.assert_array_equal(rows["geo"][at], np.flatnonzero(world["lift_geos"][test]))
        _sums_back(world, rows, test)
    np.testing.assert_array_equal(keep, np.repeat(world["lift_delta_y"] < 0.0, 5))
    assert int(keep.sum()) == 20

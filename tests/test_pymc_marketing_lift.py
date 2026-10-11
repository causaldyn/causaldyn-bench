"""The PyMC-Marketing arm's own rule for the lift rows its likelihood takes, read from
``scripts/pymc_marketing_lift.py`` without PyMC-Marketing: an export carries every test, and the arm
drops the tests whose sales did not fall, and counts them."""

import importlib.util
from pathlib import Path

import numpy as np

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


def test_on_a_geo_export_the_arm_takes_a_test_of_one_geo_whose_sales_fell():
    """A geo model reads a lift row at one geo, so a test of several geos is dropped whatever its
    sales did, and counted as that; a test of one geo is taken where its sales fell."""
    world = {
        "lift_geos": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
        "lift_delta_x": np.array([-1.0, -1.0, -1.0, -1.0, -2.0]),
        "lift_delta_y": np.array([-1.0, -1.0, 0.5, 0.5, 0.0]),
    }
    keep, dropped, why = lift.taken_geo(world)
    np.testing.assert_array_equal(keep, [False, True, False, False, False])
    assert dropped == 4
    assert why == {lift.SEVERAL: 2, lift.AGAINST: 2}


def test_on_the_familys_geo_export_every_test_takes_several_geos_and_is_dropped(tmp_path):
    seed = GEO.pilots["policy-0"][3]
    path = export(GEO.observe(GEO.world("policy-0", seed), 4), tmp_path)
    with np.load(path) as world:
        keep, dropped, why = lift.taken_geo(world)
    assert not keep.any() and dropped == 12
    assert why == {lift.SEVERAL: 12, lift.AGAINST: 0}

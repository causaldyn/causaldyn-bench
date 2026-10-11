"""What the tools' arms read of a geo export, from ``scripts/geo_export.py`` without any tool: a
channel's quarter is its cells', and the harness allots it back to them; a cell's spend goes to the
geos as the world allots it, and back; the market is the geos summed; a test withheld its own
geos' spend; and a test read as the market's keeps its ROI, while a national export's tests read as
they are."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from causaldyn_bench.mmm_decision import PLANNED
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.observe import export
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from causaldyn_bench.scorecard.truth import allotted, geo_quarter

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "geo_export.py"
_SPEC = importlib.util.spec_from_file_location("geo_export", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
reading = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reading)


@pytest.fixture(scope="module")
def geo_world():
    return GEO.world("policy-0.5", GEO.pilots["policy-0.5"][2])


@pytest.fixture(scope="module")
def saved(geo_world, tmp_path_factory):
    path = export(GEO.observe(geo_world, 4), tmp_path_factory.mktemp("geo"))
    with np.load(path) as world:
        return {key: world[key] for key in world.files}


def test_a_geo_export_is_told_from_a_national_one(saved, tmp_path):
    path = export(TRACK_M2.observe(TRACK_M2.world("drawn", 905), 4), tmp_path)
    with np.load(path) as national:
        assert reading.geo(saved) and not reading.geo(national)


def test_a_cells_channel_is_the_quarters(geo_world, saved):
    quarter = geo_quarter(geo_world.history)
    np.testing.assert_array_equal(reading.channel_of(saved), quarter.channel)


def test_a_channels_quarter_is_its_cells_and_the_harness_allots_it_back(geo_world, saved):
    """Each cell's box is the same multiple of its own status quo, so a channel's box summed over
    its cells allots back to the cells' boxes, its status quo to theirs, and the budget is the
    status quo's over the quarter."""
    quarter = geo_quarter(geo_world.history)
    channel = reading.channel_quarter(saved)
    for key in ("lower", "upper", "status_quo"):
        assert channel[key].shape == (3,)
        np.testing.assert_allclose(
            quarter.allot(channel[key]), getattr(quarter, key), rtol=1e-13, atol=0.0
        )
    assert PLANNED * channel["status_quo"].sum() == pytest.approx(
        quarter.budget, rel=1e-14, abs=0.0
    )
    np.testing.assert_allclose(channel["lower"], 0.5 * channel["status_quo"], rtol=1e-14)
    np.testing.assert_allclose(channel["upper"], 2.0 * channel["status_quo"], rtol=1e-14)


def test_a_national_exports_quarter_is_its_own(tmp_path):
    path = export(TRACK_M2.observe(TRACK_M2.world("drawn", 905), 0), tmp_path)
    with np.load(path) as national:
        channel = reading.channel_quarter(national)
        for key in ("lower", "upper", "status_quo"):
            np.testing.assert_array_equal(channel[key], national[key])
        sales, spend = reading.market(national)
        np.testing.assert_array_equal(sales, national["sales"])
        np.testing.assert_array_equal(spend, national["spend"])


def test_a_cells_spend_goes_to_the_geos_as_the_world_allots_it_and_back(geo_world, saved):
    draw = np.random.default_rng(3).uniform(0.5, 2.0, 21) * saved["status_quo"]
    spread = reading.panel(saved, draw)
    np.testing.assert_allclose(spread, allotted(geo_world.history, draw), rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(reading.cells(saved, spread), draw, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(spread.sum(axis=0), np.bincount(reading.channel_of(saved), draw))
    with pytest.raises(ValueError, match="cells are not"):
        reading.panel(saved, draw[:20])
    with pytest.raises(ValueError, match="for each geo and channel"):
        reading.cells(saved, spread.T)


def test_the_market_is_the_geos_summed(geo_world, saved):
    sales, spend = reading.market(saved)
    np.testing.assert_allclose(sales, geo_world.history.kpi.sum(axis=0), rtol=1e-14)
    np.testing.assert_allclose(spend, geo_world.history.spend.sum(axis=0), rtol=1e-14)


def test_a_test_withheld_its_own_geos_spend_over_its_dark_weeks(geo_world, saved):
    """The spend a geo test withheld is its channel's in the geos it took dark, each dark week;
    its mean is the row's ``x``, as the ladder reads it."""
    names = [str(c) for c in saved["channels"]]
    history = geo_world.history
    assert len(saved["lift_channel"]) == 12
    for row, name in enumerate(saved["lift_channel"]):
        first = int(saved["lift_start"][row]) - 1
        geos = np.flatnonzero(saved["lift_geos"][row])
        assert 0 < geos.size < 10
        dark = history.spend[geos, first : first + 4, names.index(str(name))].sum(axis=0)
        np.testing.assert_allclose(reading.withheld(saved, row), dark, rtol=1e-14, atol=0.0)
        assert reading.withheld(saved, row).mean() == pytest.approx(
            float(saved["lift_x"][row]), rel=1e-14, abs=0.0
        )


def test_a_geo_test_read_as_the_markets_keeps_its_roi(geo_world, saved):
    """Scaled from the tested geos' spend to the market's over the dark weeks, a test's lift keeps
    its return per euro; the market spent more than the test withheld, all of TV's half again or
    more, since alternate geos hold some half of the population."""
    names = [str(c) for c in saved["channels"]]
    read = reading.market_lift(saved)
    history = geo_world.history
    for row, name in enumerate(saved["lift_channel"]):
        first = int(saved["lift_start"][row]) - 1
        spent = history.spend[:, first : first + 4, names.index(str(name))].sum()
        assert read["spend"][row] == pytest.approx(spent, rel=1e-14, abs=0.0)
        tested = 4 * float(saved["lift_x"][row])
        assert read["lift"][row] / read["spend"][row] == pytest.approx(
            -float(saved["lift_delta_y"][row]) * 4 / tested, rel=1e-12, abs=0.0
        )
        assert read["spend"][row] > 1.2 * tested


def test_a_national_test_reads_as_it_is(tmp_path):
    path = export(TRACK_M2.observe(TRACK_M2.world("drawn", 905), 4), tmp_path)
    with np.load(path) as national:
        read = reading.market_lift(national)
        np.testing.assert_array_equal(read["lift"], -national["lift_delta_y"] * 4)
        np.testing.assert_allclose(read["spend"], 4 * national["lift_x"], rtol=1e-14, atol=0.0)
        for row in range(len(national["lift_channel"])):
            assert reading.withheld(national, row).mean() == pytest.approx(
                float(national["lift_x"][row]), rel=1e-14, abs=0.0
            )


def test_a_geo_export_reads_as_a_national_one_to_a_national_tool(saved):
    """The national reading is the market's: summed over the geos, a channel at a time, without
    the geo axis, every other array as the export holds it, each test's withheld spend and its
    reading as the market's beside them."""
    view = reading.national(saved)
    assert not reading.geo(view)
    assert not set(reading.GEO_KEYS) & set(view)
    sales, spend = reading.market(saved)
    np.testing.assert_array_equal(view["sales"], sales)
    np.testing.assert_array_equal(view["spend"], spend)
    for key, value in reading.channel_quarter(saved).items():
        np.testing.assert_array_equal(view[key], value)
    for key in ("digest", "budget", "lift_delta_y", "lift_sigma", "lift_x", "kernel_length"):
        np.testing.assert_array_equal(view[key], saved[key])
    assert view["lift_withheld"].shape == (12, 4)
    for row in range(12):
        np.testing.assert_array_equal(view["lift_withheld"][row], reading.withheld(saved, row))
    read = reading.market_lift(saved)
    np.testing.assert_array_equal(view["lift_market_lift"], read["lift"])
    np.testing.assert_array_equal(view["lift_market_spend"], read["spend"])


def test_a_national_export_reads_as_itself_at_rung_nought(tmp_path):
    path = export(TRACK_M2.observe(TRACK_M2.world("drawn", 905), 0), tmp_path)
    with np.load(path) as national:
        view = reading.national(national)
        assert set(view) - set(national.files) == {
            "lift_withheld",
            "lift_market_lift",
            "lift_market_spend",
        }
        for key in national.files:
            np.testing.assert_array_equal(view[key], national[key])
        assert view["lift_withheld"].shape == (0, 4)

"""A refresh's swing and a rolling origin: the cuts plan the world's one quarter and read one
window, a world cut at its whole history is the world, and the readings count what moved and only
that."""

import dataclasses

import jax
import numpy as np
import pytest

from causaldyn_bench.lift_calibration import COOLDOWN, TEST
from causaldyn_bench.mmm_decision import Quarter
from causaldyn_bench.scorecard import refresh
from causaldyn_bench.scorecard.drift import DRIFT
from causaldyn_bench.scorecard.mapping import channels
from causaldyn_bench.scorecard.returns import true_returns
from causaldyn_bench.scorecard.short import SHORT
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from causaldyn_bench.scorecard.truth import worth

SEED = 900
DRAWS = 40


@pytest.fixture(scope="module")
def world():
    return TRACK_M2.world("drawn", SEED)


@pytest.fixture(scope="module")
def short():
    environment = "L-52"
    return SHORT.world(environment, SHORT.pilots[environment][0])


def _response(observation, world, coefficient=1.0, seed=0):
    """The world's own channels as an arm's mapped response, ``DRAWS`` draws each a little apart
    and every coefficient times ``coefficient``, with the decomposition its channels give on the
    observation's spend, so that the mapping is exact."""
    history = world.history
    rng = np.random.default_rng(seed)
    parameters = {}
    for c, name in enumerate(observation.channels):
        jitter = 1.0 + 0.02 * rng.standard_normal((3, DRAWS))
        parameters[name] = {
            "retention": list(np.clip(history.retention[c] * jitter[0], 0.01, 0.99)),
            "scale": list(2.0 / history.saturation[c] * jitter[1]),
            "coefficient": list(coefficient * history.effect[c] * jitter[2]),
        }
    response = {
        "kernel": "GeometricAdstock",
        "length": history.kernel_length,
        "normalized": True,
        "curve": "Tanh",
        "parameters": parameters,
        "decomposition": {},
    }
    with jax.enable_x64(True):
        built = channels(response, observation.channels)
        weekly = {
            name: [np.asarray(draw[c](observation.spend[:, c])).tolist() for draw in built[:1]]
            for c, name in enumerate(observation.channels)
        }
        total = {
            name: [float(np.sum(draw[c](observation.spend[:, c]))) for draw in built]
            for c, name in enumerate(observation.channels)
        }
    response["decomposition"] = {"total": total, "held": [0], "weekly": weekly}
    return response


def _record(observation, weekly, response=None, forecast=None, error=None):
    record = {
        "digest": observation.digest(),
        "version": observation.version,
        "weekly": list(map(float, weekly)),
        "error": error,
    }
    if response is not None:
        record["response"] = response
    if forecast is not None:
        record["forecast"] = forecast
    return record


def test_the_first_fit_reads_a_year_and_each_refit_four_weeks_more():
    assert refresh.lengths(156) == (132, 136, 140, 144, 148, 152, 156)
    assert refresh.lengths(80, step=2, refits=3) == (74, 76, 78, 80)
    for weeks, step, refits in ((75, 4, 6), (156, 0, 6), (156, 4, 0)):
        with pytest.raises(ValueError, match=r"year|at least"):
            refresh.lengths(weeks, step, refits)


def test_each_cut_plans_the_world_s_quarter_on_the_weeks_after_it(world):
    """Every cut plans the export's quarter and reads its returns on the year ending at the first
    cut's last week; its controls over the quarter are the history's weeks after it, then the
    quarter's; it keeps a test only where the test's readout ends inside it; the last cut is the
    export but for that window."""
    full = TRACK_M2.observe(world, 2)
    cuts = refresh.windows(full)
    assert [c.sales.shape[0] for c in cuts] == [132, 136, 140, 144, 148, 152, 156]
    for c in cuts:
        weeks = c.sales.shape[0]
        assert c.roi_window == (81, 132)
        for name in ("budget", "lower", "upper", "status_quo"):
            np.testing.assert_array_equal(getattr(c, name), getattr(full, name))
        np.testing.assert_array_equal(c.spend, full.spend[:weeks])
        for name, series in full.controls.items():
            np.testing.assert_array_equal(c.controls[name], series[:weeks])
            spliced = np.concatenate([series[weeks:], full.future_controls[name]])[:13]
            np.testing.assert_array_equal(c.future_controls[name], spliced)
        ends = np.asarray(c.lift.start) + TEST + COOLDOWN - 1
        assert np.all(ends <= weeks)
        assert len(c.lift.start) == int(
            np.sum(np.asarray(full.lift.start) + TEST + COOLDOWN - 1 <= weeks)
        )
    assert [len(c.lift.start) for c in cuts] == [3, 3, 3, 3, 3, 6, 6]
    assert dataclasses.replace(cuts[-1], roi_window=full.roi_window).digest() == full.digest()


def test_a_cut_is_refused_where_its_export_cannot_be_cut(world):
    full = TRACK_M2.observe(world, 0)
    with pytest.raises(ValueError, match="quarter's controls"):
        refresh.windows(dataclasses.replace(full, future_controls=None, version=1))
    with pytest.raises(ValueError, match="year"):
        refresh.windows(full, step=18)


def test_a_world_cut_at_its_whole_history_is_the_world(world):
    """The last origin is the world: its quarter, its quarter's sales at the status quo and its
    best plan; an earlier one is the world's first weeks with the quarter after them."""
    cuts = refresh.origins(world)
    assert [c.history.week.size for c in cuts] == [132, 136, 140, 144, 148, 152, 156]
    last, truth = TRACK_M2.truth(cuts[-1]), TRACK_M2.truth(world)
    np.testing.assert_array_equal(cuts[-1].shadow.sales, world.shadow.sales)
    assert last.best.worth == truth.best.worth
    first = cuts[0]
    np.testing.assert_array_equal(first.history.sales, world.history.sales[:132])
    np.testing.assert_array_equal(first.history.spend, world.history.spend[:132])
    after = Quarter.after(first.history)
    np.testing.assert_array_equal(TRACK_M2.truth(first).quarter.status_quo, after.status_quo)
    assert first.shadow.week[0] == 133


def test_a_world_whose_effect_moves_is_not_cut():
    walking = DRIFT.world("walk", DRIFT.pilots["walk"][0])
    with pytest.raises(ValueError, match="holds still"):
        refresh.cut(walking, 132)


def test_a_swing_of_the_world_s_own_channels_holds_still(world):
    """Fitted to the world's own channels on every cut, an arm's returns on the one window are the
    world's there, so they move by the draws' jitter alone and no refit leaves the 80 % before
    it; its plan, the best plan each time, moves the regret by nothing."""
    cuts = refresh.windows(TRACK_M2.observe(world, 0))
    best = TRACK_M2.truth(world).best.weekly
    records = [_record(c, best, _response(c, world)) for c in cuts]
    with jax.enable_x64(True):
        read = refresh.swing(TRACK_M2, world, 0, records)
    truth = true_returns(world.history, (81, 132))
    assert read["fits"] == 7
    assert read["failed"] == read["unread"] == 0
    assert read["largest move"] == pytest.approx(0.0, abs=1e-9)
    for c, column in enumerate(read["channels"].values()):
        assert column["pairs"] == 6
        assert column["leaves"] == 0
        np.testing.assert_allclose(column["means"], truth.roi[c], rtol=0.05)
        assert column["swing"] < 0.05 * truth.roi[c]


def test_a_refit_that_reads_another_return_leaves_and_its_plan_moves(world):
    """One refit whose coefficients are half again the world's moves every channel's mean out of
    the 80 % before it and back, a swing of half its return; a refit that failed plays the status
    quo, and its regret's move is the status quo's regret."""
    cuts = refresh.windows(TRACK_M2.observe(world, 0))
    truth = TRACK_M2.truth(world)
    records = [_record(c, truth.best.weekly, _response(c, world)) for c in cuts]
    records[3] = _record(cuts[3], truth.best.weekly, _response(cuts[3], world, 1.5))
    records[5] = _record(cuts[5], truth.best.weekly, error="RuntimeError: no")
    with jax.enable_x64(True):
        read = refresh.swing(TRACK_M2, world, 0, records)
    assert read["failed"] == 1
    assert read["unread"] == 1
    status_quo = (
        truth.best.worth - worth(truth.cells, truth.quarter.status_quo)
    ) / truth.quarter.budget
    assert read["largest move"] == pytest.approx(status_quo, rel=1e-12, abs=0.0)
    for column in read["channels"].values():
        assert column["pairs"] == 4  # the failed refit is read in no pair
        assert column["leaves"] == 2
        means = column["means"]
        assert means[5] is None
        assert column["swing"] == pytest.approx(0.5 * means[2], rel=0.1, abs=0.0)


def test_a_reading_refuses_records_of_other_data_or_another_count(world):
    cuts = refresh.windows(TRACK_M2.observe(world, 0))
    weekly = TRACK_M2.truth(world).best.weekly
    records = [_record(c, weekly) for c in cuts]
    with pytest.raises(ValueError, match="one record a fit"):
        refresh.swing(TRACK_M2, world, 0, records[:-1])
    swapped = [records[1], records[0], *records[2:]]
    with pytest.raises(RuntimeError, match="other data"):
        refresh.swing(TRACK_M2, world, 0, swapped)
    read = refresh.swing(TRACK_M2, world, 0, records)
    assert read["unread"] == 7
    assert all(column["pairs"] == 0 for column in read["channels"].values())


def test_a_rolling_origin_reads_each_origin_s_forecast_and_the_weeks_all_windows_hold(short):
    """Family 7 at 52 weeks rolls its window: the weeks every origin reads are the 105th to the
    132nd; each origin's forecast is scored on its own quarter, and a forecast that is its
    quarter's sales exactly scores 0."""
    cuts = refresh.origins(short)
    records = []
    for c in cuts:
        observed = SHORT.observe(c, 0)
        target = c.shadow.sales
        records.append(
            _record(
                observed,
                SHORT.truth(c).best.weekly,
                _response(observed, short),
                forecast=np.repeat(target[None, :], 5, axis=0).tolist(),
            )
        )
    with jax.enable_x64(True):
        read = refresh.rolling(SHORT, short, 0, records)
    assert read["window"] == [105, 132]
    assert read["weeks"] == [132, 136, 140, 144, 148, 152, 156]
    assert read["crps"] == pytest.approx([0.0] * 7, abs=1e-12)
    assert read["mean crps"] == pytest.approx(0.0, abs=1e-12)
    assert read["largest move"] == pytest.approx(0.0, abs=1e-9)
    for column in read["channels"].values():
        assert column["leaves"] == 0
        assert column["pairs"] == 6


def test_the_section_is_written_from_its_json(world):
    cuts = refresh.windows(TRACK_M2.observe(world, 0))
    weekly = TRACK_M2.truth(world).best.weekly
    with jax.enable_x64(True):
        read = refresh.swing(
            TRACK_M2, world, 0, [_record(c, weekly, _response(c, world)) for c in cuts]
        )
    row = {"kind": "swing", "family": 0, "environment": "drawn", "seed": SEED, "k": 0}
    data = refresh.report([row | {"arm": "stacked", "reading": read}])
    text = refresh.markdown(data)
    assert "### Refresh swing" in text
    assert "Rolling origin" not in text
    assert "| 0 | drawn | 900 | 0 | stacked | 7, 0, 0 |" in text
    with pytest.raises(ValueError, match="names"):
        refresh.report([row | {"arm": "stacked"}])

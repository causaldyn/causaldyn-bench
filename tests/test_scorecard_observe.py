"""What an arm reads of a world: the export it runs on, written once and read back bit for bit, the
digest that covers every array of it, and the budgets run's version-1 exports, still read and
still digested by that run's rule."""

import dataclasses

import numpy as np
import pytest

from causaldyn_bench.budget_regret import digest, experiments, export, lift_rows
from causaldyn_bench.lift_calibration import STARTS
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard.observe import FIRST, VERSION, Observation, read
from causaldyn_bench.scorecard.observe import export as archive
from causaldyn_bench.scorecard.track_m2 import TRACK_M2


@pytest.fixture(scope="module")
def world():
    return TRACK_M2.world("drawn", 905)


@pytest.fixture(scope="module")
def observed(world):
    return TRACK_M2.observe(world, len(STARTS))


def _same(a: Observation, b: Observation) -> None:
    for field in dataclasses.fields(Observation):
        left, right = getattr(a, field.name), getattr(b, field.name)
        if field.name == "lift":
            for part in dataclasses.fields(left):
                np.testing.assert_array_equal(getattr(left, part.name), getattr(right, part.name))
        elif field.name in ("controls", "future_controls"):
            assert list(left) == list(right)
            for name in left:
                np.testing.assert_array_equal(left[name], right[name])
        else:
            np.testing.assert_array_equal(left, right)


def test_an_export_reads_back_as_the_observation_it_wrote(observed, tmp_path):
    path = archive(observed, tmp_path)
    assert path.name == "world_905.npz"
    back = read(path)
    _same(back, observed)
    assert back.digest() == observed.digest()
    assert (back.version, back.family, back.environment, back.k) == (VERSION, 0, "drawn", 4)


def test_a_world_is_exported_once_and_its_archive_is_never_overwritten(world, observed, tmp_path):
    path = archive(observed, tmp_path)
    written = path.read_bytes()
    assert archive(observed, tmp_path) == path
    assert path.read_bytes() == written
    other = TRACK_M2.observe(world, 0)  # the same world, other lift rows
    with pytest.raises(RuntimeError, match="exported once"):
        archive(other, tmp_path)
    assert path.read_bytes() == written


def _moved(observed: Observation) -> dict[str, Observation]:
    """The observation with one thing an arm reads changed, for each thing."""
    o = observed
    replace = dataclasses.replace
    lift = o.lift
    nudged = o.sales.copy()
    nudged[17] *= 1.0 + 1e-12
    return {
        "family": replace(o, family=1),
        "environment": replace(o, environment="reference"),
        "seed": replace(o, seed=906),
        "k": replace(o, k=3),
        "channels": replace(o, channels=("pla", "meta", "radio")),
        "sales": replace(o, sales=nudged),
        "spend": replace(o, spend=o.spend * (1.0 + 1e-12)),
        "controls": replace(o, controls={**o.controls, "price": o.controls["price"] + 1e-9}),
        "control names": replace(
            o,
            controls={"promo": o.controls["promotion"], "price": o.controls["price"]},
            future_controls={
                "promo": o.future_controls["promotion"],
                "price": o.future_controls["price"],
            },
        ),
        "future controls": replace(
            o,
            future_controls={**o.future_controls, "promotion": o.future_controls["promotion"] * 0},
        ),
        "kernel length": replace(o, kernel_length=5),
        "budget": replace(o, budget=o.budget * (1 + 1e-12)),
        "lower": replace(o, lower=o.lower * 0.99),
        "upper": replace(o, upper=o.upper * 1.01),
        "status quo": replace(o, status_quo=o.status_quo[::-1].copy()),
        "window": replace(o, roi_window=(104, 155)),
        "lift starts": replace(o, lift=replace(lift, start=tuple(s + 1 for s in lift.start))),
        "lift x": replace(o, lift=replace(lift, x=lift.x * 1.001)),
        "lift delta x": replace(o, lift=replace(lift, delta_x=lift.delta_x * 1.001)),
        "lift dropped": replace(o, lift=replace(lift, dropped=lift.dropped + 1)),
        "geos": replace(
            o,
            sales=np.column_stack([o.sales, o.sales]),
            spend=np.stack([o.spend, o.spend], axis=1),
            population=np.array([0.5, 0.5]),
        ),
    }


def test_the_digest_reads_everything_an_arm_reads(observed):
    stamps = {name: o.digest() for name, o in _moved(observed).items()}
    assert observed.digest() not in stamps.values()
    assert len(set(stamps.values())) == len(stamps)


def test_version_ones_digest_is_the_budgets_runs_and_reads_what_that_run_hashed(world, observed):
    history = world.history
    assert observed.digest(FIRST) == digest(history, lift_rows(experiments(history, 905)))
    moved = _moved(observed)
    for unread in ("family", "environment", "seed", "k", "budget", "future controls", "window"):
        assert moved[unread].digest(FIRST) == observed.digest(FIRST)
    for read_ in ("sales", "spend", "controls", "lift x", "channels"):
        assert moved[read_].digest(FIRST) != observed.digest(FIRST)
    with pytest.raises(ValueError, match="national worlds alone"):
        moved["geos"].digest(FIRST)


def test_a_version_one_export_reads_as_family_nought_at_the_top_rung(world, observed, tmp_path):
    history = world.history
    rows = lift_rows(experiments(history, 905))
    (tmp_path / "drawn").mkdir()
    path = export(history, 905, rows, tmp_path / "drawn")
    first = read(path)
    assert (first.version, first.family, first.environment, first.k) == (FIRST, 0, "drawn", 4)
    assert first.digest() == observed.digest(FIRST) == str(np.load(path)["digest"])
    assert first.future_controls is None
    quarter = Quarter.after(history)
    np.testing.assert_array_equal(first.status_quo, quarter.status_quo)
    assert first.budget == quarter.budget and first.planned == PLANNED
    assert first.roi_window == (105, 156)
    np.testing.assert_array_equal(first.lift.delta_y, rows.delta_y)
    with pytest.raises(ValueError, match="never exported"):
        archive(first, tmp_path / "again")


def test_an_export_whose_bits_do_not_match_its_digest_is_refused(observed, tmp_path):
    path = archive(observed, tmp_path)
    with np.load(path) as saved:
        fields = {name: saved[name] for name in saved.files}
    fields["sales"] = fields["sales"] * (1.0 + 1e-12)
    np.savez(path, **fields)
    with pytest.raises(RuntimeError, match="is not that of what it holds"):
        read(path)


def test_a_geo_axis_is_exported_with_its_population(observed, tmp_path):
    geo = _moved(observed)["geos"]
    back = read(archive(geo, tmp_path))
    _same(back, geo)
    np.testing.assert_array_equal(back.population, [0.5, 0.5])


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"version": 3}, "no export has version"),
        ({"future_controls": None}, "given in version 2"),
        ({"spend": np.zeros((156, 2))}, "spend is"),
        ({"sales": np.zeros(155)}, "spend is"),
        ({"controls": {"promotion": np.zeros(10), "price": np.zeros(156)}}, "every control"),
        ({"future_controls": {"price": np.zeros(13)}}, "the quarter's controls"),
        ({"lower": np.zeros(2)}, "one weekly spend a cell"),
        ({"roi_window": (0, 156)}, "outside weeks"),
    ],
)
def test_a_malformed_observation_is_refused(observed, change, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(observed, **change)

"""Family 15: the bench's map is siMMMulator's own, as the simulator returned it, quirks and all;
the history, the tests, the oracle, the quarter and the returns read that map, which the daily
recursion run forward confirms; and a record is read only by its digest."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.endogenous_mmm import YEAR
from causaldyn_bench.mmm_decision import PLANNED
from causaldyn_bench.scorecard import ladder
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.observe import export, read
from causaldyn_bench.scorecard.simmmulator import (
    CHANNELS,
    COMMIT,
    DAYS,
    DIGESTS,
    HISTORY,
    HORIZON,
    PHASES,
    TAIL,
    WEEKS,
    Record,
    Simmmulator,
    adstocked,
    daily_curve,
    diminished,
    exposures,
    weekly,
)
from causaldyn_bench.scorecard.truth import worth

DATA = Path(__file__).resolve().parent / "fixtures" / "simmmulator"
SEED = 500_000
AGREE = 1e-15  # the map's agreement with the simulator's own output, relative: a few last places
FAMILY = Simmmulator(DATA)


@pytest.fixture(scope="module")
def world():
    return FAMILY.world("demo", SEED)


@pytest.fixture(scope="module")
def truth(world):
    return FAMILY.truth(world)


def test_the_map_is_the_simulators_own(world):
    record = world.record
    for c in range(len(CHANNELS)):
        media = exposures(record.spend[:, c], record.cost[:, c], bool(record.clicks[c]))
        assert np.array_equal(media, record.media[:, c])
        conversions = record.returned(c, record.spend[:, c]) / record.revenue_per_conversion
        np.testing.assert_allclose(conversions, record.conversions[:, c], rtol=AGREE, atol=0.0)
    revenue = record.baseline + sum(record.returned(c, record.spend[:, c]) for c in range(3))
    np.testing.assert_allclose(revenue, record.revenue, rtol=AGREE, atol=0.0)
    assert tuple(record.clicks) == (False, False, True)


def test_the_simulator_recycles_its_shapes_and_inflexions_over_the_days(world):
    """Read by channel, as the demo's three gammas were meant, the inflexions miss the simulator's
    conversions by far more than read by the day."""
    record = world.record
    for c in range(len(CHANNELS)):
        media = exposures(record.spend[:, c], record.cost[:, c], bool(record.clicks[c]))
        adstock = adstocked(media, record.decay[c])
        meant = diminished(adstock, record.shape[c], record.inflexion[c, c]) * record.cvr[:, c]
        missed = np.abs(meant / record.conversions[:, c] - 1.0)
        assert missed.max() > 0.1
        days = np.arange(adstock.size) % PHASES
        assert np.all(missed[days == c] <= AGREE)


def test_each_inflexion_is_its_gammas_point_of_the_adstocks_range_rounded(world):
    record = world.record
    gamma = np.array([0.1, 0.2, 0.3])  # the demo's
    for c in range(len(CHANNELS)):
        media = exposures(record.spend[:, c], record.cost[:, c], bool(record.clicks[c]))
        adstock = adstocked(media, record.decay[c])
        low, high = adstock.min(), adstock.max()
        point = low + gamma * (high - low)
        inflexion = record.inflexion[c]
        assert np.all(np.abs(inflexion - point) <= 0.5e-4 * (1 + 1e-9))
        np.testing.assert_allclose(inflexion * 1e4, np.round(inflexion * 1e4), rtol=0.0, atol=1e-3)
        assert not np.allclose(inflexion, point, rtol=0.0, atol=1e-12)


def test_the_curve_rises_without_bound_and_is_convex_below_root_three_inflexions():
    curve = daily_curve(np.full(3, 2.0), np.array([1.0, 2.0, 3.0]))
    adstock = np.array([[0.5, 1.0, 1.5], [1e6, 1e6, 1e6]])
    value = curve.value(adstock, 1.0)
    assert np.all(value[1] > 0.999 * adstock[1])
    a = np.linspace(1e-3, 8.0, 4001)
    for inflexion in (1.0, 2.0, 3.0):
        one = daily_curve(np.array([2.0]), np.array([inflexion]))
        slope = one.slope(a[:, None], 1.0)[:, 0]
        step = 1e-6 * a
        numeric = (one.value((a + step)[:, None], 1.0) - one.value((a - step)[:, None], 1.0))[
            :, 0
        ] / (2 * step)
        np.testing.assert_allclose(slope, numeric, rtol=1e-7)
        rising = np.diff(slope) > 0.0
        knee = math.sqrt(3.0) * inflexion
        assert np.all(rising[a[1:] < 0.999 * knee]) and not np.any(rising[a[:-1] > 1.001 * knee])
    assert not curve.concave
    assert daily_curve(np.array([2.0]), np.array([1.0])).slope(np.zeros((1, 1)), 1.0)[0, 0] == 0.0


def test_the_mean_rate_is_its_truncated_normals_and_the_mean_cost_nearly_its_reciprocals():
    """The docstring's numbers: each day's rate is ``true + t``, ``t`` normal truncated below at
    ``-true``; and the mean of a cost's reciprocal lies above the mean cost's by its noise."""
    record = FAMILY.world("demo", SEED).record
    data = json.loads((DATA / f"{SEED}.json").read_text())["constants"]
    for c in range(len(CHANNELS)):
        true, mean, sd = (data[k][c] for k in ("true_cvr", "cvr_noise_mean", "cvr_noise_sd"))
        mass = stats.norm.sf(-true, mean, sd)
        first, _ = integrate.quad(
            lambda t, m=mean, s=sd: t * stats.norm.pdf(t, m, s), -true, np.inf
        )
        assert record.cvr_mean[c] == pytest.approx(true + first / mass, rel=1e-12, abs=0.0)
        rates = record.cvr[:, c]
        se = rates.std(ddof=1) / math.sqrt(rates.size)
        assert abs(rates.mean() - record.cvr_mean[c]) < 4.0 * se
    gaps = []
    for c in range(len(CHANNELS)):
        mean, sd = data["cost_mean"][c], data["cost_sd"][c]
        reciprocal, _ = integrate.quad(
            lambda x, m=mean, s=sd: stats.norm.pdf(x, m, s) / x, mean - 12 * sd, mean + 12 * sd
        )
        gaps.append(reciprocal * mean - 1.0)
    np.testing.assert_allclose(gaps, [1.1e-5, 5.6e-5, 1.5e-3], rtol=0.05)


def test_the_history_is_the_records_first_156_weeks(world):
    record = world.record
    assert np.array_equal(world.week, np.arange(1, WEEKS + 1)) and HISTORY == WEEKS * DAYS
    assert np.array_equal(world.sales, record.revenue[:HISTORY].reshape(WEEKS, DAYS).sum(axis=1))
    assert np.array_equal(world.spend, record.spend[:HISTORY].reshape(WEEKS, DAYS, 3).sum(axis=1))
    baseline = weekly(record.baseline[:HISTORY])
    np.testing.assert_allclose(baseline + world.media.sum(axis=1), world.sales, rtol=AGREE)


def test_a_geo_test_reads_the_worlds_curves_and_its_carryover_dies_out_to_the_bit(world):
    start = 100
    gap = world.geo_test("TV", (start,), test=4).true_gap
    assert np.all(gap[: start - 1] == 0.0)
    assert np.all(gap[start - 1 : start + 3] < 0.0)
    assert np.all(gap[start + 6 :] == 0.0)
    held = world.geo_test("Search", (start,), test=4, multiplier=1.0).true_gap
    assert np.all(held == 0.0)
    heavy = world.geo_test("Facebook", (start,), test=4, multiplier=2.0).true_gap
    assert np.all(heavy[start - 1 : start + 3] > 0.0)


def test_the_rungs_nest_to_the_bit(world):
    tests = [ladder.rung(world, SEED, k) for k in (1, 4)]
    for name in CHANNELS:
        assert np.array_equal(tests[0][name][-1].difference, tests[1][name][-1].difference)


def _forward(world, path, weekly_spend):
    """Each channel's revenue on the quarter's days and the tail's: the record run on from its
    first day with ``path``'s costs and rates over the cells' days, the history's spend, then
    ``weekly_spend`` spread evenly over the quarter's days, then nothing."""
    record = world.record
    span = slice(HISTORY, HISTORY + HORIZON)
    cost, cvr = record.cost.copy(), record.cvr.copy()
    if path == "expected":
        cost[span], cvr[span] = record.cost_mean, record.cvr_mean
    moved = dataclasses.replace(record, cost=cost, cvr=cvr)
    out = []
    for c in range(len(CHANNELS)):
        spend = np.zeros(HISTORY + HORIZON)
        spend[:HISTORY] = record.spend[:HISTORY, c]
        spend[HISTORY : HISTORY + PLANNED * DAYS] = weekly_spend[c] / DAYS
        out.append(moved.returned(c, spend)[HISTORY:])
    return np.stack(out)


def test_the_cells_are_the_daily_map_run_forward(world, truth):
    quarter = truth.quarter
    rng = np.random.default_rng(15)
    for path, cells in (("expected", truth.cells), ("realised", truth.realised)):
        for weekly_spend in (
            quarter.status_quo,
            quarter.project(rng.uniform(quarter.lower, quarter.upper)),
        ):
            forward = _forward(world, path, weekly_spend)
            for c, cell in enumerate(cells):
                assert cell.carry.size == HORIZON == PLANNED * DAYS + TAIL
                returned = cell.returns(float(weekly_spend[c]))
                np.testing.assert_allclose(returned, forward[c], rtol=1e-12, atol=0.0)
            # the tail holds all the carryover a float64 sum can: its last week adds nothing
            assert np.all(forward[:, -DAYS:].sum(axis=1) < 1e-16 * forward.sum(axis=1))


def test_the_target_is_the_quarters_revenue_at_the_status_quo(world, truth):
    forward = _forward(world, "realised", truth.quarter.status_quo)[:, : PLANNED * DAYS]
    baseline = world.record.baseline[HISTORY : HISTORY + PLANNED * DAYS]
    np.testing.assert_allclose(truth.target, weekly(baseline + forward.sum(axis=0)), rtol=1e-12)
    assert truth.scale == pytest.approx(float(np.mean(world.sales)), rel=1e-15, abs=0.0)


def test_the_true_returns_are_the_daily_maps_with_the_windows_spend_removed(world, truth):
    record, returns = world.record, truth.returns
    first, last = returns.window
    assert returns.window == (WEEKS - YEAR + 1, WEEKS)
    far = HISTORY + 300  # further than the tail: what it adds is below a float64 sum's last place
    inside = np.zeros(far, dtype=bool)
    inside[(first - 1) * DAYS : last * DAYS] = True
    for c in range(len(CHANNELS)):
        spend = np.zeros(far)
        spend[:HISTORY] = record.spend[:HISTORY, c]
        spent = spend[inside].sum()

        def total(series, c=c):
            return record.returned(c, series).sum()

        roi = (total(spend) - total(np.where(inside, 0.0, spend))) / spent
        assert returns.roi[c] == pytest.approx(roi, rel=1e-12, abs=0.0)
        step = 1e-6
        up = total(np.where(inside, spend * (1 + step), spend))
        down = total(np.where(inside, spend * (1 - step), spend))
        assert returns.marginal[c] == pytest.approx((up - down) / (2 * step * spent), rel=1e-6)


def test_no_plan_in_the_box_beats_the_oracle(truth):
    quarter = truth.quarter
    rng = np.random.default_rng(15)
    for cells, best in ((truth.cells, truth.best), (truth.realised, truth.hindsight)):
        assert quarter.feasible(best.weekly)
        for _ in range(300):
            weekly_spend = quarter.project(rng.uniform(quarter.lower, quarter.upper))
            assert worth(cells, weekly_spend) <= best.worth + TIE * quarter.budget
        assert best.worth == pytest.approx(worth(cells, best.weekly), rel=1e-15, abs=0.0)
    # the demo's largest channel returns the most: the best plan holds the others at their floors
    assert np.allclose(truth.best.weekly[1:], quarter.lower[1:], rtol=1e-9, atol=0.0)
    assert np.argmax(quarter.status_quo) == 0


def test_every_arm_reads_the_weekly_table_and_no_control(world, tmp_path):
    observation = FAMILY.observe(world, 4)
    assert number(FAMILY) == 15
    assert (observation.family, observation.environment, observation.seed) == (15, "demo", SEED)
    assert observation.kernel_length == WEEKS == world.week.size
    assert dict(observation.controls) == {} and dict(observation.future_controls or {}) == {}
    assert np.array_equal(observation.sales, world.sales)
    assert np.array_equal(observation.spend, world.spend)
    assert observation.lift.x.size + observation.lift.dropped == 4 * len(CHANNELS)
    again = read(export(observation, tmp_path))
    assert again.digest() == observation.digest()
    assert set(FAMILY.pilots) == set(FAMILY.scored) == {"demo"}


def test_a_record_is_read_by_its_digest_alone(tmp_path):
    assert SEED in DIGESTS
    data = (DATA / f"{SEED}.json").read_bytes()
    (tmp_path / f"{SEED}.json").write_bytes(data.replace(b"500000", b"500001", 1))
    with pytest.raises(ValueError, match="not world 500000 as recorded"):
        Simmmulator(tmp_path).world("demo", SEED)
    unrecorded = max(FAMILY.scored["demo"])
    (tmp_path / f"{unrecorded}.json").write_bytes(data)
    with pytest.raises(ValueError, match="no world"):
        Simmmulator(tmp_path).world("demo", unrecorded)
    with pytest.raises(ValueError, match="no environment"):
        FAMILY.world("genre", SEED)


@pytest.mark.parametrize(
    ("broken", "match"),
    [
        (lambda d: d.update(commit="0" * 40), "not a record of this family's demo"),
        (lambda d: d["daily"]["cost"][2].__setitem__(1200, 0.0), "a cost of 0 or less"),
        (
            lambda d: [d["daily"]["spend"][1].__setitem__(day, 0.0) for day in range(700, 707)],
            "spent nothing",
        ),
        (lambda d: d["constants"]["inflexions"].pop(), "three shapes"),
        (lambda d: d["daily"]["revenue"].pop(), "a record holds 1460 days"),
    ],
)
def test_a_record_that_breaks_an_invariant_is_refused(broken, match):
    data = json.loads((DATA / f"{SEED}.json").read_text())
    assert data["commit"] == COMMIT
    Record.parse(data)
    broken(data)
    with pytest.raises(ValueError, match=match):
        Record.parse(data)

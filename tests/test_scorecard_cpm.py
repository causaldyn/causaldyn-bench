"""Family 1's cpm environment: the CPM law draws the path its formula gives, the export carries the
impressions the spend bought, the response acts on them, a CPM held at its level is the static world
bit for bit, the best plan moves with the quarter's prices, and the bench's own arms plan a pilot
world of it end to end."""

import dataclasses
import math

import jax
import numpy as np
import pytest

from causaldyn_bench.budget_regret import TIE, plans
from causaldyn_bench.endogenous_mmm import YEAR, MediaMixWorld, _media
from causaldyn_bench.lift_calibration import STARTS
from causaldyn_bench.mmm_decision import PLANNED
from causaldyn_bench.scorecard import cpm, ladder
from causaldyn_bench.scorecard.continuation import Shadow
from causaldyn_bench.scorecard.drift import CPM, DRIFT, ENVIRONMENTS, FAMILY, LAWS, priced_world
from causaldyn_bench.scorecard.observe import export, read
from causaldyn_bench.scorecard.scoring import BUILT_IN, score_world
from causaldyn_bench.scorecard.seeds import Role, rng
from causaldyn_bench.scorecard.track_m2 import window

SEED = DRIFT.pilots[CPM][0]
WEEKS = 156


@pytest.fixture(scope="module")
def world():
    return DRIFT.world(CPM, SEED)


def _streams(seed, role, shape):
    return rng(FAMILY, role, seed).standard_normal(shape)


def test_the_cpm_law_draws_the_path_its_formula_gives(world):
    law = cpm.draw(rng(FAMILY, Role.PARAMETERS, SEED), world.history.channels)
    price = world.price
    assert price is not None
    np.testing.assert_array_equal(price.level, [5.0, 3.0, 20.0])
    horizon = PLANNED + world.history.kernel_length - 1
    assert price.cpm.shape == (WEEKS + horizon, 3)
    t = np.arange(1, WEEKS + horizon + 1)[:, None]
    y = np.log(price.cpm / price.level) - law.amplitude * np.sin(2 * math.pi * t / YEAR)
    shocks = np.vstack(
        [_streams(SEED, Role.PATH, (WEEKS + 1, 3)), _streams(SEED, Role.CONTINUATION, (horizon, 3))]
    )
    phi, s = law.persistence, law.sd
    before = np.vstack([s * shocks[0], y[:-1]])
    np.testing.assert_allclose(
        y - phi * before, s * np.sqrt(1 - phi**2) * shocks[1:], rtol=0.0, atol=1e-13
    )


def test_each_channels_cpm_parameters_lie_in_their_ranges():
    drawn = [
        cpm.draw(rng(FAMILY, Role.PARAMETERS, seed), ("pla", "meta", "tv"))
        for seed in DRIFT.scored[CPM][:200]
    ]
    for name, (low, high) in (
        ("amplitude", cpm.AMPLITUDE),
        ("persistence", cpm.PERSISTENCE),
        ("sd", cpm.SD),
    ):
        values = np.stack([getattr(law, name) for law in drawn])
        assert low <= values.min() and values.max() <= high
        assert values.max() - values.min() > 0.8 * (high - low)  # the range is drawn, not a point


def test_the_export_carries_the_impressions_the_spend_bought_and_the_response_acts_on_them(
    world, tmp_path
):
    """Read back from the export, the impressions are ``spend / CPM * 1000`` to the bit, and each
    channel's media are the drawn world's curve of them at ``lambda level / 1000`` an impression
    (to rounding: the world computes the same input as the spend times ``level / CPM``)."""
    observation = DRIFT.observe(world, len(STARTS))
    back = read(export(observation, tmp_path))
    price = world.price
    assert price is not None and back.impressions is not None
    cpm_history = price.cpm[:WEEKS]
    assert np.array_equal(back.impressions, back.spend / cpm_history * 1000.0)
    assert np.array_equal(back.impressions, price.impressions(back.spend))
    assert back.digest() == observation.digest()
    nudged = dataclasses.replace(observation, impressions=back.impressions * 1.5)
    assert nudged.digest() != observation.digest()
    history = world.history
    for c, (alpha, lam, beta) in enumerate(
        zip(history.retention, history.saturation, history.effect, strict=True)
    ):
        read_on = _media(
            back.impressions[:, c],
            alpha,
            lam * price.level[c] / 1000.0,
            beta,
            history.kernel_length,
            history.curve,
        )
        np.testing.assert_allclose(history.media[:, c], read_on, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(history.sales, history.premedia + history.media.sum(axis=1))


def test_impressions_are_refused_where_they_are_not_a_national_worlds_spend(world):
    observation = DRIFT.observe(world, 0)
    shown = observation.impressions
    assert shown is not None
    for wrong in (shown[:, :2], -shown, shown * np.nan):
        with pytest.raises(ValueError, match="impressions are a national world's"):
            dataclasses.replace(observation, impressions=wrong)
    plain = DRIFT.observe(DRIFT.world("static", SEED), 0)
    assert plain.impressions is None


def _same(a, b, kind):
    for field in dataclasses.fields(kind):
        assert np.array_equal(getattr(a, field.name), getattr(b, field.name)), field.name


@pytest.mark.parametrize("seed", [SEED, SEED + 1])
def test_a_cpm_held_at_its_level_is_the_static_world_bit_for_bit(seed):
    drawn = DRIFT.world(CPM, seed)
    assert drawn.price is not None
    held = priced_world(seed, cpm.Law.held(drawn.price.level))
    static = DRIFT.world("static", seed)
    _same(held.history, static.history, MediaMixWorld)
    _same(held.shadow, static.shadow, Shadow)
    assert np.array_equal(held.expected, static.expected)
    ours, theirs = DRIFT.truth(held), DRIFT.truth(static)
    assert ours.best.worth == theirs.best.worth and ours.best.price == theirs.best.price
    assert np.array_equal(ours.best.weekly, theirs.best.weekly)
    for mine, other in zip(ours.cells, theirs.cells, strict=True):
        assert np.array_equal(mine.carry, other.carry) and np.array_equal(mine.reach, other.reach)
    assert np.array_equal(ours.returns.roi, theirs.returns.roi)
    assert np.array_equal(ours.returns.marginal, theirs.returns.marginal)
    for k in (0, len(STARTS)):
        mine = DRIFT.observe(held, k)
        assert mine.impressions is not None
        unpriced = dataclasses.replace(mine, environment="static", impressions=None)
        assert unpriced.digest() == DRIFT.observe(static, k).digest()
    # the drawn law moves the prices, and with them the world
    assert np.ptp(np.log(drawn.price.cpm), axis=0).min() > 0.1
    assert not np.array_equal(drawn.history.sales, static.history.sales)


def _priced(world, factor, channel):
    """``world`` with ``channel``'s CPM times ``factor`` over the quarter and the tail."""
    price = world.price
    moved = price.cpm.copy()
    moved[WEEKS:, channel] *= factor
    return dataclasses.replace(world, price=cpm.Price(price.level, moved))


@pytest.mark.parametrize("channel, factor", [(0, 2.0), (1, 0.5), (2, 0.5)])
def test_the_best_plan_moves_with_the_quarters_prices(world, channel, factor):
    """On the first pilot world the best plan holds Meta and television at the foot of their boxes
    and gives paid shopping the rest: dearer impressions of paid shopping over the quarter, or
    cheaper ones of Meta or television, move euros to the channel whose price fell against the
    others'. The history's carryover is read at the history's prices either way, and each week of
    the quarter's spend reaches as far as it buys."""
    base = DRIFT.truth(world)
    quarter = base.quarter
    np.testing.assert_array_equal(base.best.weekly[1:], quarter.lower[1:])
    assert quarter.lower[0] + 1.0 < base.best.weekly[0] < quarter.upper[0] - 1.0
    moved = DRIFT.truth(_priced(world, factor, channel))
    shift = moved.best.weekly - base.best.weekly
    assert quarter.feasible(moved.best.weekly)
    assert (shift[channel] < -1.0) if factor > 1.0 else (shift[channel] > 1.0)
    cell, before = moved.cells[channel], base.cells[channel]
    assert np.array_equal(cell.carry, before.carry)
    np.testing.assert_allclose(cell.reach, before.reach / factor, rtol=1e-15, atol=0.0)


def _total(history, price, c, spend):
    """Channel ``c``'s effect summed over the history and its kernel's tail had it spent ``spend``
    over the history and nothing after it, at the world's prices."""
    tail = history.kernel_length - 1
    bought = np.concatenate([spend * price.buys[:WEEKS, c], np.zeros(tail)])
    return _media(
        bought,
        history.retention[c],
        history.saturation[c],
        history.effect[c],
        history.kernel_length,
        history.curve,
    ).sum()


def test_a_return_is_per_euro_spent_through_the_prices(world):
    """Each channel's ROI on last year is what its spend there returned, the effect it gave up
    gone dark, over the euros, not over the impressions; its mROI the slope of that return in a
    scaling of the year's spend, by a central difference."""
    truth = DRIFT.truth(world)
    history, price = world.history, world.price
    first, last = window(history)
    inside = np.zeros(WEEKS, dtype=bool)
    inside[first - 1 : last] = True
    for c in range(len(history.channels)):
        spend = history.spend[:, c]
        euros = spend[inside].sum()
        returned = _total(history, price, c, spend) - _total(
            history, price, c, np.where(inside, 0.0, spend)
        )
        assert truth.returns.roi[c] == pytest.approx(returned / euros, rel=1e-12, abs=0.0)
        h = 1e-5
        up, down = (
            _total(history, price, c, np.where(inside, (1.0 + step) * spend, spend))
            for step in (h, -h)
        )
        assert truth.returns.marginal[c] == pytest.approx(
            (up - down) / (2.0 * h) / euros, rel=1e-7, abs=0.0
        )
    # the same spends at the drawn world's level return another amount per euro
    static = DRIFT.truth(DRIFT.world("static", SEED)).returns
    assert not np.allclose(truth.returns.roi, static.roi, rtol=1e-3, atol=0.0)


def test_a_lift_test_reads_the_prices_of_its_weeks(world):
    history = world.history
    static = DRIFT.world("static", SEED).history
    for c, channel in enumerate(history.channels):
        ours = history.geo_test(channel, STARTS, seed=5).true_gap
        treated = history.spend[:, c].copy()
        for start in STARTS:
            treated[start - 1 : start + 3] = 0.0
        expected = (
            _media(
                treated * world.price.buys[:WEEKS, c],
                history.retention[c],
                history.saturation[c],
                history.effect[c],
                history.kernel_length,
                history.curve,
            )
            - history.media[:, c]
        )
        np.testing.assert_allclose(ours, expected, rtol=1e-12, atol=1e-9)
        assert np.any(ours != static.geo_test(channel, STARTS, seed=5).true_gap)


def test_the_bench_arms_plan_a_pilot_world_of_the_environment_end_to_end(world):
    """The arms the bench runs in its own environment, the budgets run's: the observational fit to
    the history, CHC's reading of the rung's tests and the myopic planner of it, beside the status
    quo and the equal split; each plans, none fails, and each is scored against the world's
    truth."""
    k = len(STARTS)
    observation = DRIFT.observe(world, k)
    with jax.enable_x64(True):  # the budgets run's arms run at float64 only
        arms, fits = plans(world.history, ladder.rung(world.history, SEED, k))
    records = {
        name: {
            "error": None,
            "weekly": [float(w) for w in plan],
            "digest": observation.digest(),
            "version": observation.version,
        }
        for name, plan in arms.items()
        if name not in BUILT_IN
    }
    assert set(records) == {"observational", "CHC", "myopic"} and set(fits) == {"pla", "meta", "tv"}
    scored = score_world(DRIFT, CPM, SEED, k, records, pilot=True)
    assert (scored.family, scored.environment, scored.index) == (1, CPM, 0)
    assert set(scored.arms) == {*BUILT_IN, *records}
    for score in scored.arms.values():
        assert score.failure is None and score.regret >= -TIE and score.realised is None


def test_the_family_names_the_cpm_environment_after_its_laws():
    assert ENVIRONMENTS == (*LAWS, CPM) == tuple(DRIFT.scored) == tuple(DRIFT.pilots)
    assert DRIFT.pilots[CPM] == range(220_600, 220_700)
    assert DRIFT.scored[CPM] == range(227_000, 228_000)
    assert any("cpm environment" in label for label in DRIFT.labels)

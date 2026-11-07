"""Family 1's cpm environment: the CPM law draws the path its formula gives, the export carries the
impressions the spend bought, the response acts on them, a CPM held at its level is the static world
bit for bit, the best plan reads the prices the law expects and the plan in hindsight the realised
ones, each moving with its own, and the bench's own arms plan a pilot world of it end to end."""

import dataclasses
import math

import jax
import numpy as np
import pytest

from causaldyn_bench.budget_regret import TIE, plans
from causaldyn_bench.endogenous_mmm import YEAR, MediaMixWorld, _media
from causaldyn_bench.lift_calibration import STARTS
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard import cpm, ladder
from causaldyn_bench.scorecard.continuation import Shadow
from causaldyn_bench.scorecard.drift import CPM, DRIFT, ENVIRONMENTS, FAMILY, LAWS, priced_world
from causaldyn_bench.scorecard.observe import export, read
from causaldyn_bench.scorecard.scoring import BUILT_IN, score_world
from causaldyn_bench.scorecard.seeds import Role, rng
from causaldyn_bench.scorecard.track_m2 import still, window
from causaldyn_bench.scorecard.truth import Cell, cells, oracle, worth

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
    assert ours.realised is None and ours.hindsight is None  # a held price has nothing to realise
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


def _priced(world, factor, channel, which):
    """``world`` with ``channel``'s CPM times ``factor`` over the quarter and the tail, as the law
    expects it or as the quarter realises it."""
    price = world.price
    drawn, expected = price.cpm.copy(), price.expected.copy()
    if which == "realised":
        drawn[WEEKS:, channel] *= factor
    else:
        expected[WEEKS:, channel] /= factor
    return dataclasses.replace(world, price=cpm.Price(price.level, drawn, expected))


@pytest.mark.parametrize("which", ["expected", "realised"])
@pytest.mark.parametrize("channel, factor", [(0, 2.0), (1, 0.5), (2, 0.5)])
def test_each_best_plan_moves_with_its_own_prices(world, channel, factor, which):
    """On the first pilot world both the best plan and the plan in hindsight hold Meta and
    television at the foot of their boxes and give paid shopping the rest. Dearer impressions of
    paid shopping over the quarter, or cheaper ones of Meta or television, move euros to the
    channel whose price fell against the others': in the best plan where the law expects it, in
    the plan in hindsight where the quarter realises it, and in that one alone. The history's
    carryover is read at the history's prices either way, and each week of the quarter's spend
    reaches as far as it buys."""
    base = DRIFT.truth(world)
    quarter = base.quarter
    assert base.hindsight is not None
    for plan in (base.best, base.hindsight):
        np.testing.assert_array_equal(plan.weekly[1:], quarter.lower[1:])
        assert quarter.lower[0] + 1.0 < plan.weekly[0] < quarter.upper[0] - 1.0
    moved = DRIFT.truth(_priced(world, factor, channel, which))
    on, still_on = ("best", "hindsight") if which == "expected" else ("hindsight", "best")
    shift = getattr(moved, on).weekly - getattr(base, on).weekly
    assert quarter.feasible(getattr(moved, on).weekly)
    assert (shift[channel] < -1.0) if factor > 1.0 else (shift[channel] > 1.0)
    assert np.array_equal(getattr(moved, still_on).weekly, getattr(base, still_on).weekly)
    cells_on = "cells" if which == "expected" else "realised"
    cell, before = getattr(moved, cells_on)[channel], getattr(base, cells_on)[channel]
    assert np.array_equal(cell.carry, before.carry)
    np.testing.assert_allclose(cell.reach, before.reach / factor, rtol=1e-15, atol=0.0)


def _quarters(world, draws, seed):
    """``draws`` quarters of what a euro buys, ``(draws, horizon, channels)``, run on from the log
    in the history's last week by the world's law, from a generator of the test's own."""
    law = cpm.draw(rng(FAMILY, Role.PARAMETERS, world.seed), world.history.channels)
    price = world.price
    horizon = price.cpm.shape[0] - WEEKS
    t = np.arange(1, WEEKS + horizon + 1)[:, None]
    season = law.amplitude * np.sin(2 * math.pi * t / YEAR)
    log = np.log(price.cpm[:WEEKS] / price.level) - season[:WEEKS]
    shocks = np.random.default_rng(seed).standard_normal((draws, horizon, 3))
    y = np.broadcast_to(log[-1], (draws, 3)).copy()
    out = np.empty_like(shocks)
    for h in range(horizon):
        y = law.persistence * y + law.sd * np.sqrt(1 - law.persistence**2) * shocks[:, h]
        out[:, h] = np.exp(-season[WEEKS + h] - y)
    return out


def test_the_best_plan_reads_what_a_euro_is_expected_to_buy(world):
    """The expected prices are the history's as they were and, after it, the mean of the quarters
    the law runs on from the history's last week: within five standard errors of 20 000 sampled
    quarters at every week and channel (a false red below 1e-4 over its 54 cells). The plan best
    on them is the certainty-equivalent plan: on 2 000 sampled quarters the plan best on their
    average returns more by at most 1e-3 a euro of the budget, here and on the pilot world where
    the difference is largest of the first eight."""
    price = world.price
    np.testing.assert_array_equal(price.expected[:WEEKS], price.buys[:WEEKS])
    sampled = _quarters(world, 20_000, 7)
    mean, error = sampled.mean(axis=0), sampled.std(axis=0, ddof=1) / math.sqrt(20_000)
    assert np.all(np.abs(mean - price.expected[WEEKS:]) < 5.0 * error)
    assert not np.array_equal(price.expected[WEEKS:], price.buys[WEEKS:])
    for case in (world, DRIFT.world(CPM, SEED + 5)):
        assert case.price is not None
        history, known = case.history, case.price.buys[:WEEKS]
        quarter, path = Quarter.after(history), still(history)
        best = DRIFT.truth(case).best
        each = [
            cells(history, quarter, path, buys=np.vstack([known, q]))
            for q in _quarters(case, 2_000, 8)
        ]
        average = tuple(
            Cell(
                np.concatenate([e[c].carry for e in each]),
                np.concatenate([e[c].reach for e in each]),
                each[0][c].curve,
                each[0][c].scale,
                np.concatenate([e[c].effect for e in each]) / len(each),
            )
            for c in range(3)
        )
        gap = (oracle(average, quarter).worth - worth(average, best.weekly)) / quarter.budget
        assert -1e-12 <= gap < 1e-3


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
        assert score.failure is None and score.regret >= -TIE
        assert score.realised is not None and score.realised >= -TIE


def test_the_family_names_the_cpm_environment_after_its_laws():
    assert ENVIRONMENTS == (*LAWS, CPM) == tuple(DRIFT.scored) == tuple(DRIFT.pilots)
    assert DRIFT.pilots[CPM] == range(220_600, 220_700)
    assert DRIFT.scored[CPM] == range(227_000, 228_000)
    assert any("cpm environment" in label for label in DRIFT.labels)

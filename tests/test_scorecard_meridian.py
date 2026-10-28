"""Family 12: the world's transforms are Meridian's own, as Meridian's library returned them, and
its priors Meridian's defaults; the world is Track M v2's beneath its media; and the oracle, the
quarter and the returns read Meridian's window and curve, which the formula run on past the history
confirms."""

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chc.response import Channel, GeometricAdstock, Hill, marginal_roi, roi
from scipy.stats import norm

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.mmm_decision import PLANNED, Quarter, drawn
from causaldyn_bench.scorecard.continuation import shadow
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.meridian import (
    DECAY,
    EC,
    MAX_LAG,
    MERIDIAN,
    ROI,
    adstocked,
    hill,
    media_scale,
    parameters,
    saturated,
)
from causaldyn_bench.scorecard.seeds import BLOCK, FIRST
from causaldyn_bench.scorecard.track_m2 import still
from causaldyn_bench.scorecard.truth import worth

ROOT = Path(__file__).resolve().parents[1]
WEEKS = 156
WINDOW = MAX_LAG + 1
AGREE = 1e-9  # the record's agreement with Meridian's own transforms, relative


@pytest.fixture(scope="module")
def worlds():
    return [MERIDIAN.world("priors", seed) for seed in MERIDIAN.scored["priors"][:4]]


def _plain(spend: np.ndarray, decay: float, inflexion: float, effect: float) -> np.ndarray:
    """The module's formula written out week by week: the window's geometric weights over their
    sum, nothing before the first week, and the Hill of slope 1."""
    total = sum(decay**i for i in range(WINDOW))
    out = np.empty(spend.size)
    for t in range(spend.size):
        x = sum(decay**i / total * spend[t - i] for i in range(WINDOW) if t - i >= 0)
        out[t] = effect * x / (x + inflexion)
    return out


def test_the_worlds_transforms_are_meridians_own():
    record = json.loads((ROOT / "results" / "meridian_form.json").read_text())
    assert record["meridian"] == "2.1.0" and record["dtype"] == "float64"
    defaults = record["defaults"]
    assert defaults["max_lag"] == MAX_LAG
    assert tuple(defaults["alpha_m"]) == DECAY
    assert tuple(defaults["ec_m"]) == EC
    assert defaults["slope_m"] == 1.0
    assert tuple(defaults["roi_m"]) == ROI
    assert defaults["distributions"] == {
        "alpha_m": "Uniform",
        "ec_m": "TruncatedNormal",
        "slope_m": "Deterministic",
        "roi_m": "LogNormal",
    }
    histories = {}
    for case in record["cases"]:
        seed = case["seed"]
        history = histories.setdefault(seed, MERIDIAN.world("priors", seed).history)
        c = history.channels.index(case["channel"])
        spend = history.spend[:, c]
        scale = media_scale(spend)
        assert scale == pytest.approx(case["scale"], rel=AGREE, abs=0.0)
        adstock = adstocked(spend, case["decay"])
        np.testing.assert_allclose(adstock / scale, case["adstock"], rtol=AGREE, atol=0.0)
        curve = saturated(adstock, case["ec"] * scale)
        np.testing.assert_allclose(curve, case["saturated"], rtol=AGREE, atol=0.0)
        if case["corner"] is None:
            assert (history.retention[c], history.ec[c]) == (case["decay"], case["ec"])
            meridians = history.effect[c] * np.asarray(case["saturated"])
            np.testing.assert_allclose(history.media[:, c], meridians, rtol=AGREE, atol=0.0)
    corners = {(case["channel"], case["corner"]) for case in record["cases"]}
    assert corners >= {(name, end) for name in ("pla", "meta", "tv") for end in ("low", "high")}


def test_a_world_is_track_m2s_beneath_its_media(worlds):
    for world in worlds:
        history = world.history
        generator = drawn(world.seed)
        base = generator.simulate(world.seed)
        assert np.array_equal(history.spend, base.spend)
        assert np.array_equal(history.premedia, base.premedia)
        assert np.array_equal(history.observed_price, base.observed_price)
        assert np.array_equal(history.observed_promotion, base.observed_promotion)
        assert np.array_equal(history.sales, base.premedia + history.media.sum(axis=1))
        assert not np.allclose(history.media, base.media, rtol=0.01)
        ours, theirs = Quarter.after(history), Quarter.after(base)
        for field in ("budget", "lower", "upper", "status_quo", "history"):
            assert np.array_equal(getattr(ours, field), getattr(theirs, field))
        beneath = shadow(generator, base, world.seed, still(base))
        for field in ("premedia", "observed_price", "observed_promotion", "sentiment"):
            assert np.array_equal(getattr(world.shadow, field), getattr(beneath, field))


def test_a_seed_always_draws_the_same_world_from_the_familys_own_block():
    seed = MERIDIAN.pilots["priors"][5]
    one, two = (MERIDIAN.world("priors", seed).history for _ in range(2))
    assert np.array_equal(one.media, two.media) and one.effect == two.effect
    assert one.retention != MERIDIAN.world("priors", seed + 1).history.retention
    first = FIRST + BLOCK * 12
    for seeds in (*MERIDIAN.pilots.values(), *MERIDIAN.scored.values()):
        assert first <= seeds[0] and seeds[-1] < first + BLOCK
    assert MERIDIAN.pilots["priors"] == range(first, first + 100)
    assert MERIDIAN.scored["priors"] == range(first + 1_000, first + 2_000)


def test_the_media_are_the_documented_formula(worlds):
    """Each channel's media written out in plain numpy, its inflexion ec times the median of its
    positive weeks and its coefficient set by Meridian's ROI identity over the history."""
    for world in worlds:
        history = world.history
        decay, ec, drawn_roi = parameters(history.channels, world.seed)
        for c in range(len(history.channels)):
            spend = history.spend[:, c]
            inflexion = float(ec[c]) * float(np.median(spend[spend > 0.0]))
            curve = _plain(spend, float(decay[c]), inflexion, 1.0)
            effect = float(drawn_roi[c]) * spend.sum() / curve.sum()
            assert history.saturation[c] == pytest.approx(inflexion, rel=1e-15, abs=0.0)
            assert history.effect[c] == pytest.approx(effect, rel=1e-12, abs=0.0)
            np.testing.assert_allclose(history.media[:, c], effect * curve, rtol=1e-12, atol=0.0)
            roas = history.media[:, c].sum() / spend.sum()
            assert roas == pytest.approx(history.roi[c], rel=1e-12, abs=0.0)
        assert history.kernel_length == WINDOW


def test_each_channels_parameters_follow_meridians_default_priors():
    """On 300 scored worlds' 900 draws a parameter's mean lies within four of its standard errors
    of the prior's: the decay's uniform, ec's normal truncated to [0.1, 10], and the log ROI's
    normal."""
    seeds = MERIDIAN.scored["priors"][:300]
    decay, ec, drawn_roi = (
        np.concatenate(p)
        for p in zip(*(parameters(("pla", "meta", "tv"), seed) for seed in seeds), strict=True)
    )
    n = decay.size
    assert 0.0 <= decay.min() < 0.01 and 0.99 < decay.max() < 1.0
    assert abs(decay.mean() - 0.5) < 4 * math.sqrt(1 / 12 / n)
    location, spread, lowest, highest = EC
    a, b = (lowest - location) / spread, (highest - location) / spread
    mass = norm.cdf(b) - norm.cdf(a)
    mean = location + spread * (norm.pdf(a) - norm.pdf(b)) / mass
    sd = spread * math.sqrt(
        1 + (a * norm.pdf(a) - b * norm.pdf(b)) / mass - ((norm.pdf(a) - norm.pdf(b)) / mass) ** 2
    )
    assert lowest <= ec.min() < lowest + 0.05 and ec.max() <= highest
    assert abs(ec.mean() - mean) < 4 * sd / math.sqrt(n)
    assert abs(ec.std() - sd) < 4 * sd / math.sqrt(2 * n)
    mu, sigma = ROI
    logged = np.log(drawn_roi)
    assert abs(logged.mean() - mu) < 4 * sigma / math.sqrt(n)
    assert abs(logged.std() - sigma) < 4 * sigma / math.sqrt(2 * n)


def test_hills_slope_is_its_values_derivative_and_the_curve_is_concave():
    adstock = np.linspace(0.0, 400.0, 50)
    curve = hill()
    step = 1e-5 * (adstock + 1.0)
    numeric = (curve.value(adstock + step, 120.0) - curve.value(adstock - step, 120.0)) / (2 * step)
    np.testing.assert_allclose(curve.slope(adstock, 120.0), numeric, rtol=1e-7)
    assert curve.value(np.array([120.0]), 120.0)[0] == 0.5 and curve.concave


def test_a_geo_test_reads_meridians_carryover_for_its_window_alone(worlds):
    history = worlds[0].history
    start, test = 100, 4
    gap = history.geo_test("tv", (start,), test=test).true_gap
    assert np.all(gap[: start - 1] == 0.0)
    assert np.all(gap[start - 1 : start - 1 + test + MAX_LAG] < 0.0)
    assert np.all(gap[start - 1 + test + MAX_LAG :] == 0.0)


def _returned(history, c: int, spend: np.ndarray) -> np.ndarray:
    """What channel ``c`` returns in each week of the formula run on ``spend``, then the window's
    weeks with nothing spent, beyond which it returns nothing."""
    return _plain(
        np.concatenate([spend, np.zeros(MAX_LAG)]),
        history.retention[c],
        history.saturation[c],
        history.effect[c],
    )


def test_a_plans_worth_is_the_world_run_on_over_the_quarter_and_the_tail(worlds):
    """Over the quarter each cell returns what the formula run on past the history does, and the
    shadow's media are the status quo's; a plan's worth is all the formula returns over the
    quarter and the window's tail, which is all its spend returns."""
    rng = np.random.default_rng(12)
    for world in worlds:
        history, truth = world.history, MERIDIAN.truth(world)
        quarter = truth.quarter
        plans = [quarter.status_quo, truth.best.weekly]
        plans += [quarter.project(rng.uniform(quarter.lower, quarter.upper)) for _ in range(3)]
        for weekly in plans:
            run = 0.0
            for c, cell in enumerate(truth.cells):
                spend = np.concatenate([history.spend[:, c], np.full(PLANNED, weekly[c])])
                ours = _returned(history, c, spend)[WEEKS:]
                assert cell.carry.size == PLANNED + WINDOW - 1 == ours.size
                np.testing.assert_allclose(cell.returns(weekly[c]), ours, rtol=1e-12, atol=0.0)
                run += ours.sum()
            assert worth(truth.cells, weekly) == pytest.approx(run, rel=1e-12, abs=0.0)
        for c in range(len(history.channels)):
            spend = np.concatenate([history.spend[:, c], np.full(PLANNED, quarter.status_quo[c])])
            ours = _returned(history, c, spend)[WEEKS : WEEKS + PLANNED]
            np.testing.assert_allclose(world.shadow.media[:, c], ours, rtol=1e-12, atol=0.0)


def test_the_true_returns_are_the_formula_with_the_windows_spend_removed(worlds):
    for world in worlds:
        history, returns = world.history, MERIDIAN.truth(world).returns
        first, last = returns.window
        inside = np.zeros(WEEKS, dtype=bool)
        inside[first - 1 : last] = True
        for c in range(len(history.channels)):
            spend = history.spend[:, c]
            spent = spend[inside].sum()

            def total(series, c=c, history=history):
                return _returned(history, c, series).sum()

            ratio = (total(spend) - total(np.where(inside, 0.0, spend))) / spent
            assert returns.roi[c] == pytest.approx(ratio, rel=1e-12, abs=0.0)
            step = 1e-6
            up = total(np.where(inside, spend * (1 + step), spend))
            down = total(np.where(inside, spend * (1 - step), spend))
            marginal = (up - down) / (2 * step * spent)
            assert returns.marginal[c] == pytest.approx(marginal, rel=1e-6, abs=0.0)


def test_the_world_and_its_returns_are_chcs_channels_in_meridians_form(worlds):
    """The arms' vocabulary (:mod:`causaldyn_bench.scorecard.mapping`) reads Meridian's model as
    chc's channel of a normalised geometric kernel over max_lag + 1 weeks and a Hill at ec times
    the media scale; the world's media and returns are that channel's."""
    history = worlds[0].history
    returns = MERIDIAN.truth(worlds[0]).returns
    first, last = returns.window
    with jax.enable_x64(True):
        for c in range(len(history.channels)):
            channel = Channel(
                GeometricAdstock(history.retention[c], length=WINDOW, normalized=True),
                Hill(history.ec[c] * history.scale[c], 1.0),
                history.effect[c],
            )
            media = np.asarray(channel(jnp.asarray(history.spend[:, c])))
            np.testing.assert_allclose(media, history.media[:, c], rtol=1e-13)
            spend = jnp.asarray(np.concatenate([history.spend[:, c], np.zeros(WINDOW - 1)]))
            window = slice(first - 1, last)
            ours = (returns.roi[c], returns.marginal[c])
            chcs = (float(roi(channel, spend, window)), float(marginal_roi(channel, spend, window)))
            assert ours == pytest.approx(chcs, rel=1e-12, abs=0.0)


def test_every_cell_is_concave_and_no_plan_in_the_box_beats_the_oracle(worlds):
    rng = np.random.default_rng(12)
    for world in worlds:
        truth = MERIDIAN.truth(world)
        quarter = truth.quarter
        assert all(cell.concave for cell in truth.cells)
        assert quarter.feasible(truth.best.weekly)
        for _ in range(200):
            weekly = quarter.project(rng.uniform(quarter.lower, quarter.upper))
            assert worth(truth.cells, weekly) <= truth.best.worth + TIE * quarter.budget


def test_every_arm_is_told_the_kernel_spans_meridians_window(worlds):
    observation = MERIDIAN.observe(worlds[0], 4)
    assert number(MERIDIAN) == 12
    assert observation.family == 12 and observation.environment == "priors"
    assert observation.kernel_length == WINDOW == 9
    assert set(MERIDIAN.pilots) == set(MERIDIAN.scored) == {"priors"}
    with pytest.raises(ValueError, match="no environment"):
        MERIDIAN.world("genre", MERIDIAN.scored["priors"][0])

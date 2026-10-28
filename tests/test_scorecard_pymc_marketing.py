"""Family 13: the world's media are PyMC-Marketing's default class at its default priors, written
here as the tool writes it; the world is Track M v2's beneath its media; the arm's mapping reads
the tool's response as this world's media; and the oracle, exact on concave cells, reads the world
run on over the quarter and the kernel's tail."""

import jax
import numpy as np
import pytest
from scipy import stats

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.mmm_decision import PLANNED, Quarter, drawn
from causaldyn_bench.scorecard.continuation import shadow
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.mapping import channels
from causaldyn_bench.scorecard.pymc_marketing import (
    PYMC_MARKETING,
    parameters,
)
from causaldyn_bench.scorecard.track_m2 import still
from causaldyn_bench.scorecard.truth import worth

WEEKS = 156
LENGTH = 6  # Track M v2's kernel length


@pytest.fixture(scope="module")
def worlds():
    return [PYMC_MARKETING.world("priors", seed) for seed in PYMC_MARKETING.scored["priors"][:4]]


def _pymc(spend, alpha, lam, beta, scale, target):
    """PyMC-Marketing's channel contribution as its MMM computes it: the spend over the channel's
    scale, through the geometric adstock normalised over ``l_max`` lags with nothing spent before,
    the logistic saturation, and ``beta`` times the target's scale."""
    x = spend / scale
    weights = alpha ** np.arange(LENGTH)
    weights /= weights.sum()
    adstock = np.array(
        [sum(weights[lag] * x[t - lag] for lag in range(min(LENGTH, t + 1))) for t in range(x.size)]
    )
    return beta * target * (1 - np.exp(-lam * adstock)) / (1 + np.exp(-lam * adstock))


def _beta(world):
    """Each channel's ``beta`` in the tool's scaling, on the history's largest sales: the
    coefficient that gives the channel the media the Track M v2 world beneath returns over the
    history (the module's CHOICE), not a draw from the tool's ``HalfNormal(2)``."""
    history = world.history
    base = drawn(world.seed).simulate(world.seed)
    alpha, lam = parameters(history.channels, world.seed)
    beta = []
    for c in range(len(history.channels)):
        spend = history.spend[:, c]
        curve = _pymc(spend, alpha[c], lam[c], 1.0, spend.max(), 1.0)
        beta.append(base.media[:, c].sum() / curve.sum() / np.max(history.sales))
    return np.array(beta)


def test_the_media_are_pymc_marketings_at_the_drawn_parameters(worlds):
    for world in worlds:
        history = world.history
        alpha, lam = parameters(history.channels, world.seed)
        beta = _beta(world)
        for c in range(len(history.channels)):
            spend = history.spend[:, c]
            ours = _pymc(spend, alpha[c], lam[c], beta[c], spend.max(), np.max(history.sales))
            np.testing.assert_allclose(history.media[:, c], ours, rtol=1e-12, atol=0.0)


def test_each_channel_returns_over_the_history_what_track_m2_beneath_it_returns(worlds):
    """The CHOICE for ``beta``, family 14's: the sales, media's share of them and each channel's
    return per euro over the history are those of the Track M v2 world beneath."""
    for world in worlds:
        history = world.history
        base = drawn(world.seed).simulate(world.seed)
        np.testing.assert_allclose(history.media.sum(axis=0), base.media.sum(axis=0), rtol=1e-13)
        np.testing.assert_allclose(history.roas(), base.roas(), rtol=1e-13)
        assert history.sales.sum() == pytest.approx(base.sales.sum(), rel=1e-13, abs=0.0)


def test_a_world_is_track_m2s_beneath_its_media(worlds):
    for world in worlds:
        history = world.history
        base = drawn(world.seed).simulate(world.seed)
        assert np.array_equal(history.spend, base.spend)
        assert np.array_equal(history.premedia, base.premedia)
        assert np.array_equal(history.observed_price, base.observed_price)
        assert np.array_equal(history.observed_promotion, base.observed_promotion)
        assert np.array_equal(history.sales, base.premedia + history.media.sum(axis=1))
        assert history.kernel_length == base.kernel_length == LENGTH
        assert history.curve == base.curve == "tanh"
        assert not np.allclose(history.media, base.media, rtol=0.01)
        beneath = shadow(world.generator, base, world.seed, still(base))
        for name in ("week", "baseline", "promotion", "premedia", "observed_price"):
            assert np.array_equal(getattr(world.shadow, name), getattr(beneath, name))
        assert np.array_equal(world.shadow.observed_promotion, beneath.observed_promotion)
        ours, theirs = PYMC_MARKETING.truth(world).quarter, Quarter.after(base)
        assert ours.budget == theirs.budget
        assert np.array_equal(ours.status_quo, theirs.status_quo)


def test_a_seed_always_draws_the_same_world(worlds):
    world = worlds[0]
    again = PYMC_MARKETING.world("priors", world.seed)
    for name in ("media", "sales", "spend", "premedia"):
        assert np.array_equal(getattr(again.history, name), getattr(world.history, name))
    assert again.history.effect == world.history.effect
    assert np.array_equal(again.shadow.sales, world.shadow.sales)
    assert not np.array_equal(worlds[1].history.media, world.history.media)


def test_the_family_draws_its_own_block():
    assert number(PYMC_MARKETING) == 13
    assert set(PYMC_MARKETING.pilots) == set(PYMC_MARKETING.scored) == {"priors"}
    assert PYMC_MARKETING.pilots["priors"] == range(460_000, 460_100)
    assert PYMC_MARKETING.scored["priors"] == range(461_000, 462_000)
    assert all(label.endswith(".") for label in PYMC_MARKETING.labels)
    with pytest.raises(ValueError, match="no environment 'drawn'"):
        PYMC_MARKETING.world("drawn", 461_000)


def test_the_parameters_follow_the_default_priors():
    """Over 300 scored worlds each channel's ``alpha`` and ``lam`` pass a Kolmogorov-Smirnov test,
    at 1 %, of the prior PyMC-Marketing 1.2.0's source states, and no two channels' draws are
    one; ``beta`` is not drawn."""
    seeds = PYMC_MARKETING.scored["priors"][:300]
    draws = np.stack([np.stack(parameters(("pla", "meta", "tv"), seed)) for seed in seeds])
    priors = (
        stats.beta(1.0, 3.0),
        stats.gamma(3.0, scale=1.0),
    )
    for p, prior in enumerate(priors):
        for c in range(3):
            assert stats.kstest(draws[:, p, c], prior.cdf).pvalue > 0.01
        assert not np.array_equal(draws[:, p, 0], draws[:, p, 1])


def test_the_arms_mapping_of_a_pymc_marketing_response_is_the_worlds_media(worlds):
    """The arm reads its response as chc's channel at ``2 S / lam`` and ``beta`` times the
    observed sales' largest week (:mod:`causaldyn_bench.scorecard.mapping`); at the world's own
    parameters in the tool's scaling it returns the world's media."""
    for world in worlds:
        observed = PYMC_MARKETING.observe(world, 0)
        history = world.history
        alpha, lam = parameters(history.channels, world.seed)
        beta = _beta(world)
        target = float(np.max(observed.sales))
        response = {
            "kernel": "GeometricAdstock",
            "length": observed.kernel_length,
            "normalized": True,
            "curve": "Tanh",
            "parameters": {
                name: {
                    "retention": [float(alpha[c])],
                    "scale": [2.0 * float(np.max(observed.spend[:, c])) / float(lam[c])],
                    "coefficient": [float(beta[c]) * target],
                }
                for c, name in enumerate(observed.channels)
            },
            "decomposition": {},
        }
        with jax.enable_x64(True):
            (draw,) = channels(response, observed.channels)
            for c in range(len(observed.channels)):
                media = np.asarray(draw[c](observed.spend[:, c]))
                np.testing.assert_allclose(media, history.media[:, c], rtol=1e-12, atol=0.0)


def test_every_cell_is_concave_and_the_oracle_is_exact(worlds):
    rng = np.random.default_rng(13)
    for world in worlds:
        truth = PYMC_MARKETING.truth(world)
        quarter = truth.quarter
        assert all(cell.concave for cell in truth.cells)
        assert quarter.feasible(truth.best.weekly)
        for _ in range(200):
            weekly = quarter.project(rng.uniform(quarter.lower, quarter.upper))
            assert worth(truth.cells, weekly) <= truth.best.worth + TIE * quarter.budget


def test_a_plans_worth_is_the_world_run_on_over_the_quarter_and_the_tail(worlds):
    """The world run on is PyMC-Marketing's formula at the drawn parameters, on the history's
    scales, over the history, a quarter at the plan and the kernel's tail with nothing spent."""
    rng = np.random.default_rng(113)
    for world in worlds:
        history, truth = world.history, PYMC_MARKETING.truth(world)
        alpha, lam = parameters(history.channels, world.seed)
        beta = _beta(world)
        quarter = truth.quarter
        for weekly in (
            truth.best.weekly,
            quarter.project(rng.uniform(quarter.lower, quarter.upper)),
        ):
            ran = 0.0
            for c in range(len(history.channels)):
                spend = np.concatenate(
                    [history.spend[:, c], np.full(PLANNED, weekly[c]), np.zeros(LENGTH - 1)]
                )
                scale = history.spend[:, c].max()
                media = _pymc(spend, alpha[c], lam[c], beta[c], scale, np.max(history.sales))
                ran += media[WEEKS:].sum()
            assert worth(truth.cells, weekly) == pytest.approx(ran, rel=1e-12, abs=0.0)


def test_every_arm_is_told_the_kernels_length(worlds):
    observation = PYMC_MARKETING.observe(worlds[0], 4)
    assert observation.family == 13 and observation.environment == "priors"
    assert observation.kernel_length == LENGTH

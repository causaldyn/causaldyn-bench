"""Family 2: every channel's log spend moves by kappa 0.15 for each stationary sd of the sentiment,
so the realised kappa, the slope of the log spend on the sentiment beyond Track M v2's, is kappa to
rounding; the sentiment holds the stationary sd that reaction is read in, within its Monte Carlo
error; and the media and the sales follow the spend while all the business read before media
stays. In the search-demand environment paid shopping's spend alone reads the sentiment, every arm
reads the queries, and the truth states television's return with them held beside its whole."""

import dataclasses
import math

import numpy as np
import pytest

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.endogenous_mmm import MediaMixWorld, _media
from causaldyn_bench.mmm_decision import PLANNED, drawn
from causaldyn_bench.scorecard.confounded import CONFOUNDED, ETA, KAPPAS, SEARCH, confound
from causaldyn_bench.scorecard.observe import VERSION, export, read
from causaldyn_bench.scorecard.scoring import score_world
from causaldyn_bench.scorecard.seeds import Role, rng
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from causaldyn_bench.scorecard.truth import oracle, worth

SD = 0.03 / math.sqrt(1.0 - 0.7**2)  # his sentiment's stationary sd, 0.0420084025208403 (Maxima)


def _slope(y: np.ndarray, x: np.ndarray) -> float:
    x = x - x.mean()
    return float(x @ (y - y.mean()) / (x @ x))


@pytest.mark.parametrize("name", KAPPAS)
def test_the_realised_kappa_is_kappa(name):
    kappa = KAPPAS[name]
    assert name == f"kappa-{kappa:g}"
    for seed in CONFOUNDED.pilots[name][:4]:
        world = CONFOUNDED.world(name, seed).history
        base = TRACK_M2.world("drawn", seed).history
        z = base.sentiment / SD
        moved = np.log(world.spend) - np.log(base.spend)
        np.testing.assert_allclose(moved, kappa * 0.15 * z[:, None] * np.ones(3), atol=1e-13)
        realised = [
            (_slope(np.log(world.spend[:, c]), z) - _slope(np.log(base.spend[:, c]), z)) / 0.15
            for c in range(3)
        ]
        np.testing.assert_allclose(realised, kappa, rtol=1e-12)


def test_the_sentiment_holds_the_stationary_sd_its_reaction_is_read_in():
    """Over the 300 worlds of the pilot blocks, the sentiment's variance over its 156 weeks, in
    units of its stationary variance, has the mean its AR(1) from its N(0, 0.05^2) start gives,
    0.973024731903606 (Maxima, from the weeks' covariance), within four standard errors (0.011).
    So kappa is read per sd of the demand the history shows."""
    variances = np.array(
        [
            np.var(drawn(seed).simulate(seed).sentiment / SD, ddof=1)
            for name in KAPPAS
            for seed in CONFOUNDED.pilots[name]
        ]
    )
    error = variances.std(ddof=1) / math.sqrt(variances.size)
    assert abs(variances.mean() - 0.973024731903606) <= 4.0 * error
    assert error < 0.012


def test_kappa_nought_is_track_m2_bit_for_bit():
    base = TRACK_M2.world("drawn", CONFOUNDED.pilots["kappa-1"][0]).history
    same = confound(base, 0.0)
    for field in dataclasses.fields(MediaMixWorld):
        assert np.array_equal(getattr(same, field.name), getattr(base, field.name)), field.name


def test_the_media_and_the_sales_follow_the_spend_and_nothing_the_business_read_before_media():
    seed = CONFOUNDED.pilots["kappa-2"][1]
    world = CONFOUNDED.world("kappa-2", seed).history
    base = TRACK_M2.world("drawn", seed).history
    for field in ("premedia", "budget", "bidding", "plan_noise", "sentiment", "observed_price"):
        assert np.array_equal(getattr(world, field), getattr(base, field)), field
    for c, (alpha, lam, beta) in enumerate(
        zip(world.retention, world.saturation, world.effect, strict=True)
    ):
        media = _media(world.spend[:, c], alpha, lam, beta, world.kernel_length)
        np.testing.assert_array_equal(world.media[:, c], media)
    np.testing.assert_array_equal(world.sales, world.premedia + world.media.sum(axis=1))
    assert np.all(world.spend != base.spend)
    # the spend reads the demand: its correlation with the sentiment rises well past Track M v2's
    rises = [np.corrcoef(np.log(w.spend[:, 1]), w.sentiment)[0, 1] for w in (base, world)]
    assert rises[1] > rises[0] + 0.2


def test_an_unknown_environment_is_refused():
    with pytest.raises(ValueError, match="no environment 'kappa-0'"):
        CONFOUNDED.world("kappa-0", CONFOUNDED.scored["kappa-1"][0])


def test_in_the_search_demand_world_paid_shoppings_spend_alone_reads_the_sentiment():
    for seed in CONFOUNDED.pilots[SEARCH][:4]:
        world = CONFOUNDED.world(SEARCH, seed).history
        base = TRACK_M2.world("drawn", seed).history
        pla = world.channels.index("pla")
        moved = np.log(world.spend[:, pla]) - np.log(base.spend[:, pla])
        np.testing.assert_allclose(moved, 2.0 * 0.15 * base.sentiment / SD, rtol=0.0, atol=1e-13)
        others = [c for c in range(3) if c != pla]
        np.testing.assert_array_equal(world.spend[:, others], base.spend[:, others])
        np.testing.assert_array_equal(world.media[:, others], base.media[:, others])
        np.testing.assert_array_equal(world.sales, world.premedia + world.media.sum(axis=1))


def test_every_arm_reads_the_queries_which_measure_the_demand_and_televisions_part():
    """The queries over the history are the sales before media and half television's media, over
    the base, read with a log-normal error of sd 0.02 from the family's measurement stream; over
    the quarter, the same at the status quo from its continuation stream. The error's sd over the
    156 weeks of each of four worlds is within 0.005 of 0.02, past four of its standard errors
    (0.0011)."""
    for seed in CONFOUNDED.pilots[SEARCH][:4]:
        world = CONFOUNDED.world(SEARCH, seed)
        history, shadow, base = world.history, world.shadow, world.generator.base
        tv = history.channels.index("tv")
        observation = CONFOUNDED.observe(world, 0)
        demand = (history.premedia + ETA * history.media[:, tv]) / base
        error = rng(2, Role.MEASUREMENT, seed).standard_normal(history.week.size)
        np.testing.assert_allclose(
            observation.controls["queries"], demand * np.exp(0.02 * error - 0.0002), rtol=1e-14
        )
        assert abs(np.std(np.log(observation.controls["queries"] / demand), ddof=1) - 0.02) < 0.005
        assert observation.future_controls is not None
        ahead = (shadow.premedia + ETA * shadow.media[:, tv]) / base
        error = rng(2, Role.CONTINUATION, seed).standard_normal(PLANNED)
        np.testing.assert_allclose(
            observation.future_controls["queries"],
            ahead * np.exp(0.02 * error - 0.0002),
            rtol=1e-14,
        )
    for name in KAPPAS:
        world = CONFOUNDED.world(name, CONFOUNDED.pilots[name][0])
        assert set(CONFOUNDED.observe(world, 0).controls) == {"promotion", "price"}


def test_the_queries_read_back_from_the_export(tmp_path):
    world = CONFOUNDED.world(SEARCH, CONFOUNDED.pilots[SEARCH][0])
    for k in (0, 4):
        observation = CONFOUNDED.observe(world, k)
        back = read(export(observation, tmp_path / str(k)))
        assert back.digest() == observation.digest()
        for name in ("controls", "future_controls"):
            np.testing.assert_array_equal(
                getattr(back, name)["queries"], getattr(observation, name)["queries"]
            )


def test_the_truth_states_televisions_return_with_the_queries_held_beside_its_whole():
    seed = CONFOUNDED.pilots[SEARCH][0]
    world = CONFOUNDED.world(SEARCH, seed)
    truth = CONFOUNDED.truth(world)
    tv = world.history.channels.index("tv")
    assert truth.direct is not None and truth.direct.window == truth.returns.window
    for whole, held in (
        (truth.returns.roi, truth.direct.roi),
        (truth.returns.marginal, truth.direct.marginal),
    ):
        np.testing.assert_array_equal(np.delete(held, tv), np.delete(whole, tv))
        assert held[tv] == pytest.approx((1.0 - ETA) * whole[tv], rel=1e-15, abs=0.0)
        assert held[tv] < whole[tv]
    for name in KAPPAS:
        assert CONFOUNDED.truth(CONFOUNDED.world(name, CONFOUNDED.pilots[name][0])).direct is None


def test_a_plan_on_televisions_direct_return_leaves_the_best_plan():
    """Swap television's cell for one that earns its direct return only, the return a model that
    holds the queries reads, and the oracle's plan moves off the best: on the first six pilot worlds
    it never spends more on television, and on three it leaves more than a cent of each euro. On the
    other three television is at its floor in both plans."""
    regrets = []
    for seed in CONFOUNDED.pilots[SEARCH][:6]:
        world = CONFOUNDED.world(SEARCH, seed)
        truth = CONFOUNDED.truth(world)
        tv = world.history.channels.index("tv")
        swapped = list(truth.cells)
        swapped[tv] = dataclasses.replace(swapped[tv], effect=swapped[tv].effect * (1.0 - ETA))
        plan = oracle(swapped, truth.quarter)
        assert plan.weekly[tv] <= truth.best.weekly[tv] + 1e-9
        regrets.append((truth.best.worth - worth(truth.cells, plan.weekly)) / truth.quarter.budget)
    assert min(regrets) >= -TIE
    assert sum(regret > 0.01 for regret in regrets) == 3


def test_a_search_demand_world_is_scored_and_its_best_plan_scores_nought():
    seed = CONFOUNDED.pilots[SEARCH][0]
    world = CONFOUNDED.world(SEARCH, seed)
    best = CONFOUNDED.truth(world).best
    observation = CONFOUNDED.observe(world, 4)
    record = {
        "digest": observation.digest(),
        "version": VERSION,
        "error": None,
        "weekly": best.weekly.tolist(),
    }
    scores = score_world(CONFOUNDED, SEARCH, seed, 4, {"oracle": record}, pilot=True)
    assert abs(scores.arms["oracle"].regret) <= TIE
    assert scores.arms["status quo"].regret > TIE

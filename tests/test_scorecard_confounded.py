"""Family 2: every channel's log spend moves by kappa 0.15 for each stationary sd of the sentiment,
so the realised kappa, the slope of the log spend on the sentiment beyond Track M v2's, is kappa to
rounding; the sentiment holds the stationary sd that reaction is read in, within its Monte Carlo
error; and the media and the sales follow the spend while all the business read before media
stays."""

import dataclasses
import math

import numpy as np
import pytest

from causaldyn_bench.endogenous_mmm import MediaMixWorld, _media
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.confounded import CONFOUNDED, KAPPAS, confound
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

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

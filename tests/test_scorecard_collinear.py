"""Family 3: Meta's planning noise is PLA's at correlation r, from the drawn world's own noise; its
spend follows the bidding rule as PLA's does; nothing else moves; and over each pilot block the two
channels' noise is correlated at r within its Monte Carlo error."""

import dataclasses
import math

import numpy as np
import pytest

from causaldyn_bench.endogenous_mmm import MediaMixWorld, _media
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.collinear import COLLINEAR, CORRELATIONS, couple
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

PLA, META, TV = 0, 1, 2
WEEKS = 156


@pytest.mark.parametrize("name", CORRELATIONS)
def test_metas_noise_is_plas_at_r_and_its_spend_follows_the_bidding_rule(name):
    r = CORRELATIONS[name]
    assert name == f"r-{r:g}"
    for seed in COLLINEAR.pilots[name][:3]:
        world = COLLINEAR.world(name, seed).history
        base = TRACK_M2.world("drawn", seed).history
        assert drawn(seed).bidding_channel == "pla" and base.channels == ("pla", "meta", "tv")
        np.testing.assert_array_equal(world.plan_noise[:, [PLA, TV]], base.plan_noise[:, [PLA, TV]])
        coupled = r * base.plan_noise[:, PLA] + math.sqrt(1.0 - r**2) * base.plan_noise[:, META]
        np.testing.assert_allclose(world.plan_noise[:, META], coupled, rtol=1e-15, atol=0.0)
        np.testing.assert_allclose(
            np.log(world.spend[:, META]) - np.log(base.spend[:, META]),
            world.plan_noise[:, META] - base.plan_noise[:, META] + np.log(base.bidding),
            rtol=0.0,
            atol=1e-13,
        )
        for field in ("spend", "media"):
            kept = getattr(world, field)[:, [PLA, TV]]
            np.testing.assert_array_equal(kept, getattr(base, field)[:, [PLA, TV]])
        alpha, lam, beta = world.retention[META], world.saturation[META], world.effect[META]
        media = _media(world.spend[:, META], alpha, lam, beta, world.kernel_length)
        np.testing.assert_array_equal(world.media[:, META], media)
        np.testing.assert_array_equal(world.sales, world.premedia + world.media.sum(axis=1))
        for field in ("premedia", "budget", "bidding", "sentiment", "observed_price"):
            assert np.array_equal(getattr(world, field), getattr(base, field)), field


def test_at_r_nought_meta_moves_by_the_bidding_rule_alone():
    base = TRACK_M2.world("drawn", COLLINEAR.pilots["r-0.9"][0]).history
    bid = couple(base, 0.0)
    np.testing.assert_array_equal(bid.plan_noise, base.plan_noise)
    np.testing.assert_array_equal(bid.spend[:, META], base.spend[:, META] * base.bidding)
    unbid = dataclasses.replace(base, bidding=np.ones(WEEKS))
    same = couple(unbid, 0.0)
    for field in dataclasses.fields(MediaMixWorld):
        assert np.array_equal(getattr(same, field.name), getattr(unbid, field.name)), field.name


@pytest.mark.parametrize("name", CORRELATIONS)
def test_the_realised_correlation_is_r_within_its_monte_carlo_error(name):
    """Fisher's z of each world's correlation of the two channels' noise over its 156 weeks has
    mean ``atanh r + r / (2 (n - 1))`` and variance ``1 / (n - 3)`` to first order; over the 100
    pilot worlds the mean lies within four of its standard errors, 0.0081 each."""
    r = CORRELATIONS[name]
    z = np.array(
        [
            math.atanh(np.corrcoef(*COLLINEAR.world(name, s).history.plan_noise[:, :2].T)[0, 1])
            for s in COLLINEAR.pilots[name]
        ]
    )
    error = 1.0 / math.sqrt((WEEKS - 3) * z.size)
    assert abs(z.mean() - math.atanh(r) - r / (2 * (WEEKS - 1))) <= 4.0 * error
    assert z.std(ddof=1) == pytest.approx(1.0 / math.sqrt(WEEKS - 3), rel=0.25, abs=0.0)

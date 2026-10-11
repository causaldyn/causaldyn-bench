"""Family 5: from week 12 every channel's spend is the drawn world's times exp(rho s), s last week's
sales over the ten weeks before it, logged, read off the world's own sales, so the realised rho is
rho to rounding: the rule is deterministic given the world, with no Monte Carlo error; at rho = 0
the world is Track M v2's bit for bit; and the media and the sales follow the spend while all the
business read before media stays."""

import dataclasses

import numpy as np
import pytest

from causaldyn_bench.endogenous_mmm import MediaMixWorld, _media
from causaldyn_bench.scorecard.performance import PERFORMANCE, RHOS, follow
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

WEEKS = 156


def _surprise(sales: np.ndarray) -> np.ndarray:
    """``s_{t-1}`` for weeks 12 to 156, numbered from 1: last week's sales over the ten before."""
    return np.array(
        [np.log(sales[t - 1] / np.mean(sales[t - 11 : t - 1])) for t in range(11, WEEKS)]
    )


@pytest.mark.parametrize("name", RHOS)
def test_the_spend_follows_last_weeks_surprise_at_rho(name):
    rho = RHOS[name]
    assert name == f"rho-{rho:g}"
    for seed in PERFORMANCE.pilots[name][:4]:
        world = PERFORMANCE.world(name, seed).history
        base = TRACK_M2.world("drawn", seed).history
        np.testing.assert_array_equal(world.spend[:11], base.spend[:11])
        moved = np.log(world.spend[11:] / base.spend[11:])
        surprise = _surprise(world.sales)
        np.testing.assert_allclose(
            moved, rho * surprise[:, None] * np.ones(3), rtol=0.0, atol=1e-14
        )
        realised = float(np.sum(moved * surprise[:, None]) / (3.0 * surprise @ surprise))
        assert realised == pytest.approx(rho, rel=0.0, abs=1e-13)
        if rho > 0.0:
            assert np.abs(moved).max() > 0.05


def test_rho_nought_is_track_m2_bit_for_bit():
    seed = PERFORMANCE.pilots["rho-0"][0]
    world, base = PERFORMANCE.world("rho-0", seed), TRACK_M2.world("drawn", seed)
    for field in dataclasses.fields(MediaMixWorld):
        name = field.name
        assert np.array_equal(getattr(world.history, name), getattr(base.history, name)), name
    np.testing.assert_array_equal(world.shadow.sales, base.shadow.sales)
    same = follow(base.history, 0.0)
    assert all(
        np.array_equal(getattr(same, f.name), getattr(base.history, f.name))
        for f in dataclasses.fields(MediaMixWorld)
    )


def test_the_media_and_the_sales_follow_the_spend_and_nothing_the_business_read_before_media():
    seed = PERFORMANCE.pilots["rho-0.6"][1]
    world = PERFORMANCE.world("rho-0.6", seed).history
    base = TRACK_M2.world("drawn", seed).history
    for field in ("premedia", "budget", "bidding", "plan_noise", "sentiment", "observed_price"):
        assert np.array_equal(getattr(world, field), getattr(base, field)), field
    for c, (alpha, lam, beta) in enumerate(
        zip(world.retention, world.saturation, world.effect, strict=True)
    ):
        media = _media(world.spend[:, c], alpha, lam, beta, world.kernel_length)
        np.testing.assert_array_equal(world.media[:, c], media)
    np.testing.assert_array_equal(world.sales, world.premedia + world.media.sum(axis=1))
    assert np.all(world.spend[11:] != base.spend[11:])
    # the rule read the sales its own spend bought: on the sales before media it would differ
    on_premedia = _surprise(world.premedia)
    assert np.abs(np.log(world.spend[11:, 0] / base.spend[11:, 0]) - 0.6 * on_premedia).max() > 1e-3


def test_an_unknown_environment_is_refused():
    with pytest.raises(ValueError, match="no environment 'rho-1'"):
        PERFORMANCE.world("rho-1", PERFORMANCE.scored["rho-0"][0])

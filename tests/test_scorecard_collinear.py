"""Family 3: Meta takes the share c of PLA's movements, its planning noise PLA's at correlation c
and the bidding rule to the power c, from the drawn world's own; nothing else moves; each
environment's c is the one its pilot worlds put at r; and over the scored worlds the correlation an
arm sees, of the two channels' log spend, is r within its Monte Carlo error."""

import dataclasses
import math

import numpy as np
import pytest

from causaldyn_bench.endogenous_mmm import MediaMixWorld, _media
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.collinear import COLLINEAR, CORRELATIONS, SHARES, couple
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

PLA, META, TV = 0, 1, 2


def _seen(history: MediaMixWorld) -> float:
    """The correlation of PLA's and Meta's log spend over the history, as an arm reads it."""
    return float(np.corrcoef(np.log(history.spend[:, [PLA, META]]).T)[0, 1])


@pytest.mark.parametrize("name", CORRELATIONS)
def test_meta_takes_the_share_c_of_plas_noise_and_bidding_rule(name):
    r, c = CORRELATIONS[name], SHARES[name]
    assert name == f"r-{r:g}" and 0.0 < c < 1.0
    for seed in COLLINEAR.pilots[name][:3]:
        world = COLLINEAR.world(name, seed).history
        base = TRACK_M2.world("drawn", seed).history
        assert drawn(seed).bidding_channel == "pla" and base.channels == ("pla", "meta", "tv")
        np.testing.assert_array_equal(world.plan_noise[:, [PLA, TV]], base.plan_noise[:, [PLA, TV]])
        coupled = c * base.plan_noise[:, PLA] + math.sqrt(1.0 - c**2) * base.plan_noise[:, META]
        np.testing.assert_allclose(world.plan_noise[:, META], coupled, rtol=1e-15, atol=0.0)
        np.testing.assert_allclose(
            np.log(world.spend[:, META]) - np.log(base.spend[:, META]),
            world.plan_noise[:, META] - base.plan_noise[:, META] + c * np.log(base.bidding),
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


def test_at_c_nought_the_world_is_track_m2s_and_at_one_the_log_spends_differ_by_a_constant():
    base = TRACK_M2.world("drawn", COLLINEAR.pilots["r-0.9"][0]).history
    same = couple(base, 0.0)
    for field in dataclasses.fields(MediaMixWorld):
        assert np.array_equal(getattr(same, field.name), getattr(base, field.name)), field.name
    whole = couple(base, 1.0)
    gap = np.log(whole.spend[:, META]) - np.log(whole.spend[:, PLA])
    np.testing.assert_allclose(gap, math.log(40.0 / 60.0), rtol=0.0, atol=1e-13)
    assert _seen(whole) == pytest.approx(1.0, rel=0.0, abs=1e-14)
    assert _seen(base) < 0.9


@pytest.mark.parametrize("name", CORRELATIONS)
def test_each_share_is_where_the_pilot_worlds_see_r(name):
    """c is the bisection's root on the 300 pilot worlds, to its six places: there their mean is r
    to 1e-6, and a millionth of c either side brackets r."""
    r = CORRELATIONS[name]
    worlds = [drawn(s).simulate(s) for e in CORRELATIONS for s in COLLINEAR.pilots[e]]

    def seen(c: float) -> float:
        return float(np.mean([_seen(couple(h, c)) for h in worlds]))

    assert len(worlds) == 300
    assert seen(SHARES[name]) == pytest.approx(r, rel=0.0, abs=1e-6)
    assert seen(SHARES[name] - 1e-6) < r < seen(SHARES[name] + 1e-6)


@pytest.mark.parametrize("name", CORRELATIONS)
def test_over_the_scored_worlds_an_arm_sees_r_within_its_monte_carlo_error(name):
    """Over the first 100 scored worlds the mean correlation an arm sees lies within four standard
    errors of r. The error is the scored mean's and the 300 pilot worlds' that set c together,
    ``sqrt(1 + 1/3)`` times the scored mean's alone: 0.00112, 0.00045 and 0.00015 here."""
    r = CORRELATIONS[name]
    seen = np.array([_seen(COLLINEAR.world(name, s).history) for s in COLLINEAR.scored[name][:100]])
    error = seen.std(ddof=1) / math.sqrt(seen.size) * math.sqrt(1.0 + 1.0 / 3.0)
    assert abs(seen.mean() - r) <= 4.0 * error
    assert error < {0.9: 0.0015, 0.97: 0.0006, 0.99: 0.0002}[r]

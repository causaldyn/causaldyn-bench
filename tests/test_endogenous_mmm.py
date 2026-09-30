"""Track M v2's world against the paper it is written from.

The generator is Heusch (2026b)'s, written clean-room from his equations. A reading of a paper can
go wrong in ways no single equation shows, so the tests hold the whole: his reference instance's
reported numbers, which a correct reading reproduces in distribution over histories, and the two
properties his geo-test procedure asserts.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from chc.response import Channel, GeometricAdstock, Tanh

from causaldyn_bench.endogenous_mmm import EndogenousMediaMix

SEEDS = range(40)


@pytest.fixture(scope="module")
def worlds():
    return [EndogenousMediaMix().simulate(seed) for seed in SEEDS]


def test_each_media_effect_is_a_chc_response_channel(worlds) -> None:
    """His Eq. 7-8 are a normalised six-week geometric kernel through ``tanh(lambda x / 2)``: the
    library's ``Channel`` with ``Tanh(2 / lambda)``, computed here without it."""
    world = worlds[0]
    tolerance = 1e3 * float(jnp.finfo(jnp.asarray(1.0).dtype).eps)
    for c, (alpha, lam, beta) in enumerate(
        zip(world.retention, world.saturation, world.effect, strict=True)
    ):
        channel = Channel(GeometricAdstock(alpha, length=6, normalized=True), Tanh(2 / lam), beta)
        np.testing.assert_allclose(
            np.asarray(channel(world.spend[:, c])), world.media[:, c], rtol=tolerance
        )


def test_the_reference_instance_is_reproduced_in_distribution(worlds) -> None:
    """His seed-42 instance's numbers (Section IV, Table II) against the medians of 40 histories.

    Returns and shares sit within about a percent. Television's return pins the burst weeks
    (48-52). PLA's mean spend runs above his 75.8, his instance in the lower tail of ours, so it
    gets the widest band.
    """

    def median(statistic) -> np.ndarray:
        return np.median([statistic(world) for world in worlds], axis=0)

    np.testing.assert_allclose(median(lambda w: w.roas()), [4.20, 2.90, 1.56], rtol=0.01)
    blended = median(lambda w: w.media.sum() / w.spend.sum())
    np.testing.assert_allclose(blended, 3.30, rtol=0.01)
    shares = median(lambda w: [w.baseline.sum(), w.promotion.sum(), w.media.sum()] / w.sales.sum())
    np.testing.assert_allclose(shares, [0.703, 0.111, 0.187], atol=0.01)
    spend = median(lambda w: w.spend.mean(axis=0))
    assert np.all(np.abs(spend / np.array([75.8, 52.5, 27.5]) - 1.0) <= [0.03, 0.02, 0.02])


def test_the_measurement_layer_reads_as_the_paper_reports(worlds) -> None:
    """A promotion indicator agreeing with the truth in 82.1 % of weeks, and a price reading that
    correlates 0.89 with the true price."""
    agreement = np.median([np.mean(w.observed_promotion == w.promotion_weeks) for w in worlds])
    correlation = np.median(
        [np.corrcoef(w.observed_price, w.price_position * (1.0 - w.discount))[0, 1] for w in worlds]
    )
    assert agreement == pytest.approx(0.821, abs=0.02)
    assert correlation == pytest.approx(0.89, abs=0.02)


def test_a_geo_test_moves_only_the_tested_channel(worlds) -> None:
    """His Section V: the gap is zero before the first test and once each test's carryover is
    spent, never positive for a go-dark test, and the noise of opposite signs keeps the total."""
    world = worlds[0]
    experiment = world.geo_test("pla", (20, 55), seed=1)
    gap = experiment.true_gap
    assert np.all(gap[:19] == 0.0)
    assert np.all(gap[19 + 4 + 5 : 54] == 0.0)  # four dark weeks, then the kernel's five lags
    assert np.all(gap[19:23] < 0.0)
    assert np.all(gap <= 0.0)
    np.testing.assert_allclose(
        experiment.sales_treated + experiment.sales_control, 2 * world.sales + gap
    )


def test_with_every_mechanism_off_spend_follows_only_its_plan() -> None:
    """Each coordination mechanism is a switch: off, the budget and the bid hold at 1 and
    television has no bursts."""
    quiet = EndogenousMediaMix(
        budget_feedback=False, anticipatory=False, bursts=False, bidding=False
    ).simulate(3)
    assert np.all(quiet.budget == 1.0)
    assert np.all(quiet.bidding == 1.0)
    tilt = np.log(quiet.spend / np.array([60.0, 40.0, 10.0])) - 0.15 * quiet.seasonality[:, None]
    assert np.abs(tilt).max() < 0.5  # the planning noise alone, sd 0.1

"""Track M v2's world against the paper it is written from.

The generator is Heusch (2026b)'s, written clean-room from his equations. A reading of a paper can
go wrong in ways no single equation shows, so the tests hold the whole: his reference instance's
reported numbers, which a correct reading reproduces in distribution over histories, and the two
properties his geo-test procedure asserts.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chc.response import (
    Channel,
    Exponential,
    GeometricAdstock,
    Hill,
    Logistic,
    MichaelisMenten,
    Tanh,
    Weibull,
)
from scipy.optimize import brentq

from causaldyn_bench.endogenous_mmm import CURVES, EndogenousMediaMix, _media

SEEDS = range(40)


def _library(curve: str, lam: float):
    """The library's family at the scale that puts half its ceiling at ``ln 3 / lam``; the
    logistic's midpoint is solved for here, apart from the generator's closed form."""
    half = math.log(3.0) / lam
    if curve == "logistic-4":
        scale = brentq(lambda k: float(Logistic(k, 4.0)(half)) - 0.5, 0.5 * half, 2.0 * half)
        return Logistic(scale, 4.0)
    return {
        "tanh": lambda: Tanh(2.0 / lam),
        "exponential": lambda: Exponential(half / math.log(2.0)),
        "michaelis-menten": lambda: MichaelisMenten(half),
        "hill-2": lambda: Hill(half, 2.0),
        "weibull-2": lambda: Weibull(half / math.sqrt(math.log(2.0)), 2.0),
    }[curve]()


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


@pytest.mark.parametrize("curve", sorted(CURVES))
def test_every_curve_is_the_library_s_family_at_half_its_ceiling_where_his_is(curve) -> None:
    """Each curve, value and slope, against ``chc.response``'s family at the matching scale, and
    half its ceiling at ``ln 3 / lambda``, where his ``tanh(lambda x / 2)`` is."""
    lam = 0.0073
    half = math.log(3.0) / lam
    adstock = np.linspace(0.0, 4.0 * half, 97)
    with jax.enable_x64(True):
        library = _library(curve, lam)
        value = np.asarray(library(jnp.asarray(adstock)))
        slope = np.asarray(jax.vmap(jax.grad(lambda a: library(a)))(jnp.asarray(adstock)))
    np.testing.assert_allclose(CURVES[curve].value(adstock, lam), value, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(CURVES[curve].slope(adstock, lam), slope, rtol=1e-10, atol=1e-15)
    assert float(CURVES[curve].value(np.array(half), lam)) == pytest.approx(0.5, rel=1e-12, abs=0.0)
    assert CURVES[curve].concave == (library.inflection() == 0.0)


def test_a_curve_moves_the_media_and_nothing_the_business_sets() -> None:
    """A seed's spend, baseline and mechanisms do not read the curve: two curves on one seed
    differ in the media alone, and each world's media is the library's channel on its spend."""
    his = EndogenousMediaMix().simulate(7)
    hill = EndogenousMediaMix(curve="hill-2").simulate(7)
    for name in ("spend", "premedia", "budget", "bidding", "observed_price", "observed_promotion"):
        np.testing.assert_array_equal(getattr(hill, name), getattr(his, name))
    assert hill.curve == "hill-2"
    assert not np.allclose(hill.media, his.media)
    np.testing.assert_array_equal(hill.sales, hill.premedia + hill.media.sum(axis=1))
    with jax.enable_x64(True):
        for c, (alpha, lam, beta) in enumerate(
            zip(hill.retention, hill.saturation, hill.effect, strict=True)
        ):
            kernel = GeometricAdstock(alpha, length=6, normalized=True)
            channel = Channel(kernel, _library("hill-2", lam), beta)
            np.testing.assert_allclose(
                np.asarray(channel(hill.spend[:, c])), hill.media[:, c], rtol=1e-12
            )


def test_a_geo_test_reads_the_world_s_curve() -> None:
    world = dataclasses.replace(EndogenousMediaMix(), curve="logistic-4").simulate(2)
    experiment = world.geo_test("meta", (20, 55), seed=1)
    alpha, lam, beta = world.retention[1], world.saturation[1], world.effect[1]
    with jax.enable_x64(True):
        channel = Channel(
            GeometricAdstock(alpha, length=6, normalized=True), _library("logistic-4", lam), beta
        )
        dark = np.asarray(channel(experiment.spend_treated))
    np.testing.assert_allclose(experiment.true_gap, dark - world.media[:, 1], atol=1e-9)
    assert np.any(experiment.true_gap < 0.0)


def test_a_curve_that_is_not_one_of_the_families_is_refused() -> None:
    with pytest.raises(ValueError, match="not one of"):
        EndogenousMediaMix(curve="gompertz")


def _go_dark(world, channel: str, starts: tuple[int, ...], seed: int):
    """His go-dark as the procedure ran before a test took a multiplier: the test weeks zeroed,
    the channel's effect recomputed, and noise of opposite signs from the seed."""
    column = world.channels.index(channel)
    dark = world.spend[:, column].copy()
    for start in starts:
        dark[start - 1 : start - 1 + 4] = 0.0
    alpha, lam, beta = world.retention[column], world.saturation[column], world.effect[column]
    gap = _media(dark, alpha, lam, beta, world.kernel_length, world.curve) - world.media[:, column]
    noise = np.random.default_rng(seed).normal(0.0, 0.01 * np.mean(world.sales), world.week.size)
    return dark, gap, world.sales + noise, world.sales + gap - noise


@pytest.mark.parametrize("curve", ["tanh", "hill-2"])
def test_a_multiplier_of_nought_is_the_go_dark_bit_for_bit(curve) -> None:
    """The default, and 0 asked for, give the committed records' tests exactly, overlapping
    tests' weeks dark once."""
    world = EndogenousMediaMix(curve=curve).simulate(5)
    for channel, starts in (("pla", (20, 55, 100, 140)), ("tv", (30, 32))):
        expected = _go_dark(world, channel, starts, seed=9)
        for experiment in (
            world.geo_test(channel, starts, seed=9),
            world.geo_test(channel, starts, seed=9, multiplier=0.0),
        ):
            for got, want in zip(
                (
                    experiment.spend_treated,
                    experiment.true_gap,
                    experiment.sales_control,
                    experiment.sales_treated,
                ),
                expected,
                strict=True,
            ):
                np.testing.assert_array_equal(got, want)


def test_a_multiplier_of_one_moves_nothing() -> None:
    world = EndogenousMediaMix().simulate(6)
    experiment = world.geo_test("meta", (20, 55, 100, 140), seed=2, multiplier=1.0)
    np.testing.assert_array_equal(experiment.spend_treated, experiment.spend_control)
    assert np.all(experiment.true_gap == 0.0)
    np.testing.assert_allclose(
        experiment.sales_treated + experiment.sales_control, 2.0 * world.sales
    )


@pytest.mark.parametrize("curve", sorted(name for name, c in CURVES.items() if c.concave))
def test_on_a_concave_curve_the_gap_rises_with_the_multiplier_and_bends_down(curve) -> None:
    """The treated adstock is affine in the multiplier and the curve rises, so every week's gap
    rises with it, strictly in the test weeks; a concave curve makes it concave in it too."""
    world = EndogenousMediaMix(curve=curve).simulate(7)
    multipliers = np.linspace(0.0, 2.5, 11)
    gaps = np.array([world.geo_test("pla", (20, 100), multiplier=m).true_gap for m in multipliers])
    rises = np.diff(gaps, axis=0)
    assert np.all(rises >= 0.0)
    tested = np.r_[19:23, 99:103]
    assert np.all(rises[:, tested] > 0.0)
    scale = float(np.max(np.abs(gaps)))
    assert np.all(np.diff(gaps, n=2, axis=0) <= 1e-12 * scale)


def test_any_multiplier_scales_the_test_weeks_alone_and_the_library_channel_reads_the_gap() -> None:
    """Over random multipliers: the treated spend is the multiple in the test weeks and the
    control's elsewhere; the gap is nought before the first test week and once a test's carryover
    is spent, signed as the multiplier less one in the test weeks, and the library's channel,
    computed apart, reads it; the two universes' sales sum to twice the world's and the gap."""
    world = EndogenousMediaMix().simulate(8)
    starts, length = (20, 55, 100, 140), world.kernel_length
    tested = np.zeros(world.week.size, dtype=bool)
    for start in starts:
        tested[start - 1 : start - 1 + 4] = True
    reached = np.zeros(world.week.size, dtype=bool)  # weeks a test's spend can reach
    for start in starts:
        reached[start - 1 : start - 1 + 4 + length - 1] = True
    alpha, lam, beta = world.retention[0], world.saturation[0], world.effect[0]
    rng = np.random.default_rng(0)
    with jax.enable_x64(True):
        channel = Channel(
            GeometricAdstock(alpha, length=length, normalized=True), Tanh(2 / lam), beta
        )
        control = np.asarray(channel(world.spend[:, 0]))
        for m in rng.uniform(0.0, 3.0, 60):
            experiment = world.geo_test("pla", starts, seed=3, multiplier=float(m))
            spend = world.spend[:, 0]
            np.testing.assert_array_equal(experiment.spend_treated[tested], m * spend[tested])
            np.testing.assert_array_equal(experiment.spend_treated[~tested], spend[~tested])
            gap = experiment.true_gap
            assert np.all(gap[~reached] == 0.0)
            assert np.all(np.sign(gap[tested]) == np.sign(m - 1.0))
            treated = np.asarray(channel(experiment.spend_treated))
            np.testing.assert_allclose(gap, treated - control, rtol=0.0, atol=1e-9 * beta)
            np.testing.assert_allclose(
                experiment.sales_treated + experiment.sales_control, 2.0 * world.sales + gap
            )
            overlapping = world.geo_test("pla", (30, 32), multiplier=float(m))
            np.testing.assert_array_equal(overlapping.spend_treated[29:35], m * spend[29:35])


@pytest.mark.parametrize("multiplier", [-0.5, math.inf, math.nan])
def test_a_multiplier_below_nought_or_not_finite_is_refused(multiplier) -> None:
    world = EndogenousMediaMix().simulate(1)
    with pytest.raises(ValueError, match="multiplier"):
        world.geo_test("pla", (20,), multiplier=multiplier)

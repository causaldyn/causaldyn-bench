"""Track M v2's world: weekly retail sales whose media spend is set by the business, not drawn.

Written from the equations of Heusch (2026), "A Synthetic Benchmark Dataset with Endogenous
Marketing Spend for Validating Marketing Mix Models", arXiv:2608.21130 (CC-BY-4.0), and credited
under that licence. The author's generator (github.com/niklas-heusch/mmm-materials) carries no
licence, so none of its code was read or copied: this is a clean-room implementation of the
paper's Section III and its geo-test procedure of Section V.

Sales are a baseline, a promotional uplift and three media effects, ``y_t = b_t + p_t + sum_c
m_{c,t}`` (his Eq. 1):

* the baseline multiplies a seasonality, a persistent product quality and a competitive price
  position, and adds a market sentiment nobody observes, ``b_t = B (1 + s_t)(1 + q_t)
  pi_t^-0.9 + B u_t + eps_t`` (Eq. 2), floored at ``B / 2``;
* five promotions a year (his Table I) lift it, ``p_t = b_t (mu_t - 1)`` with ``mu_t =
  max(1 - d_t, 0.5)^{e_t}`` clipped to ``[0.8, 3]`` (Eq. 3);
* media spend is planned on the log scale from a base level, a quarterly budget that follows the
  last two quarters' sales before media, a seasonal tilt, spend raised ahead of promotions, and
  television's scheduled bursts (Eq. 4-5); paid shopping (PLA) is then scaled by a bidding rule
  that chases last week's sales before media over their ten-week average (Eq. 6);
* each channel's effect is a normalised six-week geometric adstock through ``tanh(lambda x / 2)``,
  ``m_{c,t} = beta_c g(xbar_{c,t}; lambda_c)`` (Eq. 7-8), which is
  ``chc.response.Channel(GeometricAdstock(alpha, length=6, normalized=True), Tanh(2 / lambda),
  beta)``; a test holds the two equal, and this module computes it on its own.

What an analyst observes is the paper's measurement layer: sales, spend, a promotion indicator
with 70 % recall and 15 % false positives, and a price read with autocorrelated error. Sentiment
has no observed counterpart.

:meth:`MediaMixWorld.geo_test` is his Section V: two universes at the market's scale, the control
as simulated and the treated one with the tested channel dark in the test weeks, its effect
recomputed over the whole history, and noise of opposite signs, one percent of mean weekly sales
each, added to the two. Beyond the paper, its ``multiplier`` scales the tested channel's spend in
the test weeks instead of zeroing it: a partial cut below 1, a heavy-up above; 0, the default, is
his go-dark.

Where the paper leaves a detail open, the choice is named where it is made (``CHOICE``): the
television burst weeks, the processes' starting values, the quarters' boundaries, the weeks
before the bidding rule has ten weeks of history, and the price the measurement layer reads.

Beyond the paper, ``curve`` puts another family in his curve's place (:data:`CURVES`), each at half
its ceiling where his is, at ``ln 3 / lambda``, so a world can hold a curve its analyst's family is
not. His is the default, and computed as he writes it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

Series = NDArray[np.float64]
# a one-axis Series, typed so: iterated, its entries are scalars, where a Series leaves them open
Vector = np.ndarray[tuple[int], np.dtype[np.float64]]

# his Table I: first week of the year, last week, discount, elasticity multiplier
PROMOTIONS: tuple[tuple[int, int, float, float], ...] = (
    (2, 4, 0.40, 1.5),  # winter clearance
    (16, 16, 0.30, 1.3),  # spring mid-season
    (28, 29, 0.40, 1.5),  # summer clearance
    (40, 40, 0.30, 1.3),  # autumn mid-season
    (46, 47, 0.40, 1.8),  # cyber week
)
# CHOICE: the paper gives a five-week pre-Christmas burst at x15 and a three-week early-summer one
# at x8, not their weeks. Weeks 48-52 give his television return of 1.56 (the reference-instance
# test holds it within a percent); a week earlier reads it higher, since the last burst's carryover
# then stays inside the history. No number he reports pins the summer weeks, which are where his
# Fig. 3 shows them
TV_BURSTS: tuple[tuple[int, int, float], ...] = ((48, 52, 15.0), (23, 25, 8.0))
YEAR = 52
QUARTER = 13
ELASTICITY = -0.9

LN2, LN3 = math.log(2.0), math.log(3.0)
STEEPNESS = 4.0  # the logistic curve's
# the logistic's midpoint in half-saturations: there sigma(s (z - 1)) is (1 + sigma(-s)) / 2
_MIDPOINT = 1.0 / (1.0 + float(logit((1.0 + expit(-STEEPNESS)) / 2.0)) / STEEPNESS)


@dataclass(frozen=True)
class Curve:
    """A channel's curve on its adstock ``a``, given his ``lambda``: its value, rising from 0 to 1,
    and its slope in ``a``."""

    value: Callable[[Series, float], Series]
    slope: Callable[[Series, float], Series]
    concave: bool  # from zero adstock; otherwise S-shaped


def _hill(a: Series, lam: float) -> Series:
    z = a * lam / LN3
    return z**2 / (1.0 + z**2)


def _hill_slope(a: Series, lam: float) -> Series:
    z = a * lam / LN3
    return 2.0 * z / (1.0 + z**2) ** 2 * lam / LN3


def _weibull(a: Series, lam: float) -> Series:
    return -np.expm1(-LN2 * (a * lam / LN3) ** 2)


def _weibull_slope(a: Series, lam: float) -> Series:
    z = a * lam / LN3
    return 2.0 * LN2 * z * np.exp(-LN2 * z**2) * lam / LN3


def _logistic(a: Series, lam: float) -> Series:
    z = a * lam / LN3 / _MIDPOINT
    return (expit(STEEPNESS * (z - 1.0)) - expit(-STEEPNESS)) / expit(STEEPNESS)


def _logistic_slope(a: Series, lam: float) -> Series:
    z = a * lam / LN3 / _MIDPOINT
    rise = expit(STEEPNESS * (z - 1.0))
    return STEEPNESS * rise * (1.0 - rise) / expit(STEEPNESS) * lam / LN3 / _MIDPOINT


# his curve, tanh(lambda a / 2), and five others, each written from its definition and at half its
# ceiling where his is, at ln 3 / lambda: the exponential and Michaelis-Menten, concave from zero as
# his is, and Hill of slope 2, Weibull of shape 2 and the logistic of steepness 4, S-shaped
CURVES: dict[str, Curve] = {
    "tanh": Curve(
        lambda a, lam: np.tanh(lam * a / 2.0),
        lambda a, lam: lam / 2.0 / np.cosh(lam * a / 2.0) ** 2,
        concave=True,
    ),
    "exponential": Curve(
        lambda a, lam: -np.expm1(-LN2 * a * lam / LN3),
        lambda a, lam: LN2 * lam / LN3 * np.exp(-LN2 * a * lam / LN3),
        concave=True,
    ),
    "michaelis-menten": Curve(
        lambda a, lam: a / (LN3 / lam + a),
        lambda a, lam: LN3 / lam / (LN3 / lam + a) ** 2,
        concave=True,
    ),
    "hill-2": Curve(_hill, _hill_slope, concave=False),
    "weibull-2": Curve(_weibull, _weibull_slope, concave=False),
    "logistic-4": Curve(_logistic, _logistic_slope, concave=False),
}


@dataclass(frozen=True)
class MediaMixWorld:
    """One simulated history, with its ground truth beside what an analyst would observe.

    Money is in thousands of euros. Every series runs over the weeks; ``spend`` and ``media`` are
    ``(weeks, channels)``.
    """

    channels: tuple[str, ...]
    week: NDArray[np.int64]  # 1, 2, ..., weeks
    sales: Series
    baseline: Series
    promotion: Series  # the promotional uplift p_t
    premedia: Series  # b_t + p_t, what the budget and the bidding rule read
    media: Series  # (weeks, channels) each channel's true effect
    spend: Series  # (weeks, channels)
    seasonality: Series
    quality: Series
    price_position: Series
    sentiment: Series  # unobserved by construction
    discount: Series
    budget: Series  # the quarterly multiplier Q_t
    bidding: Series  # PLA's multiplier rho_t
    observed_promotion: NDArray[np.bool_]
    observed_price: Series
    retention: tuple[float, ...]
    saturation: tuple[float, ...]
    effect: tuple[float, ...]
    kernel_length: int
    curve: str = "tanh"  # every channel's, a key of CURVES

    @property
    def promotion_weeks(self) -> NDArray[np.bool_]:
        return self.discount > 0.0

    def roas(self) -> Series:
        """Each channel's true return per euro over the history: its effect over its spend."""
        return self.media.sum(axis=0) / self.spend.sum(axis=0)

    def geo_test(
        self,
        channel: str,
        starts: tuple[int, ...],
        *,
        test: int = 4,
        noise_share: float = 0.01,
        seed: int = 0,
        multiplier: float = 0.0,
    ) -> GeoExperiment:
        """His Section V: the tested channel's spend times ``multiplier`` in each test's weeks in
        the treated universe, dark at the default 0, his go-dark; a partial cut below 1 and a
        heavy-up above it.

        ``starts`` are the tests' first weeks, numbered from 1 as :attr:`week` is. Both universes
        stay at the market's scale; each gets noise ``N(0, (noise_share * mean sales)^2)``, of
        opposite signs, so the sum of the two is unchanged.
        """
        if not 0.0 <= multiplier < math.inf:
            raise ValueError(f"the multiplier {multiplier} is not a finite number of 0 or more")
        column = self.channels.index(channel)
        weeks = np.zeros(self.week.size, dtype=bool)
        for start in starts:
            if not 1 <= start <= self.week.size - test + 1:
                raise ValueError(f"a test from week {start} runs past the history")
            weeks[start - 1 : start - 1 + test] = True
        treated = self.spend[:, column].copy()
        # once a week however many tests hold it; at 0 every product is +0.0, the go-dark's bits
        treated[weeks] *= multiplier
        gap = self._effect_of(column, treated) - self.media[:, column]
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_share * float(np.mean(self.sales)), self.sales.size)
        return GeoExperiment(
            channel=channel,
            starts=starts,
            test=test,
            spend_control=self.spend[:, column].copy(),
            spend_treated=treated,
            sales_control=self.sales + noise,
            sales_treated=self.sales + gap - noise,
            true_gap=gap,
        )

    def _effect_of(self, column: int, spend: Series) -> Series:
        """The effect of channel ``column`` in each week had it spent ``spend``, as :attr:`media`
        holds its effect: a world whose effect moves over the weeks moves this with it."""
        return _media(
            spend,
            self.retention[column],
            self.saturation[column],
            self.effect[column],
            self.kernel_length,
            self.curve,
        )


@dataclass(frozen=True)
class GeoExperiment:
    """Two universes that differ only in one channel's spend in the test weeks."""

    channel: str
    starts: tuple[int, ...]
    test: int
    spend_control: Series
    spend_treated: Series
    sales_control: Series
    sales_treated: Series
    true_gap: Series  # treated less control, before the noise

    def readout(self, start: int, *, pre: int = 4, cooldown: int = 8) -> slice:
        """The weeks a test is read over, as a slice of the history: ``pre`` weeks before its
        first, its own, and ``cooldown`` after (his 2026a: 4, 4 and 8)."""
        first = start - 1 - pre
        if first < 0 or start - 1 + self.test + cooldown > self.spend_control.size:
            raise ValueError(f"the readout of the test from week {start} leaves the history")
        return slice(first, start - 1 + self.test + cooldown)


@dataclass(frozen=True)
class EndogenousMediaMix:
    """His generator's parameters; :meth:`simulate` draws one history.

    The defaults are his reference instance's (Table II), with each coordination mechanism a
    switch, as in his generator.
    """

    weeks: int = 156
    base: float = 2000.0
    channels: tuple[str, ...] = ("pla", "meta", "tv")
    retention: tuple[float, ...] = (0.2, 0.4, 0.7)  # alpha
    saturation: tuple[float, ...] = (0.008, 0.010, 0.006)  # lambda, per thousand euros
    effect: tuple[float, ...] = (1100.0, 600.0, 560.0)  # beta, thousands of euros
    planned_spend: tuple[float, ...] = (60.0, 40.0, 10.0)  # base weekly levels
    anticipation: tuple[float, ...] = (0.4, 0.4, 0.0)  # gamma_c on promotions in the next 3 weeks
    bidding_channel: str = "pla"
    kernel_length: int = 6
    budget_feedback: bool = True
    anticipatory: bool = True
    bursts: bool = True
    bidding: bool = True
    curve: str = "tanh"  # every channel's, a key of CURVES; his is tanh

    def __post_init__(self) -> None:
        sizes = {
            len(self.channels),
            len(self.retention),
            len(self.saturation),
            len(self.effect),
            len(self.planned_spend),
            len(self.anticipation),
        }
        if len(sizes) != 1:
            raise ValueError("every channel parameter needs one value per channel")
        if self.bidding_channel not in self.channels:
            raise ValueError(f"the bidding channel {self.bidding_channel!r} is not a channel")
        if self.weeks < 2 * QUARTER + 1:
            raise ValueError(f"{self.weeks} weeks leave the quarterly budget nothing to review")
        if self.curve not in CURVES:
            raise ValueError(f"the curve {self.curve!r} is not one of {sorted(CURVES)}")

    def simulate(self, seed: int = 0) -> MediaMixWorld:
        rng = np.random.default_rng(seed)
        weeks = self.weeks
        week = np.arange(1, weeks + 1)
        angle = 2.0 * math.pi * week / YEAR
        seasonality = 0.15 * np.sin(angle) + 0.08 * np.sin(2 * angle) + 0.05 * np.sin(3 * angle)
        # CHOICE: quality starts at 0, the price position at its mean 1, sentiment at a draw from
        # N(0, 0.05^2), each one step before week 1
        quality = _ar1(rng, weeks, 0.95, 0.0, 0.02, start=0.0)
        price_position = _ar1(rng, weeks, 0.9, 0.1, 0.03, start=1.0)
        sentiment = _ar1(rng, weeks, 0.7, 0.0, 0.03, start=float(rng.normal(0.0, 0.05)))
        noise = rng.normal(0.0, 0.02 * self.base, weeks)
        baseline = (
            self.base * (1 + seasonality) * (1 + quality) * price_position**ELASTICITY
            + self.base * sentiment
            + noise
        )
        baseline = np.maximum(baseline, 0.5 * self.base)
        discount, multiplier = _calendar(week)
        uplift = np.where(
            discount > 0.0,
            np.clip(np.maximum(1.0 - discount, 0.5) ** (ELASTICITY * multiplier), 0.8, 3.0),
            1.0,
        )
        promotion = baseline * (uplift - 1.0)
        premedia = baseline + promotion

        ahead = np.sum([_calendar(week + lead)[0] > 0.0 for lead in (1, 2, 3)], axis=0, dtype=float)
        budget = _budget(premedia) if self.budget_feedback else np.ones(weeks)
        channels = len(self.channels)
        log_plan = (
            np.log(np.array(self.planned_spend))[None, :]
            + np.log(budget)[:, None]
            + 0.15 * seasonality[:, None]
            + rng.normal(0.0, 0.1, (weeks, channels))
        )
        if self.anticipatory:
            log_plan += ahead[:, None] * np.array(self.anticipation)[None, :]
        if self.bursts and "tv" in self.channels:
            log_plan[:, self.channels.index("tv")] += np.log(_bursts(week))
        spend = np.exp(log_plan)
        bidding = _bidding(premedia) if self.bidding else np.ones(weeks)
        spend[:, self.channels.index(self.bidding_channel)] *= bidding

        media = np.column_stack(
            [
                _media(spend[:, c], alpha, lam, beta, self.kernel_length, self.curve)
                for c, (alpha, lam, beta) in enumerate(
                    zip(self.retention, self.saturation, self.effect, strict=True)
                )
            ]
        )
        sales = premedia + media.sum(axis=1)

        promoted = discount > 0.0
        caught = rng.random(weeks)
        observed_promotion = np.where(promoted, caught < 0.7, caught < 0.15)
        # CHOICE: the "true promotional price" is the price position times one less the discount;
        # it gives the paper's correlation of 0.89 between the price and its reading
        error = _ar1(rng, weeks, 0.2, 0.0, 0.08, start=0.0)
        observed_price = np.clip(price_position * (1.0 - discount) + error, 0.5, 1.8)
        return MediaMixWorld(
            channels=self.channels,
            week=week,
            sales=sales,
            baseline=baseline,
            promotion=promotion,
            premedia=premedia,
            media=media,
            spend=spend,
            seasonality=seasonality,
            quality=quality,
            price_position=price_position,
            sentiment=sentiment,
            discount=discount,
            budget=budget,
            bidding=bidding,
            observed_promotion=observed_promotion,
            observed_price=observed_price,
            retention=self.retention,
            saturation=self.saturation,
            effect=self.effect,
            kernel_length=self.kernel_length,
            curve=self.curve,
        )


def _ar1(
    rng: np.random.Generator,
    weeks: int,
    persistence: float,
    drift: float,
    sd: float,
    *,
    start: float,
) -> Series:
    shocks = rng.normal(0.0, sd, weeks)
    path = np.empty(weeks)
    level = start
    for t in range(weeks):
        level = persistence * level + drift + shocks[t]
        path[t] = level
    return path


def _calendar(week: NDArray[np.int64]) -> tuple[Series, Series]:
    """Each week's discount and elasticity multiplier, from the week of its year."""
    of_year = (week - 1) % YEAR + 1
    discount = np.zeros(week.shape)
    multiplier = np.ones(week.shape)
    for first, last, cut, scale in PROMOTIONS:
        inside = (of_year >= first) & (of_year <= last)
        discount[inside] = cut
        multiplier[inside] = scale
    return discount, multiplier


def _bursts(week: NDArray[np.int64]) -> Series:
    of_year = (week - 1) % YEAR + 1
    factor = np.ones(week.shape)
    for first, last, scale in TV_BURSTS:
        factor[(of_year >= first) & (of_year <= last)] = scale
    return factor


def _budget(premedia: Series) -> Series:
    """Eq. 5: the multiplier held over each quarter, from the second quarter back and the first.

    CHOICE: quarters are weeks 1-13, 14-26, ...; the first review is at the start of the third,
    the first quarter with two full quarters of history before it.
    """
    weeks = premedia.size
    budget = np.ones(weeks)
    for quarter in range(2, math.ceil(weeks / QUARTER)):
        last = premedia[(quarter - 1) * QUARTER : quarter * QUARTER].sum()
        before = premedia[(quarter - 2) * QUARTER : (quarter - 1) * QUARTER].sum()
        budget[quarter * QUARTER : (quarter + 1) * QUARTER] = np.clip(
            1.0 + 0.3 * (last / before - 1.0), 0.8, 1.4
        )
    return budget


def _bidding(premedia: Series) -> Series:
    """Eq. 6: last week's sales before media over the ten weeks before it.

    CHOICE: the rule starts once it has those eleven weeks, at week 12; before, it is 1.
    """
    rho = np.ones(premedia.size)
    for t in range(11, premedia.size):
        ratio = premedia[t - 1] / premedia[t - 11 : t - 1].mean()
        rho[t] = np.clip(1.0 + 0.8 * (ratio - 1.0), 0.5, 3.0)
    return rho


def _media(
    spend: Series, alpha: float, lam: float, beta: float, length: int, curve: str = "tanh"
) -> Series:
    """Eq. 7-8 on one channel: the normalised geometric adstock through ``tanh(lam x / 2)``, or
    through ``curve``, with nothing spent before the history."""
    weights = alpha ** np.arange(length)
    weights = weights / weights.sum()
    adstock = np.convolve(spend, weights)[: spend.size]
    if curve == "tanh":  # Eq. 8 as he writes it, which the committed results were drawn with
        return beta * (1.0 - np.exp(-lam * adstock)) / (1.0 + np.exp(-lam * adstock))
    return beta * CURVES[curve].value(adstock, lam)

"""Family 17: twin worlds one history cannot tell apart, a line whose slope drifts and a fixed
S-shaped curve, whose best plans differ.

Two channels, television and search, neither carried over. Television's spend rises from 30 to 100
a week over the first :data:`RISE` weeks, ``ramp(t)``, and holds there, each week's spend that level
times a log-normal of log-scale 0.05 and mean 1; search's is a log-normal of log-scale 0.5 and mean
50 a week. Sales are a base of 1000, television's response, search's ``252 tanh(x / 50)`` and a
normal noise of sd 40, the same draws in both twins:

* ``curve``: television's response is fixed on a Hill of slope 3, ``350 x^3 / (x^3 + 60^3)``;
* ``line``: it is a line in its spend whose slope drifts over the weeks, ``drift(t) x``, the
  curve's average return at the week's level, ``drift(t) = curve(ramp(t)) / ramp(t)``: 1.30 in the
  first week, 3.09 at most, where the ramp passes 76, and 2.88 once it holds at 100.

Along the ramp the twins return the same, and they differ only by how far each week's spend is off
its level, which the spread of 0.05 keeps small: the conflation of a time-varying effect with a
curve that Dew, Padilla and Shchetkina (arXiv 2408.07678) measure. A go-dark test reads
``curve(x)`` in one twin and ``drift(t) x`` in the other at a spend ``x`` near the week's level, so
the ladder's tests do not tell them apart either; a test that moves the spend both ways about the
plan does.

The twins run over the scorecard's 156 weeks, the ramp over three quarters of the history
(CHOICE); :func:`history` and :func:`sales` take both lengths, and draw the spreads of television
and of search and then the noise, in that order. The quarter is the twins' own (CHOICE: a fixed
decision rather than the scorecard's box rule, under which the twins' best plans come closer): 150
a week over 13 weeks, television within 50 to 140 and search within 10 to 100, from the status quo
of 100 and 50, the levels the history's spend is drawn about at its end. The line's slope over the
quarter is the one the history left, 2.88, the ramp being over. Television's best spend a week is
then 110.81 in the line's twin and 94.69 in the curve's, and each twin's best plan leaves the
other's 0.095 (in the curve's twin) and 0.061 (in the line's) of a euro of the budget.

A seed draws the same history in both environments, its spreads and noise from the family's path
stream and the quarter's noise from its continuation stream (:mod:`.seeds`): a pair of twins is one
seed read in both, so both environments read the same block of seeds in the same order. The truth
reads each twin's own curve; the curve's best plan is exact to the oracle's gap
(:func:`causaldyn_bench.scorecard.truth.oracle`), the Hill's bend from Maxima
(``scripts/twin_hill_bend.mac``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from causaldyn_bench.budget_regret import lift_rows
from causaldyn_bench.endogenous_mmm import CURVES, Curve, Market, Series
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard import ladder
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.returns import Returns
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, window
from causaldyn_bench.scorecard.truth import Cell, oracle

FAMILY = 17
ENVIRONMENTS = ("curve", "line")
CHANNELS = ("tv", "search")
WEEKS = 156
RISE = 117  # the weeks television's spend rises over; it holds at its top for the rest
LEVELS = (30.0, 100.0)  # television's spend a week where the ramp starts and where it holds
SPREAD = 0.05  # the log-scale of television's spend about its level
PAID, PAID_SPREAD = 50.0, 0.5  # search's mean spend a week and its log-scale
BASE, NOISE = 1000.0, 40.0
HEIGHT, HALF, SLOPE = 350.0, 60.0, 3.0  # television's curve: its ceiling, half-saturation, slope
SEARCH_HEIGHT, SEARCH_SCALE = 252.0, 50.0
BUDGET = 150.0  # a week of the quarter
LOWER = (50.0, 10.0)  # each channel's box over the quarter, a week
UPPER = (140.0, 100.0)
STATUS_QUO = (100.0, 50.0)
# the Hill of slope 3's greatest concavity in its half-saturations, the largest -f'' of
# f(z) = z^3 / (1 + z^3): at z^3 = (8 + 3 sqrt 6) / 10 it is 0.879475680097885 (Maxima,
# scripts/twin_hill_bend.mac)
_U = (8.0 + 3.0 * math.sqrt(6.0)) / 10.0
BEND = 6.0 * _U ** (1.0 / 3.0) * (2.0 * _U - 1.0) / (1.0 + _U) ** 3


def _hill(a: Series, half: float) -> Series:
    a = np.asarray(a, dtype=float)
    return a**SLOPE / (a**SLOPE + half**SLOPE)


def _hill_slope(a: Series, half: float) -> Series:
    a = np.asarray(a, dtype=float)
    return SLOPE * a ** (SLOPE - 1.0) * half**SLOPE / (a**SLOPE + half**SLOPE) ** 2


HILL = Curve(_hill, _hill_slope, concave=False, bend=lambda half: BEND / half**2)
"""Television's curve in the curve's twin, read at its half-saturation."""
LINE = Curve(
    lambda a, scale: np.asarray(a, dtype=float),
    lambda a, scale: np.ones_like(np.asarray(a, dtype=float)),
    concave=True,
    bend=lambda scale: 0.0,
)
"""Television's response per unit of its slope in the line's twin: the spend itself."""
SEARCH = (CURVES["tanh"], 2.0 / SEARCH_SCALE)  # ``tanh(lambda x / 2)`` is ``tanh(x / 50)``


def ramp(t: ArrayLike, rise: int = RISE) -> Series:
    """Television's level a week at the weeks ``t``, counted from 0."""
    low, high = LEVELS
    return low + (high - low) * np.minimum(np.asarray(t, dtype=float) / (rise - 1), 1.0)


def curve(x: ArrayLike) -> Series:
    """Television's response a week in the curve's twin."""
    return HEIGHT * _hill(np.asarray(x, dtype=float), HALF)


def drift(t: ArrayLike, rise: int = RISE) -> Series:
    """The line's slope at the weeks ``t``: the curve's average return at the week's level."""
    level = ramp(t, rise)
    return curve(level) / level


def search(x: ArrayLike) -> Series:
    """Search's response a week in both twins."""
    return SEARCH_HEIGHT * np.tanh(np.asarray(x, dtype=float) / SEARCH_SCALE)


def history(
    draw: np.random.Generator, weeks: int = WEEKS, rise: int = RISE
) -> tuple[Series, Series, Series]:
    """Television's and search's spend over ``weeks`` and the sales' noise, drawn from ``draw`` in
    that order."""
    level = ramp(np.arange(weeks), rise)
    tv = level * np.exp(SPREAD * draw.standard_normal(weeks) - SPREAD**2 / 2.0)
    paid = PAID * np.exp(PAID_SPREAD * draw.standard_normal(weeks) - PAID_SPREAD**2 / 2.0)
    return tv, paid, NOISE * draw.standard_normal(weeks)


def slope(environment: str, weeks: int, rise: int = RISE) -> Series:
    """Television's coefficient in each week of ``environment``: the curve's height, or the line's
    drifting slope, on the response :func:`parts` names."""
    if environment == "curve":
        return np.full(weeks, HEIGHT)
    if environment == "line":
        return drift(np.arange(weeks), rise)
    raise ValueError(f"the twins are {ENVIRONMENTS}, not {environment!r}")


def parts(environment: str) -> tuple[Curve, float]:
    """Television's response in ``environment`` per unit of its coefficient, and its scale."""
    return (HILL, HALF) if environment == "curve" else (LINE, 1.0)


def sales(environment: str, tv: Series, paid: Series, noise: Series, rise: int = RISE) -> Series:
    """The twin's weekly sales over the spend and the noise :func:`history` drew."""
    response, scale = parts(environment)
    effect = slope(environment, tv.size, rise)
    return BASE + effect * response.value(tv, scale) + search(paid) + noise


@dataclass(frozen=True, eq=False)
class TwinWorld(Market):
    """One twin over the history: its sales, each channel's spend and media, ``(weeks,
    channels)``, television's coefficient in each week, and the quarter's sales at the status
    quo."""

    environment: str
    seed: int
    channels: tuple[str, ...]
    week: NDArray[np.int64]  # 1, 2, ..., 156
    sales: Series
    media: Series
    spend: Series
    effect: Series  # television's coefficient a week
    target: Series

    def _effect_of(self, column: int, spend: Series) -> Series:
        if column == 1:
            return search(spend)
        response, scale = parts(self.environment)
        return self.effect * response.value(np.asarray(spend, dtype=float), scale)


def twin(environment: str, seed: int) -> TwinWorld:
    """The twin of ``environment`` drawn from ``seed``."""
    if environment not in ENVIRONMENTS:
        raise ValueError(f"the twins are {ENVIRONMENTS}, not {environment!r}")
    tv, paid, noise = history(rng(FAMILY, Role.PATH, seed))
    effect = slope(environment, WEEKS)
    response, scale = parts(environment)
    media = np.column_stack([effect * response.value(tv, scale), search(paid)])
    quarter = rng(FAMILY, Role.CONTINUATION, seed).standard_normal(PLANNED)
    at = effect[-1] * response.value(np.array(STATUS_QUO[0]), scale) + search(STATUS_QUO[1])
    return TwinWorld(
        environment=environment,
        seed=seed,
        channels=CHANNELS,
        week=np.arange(1, WEEKS + 1),
        sales=BASE + media.sum(axis=1) + noise,
        media=media,
        spend=np.column_stack([tv, paid]),
        effect=effect,
        target=BASE + at + NOISE * quarter,
    )


def quarter(world: TwinWorld) -> Quarter:
    """The twins' quarter: :data:`BUDGET` a week within each channel's box, from the status quo."""
    return Quarter(
        budget=PLANNED * BUDGET,
        lower=np.array(LOWER),
        upper=np.array(UPPER),
        status_quo=np.array(STATUS_QUO),
        history=world.spend.copy(),
    )


def cells(world: TwinWorld) -> tuple[Cell, ...]:
    """Each channel as a cell over the quarter, nothing carried in or out: television on the twin's
    own response at the coefficient the history left, search on its curve."""
    nothing, unit = np.zeros(PLANNED), np.ones(PLANNED)
    response, scale = parts(world.environment)
    paid, lam = SEARCH
    return (
        Cell(nothing, unit, response, scale, np.full(PLANNED, world.effect[-1])),
        Cell(nothing, unit, paid, lam, np.full(PLANNED, SEARCH_HEIGHT)),
    )


def true_returns(world: TwinWorld) -> Returns:
    """Each channel's ROI and mROI on last year: with nothing carried over, what each week's spend
    returns, and its slope times the spend, over what the window spent."""
    first, last = window(world)
    inside = slice(first - 1, last)
    response, scale = parts(world.environment)
    paid, lam = SEARCH
    roi, marginal = [], []
    for c, (curve_, at, effect) in enumerate(
        ((response, scale, world.effect), (paid, lam, np.full(WEEKS, SEARCH_HEIGHT)))
    ):
        spend = world.spend[inside, c]
        spent = float(spend.sum())
        weights = effect[inside]
        roi.append(float(np.sum(weights * curve_.value(spend, at))) / spent)
        marginal.append(float(np.sum(weights * curve_.slope(spend, at) * spend)) / spent)
    return Returns((first, last), np.array(roi), np.array(marginal))


class Twins:
    """Family 17."""

    name = "twins"
    stream = stream(FAMILY)
    labels = (
        "Two twins share one history: in one television's response is a line whose slope drifts "
        "with its spend's level, in the other a fixed S-shaped curve; a fit to the history, or a "
        "go-dark test, cannot tell them apart, and their best plans differ.",
        "Neither channel carries over; every arm is told the kernel's length, 1.",
        "The quarter is the twins' own: 150 a week, television within 50 to 140 and search within "
        "10 to 100, from a status quo of 100 and 50.",
        "An arm reads the sales and the spend, and no control.",
        TRACK_M2.labels[2],
    )
    pilots = MappingProxyType({name: pilot(FAMILY, 0) for name in ENVIRONMENTS})
    scored = MappingProxyType({name: scored(FAMILY, 0) for name in ENVIRONMENTS})

    def world(self, environment: str, seed: int) -> TwinWorld:
        if environment not in ENVIRONMENTS:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        return twin(environment, seed)

    def observe(self, world: TwinWorld, k: int) -> Observation:
        planned = quarter(world)
        return Observation(
            family=FAMILY,
            environment=world.environment,
            seed=world.seed,
            k=k,
            channels=world.channels,
            sales=world.sales,
            spend=world.spend,
            controls={},
            future_controls={},
            kernel_length=1,
            planned=PLANNED,
            budget=planned.budget,
            lower=planned.lower,
            upper=planned.upper,
            status_quo=planned.status_quo,
            roi_window=window(world),
            lift=lift_rows(ladder.rung(world, world.seed, k)),
        )

    def truth(self, world: TwinWorld) -> Truth:
        planned = quarter(world)
        expected = cells(world)
        return Truth(
            quarter=planned,
            cells=expected,
            best=oracle(expected, planned),
            realised=None,
            hindsight=None,
            target=world.target,
            scale=float(np.mean(world.sales)),
            returns=true_returns(world),
        )


TWINS = Twins()

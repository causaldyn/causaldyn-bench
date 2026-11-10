"""Family 0: Track M v2 itself, through the scorecard.

The worlds are the budgets run's (:mod:`causaldyn_bench.budget_regret`): Heusch's generator over
156 weeks, every channel's retention, saturation and effect drawn on the *drawn* worlds and his own
on the *reference* ones, at the run's seeds, pilots and scored alike, each scored block going on
past the run's to 260 worlds (:data:`~causaldyn_bench.scorecard.seeds.TRACK_M2_EXTENSIONS`), drawn
as the run's are. Each channel's effect holds still, so the path the best plan is made for is the
one the quarter realises: the channel's coefficient in every week. The ladder's top rung is the
run's experiments, so a record fitted to its exports (version 1) is scored here unchanged.

The worlds of other families that are built on Track M v2's are read the same way:
:func:`observation` is what an arm reads of such a world and :func:`truth` what its plan is scored
against, whatever moved the world's effect. Where a channel's tests ran in the market whose history
an arm reads, an overlapping rung (:func:`causaldyn_bench.scorecard.ladder.overlap`), the arm reads
:func:`overlapping`'s observation, and :func:`truth` reads the returns on the history as it ran
them. :func:`respent` is a history whose business spent otherwise, and :func:`settled` the world of
a history whose effect holds still.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from causaldyn_bench.budget_regret import ENVIRONMENTS, PILOTS, lift_rows
from causaldyn_bench.endogenous_mmm import YEAR, EndogenousMediaMix, Market, MediaMixWorld, Series
from causaldyn_bench.lift_calibration import TEST
from causaldyn_bench.mmm_decision import BOX, PLANNED, Quarter
from causaldyn_bench.scorecard import ladder
from causaldyn_bench.scorecard.continuation import Shadow, shadow
from causaldyn_bench.scorecard.cpm import Price
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Held, Observation
from causaldyn_bench.scorecard.returns import true_returns
from causaldyn_bench.scorecard.seeds import TRACK_M2_EXTENSIONS, stream
from causaldyn_bench.scorecard.truth import cells, oracle


@dataclass(frozen=True, eq=False)
class World:
    """A world built on Track M v2's, as a family draws it: the history an arm reads, the quarter
    after it at the status quo, and each channel's effect over the quarter and the kernel's tail,
    ``(channels, 13 + L - 1)``, as the best plan expects it and as the quarter realises it.
    ``measured`` holds each control a family measures beyond the measurement layer's, over the
    history and over the quarter as it will be read: none on Track M v2. ``price`` is each
    channel's cost per thousand impressions, where the family prices them (:mod:`.cpm`)."""

    environment: str
    seed: int
    generator: EndogenousMediaMix
    history: MediaMixWorld
    shadow: Shadow
    expected: Series
    realised: Series
    measured: Mapping[str, tuple[Series, Series]] = field(default_factory=dict)
    price: Price | None = None


def still(history: MediaMixWorld) -> Series:
    """Each channel's effect path where the effect holds still: its coefficient in every week."""
    horizon = PLANNED + history.kernel_length - 1
    return np.repeat(np.asarray(history.effect, dtype=float)[:, None], horizon, axis=1)


def respent(history: MediaMixWorld, spend: Series) -> MediaMixWorld:
    """``history`` had its business spent ``spend``, ``(weeks, channels)``: each channel's media
    and the sales follow the spend, and all the business read before media, its budget and its
    bidding rule among it, stays."""
    media = np.column_stack([history._effect_of(c, spend[:, c]) for c in range(spend.shape[1])])
    return dataclasses.replace(
        history, spend=spend, media=media, sales=history.premedia + media.sum(axis=1)
    )


def settled(
    environment: str, seed: int, generator: EndogenousMediaMix, history: MediaMixWorld
) -> World:
    """The world of ``history``, which ``generator`` drew from ``seed``, where each channel's effect
    holds still: the path the best plan is made for is the one the quarter realises."""
    path = still(history)
    return World(
        environment, seed, generator, history, shadow(generator, history, seed, path), path, path
    )


def window(history: Market) -> tuple[int, int]:
    """The history's last year, its first and last weeks numbered from 1: the window a return is
    read over."""
    weeks = history.week.size
    return (weeks - YEAR + 1, weeks)


def observation(family: int, world: World, k: int, box: tuple[float, float] = BOX) -> Observation:
    """All an arm reads of ``world`` at rung ``k``: the history as the measurement layer reads it,
    and each control the family measures beside it, the quarter's budget, box and status quo, each
    channel's box ``box`` times its mean, the quarter's promotion calendar, price and measured
    controls as they will be read, last year's weeks as the window a return is read over, the
    rung's tests, and each channel's impressions where the world prices them."""
    history = world.history
    quarter = Quarter.after(history, box)
    return Observation(
        family=family,
        environment=world.environment,
        seed=world.seed,
        k=k,
        channels=history.channels,
        sales=history.sales,
        spend=history.spend,
        controls={
            "promotion": history.observed_promotion.astype(float),
            "price": history.observed_price,
            **{name: series for name, (series, _) in world.measured.items()},
        },
        future_controls={
            "promotion": world.shadow.observed_promotion.astype(float),
            "price": world.shadow.observed_price,
            **{name: series for name, (_, series) in world.measured.items()},
        },
        kernel_length=history.kernel_length,
        planned=PLANNED,
        budget=quarter.budget,
        lower=quarter.lower,
        upper=quarter.upper,
        status_quo=quarter.status_quo,
        roi_window=window(history),
        lift=lift_rows(ladder.rung(history, world.seed, k)),
        impressions=None if world.price is None else world.price.impressions(history.spend),
    )


def overlapping(family: int, world: World, held: ladder.Overlap) -> Observation:
    """All an arm reads of ``world`` where ``held``'s tests ran in the market whose history it
    reads (:func:`causaldyn_bench.scorecard.ladder.overlap`): the history's sales and spend as that
    market ran them, the tests' rows at the rung of their count, and where the history holds them
    (:class:`~causaldyn_bench.scorecard.observe.Held`). The rest is :func:`observation`'s, the
    quarter's budget, box and status quo among it read off the history as planned, so that the
    quarter is the one the ladder's rungs plan."""
    rows = lift_rows(held.tests)
    (tests,) = held.tests.values()
    first = np.asarray(rows.start, dtype=np.int64) - 1
    planned = held.planned[first[:, None] + np.arange(TEST)].reshape(first.size, TEST)
    return dataclasses.replace(
        observation(family, world, 0),
        k=len(tests),
        sales=held.history.sales,
        spend=held.history.spend,
        lift=rows,
        held=Held(held.share, planned),
        impressions=(None if world.price is None else world.price.impressions(held.history.spend)),
    )


def truth(
    world: World, held: ladder.Overlap | None = None, box: tuple[float, float] = BOX
) -> Truth:
    """The best plan in the box ``box`` times each channel's mean, on the expected path, the best on
    the realised one where it is another, the quarter's sales at the status quo, and each channel's
    returns on last year's spend: on the history as planned, or, where ``held``'s tests ran in the
    market, on the history as it ran them. The quarter's carryover is the same, since a test's
    darkness is over before the kernel reaches the quarter. Where the world prices its
    impressions (:mod:`.cpm`), the best plan reads the prices the world expects, and the realised
    cells and the returns the prices it drew."""
    quarter = Quarter.after(world.history, box)
    buys = anticipated = None
    moved = not np.array_equal(world.realised, world.expected)
    if world.price is not None:
        buys, anticipated = world.price.buys, world.price.anticipated
        moved = moved or not np.array_equal(buys, anticipated)
    expected = cells(world.history, quarter, world.expected, buys=anticipated)
    realised = hindsight = None
    if moved:
        realised = cells(world.history, quarter, world.realised, buys=buys)
        hindsight = oracle(realised, quarter)
    return Truth(
        quarter=quarter,
        cells=expected,
        best=oracle(expected, quarter),
        realised=realised,
        hindsight=hindsight,
        target=world.shadow.sales,
        scale=float(np.mean(world.history.sales)),
        returns=true_returns(
            world.history if held is None else held.history, window(world.history), buys
        ),
    )


class TrackM2:
    """Family 0."""

    name = "Track M v2"
    stream = stream(0)
    labels = (
        "The world's media are PyMC-Marketing's default class: its logistic saturation is the "
        "world's tanh curve, and its normalised geometric adstock at the kernel's length is the "
        "world's kernel.",
        "Every arm is told the kernel's length.",
        "A geo test is two universes at the market's scale, so its gap holds no noise between "
        "regions beyond the noise drawn.",
    )
    pilots = MappingProxyType({e.name: e.seeds for e in PILOTS})
    scored = MappingProxyType(
        {e.name: (*e.seeds, *TRACK_M2_EXTENSIONS[e.name]) for e in ENVIRONMENTS}
    )

    def world(self, environment: str, seed: int) -> World:
        if environment not in self.scored:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = next(e for e in ENVIRONMENTS if e.name == environment).generator(seed)
        history = generator.simulate(seed)
        path = still(history)
        return World(
            environment,
            seed,
            generator,
            history,
            shadow(generator, history, seed, path),
            path,
            path,
        )

    def observe(self, world: World, k: int) -> Observation:
        return observation(0, world, k)

    def truth(self, world: World) -> Truth:
        return truth(world)


TRACK_M2 = TrackM2()

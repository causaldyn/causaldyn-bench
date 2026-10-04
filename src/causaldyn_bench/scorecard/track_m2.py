"""Family 0: Track M v2 itself, through the scorecard.

The worlds are the budgets run's (:mod:`causaldyn_bench.budget_regret`): Heusch's generator over
156 weeks, every channel's retention, saturation and effect drawn on the *drawn* worlds and his own
on the *reference* ones, at the run's seeds, pilots and scored alike. Each channel's effect holds
still, so the path the best plan is made for is the one the quarter realises: the channel's
coefficient in every week. The ladder's top rung is the run's experiments, so a record fitted to
its exports (version 1) is scored here unchanged.

The worlds of other families that are built on Track M v2's are read the same way:
:func:`observation` is what an arm reads of such a world and :func:`truth` what its plan is scored
against, whatever moved the world's effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from causaldyn_bench.budget_regret import ENVIRONMENTS, PILOTS, lift_rows
from causaldyn_bench.endogenous_mmm import YEAR, EndogenousMediaMix, MediaMixWorld, Series
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard import ladder
from causaldyn_bench.scorecard.continuation import Shadow, shadow
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.returns import true_returns
from causaldyn_bench.scorecard.seeds import stream
from causaldyn_bench.scorecard.truth import cells, oracle


@dataclass(frozen=True, eq=False)
class World:
    """A world built on Track M v2's, as a family draws it: the history an arm reads, the quarter
    after it at the status quo, and each channel's effect over the quarter and the kernel's tail,
    ``(channels, 13 + L - 1)``, as the best plan expects it and as the quarter realises it."""

    environment: str
    seed: int
    generator: EndogenousMediaMix
    history: MediaMixWorld
    shadow: Shadow
    expected: Series
    realised: Series


def still(history: MediaMixWorld) -> Series:
    """Each channel's effect path where the effect holds still: its coefficient in every week."""
    horizon = PLANNED + history.kernel_length - 1
    return np.repeat(np.asarray(history.effect, dtype=float)[:, None], horizon, axis=1)


def window(history: MediaMixWorld) -> tuple[int, int]:
    """The history's last year, its first and last weeks numbered from 1: the window a return is
    read over."""
    weeks = history.week.size
    return (weeks - YEAR + 1, weeks)


def observation(family: int, world: World, k: int) -> Observation:
    """All an arm reads of ``world`` at rung ``k``: the history as the measurement layer reads it,
    the quarter's budget, box and status quo, the quarter's promotion calendar and price as they
    will be read, last year's weeks as the window a return is read over, and the rung's tests."""
    history = world.history
    quarter = Quarter.after(history)
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
        },
        future_controls={
            "promotion": world.shadow.observed_promotion.astype(float),
            "price": world.shadow.observed_price,
        },
        kernel_length=history.kernel_length,
        planned=PLANNED,
        budget=quarter.budget,
        lower=quarter.lower,
        upper=quarter.upper,
        status_quo=quarter.status_quo,
        roi_window=window(history),
        lift=lift_rows(ladder.rung(history, world.seed, k)),
    )


def truth(world: World) -> Truth:
    """The best plan on the expected path, the best on the realised one where it is another, the
    quarter's sales at the status quo, and each channel's returns on last year's spend."""
    quarter = Quarter.after(world.history)
    expected = cells(world.history, quarter, world.expected)
    realised = hindsight = None
    if not np.array_equal(world.realised, world.expected):
        realised = cells(world.history, quarter, world.realised)
        hindsight = oracle(realised, quarter)
    return Truth(
        quarter=quarter,
        cells=expected,
        best=oracle(expected, quarter),
        realised=realised,
        hindsight=hindsight,
        target=world.shadow.sales,
        scale=float(np.mean(world.history.sales)),
        returns=true_returns(world.history, window(world.history)),
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
    scored = MappingProxyType({e.name: e.seeds for e in ENVIRONMENTS})

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

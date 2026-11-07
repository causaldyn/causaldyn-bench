"""Family 20: Track M v2's drawn worlds whose promotions follow television, with the price a lever
whose counterfactual is a stated level.

Each week of the history and the quarter is a promotion week by a draw of its own: with chance
``p_heavy`` in a week television bursts in (his burst weeks, 23 to 25 and 48 to 52 of each year,
:data:`~causaldyn_bench.endogenous_mmm.TV_BURSTS`) and ``p_light`` in any other. A promotion week
cuts the price by :data:`CUT`, 10 %, at his elasticity multiplier of 1, which lifts the baseline
``0.9^-0.9 - 1``, about 10 %: near what a burst week's television brings. Each environment is one
pair of chances:

* ``independent``: 0.16 in every week, so promotions do not follow television;
* ``tv-0.5``: 0.5 in a burst week and 0.1 in any other;
* ``tv-0.8``: 0.8 and 0.05.

Each holds about 8.4 promotion weeks a year, his calendar's 9 within one. The generator runs on the
drawn calendar in his calendar's place (:meth:`Promoted.calendar`), so everything that reads the
promotions reads it: the uplift, the spend raised ahead of a promotion, the budget and the bidding,
which read the sales before media, the measurement layer's promotion indicator and price, and the
quarter's processes. An arm that reads television's sales without the promotions that come with
them reads the promotions' sales as television's.

The price is a lever whose counterfactual is a stated level, :data:`LEVEL`, 1: the price position
at its mean, undiscounted, which the measurement layer's price reads about. Its contribution is
what the sales before media over the window a return is read on hold above what they would at that
level, the baseline's price position and the promotions' discounts both set to it, every other
process as it ran and the media as they ran (CHOICE: the lever is read with the spend held, as a
channel's return is read with the others' held):

    sum_t b_t mu_t - max(b_t - B (1 + s_t)(1 + q_t)(pi_t^-0.9 - l^-0.9), B / 2),

``b_t`` the baseline as it ran, ``mu_t`` the promotion's uplift, ``B`` the base, ``s_t`` and ``q_t``
the seasonality and the quality, ``pi_t`` the price position and ``l`` the level: the second term is
the baseline at the level, on his floor of half the base. Where the baseline as it ran sits on that
floor the counterfactual is not read off the world, and the truth refuses it. The truth states it
beside every channel's return (:func:`contribution`).

The calendar is drawn from the family's parameter stream for the world's seed (:mod:`.seeds`);
beneath it each world is Track M v2's drawn world for its seed, its processes, its quarter's and its
lift tests' noise among them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from causaldyn_bench.endogenous_mmm import (
    ELASTICITY,
    TV_BURSTS,
    YEAR,
    EndogenousMediaMix,
    Series,
)
from causaldyn_bench.mmm_decision import PLANNED, drawn
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, settled, truth, window

FAMILY = 20
CUT = 0.1  # a promotion week's discount
LEVEL = 1.0  # the price's counterfactual: the position at its mean, undiscounted
LEAD = 3  # the weeks ahead the spend reads the promotions over
# each environment's chance of a promotion in a week television bursts in, and in any other
CHANCES: MappingProxyType[str, tuple[float, float]] = MappingProxyType(
    {"independent": (0.16, 0.16), "tv-0.5": (0.5, 0.1), "tv-0.8": (0.8, 0.05)}
)


def bursting(week: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Whether television bursts in each week, by his burst calendar."""
    of_year = (np.asarray(week) - 1) % YEAR + 1
    out = np.zeros(of_year.shape, dtype=bool)
    for first, last, _ in TV_BURSTS:
        out |= (of_year >= first) & (of_year <= last)
    return out


@dataclass(frozen=True)
class Promoted(EndogenousMediaMix):
    """His generator on a drawn calendar: a promotion at a discount of ``cut`` in each week of
    ``promoted``, numbered from 1, at an elasticity multiplier of 1, and none in any other."""

    promoted: tuple[int, ...] = ()
    cut: float = CUT

    def calendar(self, week: NDArray[np.int64]) -> tuple[Series, Series]:
        on = np.isin(week, self.promoted)
        return np.where(on, self.cut, 0.0), np.ones(np.shape(week))


def promotions(environment: str, seed: int, weeks: int) -> tuple[int, ...]:
    """The promotion weeks of world ``seed`` of ``environment``, numbered from 1, over the history's
    ``weeks``, the quarter's and the weeks the spend reads ahead of the quarter's last."""
    heavy, light = CHANCES[environment]
    week = np.arange(1, weeks + PLANNED + LEAD + 1)
    chance = np.where(bursting(week), heavy, light)
    drawn_ = rng(FAMILY, Role.PARAMETERS, seed).random(week.size) < chance
    return tuple(int(w) for w in week[drawn_])


def contribution(world: World, level: float = LEVEL) -> float:
    """The price's contribution over the window a return is read on, against the position ``level``
    undiscounted: the sales before media less the baseline at that level, see the module.

    Raises:
        ValueError: a week of the window whose baseline sits on its floor.
    """
    history = world.history
    base = world.generator.base
    first, last = window(history)
    inside = slice(first - 1, last)
    baseline = history.baseline[inside]
    if np.any(baseline <= 0.5 * base):
        raise ValueError("the baseline sits on its floor in the window: its counterfactual is lost")
    demand = base * (1.0 + history.seasonality[inside]) * (1.0 + history.quality[inside])
    moved = demand * (history.price_position[inside] ** ELASTICITY - level**ELASTICITY)
    floor = np.maximum(baseline - moved, 0.5 * base)
    return float(np.sum(history.premedia[inside] - floor))


class Promotions:
    """Family 20."""

    name = "promotions with television"
    stream = stream(FAMILY)
    labels = (
        "The promotions fall in television's burst weeks more often than in others, so they move "
        "the sales with television; an arm reads them through the measurement layer's promotion "
        "indicator and price.",
        "The price is a lever whose counterfactual is a stated level, 1: the price position at its "
        "mean, undiscounted.",
        "Beneath its calendar each world is a Track M v2 drawn world, whose media are "
        "PyMC-Marketing's default class.",
        *TRACK_M2.labels[1:],
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(CHANCES)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(CHANCES)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in CHANCES:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        base = drawn(seed)
        kept = {f.name: getattr(base, f.name) for f in dataclasses.fields(EndogenousMediaMix)}
        generator = Promoted(**kept, promoted=promotions(environment, seed, base.weeks))
        return settled(environment, seed, generator, generator.simulate(seed))

    def observe(self, world: World, k: int) -> Observation:
        return observation(FAMILY, world, k)

    def truth(self, world: World) -> Truth:
        return dataclasses.replace(truth(world), levers={"price": contribution(world)})


PROMOTIONS = Promotions()

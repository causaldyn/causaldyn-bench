"""Family 5: Track M v2's drawn worlds whose budget follows the sales' surprise of last week.

From week 12 every channel's spend in week ``t`` is the drawn world's times ``exp(rho s_{t-1})``,

    s_{t-1} = log(y_{t-1} / mean(y_{t-11}, ..., y_{t-2})),

the sales of last week over the ten weeks before it: the window his bidding rule reads (his Eq. 6),
on the sales, media included, where his rule reads them before media. The spend then answers the
media it bought, so the history is run week by week: a week's media follow its spend, its sales
follow its media, and next week's spend follows those. Each environment is one ``rho``: 0, 0.3 and
0.6; at 0 the world is Track M v2's, bit for bit.

The world's budget and bidding rule read its sales before media, so neither moves. The quarter's
spend is the plan's, so the rule moves the quarter's box, budget and status quo, which last year's
spend sets, and not the plan's worth. Beneath the rule each world is Track M v2's drawn world for
its seed, its quarter's processes and its lift tests' noise among it: the family draws no variate
of its own.
"""

from __future__ import annotations

import math
from types import MappingProxyType

from causaldyn_bench.endogenous_mmm import MediaMixWorld, Series
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import pilot, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, respent, settled, truth

FAMILY = 5
TRAILING = 10  # the weeks before last week its sales are read against
FIRST = TRAILING + 1  # week 12, numbered from 0: the first with all of them
RHOS: MappingProxyType[str, float] = MappingProxyType(
    {"rho-0": 0.0, "rho-0.3": 0.3, "rho-0.6": 0.6}
)


def surprise(sales: Series, t: int) -> float:
    """``s_{t-1}``, the sales of the week before week ``t``, numbered from 0, over the ten before
    it, logged."""
    return math.log(sales[t - 1] / sales[t - 1 - TRAILING : t - 1].mean())


def follow(history: MediaMixWorld, rho: float) -> MediaMixWorld:
    """``history`` with every channel's spend from week 12 its own times ``exp(rho s_{t-1})``, run
    week by week, and the media and the sales with it."""
    spend, media, sales = history.spend.copy(), history.media.copy(), history.sales.copy()
    for t in range(FIRST, sales.size):
        spend[t] = history.spend[t] * math.exp(rho * surprise(sales, t))
        for c in range(spend.shape[1]):
            media[t, c] = history._effect_of(c, spend[: t + 1, c])[t]
        sales[t] = history.premedia[t] + media[t].sum()
    return respent(history, spend)


class Performance:
    """Family 5."""

    name = "budget follows performance"
    stream = stream(FAMILY)
    labels = (
        "Beneath the rule each world is a Track M v2 drawn world, whose media are "
        "PyMC-Marketing's default class; at rho = 0 it is that world, bit for bit.",
        *TRACK_M2.labels[1:],
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(RHOS)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(RHOS)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in RHOS:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = drawn(seed)
        history = follow(generator.simulate(seed), RHOS[environment])
        return settled(environment, seed, generator, history)

    def observe(self, world: World, k: int) -> Observation:
        return observation(FAMILY, world, k)

    def truth(self, world: World) -> Truth:
        return truth(world)


PERFORMANCE = Performance()

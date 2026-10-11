"""Family 2: Track M v2's drawn worlds whose spend follows the market's sentiment, which moves the
sales too and which no arm observes.

The sentiment ``u_t`` is the drawn world's own: the AR(1) of persistence 0.7 and shocks of sd 0.03
whose ``B u_t`` the baseline carries (his Eq. 2), and which his measurement layer never reads. Every
channel's log spend in week ``t`` reads it,

    log s_{c,t} = log s^0_{c,t} + kappa 0.15 u_t / sd(u),

with ``s^0`` the drawn world's spend and ``sd(u) = 0.03 / sqrt(1 - 0.7^2)`` the sentiment's
stationary sd: at ``kappa = 1`` a channel's spend moves 0.15 on the log scale, half again his
plan's own noise of 0.1, for each sd of the demand it senses. An arm that reads the spend and the
sales without the sentiment reads the sentiment's sales as the media's.

Each environment is one ``kappa``: 0.5, 1 and 2. The world's budget and bidding rule read its
sales before media, so neither moves; the media and the sales follow the spend
(:func:`~causaldyn_bench.scorecard.track_m2.respent`). The quarter's spend is the plan's, so the
sentiment moves the quarter's box, budget and status quo, which last year's spend sets, and not
the plan's worth. Beneath the confounding each world is Track M v2's drawn world for its seed, its
quarter's processes and its lift tests' noise among it: the family draws no variate of its own.
"""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np

from causaldyn_bench.endogenous_mmm import MediaMixWorld
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import pilot, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, respent, settled, truth

FAMILY = 2
REACTION = 0.15  # a channel's log spend per sd of the sentiment, at kappa = 1
STATIONARY = 0.03 / math.sqrt(1.0 - 0.7**2)  # the sentiment's stationary sd, his AR(1)'s
KAPPAS: MappingProxyType[str, float] = MappingProxyType(
    {"kappa-0.5": 0.5, "kappa-1": 1.0, "kappa-2": 2.0}
)


def confound(history: MediaMixWorld, kappa: float) -> MediaMixWorld:
    """``history`` with every channel's log spend moved by ``kappa`` times :data:`REACTION` for
    each stationary sd of its sentiment, and the media and the sales with it."""
    reaction = np.exp(kappa * REACTION * history.sentiment / STATIONARY)
    return respent(history, history.spend * reaction[:, None])


class Confounded:
    """Family 2."""

    name = "confounded with demand"
    stream = stream(FAMILY)
    labels = (
        "Every channel's spend follows the market's sentiment, which moves the sales too: the "
        "confounder is unobserved by every arm.",
        "Beneath the confounding each world is a Track M v2 drawn world, whose media are "
        "PyMC-Marketing's default class.",
        *TRACK_M2.labels[1:],
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(KAPPAS)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(KAPPAS)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in KAPPAS:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = drawn(seed)
        history = confound(generator.simulate(seed), KAPPAS[environment])
        return settled(environment, seed, generator, history)

    def observe(self, world: World, k: int) -> Observation:
        return observation(FAMILY, world, k)

    def truth(self, world: World) -> Truth:
        return truth(world)


CONFOUNDED = Confounded()

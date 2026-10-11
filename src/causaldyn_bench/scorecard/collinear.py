"""Family 3: Track M v2's drawn worlds whose paid shopping and Meta spend move together.

His business plans each channel's log spend with noise of its own, of sd 0.1 a week (his Eq. 4),
and its bidding rule scales paid shopping's (PLA's) spend alone (his Eq. 6). Here Meta's planning
noise is PLA's at correlation ``r``,

    e'_meta = r e_pla + sqrt(1 - r^2) e_meta,

from the drawn world's own noise (:attr:`~causaldyn_bench.endogenous_mmm.MediaMixWorld.plan_noise`),
and the bidding rule scales Meta's spend as it scales PLA's. The two channels then share every
mechanism that sets their spend but their base levels, and differ by ``sqrt(1 - r^2)`` of Meta's own
noise alone. Each environment is one ``r``: 0.9, 0.97 and 0.99.

The world's budget and bidding rule read its sales before media, so neither moves; Meta's media and
the sales follow its spend (:func:`~causaldyn_bench.scorecard.track_m2.respent`). Beneath the
coupling each world is Track M v2's drawn world for its seed, its quarter's processes and its lift
tests' noise among it: the family draws no variate of its own.
"""

from __future__ import annotations

import dataclasses
import math
from types import MappingProxyType

import numpy as np

from causaldyn_bench.endogenous_mmm import MediaMixWorld
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import pilot, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, respent, settled, truth

FAMILY = 3
COUPLED = ("pla", "meta")  # the bidding rule's channel, and the channel coupled to it
CORRELATIONS: MappingProxyType[str, float] = MappingProxyType(
    {"r-0.9": 0.9, "r-0.97": 0.97, "r-0.99": 0.99}
)


def couple(history: MediaMixWorld, r: float) -> MediaMixWorld:
    """``history`` with Meta's planning noise PLA's at correlation ``r`` and Meta's spend scaled by
    the bidding rule, and Meta's media and the sales with it."""
    lead, follower = (history.channels.index(name) for name in COUPLED)
    noise = history.plan_noise.copy()
    noise[:, follower] = r * noise[:, lead] + math.sqrt(1.0 - r**2) * noise[:, follower]
    moved = np.exp(noise[:, follower] - history.plan_noise[:, follower]) * history.bidding
    spend = history.spend.copy()
    spend[:, follower] *= moved
    return dataclasses.replace(respent(history, spend), plan_noise=noise)


class Collinear:
    """Family 3."""

    name = "collinear channels"
    stream = stream(FAMILY)
    labels = (
        "Beneath the coupling each world is a Track M v2 drawn world, whose media are "
        "PyMC-Marketing's default class.",
        *TRACK_M2.labels[1:],
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(CORRELATIONS)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(CORRELATIONS)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in CORRELATIONS:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = drawn(seed)
        history = couple(generator.simulate(seed), CORRELATIONS[environment])
        return settled(environment, seed, generator, history)

    def observe(self, world: World, k: int) -> Observation:
        return observation(FAMILY, world, k)

    def truth(self, world: World) -> Truth:
        return truth(world)


COLLINEAR = Collinear()

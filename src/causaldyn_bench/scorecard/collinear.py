"""Family 3: Track M v2's drawn worlds whose paid shopping and Meta spend move together.

His business plans each channel's log spend with noise of its own, of sd 0.1 a week (his Eq. 4),
and its bidding rule scales paid shopping's (PLA's) spend alone (his Eq. 6). Here Meta takes the
share ``c`` of PLA's own movements: its planning noise is PLA's at correlation ``c``,

    e'_meta = c e_pla + sqrt(1 - c^2) e_meta,

from the drawn world's own noise (:attr:`~causaldyn_bench.endogenous_mmm.MediaMixWorld.plan_noise`),
and the bidding rule scales Meta's spend to the power ``c``. At ``c = 0`` the world is Track M v2's;
at ``c = 1`` the two channels' log spend differs by a constant.

Each environment is the collinearity an arm sees, ``r``: 0.9, 0.97 and 0.99, the correlation of the
two channels' log spend over the history, as an arm computes it from the spend it reads, averaged
over the environment's worlds. Each environment's ``c`` (:data:`SHARES`) was found once, by
bisection on ``c`` to 1e-10 where that average over the family's pilot worlds is ``r``, and rounded
to six places: over its three pilot blocks pooled, 300 worlds, since the environments differ in
``c`` alone, so that the scored worlds never entered it. Each pilot block's average is 0.706 to
0.717 at ``c = 0``. The bidding rule alone, at Meta's noise its own, gives 0.927 to 0.930, past
0.9; and Meta's noise PLA's alone, without the rule, at most 0.789 to 0.792, short of 0.9: so ``c``
moves both.

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
# each environment's share c, from the family's pilot worlds alone; see the module
SHARES: MappingProxyType[str, float] = MappingProxyType(
    {"r-0.9": 0.457375, "r-0.97": 0.748522, "r-0.99": 0.890720}
)


def couple(history: MediaMixWorld, share: float) -> MediaMixWorld:
    """``history`` with Meta's planning noise PLA's at correlation ``share`` and Meta's spend scaled
    by the bidding rule to the power ``share``, and Meta's media and the sales with it."""
    lead, follower = (history.channels.index(name) for name in COUPLED)
    noise = history.plan_noise.copy()
    noise[:, follower] = share * noise[:, lead] + math.sqrt(1.0 - share**2) * noise[:, follower]
    moved = np.exp(noise[:, follower] - history.plan_noise[:, follower]) * history.bidding**share
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
        history = couple(generator.simulate(seed), SHARES[environment])
        return settled(environment, seed, generator, history)

    def observe(self, world: World, k: int) -> Observation:
        return observation(FAMILY, world, k)

    def truth(self, world: World) -> Truth:
        return truth(world)


COLLINEAR = Collinear()

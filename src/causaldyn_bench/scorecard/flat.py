"""Family 19: Track M v2's drawn worlds in which one channel's spend never moves.

Each environment, ``flat-<channel>``, names the channel held flat: its spend is its mean weekly
spend over the drawn history in every week of the history, and in the kernel's weeks before it
(CHOICE: a channel bought at one level for as long as it has run, so its media hold one level from
the first week and no week reads its adstock filling up). Its media and the sales follow; every
other channel, the business's rules among them, is the drawn world's, so the budget and the bidding
move the other channels' spend and not the flat one's, as a contract fixed for the history would. A
history then holds nothing that moves the flat channel's effect: its return is told apart from the
base only by the curve and the coefficient an arm assumes, and an honest arm reads it with a wide
interval, or holds its spend where no test has read it.

The truth states the flat channel's return as every channel's (:mod:`.returns`): its ROI and mROI
on last year's spend, which a lift test at a rung of the ladder reads, by taking it dark. Its box
over the quarter is the scorecard's, ``0.5`` to ``2`` times its spend, so the best plan may move it.
Beneath the flat channel each world is Track M v2's drawn world for its seed, its quarter's
processes and its lift tests' noise among it: the family draws no variate of its own.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from causaldyn_bench.endogenous_mmm import MediaMixWorld, Series
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import pilot, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, settled, truth

FAMILY = 19
# each environment and the channel it holds flat, one of Track M v2's
FLATTENED: MappingProxyType[str, str] = MappingProxyType(
    {"flat-pla": "pla", "flat-meta": "meta", "flat-tv": "tv"}
)
ENVIRONMENTS = tuple(FLATTENED)


@dataclass(frozen=True, kw_only=True)
class Flat(MediaMixWorld):
    """A history whose channel ``flat`` spent ``level`` in every week, and in the kernel's weeks
    before the first: its geo tests and its media read that level before the history."""

    flat: int
    level: float

    def _effect_of(self, column: int, spend: Series) -> Series:
        if column != self.flat:
            return super()._effect_of(column, spend)
        before = np.full(self.kernel_length - 1, self.level)
        whole = np.concatenate([before, np.asarray(spend, dtype=float)])
        return super()._effect_of(column, whole)[before.size :]


def flatten(history: MediaMixWorld, channel: str) -> Flat:
    """``history`` with ``channel``'s spend at its mean weekly spend in every week and before the
    first, and its media and the sales with it; all the business read before media stays."""
    column = history.channels.index(channel)
    level = float(np.mean(history.spend[:, column]))
    spend = history.spend.copy()
    spend[:, column] = level
    kept = {f.name: getattr(history, f.name) for f in dataclasses.fields(MediaMixWorld)}
    world = Flat(**kept, flat=column, level=level)
    media = history.media.copy()
    media[:, column] = world._effect_of(column, spend[:, column])
    sales = history.premedia + media.sum(axis=1)
    return dataclasses.replace(world, spend=spend, media=media, sales=sales)


class FlatSpend:
    """Family 19."""

    name = "flat spend"
    stream = stream(FAMILY)
    labels = (
        "One channel's spend never moves, before the history or in it: its effect is not told "
        "apart from the base by the history, only by a lift test, which takes it dark.",
        "Beneath the flat channel each world is a Track M v2 drawn world, whose media are "
        "PyMC-Marketing's default class.",
        *TRACK_M2.labels[1:],
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(ENVIRONMENTS)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(ENVIRONMENTS)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in ENVIRONMENTS:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = drawn(seed)
        history = flatten(generator.simulate(seed), FLATTENED[environment])
        return settled(environment, seed, generator, history)

    def observe(self, world: World, k: int) -> Observation:
        return observation(FAMILY, world, k)

    def truth(self, world: World) -> Truth:
        return truth(world)


FLAT = FlatSpend()

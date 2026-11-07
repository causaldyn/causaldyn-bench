"""Family 18: Track M v2's drawn worlds beside a brand measure, in which nothing media build lasts
in sales: the long-term share of every channel's return is 0.

Every world exports a weekly brand measure, the branded searches, beside the measurement layer's
controls: a level of 500 a week with a yearly wave of 50 and a normal noise of sd 15,

    s_t = 500 + 50 sin(2 pi t / 52) + b p h_t + 15 z_t,

``h`` television's spend over its mean positive week of the history, carried by a normalised
geometric kernel of retention 0.7 over the whole series and read through ``a / (a + 1)``, and ``p``
the history's ``sum(tv) / sum(h)``, so that television builds ``b`` searches a thousand euros of its
spend, most of them in the weeks after it. Sales read ``l (s_t - 500)`` beside the drawn world's.
The path from television through the searches to sales is cut at one of its two links, so the
searches carry none of any channel's return to sales:

* ``unread``: television builds the searches, ``b = 1.25``, and sales do not read them, ``l = 0``;
* ``unbuilt``: sales read the searches, ``l = 2`` a search, and television does not build them,
  ``b = 0``.

Beneath the searches lie Track M v2's channels, spend and sales (CHOICE: a drawn world's, in
place of a world of two channels of its own): the case Cain (2025, Applied Marketing
Analytics 11(1)) asks a model to rule out before it reads a long-term effect. A model that reads
television's return through the searches, as a nested model does, or one that gives television a
second, long column, as a dual adstock does, must read a share of 0. The truth states each
channel's long-term share (:func:`long_term`): its return through the searches over its whole
return, 0 in both cuts, read off the world's ``b`` and ``l`` rather than set.

An arm is told the searches over the quarter with television at the status quo; their noise is the
family's own, from its measurement stream over the history and its continuation stream over the
quarter (:mod:`.seeds`). In ``unbuilt`` the quarter's sales read them as the history's do;
television builds none there, so no plan moves them. The business's budget and bidding read its
sales before media, which hold no searches (CHOICE), so the drawn world's spend stays. Beneath the
searches each world is Track M v2's drawn world for its seed, its quarter's processes and its lift
tests' noise among it.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from causaldyn_bench.endogenous_mmm import YEAR, Series
from causaldyn_bench.mmm_decision import PLANNED, Quarter, drawn
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, settled, truth, window

FAMILY = 18
LEVEL, WAVE, NOISE = 500.0, 50.0, 15.0  # the searches' level, yearly wave and noise, a week
RETENTION = 0.7  # television's carryover into the searches
BUILT, LINK = 1.25, 2.0  # searches a thousand euros of television builds; sales a search
# each cut's (b, l): what television builds of the searches, and what sales read of them
CUTS: MappingProxyType[str, tuple[float, float]] = MappingProxyType(
    {"unread": (BUILT, 0.0), "unbuilt": (0.0, LINK)}
)
TAIL = 104  # the weeks after the history a return through the searches is counted over


def carried(tv: Series, scale: float) -> Series:
    """``h``: the spend over ``scale`` carried by the normalised geometric kernel of retention
    :data:`RETENTION` over the whole series, through ``a / (a + 1)``."""
    weights = RETENTION ** np.arange(tv.size)
    adstock = np.convolve(tv / scale, weights / weights.sum())[: tv.size]
    return adstock / (adstock + 1.0)


@dataclass(frozen=True, eq=False, kw_only=True)
class Branded(World):
    """A world beside its searches: what television builds of them a thousand euros, ``built``,
    what sales read a search above the level, ``link``, television's mean positive week over the
    history, ``scale``, and the history's ``sum(tv) / sum(h)``, ``per``."""

    built: float
    link: float
    scale: float
    per: float

    def built_searches(self, tv: Series) -> Series:
        """The searches television's spend ``tv`` builds in each week from the history's first."""
        return self.built * self.per * carried(tv, self.scale)


def branded(world: World, built: float, link: float) -> Branded:
    """``world`` beside its searches, built at ``built`` and read by sales at ``link``: the
    history's sales and the quarter's read them, and an arm reads them over both."""
    history = world.history
    tv = history.spend[:, history.channels.index("tv")]
    weeks = tv.size
    scale = float(np.mean(tv[tv > 0.0]))
    quo = Quarter.after(history).status_quo[history.channels.index("tv")]
    whole = np.concatenate([tv, np.full(PLANNED, quo)])
    kept = {f.name: getattr(world, f.name) for f in dataclasses.fields(World)}
    out = Branded(
        **kept,
        built=built,
        link=link,
        scale=scale,
        per=float(tv.sum() / carried(whole, scale)[:weeks].sum()),
    )
    t = np.arange(1, weeks + PLANNED + 1)
    noise = np.concatenate(
        [
            rng(FAMILY, Role.MEASUREMENT, world.seed).standard_normal(weeks),
            rng(FAMILY, Role.CONTINUATION, world.seed).standard_normal(PLANNED),
        ]
    )
    searches = (
        LEVEL + WAVE * np.sin(2.0 * math.pi * t / YEAR) + out.built_searches(whole) + NOISE * noise
    )
    past, future = searches[:weeks], searches[weeks:]
    return dataclasses.replace(
        out,
        history=dataclasses.replace(history, sales=history.sales + link * (past - LEVEL)),
        shadow=dataclasses.replace(
            world.shadow, sales=world.shadow.sales + link * (future - LEVEL)
        ),
        measured={"searches": (past, future)},
    )


def through(world: Branded) -> Series:
    """Each channel's return on last year's spend through the searches, a euro: what the searches
    it builds over the history and :data:`TAIL` weeks after, nothing spent after the history, lose
    with the window's spend removed, times what sales read a search, over what the window spent.
    Television alone builds searches."""
    history = world.history
    first, last = window(history)
    out = np.zeros(len(history.channels))
    c = history.channels.index("tv")
    tv = np.concatenate([history.spend[:, c], np.zeros(TAIL)])
    without = tv.copy()
    without[first - 1 : last] = 0.0
    lost = world.built_searches(tv) - world.built_searches(without)
    out[c] = world.link * float(lost.sum()) / float(tv[first - 1 : last].sum())
    return out


def long_term(world: Branded, returns: Series) -> Series:
    """Each channel's long-term share: its return through the searches over that and ``returns``,
    its return in the sales beside them."""
    lasting = through(world)
    return lasting / (lasting + returns)


class LongTermNull:
    """Family 18."""

    name = "long-term null"
    stream = stream(FAMILY)
    labels = (
        "An arm reads a brand measure, the branded searches, beside the promotion and the price: "
        "television builds them in one cut and sales read them in the other, so nothing a channel "
        "builds lasts in sales, and every channel's long-term share is 0.",
        "Beneath the searches each world is a Track M v2 drawn world, whose media are "
        "PyMC-Marketing's default class.",
        *TRACK_M2.labels[1:],
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(CUTS)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(CUTS)})

    def world(self, environment: str, seed: int) -> Branded:
        if environment not in CUTS:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = drawn(seed)
        world = settled(environment, seed, generator, generator.simulate(seed))
        return branded(world, *CUTS[environment])

    def observe(self, world: Branded, k: int) -> Observation:
        return observation(FAMILY, world, k)

    def truth(self, world: Branded) -> Truth:
        base = truth(world)
        return dataclasses.replace(base, long_term=long_term(world, base.returns.roi))


LONG_TERM = LongTermNull()

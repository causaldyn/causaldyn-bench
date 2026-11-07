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
quarter's processes and its lift tests' noise among it.

A fourth environment, ``search-demand``, is paid search's: only paid shopping's spend follows the
sentiment, at ``kappa = 2``, as paid search buys against the demand its auctions show (CHOICE: the
strongest of the three), and every arm reads a query volume beside the promotion and the price,
the demand measured with a log-normal error of sd 0.02 (CHOICE, the error):

    queries_t = (y^0_t + eta m_{tv,t}) / B exp(0.02 z_t - 0.02^2 / 2),

``y^0`` the sales before media, ``m_tv`` television's media, ``B`` the base and ``eta = 0.5``.
Television's media raise the demand the queries measure by ``eta`` of what they bring, and bring
the rest directly: the sales are the drawn world's, ``y^0 + sum_c m_c``, so each channel's return is
the world's own, and the queries confound paid shopping, whose spend reads the demand, and mediate
television. Held as a control, they block ``eta`` of television's return: the truth states each
channel's return with the queries held (:func:`direct`), ``1 - eta`` of television's and every other
channel's whole. The queries' errors are drawn from the family's measurement stream over the
history and its continuation stream over the quarter (:mod:`.seeds`), the quarter's read at the
status quo; the other environments draw no variate of their own.
"""

from __future__ import annotations

import dataclasses
import math
from types import MappingProxyType

import numpy as np

from causaldyn_bench.endogenous_mmm import MediaMixWorld, Series
from causaldyn_bench.mmm_decision import PLANNED, drawn
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.returns import Returns
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, respent, settled, truth

FAMILY = 2
REACTION = 0.15  # a channel's log spend per sd of the sentiment, at kappa = 1
STATIONARY = 0.03 / math.sqrt(1.0 - 0.7**2)  # the sentiment's stationary sd, his AR(1)'s
KAPPAS: MappingProxyType[str, float] = MappingProxyType(
    {"kappa-0.5": 0.5, "kappa-1": 1.0, "kappa-2": 2.0}
)
SEARCH = "search-demand"  # the environment in which paid shopping alone reads the demand
SEARCHED, MEDIATED = "pla", "tv"  # the channel that buys against the demand; the one that raises it
SEARCH_KAPPA = 2.0
ETA = 0.5  # the share of television's media that runs through the demand the queries measure
QUERY_ERROR = 0.02  # the queries' log-normal error's sd
ENVIRONMENTS = (*KAPPAS, SEARCH)


def confound(
    history: MediaMixWorld, kappa: float, channels: tuple[str, ...] | None = None
) -> MediaMixWorld:
    """``history`` with each of ``channels``' log spend, every channel's where it is None, moved by
    ``kappa`` times :data:`REACTION` for each stationary sd of its sentiment, and the media and the
    sales with it."""
    reaction = np.exp(kappa * REACTION * history.sentiment / STATIONARY)
    moved = [c in (history.channels if channels is None else channels) for c in history.channels]
    return respent(history, history.spend * np.where(moved, reaction[:, None], 1.0))


def queries(premedia: Series, media: Series, base: float, error: Series) -> Series:
    """The query volume over the weeks: the sales before media and :data:`ETA` of television's
    media, over the base, read with the log-normal error of standard normal draws ``error``."""
    spread = QUERY_ERROR * error - QUERY_ERROR**2 / 2.0
    return (premedia + ETA * media) / base * np.exp(spread)


def searched(world: World) -> World:
    """``world`` with its query volume over the history and over the quarter at the status quo."""
    history, generator = world.history, world.generator
    tv = history.channels.index(MEDIATED)
    weeks = history.week.size
    past = queries(
        history.premedia,
        history.media[:, tv],
        generator.base,
        rng(FAMILY, Role.MEASUREMENT, world.seed).standard_normal(weeks),
    )
    future = queries(
        world.shadow.premedia,
        world.shadow.media[:, tv],
        generator.base,
        rng(FAMILY, Role.CONTINUATION, world.seed).standard_normal(PLANNED),
    )
    return dataclasses.replace(world, measured={"queries": (past, future)})


def direct(returns: Returns, channels: tuple[str, ...]) -> Returns:
    """Each channel's ``returns`` with the queries held: ``1 - eta`` of television's, whose media
    raise the demand they measure by ``eta``, and every other channel's whole."""
    held = np.array([1.0 - ETA if c == MEDIATED else 1.0 for c in channels])
    return Returns(returns.window, returns.roi * held, returns.marginal * held)


class Confounded:
    """Family 2."""

    name = "confounded with demand"
    stream = stream(FAMILY)
    labels = (
        "Every channel's spend follows the market's sentiment, which moves the sales too: the "
        "confounder is unobserved by every arm.",
        "In the search-demand environment only paid shopping's spend follows it, and every arm "
        "reads a query volume that measures the demand, television's part of it included.",
        "Beneath the confounding each world is a Track M v2 drawn world, whose media are "
        "PyMC-Marketing's default class.",
        *TRACK_M2.labels[1:],
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(ENVIRONMENTS)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(ENVIRONMENTS)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in ENVIRONMENTS:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = drawn(seed)
        if environment == SEARCH:
            history = confound(generator.simulate(seed), SEARCH_KAPPA, (SEARCHED,))
            return searched(settled(environment, seed, generator, history))
        history = confound(generator.simulate(seed), KAPPAS[environment])
        return settled(environment, seed, generator, history)

    def observe(self, world: World, k: int) -> Observation:
        return observation(FAMILY, world, k)

    def truth(self, world: World) -> Truth:
        base = truth(world)
        if world.environment != SEARCH:
            return base
        return dataclasses.replace(base, direct=direct(base.returns, world.history.channels))


CONFOUNDED = Confounded()

"""Family 8: geo worlds, drawn by :mod:`chc.geo_world`.

Each world is ten geos over 104 weeks. A geo's KPI a week is its base on a season every geo shares,
plus its media's lifts, plus noise; a channel lifts a geo's population times a curve of its spend
per head through a normalised geometric adstock of eight weeks, times the geo's effect, log-normal
about the channel's national median with a spread ``tau_c`` drawn for each world uniform on
``[0.2, 0.6]``. Three channels: search and social, which each geo buys itself, and TV, bought
nationally and allotted to the geos by population:

    search  Tanh at 2 a head, effect 3, retention 0.3, spend 1.5 a head
    social  Hill of slope 1 at 1 a head, effect 1.5, retention 0.5, spend 0.8 a head
    tv      Tanh at 4 a head, effect 2, retention 0.7, spend 2 a head, national

CHOICE: search and TV saturate as a Tanh and social as a Hill of slope 1, so every cell is concave;
their scales, and the effects, retentions and spends, are those of the geo mix chc's geo DLM
benchmark (``scripts/bench_geo_dlm.py``) draws, where search is Michaelis-Menten, social exponential
and a fourth channel, video, is left out here. A geo's base, 19 a head, and its noise, 230 a head,
make a geo read like a Track M v2 market: on the family's first 40 pilot worlds at policy 0 media
are 0.164 of the KPI, and the noise of a geo of 1e5 people is 0.031 of its mean KPI, where on Track
M v2's 40 drawn pilot worlds media are 0.168 of the sales and the sentiment and noise together
0.033. chc's own defaults, a base and a noise of 1 a head, put media at 0.79 of the KPI and the
noise at 0.0007.

Each environment is one ``policy``, 0 and 0.5: how far the log's spend rises with the season, the
confounding of :mod:`chc.mmm`. At one seed the two policies draw the same variates, so the policy
alone moves the spend; each environment draws its worlds from a block of seeds of its own, as every
family's do. Every other parameter is chc's default: populations log-normal about 1e5 with a log-SD
of 1, bases per head with 0.2, each geo's intensity on a channel with 0.5, a week's spend jitter
with 0.3, a season of 0.2 of the base, no spillover and no drift.

The decision is the quarter after the history over the world's cells: each geo's own channels, geo
by geo, then TV once, its plan allotted to the geos by population
(:func:`~causaldyn_bench.scorecard.truth.geo_parts`), each within half and twice its mean over the
last year, at the budget of last year's rate. The status quo is every cell at its own mean, TV's
its spend over the geos summed. The equal split gives each channel, not each cell, an equal share of
the budget, a per-geo channel's share split across the geos by population, then moves it into the
box (:class:`~causaldyn_bench.scorecard.truth.GeoQuarter`): split over the 21 cells, TV would get a
21st. An arm may plan one weekly spend a channel instead of a cell: where plans are scored
(:mod:`~causaldyn_bench.scorecard.scoring`) it is allotted to the cells once, for every arm, a
per-geo channel's across the geos as the status quo splits it and TV's to its one cell. The oracle
is exact: every cell is concave. The ladder
is geo experiments read by difference-in-differences
(:func:`~causaldyn_bench.scorecard.ladder.geo_rung`), noisy ones: on the family's first 30 pilot
worlds at policy 0 a reading of a world without noise is off the true change, relative to it, by
+0.06 on the mean with a standard deviation of 0.35 for search, +0.21 (1.19) for social and +0.13
(0.71) for TV, from the spend's weekly jitter and the season, and exact without both. A test's
standard error is read in time, from placebo readings that hold that error too: on the family's
first 100 pilot worlds of each environment a 95 % interval holds the true change for 0.920 to 0.958
of a channel's tests. The quarter at the status quo, the target of an arm's forecast, is the
market's, the geos' KPI summed
(:func:`~causaldyn_bench.scorecard.continuation.geo_shadow`); and a channel's returns are the
market's (:func:`~causaldyn_bench.scorecard.returns.true_geo_returns`).

chc's curves are JAX's, so the world is drawn at double precision whatever the caller's setting:
its exports are the same bits either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import jax
import numpy as np
from chc.geo_world import GeoMediaMix, GeoMediaWorld, MediaChannel
from chc.response import Hill, Tanh

from causaldyn_bench.endogenous_mmm import YEAR
from causaldyn_bench.mmm_decision import PLANNED
from causaldyn_bench.scorecard.continuation import GeoShadow, geo_shadow
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.ladder import geo_rung
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.returns import true_geo_returns
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored, stream
from causaldyn_bench.scorecard.truth import geo_cells, geo_quarter, oracle

FAMILY = 8
GEOS = 10
WEEKS = 104
SPREADS = (0.2, 0.6)  # the range each channel's spread of the geos' effects is drawn from
BASE = 19.0  # a geo's base per head a week, at the median
NOISE = 230.0  # the standard deviation of one head's KPI a week
POLICIES: MappingProxyType[str, float] = MappingProxyType({"policy-0": 0.0, "policy-0.5": 0.5})


def mix(policy: float, seed: int, geos: int = GEOS) -> GeoMediaMix:
    """The mix world ``seed`` is drawn from, its spreads drawn from the family's stream; ``geos``
    other than ten for a test alone."""
    spreads = rng(FAMILY, Role.PARAMETERS, seed).uniform(*SPREADS, 3)
    with jax.enable_x64(True):
        channels = (
            MediaChannel(
                "search", Tanh(2.0), 3.0, spread=float(spreads[0]), retention=0.3, spend=1.5
            ),
            MediaChannel(
                "social", Hill(1.0, 1.0), 1.5, spread=float(spreads[1]), retention=0.5, spend=0.8
            ),
            MediaChannel(
                "tv",
                Tanh(4.0),
                2.0,
                spread=float(spreads[2]),
                retention=0.7,
                spend=2.0,
                national=True,
            ),
        )
    return GeoMediaMix(
        channels, geos=geos, weeks=WEEKS, base=BASE, noise=NOISE, policy=policy, length=8
    )


@dataclass(frozen=True, eq=False)
class World:
    """A geo world as the family draws it: its history and its quarter at the status quo."""

    environment: str
    seed: int
    history: GeoMediaWorld
    shadow: GeoShadow


def drawn(environment: str, seed: int, generator: GeoMediaMix) -> World:
    """The world ``generator`` draws from ``seed``, at double precision, and its quarter."""
    with jax.enable_x64(True):
        history = generator.draw(seed)
    return World(environment, seed, history, geo_shadow(history, FAMILY, seed))


def window(history: GeoMediaWorld) -> tuple[int, int]:
    """The history's last year, its first and last weeks numbered from 1."""
    weeks = history.spend.shape[1]
    return (weeks - YEAR + 1, weeks)


class Geo:
    """Family 8."""

    name = "geo"
    stream = stream(FAMILY)
    labels = (
        "Each geo's effect on a channel is drawn from the hierarchy chc's geo model takes: "
        "log-normal about the channel's national median, independent across geos and channels.",
        "A geo's lift is its population times a curve of its spend per head.",
        "Every arm is told the kernel's length.",
        "National media is allotted to the geos by population, so outside its tests every geo "
        "spends the same on it a head.",
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(POLICIES)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(POLICIES)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in POLICIES:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        return drawn(environment, seed, mix(POLICIES[environment], seed))

    def observe(self, world: World, k: int) -> Observation:
        """All an arm reads of ``world`` at rung ``k``: each geo's KPI and spend, each geo's share
        of the population, the channels bought nationally, the quarter's budget and each cell's box
        and status quo, last year's weeks as the window a return is read over, and the rung's geo
        tests. The world has no controls."""
        history = world.history
        quarter = geo_quarter(history)
        return Observation(
            family=FAMILY,
            environment=world.environment,
            seed=world.seed,
            k=k,
            channels=tuple(channel.name for channel in history.mix.channels),
            sales=np.ascontiguousarray(history.kpi.T),
            spend=np.ascontiguousarray(history.spend.transpose(1, 0, 2)),
            controls={},
            future_controls={},
            kernel_length=history.mix.length,
            planned=PLANNED,
            budget=quarter.budget,
            lower=quarter.lower,
            upper=quarter.upper,
            status_quo=quarter.status_quo,
            roi_window=window(history),
            lift=geo_rung(history, FAMILY, world.seed, k),
            population=history.shares,
            national=tuple(c.name for c in history.mix.channels if c.national),
        )

    def truth(self, world: World) -> Truth:
        """The best plan over the world's cells, the market's quarter at the status quo, and each
        channel's returns on last year's spend; the effect holds still, so there is one path."""
        history = world.history
        quarter = geo_quarter(history)
        cells = geo_cells(history, quarter)
        return Truth(
            quarter=quarter,
            cells=cells,
            best=oracle(cells, quarter),
            realised=None,
            hindsight=None,
            target=world.shadow.sales.sum(axis=1),
            scale=float(np.mean(history.kpi.sum(axis=0))),
            returns=true_geo_returns(history, window(history)),
        )


GEO = Geo()

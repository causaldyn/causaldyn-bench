"""The ladder of lift tests: how many of Track M v2's go-dark tests each channel gets.

At rung ``k`` each channel gets the latest ``k`` of his four tests, from weeks 20, 55, 100 and 140,
each channel's in a treated universe of its own whose noise is drawn from ``1000 seed + c`` for the
``c``-th channel, as the budgets run drew it. A universe's noise is one draw over the whole history
whichever weeks go dark, and a test's carryover dies out before the next test's readout opens, so
the rungs nest under common random numbers: a test's readout is the same at every rung that holds
it. Rung 0 has no tests; rung 4 is the budgets run's experiments, bit for bit.

The tests read the world's own geo test (:meth:`MediaMixWorld.geo_test`), so where a family's world
moves its effect, its tests read the moved effect.

A rung's tests run in universes of their own: the history an arm reads holds neither their dark
weeks nor the sales those moved, so a test and the history read different data. An *overlapping*
rung (:func:`overlap`) runs one channel's tests in the market whose history an arm reads: each
test's readout weeks are weeks of that history, its spend there is the test's, and its sales hold
the sales the test moved. The tests are the rung's own, read from the same universes, and the
history is built from those universes at the share of the market the tests ran in:

* ``share=1``: the market ran them. The history is the treated universe, its noise in every week.
  A test's gap, the treated universe less the control, holds the treated noise over the readout,
  and so does the history's sales there: an arm that reads both reads that noise twice.
* ``share=1/2``: half the market ran them and half did not. The history is the sum of the two
  universes' halves. Their noises, of opposite signs, cancel in it, so the history holds the
  tests' change in spend and their response but none of the gaps' noise.

A geo world (:func:`geo_rung`, a world :mod:`chc.geo_world` drew) runs its tests as geo
experiments: each test takes alternate geos of the ring dark for the channel over its four weeks,
the even geos in the first and third test, the odd ones in the second and fourth, and is read by
difference-in-differences against the other geos. Its four tests start in weeks 20, 44, 68 and 92
of the 104, 24 weeks apart, the last one's readout closing in week 103. Each channel's tests run in
a universe of their own: the world's base and spend, the channel dark in the tested geos and weeks,
and fresh noise of the world's size in every geo and week, drawn from the family's stream
``(100 + f, 2, seed, c + 1)`` for the ``c``-th channel. The history holds none of it. A test's
carryover dies out before the next one's readout opens, as on Track M v2, so the rungs nest under
common random numbers here too.

The reading scales the other geos' KPI to the tested geos' by the ratio of the two groups' means
over the four weeks before the test, and sums the tested geos' KPI less that over the dark weeks and
the cooldown: a difference-in-differences whose groups' trends are proportional rather than
parallel. Where every geo's KPI holds still outside the test the reading is exact. A drawn world's
does not: each geo's spend jitters week by week, which no pre-period reads, and the season
multiplies the base alone, so the two groups' KPI do not move in proportion; the reading's error,
beside its noise, is what is left of both. It beat the difference in KPI a head, the other geos'
scaled by population, on the geo family's pilots.
Its standard error is the weekly noise of the difference, read from the four weeks before every
test of the channel, times ``sqrt(Q + Q^2 / P)`` over the dark weeks, ``Q`` the readout's twelve
weeks from the first dark one and ``P`` its four before, the second term the ratio's error
(Maxima). Controls next to tested geos receive what spills from them, as in practice.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import numpy as np
from chc.geo_world import GeoMediaWorld
from chc.lift import LiftTest

from causaldyn_bench.budget_regret import LiftRows
from causaldyn_bench.endogenous_mmm import Market, MediaMixWorld, Series
from causaldyn_bench.lift_calibration import (
    COOLDOWN,
    NOISE,
    PRE,
    STARTS,
    TEST,
    Setting,
    lift_tests,
)
from causaldyn_bench.scorecard.seeds import Role, rng

RUNGS = tuple(range(len(STARTS) + 1))  # the tests a channel may get: none to all of his four
SHARES = (0.5, 1.0)  # the shares of the market an overlapping rung's tests may run in
GEO_STARTS = (20, 44, 68, 92)  # a geo world's tests' first dark weeks, of its 104


def starts(k: int) -> tuple[int, ...]:
    """The first weeks of the latest ``k`` of his tests."""
    if k not in RUNGS:
        raise ValueError(f"a rung holds 0 to {len(STARTS)} tests a channel, not {k}")
    return STARTS[len(STARTS) - k :]


def rung(world: Market, seed: int, k: int) -> dict[str, tuple[LiftTest, ...]]:
    """Rung ``k``'s tests on ``world``, drawn from ``seed``: each channel's latest ``k``."""
    weeks = starts(k)
    if not weeks:
        return {}
    return {
        name: lift_tests(world, Setting(name, starts=weeks), 1000 * seed + c)
        for c, name in enumerate(world.channels)
    }


@dataclass(frozen=True, eq=False)
class Overlap:
    """Rung ``k``'s tests of one channel, run in the market an arm reads (:func:`overlap`).

    ``history`` is the world with the channel's spend as the market ran it, and its media and sales
    as they followed; ``tests`` are the channel's tests as :func:`rung` reads them; ``planned`` is
    the channel's spend as planned in every week, the world's own; ``share`` is the share of the
    market the tests ran in."""

    history: MediaMixWorld
    tests: dict[str, tuple[LiftTest, ...]]
    planned: Series
    share: float


def overlap(
    world: MediaMixWorld,
    seed: int,
    k: int,
    channel: str,
    share: float = 1.0,
    noise_share: float = NOISE,
) -> Overlap:
    """Rung ``k``'s tests of ``channel`` on ``world``, drawn from ``seed``, run in ``share`` of the
    market whose history an arm reads; see the module. Each universe's noise is ``noise_share``
    of mean weekly sales, the ladder's by default.

    Raises:
        ValueError: on rung 0, which runs no test, a rung off the ladder, a channel the world does
            not have, and a share other than 1/2 and 1.
    """
    weeks = starts(k)
    if not weeks:
        raise ValueError("rung 0 runs no test for a history to hold")
    if share not in SHARES:
        raise ValueError(f"an overlapping rung's tests run in a share of {SHARES}, not {share}")
    if channel not in world.channels:
        raise ValueError(f"the world has no channel {channel!r}: it has {world.channels}")
    c = world.channels.index(channel)
    setting = Setting(channel, starts=weeks, noise_share=noise_share)
    tests = lift_tests(world, setting, 1000 * seed + c)
    # the universes `lift_tests` read the tests from, drawn again from the same seed
    universes = world.geo_test(
        channel, weeks, test=TEST, noise_share=noise_share, seed=1000 * seed + c
    )
    spend = world.spend.copy()
    spend[:, c] = share * universes.spend_treated + (1.0 - share) * universes.spend_control
    media = world.media.copy()
    media[:, c] += share * universes.true_gap
    sales = share * universes.sales_treated + (1.0 - share) * universes.sales_control
    history = dataclasses.replace(world, spend=spend, media=media, sales=sales)
    return Overlap(history, {channel: tests}, universes.spend_control.copy(), share)


def dark_geos(geos: int, test: int) -> tuple[int, ...]:
    """The geos the ``test``-th of :data:`GEO_STARTS`, from 0, takes dark: every other geo of the
    ring, the even ones in an even test and the odd ones in an odd test."""
    return tuple(range(test % 2, geos, 2))


def geo_rung(world: GeoMediaWorld, family: int, seed: int, k: int) -> LiftRows:
    """Rung ``k``'s geo tests on ``world``, which family ``family`` drew from ``seed``: each
    channel's latest ``k`` of :data:`GEO_STARTS`, read by difference-in-differences; see the
    module. Each row names its tested geos; a test whose KPI did not fall is dropped and counted,
    as :func:`causaldyn_bench.budget_regret.lift_rows` drops it.

    Raises:
        ValueError: a rung off the ladder, a world with fewer than two geos, or one too short for a
            test's readout.
    """
    if k not in RUNGS:
        raise ValueError(f"a rung holds 0 to {len(GEO_STARTS)} tests a channel, not {k}")
    geos, weeks = world.mix.geos, world.spend.shape[1]
    if geos < 2:
        raise ValueError(f"a geo test reads its geos against others, and the world has {geos}")
    if GEO_STARTS[-1] + TEST + COOLDOWN - 1 > weeks:
        raise ValueError(f"the tests' readouts run past the world's {weeks} weeks")
    starts = GEO_STARTS[len(GEO_STARTS) - k :]
    after = TEST + COOLDOWN
    columns: dict[str, list] = {key: [] for key in ("channel", "start", "x", "change", "geos")}
    sigma: list[float] = []
    dropped = 0
    size = world.mix.noise * np.sqrt(world.population)[:, None]
    for c, channel in enumerate(world.mix.channels):
        noise = size * rng(family, Role.TESTS, seed, c + 1).standard_normal((geos, weeks))
        dark = world.spend.copy()
        for start in starts:
            dark[
                list(dark_geos(geos, GEO_STARTS.index(start))), start - 1 : start - 1 + TEST, c
            ] = 0.0
        with jax.enable_x64(True):
            kpi = world.base + np.asarray(world.lift(dark), dtype=float) + noise
        gaps = []
        for start in starts:
            treated = dark_geos(geos, GEO_STARTS.index(start))
            control = [g for g in range(geos) if g not in treated]
            first = start - 1
            pre, readout = slice(first - PRE, first), slice(first, first + after)
            ours, theirs = kpi[list(treated)].sum(axis=0), kpi[control].sum(axis=0)
            gap = ours - float(ours[pre].mean()) / float(theirs[pre].mean()) * theirs
            gaps.append(gap[pre])
            change = float(gap[readout].sum()) / TEST
            if not change < 0.0:
                dropped += 1
                continue
            columns["channel"].append(channel.name)
            columns["start"].append(start)
            columns["x"].append(
                float(world.spend[list(treated), first : first + TEST, c].sum(axis=0).mean())
            )
            columns["change"].append(change)
            columns["geos"].append(treated)
        if gaps:
            weekly = float(np.sqrt(np.mean([np.var(g, ddof=1) for g in gaps])))
            kept = len(columns["channel"]) - len(sigma)
            sigma += [weekly * float(np.sqrt(after + after**2 / PRE)) / TEST] * kept
    x = np.array(columns["x"], dtype=float)
    return LiftRows(
        channel=tuple(columns["channel"]),
        start=tuple(columns["start"]),
        x=x,
        delta_x=-x,
        delta_y=np.array(columns["change"], dtype=float),
        sigma=np.array(sigma, dtype=float),
        dropped=dropped,
        geos=tuple(columns["geos"]),
    )

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
and fresh noise of the world's size and law in every geo and week
(:func:`~causaldyn_bench.scorecard.geo_noise.kpi_noise`), drawn from the family's stream
``(100 + f, 2, seed, c + 1)`` for the ``c``-th channel. The history holds none of it. A test's
carryover dies out before the next one's readout opens, as on Track M v2, so the rungs nest under
common random numbers here too.

The reading is a difference-in-differences whose control is scaled by the slope of the two groups'
co-movement: each group's KPI less its mean over the four weeks before the test, the other geos'
times the slope, summed over the dark weeks and the cooldown. The slope is the least-squares slope
of the tested geos' KPI on the others' over the weeks clear of every test's window, its four weeks
before and twelve of readout, and of the adstock's first weeks; the same weeks at every
rung, so the rungs still nest. Where every geo's KPI holds still outside the test the reading is
exact, whatever the slope. A drawn world's does not: each geo's spend jitters week by week, which
no pre-period reads, and the season multiplies the base alone. Under a season the two groups' KPI
are not in proportion, `b_1 s_t + m_1` against `b_2 s_t + m_2`, but they are on a line of slope
`b_1 / b_2`, so the slope removes the season, and what is left of the reading's error, beside its
noise, is the jitter.

CHOICE: the reading first scaled the others' KPI by the ratio of the two groups' means over the four
weeks before the test, which is exact only where the groups' media shares are equal. Its error then
held the season, by how far the season moved over the readout from its level before it: on the
family's first 100 pilot worlds at policy 0, the tests of weeks 44 and 92 covered the true change in
0.918 of a channel's tests and those of weeks 20 and 68 in 0.970, and two other draws of the
tests' noise (0.923 against 0.982, 0.933 against 0.975), and the two came together without the
season. A placebo at the test's own phase of the season cannot mend it: the season's leak depends on
its level before the test as well as its slope, so a window half a year off does not match it, and
in 104 weeks no window shares the week-44 test's phase a year apart. Read by the slope, the tests of
odd and even weeks cover 0.925 and 0.952, 0.940 and 0.962, and 0.955 and 0.950 on those three draws
of the noise, and 0.943 and 0.937 at policy 0.5, where the ratio gave 0.910 and 0.955; the week-44
tests alone cover 0.903, 0.930 and 0.960 on the three draws, from 0.877 on the first. Controls
next to tested geos receive what spills from them, as in practice.

A reading's standard error is read in time, as an A/A test reads one. The same reading, of the same
geos, is taken at every other window of the channel's universe had it held no test (the world's base
and logged spend, and the universe's own noise), clear of the test's own four weeks before and
twelve of readout and of the adstock's first weeks. The standard error is the root mean square of
those placebo readings, widened by ``sqrt(nu / (nu - 2))``, a t's spread at their effective count
``nu`` over a normal's. Over independent weeks, two placebo readings ``h`` weeks apart correlate by
``rho(h)``, the reading's weights (``-Q/P`` on the ``P`` weeks before, 1 on the ``Q`` of readout)
correlated with themselves ``h`` weeks on, so their mean square holds
``nu = n^2 / sum rho(|f - f'|)^2`` independent squares (both checked in Maxima).

The noise alone, ``sqrt(Q + Q^2 / P)`` times the difference's weekly noise, the second term the
pre-period's error, misses two parts of a reading's error. The first is the KPI's movement that is
not on the groups' line: each geo's spend jitter carried by the adstock, and at policy 0.5 the
spend following the season. The four weeks before a test do not read it. The second is the noise's
own variance: ``A + c^2 B`` for a test at the slope ``c``, ``A`` and ``B`` the tested and the other
geos' summed variances, the even and the odd geos' trading places from one test to the next; one
weekly noise pooled over a channel's tests fits neither. A placebo reading holds both.

An *understated* rung (:func:`understated`) is a rung whose tests report their standard error
``factor`` times too small, as an analysis does that reads a geo's noise as independent from week
to week where it carries over: under a first-order autoregression of coefficient ``phi`` a sum over
12 weeks has ``1 + 2 sum_{k<12} (1 - k/12) phi^k`` times the variance of 12 independent weeks, so
such an analysis understates its error by a factor of 1.63 at ``phi = 0.5`` and 2.43 at
``phi = 0.8`` (Maxima). Its readings are the rung's own; a 95 % interval on its standard error holds
the true change about as often as one of ``1.96 / factor`` of the rung's standard errors does,
``2 Phi(1.96 / factor) - 1``: 0.67 at a factor of 2 and 0.49 at 3. It tests how an arm reads a
test whose error is understated.
"""

from __future__ import annotations

import dataclasses
import math
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
from causaldyn_bench.scorecard.geo_noise import kpi_noise
from causaldyn_bench.scorecard.seeds import Role, rng

RUNGS = tuple(range(len(STARTS) + 1))  # the tests a channel may get: none to all of his four
UNDERSTATED = (2.0, 3.0)  # the factors an understated rung's tests understate their error by
SHARES = (0.5, 1.0)  # the shares of the market an overlapping rung's tests may run in
GEO_STARTS = (20, 44, 68, 92)  # a geo world's tests' first dark weeks, of its 104
# a geo reading's weights over its weeks before and its readout, and their correlation with
# themselves h weeks on, 0 from P + Q
_WEIGHTS = np.concatenate([np.full(PRE, -(TEST + COOLDOWN) / PRE), np.ones(TEST + COOLDOWN)])
_RHO = np.append(
    [float(_WEIGHTS[: _WEIGHTS.size - h] @ _WEIGHTS[h:]) for h in range(_WEIGHTS.size)], 0.0
) / float(_WEIGHTS @ _WEIGHTS)


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


def understated(rows: LiftRows, factor: float) -> LiftRows:
    """``rows`` as an analysis reports them that understates each test's error ``factor`` times:
    each standard error the rung's own over ``factor``, every other column the rung's own; see the
    module.

    Raises:
        ValueError: on a factor not finite or below 1, which would overstate the error.
    """
    if not (math.isfinite(factor) and factor >= 1.0):
        raise ValueError(
            f"a rung understates its tests' error by a finite factor of 1 or more, not {factor}"
        )
    return dataclasses.replace(rows, sigma=rows.sigma / factor)


def dark_geos(geos: int, test: int) -> tuple[int, ...]:
    """The geos the ``test``-th of :data:`GEO_STARTS`, from 0, takes dark: every other geo of the
    ring, the even ones in an even test and the odd ones in an odd test."""
    return tuple(range(test % 2, geos, 2))


def _geo_gaps(ours: Series, theirs: Series, first: int, slope: float) -> Series:
    """The gap a geo test reads in each week of its dark weeks and cooldown from its tested geos'
    KPI ``ours`` and the others' ``theirs``, its first dark week ``first`` counted from 0: each
    group's KPI less its mean over the ``PRE`` weeks before, the others' times ``slope``."""
    pre = slice(first - PRE, first)
    gap = (ours - float(ours[pre].mean())) - slope * (theirs - float(theirs[pre].mean()))
    return gap[first : first + TEST + COOLDOWN]


def _geo_reading(ours: Series, theirs: Series, first: int, slope: float) -> float:
    """The change a geo test reads: its gaps (:func:`_geo_gaps`) summed over the dark weeks and the
    cooldown, a dark week."""
    return float(_geo_gaps(ours, theirs, first, slope).sum()) / TEST


def _slope(ours: Series, theirs: Series, first: int, length: int) -> float:
    """The least-squares slope of ``ours`` on ``theirs`` over the weeks clear of the adstock's first
    ``length - 1``, of every test's window in :data:`GEO_STARTS` and of the test's own from week
    ``first``: its ``PRE`` weeks before and its dark weeks and cooldown. Where ``theirs`` does not
    move over them, 0: the least-squares solution of least norm."""
    clear = np.ones(ours.size, dtype=bool)
    clear[: length - 1] = False
    for start in (*(s - 1 for s in GEO_STARTS), first):
        clear[max(start - PRE, 0) : start + TEST + COOLDOWN] = False
    x, y = theirs[clear], ours[clear]
    (slope,), *_ = np.linalg.lstsq((x - x.mean())[:, None], y - y.mean())
    return float(slope)


def placebo_error(kpi: Series, treated: tuple[int, ...], first: int, length: int) -> float:
    """The standard error of the reading of a test that took ``treated`` dark from week ``first``,
    counted from 0, read in time on ``kpi``, ``(geos, weeks)``, the channel's universe had it held
    no test: the root mean square of the same reading, at the test's slope, at every window clear
    of the test's own and of the adstock's first ``length - 1`` weeks, widened by
    ``sqrt(nu / (nu - 2))`` at their effective count ``nu``; see the module.

    Raises:
        ValueError: on a universe too short for a t's spread, two effective placebo readings or
            fewer.
    """
    others = [g for g in range(kpi.shape[0]) if g not in treated]
    ours, theirs = kpi[list(treated)].sum(axis=0), kpi[others].sum(axis=0)
    slope = _slope(ours, theirs, first, length)
    after = TEST + COOLDOWN
    firsts = np.array(
        [
            f
            for f in range(length - 1 + PRE, kpi.shape[1] - after + 1)
            if f + after <= first - PRE or f - PRE >= first + after
        ],
        dtype=np.int64,
    )
    readings = np.array([_geo_reading(ours, theirs, int(f), slope) for f in firsts])
    apart = np.minimum(np.abs(firsts[:, None] - firsts[None, :]), _RHO.size - 1)
    nu = firsts.size**2 / float(np.sum(_RHO[apart] ** 2)) if firsts.size else 0.0
    if not nu > 2.0:
        raise ValueError(f"a test's placebo readings count {nu:.3g}, too few for a t's spread")
    return float(np.sqrt(np.mean(readings**2) * nu / (nu - 2.0)))


def geo_rung(world: GeoMediaWorld, family: int, seed: int, k: int) -> LiftRows:
    """Rung ``k``'s geo tests on ``world``, which family ``family`` drew from ``seed``: each
    channel's latest ``k`` of :data:`GEO_STARTS`, read by difference-in-differences; see the
    module. Each row names its tested geos. Every test is a row, whichever way its KPI moved, as
    :func:`causaldyn_bench.budget_regret.lift_rows` keeps it. A test's standard error is read in
    time (:func:`placebo_error`), at the slope its reading reads: its weeks are clear of every
    test, so the channel's universe and the universe had it held no test agree on it. A row's gaps
    are its reading week by week (:func:`_geo_gaps`); every week's is read against the same
    pre-period means, so their errors are correlated across the weeks.

    Raises:
        ValueError: a rung off the ladder, a world with fewer than two geos, or one too short for a
            test's readout or its placebo readings.
    """
    if k not in RUNGS:
        raise ValueError(f"a rung holds 0 to {len(GEO_STARTS)} tests a channel, not {k}")
    geos, weeks = world.mix.geos, world.spend.shape[1]
    if geos < 2:
        raise ValueError(f"a geo test reads its geos against others, and the world has {geos}")
    if GEO_STARTS[-1] + TEST + COOLDOWN - 1 > weeks:
        raise ValueError(f"the tests' readouts run past the world's {weeks} weeks")
    starts = GEO_STARTS[len(GEO_STARTS) - k :]
    columns: dict[str, list] = {
        key: [] for key in ("channel", "start", "x", "change", "sigma", "geos", "gaps")
    }
    with jax.enable_x64(True):
        logged = world.base + np.asarray(world.lift(world.spend), dtype=float)
    for c, channel in enumerate(world.mix.channels):
        shocks = rng(family, Role.TESTS, seed, c + 1).standard_normal((geos, weeks))
        dark = world.spend.copy()
        for start in starts:
            dark[
                list(dark_geos(geos, GEO_STARTS.index(start))), start - 1 : start - 1 + TEST, c
            ] = 0.0
        with jax.enable_x64(True):
            level = world.base + np.asarray(world.lift(dark), dtype=float)
        noise = kpi_noise(world, level, shocks)
        kpi = level + noise
        for start in starts:
            treated = dark_geos(geos, GEO_STARTS.index(start))
            control = [g for g in range(geos) if g not in treated]
            first = start - 1
            ours, theirs = kpi[list(treated)].sum(axis=0), kpi[control].sum(axis=0)
            slope = _slope(ours, theirs, first, world.mix.length)
            columns["channel"].append(channel.name)
            columns["start"].append(start)
            columns["x"].append(
                float(world.spend[list(treated), first : first + TEST, c].sum(axis=0).mean())
            )
            columns["change"].append(_geo_reading(ours, theirs, first, slope))
            columns["gaps"].append(_geo_gaps(ours, theirs, first, slope))
            columns["sigma"].append(placebo_error(logged + noise, treated, first, world.mix.length))
            columns["geos"].append(treated)
    x = np.array(columns["x"], dtype=float)
    return LiftRows(
        channel=tuple(columns["channel"]),
        start=tuple(columns["start"]),
        x=x,
        delta_x=-x,
        delta_y=np.array(columns["change"], dtype=float),
        sigma=np.array(columns["sigma"], dtype=float),
        dropped=0,
        geos=tuple(columns["geos"]),
        gaps=np.array(columns["gaps"], dtype=float).reshape(-1, TEST + COOLDOWN),
    )

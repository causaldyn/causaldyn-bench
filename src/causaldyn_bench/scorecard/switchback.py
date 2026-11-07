"""A switchback test of a geo world's national channel, drawn on the world from streams of its own,
and its truth.

A channel bought nationally cannot be cut by a geo test without cutting the national buy, so a
switchback test randomises time instead (Bojinov, Simchi-Levi and Zhao 2023, "Design and analysis of
switchback experiments", Management Science 69). On a world of the geo families (:mod:`.geo`,
:mod:`.spillover`, :mod:`.few_geos`) the test cuts the history's last :data:`BLOCKS` blocks of the
kernel's length, weeks 9 to 104 of 104 in blocks of eight, and turns the national channel off in
every geo at once in half of them, in an order drawn uniformly from every such arrangement, 924 of
them; outside the blocks off its spend is the history's. A block as long as the kernel lets a block
on reach its steady state, and the kernel's length is in every export, so a reader chooses its own
washout. The test runs in a universe of its own, as the ladder's tests do: the world's base, the
test's spend, and fresh noise of the world's size and law in every geo and week
(:func:`~causaldyn_bench.scorecard.geo_noise.kpi_noise`). The history holds none of it.

**Streams.** The blocks' order is drawn from the family's stream of tests at the further index
:data:`ORDER`, ``(100 + f, 2, seed, 1001)``, and the universe's noise at :data:`NOISE`, ``(100 + f,
2, seed, 1000)``. The ladder's tests read the indices 1 to the count of channels and nothing else
reads the stream, so the test moves no world, no ladder test and no quarter the families drew
before it.

**The truth** (:func:`lift`) is what the weeks on raised the KPI over the channel off through the
whole test, summed over the geos: the world's lift at the test's spend less its lift with the
channel off in every week of the test, from the test's first week through the kernel's last week
after it, nothing spent past the history. It holds the carry-out of the last blocks on past the
history, which a reader of a week on's steady state times the weeks on counts too.

**The steady truth** (:func:`steady`) is the lift a week on gains as a reader with a washout of
half a block reads it: the mean, over the weeks on that such a reading keeps
(:meth:`Design.kept`), of what the test's spend raised the KPI that week over the channel off
through the test, summed over the geos. A week whose kernel's lags all fall in weeks on holds the
channel's whole steady-state lift; the kernel is a block long, longer than the washout, so the
first weeks kept after a switch miss the lags that fall in the block before and sit below it. A
kept week off still carries the lift of those lags, so a reading of the mean KPI over the kept
weeks on less the mean over the kept weeks off reads this less their mean, beside its noise and
what the rest of the KPI differs by between those weeks. That difference's mean over the
arrangements is nought: an arrangement and its complement switch in the same blocks and keep the
same weeks.

An export of version 4 (:mod:`.observe`) carries the test: its design and its universe
(:class:`~causaldyn_bench.scorecard.observe.SwitchbackTest`); a geo family's truth carries its lift
and its steady truth (:attr:`~causaldyn_bench.scorecard.family.Truth.switchback` and
:attr:`~causaldyn_bench.scorecard.family.Truth.switchback_steady`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import numpy as np
from chc.geo_world import GeoMediaWorld

from causaldyn_bench.endogenous_mmm import Series
from causaldyn_bench.scorecard.geo_noise import kpi_noise
from causaldyn_bench.scorecard.seeds import Role, rng

BLOCKS = 12  # the test's blocks, each the kernel's length
NOISE = 1000  # the universe's noise's further index on the family's stream of tests
ORDER = 1001  # the blocks' order's further index on it


@dataclass(frozen=True)
class Design:
    """A switchback test: the ``channel``-th channel, from week ``first``, numbered from 1, in
    blocks of ``block`` weeks, on in the blocks where ``on`` holds and off in the others."""

    channel: int
    first: int
    block: int
    on: tuple[bool, ...]

    def state(self, weeks: int) -> np.ndarray:
        """``(weeks,)``: whether the channel runs in each week of a history of ``weeks``, as logged
        before the test and in its blocks on."""
        state = np.ones(weeks, dtype=bool)
        state[self.first - 1 : self.first - 1 + len(self.on) * self.block] = np.repeat(
            self.on, self.block
        )
        return state

    def kept(self, weeks: int, washout: int) -> np.ndarray:
        """``(weeks,)``: whether a reading with a washout of ``washout`` weeks reads each week of a
        history of ``weeks``: every week of the test but the first ``washout`` of its first block,
        whose state before it the reading does not know, and of every block after a switch; a
        block in the state of the one before it is read whole, and no week outside the test is.

        Raises:
            ValueError: a washout below 0, or one that leaves nothing of a block.
        """
        if not 0 <= washout < self.block:
            raise ValueError(f"a washout is 0 to {self.block - 1} weeks of a block, not {washout}")
        on = np.array(self.on)
        switched = np.ones(on.size, dtype=bool)
        switched[1:] = on[1:] != on[:-1]
        read = np.ones((on.size, self.block), dtype=bool)
        read[:, :washout] = ~switched[:, None]
        kept = np.zeros(weeks, dtype=bool)
        kept[self.first - 1 : self.first - 1 + on.size * self.block] = read.ravel()
        return kept


def design(world: GeoMediaWorld, family: int, seed: int) -> Design:
    """The switchback test of ``world``, which family ``family`` drew from ``seed``: its national
    channel over the history's last :data:`BLOCKS` blocks of the kernel's length, half of them on in
    an order drawn from :data:`ORDER`; see the module.

    Raises:
        ValueError: a world with no national channel or several, or one shorter than the blocks.
    """
    national = [c for c, channel in enumerate(world.mix.channels) if channel.national]
    if len(national) != 1:
        raise ValueError(f"a switchback test cuts one national channel; the world has {national}")
    block, weeks = world.mix.length, world.spend.shape[1]
    first = weeks - BLOCKS * block + 1
    if first < 1:
        raise ValueError(f"{BLOCKS} blocks of {block} weeks run past a history of {weeks}")
    on = np.zeros(BLOCKS, dtype=bool)
    on[rng(family, Role.TESTS, seed, ORDER).permutation(BLOCKS)[: BLOCKS // 2]] = True
    return Design(national[0], first, block, tuple(bool(b) for b in on))


def tested(world: GeoMediaWorld, test: Design) -> Series:
    """``(geos, weeks, channels)``: the world's spend with ``test``'s channel off in its blocks
    off."""
    spend = world.spend.copy()
    spend[:, ~test.state(spend.shape[1]), test.channel] = 0.0
    return spend


def universe(world: GeoMediaWorld, family: int, seed: int, test: Design) -> tuple[Series, Series]:
    """The universe ``test`` runs in on ``world``, which family ``family`` drew from ``seed``: its
    KPI, ``(geos, weeks)``, the world's base and its lift at the test's spend with noise from
    :data:`NOISE`, and that spend, ``(geos, weeks, channels)``."""
    spend = tested(world, test)
    shocks = rng(family, Role.TESTS, seed, NOISE).standard_normal(spend.shape[:2])
    with jax.enable_x64(True):
        level = world.base + np.asarray(world.lift(spend), dtype=float)
    return level + kpi_noise(world, level, shocks), spend


def lift(world: GeoMediaWorld, test: Design) -> float:
    """What ``test``'s weeks on raised the KPI over its channel off through the whole test, summed
    over the geos and over the weeks from its first through the kernel's last after the history,
    nothing spent past it; see the module.

    Raises:
        ValueError: the world's lifts drift, whose path past the history is a forecast.
    """
    return float(_gained(world, test).sum())


def steady(world: GeoMediaWorld, test: Design) -> float:
    """The lift a week on of ``test`` gains as a reader with a washout of ``test.block // 2``
    reads it: the mean, over the weeks on :meth:`Design.kept` keeps at that washout, of what the
    test's spend raised the KPI that week over its channel off through the test, summed over the
    geos; see the module.

    Raises:
        ValueError: the world's lifts drift.
    """
    weeks = world.spend.shape[1]
    read = test.state(weeks) & test.kept(weeks, test.block // 2)
    return float(_gained(world, test)[:, :weeks].sum(axis=0)[read].mean())


def _gained(world: GeoMediaWorld, test: Design) -> Series:
    """``(geos, weeks + length - 1)``: what ``test``'s spend raised each geo's KPI each week over
    its channel off from the test's first week, through the kernel's last week after the history."""
    if world.mix.drift > 0.0:
        raise ValueError("a geo world whose lifts drift has no lift past its history here")
    geos, _, channels = world.spend.shape
    off = world.spend.copy()
    off[:, test.first - 1 :, test.channel] = 0.0
    after = np.zeros((geos, world.mix.length - 1, channels))
    on, off = (np.concatenate([spend, after], axis=1) for spend in (tested(world, test), off))
    longer = dataclasses.replace(world, spend=on, drift=np.ones(on.shape[1:]))
    with jax.enable_x64(True):
        return np.asarray(longer.lift(on), dtype=float) - np.asarray(longer.lift(off), dtype=float)

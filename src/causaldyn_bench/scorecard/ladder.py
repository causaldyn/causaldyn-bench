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
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from chc.lift import LiftTest

from causaldyn_bench.endogenous_mmm import Market, MediaMixWorld, Series
from causaldyn_bench.lift_calibration import NOISE, STARTS, TEST, Setting, lift_tests

RUNGS = tuple(range(len(STARTS) + 1))  # the tests a channel may get: none to all of his four
SHARES = (0.5, 1.0)  # the shares of the market an overlapping rung's tests may run in


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

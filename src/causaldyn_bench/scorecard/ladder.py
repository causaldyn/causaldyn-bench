"""The ladder of lift tests: how many of Track M v2's go-dark tests each channel gets.

At rung ``k`` each channel gets the latest ``k`` of his four tests, from weeks 20, 55, 100 and 140,
each channel's in a treated universe of its own whose noise is drawn from ``1000 seed + c`` for the
``c``-th channel, as the budgets run drew it. A universe's noise is one draw over the whole history
whichever weeks go dark, and a test's carryover dies out before the next test's readout opens, so
the rungs nest under common random numbers: a test's readout is the same at every rung that holds
it. Rung 0 has no tests; rung 4 is the budgets run's experiments, bit for bit.

The tests read the world's own geo test (:meth:`MediaMixWorld.geo_test`), so where a family's world
moves its effect, its tests read the moved effect.
"""

from __future__ import annotations

from chc.lift import LiftTest

from causaldyn_bench.endogenous_mmm import Market
from causaldyn_bench.lift_calibration import STARTS, Setting, lift_tests

RUNGS = tuple(range(len(STARTS) + 1))  # the tests a channel may get: none to all of his four


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

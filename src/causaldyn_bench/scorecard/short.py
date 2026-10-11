"""Family 7: Track M v2's drawn worlds of which an arm reads only the last ``L`` weeks.

Each environment is one length ``L``: 52, 78 and 104 of the history's 156 weeks. The world is the
drawn world, and so are its quarter, box, budget, status quo, best plan and returns: the arm reads
less of it, not another. The window a return is read over is the last year, inside every ``L``.

A lift test counts only where its readout, from four weeks before its first dark week to eight
after its last, falls inside the ``L`` weeks. A rung holds each channel's latest ``k`` tests, so
the deepest rung whose tests all count, ``k_max(L)`` (:data:`DEEPEST`), holds every test that
does: 1, 2 and 2 at ``L`` = 52, 78 and 104, since his tests start in weeks 20, 55, 100 and 140,
and the readout of the one in week 55 opens in week 51, two weeks before the last 104 do. A deeper
rung is refused. The rows' first dark weeks are numbered from the first week the arm reads, as
its history's are, and so is the window of the world's returns; the returns are the world's, the
carryover of the weeks the arm does not read among them.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

from causaldyn_bench.endogenous_mmm import EndogenousMediaMix
from causaldyn_bench.lift_calibration import COOLDOWN, PRE, STARTS, TEST
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import pilot, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, settled, truth

FAMILY = 7
WEEKS = EndogenousMediaMix().weeks  # the drawn world's history
LENGTHS: MappingProxyType[str, int] = MappingProxyType({"L-52": 52, "L-78": 78, "L-104": 104})


def deepest(length: int, weeks: int = WEEKS) -> int:
    """``k_max``: how many of his tests read out inside the last ``length`` of ``weeks``."""
    first = weeks - length + 1
    return sum(start - PRE >= first and start + TEST + COOLDOWN - 1 <= weeks for start in STARTS)


DEEPEST: MappingProxyType[str, int] = MappingProxyType(
    {name: deepest(length) for name, length in LENGTHS.items()}
)


def shortened(full: Observation, length: int) -> Observation:
    """``full`` with its history, its controls and its return's window cut to their last
    ``length`` weeks, and its tests' first dark weeks numbered from the first of them."""
    cut = full.sales.shape[0] - length
    first, last = full.roi_window
    rows = full.lift
    return dataclasses.replace(
        full,
        sales=full.sales[cut:],
        spend=full.spend[cut:],
        controls={name: series[cut:] for name, series in full.controls.items()},
        roi_window=(first - cut, last - cut),
        lift=dataclasses.replace(rows, start=tuple(start - cut for start in rows.start)),
    )


class Short:
    """Family 7."""

    name = "short series"
    stream = stream(FAMILY)
    labels = (
        "Each world is a Track M v2 drawn world, whose media are PyMC-Marketing's default "
        "class, of which an arm reads the last weeks alone.",
        *TRACK_M2.labels[1:],
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(LENGTHS)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(LENGTHS)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in LENGTHS:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = drawn(seed)
        return settled(environment, seed, generator, generator.simulate(seed))

    def observe(self, world: World, k: int) -> Observation:
        """The rung's tests, of each channel's latest ``k`` up to ``k_max(L)``, and the last ``L``
        weeks of the world.

        Raises:
            ValueError: a rung deeper than ``k_max(L)``, whose earliest tests read out before the
                weeks the arm reads.
        """
        length = LENGTHS[world.environment]
        top = deepest(length, world.history.week.size)
        if k > top:
            raise ValueError(
                f"{top} of a channel's tests read out inside the last {length} weeks, not {k}"
            )
        return shortened(observation(FAMILY, world, k), length)

    def truth(self, world: World) -> Truth:
        """The drawn world's, the window of its returns numbered as the arm's weeks are."""
        full = truth(world)
        cut = world.history.week.size - LENGTHS[world.environment]
        first, last = full.returns.window
        returns = dataclasses.replace(full.returns, window=(first - cut, last - cut))
        return dataclasses.replace(full, returns=returns)


SHORT = Short()

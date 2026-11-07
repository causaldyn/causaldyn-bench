"""A family of worlds: one strategy object that draws a world, says what an arm reads of it, and
holds what the arm's plan is scored against.

Families differ in the shape of their worlds, a market's history, a geo panel or a black box, so
each is an object of its own behind one protocol rather than one generator grown by flags. A family
names its environments, each a setting of its mechanism, and for each the worlds its pilots and its
scored runs draw, in their seeded order (:mod:`.seeds`): a run's sample, and every look at it, is
a prefix of that order. Its labels say, in plain words, what its worlds hand some arm by
construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar

from causaldyn_bench.endogenous_mmm import Series, Vector
from causaldyn_bench.mmm_decision import Plan, Quarter
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.returns import Returns
from causaldyn_bench.scorecard.seeds import TRACK_M2
from causaldyn_bench.scorecard.truth import Cell

W = TypeVar("W")


@dataclass(frozen=True, eq=False)
class Truth:
    """What a plan for one world is scored against, and what else its family states of the world:
    each channel's share of its return that lasts through a brand measure, each lever's
    contribution, each channel's returns with a mediator held, and the lift of the switchback test
    its worlds carry, where the family states them."""

    quarter: Quarter
    cells: tuple[Cell, ...]  # on the effect path the best plan is made for, the law's expected one
    best: Plan
    realised: tuple[Cell, ...] | None  # on the path the quarter realised, where that is another
    hindsight: Plan | None  # the best plan on the realised path
    target: Series  # the quarter's sales at the status quo, an arm's forecast's target
    scale: float  # mean weekly sales over the history, the unit a forecast's score is read in
    returns: Returns  # each channel's ROI and mROI on the window an export names
    # each channel's share of its return that runs through the brand measure the family exports
    long_term: Vector | None = None
    # each lever's contribution over the window the returns are read on, against the level the
    # family states it at
    levers: Mapping[str, float] | None = None
    # each channel's ROI and mROI on that window with the mediator the family exports held
    direct: Returns | None = None
    # what the switchback test the family's worlds carry raised the KPI over its channel off, summed
    # over the geos, its carry-out past the history included (:mod:`.switchback`)
    switchback: float | None = None


class Family(Protocol[W]):
    """One family of worlds; ``W`` is the type of its worlds. A family is read, never written."""

    @property
    def name(self) -> str: ...

    @property
    def stream(self) -> int:
        """Its variates' stream, ``100 + f`` for family ``f``."""
        ...

    @property
    def labels(self) -> tuple[str, ...]:
        """What its worlds hand some arm by construction."""
        ...

    @property
    def pilots(self) -> Mapping[str, range]:
        """Each environment's pilot worlds, in their seeded order."""
        ...

    @property
    def scored(self) -> Mapping[str, range]:
        """Each environment's scored worlds, in their seeded order."""
        ...

    def world(self, environment: str, seed: int) -> W:
        """The world of ``environment`` drawn from ``seed``."""
        ...

    def observe(self, world: W, k: int) -> Observation:
        """All an arm reads of ``world`` at rung ``k`` of the ladder; its digest covers it."""
        ...

    def truth(self, world: W) -> Truth:
        """What a plan for ``world`` is scored against."""
        ...


def number(family: Family[W]) -> int:  # noqa: UP047 -- CI runs 3.11
    """The family's number, as its exports name it."""
    return family.stream - TRACK_M2


def index(  # noqa: UP047 -- CI runs 3.11
    family: Family[W], environment: str, seed: int, *, pilot: bool = False
) -> int:
    """The world's place in its environment's seeded order, from 0.

    Raises:
        ValueError: the family draws no such world.
    """
    samples = family.pilots if pilot else family.scored
    if environment not in samples:
        raise ValueError(f"{family.name} has no environment {environment!r}")
    if seed not in samples[environment]:
        kind = "pilot" if pilot else "scored"
        raise ValueError(f"{family.name}'s {environment} draws no {kind} world {seed}")
    return samples[environment].index(seed)

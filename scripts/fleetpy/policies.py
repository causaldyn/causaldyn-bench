"""Track R's pricing policies on FleetPy: each is ``policy(period, ledger) -> factor per zone``.

The status quo is a surge rule fixed before any scored run. A zone's factor rises from 1, the list
fare, by half for every request per idle vehicle above one: ``1 + (u - 1) / 2``, with ``u`` the
zone's requests over the last period per vehicle idle in it now, held to ``[1, 2]``. It never
discounts.

The logger adds a Gaussian dither to the status quo. The dither is drawn ahead for the whole day
from its own stream, so the draw a period gets does not depend on what ran before it.

Only numpy is needed here, so the bench's tests read this file without FleetPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

BOX = (0.6, 2.0)
PERIODS = 96


class Book(Protocol):
    """What a policy reads of the run's ledger."""

    zones: int
    idle: dict

    def closed(self, period: int) -> dict[str, np.ndarray]: ...


def utilisation(period: int, book: Book) -> np.ndarray:
    """Per zone, the last period's requests per vehicle idle now; 0 in the first period."""
    if period == 0:
        return np.zeros(book.zones)
    return book.closed(period - 1)["requests"] / np.maximum(book.idle[period], 1.0)


@dataclass
class Constant:
    factor: float

    def __call__(self, period: int, book: Book) -> np.ndarray:
        return np.full(book.zones, self.factor)


@dataclass
class Surge:
    def __call__(self, period: int, book: Book) -> np.ndarray:
        return np.clip(1.0 + (utilisation(period, book) - 1.0) / 2.0, 1.0, BOX[1])


@dataclass
class Dithered:
    """The status quo plus ``scale`` N(0, 1), held to the box. ``base`` and ``clipped`` keep, per
    period, the status quo's factor and where the dither left the box, for the log."""

    scale: float
    draws: np.ndarray  # (PERIODS, zones) standard normals
    status_quo: Surge = field(default_factory=Surge)
    base: dict = field(default_factory=dict)
    clipped: dict = field(default_factory=dict)

    def __call__(self, period: int, book: Book) -> np.ndarray:
        base = self.status_quo(period, book)
        wanted = base + self.scale * self.draws[period]
        self.base[period] = base
        self.clipped[period] = (wanted < BOX[0]) | (wanted > BOX[1])
        return np.clip(wanted, *BOX)


@dataclass
class Schedule:
    """A fixed factor per period and zone, ``(PERIODS, zones)``."""

    table: np.ndarray

    def __call__(self, period: int, book: Book) -> np.ndarray:
        return self.table[period]


@dataclass
class Switchback:
    """``arms[assignment[period]]`` decides the period."""

    arms: tuple
    assignment: np.ndarray  # (PERIODS,) indices into arms

    def __call__(self, period: int, book: Book) -> np.ndarray:
        return self.arms[int(self.assignment[period])](period, book)

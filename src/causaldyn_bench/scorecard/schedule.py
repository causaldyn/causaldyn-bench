"""A schedule scored against a world's truth: a spend in each week of the quarter on each cell, its
regret against the best schedule in the box at the budget.

A plan (:mod:`.truth`) holds one weekly spend a cell over the quarter's 13 weeks; a schedule gives
each week its own, as a flighting plan does. Its worth is what it returns over the quarter and the
kernel's tail, carryover in and out: a cell's adstock in each week of that horizon is its carry
plus the reach of each quarter week's spend (:attr:`Cell.unit
<causaldyn_bench.scorecard.truth.Cell.unit>`), so a schedule held flat is worth its plan. Each
week's spend on a cell is held within the cell's box, as a plan's weekly spend is, and the
schedule spends the quarter's budget over its weeks and cells; a plan is a schedule, so the best
schedule returns at least the best plan.

**The best schedule** (:func:`oracle`), where every cell is concave and none is a line, as
:func:`~causaldyn_bench.scorecard.truth.oracle` prices its plan: at a price ``mu`` a euro each cell
spends where each week's slope over the horizon meets ``mu``, or at an end of its box, a concave
programme over its 13 weeks solved by projected Newton (Bertsekas 1982, "Projected Newton methods
for optimization problems with simple constraints", SIAM J. Control Optim. 20(2)), each week's
curvature read as a central difference of the curve's slope; and ``mu`` is found by bisection where
the schedules spend the budget. The worth is concave, so it lies below its tangent plane: what the
best move along the gradient within the box at the budget gains, the Frank-Wolfe gap, bounds what
any schedule returns over the one found. The schedule is kept only where that bound is below
:data:`GAP` per euro of the budget. A cell that is not concave, or a line, is refused: its best
schedule is not a concave programme.

**A schedule's score** (:func:`score_schedule`): its regret per euro of the budget against the
best schedule, on the effect path the best plan is made for, and on the realised path against the
best schedule in hindsight where the quarter realised another; the best schedule's worth over the
best plan's per euro of the budget, the most a schedule gains over holding a rate; and how far the
schedule was moved into the box at the budget, where it left them. A schedule of one spend a
channel each week on a geo world is allotted to the cells week by week, as a plan is
(:meth:`GeoQuarter.allot <causaldyn_bench.scorecard.truth.GeoQuarter.allot>`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from causaldyn_bench.budget_regret import OUT_OF_PLAN, TIE
from causaldyn_bench.endogenous_mmm import Series
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.truth import Cell

GAP = 1e-10  # the best schedule's certified gap per euro of the budget, a tenth of a tie
STEP = 1e-4  # the central difference a week's curvature is read at, relative to its adstock
NEWTON = 100  # the Newton steps a cell's programme may take at one price
NEAR = 1e-9  # a Newton step's rise, of its cell's value, below which it is taken whole


@dataclass(frozen=True)
class Schedule:
    """The best schedule: each week's spend on each cell, ``(13, cells)``, its worth, the price a
    euro its weeks' slopes meet, and the certified gap: no schedule in the box at the budget returns
    more than ``worth + gap``."""

    weekly: Series
    worth: float
    price: float
    gap: float


@dataclass(frozen=True)
class ScheduleScore:
    """A schedule scored: its regret per euro of the budget against the best schedule on the path
    the best plan is made for, and on the realised path against the best schedule in hindsight
    where the quarter realised another; the best schedule's worth over the best plan's, per euro;
    and how far the schedule was moved into the box at the budget, where it left them."""

    regret: float
    realised: float | None
    edge: float
    moved: float | None


def _units(cells: Sequence[Cell]) -> list[Series]:
    units = [cell.unit for cell in cells]
    if any(unit is None for unit in units):
        raise ValueError("a schedule is scored on cells that state each quarter week's reach")
    return [np.asarray(unit, dtype=float) for unit in units if unit is not None]


def _cell_worth(cell: Cell, unit: Series, spends: Series) -> float:
    return float(np.sum(cell.effect * cell.curve.value(cell.carry + unit @ spends, cell.scale)))


def worth(cells: Sequence[Cell], schedule: ArrayLike) -> float:
    """What ``schedule``, ``(13, cells)``, returns over the quarter and the tail, carryover in and
    out.

    Raises:
        ValueError: a schedule of another shape, or a cell that does not state its reach week by
            week.
    """
    weekly = np.asarray(schedule, dtype=float)
    if weekly.shape != (PLANNED, len(cells)):
        raise ValueError(
            f"a schedule is {(PLANNED, len(cells))}, a week by a cell, not {weekly.shape}"
        )
    units = _units(cells)
    return sum(
        _cell_worth(c, u, weekly[:, i]) for i, (c, u) in enumerate(zip(cells, units, strict=True))
    )


def _slopes(cells: Sequence[Cell], units: Sequence[Series], weekly: Series) -> Series:
    """``(13, cells)``: what one more euro in each week on each cell returns."""
    out = np.empty_like(weekly)
    for i, (cell, unit) in enumerate(zip(cells, units, strict=True)):
        adstock = cell.carry + unit @ weekly[:, i]
        out[:, i] = unit.T @ (cell.effect * cell.curve.slope(adstock, cell.scale))
    return out


def gap(cells: Sequence[Cell], quarter: Quarter, schedule: ArrayLike) -> float:
    """What the best move from ``schedule`` along its slopes within the box at the budget gains:
    every week on every cell from its floor, the rest of the budget spent where the slope is
    steepest first. The worth is concave, so no schedule in the box at the budget returns more than
    ``schedule``'s worth and this."""
    weekly = np.asarray(schedule, dtype=float)
    slopes = _slopes(cells, _units(cells), weekly).ravel()
    low, high = (np.tile(bound, (PLANNED, 1)).ravel() for bound in (quarter.lower, quarter.upper))
    vertex = low.copy()
    rest = quarter.budget - float(low.sum())
    for k in np.argsort(-slopes, kind="stable"):
        if rest <= 0.0:
            break
        taken = min(float(high[k] - low[k]), rest)
        vertex[k] += taken
        rest -= taken
    return float(slopes @ (vertex - weekly.ravel()))


def _curvature(cell: Cell, adstock: Series) -> Series:
    step = STEP * np.maximum(np.abs(adstock), 1.0 / cell.scale)
    rise = cell.curve.slope(adstock + step, cell.scale) - cell.curve.slope(
        adstock - step, cell.scale
    )
    return rise / (2.0 * step)


def _cell_schedule(
    cell: Cell, unit: Series, low: float, high: float, price: float, start: Series
) -> Series:
    """The cell's best spends over the quarter's weeks at ``price`` a euro, each within ``[low,
    high]``: projected Newton from ``start``, each step searched back along its line while the
    rise it predicts is above :data:`NEAR` of the cell's value and taken whole below it, where the
    worth's quadratic model is exact to its rounding, until a step moves no spend."""

    def value(spends: Series) -> float:
        return _cell_worth(cell, unit, spends) - price * float(spends.sum())

    spends = np.clip(start, low, high)
    current = value(spends)
    for _ in range(NEWTON):
        adstock = cell.carry + unit @ spends
        slope = unit.T @ (cell.effect * cell.curve.slope(adstock, cell.scale)) - price
        free = ~(((spends <= low) & (slope <= 0.0)) | ((spends >= high) & (slope >= 0.0)))
        if not free.any():
            break
        reach = unit[:, free]
        hessian = reach.T @ ((cell.effect * _curvature(cell, adstock))[:, None] * reach)
        move = np.zeros_like(spends)
        move[free] = np.linalg.lstsq(hessian, -slope[free])[0]
        rise = float(slope @ move)
        # the worth is concave, so only a slope of nought, or its rounding, rises by none
        if not rise > 0.0:
            break
        trial = np.clip(spends + move, low, high)
        if rise > NEAR * max(abs(current), 1.0):
            size = 1.0
            while value(trial) < current + 1e-4 * float(slope @ (trial - spends)) and size > 1e-12:
                size /= 2.0
                trial = np.clip(spends + size * move, low, high)
        if np.array_equal(trial, spends):
            break
        spends, current = trial, value(trial)
    return spends


def _project(quarter: Quarter, weekly: Series) -> Series:
    """The schedule in the box at the budget nearest ``weekly``: every entry moved by one shift and
    clipped to its box, the shift found by bisection."""
    low, high = (np.broadcast_to(bound, weekly.shape) for bound in (quarter.lower, quarter.upper))

    def excess(shift: float) -> float:
        return float(np.clip(weekly - shift, low, high).sum() - quarter.budget)

    reach = float(np.max(np.abs(weekly)) + np.max(quarter.upper))
    shift = brentq(
        excess,
        -reach,
        reach,
        xtol=1e-12 * quarter.budget / weekly.size,
        rtol=4 * np.finfo(float).eps,
    )
    return np.clip(weekly - shift, low, high)


def oracle(cells: Sequence[Cell], quarter: Quarter) -> Schedule:
    """The best schedule in ``quarter``'s box at its budget on ``cells``; see the module.

    Raises:
        ValueError: a cell that does not state its reach week by week, one that is not concave or
            is a line, or a box that cannot spend the budget.
        RuntimeError: the schedule found is not certified to :data:`GAP` per euro of the budget.
    """
    units = _units(cells)
    if len(cells) != quarter.lower.size:
        raise ValueError(f"{len(cells)} cells against a box of {quarter.lower.size}")
    if not all(cell.concave and cell.bend != 0.0 for cell in cells):
        raise ValueError("the best schedule is solved on concave cells alone, none of them a line")
    if not PLANNED * quarter.lower.sum() <= quarter.budget <= PLANNED * quarter.upper.sum():
        raise ValueError(f"no schedule in the box spends {quarter.budget:.6g}")
    weekly = np.tile(quarter.status_quo, (PLANNED, 1)).astype(float)

    def excess(price: float) -> float:
        for i, (cell, unit) in enumerate(zip(cells, units, strict=True)):
            weekly[:, i] = _cell_schedule(
                cell, unit, quarter.lower[i], quarter.upper[i], price, weekly[:, i]
            )
        return float(weekly.sum()) - quarter.budget

    ceiling = float(_slopes(cells, units, np.tile(quarter.lower, (PLANNED, 1))).max())
    if excess(0.0) <= 0.0:
        price = 0.0
    elif excess(ceiling) >= 0.0:
        price = ceiling
    else:
        price = brentq(excess, 0.0, ceiling, xtol=1e-300, rtol=4 * np.finfo(float).eps)
        excess(price)
    best = _project(quarter, weekly)
    bound = gap(cells, quarter, best)
    if not bound <= GAP * quarter.budget:
        raise RuntimeError(
            f"the best schedule is certified to {bound / quarter.budget:.3g} a euro, not {GAP:g}"
        )
    return Schedule(best, worth(cells, best), price, max(bound, 0.0))


def _regret(cells: Sequence[Cell], best: Schedule, weekly: Series, quarter: Quarter) -> float:
    """What ``best`` returns over ``weekly`` on ``cells``, per euro of the budget.

    Raises:
        RuntimeError: ``weekly`` beat ``best`` by more than a tie and its certified gap.
    """
    regret = (best.worth - worth(cells, weekly)) / quarter.budget
    if regret < -(TIE + best.gap / quarter.budget):
        raise RuntimeError(f"a schedule beat the best schedule by {-regret:.3g} a euro")
    return regret


def score_schedule(truth: Truth, schedule: ArrayLike) -> ScheduleScore:
    """``schedule``, a spend each week on each cell or each channel, ``(13, cells)`` or ``(13,
    channels)``, scored against ``truth``; see the module.

    Raises:
        ValueError: a schedule of another shape, or what :func:`oracle` refuses.
        RuntimeError: the schedule beat the best schedule by more than a tie and its certified gap,
            which only a broken oracle allows.
    """
    quarter = truth.quarter
    weekly = np.asarray(schedule, dtype=float)
    if weekly.ndim != 2 or weekly.shape[0] != PLANNED:
        raise ValueError(
            f"a schedule holds a spend in each of the quarter's {PLANNED} weeks, not {weekly.shape}"
        )
    allotted = np.array([quarter.allot(row) for row in weekly])
    moved = None
    scale = OUT_OF_PLAN * quarter.budget
    if not (
        abs(float(allotted.sum()) - quarter.budget) <= scale
        and np.all(allotted >= quarter.lower - scale)
        and np.all(allotted <= quarter.upper + scale)
    ):
        inside = _project(quarter, allotted)
        moved = float(np.max(np.abs(inside - allotted)))
        allotted = inside
    best = oracle(truth.cells, quarter)
    regret = _regret(truth.cells, best, allotted, quarter)
    realised = None
    if truth.realised is not None:
        realised = _regret(truth.realised, oracle(truth.realised, quarter), allotted, quarter)
    edge = (best.worth - truth.best.worth) / quarter.budget
    return ScheduleScore(regret, realised, edge, moved)

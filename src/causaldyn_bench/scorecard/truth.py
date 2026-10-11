"""The scorecard's oracle: the best plan in the box at the budget, for any number of cells, each
with an effect that may move over the quarter.

A *cell* is one thing a plan spends on: one of Track M v2's channels, or later a channel in one
geo. Its spend is one weekly level held over the quarter's 13 weeks, and its adstock over the
quarter and the kernel's tail after it, ``H = 13 + L - 1`` weeks, is affine in that level: the
history's carryover plus the level times the reach of one unit a week. Its worth is what it
returns over those weeks,

    worth(w) = sum_t effect_t * curve(carry_t + w * reach_t),

with ``effect`` a path over the ``H`` weeks: the channel's constant coefficient where the effect
holds still, as on Track M v2, and a law's expected path where it drifts. A plan's worth is the
sum of its cells'.

The best plan in the box at the budget:

* where every cell is concave, a concave curve with an effect never negative, exact for any number
  of cells: at a price ``mu`` per euro each cell spends where its slope over the quarter meets
  ``13 mu``, or sits at an end of its box, and ``mu`` is found by bisection where the plan spends
  the budget;
* otherwise, where every cell's curve states its bend (:attr:`Curve.bend
  <causaldyn_bench.endogenous_mmm.Curve.bend>`), as every curve of
  :data:`~causaldyn_bench.endogenous_mmm.CURVES` does, exact for any number of cells to a gap of
  :data:`GAP` per euro of the budget, a tenth of a tie: Falk and Soland's branch and bound
  (:func:`_bounded`);
* otherwise, for up to three cells, every split on a grid of 401 points a side of every cell's box
  but the last's, the last spending the rest, refined by SLSQP from the grid's best points, each
  some points from the others: Track M v2's searched plan;
* for more cells, a max-plus dynamic programme over 2 000 steps of the budget above the cells'
  floors, the cells taken in turn, refined by SLSQP from its best plan.

Written apart from :mod:`chc.allocation`, whose plans it scores, and apart from
:func:`causaldyn_bench.mmm_decision.oracle`, which it reproduces: on every committed Track M v2
world a test holds its worth and the status quo's regret to the budgets run's ``TIE``.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq, minimize

from causaldyn_bench.endogenous_mmm import Curve, MediaMixWorld, Series
from causaldyn_bench.mmm_decision import GRID, PLANNED, STARTS, Plan, Quarter

STEPS = 2_000  # the dynamic programme's steps of the budget above the cells' floors
APART = 4  # a grid point this many points or fewer from a start, along every axis, is no start
GAP = 1e-10  # the bound's certified gap per euro of the budget, a tenth of budget_regret's tie
SAMPLES = 128  # the pieces a cell's interval is read in, for its envelope
NODES = 100_000  # the boxes the bound may split before it gives up
ROUNDING = 64.0 * float(np.finfo(float).eps)  # a sampled worth's rounding, of the largest one


@dataclass(frozen=True, eq=False)
class Cell:
    """One cell over the quarter and the kernel's tail: its adstock ``carry + weekly * reach``, its
    curve at its ``scale`` (his ``lambda``), and the effect each of those weeks carries."""

    carry: Series
    reach: Series
    curve: Curve
    scale: float
    effect: Series

    def __post_init__(self) -> None:
        shapes = {np.shape(self.carry), np.shape(self.reach), np.shape(self.effect)}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise ValueError(f"carry, reach and effect are paths over the same weeks, not {shapes}")

    @property
    def concave(self) -> bool:
        """Whether the worth is concave in the weekly spend: a concave curve of an affine adstock,
        weighted by an effect that is never negative."""
        return self.curve.concave and bool(np.all(self.effect >= 0.0))

    @property
    def bend(self) -> float | None:
        """The worth's greatest concavity in the weekly spend, the largest ``-worth''``: its curve's
        bend times ``sum effect reach^2``, where the curve states its bend and the effect is never
        negative; None otherwise."""
        if self.curve.bend is None or not np.all(self.effect >= 0.0):
            return None
        return float(np.sum(self.effect * self.reach**2)) * self.curve.bend(self.scale)

    def worths(self, weekly: ArrayLike) -> Series:
        """The worth at every entry of ``weekly``, whatever its shape."""
        adstock = self.carry + np.asarray(weekly, dtype=float)[..., None] * self.reach
        return np.sum(self.effect * self.curve.value(adstock, self.scale), axis=-1)

    def worth(self, weekly: float) -> float:
        return float(self.worths(weekly))

    def returns(self, weekly: float) -> Series:
        """What the cell returns in each of its weeks at the weekly spend ``weekly``."""
        return self.effect * self.curve.value(self.carry + weekly * self.reach, self.scale)

    def slope(self, weekly: float) -> float:
        adstock = self.carry + weekly * self.reach
        return float(np.sum(self.effect * self.reach * self.curve.slope(adstock, self.scale)))


def cells(
    world: MediaMixWorld,
    quarter: Quarter,
    effect: ArrayLike | None = None,
    curves: Sequence[Curve] | None = None,
) -> tuple[Cell, ...]:
    """Each of ``world``'s channels as a cell over the quarter after ``quarter.history``: the
    history's carryover and the reach of a unit a week through the channel's kernel, the world's
    own (:meth:`MediaMixWorld.kernel`), nothing spent after the quarter; on the world's curves, or
    on ``curves``, one a channel, each at the channel's ``saturation``, his ``lambda``; and the
    channel's constant effect, or ``effect``, a path a channel over the quarter and the kernel's
    tail, ``(channels, 13 + L - 1)``."""
    length = world.kernel_length
    horizon = PLANNED + length - 1
    history = quarter.history
    weeks = history.shape[0]
    # the week each of the horizon's weeks reads at each lag of the kernel, numbered from 0
    source = weeks + np.arange(horizon)[:, None] - np.arange(length)[None, :]
    before = source < weeks
    during = (source >= weeks) & (source < weeks + PLANNED)
    channels = len(world.channels)
    paths = (
        np.repeat(np.asarray(world.effect, dtype=float)[:, None], horizon, axis=1)
        if effect is None
        else np.asarray(effect, dtype=float)
    )
    if paths.shape != (channels, horizon):
        raise ValueError(f"an effect path is {(channels, horizon)}, not {paths.shape}")
    curves = curves or world.curves()
    out = []
    for c, lam in enumerate(world.saturation):
        kernel = world.kernel(c)
        spent = np.where(before, history[np.minimum(source, weeks - 1), c], 0.0)
        out.append(Cell(spent @ kernel, during @ kernel, curves[c], lam, paths[c]))
    return tuple(out)


def worth(cells: Sequence[Cell], weekly: ArrayLike) -> float:
    """What a plan's weekly spends return over the quarter and the tail, carryover in and out."""
    return sum(cell.worth(float(w)) for cell, w in zip(cells, np.asarray(weekly), strict=True))


def oracle(cells: Sequence[Cell], quarter: Quarter) -> Plan:
    """The best plan in ``quarter``'s box at its budget on ``cells``: exact where every cell is
    concave, searched otherwise. Its price is the budget's: what one more euro of it returns, nan
    where a searched plan holds every cell at an end of its box."""
    if len(cells) != quarter.lower.size:
        raise ValueError(f"{len(cells)} cells against a box of {quarter.lower.size}")
    rate = quarter.budget / PLANNED
    if not quarter.lower.sum() <= rate <= quarter.upper.sum():
        raise ValueError(f"no plan in the box spends {rate:.6g} a week")
    if all(cell.concave for cell in cells):
        return _priced(cells, quarter)
    if all(cell.bend is not None for cell in cells):
        return _bounded(cells, quarter)
    if len(cells) <= 3:
        return _gridded(cells, quarter)
    return _programmed(cells, quarter)


def _priced(cells: Sequence[Cell], quarter: Quarter) -> Plan:
    """Concave cells: each spends where its slope meets the price, the price bisected to the
    budget."""

    def weekly_at(price: float) -> Series:
        target = PLANNED * price
        spends = []
        for cell, low, high in zip(cells, quarter.lower, quarter.upper, strict=True):
            if cell.slope(low) <= target:
                spends.append(low)
            elif cell.slope(high) >= target:
                spends.append(high)
            else:
                spends.append(brentq(lambda w, c=cell: c.slope(w) - target, low, high, xtol=1e-12))
        return np.array(spends, dtype=float)

    def excess(price: float) -> float:
        return PLANNED * float(weekly_at(price).sum()) - quarter.budget

    ceiling = max(cell.slope(low) for cell, low in zip(cells, quarter.lower, strict=True))
    ceiling /= PLANNED
    if excess(0.0) <= 0.0:
        price = 0.0
    elif excess(ceiling) >= 0.0:
        price = ceiling
    else:
        price = brentq(excess, 0.0, ceiling, xtol=1e-14, rtol=4 * np.finfo(float).eps)
    weekly = weekly_at(price)
    return Plan(weekly, worth(cells, weekly), price)


@dataclass(frozen=True, eq=False)
class _Envelope:
    """A cell's worth over an interval, from above: concave and piecewise linear through the points
    ``(x, y)``, ``x`` rising."""

    x: Series
    y: Series

    def at(self, weekly: float) -> float:
        return float(np.interp(weekly, self.x, self.y))


def _envelope(cell: Cell, low: float, high: float) -> _Envelope:
    """The upper concave hull of the cell's worth at :data:`SAMPLES` + 1 points evenly spread over
    ``[low, high]``, lifted by what the worth can rise above a chord between two neighbours, ``bend
    h^2 / 8`` for their spacing ``h``, and by the points' rounding: above the worth everywhere on
    the interval."""
    bend = cell.bend
    if bend is None:
        raise ValueError("a cell bounded from above states its bend")
    x = np.linspace(low, high, SAMPLES + 1)
    y = cell.worths(x)
    step = (high - low) / SAMPLES
    lift = bend * step**2 / 8.0 + ROUNDING * float(np.max(np.abs(y)))
    if step <= 0.0:
        return _Envelope(x[:1], y[:1] + lift)
    hull: list[int] = []
    for i in range(x.size):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            if (x[a] - x[o]) * (y[i] - y[o]) < (y[a] - y[o]) * (x[i] - x[o]):
                break  # the last point lies above the chord past it, so it stays on the hull
            hull.pop()
        hull.append(i)
    return _Envelope(x[hull], y[hull] + lift)


def _relaxed(envelopes: Sequence[_Envelope], rate: float) -> tuple[float, Series]:
    """The best of the envelopes' sum where the weekly spends, each in its envelope's interval, sum
    to ``rate``, and those spends: every cell from its interval's start, the rest of the rate spent
    on the steepest pieces first, which is exact on concave pieces."""
    weekly = np.array([float(e.x[0]) for e in envelopes])
    bound = float(sum(float(e.y[0]) for e in envelopes))
    owner = np.concatenate([np.full(e.x.size - 1, c) for c, e in enumerate(envelopes)])
    length = np.concatenate([np.diff(e.x) for e in envelopes])
    slope = np.concatenate([np.diff(e.y) for e in envelopes]) / length
    rest = rate - float(weekly.sum())
    for k in np.argsort(-slope, kind="stable"):
        if rest <= 0.0:
            break
        taken = min(float(length[k]), rest)
        weekly[owner[k]] += taken
        bound += float(slope[k]) * taken
        rest -= taken
    return bound, weekly


def _bounded(cells: Sequence[Cell], quarter: Quarter) -> Plan:
    """Cells whose curves state their bend: Falk and Soland's branch and bound for separable
    programmes (Management Science 15(9), 1969).

    On a box of intervals, one a cell, each cell's worth lies below its envelope
    (:func:`_envelope`), so the best of the envelopes' sum at the budget, exact on concave pieces
    (:func:`_relaxed`), bounds every plan in the box. Its spends are a plan, which SLSQP refines
    where it beats the best so far. The box of the highest bound is split in two at its plan's
    spend of the cell whose envelope lies furthest above its worth there, or at the interval's
    middle where that spend lies within a tenth of an end, and a box no better than the best plan
    by :data:`GAP` per euro of the budget is dropped. Once no box is left, no plan in the box at
    the budget returns more than the best plan by that gap.

    Raises:
        RuntimeError: :data:`NODES` boxes were split without closing the gap.
    """
    rate = quarter.budget / PLANNED
    tolerance = GAP * quarter.budget
    held: dict[tuple[int, float, float], _Envelope] = {}

    def relaxed(lows: Series, highs: Series) -> tuple[float, Series, list[_Envelope]]:
        envelopes = []
        for c, (low, high) in enumerate(zip(lows.tolist(), highs.tolist(), strict=True)):
            if (c, low, high) not in held:
                held[(c, low, high)] = _envelope(cells[c], low, high)
            envelopes.append(held[(c, low, high)])
        bound, weekly = _relaxed(envelopes, rate)
        return bound, weekly, envelopes

    bound, weekly, envelopes = relaxed(quarter.lower, quarter.upper)
    best = _refined(cells, quarter, [weekly])
    order = itertools.count()
    boxes = [(-bound, next(order), quarter.lower, quarter.upper, weekly, envelopes)]
    for _ in range(NODES):
        if not boxes or -boxes[0][0] <= best.worth + tolerance:
            return best
        _, _, lows, highs, weekly, envelopes = heapq.heappop(boxes)
        above = [
            e.at(w) - cell.worth(w) if high > low else -math.inf
            for e, cell, w, low, high in zip(
                envelopes, cells, weekly.tolist(), lows, highs, strict=True
            )
        ]
        c = int(np.argmax(above))
        low, high = float(lows[c]), float(highs[c])
        inside = low + 0.1 * (high - low) < weekly[c] < high - 0.1 * (high - low)
        split = float(weekly[c]) if inside else (low + high) / 2.0
        for side in (0, 1):
            child_lows, child_highs = lows.copy(), highs.copy()
            (child_highs if side == 0 else child_lows)[c] = split
            if child_lows.sum() > rate or child_highs.sum() < rate:
                continue
            child = relaxed(child_lows, child_highs)
            if worth(cells, child[1]) > best.worth + tolerance:
                best = max(best, _refined(cells, quarter, [child[1]]), key=lambda plan: plan.worth)
            if child[0] > best.worth + tolerance:
                heapq.heappush(boxes, (-child[0], next(order), child_lows, child_highs, *child[1:]))
    raise RuntimeError(f"{NODES} boxes split and the bound still above the best plan by {GAP:g}")


def _refined(cells: Sequence[Cell], quarter: Quarter, starts: Sequence[Series]) -> Plan:
    """The best plan SLSQP reaches from ``starts``, each moved into the box at the budget."""
    rate = quarter.budget / PLANNED
    bounds = list(zip(quarter.lower, quarter.upper, strict=True))

    def slopes(weekly: Series) -> Series:
        return np.array([cell.slope(float(w)) for cell, w in zip(cells, weekly, strict=True)])

    weekly, best = quarter.status_quo, -math.inf
    for start in starts:
        solution = minimize(
            lambda w: -worth(cells, w),
            start,
            jac=lambda w: -slopes(w),
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - rate}],
            options={"ftol": 1e-15, "maxiter": 1000},
        )
        for candidate in (quarter.project(solution.x), quarter.project(start)):
            value = worth(cells, candidate)
            if value > best:
                weekly, best = candidate, value
    low, high = quarter.lower, quarter.upper
    free = (weekly > low * (1 + 1e-9)) & (weekly < high * (1 - 1e-9))
    price = float(np.mean(slopes(weekly)[free])) / PLANNED if free.any() else math.nan
    return Plan(weekly, best, price)


def _gridded(cells: Sequence[Cell], quarter: Quarter) -> Plan:
    """Up to three cells: every split on a grid of every box but the last's, the last cell spending
    the rest, refined from the grid's best points."""
    rate = quarter.budget / PLANNED
    low, high = quarter.lower, quarter.upper
    free = len(cells) - 1
    axes = [np.linspace(low[c], high[c], GRID) for c in range(free)]
    values = np.zeros((GRID,) * free)
    last = np.full(values.shape, rate)
    for c, axis in enumerate(axes):
        along = [1] * free
        along[c] = GRID
        values = values + cells[c].worths(axis).reshape(along)
        last = last - axis.reshape(along)
    inside = (last >= low[-1]) & (last <= high[-1])
    values = np.where(inside, values + cells[-1].worths(np.clip(last, low[-1], high[-1])), -np.inf)
    chosen: list[tuple[int, ...]] = []
    for flat in np.argsort(values, axis=None)[::-1]:
        index = tuple(int(i) for i in np.unravel_index(flat, values.shape))
        if not np.isfinite(values[index]) or len(chosen) == STARTS:
            break
        if all(
            max(abs(i - j) for i, j in zip(index, other, strict=True)) > APART for other in chosen
        ):
            chosen.append(index)
    starts = [
        np.array([*(axes[c][i] for c, i in enumerate(index)), last[index]]) for index in chosen
    ]
    return _refined(cells, quarter, starts)


def _programmed(cells: Sequence[Cell], quarter: Quarter) -> Plan:
    """More cells: the best split of the budget above the floors in :data:`STEPS` steps, refined
    from that split."""
    return _refined(cells, quarter, [_split(cells, quarter, STEPS)])


def _split(cells: Sequence[Cell], quarter: Quarter, steps: int) -> Series:
    """The best plan whose cells each spend their floor and a whole number of the ``steps`` equal
    steps of the budget above the floors: each cell taken in turn against the best of the cells
    before it for every number of steps they spend together, then read back from the last."""
    rate = quarter.budget / PLANNED
    low, high = quarter.lower, quarter.upper
    step = (rate - float(low.sum())) / steps
    if step <= 0.0:  # the floors spend the budget: there is nothing to split
        return low.copy()
    counts = np.minimum(np.floor((high - low) / step).astype(int), steps)
    totals = np.arange(steps + 1)
    best = np.full(steps + 1, -np.inf)
    best[: counts[0] + 1] = cells[0].worths(low[0] + step * np.arange(counts[0] + 1))
    choices = []
    for c in range(1, len(cells)):
        own = np.arange(counts[c] + 1)
        before = totals[:, None] - own[None, :]
        values = cells[c].worths(low[c] + step * own)
        candidates = np.where(before >= 0, best[np.maximum(before, 0)] + values, -np.inf)
        choice = np.argmax(candidates, axis=1)
        best = candidates[totals, choice]
        choices.append(choice)
    if not np.isfinite(best[steps]):
        raise RuntimeError("the boxes' steps cannot spend the budget")
    taken = np.zeros(len(cells), dtype=int)
    remaining = steps
    for c in range(len(cells) - 1, 0, -1):
        taken[c] = choices[c - 1][remaining]
        remaining -= taken[c]
    taken[0] = remaining
    return low + step * taken

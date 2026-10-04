"""The gate a comparison passes: a paired difference read against a margin at looks fixed before the
run starts, the run stopping for the comparison where its verdict is plain.

A comparison claims that the mean of the first arm's axis less the second's lies below a margin
``delta``: the first arm is no worse than the second by more than ``delta`` where it is above 0, and
better where it is 0. A :class:`Gate` reads it at its looks, each a count of every environment's
worlds, so each look is a prefix of the run (:func:`~causaldyn_bench.scorecard.scoring.look`):
``Gate((25, 50, 75, 100))`` reads a quarter, a half, three quarters and all of them.

**The boundaries** spend the one-sided ``alpha`` over the looks by Lan and DeMets' (1983)
O'Brien-Fleming type function ``alpha(t) = 2 - 2 Phi(z_{1 - alpha/2} / sqrt(t))`` of the share
``t`` of the worlds a look reads (:func:`spent`), so an early look shows the claim only where it is
plain: at ``alpha = 0.025`` and four equal looks the statistic must reach 4.33, 2.96, 2.36 and 2.01,
against a single look's 1.96. :func:`boundaries` solves for them by Armitage, McPherson and Rowe's
(1969) recursion on the sub-density of the score statistic, its integrals by Simpson's rule
(Jennison and Turnbull 2000, ch. 19), so the looks may fall anywhere.

**A look** reads the statistic ``(delta - mean) / (sd / sqrt(n))`` over its ``n`` paired
differences against its boundary moved to Student's t on ``n - 1`` degrees of freedom at the same
nominal level, Jennison and Turnbull's significance-level approach to a variance the look
estimates. At or above it the claim is shown; below it at the last look, not shown. A gate may name
a futility threshold, read on the same scale at every look but the last. It is non-binding: the
boundaries are solved without it, so a run that ignores it keeps its ``alpha``, and one that obeys
it stops a comparison it would not show sooner and shows fewer of the rest
(:meth:`Gate.crossings` prices both).

**A claim over many comparisons**, every arm on every family, is an intersection-union test (Berger
1982): it holds only where each comparison shows its own at level ``alpha``, so it holds at
``alpha`` with no adjustment for how many it needs.

**A list over many comparisons**, the losses printed beside the unadjusted verdicts, is adjusted
instead, by :func:`holm` on each comparison's sequential level (:meth:`Gate.level`): the least
``alpha`` at which a look it read crosses that look's boundary solved at that ``alpha``. That is
Maurer and Bretz's (2013) graphical procedure for group sequential designs, with Holm's weights,
looking back at every look read. It holds the familywise error at ``alpha`` under any dependence
between the comparisons, since every boundary falls as ``alpha`` rises; the move to Student's t
holds it as nearly as it holds one comparison's size.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Literal, TypeVar

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm
from scipy.stats import t as student

from causaldyn_bench.endogenous_mmm import Series
from causaldyn_bench.scorecard.scoring import WorldRecord, paired

ALPHA = 0.025  # one-sided: a paired 95 % interval's upper end
REACH = 10.0  # standard deviations the grid runs from its mean where no boundary cuts it
PER_SD = 16  # Simpson intervals a standard deviation of the narrowest Gaussian the grid integrates
CHUNK = 1 << 22  # kernel entries evaluated at once, bounding a convolution's memory at 32 MiB
FLOOR = 1e-12  # the least sequential level read; a comparison that crosses below it reads it

Verdict = Literal["shown", "futile", "continue", "not shown"]
K = TypeVar("K")


def spent(t: float, alpha: float = ALPHA) -> float:
    """The ``alpha`` a look at share ``t`` of the worlds has spent, with the looks before it.

    Raises:
        ValueError: on ``t`` outside (0, 1].
    """
    if not 0.0 < t <= 1.0:
        raise ValueError(f"a look reads a share of the worlds in (0, 1], not {t}")
    return 2.0 * float(norm.sf(norm.isf(alpha / 2.0) / math.sqrt(t)))


def boundaries(fractions: Sequence[float], alpha: float = ALPHA) -> tuple[float, ...]:
    """Each look's boundary on the standard-normal scale, at the shares ``fractions`` of the worlds,
    increasing to 1: the chance the look crosses it, no look before having crossed its own, is
    ``spent(t_k) - spent(t_{k-1})`` where the claim is false by a hair. A look whose share spends
    less than a double holds has an infinite boundary.

    Raises:
        ValueError: on shares that do not increase strictly from above 0 to 1, or an ``alpha``
            outside (0, 1/2).
    """
    _check(fractions, alpha)
    return _recursion(fractions, 0.0, None, None, alpha)[0]


@dataclass(frozen=True)
class Look:
    """A comparison read at one look."""

    worlds: int  # the look's worlds of each environment
    n: int  # the paired differences it reads: its worlds of every environment
    mean: float
    sd: float
    statistic: float  # (margin - mean) / (sd / sqrt(n))
    boundary: float  # the look's boundary, moved to Student's t on n - 1 degrees of freedom
    verdict: Verdict


@dataclass(frozen=True)
class Crossings:
    """A gate's chances at one drift, look by look."""

    shown: tuple[float, ...]  # that the look shows the claim, none before having stopped
    futile: tuple[
        float, ...
    ]  # that it stops for futility, none before having stopped; 0 at the last
    expected: float  # the share of the last look's worlds a comparison reads before it stops


@dataclass(frozen=True)
class Gate:
    """The looks a run's comparisons are read at, each a count of every environment's worlds, a
    futility threshold on the standard-normal scale or none, and the one-sided ``alpha`` the looks
    spend.

    Raises:
        ValueError: on looks that do not increase strictly from at least 2 worlds, an ``alpha``
            outside (0, 1/2), or a futility threshold not below every boundary but the last.
    """

    looks: tuple[int, ...]
    futility: float | None = None
    alpha: float = ALPHA

    def __post_init__(self) -> None:
        if not self.looks or self.looks[0] < 2:
            raise ValueError(f"a gate's first look reads at least 2 worlds: {self.looks}")
        _check(self.fractions, self.alpha)
        if self.futility is not None and self.futility >= min(self.bounds[:-1], default=math.inf):
            raise ValueError(
                f"a futility threshold of {self.futility} stops every look it does not show: "
                f"the boundaries are {self.bounds}"
            )

    @property
    def fractions(self) -> tuple[float, ...]:
        return tuple(worlds / self.looks[-1] for worlds in self.looks)

    @cached_property
    def bounds(self) -> tuple[float, ...]:
        """Each look's boundary on the standard-normal scale (:func:`boundaries`)."""
        return boundaries(self.fractions, self.alpha)

    def crossings(self, drift: float) -> Crossings:
        """The gate's chances where the statistic at the last look has mean ``drift``, the variance
        taken as known: ``(delta - mu) sqrt(N) / sigma`` for paired differences of mean ``mu`` and
        SD ``sigma``, ``N`` of them at the last look.
        """
        _, shown, futile = _recursion(self.fractions, drift, self.futility, self.bounds, self.alpha)
        stops = [s + f for s, f in zip(shown[:-1], futile[:-1], strict=True)]
        read = sum(t * p for t, p in zip(self.fractions[:-1], stops, strict=True))
        return Crossings(shown, futile, read + 1.0 - sum(stops))

    def read(
        self,
        records: Sequence[WorldRecord],
        first: str,
        second: str,
        *,
        margin: float,
        axis: str = "regret",
    ) -> tuple[Look, ...]:
        """``first``'s ``axis`` less ``second``'s read against ``margin`` at each of the gate's
        looks the records hold whole, to the first whose verdict is not ``"continue"``.

        Raises:
            ValueError: as :func:`~causaldyn_bench.scorecard.scoring.paired` does on a look.
        """
        held = _held(records)
        environments = len({record.environment for record in records})

        def seen() -> Iterator[tuple[int, int, float, float]]:
            for worlds in self.looks:
                if worlds > held:
                    return
                difference = paired(records, first, second, worlds, axis).difference
                yield worlds, worlds * environments, difference.mean, difference.sd

        return _reading(seen(), self.bounds, margin, self.futility)

    def level(self, looks: Sequence[Look]) -> float:
        """The least one-sided ``alpha`` in ``[FLOOR, 1/2)`` at which one of ``looks``, a reading
        of this gate, crosses its look's boundary solved at that ``alpha`` and moved to Student's t
        as at the gate's own: the comparison's sequential level. ``FLOOR`` where one crosses there
        already, and 1 where none crosses below 1/2 or nothing was read.

        A reading stops at its first verdict, so the level reads only the looks before it, and a
        look left unread could only have lowered it: the level errs high, which keeps its
        guarantee.

        Raises:
            ValueError: on more looks than the gate has.
        """
        if len(looks) > len(self.looks):
            raise ValueError(f"{len(looks)} looks read on a gate of {len(self.looks)}")
        # each statistic on the normal scale at the same tail, which a boundary crosses as the
        # Student move would
        scaled = [float(norm.isf(student.sf(look.statistic, look.n - 1))) for look in looks]

        def excess(log_alpha: float) -> float:
            bounds = boundaries(self.fractions, math.exp(log_alpha))
            return max(z - bound for z, bound in zip(scaled, bounds, strict=False))

        low, high = math.log(FLOOR), math.log(math.nextafter(0.5, 0.0))
        if not scaled or excess(high) < 0.0:
            return 1.0
        if excess(low) >= 0.0:
            return FLOOR
        return math.exp(brentq(excess, low, high, xtol=1e-12))


def holm(levels: Mapping[K, float], alpha: float = ALPHA) -> frozenset[K]:  # noqa: UP047 -- CI runs 3.11
    """The comparisons Holm's (1979) step-down shows at familywise ``alpha``: in order of their
    levels, ties by key, each while its level is at most ``alpha`` over the number not yet shown.

    Raises:
        ValueError: on an ``alpha`` outside (0, 1/2).
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"a one-sided alpha lies in (0, 1/2), not {alpha}")
    ordered = sorted(levels, key=lambda key: (levels[key], str(key)))
    shown: list[K] = []
    for rank, key in enumerate(ordered):
        if levels[key] > alpha / (len(ordered) - rank):
            break
        shown.append(key)
    return frozenset(shown)


def _check(fractions: Sequence[float], alpha: float) -> None:
    shares = np.asarray(fractions, dtype=float)
    if shares.size == 0 or shares[0] <= 0.0 or shares[-1] != 1.0 or np.any(np.diff(shares) <= 0):
        raise ValueError(f"the looks' shares increase strictly from above 0 to 1: {fractions}")
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"a one-sided alpha lies in (0, 1/2), not {alpha}")


def _held(records: Sequence[WorldRecord]) -> int:
    """The worlds of every environment the records hold from the first on."""
    places: dict[str, set[int]] = {}
    for record in records:
        places.setdefault(record.environment, set()).add(record.index)
    return min(
        (next(i for i in range(len(held) + 1) if i not in held) for held in places.values()),
        default=0,
    )


def _reading(
    seen: Iterable[tuple[int, int, float, float]],
    bounds: Sequence[float],
    margin: float,
    futility: float | None,
) -> tuple[Look, ...]:
    """The looks ``seen`` yields, as (worlds, n, mean, sd), read to the first verdict that stops."""
    looks: list[Look] = []
    for k, (worlds, n, mean, sd) in enumerate(seen):
        statistic = _statistic(margin - mean, sd / math.sqrt(n))
        boundary = _moved(bounds[k], n)
        verdict: Verdict
        if statistic >= boundary:
            verdict = "shown"
        elif k == len(bounds) - 1:
            verdict = "not shown"
        elif futility is not None and statistic < _moved(futility, n):
            verdict = "futile"
        else:
            verdict = "continue"
        looks.append(Look(worlds, n, mean, sd, statistic, boundary, verdict))
        if verdict != "continue":
            break
    return tuple(looks)


def _statistic(gap: float, se: float) -> float:
    """``gap / se``, where differences that do not spread show a gap of either sign plainly."""
    if se > 0.0:
        return gap / se
    return math.copysign(math.inf, gap) if gap else 0.0


@cache
def _moved(z: float, n: int) -> float:
    """Student's t on ``n - 1`` degrees of freedom at the nominal level of ``z``."""
    return float(student.isf(norm.sf(z), n - 1))


def _recursion(
    fractions: Sequence[float],
    drift: float,
    futility: float | None,
    bounds: Sequence[float] | None,
    alpha: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Each look's boundary, solved where ``bounds`` is None, and its chances to show the claim and
    to stop for futility, none before having stopped.

    The score statistic ``S_k = Z_k sqrt(t_k)`` moves by independent steps
    ``N(drift d_k, d_k)``, ``d_k = t_k - t_{k-1}``. Before each look the sub-density of ``S`` on
    the region where the comparison went on is held as Simpson's nodes with their weighted mass, a
    point mass at 0 before the first.
    """
    shares = np.asarray(fractions, dtype=float)
    steps = np.diff(shares, prepend=0.0)
    last = shares.size - 1
    solved: list[float] = []
    shown = [0.0] * shares.size
    futile = [0.0] * shares.size
    nodes, mass = np.zeros(1), np.ones(1)
    for k in range(shares.size):
        sd, root = math.sqrt(steps[k]), math.sqrt(shares[k])
        centre = nodes + drift * steps[k]
        if bounds is None:
            share = spent(shares[k], alpha) - (spent(shares[k - 1], alpha) if k else 0.0)
            low, high = float(centre.min()) - REACH * sd, float(centre.max()) + 4 * REACH * sd
            edge = (
                brentq(_excess, low, high, (centre, mass, sd, share), xtol=1e-14, rtol=1e-15)
                if share > 0.0
                else math.inf
            )
            solved.append(edge / root)
        else:
            edge = bounds[k] * root
        line = futility * root if futility is not None and k < last else -math.inf
        shown[k] = _above(edge, centre, mass, sd)
        futile[k] = float(mass @ ndtr((line - centre) / sd))
        if k == last:
            break
        low = max(line, drift * shares[k] - REACH * root)
        high = min(edge, drift * shares[k] + REACH * root)
        if high <= low:  # the comparison goes on with less chance than a double holds
            break
        x, weights = _simpson(low, high, min(sd, math.sqrt(steps[k + 1])) / PER_SD)
        nodes, mass = x, weights * _convolved(x, centre, mass, sd)
    return tuple(solved) if bounds is None else tuple(bounds), tuple(shown), tuple(futile)


def _above(edge: float, centre: Series, mass: Series, sd: float) -> float:
    """The chance a step of ``sd`` from the weighted nodes ``centre`` ends above ``edge``."""
    return float(mass @ ndtr((centre - edge) / sd))


def _excess(edge: float, centre: Series, mass: Series, sd: float, share: float) -> float:
    return _above(edge, centre, mass, sd) - share


def _simpson(low: float, high: float, step: float) -> tuple[Series, Series]:
    """Simpson's nodes and weights on ``[low, high]``, at most ``step`` apart."""
    intervals = 2 * max(1, math.ceil((high - low) / step / 2))
    weights = np.full(intervals + 1, 2.0)
    weights[1::2] = 4.0
    weights[[0, -1]] = 1.0
    return np.linspace(low, high, intervals + 1), weights * (high - low) / (3 * intervals)


def _convolved(x: Series, centre: Series, mass: Series, sd: float) -> Series:
    """``sum_j mass_j phi((x_i - centre_j) / sd) / sd`` at each node ``x_i``."""
    out = np.empty(x.size)
    rows = max(1, CHUNK // centre.size)
    for i in range(0, x.size, rows):
        z = (x[i : i + rows, None] - centre) / sd
        out[i : i + rows] = np.exp(-0.5 * z * z) @ mass
    return out / (sd * math.sqrt(2.0 * math.pi))

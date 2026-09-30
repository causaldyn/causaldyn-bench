"""Track M v2, decisions: how much of next quarter's media budget each arm puts where, scored on the
world's own channels.

The world is :mod:`causaldyn_bench.endogenous_mmm`, Heusch's generator written from his paper, over
its 156 weeks of history. The decision is the quarter after it: a weekly spend for each channel,
held for the quarter's 13 weeks, with

* **the budget** thirteen times last year's mean weekly spend over all channels, the quarter
  planned at last year's rate;
* **the box** each channel's weekly spend kept within half and twice its own mean over the same
  52 weeks, as a planner bounds a channel near what it has run at;
* **the status quo** last year's mix at that budget, each channel at its mean.

A plan's worth is what its spend returns on the world's own channels over the quarter, carryover
included: the adstock the history leaves runs into the quarter, and the quarter's spend runs on for
the kernel's length after it with nothing spent. Media's effect is additive in this world, so a
plan's worth is the sum of its channels', and the baseline, the promotions and every mechanism that
set the history's spend leave it alone. The regret of a plan is what the best plan in the box at
the budget returns over what it does.

**Why the channels are drawn.** On his reference instance a quarter's best plan is a corner: paid
shopping's return per euro at twice its spend still tops Meta's and television's at their floors,
so the box's best plan runs PLA at whatever the others' floors leave (a test holds it). Every
arm that ranks PLA first then scores a regret of nought, whether or not its reading of PLA is
right, and his observational model reads PLA's return two and a half times too high (Heusch
2026a). One instance cannot tell a right reading from a lucky one. So each world draws every
channel's retention, saturation and effect independently from the range his three channels span,
log-uniform for the two scales; the mechanisms that set spend stay each channel's own, so the
bidding rule still chases demand through PLA whatever PLA returns. His reference instance is kept
as a setting of its own.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from causaldyn_bench.endogenous_mmm import YEAR, EndogenousMediaMix, MediaMixWorld, Series

PLANNED = 13  # weeks in the quarter planned
BOX = (0.5, 2.0)  # a channel's weekly spend, as a multiple of its mean over the last year
PARAMETER_STREAM = 7  # the channels' draw, apart from the world's own seed


def drawn(seed: int, reference: EndogenousMediaMix | None = None) -> EndogenousMediaMix:
    """His generator with each channel's retention, saturation and effect drawn from the range his
    three channels span: uniform for the retention, log-uniform for the saturation and the effect.
    """
    reference = reference or EndogenousMediaMix()
    rng = np.random.default_rng((PARAMETER_STREAM, seed))
    channels = len(reference.channels)

    def spread(values: tuple[float, ...], logged: bool) -> tuple[float, ...]:
        low, high = min(values), max(values)
        if logged:
            return tuple(np.exp(rng.uniform(math.log(low), math.log(high), channels)).tolist())
        return tuple(rng.uniform(low, high, channels).tolist())

    return dataclasses.replace(
        reference,
        retention=spread(reference.retention, logged=False),
        saturation=spread(reference.saturation, logged=True),
        effect=spread(reference.effect, logged=True),
    )


@dataclass(frozen=True)
class Quarter:
    """What every arm is given to decide with: the budget, each channel's box, the status quo and
    the spend before the quarter. Weekly spends are ``(channels,)``, in thousands of euros."""

    budget: float  # over the quarter, every channel
    lower: Series
    upper: Series
    status_quo: Series
    history: Series  # (weeks, channels)

    @classmethod
    def after(cls, world: MediaMixWorld) -> Quarter:
        mean = world.spend[-YEAR:].mean(axis=0)
        return cls(
            budget=PLANNED * float(mean.sum()),
            lower=BOX[0] * mean,
            upper=BOX[1] * mean,
            status_quo=mean,
            history=world.spend.copy(),
        )

    def feasible(self, weekly: Series, tolerance: float = 1e-9) -> bool:
        scale = tolerance * self.budget
        return bool(
            abs(PLANNED * float(np.sum(weekly)) - self.budget) <= scale
            and np.all(weekly >= self.lower - scale)
            and np.all(weekly <= self.upper + scale)
        )

    def project(self, weekly: Series) -> Series:
        """The plan in the box at the budget nearest ``weekly``: each channel moved by one shift
        and clipped to its box, the shift found by bisection."""
        target = self.budget / PLANNED

        def excess(shift: float) -> float:
            return float(np.clip(weekly - shift, self.lower, self.upper).sum() - target)

        reach = float(np.max(np.abs(weekly)) + np.max(self.upper))
        shift = brentq(excess, -reach, reach, xtol=1e-12 * target, rtol=4 * np.finfo(float).eps)
        return np.clip(weekly - shift, self.lower, self.upper)

    def equal_split(self) -> Series:
        return self.project(
            np.full(self.status_quo.size, self.budget / PLANNED / self.status_quo.size)
        )


@dataclass(frozen=True)
class _Reach:
    """One channel's adstock over the quarter and the tail as ``carry + weekly * reach``."""

    carry: Series
    reach: Series
    saturation: float
    effect: float

    def worth(self, weekly: float) -> float:
        adstock = self.carry + weekly * self.reach
        return self.effect * float(np.sum(np.tanh(self.saturation * adstock / 2.0)))

    def slope(self, weekly: float) -> float:
        adstock = self.carry + weekly * self.reach
        half = self.saturation / 2.0
        return self.effect * half * float(np.sum(self.reach / np.cosh(half * adstock) ** 2))


def _reaches(world: MediaMixWorld, quarter: Quarter) -> list[_Reach]:
    length = world.kernel_length
    weeks = quarter.history.shape[0]
    reaches = []
    for c, (alpha, lam, beta) in enumerate(
        zip(world.retention, world.saturation, world.effect, strict=True)
    ):
        weights = alpha ** np.arange(length)
        weights = weights / weights.sum()

        def adstock(weekly: float, column: int = c, kernel: Series = weights) -> Series:
            spend = np.concatenate(
                [quarter.history[:, column], np.full(PLANNED, weekly), np.zeros(length - 1)]
            )
            return np.convolve(spend, kernel)[weeks : spend.size]

        carry = adstock(0.0)
        reaches.append(_Reach(carry, adstock(1.0) - carry, lam, beta))
    return reaches


def worth(world: MediaMixWorld, quarter: Quarter, weekly: Series) -> float:
    """What a plan's spend returns on the world's channels, carryover in and out included."""
    return sum(r.worth(float(w)) for r, w in zip(_reaches(world, quarter), weekly, strict=True))


@dataclass(frozen=True)
class Plan:
    weekly: Series
    worth: float
    price: float  # the budget's shadow price: what one more euro of budget returns


def oracle(world: MediaMixWorld, quarter: Quarter) -> Plan:
    """The best plan in the box at the budget, on the world's own channels.

    Each channel's worth is concave in its weekly spend, ``tanh`` of an affine adstock, and the
    channels add, so the plan is exact: at a price ``mu`` per euro, each channel runs where its
    slope over the quarter meets ``PLANNED * mu`` or sits at the end of its box, and ``mu`` is
    found where the plan spends the budget. Written apart from every arm's planner, which this
    scores.
    """
    reaches = _reaches(world, quarter)

    def weekly_at(price: float) -> Series:
        rates = []
        for r, low, high in zip(reaches, quarter.lower, quarter.upper, strict=True):
            target = PLANNED * price
            if r.slope(low) <= target:
                rates.append(low)
            elif r.slope(high) >= target:
                rates.append(high)
            else:
                rates.append(brentq(lambda w, r=r, t=target: r.slope(w) - t, low, high, xtol=1e-12))
        return np.array(rates, dtype=float)

    def excess(price: float) -> float:
        return PLANNED * float(weekly_at(price).sum()) - quarter.budget

    ceiling = max(r.slope(low) for r, low in zip(reaches, quarter.lower, strict=True)) / PLANNED
    if excess(0.0) <= 0.0:
        price = 0.0
    elif excess(ceiling) >= 0.0:
        price = ceiling
    else:
        price = brentq(excess, 0.0, ceiling, xtol=1e-14, rtol=4 * np.finfo(float).eps)
    weekly = weekly_at(price)
    return Plan(weekly, worth(world, quarter, weekly), price)


def regret(world: MediaMixWorld, quarter: Quarter, weekly: Series, best: Plan) -> float:
    """What ``best`` returns over ``weekly``, which must spend the budget within the box."""
    if not quarter.feasible(weekly, tolerance=1e-6):
        raise ValueError(f"the plan {np.round(weekly, 3)} leaves the box or misses the budget")
    return best.worth - worth(world, quarter, weekly)

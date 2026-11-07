"""The CPM law: each channel's cost per thousand impressions moves week by week, and its response
acts on the impressions its spend buys. Family 1's *cpm* environment (:mod:`.drift`) draws it.

**The law.** Channel ``c``'s cost per thousand impressions in week ``t``, numbered from the
history's first, is

    CPM_{c,t} = level_c exp(a_c sin(2 pi t / 52) + y_{c,t}),
    y_{c,t} = phi_c y_{c,t-1} + s_c sqrt(1 - phi_c^2) eps_{c,t},

a first-order autoregression in its log about a seasonal mean, ``eps`` standard normal, started from
its stationary law ``N(0, s_c^2)`` the week before the history's first and run on over the quarter
and the kernel's tail. Each channel's parameters are drawn apart (:func:`draw`):

* ``a_c`` uniform on ``[0.05, 0.25]``: a yearly swing of 5 % to 28 % above the level and 5 % to
  22 % below it, in phase with the first harmonic of the world's own seasonality of sales,
  ``0.15 sin(2 pi t / 52)`` (:class:`~causaldyn_bench.endogenous_mmm.EndogenousMediaMix`), since an
  auction's price rises with the demand its season brings;
* ``phi_c`` uniform on ``[0.5, 0.9]`` a week: a shock's half-life of one to six and a half weeks,
  so the history's last weeks tell an arm part, not all, of the quarter's prices;
* ``s_c`` uniform on ``[0.05, 0.2]``: the log's stationary sd about its seasonal mean;
* ``level_c``, :data:`LEVELS`: 3 a thousand for Meta and 20 for television, the costs the demo of
  siMMMulator draws around (``scripts/simmmulator_worlds.R``), and 5 for paid shopping, which is
  bought by the click and has no such source.

CHOICE: the three ranges are the bench's own; no published series of prices is read. The level sets
the unit the impressions are counted in and nothing else: it moves no plan, regret or return.

**The response acts on impressions.** A channel's spend ``s_{c,t}`` buys
``n_{c,t} = s_{c,t} / CPM_{c,t} * 1000`` impressions (:meth:`Price.impressions`). The drawn world's
curve reads a thousand impressions as ``level_c`` euros of its spend: its adstock is the kernel's
over ``level_c n_{c,t} / 1000``, so its saturation of ``lambda_c`` a euro is
``lambda_c level_c / 1000`` an impression. A history computes that input as the spend times
``level_c / CPM_{c,t}``, what a euro buys against what it buys at the level (:attr:`Price.buys`),
so a CPM held at its level (:meth:`Law.held`) leaves the drawn world as it was, bit for bit.

The world's budget and bidding read its sales before media, so its spend is the drawn world's; its
media and sales move with the prices, and so does every lift test, which reads the response of its
own weeks. The export carries each channel's impressions over the history beside its spend
(:class:`~causaldyn_bench.scorecard.observe.Observation`).

**The best plan** is made in spend as family 1 makes it under every law (:mod:`.drift`): on the
quarter as the bench expects it from the history, which no arm reads, with the quarter's own prices
giving the best plan in hindsight, against which a plan's realised regret is read. Given the log's
state ``y_{c,T}`` in the history's last week, ``y_{c,T+h}`` is normal with mean
``phi_c^h y_{c,T}`` and variance ``s_c^2 (1 - phi_c^{2h})``, so a euro is expected to buy

    E[level_c / CPM_{c,T+h}] = exp(-a_c sin(2 pi (T + h) / 52) - phi_c^h y_{c,T}
                                   + s_c^2 (1 - phi_c^{2h}) / 2)

(:attr:`Price.expected`; ``scripts/cpm_expectation.mac`` checks it). A cell's adstock is affine in
its weekly spend at any prices (:func:`causaldyn_bench.scorecard.truth.cells`): the history's
carryover is read at the history's prices and each week of the quarter's spend at that week's,
expected or realised. What a euro buys enters the adstock, inside the curve, where under the other
laws the effect multiplies the curve from outside, so the plan best on the expected prices is the
certainty-equivalent plan, best for the expected adstock, and not exactly the plan best in
expectation. On the first eight pilot worlds the plan best on 2 000 sampled quarters returns at most
3e-4 a euro of the budget more than it, on that sample, and nothing more on four of them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from types import MappingProxyType

import numpy as np

from causaldyn_bench.endogenous_mmm import YEAR, MediaMixWorld, Series

# each channel's level, in euros a thousand impressions: the unit its impressions are counted in
LEVELS: Mapping[str, float] = MappingProxyType({"pla": 5.0, "meta": 3.0, "tv": 20.0})
AMPLITUDE = (0.05, 0.25)  # the yearly swing in the log, uniform
PERSISTENCE = (0.5, 0.9)  # phi, a week, uniform
SD = (0.05, 0.2)  # the log's stationary sd about its seasonal mean, uniform


@dataclass(frozen=True)
class Law:
    """Each channel's CPM law, ``(channels,)`` each: its level, the yearly swing of its log, the
    autoregression's persistence and the log's stationary sd about its seasonal mean."""

    level: Series
    amplitude: Series
    persistence: Series
    sd: Series

    @classmethod
    def held(cls, level: Series) -> Law:
        """The CPM held at ``level`` in every week."""
        level = np.asarray(level, dtype=float)
        nought = np.zeros_like(level)
        return cls(level, nought, nought, nought)

    def price(self, shocks: Series, onward: Series) -> Price:
        """The prices over the history and the weeks after it, ``(weeks + horizon, channels)``:
        ``shocks``' first row starts the log's autoregression the week before the history's first,
        one row a week of the history after it, and ``onward``'s rows run it on, one a week; and
        what a euro is expected to buy in each week after the history, given the log in its last."""
        weeks = shocks.shape[0] - 1
        steps = np.vstack([shocks[1:], onward])
        innovation = self.sd * np.sqrt(1.0 - self.persistence**2)
        y = self.sd * shocks[0]
        logs = []
        for row in steps:
            y = self.persistence * y + innovation * row
            logs.append(y)
        log = np.stack(logs)
        week = np.arange(1, steps.shape[0] + 1)[:, None]
        season = self.amplitude * np.sin(2.0 * math.pi * week / YEAR)
        cpm = self.level * np.exp(season + log)
        decay = self.persistence ** np.arange(1, onward.shape[0] + 1)[:, None]
        ahead = np.exp(
            -season[weeks:] - decay * log[weeks - 1] + self.sd**2 * (1.0 - decay**2) / 2.0
        )
        return Price(self.level, cpm, np.vstack([self.level / cpm[:weeks], ahead]))


def draw(generator: np.random.Generator, channels: Sequence[str]) -> Law:
    """Each of ``channels``' law, its parameters drawn from ``generator`` in the order of the
    ranges: the swing, the persistence and the sd, each over the channels."""
    n = len(channels)
    amplitude = generator.uniform(*AMPLITUDE, n)
    persistence = generator.uniform(*PERSISTENCE, n)
    sd = generator.uniform(*SD, n)
    return Law(np.array([LEVELS[c] for c in channels]), amplitude, persistence, sd)


@dataclass(frozen=True, eq=False)
class Price:
    """A world's prices: each channel's level, ``(channels,)``, its CPM in each week of the
    history, the quarter and the kernel's tail, ``(weeks, channels)``, and what a euro is
    ``expected`` to buy in each of those weeks against what it buys at the level: the history's
    weeks as they were, each later one as the law expects it given the history; None where the
    prices are known in advance."""

    level: Series
    cpm: Series
    expected: Series | None = None

    @property
    def anticipated(self) -> Series:
        """What the best plan reads a euro to buy in each week: :attr:`expected`, or :attr:`buys`
        where the prices are known in advance."""
        return self.buys if self.expected is None else self.expected

    @property
    def buys(self) -> Series:
        """What a euro buys in each week against what it buys at the level, ``level / CPM``."""
        return self.level / self.cpm

    def impressions(self, spend: Series) -> Series:
        """The impressions ``spend``, ``(weeks, channels)`` from the history's first week, buys:
        ``spend / CPM * 1000``."""
        return spend / self.cpm[: spend.shape[0]] * 1000.0


@dataclass(frozen=True, kw_only=True)
class Priced(MediaMixWorld):
    """A history whose channels' curves read the impressions their spend bought: ``buys``,
    ``(weeks, channels)``, what a euro bought in each week against what it buys at the level."""

    buys: Series

    def _effect_of(self, column: int, spend: Series) -> Series:
        return super()._effect_of(column, spend * self.buys[:, column])


def priced(history: MediaMixWorld, buys: Series) -> Priced:
    """``history`` with each channel's curve reading its spend times ``buys``, ``(weeks,
    channels)``, and its media and sales with it; its spend, and all its business read before
    media, as they were."""
    kept = {f.name: getattr(history, f.name) for f in fields(MediaMixWorld)}
    world = Priced(**kept, buys=buys)
    media = np.column_stack(
        [world._effect_of(c, history.spend[:, c]) for c in range(len(history.channels))]
    )
    return replace(world, media=media, sales=history.premedia + media.sum(axis=1))

"""Family 1: Track M v2's drawn worlds with each channel's effect drifting over the weeks.

Each channel's media in week ``t`` are its media in the drawn world times ``m_{c,t} = e^{x_{c,t}}``,
so its effect in that week is ``beta_c m_{c,t}``. The world's budget and bidding read its sales
before media, so the spend path does not move; the sales do, and so does every lift test, which
reads the effect of its own weeks: as the effect drifts, the earlier tests go stale.

Each environment is one law of ``x``, the same for every channel, with each channel's parameters
drawn apart:

* *walk*: ``x_t = x_{t-1} + sigma_c eps_t`` from ``x_0 = 0``, ``sigma_c`` log-uniform on
  ``[0.01, 0.04]`` a week; the expected multiplier ``h`` weeks past the history's last week ``T``
  is ``m_T e^{h sigma_c^2 / 2}``;
* *trend*: ``x_t = kappa_c (t - 1) / (T - 1)``, ``kappa_c`` uniform on ``[-0.7, 0.7]``; the line
  continued;
* *step*: ``x_t = Delta_c 1{t >= tau_c}``, ``tau_c`` uniform on weeks 53 to 143, ``Delta_c`` of
  either sign with equal chance and size uniform on ``[0.2, 0.7]``; ``m_T``;
* *season*: ``x_t = a_c sin(2 pi t / 52 + phase_c)``, ``a_c`` uniform on ``[0.1, 0.4]``; the wave
  continued;
* *revert*: ``x_t = phi_c x_{t-1} + s_c sqrt(1 - phi_c^2) eps_t``, started from its stationary
  law ``N(0, s_c^2)``, ``phi_c`` uniform on ``[0.8, 0.97]`` and ``s_c`` on ``[0.15, 0.4]``;
  ``exp(phi_c^h x_T + s_c^2 (1 - phi_c^{2h}) / 2)``;
* *static*: ``x = 0``; Track M v2's world, bit for bit.

Given its state at ``T``, ``x_{T+h}`` is normal under every law, so its expected multiplier is
``exp(mean + variance / 2)``. The best plan is made for each channel's effect times that, from the
state at ``T``, which the bench knows and no arm does: a plan's worth is linear in the effect, so
the plan best on the expected path is the plan best in expectation. The quarter realises a path of
its own, drawn from the family's continuation stream; where it is another, a plan's regret against
the best plan in hindsight is scored beside it.

The parameters, the history's shocks and the quarter's are drawn from the family's streams
(:mod:`.seeds`) for the world's seed, in that order of roles; the world beneath the drift, its
quarter's processes and its lift tests' noise are Track M v2's for the same seed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from causaldyn_bench.endogenous_mmm import YEAR, MediaMixWorld, Series
from causaldyn_bench.mmm_decision import PLANNED, drawn
from causaldyn_bench.scorecard.continuation import shadow
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, truth

FAMILY = 1


class Law(Protocol):
    """A law of each channel's log multiplier ``x``, on arrays whose last axis is the channels."""

    def start(self, shock: Series) -> Series:
        """``x_0``, the week before the history's first, from a standard normal shock."""
        ...

    def step(self, x: Series, t: int, shock: Series) -> Series:
        """``x_t`` from ``x_{t-1}`` and a standard normal shock."""
        ...

    def moments(self, x: Series, t: int, h: int) -> tuple[Series, Series]:
        """The mean and the variance of ``x_{t+h}`` given ``x_t = x``."""
        ...


@dataclass(frozen=True)
class Walk:
    """A random walk in ``x``, from 0 before the first week."""

    sd: Series  # each channel's, a week

    def start(self, shock: Series) -> Series:
        # CHOICE: at the drawn world's effect the week before the history's first
        return np.zeros_like(shock)

    def step(self, x: Series, t: int, shock: Series) -> Series:
        return x + self.sd * shock

    def moments(self, x: Series, t: int, h: int) -> tuple[Series, Series]:
        return x, np.broadcast_to(h * self.sd**2, np.shape(x))


@dataclass(frozen=True)
class Revert:
    """A stationary first-order autoregression in ``x``, started from its stationary law."""

    persistence: Series
    sd: Series  # the stationary sd, not the shock's

    def start(self, shock: Series) -> Series:
        # CHOICE: from its stationary law, so the history holds no transient
        return self.sd * shock

    def step(self, x: Series, t: int, shock: Series) -> Series:
        return self.persistence * x + self.sd * np.sqrt(1.0 - self.persistence**2) * shock

    def moments(self, x: Series, t: int, h: int) -> tuple[Series, Series]:
        decay = self.persistence**h
        return decay * x, np.broadcast_to(self.sd**2 * (1.0 - decay**2), np.shape(x))


class Fixed:
    """A law whose path is set in advance: ``x_t`` is :meth:`level` at ``t`` whatever the shock."""

    def level(self, t: int) -> Series:
        raise NotImplementedError

    def start(self, shock: Series) -> Series:
        return np.zeros_like(shock)

    def step(self, x: Series, t: int, shock: Series) -> Series:
        return np.broadcast_to(self.level(t), np.shape(shock))

    def moments(self, x: Series, t: int, h: int) -> tuple[Series, Series]:
        return np.broadcast_to(self.level(t + h), np.shape(x)), np.zeros(np.shape(x))


@dataclass(frozen=True)
class Trend(Fixed):
    """A line in ``x`` from 0 in the first week to ``slope`` in the history's last."""

    slope: Series
    weeks: int  # the history's

    def level(self, t: int) -> Series:
        return self.slope * (t - 1) / (self.weeks - 1)


@dataclass(frozen=True)
class Step(Fixed):
    """``x`` steps from 0 to ``size`` in week ``when``."""

    when: NDArray[np.int64]
    size: Series

    def level(self, t: int) -> Series:
        return np.where(t >= self.when, self.size, 0.0)


@dataclass(frozen=True)
class Season(Fixed):
    """A yearly wave in ``x``."""

    amplitude: Series
    phase: Series

    def level(self, t: int) -> Series:
        return self.amplitude * np.sin(2.0 * math.pi * t / YEAR + self.phase)


@dataclass(frozen=True)
class Static(Fixed):
    """No drift."""

    channels: int

    def level(self, t: int) -> Series:
        return np.zeros(self.channels)


def _walk(draw: np.random.Generator, channels: int, weeks: int) -> Law:
    return Walk(np.exp(draw.uniform(math.log(0.01), math.log(0.04), channels)))


def _trend(draw: np.random.Generator, channels: int, weeks: int) -> Law:
    return Trend(draw.uniform(-0.7, 0.7, channels), weeks)


def _step(draw: np.random.Generator, channels: int, weeks: int) -> Law:
    when = draw.integers(53, 144, channels)
    # CHOICE: each sign with equal chance
    sign = np.where(draw.random(channels) < 0.5, -1.0, 1.0)
    return Step(when, sign * draw.uniform(0.2, 0.7, channels))


def _season(draw: np.random.Generator, channels: int, weeks: int) -> Law:
    # CHOICE: the phase uniform over the year
    return Season(draw.uniform(0.1, 0.4, channels), draw.uniform(0.0, 2.0 * math.pi, channels))


def _revert(draw: np.random.Generator, channels: int, weeks: int) -> Law:
    return Revert(draw.uniform(0.8, 0.97, channels), draw.uniform(0.15, 0.4, channels))


def _static(draw: np.random.Generator, channels: int, weeks: int) -> Law:
    return Static(channels)


# each law's parameters for each channel, in the order of the family's environments
LAWS: dict[str, Callable[[np.random.Generator, int, int], Law]] = {
    "walk": _walk,
    "trend": _trend,
    "step": _step,
    "season": _season,
    "revert": _revert,
    "static": _static,
}


def law(environment: str, seed: int, channels: int, weeks: int) -> Law:
    """The law of ``environment`` with each channel's parameters drawn for world ``seed``."""
    if environment not in LAWS:
        raise ValueError(f"the drift family has no law {environment!r}")
    return LAWS[environment](rng(FAMILY, Role.PARAMETERS, seed), channels, weeks)


def onward(moving: Law, x: Series, week: int, shocks: Series) -> Series:
    """``x`` over the weeks after ``week``, one a row of ``shocks``, from ``x`` in ``week``."""
    out = []
    for h in range(1, shocks.shape[0] + 1):
        x = moving.step(x, week + h, shocks[h - 1])
        out.append(x)
    return np.stack(out)


def path(moving: Law, shocks: Series) -> Series:
    """``x`` over the history's weeks, from 1: ``shocks``' first row starts it, one row a week."""
    return onward(moving, moving.start(shocks[0]), 0, shocks[1:])


def expected(moving: Law, x: Series, week: int, horizon: int) -> Series:
    """The expected multiplier ``e^x`` over the ``horizon`` weeks after ``week`` given ``x`` in it,
    ``(horizon, channels)``: ``x`` is normal there, so it is ``exp(mean + variance / 2)``."""
    out = []
    for h in range(1, horizon + 1):
        mean, variance = moving.moments(x, week, h)
        out.append(np.exp(mean + variance / 2.0))
    return np.stack(out)


@dataclass(frozen=True, kw_only=True)
class Drifted(MediaMixWorld):
    """A history whose channels' media are the drawn world's times ``multiplier``,
    ``(weeks, channels)``; its geo tests read the multiplier of their weeks."""

    multiplier: Series

    def _effect_of(self, column: int, spend: Series) -> Series:
        return self.multiplier[:, column] * super()._effect_of(column, spend)


def drift(history: MediaMixWorld, multiplier: Series) -> Drifted:
    """``history`` with each channel's media times ``multiplier``, and its sales with them."""
    media = history.media * multiplier
    kept = {f.name: getattr(history, f.name) for f in fields(MediaMixWorld)}
    kept |= {"media": media, "sales": history.premedia + media.sum(axis=1)}
    return Drifted(**kept, multiplier=multiplier)


class Drift:
    """Family 1."""

    name = "drift"
    stream = stream(FAMILY)
    labels = (
        "The walk law is a dynamic linear model's own state law: a random walk in the log of each "
        "channel's effect. The other laws, and every law's ranges, are the bench's own.",
        "Beneath the drift each world is a Track M v2 drawn world, whose media are "
        "PyMC-Marketing's default class; under the static law it is that world, bit for bit.",
        *TRACK_M2.labels[1:],
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(LAWS)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(LAWS)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in LAWS:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = drawn(seed)
        base = generator.simulate(seed)
        weeks, channels = base.spend.shape
        horizon = PLANNED + base.kernel_length - 1
        moving = law(environment, seed, channels, weeks)
        x = path(moving, rng(FAMILY, Role.PATH, seed).standard_normal((weeks + 1, channels)))
        history = drift(base, np.exp(x))
        shocks = rng(FAMILY, Role.CONTINUATION, seed).standard_normal((horizon, channels))
        effect = np.asarray(base.effect, dtype=float)[:, None]
        anticipated = effect * expected(moving, x[-1], weeks, horizon).T
        realised = effect * np.exp(onward(moving, x[-1], weeks, shocks)).T
        quarter = shadow(generator, history, seed, realised)
        return World(environment, seed, generator, history, quarter, anticipated, realised)

    def observe(self, world: World, k: int) -> Observation:
        return observation(FAMILY, world, k)

    def truth(self, world: World) -> Truth:
        return truth(world)


DRIFT = Drift()

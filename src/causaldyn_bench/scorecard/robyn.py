"""Family 14: Robyn's home ground, worlds whose media are in Robyn's own class, drawn by the bench.

Robyn ships no simulator and no prior: it searches its hyperparameters over ranges and fits its
coefficients by a ridge. So the bench writes the world in Robyn's form and draws it on the Robyn
arm's own ranges. Each channel's media in week ``t`` are

    m_{c,t} = beta_c x_{c,t}^{a_c} / (x_{c,t}^{a_c} + K_c^{a_c}),
    x_{c,t} = s_{c,t} + theta_c x_{c,t-1},  x_{c,1} = s_{c,1},

Robyn 3.12.1's ``saturation_hill`` of its ``adstock_geometric``: the adstock unnormalised and
carried over every later week, and the curve's inflexion ``K_c = gamma_c max_t x_{c,t}``, gamma
times the channel's peak adstock over the history, Robyn's modelling window as the arm sets it.
Once drawn, ``K_c`` is the world's: a geo test's treated universe and the quarter read the same
curve.

* ``theta_c``, ``a_c`` and ``gamma_c`` are uniform on the Robyn arm's genre ranges, the demo's for
  the demo channel each world channel stands for: decay 0-0.3, shape 0.5-3 and gamma 0.3-1 for pla
  and meta, a search and a social channel, and 0.3-0.8, 0.5-1 and 0.3-1 for tv;
* ``beta_c`` (CHOICE) gives each channel the media the Track M v2 world beneath it returns over the
  history, so the sales, the media's share of them and each channel's return per euro over the
  history are Track M v2's, and only the media's form is Robyn's.

Beneath the media each world is Track M v2's drawn world for its seed: its base, its spend, whose
budget and bidding rules read the sales before media and so do not move, its measurement layer and
the quarter's processes. Robyn's kernel takes no length; every arm is told it spans the history,
``L = 156`` weeks, the length the oracle and the returns read it at, so a plan's worth and a
return count what spend returns over the ``L - 1`` weeks after the quarter or the window. Robyn's
carryover runs on past them, and a Hill of shape below 1 rises steeply from zero, so what it
returns later is left out for every plan alike: on drawn spend, under 4e-8 of a channel's worth
and 2e-9 of its return per euro at the ranges' corner, decay 0.8 and shape 0.5, and under 1e-13
at shape 1. The tests read that carryover too, which never dies out, so the ladder's rungs nest
only up to what an earlier test carries into a later test's readout, at least 28 weeks on.

``scripts/robyn_form.py`` runs Robyn's own functions in the Robyn arm's pinned library on worlds of
this family and records what they return in ``results/robyn_form.json``; a test holds the world's
transforms to that record within 1e-9.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import lfilter

from causaldyn_bench.endogenous_mmm import Curve, MediaMixWorld, Series
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.continuation import shadow
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, still, truth

FAMILY = 14
# the Robyn arm's genre ranges (scripts/robyn_arm.R), lowest and highest: each channel's decay
# (Robyn's thetas), shape (alphas) and gamma (gammas)
GENRE: dict[str, tuple[tuple[float, float], ...]] = {
    "pla": ((0.0, 0.3), (0.5, 3.0), (0.3, 1.0)),
    "meta": ((0.0, 0.3), (0.5, 3.0), (0.3, 1.0)),
    "tv": ((0.3, 0.8), (0.5, 1.0), (0.3, 1.0)),
}


def adstocked(spend: ArrayLike, decay: float) -> Series:
    """Robyn's ``adstock_geometric``: ``x_t = s_t + decay x_{t-1}`` from ``x_1 = s_1``."""
    return lfilter([1.0], [1.0, -decay], np.asarray(spend, dtype=float))


def saturated(adstock: ArrayLike, shape: float, inflexion: float) -> Series:
    """Robyn's ``saturation_hill`` at ``inflexion``: ``x^shape / (x^shape + inflexion^shape)``."""
    rising = np.asarray(adstock, dtype=float) ** shape
    return rising / (rising + inflexion**shape)


def hill(shape: float) -> Curve:
    """Robyn's Hill of ``shape`` as the scorecard reads a curve, at its inflexion. Its slope is
    read where the adstock is positive and is 0 where it is not: there a cell's reach and a
    window's adstock are 0 too, and below a shape of 1 the slope at 0 is infinite."""

    def slope(adstock: Series, inflexion: float) -> Series:
        a = np.asarray(adstock, dtype=float)
        rising, floor = a**shape, inflexion**shape
        return np.divide(
            shape * rising * floor,
            a * (rising + floor) ** 2,
            out=np.zeros_like(a),
            where=a > 0.0,
        )

    return Curve(lambda a, inflexion: saturated(a, shape, inflexion), slope, shape <= 1.0)


@dataclass(frozen=True, kw_only=True)
class RobynWorld(MediaMixWorld):
    """A history whose media are in Robyn's form: :attr:`retention` holds each channel's decay,
    :attr:`saturation` its inflexion, in thousands of euros of adstock, and :attr:`effect` its
    coefficient; the kernel spans the history."""

    shape: tuple[float, ...]  # Robyn's alphas
    gamma: tuple[float, ...]  # Robyn's gammas: each inflexion over its channel's peak adstock

    def kernel(self, column: int) -> Series:
        return self.retention[column] ** np.arange(self.kernel_length, dtype=float)

    def curves(self) -> tuple[Curve, ...]:
        return tuple(hill(shape) for shape in self.shape)

    def _effect_of(self, column: int, spend: Series) -> Series:
        adstock = adstocked(spend, self.retention[column])
        return self.effect[column] * saturated(adstock, self.shape[column], self.saturation[column])


def parameters(channels: tuple[str, ...], seed: int) -> tuple[Series, Series, Series]:
    """Each channel's decay, shape and gamma for world ``seed``, uniform on its genre's ranges."""
    low, high = np.array([GENRE[name] for name in channels]).transpose(2, 1, 0)
    draw = rng(FAMILY, Role.PARAMETERS, seed)
    decay = draw.uniform(low[0], high[0])
    shape = draw.uniform(low[1], high[1])
    gamma = draw.uniform(low[2], high[2])
    return decay, shape, gamma


def robyn(history: MediaMixWorld, decay: Series, shape: Series, gamma: Series) -> RobynWorld:
    """``history`` with each channel's media in Robyn's form at ``decay``, ``shape`` and
    ``gamma``, each channel returning over the history what it returned there, and the sales with
    them."""
    adstock = np.column_stack(
        [adstocked(history.spend[:, c], float(d)) for c, d in enumerate(decay)]
    )
    inflexion = gamma * adstock.max(axis=0)
    curve = np.column_stack(
        [saturated(adstock[:, c], float(shape[c]), float(inflexion[c])) for c in range(len(decay))]
    )
    effect = history.media.sum(axis=0) / curve.sum(axis=0)
    media = effect * curve
    kept = {f.name: getattr(history, f.name) for f in dataclasses.fields(MediaMixWorld)}
    kept |= {
        "media": media,
        "sales": history.premedia + media.sum(axis=1),
        "retention": tuple(decay.tolist()),
        "saturation": tuple(inflexion.tolist()),
        "effect": tuple(effect.tolist()),
        "kernel_length": history.week.size,
        "curve": "hill",  # each channel's of its own shape (curves), no key of CURVES
    }
    return RobynWorld(**kept, shape=tuple(shape.tolist()), gamma=tuple(gamma.tolist()))


class Robyn:
    """Family 14."""

    name = "Robyn's home ground"
    stream = stream(FAMILY)
    labels = (
        "The world's media are Robyn's class, by the bench's generator: each channel's spend "
        "through Robyn 3.12.1's geometric adstock, carried over every later week, and its Hill, "
        "at gamma times the channel's peak adstock over the history; decay, shape and gamma drawn "
        "uniformly on the Robyn arm's genre ranges.",
        "Every arm is told the kernel spans the history, as Robyn's adstock does.",
        "Beneath the media each world is a Track M v2 drawn world, each channel returning over the "
        "history what it returns there.",
        TRACK_M2.labels[2],
    )
    pilots = MappingProxyType({"genre": pilot(FAMILY, 0)})
    scored = MappingProxyType({"genre": scored(FAMILY, 0)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in self.scored:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = drawn(seed)
        base = generator.simulate(seed)
        history = robyn(base, *parameters(base.channels, seed))
        path = still(history)
        return World(
            environment,
            seed,
            generator,
            history,
            shadow(generator, history, seed, path),
            path,
            path,
        )

    def observe(self, world: World, k: int) -> Observation:
        return observation(FAMILY, world, k)

    def truth(self, world: World) -> Truth:
        return truth(world)


ROBYN = Robyn()

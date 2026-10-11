"""Family 6: Track M v2's drawn worlds whose curves change outside the range the history shows.

Each channel's history shows its curve over the range of its adstock, ``[a_min, a_max]``, over
every week of the history, the first weeks' too, so its own media are the drawn world's, bit for
bit. Outside that range the curve ``g`` is the bench's own, continuous at both ends:

* below ``a_min`` a cliff, ``g(a_min) (a / a_min)^p``, with each channel's ``p`` uniform on
  ``[2, 4]``, drawn from the family's parameter stream for the world's seed; its slope drops at
  ``a_min``, a kink the oracle reads it at;
* above ``a_max`` an early saturation, ``g(a_max) + g'(a_max) h (1 - e^{-(a - a_max) / h})``, its
  slope continuous; CHOICE: ``h = a_max / 4``, so past the range the curve rises at most by its
  slope at ``a_max`` over a quarter of ``a_max``.

Each environment replaces one side or both: *cliff*, *saturation* and *both*. Each channel's box
widens from ``[0.5, 2]`` to ``[0.25, 3]`` times its mean, so a plan can leave the range at either
end, and the arms read the widened box. A plan is scored on the replaced curves, a geo test reads
them in its dark weeks, which leave the range from below, and each channel's returns read them on
the window's spend removed; so does the kernel's tail after the quarter, where nothing is spent.

The worth of a cell with a cliff is not concave, and the oracle solves it exactly to its gap by
branch and bound on the curves' bends and kinks (:func:`causaldyn_bench.scorecard.truth.oracle`).
Beneath its curves each world is Track M v2's drawn world for its seed, its quarter's processes and
its lift tests' noise among it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType

import numpy as np

from causaldyn_bench.endogenous_mmm import Curve, MediaMixWorld, Series
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, settled, truth

FAMILY = 6
# whether each environment's curve is replaced below the range, and above it
SIDES: MappingProxyType[str, tuple[bool, bool]] = MappingProxyType(
    {"cliff": (True, False), "saturation": (False, True), "both": (True, True)}
)
WIDENED = (0.25, 3.0)  # each channel's box, times its mean
POWERS = (2.0, 4.0)  # the cliff's exponent, uniform between
EARLY = 0.25  # the saturation's scale over the range's top


def beyond(base: Curve, low: float, high: float, power: float | None, scale: float | None) -> Curve:
    """``base`` on ``[low, high]``; below ``low`` the cliff of exponent ``power``, and above
    ``high`` the saturation of scale ``scale``, each where it is given."""

    ends = np.array([low, high])

    def value(a: Series, lam: float) -> Series:
        a = np.asarray(a, dtype=float)
        out = base.value(a, lam)
        at, rate = base.value(ends, lam), base.slope(ends, lam)
        if power is not None:
            out = np.where(a < low, at[0] * (a / low) ** power, out)
        if scale is not None:
            rise = rate[1] * scale * -np.expm1(-(a - high) / scale)
            out = np.where(a > high, at[1] + rise, out)
        return out

    def slope(a: Series, lam: float) -> Series:
        a = np.asarray(a, dtype=float)
        out = base.slope(a, lam)
        at, rate = base.value(ends, lam), base.slope(ends, lam)
        if power is not None:
            out = np.where(a < low, at[0] * power * (a / low) ** (power - 1.0) / low, out)
        if scale is not None:
            out = np.where(a > high, rate[1] * np.exp(-(a - high) / scale), out)
        return out

    def bend(lam: float) -> float:
        # the cliff is convex; the saturation's -g'' is greatest at the range's top
        if base.bend is None:
            raise ValueError("the curve replaced outside the range states its bend")
        own = base.bend(lam)
        return own if scale is None else max(own, float(base.slope(ends, lam)[1]) / scale)

    return Curve(
        value,
        slope,
        concave=base.concave and power is None,
        bend=bend,
        kinks=() if power is None else (low,),
    )


@dataclass(frozen=True, kw_only=True)
class Replaced(MediaMixWorld):
    """A history whose channels' curves are :attr:`replaced` outside :attr:`observed`, each the
    range of the channel's adstock over the history, on :attr:`sides`, below and above; a geo test
    that takes the adstock there reads the replaced curve, and elsewhere the drawn world's media."""

    observed: tuple[tuple[float, float], ...]
    sides: tuple[bool, bool]
    replaced: tuple[Curve, ...]

    def curves(self) -> tuple[Curve, ...]:
        return self.replaced

    def _effect_of(self, column: int, spend: Series) -> Series:
        inside = super()._effect_of(column, spend)
        adstock = np.convolve(spend, self.kernel(column))[: spend.size]
        low, high = self.observed[column]
        lam = self.saturation[column]
        below, above = self.sides
        out = (below & (adstock < low)) | (above & (adstock > high))
        return np.where(
            out, self.effect[column] * self.replaced[column].value(adstock, lam), inside
        )


def replaced(history: MediaMixWorld, sides: tuple[bool, bool], powers: Series) -> Replaced:
    """``history`` with each channel's curve replaced outside its adstock's range on ``sides``,
    below and above, the cliff's exponent ``powers``, one a channel."""
    below, above = sides
    observed, curves = [], []
    for c, base in enumerate(history.curves()):
        adstock = np.convolve(history.spend[:, c], history.kernel(c))[: history.spend.shape[0]]
        low, high = float(adstock.min()), float(adstock.max())
        observed.append((low, high))
        power = float(powers[c]) if below else None
        curves.append(beyond(base, low, high, power, EARLY * high if above else None))
    kept = {f.name: getattr(history, f.name) for f in fields(MediaMixWorld)}
    return Replaced(**kept, observed=tuple(observed), sides=sides, replaced=tuple(curves))


class Outside:
    """Family 6."""

    name = "truth outside the observed range"
    stream = stream(FAMILY)
    labels = (
        "Outside the range of each channel's adstock its history shows, the curve is the bench's "
        "own, which favours plans that stay in the data.",
        "Beneath its curves each world is a Track M v2 drawn world, whose media are "
        "PyMC-Marketing's default class; its history is that world's, bit for bit.",
        *TRACK_M2.labels[1:],
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(SIDES)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(SIDES)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in SIDES:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = drawn(seed)
        base = generator.simulate(seed)
        powers = rng(FAMILY, Role.PARAMETERS, seed).uniform(*POWERS, len(base.channels))
        return settled(environment, seed, generator, replaced(base, SIDES[environment], powers))

    def observe(self, world: World, k: int) -> Observation:
        return observation(FAMILY, world, k, WIDENED)

    def truth(self, world: World) -> Truth:
        return truth(world, box=WIDENED)


OUTSIDE = Outside()

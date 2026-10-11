"""Family 4: Track M v2's drawn worlds whose every channel's curve is S-shaped.

Each environment puts one of the generator's S-shaped curves
(:data:`~causaldyn_bench.endogenous_mmm.CURVES`) in his ``tanh``'s place, at half its ceiling where
his is, at ``ln 3 / lambda``: Hill of slope 2, Weibull of shape 2 and the logistic of steepness 4.
A seed's spend, baseline and mechanisms do not read the curve, so its three worlds differ in their
media and sales alone, and so do their tests.

A channel's worth is then not concave in its weekly spend, so the best plan in the box may hold a
channel at a bound that a planner reading a concave curve would never choose, and a plan that is
best only locally is not best. The oracle solves it exactly to its gap, by branch and bound on the
curves' bends (:func:`causaldyn_bench.scorecard.truth.oracle`). Beneath its curve each world is
Track M v2's drawn world for its seed, its quarter's processes and its lift tests' noise among it:
the family draws no variate of its own.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import pilot, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, settled, truth

FAMILY = 4
SHAPES = ("hill-2", "weibull-2", "logistic-4")  # keys of CURVES, each S-shaped


class SCurves:
    """Family 4."""

    name = "S-curves"
    stream = stream(FAMILY)
    labels = (
        "Each world's curve is S-shaped, which PyMC-Marketing's default saturation is not; the "
        "hill-2 worlds' curve is in the Hill class Meridian and Robyn fit.",
        "Beneath its curve each world is a Track M v2 drawn world, whose adstock is "
        "PyMC-Marketing's default: the normalised geometric adstock at the kernel's length.",
        *TRACK_M2.labels[1:],
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(SHAPES)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(SHAPES)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in SHAPES:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = dataclasses.replace(drawn(seed), curve=environment)
        return settled(environment, seed, generator, generator.simulate(seed))

    def observe(self, world: World, k: int) -> Observation:
        return observation(FAMILY, world, k)

    def truth(self, world: World) -> Truth:
        return truth(world)


S_CURVES = SCurves()

"""A geo world's KPI noise, wherever the bench draws it: the history, each test's universe and the
quarter at the status quo.

:mod:`chc.geo_world` makes each head's KPI a week independent with standard deviation ``noise``, so
a geo's noise is normal with standard deviation ``noise sqrt(P)``. A geo's level grows as ``P`` and
its noise as ``sqrt(P)``, so a small geo's noise is large against its level. Where it is about half
the level, a normal draw puts a geo-week's KPI below 0 a few times in a world, and a KPI that is
sales cannot be. A :class:`LogNormalNoise` mix keeps chc's mean and standard deviation in each
geo-week and draws the KPI log-normal about its level instead:

    level exp(s z - s^2 / 2),  s^2 = log(1 + (noise sqrt(P) / level)^2)

from the same standard normal ``z``, so the KPI is above 0 by construction (Maxima: mean ``level``,
variance ``(noise sqrt(P))^2``). Every other mix keeps chc's normal noise, bit for bit.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from chc.geo_world import GeoMediaMix, GeoMediaWorld


@dataclass(frozen=True, eq=False)
class LogNormalNoise(GeoMediaMix):
    """chc's mix, its KPI noise log-normal about each geo-week's level at chc's standard deviation;
    see the module.

    Raises:
        ValueError: on what chc's mix refuses, or no noise.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.noise > 0.0:
            raise ValueError(f"log-normal noise needs a noise a head above 0, got {self.noise!r}")

    def draw(self, seed: int) -> GeoMediaWorld:
        """chc's world from ``seed``, its noise read from the same normal variates."""
        world = super().draw(seed)
        shocks = world.noise / (self.noise * np.sqrt(world.population)[:, None])
        level = world.base + world.media.sum(axis=2) + world.spilled
        return dataclasses.replace(world, noise=kpi_noise(world, level, shocks))


def kpi_noise(world: GeoMediaWorld, level: np.ndarray, shocks: np.ndarray) -> np.ndarray:
    """``(geos, weeks)``: the noise of a KPI whose mean is ``level``, from standard normal
    ``shocks``, each ``(geos, weeks)``: chc's, ``noise sqrt(P) shocks``, or log-normal about the
    level at that standard deviation where ``world``'s mix is a :class:`LogNormalNoise`."""
    size = world.mix.noise * np.sqrt(world.population)[:, None]
    if not isinstance(world.mix, LogNormalNoise):
        return size * shocks
    spread = np.sqrt(np.log1p((size / level) ** 2))
    return level * np.expm1(spread * shocks - spread**2 / 2.0)

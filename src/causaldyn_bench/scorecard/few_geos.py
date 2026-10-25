"""Family 10: few geos, family 8 at three geos, at eight, and at fifty small, noisy ones.

Each world is a world of family 8 (:mod:`~causaldyn_bench.scorecard.geo`) at policy 0, its first
environment, but for its count of geos and, at fifty, their size and noise. Its channels, effects
and their spreads, bases, season, spend and noise a head are family 8's, and so are its decision
over the cells, its oracle, status quo and equal split, its ladder of geo tests, its export and its
labels. Each environment draws its worlds from a block of seeds of its own, and each channel's
spread from the family's own stream, as every family's do:

    geos-3   three geos of family 8's, a median of 1e5 people, noise 230 a head   7 cells
    geos-8   eight of them                                                      17 cells
    geos-50  fifty small, noisy geos, a median of 2e4 people, noise 460 a head  101 cells
             log-normal about each geo-week's level

CHOICE: fifty small geos hold family 8's ten in expectation, a fifth of the median population at
the same log-SD of 1 (Maxima). A geo's spend and its lift scale with its people, so the market is
family 8's, and so is media's share of the KPI: on the first 30 pilot worlds of each, a market of
1.64e6 people against family 8's 1.68e6 at policy 0, and media 0.161 of the KPI against 0.163. A
small geo's noise a head is twice family 8's, so a median geo's noise over its mean KPI is
``2 sqrt(5)``, 4.47, times family 8's (Maxima): 0.141 against 0.032, and the median over the geos
0.148 against 0.033. The market's noise is twice family 8's. A fifth of the people at family 8's
noise a head was rejected: a geo ``sqrt(5)`` times as noisy and a market no noisier is small, not
noisy. At three and eight geos a geo is family 8's: media 0.162 and 0.167 of the KPI, a geo's noise
0.032 and 0.031 of its mean KPI.

CHOICE: at fifty a geo-week's noise is log-normal about its level, at the standard deviation chc's
normal noise gives it (:class:`~causaldyn_bench.scorecard.geo_noise.LogNormalNoise`), so every
geo's KPI is above 0 by construction. A geo's level grows as its people and its noise as their
square root, and the smallest of fifty geos at a log-SD of 1 holds about a tenth of the median's
people, its noise about half its level: a world's largest ratio of a geo-week's noise to its
level is 0.51 in the median of the first 30 pilot worlds and 0.85 at the most. Drawn normal,
24 of those 30 worlds had a geo-week below 0, which a KPI that is sales cannot have and Meridian
refuses. The law keeps each geo-week's mean and standard deviation, so media's share and the noise
ratios above are the normal noise's, and the tests' numbers below are read on it. At three and
eight geos a geo's noise is at most 0.16 of its level on their first 30 pilot worlds, and their
noise stays chc's normal noise, bit for bit, as family 8's does.

The ladder is family 8's (:func:`~causaldyn_bench.scorecard.ladder.geo_rung`) at every count. On
three geos the alternate geos a test takes dark are geos 0 and 2 in the first and third test and
geo 1 alone in the second and fourth, each read against the others, and a test's standard error is
read in time from placebo readings whose count does not depend on the geos: every test reads, and
every one is exported. The tests are weak: on the first 30 pilot worlds of each environment the
median test's standard error is 2.19 times its true change on three geos, 1.13 on eight, 1.45 on
fifty small ones, and 0.98 on family 8's ten. Their intervals hold: at k = 4 on those worlds a 95 %
interval holds the true change for 0.925 to 0.950 of a channel's tests on three geos, 0.933 to
0.958 on eight and 0.917 to 0.975 on fifty, where family 8's hold 0.917 to 0.950. A test of one geo
is no worse than a test of two: on the first 100 pilot worlds of three geos the tests of geo 1
alone hold 0.938 and those of geos 0 and 2 hold 0.957, where family 8's tests in the same weeks hold
0.925 and 0.952 on its 100; without the season they hold 0.948 and 0.938. A rule that exports no
test of one geo was rejected: on three geos it drops half of every rung's tests for a coverage that
family 8's own tests share.

Every cell is concave, as in family 8, so the oracle is exact.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import NamedTuple

from causaldyn_bench.scorecard.geo import NOISE, POPULATION, Geo, World, drawn, mix
from causaldyn_bench.scorecard.seeds import pilot, scored, stream

FAMILY = 10
SMALL = POPULATION / 5  # a small geo's median population: fifty of them hold ten of family 8's
NOISY = 2 * NOISE  # a small geo's noise a head a week


class Geos(NamedTuple):
    """An environment's geos: how many, their median population, their noise a head a week, and
    whether that noise is log-normal about a geo-week's level."""

    count: int
    population: float
    noise: float
    lognormal: bool = False


SIZES: MappingProxyType[str, Geos] = MappingProxyType(
    {
        "geos-3": Geos(3, POPULATION, NOISE),
        "geos-8": Geos(8, POPULATION, NOISE),
        "geos-50": Geos(50, SMALL, NOISY, lognormal=True),
    }
)


class FewGeos(Geo):
    """Family 10: family 8 at another count of geos."""

    name = "few geos"
    stream = stream(FAMILY)
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(SIZES)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(SIZES)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in SIZES:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        geos = SIZES[environment]
        generator = mix(
            0.0,
            seed,
            geos.count,
            family=FAMILY,
            population=geos.population,
            noise=geos.noise,
            lognormal=geos.lognormal,
        )
        return drawn(environment, seed, generator, FAMILY)


FEW_GEOS = FewGeos()

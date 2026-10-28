"""Family 9: spillover, family 8's geo worlds whose geos' own media lift their neighbours.

Each world is a world of family 8 (:mod:`~causaldyn_bench.scorecard.geo`) at policy 0, its first
environment, whose ten geos stand on a ring in their order, a stand-in for geography: a geo's
search and social lift each of its two ring neighbours by ``s / 2`` of what they lift at home, the
two together ``s`` of it, and TV, bought nationally, does not spill (:mod:`chc.geo_world`'s
``spillover``). Its channels, effects and their spreads, bases, season, spend and noise are family
8's, and so are its decision over the cells, its status quo and equal split, its ladder of geo
tests and its export. Each environment draws its worlds from a block of seeds of its own, and each
channel's spread from the family's own stream, as every family's do:

    spillover-0.2   s = 0.2
    spillover-0.5   s = 0.5

The truth counts what spills. A geo's own channel's cell is worth ``1 + s`` times its lift at home
(:func:`~causaldyn_bench.scorecard.truth.geo_parts`), so the oracle buys the spill, and it stays
exact: every cell is concave, as in family 8. The quarter at the status quo holds the spilled lift
(:func:`~causaldyn_bench.scorecard.continuation.geo_shadow`), and a channel's returns are the
market's (:func:`~causaldyn_bench.scorecard.returns.true_geo_returns`), search's and social's
``1 + s`` times their returns at home. On the first 30 pilot worlds of each environment media are
0.173 and 0.196 of the KPI, against 0.155 and 0.152 on the same worlds without spillover and 0.163
on family 8's own first 30 pilot worlds at policy 0, and what spills is 0.126 and 0.263 of media's
lift. On the mean over those worlds a euro of the last year's search returns 1.49 and 1.75, against
1.25 and 1.16 without spillover, social 1.00 and 1.18 against 0.83 and 0.79, and TV 0.50 in both,
with spillover or without; the marginal euro returns 0.97 and 1.13 against 0.81 and 0.75 for search,
0.55 and 0.63 against 0.46 and 0.42 for social, and 0.42 and 0.42 for TV either way. Family 8's
return 1.23, 0.79 and 0.49 a euro, 0.76, 0.42 and 0.42 at the margin.

The ladder is family 8's (:func:`~causaldyn_bench.scorecard.ladder.geo_rung`), read as in family 8.
A test takes alternate geos of the ring dark, so a tested geo's neighbours are controls: it loses
its home lift and nothing that spills into it, and the controls lose ``s`` times that, what the
tested geos no longer spill into them, as controls next to tested geos do in practice. The
difference in differences scales the controls' change by its slope ``b`` and takes it off: on a
world still but for its season a search or social reading is ``1 - b s`` times the tested geos'
change exactly, ``b`` the tested geos' base over the controls'. Over the first 30 pilot worlds'
tests ``b`` is 1.00 at the median and 1.39 and 1.36 on the mean, the populations log-normal, so
``-b s`` is -0.28 and -0.68 on the mean, and read on those worlds without noise a search or social
reading is off the tested geos' change, relative to it, by -0.25 on the mean for both at
``s = 0.2`` and by -0.67 for both at 0.5, where the same worlds without spillover are off by +0.04
for both, and by +0.07 and +0.18, from the spend's weekly jitter. TV does not spill, and its
readings take no bias from it: off by +0.01 and +0.02, against +0.01 and +0.14 on the same worlds
without spillover, the jitter's error at a standard deviation of 0.36 to 0.63.

CHOICE: a test's true change is the tested geos' change, what their KPI loses over the readout with
them dark: their home lift, since nothing that spills into them moves. It is the change the
experiment's design targets and the one family 8's tests are read against. The alternative is the
market's change, home and spill, ``1 + s`` times it, which a cell's worth counts and a plan needs;
read against it, a search or social reading is low by 0.38 for both at ``s = 0.2`` and by 0.78 at
0.5, on those worlds without noise. Neither is moved into the reading: the bias is spillover's own,
the one a geo experiment whose controls neighbour its tested geos meets.

With noise, at k = 4 on the first 30 pilot worlds of each environment, a reading's mean error
relative to the tested geos' change is -0.28, -0.15 and +0.03 for search, social and TV at
``s = 0.2`` and -0.68, -0.36 and -0.12 at 0.5, where the same worlds without spillover read +0.00,
+0.14 and +0.03, and +0.06, +0.48 and +0.00, and family 8's +0.08, +0.07 and +0.03: social's
standard error is 1.5 times its change at the median test, so its mean relative error is mostly
noise. The standard error is family 8's, read in time from placebo readings that hold no test and
so no bias, and the sharpest tests are the first to miss: search's standard error is 0.45 and 0.49
of its change at the median test. A 95 % interval holds the tested geos' change for 0.875, 0.967
and 0.950 of search's, social's and TV's tests at ``s = 0.2`` and 0.650, 0.883 and 0.950 at 0.5,
against 0.933, 0.983 and 0.925, and 0.950, 0.933 and 0.967, on the same worlds without spillover,
and family 8's 0.950, 0.950 and 0.917. On the first 100 pilot worlds they hold 0.887, 0.948 and
0.955, and 0.640, 0.915 and 0.940, against 0.950, 0.975 and 0.945, and 0.950, 0.945 and 0.955, and
family 8's 0.925, 0.940 and 0.950. An interval holds the market's change for 0.767, 0.950 and 0.950
of the tests at ``s = 0.2`` and 0.408, 0.883 and 0.950 at 0.5 on the first 30.
"""

from __future__ import annotations

from types import MappingProxyType

from causaldyn_bench.scorecard.geo import Geo, World, drawn, mix
from causaldyn_bench.scorecard.seeds import pilot, scored, stream

FAMILY = 9
SPILLOVERS: MappingProxyType[str, float] = MappingProxyType(
    {"spillover-0.2": 0.2, "spillover-0.5": 0.5}
)


class Spillover(Geo):
    """Family 9: family 8 whose geos' own media lift their ring neighbours."""

    name = "spillover"
    stream = stream(FAMILY)
    labels = (
        *Geo.labels,
        "The ring of the geos in their order stands in for geography: each geo's own media lift "
        "its two ring neighbours by s/2 each of its home lift; national media do not spill.",
        "A geo test's controls are the tested geos' neighbours, so they receive what spills from "
        "the tested geos, as in practice.",
    )
    pilots = MappingProxyType({name: pilot(FAMILY, e) for e, name in enumerate(SPILLOVERS)})
    scored = MappingProxyType({name: scored(FAMILY, e) for e, name in enumerate(SPILLOVERS)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in SPILLOVERS:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        return drawn(
            environment,
            seed,
            mix(0.0, seed, family=FAMILY, spillover=SPILLOVERS[environment]),
            FAMILY,
        )


SPILLOVER = Spillover()

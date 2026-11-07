"""The geo ladder's opt-in schedule of first weeks: the late schedule holds the invariants the
default holds and leaves 30 weeks of history before its first test; a rung run on it takes its
tests from its weeks, each test read again on its exported universe at the schedule's slope is its
row to the last bit, at either precision; a rung without tests is the default's; and a schedule
whose readings would not hold is refused, each rule at its edge."""

import itertools

import jax
import numpy as np
import pytest

from causaldyn_bench.scorecard import ladder
from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.ladder import (
    COOLDOWN,
    GEO_STARTS,
    LATE_STARTS,
    PRE,
    TEST,
    dark_geos,
    geo_rung,
    geo_universes,
)
from causaldyn_bench.scorecard.observe import UNIVERSES
from causaldyn_bench.scorecard.spillover import SPILLOVER

LENGTH = 8  # the kernel's
WORLDS = {
    "geo": (GEO, "policy-0", 3),
    "spillover": (SPILLOVER, "spillover-0.2", 0),
    "three geos": (FEW_GEOS, "geos-3", 2),
}


@pytest.fixture(scope="module", params=sorted(WORLDS))
def drawn(request):
    family, environment, place = WORLDS[request.param]
    return family, family.world(environment, family.pilots[environment][place])


def test_the_late_schedule_holds_the_ladders_invariants_and_thirty_weeks_before_its_first_test():
    assert GEO_STARTS == (20, 44, 68, 92)
    for schedule in (GEO_STARTS, LATE_STARTS):
        assert len(schedule) == 4
        assert schedule[0] - PRE >= 1 + LENGTH
        assert schedule[-1] + TEST + COOLDOWN - 1 <= 104
        assert all(b - PRE > a + TEST - 1 + LENGTH - 1 for a, b in itertools.pairwise(schedule))
    assert LATE_STARTS[0] - 1 >= 30 > GEO_STARTS[0] - 1


def test_a_late_rung_takes_its_tests_from_its_weeks_and_each_reads_again_on_its_universe(
    drawn,
):
    """Each channel's four tests start in the late weeks, alternate geos dark in turn, and read
    the planned spend there; each, read again on the exported universe by the ladder's reading at
    the late schedule's slope, is its row bit for bit, the export the same bits at either
    precision, and its error is read from placebo windows."""
    family, world = drawn
    history = world.history
    with jax.enable_x64(False):
        single = family.observe(world, 4, version=UNIVERSES, starts=LATE_STARTS)
    with jax.enable_x64(True):
        double = family.observe(world, 4, version=UNIVERSES, starts=LATE_STARTS)
    assert single.digest() == double.digest()
    assert single.digest() != family.observe(world, 4, version=UNIVERSES).digest()
    lift, universes = double.lift, double.universes
    assert lift.geos is not None and lift.gaps is not None and universes is not None
    geos = history.mix.geos
    names = [c.name for c in history.mix.channels]
    assert lift.channel == tuple(name for name in names for _ in LATE_STARTS)
    assert lift.start == LATE_STARTS * len(names)
    assert np.all(np.isfinite(lift.sigma)) and np.all(lift.sigma > 0.0)
    for row, (channel, start) in enumerate(zip(lift.channel, lift.start, strict=True)):
        c, j, first = names.index(channel), LATE_STARTS.index(start), start - 1
        treated = list(dark_geos(geos, j))
        assert lift.geos[row] == tuple(treated)
        planned = history.spend[treated, first : first + TEST, c].sum(axis=0).mean()
        assert lift.x[row] == pytest.approx(float(planned), rel=1e-14, abs=0.0)
        kpi = universes.sales[c].T
        others = [g for g in range(geos) if g not in treated]
        ours, theirs = kpi[treated].sum(axis=0), kpi[others].sum(axis=0)
        slope = ladder._slope(ours, theirs, first, LENGTH, LATE_STARTS)
        assert ladder._geo_reading(ours, theirs, first, slope) == lift.delta_y[row]
        np.testing.assert_array_equal(ladder._geo_gaps(ours, theirs, first, slope), lift.gaps[row])
    for c in range(len(names)):
        dark = history.spend.copy()
        for j, start in enumerate(LATE_STARTS):
            dark[list(dark_geos(geos, j)), start - 1 : start - 1 + TEST, c] = 0.0
        np.testing.assert_array_equal(universes.spend[c], dark.transpose(1, 0, 2))


def test_a_rung_without_tests_is_the_defaults_and_a_rung_is_its_schedules_latest(drawn):
    family, world = drawn
    late = family.observe(world, 0, version=UNIVERSES, starts=LATE_STARTS)
    assert late.digest() == family.observe(world, 0, version=UNIVERSES).digest()
    rows = geo_rung(world.history, 8, world.seed, 2, starts=LATE_STARTS)
    assert set(rows.start) == {72, 92}
    drawn_universes = geo_universes(world.history, 8, world.seed, 2, starts=LATE_STARTS)
    assert set(drawn_universes) == {c.name for c in world.history.mix.channels}


def test_a_schedule_whose_readings_would_not_hold_is_refused():
    """Each rule refuses its edge and accepts the week past it: four tests; the first's weeks
    before it past the adstock's first seven; each test's carryover, its last dark week and the
    kernel's seven after it, gone before the next one's weeks before it open; the last readout
    within the 104 weeks."""
    world = GEO.world("policy-0", GEO.pilots["policy-0"][0])
    history = world.history
    refused = {
        (44, 68, 92): "starts 4 tests",
        (12, 44, 68, 92): "adstock's first",
        (20, 34, 68, 92): "carryover",
        (20, 44, 68, 94): "run past",
    }
    for starts, match in refused.items():
        with pytest.raises(ValueError, match=match):
            geo_rung(history, 8, world.seed, 1, starts=starts)
        with pytest.raises(ValueError, match=match):
            geo_universes(history, 8, world.seed, 1, starts=starts)
        with pytest.raises(ValueError, match=match):
            GEO.observe(world, 1, starts=starts)
    for starts in ((13, 44, 68, 92), (20, 35, 68, 92), (20, 44, 68, 93)):
        ladder._check_starts(starts, 104, LENGTH)

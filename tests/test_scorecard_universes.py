"""A geo export's version 4: each channel's universe at the rung, the KPI and the spend of every geo
as the ladder ran the channel's tests there. Each test read again on its exported universe, by the
ladder's own reading, is the export's row to the last bit, the export the same bits at either
precision; a universe is the world's base and its spend with the channel dark in its tests' geos
and weeks, and noise from the family's stream of tests; version 4 holds version 3's arrays bit for
bit, gives versions 2's and 3's digests, reads back whole, and refuses a malformed universe; and
the geo families' exports at versions 2 and 3 hold no universe."""

import dataclasses

import jax
import numpy as np
import pytest

from causaldyn_bench.scorecard import ladder
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.ladder import GEO_STARTS, TEST, dark_geos
from causaldyn_bench.scorecard.observe import (
    SERIES,
    UNIVERSES,
    VERSION,
    Observation,
    export,
    read,
)
from causaldyn_bench.scorecard.spillover import SPILLOVER
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

WORLDS = {
    "geo": (GEO, "policy-0.5", 2),
    "spillover": (SPILLOVER, "spillover-0.5", 1),
    "three geos": (FEW_GEOS, "geos-3", 4),
}


@pytest.fixture(scope="module", params=sorted(WORLDS))
def drawn(request):
    family, environment, place = WORLDS[request.param]
    return family, family.world(environment, family.pilots[environment][place])


@pytest.fixture(scope="module")
def observed(drawn):
    family, world = drawn
    return family.observe(world, 4, version=UNIVERSES)


@pytest.mark.parametrize("k", [1, 4])
def test_every_test_read_again_on_its_exported_universe_is_its_row_to_the_last_bit(
    drawn, tmp_path, k
):
    """The ladder's reading of each row, on the KPI of the universe the export holds for the row's
    channel, gives the row's change and its gaps bit for bit; the export is the same bits whichever
    precision JAX runs at."""
    family, world = drawn
    with jax.enable_x64(False):
        single = family.observe(world, k, version=UNIVERSES)
    with jax.enable_x64(True):
        double = family.observe(world, k, version=UNIVERSES)
    assert single.digest() == double.digest()
    back = read(export(single, tmp_path))
    assert back.digest() == single.digest() and back.universes is not None
    lift, length = back.lift, back.kernel_length
    assert lift.geos is not None and lift.gaps is not None
    assert len(lift.channel) == 3 * k
    for row, (channel, start) in enumerate(zip(lift.channel, lift.start, strict=True)):
        kpi = back.universes.sales[back.universes.channel.index(channel)].T
        treated = list(lift.geos[row])
        others = [g for g in range(kpi.shape[0]) if g not in treated]
        ours, theirs = kpi[treated].sum(axis=0), kpi[others].sum(axis=0)
        first = start - 1
        slope = ladder._slope(ours, theirs, first, length)
        assert ladder._geo_reading(ours, theirs, first, slope) == lift.delta_y[row]
        np.testing.assert_array_equal(ladder._geo_gaps(ours, theirs, first, slope), lift.gaps[row])


def test_a_universe_is_the_worlds_base_and_spend_with_its_tests_dark_and_the_familys_noise(drawn):
    """Drawn apart from the ladder: the spend the history's with the channel dark in each test's
    geos and weeks and every other channel's as logged, and the KPI the world's base, what chc's
    world lifts at that spend, and the world's noise a head times the family's tests' normals,
    ``(100 + f, 2, seed, c + 1)``. Three geos is a world of plain normal noise, as the others."""
    family, world = drawn
    history, k = world.history, 2
    observation = family.observe(world, k, version=UNIVERSES)
    universes = observation.universes
    assert universes is not None
    names = tuple(c.name for c in history.mix.channels)
    assert universes.channel == names
    geos, weeks, _ = history.spend.shape
    for c in range(len(names)):
        dark = history.spend.copy()
        for start in GEO_STARTS[-k:]:
            taken = list(dark_geos(geos, GEO_STARTS.index(start)))
            dark[taken, start - 1 : start - 1 + TEST, c] = 0.0
        normals = np.random.default_rng((100 + number(family), 2, world.seed, c + 1))
        with jax.enable_x64(True):
            level = history.base + np.asarray(history.lift(dark))
        noise = history.mix.noise * np.sqrt(history.population)[:, None]
        kpi = level + noise * normals.standard_normal((geos, weeks))
        np.testing.assert_array_equal(universes.sales[c], kpi.T)
        np.testing.assert_array_equal(universes.spend[c], dark.transpose(1, 0, 2))
        moved = universes.spend[c] != observation.spend
        assert moved[:, :, [i for i in range(len(names)) if i != c]].sum() == 0
        assert moved.sum() == sum(
            len(dark_geos(geos, GEO_STARTS.index(s))) * TEST for s in (68, 92)
        )


def test_version_four_is_version_three_with_the_universes(drawn, observed, tmp_path):
    """Every array of version 3 with the same bits, the universes beside them, a digest of its own
    that reads them, and versions 2's and 3's digests when a record fitted to those asks for them;
    the geo family's own exports at versions 2 and 3 hold no universe."""
    family, world = drawn
    second, third = family.observe(world, 4), family.observe(world, 4, version=SERIES)
    assert (second.version, third.version) == (VERSION, SERIES)
    assert second.universes is None and third.universes is None
    paths = {v: export(o, tmp_path / str(v)) for v, o in ((2, second), (3, third), (4, observed))}
    with np.load(paths[2]) as two, np.load(paths[3]) as three, np.load(paths[4]) as four:
        assert set(three.files) - set(two.files) == {"lift_gaps"}
        added = set(four.files) - set(three.files)
        assert {name for name in added if not name.startswith("switchback_")} == {
            "universe_channel",
            "universe_sales",
            "universe_spend",
        }
        for name in three.files:
            if name not in ("version", "digest"):
                np.testing.assert_array_equal(four[name], three[name])
        assert int(four["version"]) == UNIVERSES
    back = read(paths[4])
    assert back.version == UNIVERSES and back.universes is not None
    assert observed.universes is not None
    assert back.universes.channel == observed.universes.channel
    np.testing.assert_array_equal(back.universes.sales, observed.universes.sales)
    np.testing.assert_array_equal(back.universes.spend, observed.universes.spend)
    assert back.digest() == observed.digest() not in (second.digest(), third.digest())
    assert back.digest(VERSION) == second.digest() == read(paths[2]).digest()
    assert back.digest(SERIES) == third.digest() == read(paths[3]).digest()
    nudged = observed.universes.sales.copy()
    nudged[1, 50, 0] *= 1.0 + 1e-12
    moved = dataclasses.replace(
        observed, universes=dataclasses.replace(observed.universes, sales=nudged)
    )
    assert moved.digest() != observed.digest()
    assert moved.digest(SERIES) == third.digest()
    with pytest.raises(ValueError, match="these have none"):
        third.digest(UNIVERSES)
    with pytest.raises(ValueError, match="version 2 to 4"):
        family.observe(world, 4, version=1)


def test_rung_nought_holds_no_universe(tmp_path):
    world = GEO.world("policy-0", GEO.pilots["policy-0"][0])
    nought = GEO.observe(world, 0, version=UNIVERSES)
    assert nought.universes is not None and nought.universes.channel == ()
    assert nought.universes.sales.shape == (0, 104, 10)
    assert nought.universes.spend.shape == (0, 104, 10, 3)
    back = read(export(nought, tmp_path))
    assert back.universes is not None and back.universes.channel == ()
    assert back.digest() == nought.digest()


def test_a_malformed_version_four_observation_is_refused(observed):
    o, replace = observed, dataclasses.replace
    assert o.universes is not None
    held = o.universes
    broken = held.sales.copy()
    broken[0, 3, 1] = np.nan
    negative = held.spend.copy()
    negative[2, 7, 0, 1] = -1.0

    def universes(**change) -> Observation:
        return replace(o, universes=replace(held, **change))

    cases = [
        ("only there", lambda: replace(o, version=SERIES)),
        ("only there", lambda: replace(o, universes=None)),
        ("tested channels'", lambda: universes(channel=held.channel[::-1])),
        ("sales are", lambda: universes(sales=held.sales[:, :-1])),
        ("spend", lambda: universes(spend=held.spend[..., :2])),
        ("finite", lambda: universes(sales=broken)),
        ("at least 0", lambda: universes(spend=negative)),
    ]
    for match, change in cases:
        with pytest.raises(ValueError, match=match):
            change()
    national = TRACK_M2.observe(TRACK_M2.world("drawn", 905), 4)
    with pytest.raises(ValueError, match="geo axis alone"):
        replace(national, version=UNIVERSES, universes=held)

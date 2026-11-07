"""The switchback test a geo world carries: its national channel off in every geo over half of the
history's last twelve blocks of the kernel's length, the blocks' order and the universe's noise each
from a further index of the family's stream of tests; its universe the world's base and its spend
with the channel off in those blocks; its truth the lift chc's world makes over the channel off
through the test, carry-out past the history included, which the national cell's own formula gives
too; a version-4 export carries it at every rung and reads it back, and a malformed one is refused;
and no world, test, quarter or truth the families drew before it moves."""

import dataclasses

import jax
import numpy as np
import pytest

from causaldyn_bench.mmm_decision import PLANNED
from causaldyn_bench.scorecard import switchback
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.geo_noise import kpi_noise
from causaldyn_bench.scorecard.observe import SERIES, UNIVERSES, VERSION, Observation, export, read
from causaldyn_bench.scorecard.spillover import SPILLOVER
from causaldyn_bench.scorecard.truth import geo_curve, geo_kernel

WORLDS = {
    "geo": (GEO, "policy-0.5", 0),
    "spillover": (SPILLOVER, "spillover-0.2", 2),
    "fifty geos": (FEW_GEOS, "geos-50", 1),
}


@pytest.fixture(scope="module", params=sorted(WORLDS))
def drawn(request):
    family, environment, place = WORLDS[request.param]
    return family, family.world(environment, family.pilots[environment][place])


def test_the_design_turns_half_the_last_twelve_blocks_off_in_an_order_from_its_own_stream():
    """TV, weeks 9 to 104 in blocks of eight, six on, the order ``(108, 2, seed, 1001)``'s first
    permutation of the twelve. Over 400 worlds each block is on in 200 expected, binomial with a
    standard deviation of 10; the bounds are five of them, so a sound draw leaves them about once in
    three million blocks."""
    seeds = list(GEO.pilots["policy-0"]) + list(GEO.scored["policy-0"][:300])
    world = GEO.world("policy-0", seeds[0]).history
    count = np.zeros(switchback.BLOCKS)
    orders = set()
    for seed in seeds:
        test = switchback.design(world, 8, seed)
        assert (test.channel, test.first, test.block) == (2, 9, 8)
        on = np.array(test.on)
        assert on.sum() == 6
        order = np.random.default_rng((108, 2, seed, 1001)).permutation(12)
        assert set(np.flatnonzero(on)) == set(order[:6].tolist())
        count += on
        orders.add(test.on)
    assert np.all((count >= 150) & (count <= 250)), count
    assert len(orders) > 300  # 924 arrangements: 400 draws repeat some, not most
    state = switchback.design(world, 8, seeds[0]).state(104)
    assert state[:8].all() and state[8:].sum() == 48
    np.testing.assert_array_equal(state[8:], np.repeat(switchback.design(world, 8, seeds[0]).on, 8))


def test_the_universe_is_the_worlds_base_and_its_spend_with_the_channel_off_and_its_own_noise(
    drawn,
):
    family, world = drawn
    history, seed = world.history, world.seed
    test = switchback.design(history, number(family), seed)
    kpi, spend = switchback.universe(history, number(family), seed, test)
    off = ~test.state(104)
    expected = history.spend.copy()
    expected[:, off, test.channel] = 0.0
    np.testing.assert_array_equal(spend, expected)
    assert np.all(history.spend[:, off, test.channel] > 0.0)
    normals = np.random.default_rng((100 + number(family), 2, seed, 1000))
    normals = normals.standard_normal(kpi.shape)
    with jax.enable_x64(True):
        level = history.base + np.asarray(history.lift(expected))
    np.testing.assert_array_equal(kpi, level + kpi_noise(history, level, normals))
    if family is not FEW_GEOS:
        noise = history.mix.noise * np.sqrt(history.population)[:, None] * normals
        np.testing.assert_array_equal(kpi, level + noise)


def test_the_truth_is_the_national_cells_lift_over_the_test_and_its_carry_out(drawn):
    """The national channel's spend per head is the same in every geo, so its lifts over the geos
    are one cell's: the curve at the whole population of the spend summed over the geos, through
    the kernel, times ``sum_g P_g beta_g``. Its lift at the test's spend less at the channel off
    through the test, over the history and the kernel's weeks after it, is the truth."""
    family, world = drawn
    history = world.history
    truth = family.truth(world)
    test = switchback.design(history, number(family), world.seed)
    assert truth.switchback == switchback.lift(history, test)
    channel = history.mix.channels[test.channel]
    curve, lam = geo_curve(channel.curve, float(history.population.sum()))
    kernel = geo_kernel(channel.retention, history.mix.length)
    size = float(history.population @ history.effect[:, test.channel])
    summed = history.spend[:, :, test.channel].sum(axis=0)
    tail = np.zeros(history.mix.length - 1)
    on = np.concatenate([np.where(test.state(104), summed, 0.0), tail])
    off = np.concatenate([summed[: test.first - 1], np.zeros(104 - test.first + 1), tail])
    gained = size * (
        curve.value(np.convolve(on, kernel)[: on.size], lam)
        - curve.value(np.convolve(off, kernel)[: off.size], lam)
    )
    assert truth.switchback == pytest.approx(float(gained.sum()), rel=1e-11, abs=0.0)
    assert float(gained[:104].sum()) > 0.0
    carried = float(gained[104:].sum())
    assert (carried > 0.0) == test.on[-1] and (carried == 0.0) == (not test.on[-1])


def test_a_version_four_export_carries_the_switchback_at_every_rung_and_reads_it_back(
    drawn, tmp_path
):
    family, world = drawn
    tests = [family.observe(world, k, version=UNIVERSES) for k in (0, 4)]
    nought, top = tests
    assert nought.switchback is not None and top.switchback is not None
    for name in ("channel", "first", "block", "on", "sales", "spend"):
        np.testing.assert_array_equal(
            getattr(nought.switchback, name), getattr(top.switchback, name)
        )
    test = switchback.design(world.history, number(family), world.seed)
    kpi, spend = switchback.universe(world.history, number(family), world.seed, test)
    np.testing.assert_array_equal(top.switchback.sales, kpi.T)
    np.testing.assert_array_equal(top.switchback.spend, spend.transpose(1, 0, 2))
    assert (top.switchback.channel, top.switchback.on) == ("tv", test.on)
    with np.load(export(top, tmp_path)) as saved:
        assert {name for name in saved.files if name.startswith("switchback_")} == {
            "switchback_channel",
            "switchback_first",
            "switchback_block",
            "switchback_on",
            "switchback_sales",
            "switchback_spend",
        }
    back = read(tmp_path / f"world_{world.seed}.npz")
    assert back.switchback is not None and back.digest() == top.digest()
    for name in ("channel", "first", "block", "on", "sales", "spend"):
        np.testing.assert_array_equal(getattr(back.switchback, name), getattr(top.switchback, name))
    nudged = top.switchback.sales.copy()
    nudged[60, 0] *= 1.0 + 1e-12
    moved = dataclasses.replace(top, switchback=dataclasses.replace(top.switchback, sales=nudged))
    assert moved.digest() != top.digest() and moved.digest(SERIES) == top.digest(SERIES)
    assert moved.digest(VERSION) == family.observe(world, 4).digest()


def test_a_malformed_switchback_is_refused():
    world = GEO.world("policy-0", GEO.pilots["policy-0"][0])
    o: Observation = GEO.observe(world, 2, version=UNIVERSES)
    held = o.switchback
    assert held is not None
    replace = dataclasses.replace
    broken = held.sales.copy()
    broken[10, 3] = np.inf
    negative = held.spend.copy()
    negative[3, 0, 0] = -1.0
    leaking = held.spend.copy()
    first_off = held.first - 1 + 8 * held.on.index(False)
    leaking[first_off, 4, 2] = 1.0

    def test(**change) -> Observation:
        return replace(o, switchback=replace(held, **change))

    cases = [
        ("version 4 alone", lambda: replace(o, version=SERIES, universes=None)),
        ("national channel", lambda: test(channel="search")),
        ("lie outside", lambda: test(first=10)),
        ("lie outside", lambda: test(block=9)),
        ("on in some blocks and off", lambda: test(on=(True,) * 12)),
        ("on in some blocks and off", lambda: test(on=(False,) * 12)),
        ("sales are", lambda: test(sales=held.sales[:, :-1])),
        ("spend", lambda: test(spend=held.spend[:-1])),
        ("finite", lambda: test(sales=broken)),
        ("at least 0", lambda: test(spend=negative)),
        ("spends nothing", lambda: test(spend=leaking)),
    ]
    for match, change in cases:
        with pytest.raises(ValueError, match=match):
            change()


def test_the_test_moves_no_world_test_quarter_or_truth_the_families_drew_before_it(tmp_path):
    """The test's streams are further indices of the family's stream of tests that no ladder reads;
    the versions 2 and 3 exports hold what they held, with the same digests, whether or not the
    world's switchback was drawn first; and the truth of the family's first pilot world is the one
    the bench stated before the test was added, to the tie between two regrets."""
    for family in (GEO, SPILLOVER, FEW_GEOS):
        for environment in family.pilots:
            channels = len(
                family.world(environment, family.pilots[environment][0]).history.mix.channels
            )
            assert {switchback.NOISE, switchback.ORDER}.isdisjoint(range(1, channels + 1))
    seed = GEO.pilots["policy-0"][0]
    world = GEO.world("policy-0", seed)
    before = {k: GEO.observe(world, k) for k in (0, 4)}
    test = switchback.design(world.history, 8, seed)
    switchback.universe(world.history, 8, seed, test)
    for k, observation in before.items():
        again = GEO.observe(world, k)
        assert again.digest() == observation.digest()
        assert again.universes is None and again.switchback is None
        with np.load(export(again, tmp_path / str(k))) as saved:
            assert not any(name.startswith(("universe_", "switchback_")) for name in saved.files)
    truth = GEO.truth(world)
    stated = {  # the bench's truth of world 360000 before the test was added
        "best": 136741826.21055025,
        "price": 0.5190052005883683,
        "target": 703813435.9421318,
        "roi": (1.4810126764890161, 1.173550796445108, 0.44584145947586773),
    }
    assert truth.best.worth == pytest.approx(stated["best"], rel=1e-9, abs=0.0)
    assert truth.best.price == pytest.approx(stated["price"], rel=1e-9, abs=0.0)
    assert float(truth.target.sum()) == pytest.approx(stated["target"], rel=1e-9, abs=0.0)
    np.testing.assert_allclose(truth.returns.roi, stated["roi"], rtol=1e-9, atol=0.0)
    assert truth.best.weekly.size == 21 and truth.target.size == PLANNED

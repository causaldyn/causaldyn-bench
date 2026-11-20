"""The scorecard's seeds: every family's block and stream apart from one another and from every seed
and stream a committed run before the scorecard drew from, numpy's padding included; family 0's
scored blocks extended past the budgets run's without moving one of its worlds; and every
confirmatory block apart from every block and every seed a committed run drew, the scorecard's
scored run's included, and shared by two confirmatory runs that read no world of each other's."""

from collections import Counter

import numpy as np
import pytest

from causaldyn_bench.budget_regret import ENVIRONMENTS
from causaldyn_bench.scorecard.collinear import COLLINEAR
from causaldyn_bench.scorecard.confounded import CONFOUNDED
from causaldyn_bench.scorecard.drift import DRIFT
from causaldyn_bench.scorecard.family import confirmatory as confirmatory_of
from causaldyn_bench.scorecard.family import index, number
from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.flat import FLAT
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.long_term import LONG_TERM
from causaldyn_bench.scorecard.meridian import MERIDIAN
from causaldyn_bench.scorecard.outside import OUTSIDE
from causaldyn_bench.scorecard.performance import PERFORMANCE
from causaldyn_bench.scorecard.promotions import PROMOTIONS
from causaldyn_bench.scorecard.pymc_marketing import PYMC_MARKETING
from causaldyn_bench.scorecard.robyn import ROBYN
from causaldyn_bench.scorecard.s_curves import S_CURVES
from causaldyn_bench.scorecard.seeds import (
    BLOCK,
    CONFIRMATORY,
    FAMILIES,
    FIRST,
    PILOTS,
    SCORECARD_RUNS,
    SCORECARD_STREAMS,
    SECOND_RUN,
    TRACK_M2,
    TRACK_M2_CONFIRMATORY,
    TRACK_M2_EXTENSIONS,
    USED_SEEDS,
    USED_STREAMS,
    WORLDS,
    Role,
    confirmatory,
    pilot,
    rng,
    scored,
    stream,
)
from causaldyn_bench.scorecard.short import SHORT
from causaldyn_bench.scorecard.simmmulator import Simmmulator
from causaldyn_bench.scorecard.spillover import SPILLOVER
from causaldyn_bench.scorecard.track_m2 import TRACK_M2 as FAMILY_0
from causaldyn_bench.scorecard.twins import TWINS

CHANNELS = 3  # a Track M v2 world's channels, each with tests of its own noise


def _canonical(seed: int | tuple[int, ...]) -> tuple[int, ...]:
    """The words numpy hashes: an int is one word, and a seed of fewer than four is padded with
    zeros to four."""
    words = (seed,) if isinstance(seed, int) else tuple(seed)
    return words + (0,) * (4 - len(words))


def _used(*, scorecard: bool = False) -> set[tuple[int, ...]]:
    """Every seed a committed run drew, as numpy hashes it: the runs before the scorecard's, and
    with ``scorecard`` the scorecard's runs' too."""
    held = {k: v for k, v in USED_SEEDS.items() if scorecard or k not in SCORECARD_RUNS}
    drawn = {k: v for k, v in USED_STREAMS.items() if scorecard or k not in SCORECARD_STREAMS}
    ints = {s for ranges in held.values() for r in ranges for s in r}
    streams = {seed for seeds in drawn.values() for seed in seeds}
    return {_canonical(s) for s in ints} | {_canonical(s) for s in streams}


def _blocks() -> list[range]:
    """Every family's pilot and scored worlds, environment by environment."""
    out = []
    for family in range(1, FAMILIES + 1):
        out += [pilot(family, e) for e in range(WORLDS // PILOTS)]
        out += [scored(family, e) for e in range(BLOCK // WORLDS - 1)]
    return out


def test_numpy_pads_a_seed_with_zeros_to_four_words_and_no_further():
    def first(seed):
        return np.random.default_rng(seed).integers(0, 2**62, 4).tolist()

    assert first(5) == first((5, 0)) == first((5, 0, 0, 0))
    assert first((101, 3, 7)) == first((101, 3, 7, 0))
    assert first((5, 0, 0, 0, 0)) != first(5)
    assert first((101, 3, 7, 1)) != first((101, 3, 7))


def test_the_blocks_tile_each_familys_seeds_without_overlap():
    seeds = [s for block in _blocks() for s in block]
    assert len(seeds) == len(set(seeds)) == FAMILIES * BLOCK
    assert min(seeds) == FIRST + BLOCK and max(seeds) == FIRST + BLOCK * (FAMILIES + 1) - 1
    assert pilot(1, 0) == range(220_000, 220_100)
    assert scored(1, 0) == range(221_000, 222_000)
    assert scored(1, 5) == range(226_000, 227_000)


def test_no_new_world_seed_is_one_a_committed_run_drew():
    new = range(FIRST + BLOCK, FIRST + BLOCK * (FAMILIES + 1))
    for words in _used():
        if words[1:] == (0, 0, 0):  # an int seed
            assert words[0] not in new, f"{words[0]} is a committed run's seed"


def test_no_new_worlds_test_noise_is_a_seed_a_committed_run_drew():
    """Lift tests on a family's world draw their noise from 1000 s + c, as Track M v2's do."""
    new = range(FIRST + BLOCK, FIRST + BLOCK * (FAMILIES + 1))
    for words in _used():
        if words[1:] == (0, 0, 0):
            assert words[0] // 1000 not in new, f"{words[0]} is a committed run's seed"


def test_no_new_drawn_worlds_channels_are_drawn_from_a_committed_stream():
    new = range(FIRST + BLOCK, FIRST + BLOCK * (FAMILIES + 1))
    for words in _used():
        if words[0] == 7:
            assert words[1] not in new


def test_no_family_stream_is_one_a_committed_run_drew_from():
    """A family's variates are (100 + f, role, seed, ...), seed never 0: no committed seed starts
    with a family's stream unless it is an int seed, whose other words are all 0."""
    streams = {stream(f) for f in range(FAMILIES + 1)}
    assert streams == set(range(TRACK_M2, TRACK_M2 + FAMILIES + 1))
    assert not streams & {7, 11, 12, 13}
    seeds = [s for block in _blocks() for s in block]
    assert min(seeds) > 0
    for words in _used():
        if words[0] in streams:
            assert words[1:] == (0, 0, 0)


def test_the_families_variates_are_streams_of_their_own():
    seed = scored(1, 0)[0]
    draws = {
        (family, role, index): rng(family, role, seed, *index).integers(0, 2**62, 2).tolist()
        for family in (0, 1, 2)
        for role in Role
        for index in ((), (1,), (2,))
    }
    as_tuples = {tuple(v) for v in draws.values()}
    assert len(as_tuples) == len(draws)
    np.testing.assert_array_equal(
        rng(0, Role.CONTINUATION, seed).normal(size=3),
        np.random.default_rng((TRACK_M2, 3, seed)).normal(size=3),
    )


def test_a_further_index_of_nought_and_a_family_out_of_range_are_refused():
    with pytest.raises(ValueError, match="starts at 1"):
        rng(1, Role.PATH, 221_000, 0)
    with pytest.raises(ValueError, match="not one of 0 to"):
        stream(FAMILIES + 1)
    with pytest.raises(ValueError, match="has no block"):
        scored(0, 0)
    with pytest.raises(ValueError, match="scored environments"):
        scored(1, BLOCK // WORLDS - 1)
    with pytest.raises(ValueError, match="pilots hold"):
        pilot(1, WORLDS // PILOTS)


def test_family_noughts_scored_blocks_keep_the_budgets_runs_worlds_first_then_go_on():
    """Each environment holds 260 scored worlds: the budgets run's, in their order, then its
    extension from the environment's first seed plus 1 000, the seeded order the seeds' own."""
    for environment in ENVIRONMENTS:
        held = FAMILY_0.scored[environment.name]
        assert len(held) == 260
        assert tuple(held[: environment.worlds]) == tuple(environment.seeds)
        assert tuple(held[environment.worlds :]) == tuple(TRACK_M2_EXTENSIONS[environment.name])
        assert held[environment.worlds] == environment.first + 1_000
        assert list(held) == sorted(held)
    assert tuple(FAMILY_0.scored) == tuple(TRACK_M2_EXTENSIONS) == ("drawn", "reference")


def test_every_block_is_apart_from_every_other_but_family_noughts_two_pilot_blocks():
    """No seed is in two blocks, every family's pilots and scored worlds and family 0's alike, but
    the budgets run's pilots, which draw both its environments from 900 by that run's design."""
    blocks = [*_blocks(), *FAMILY_0.pilots.values(), *FAMILY_0.scored.values()]
    twice = {s for s, n in Counter(s for block in blocks for s in block).items() if n > 1}
    assert twice == set(FAMILY_0.pilots["drawn"]) & set(FAMILY_0.pilots["reference"])
    assert twice == set(range(900, 920))


def test_family_noughts_new_worlds_draw_from_no_committed_seed_and_no_familys_block():
    """A new world of family 0 draws its history from its seed, a drawn one its channels from
    (7, seed), and its tests' noise from 1000 seed + c: none is a seed or stream a committed run
    drew from, and none a world's seed of any family's block or a test's seed of one."""
    used = _used()
    families = range(FIRST + BLOCK, FIRST + BLOCK * (FAMILIES + 1))
    for environment, block in TRACK_M2_EXTENSIONS.items():
        for seed in block:
            drawn = [seed, *range(1000 * seed, 1000 * seed + CHANNELS)]
            assert not {_canonical(s) for s in drawn} & used, seed
            assert all(s not in families and s // 1000 not in families for s in drawn), seed
            if environment == "drawn":
                assert _canonical((7, seed)) not in used, seed


def test_family_noughts_old_worlds_export_as_before_and_its_new_ones_as_drawn(other_kernels):
    """The budgets run's worlds export bit for bit as before the extension, at both ends of each
    block and of each pilot block, and the extension's first and last worlds as they were first
    drawn: digests read on x86 without AVX-512."""
    if other_kernels is not None:
        pytest.skip(other_kernels)
    pinned = {
        ("drawn", 10_000): ("7f01368ac3050d95", "e652d0ae03e53abf"),
        ("drawn", 10_199): ("c5764f912ba35800", "1010148a60211e8e"),
        ("reference", 20_000): ("6f940216e0b94fe0", "fbef107fdd1a31a9"),
        ("reference", 20_099): ("ee463176fa8bed94", "4bfd0f6977402f04"),
        ("drawn", 11_000): ("0b9fe7ec6590bb47", "62dd3a62b222f60c"),
        ("drawn", 11_059): ("e4464e8d133b77cd", "4e482872d456d575"),
        ("reference", 21_000): ("5cc243c9dcaee0a3", "3bd3405ce58702db"),
        ("reference", 21_159): ("fab0ba00c51098e8", "82313bdeeada4997"),
    }
    for (environment, seed), digests in pinned.items():
        world = FAMILY_0.world(environment, seed)
        assert len(world.history.channels) == CHANNELS
        assert tuple(FAMILY_0.observe(world, k).digest() for k in (0, 4)) == digests
    for environment, seed, digest in (
        ("drawn", 939, "8e0d1f050e04cc69"),
        ("reference", 919, "6edb91a3f961e24a"),
    ):
        assert FAMILY_0.observe(FAMILY_0.world(environment, seed), 4).digest() == digest


FAMILY_OBJECTS = (
    FAMILY_0, DRIFT, CONFOUNDED, COLLINEAR, S_CURVES, PERFORMANCE, OUTSIDE, SHORT, GEO, SPILLOVER,
    FEW_GEOS, MERIDIAN, PYMC_MARKETING, ROBYN, TWINS, LONG_TERM, FLAT, PROMOTIONS,
)  # fmt: skip
"""Every family that draws its worlds; family 15 reads its worlds from files, its one environment
drawn from its first scored block."""


def _confirmatory_blocks() -> list[range]:
    out = [confirmatory(f, e) for f in range(1, FAMILIES + 1) for e in range(CONFIRMATORY)]
    return [*out, *TRACK_M2_CONFIRMATORY.values()]


def test_every_family_s_scored_environments_are_below_its_confirmatory_blocks():
    """No family holds more environments than its confirmatory blocks are placed past: each
    environment's scored block is one of its first :data:`CONFIRMATORY`, its confirmatory block
    that one moved on, and family 0's are its own."""
    for family in FAMILY_OBJECTS:
        f = number(family)
        held = confirmatory_of(family)
        assert tuple(held) == tuple(family.scored)
        for environment, block in family.scored.items():
            if f == 0:
                assert held[environment] == TRACK_M2_CONFIRMATORY[environment][:SECOND_RUN]
                continue
            e = (block[0] - FIRST - BLOCK * f) // WORLDS - 1
            assert 0 <= e < CONFIRMATORY
            assert block == scored(f, e)
            assert confirmatory(f, e) == scored(f, e + CONFIRMATORY)
            assert held[environment] == confirmatory(f, e)[:SECOND_RUN]
    assert dict(Simmmulator.scored) == {"demo": scored(15, 0)}
    assert confirmatory(1, 0) == range(230_000, 231_000)
    assert dict(TRACK_M2_CONFIRMATORY) == {
        "drawn": range(12_000, 13_000),
        "reference": range(22_000, 23_000),
    }


def test_the_confirmatory_blocks_are_apart_from_every_block_and_every_committed_seed():
    """No confirmatory world is a world of another block, pilot, scored or confirmatory, or a
    seed any committed run drew, the scorecard's scored run's and pilots' included; nor is its
    tests' noise, 1000 s + c, any committed seed's; and family 0's draw their channels from no
    committed stream."""
    blocks = _confirmatory_blocks()
    seeds = [s for block in blocks for s in block]
    assert len(seeds) == len(set(seeds)) == FAMILIES * CONFIRMATORY * WORLDS + 2 * WORLDS
    others = [*_blocks(), *FAMILY_0.pilots.values(), *FAMILY_0.scored.values()]
    others = [b for b in others if b not in blocks]
    assert not set(seeds) & {s for block in others for s in block}
    used = _used(scorecard=True)
    ints = {words[0] for words in used if words[1:] == (0, 0, 0)}
    assert not set(seeds) & ints
    assert not {s // 1000 for s in ints} & set(seeds)
    for seed in TRACK_M2_CONFIRMATORY["drawn"]:
        assert _canonical((7, seed)) not in used, seed


def test_the_scorecard_s_runs_are_listed_as_drawn_each_the_first_worlds_of_a_block():
    """The scorecard's scored run's worlds are listed as drawn, each environment's the first of
    its scored block, family 0's the first of its extensions, and its pilots' each a pilot block
    whole."""
    held = SCORECARD_RUNS["the scorecard's scored run"]
    assert held[:2] == (range(11_000, 11_020), range(21_000, 21_120))
    assert scored(2, 1)[:137] in held
    assert len(held) == 2 + 6 + 3 * 7 + 2 + 1 + 1
    blocks = _blocks()
    for r in held[2:]:
        assert (r.start - FIRST) % WORLDS == 0
        assert any(r.start == b.start and len(r) <= len(b) == WORLDS for b in blocks)
    assert all(b in blocks and len(b) == PILOTS for b in SCORECARD_RUNS["the scorecard's pilots"])


def test_a_confirmatory_world_is_found_in_its_own_order_and_a_world_of_no_block_is_refused():
    block = confirmatory_of(DRIFT)["walk"]
    assert index(DRIFT, "walk", block[0]) == 0
    assert index(DRIFT, "walk", block[17]) == 17
    assert index(FAMILY_0, "drawn", 12_003) == 3
    with pytest.raises(ValueError, match="draws no pilot world"):
        index(DRIFT, "walk", block[0], pilot=True)
    with pytest.raises(ValueError, match="draws no scored world"):
        index(DRIFT, "trend", block[0])
    with pytest.raises(ValueError, match="confirmatory blocks hold"):
        confirmatory(1, CONFIRMATORY)


DEVELOPMENT = range(700_000, 900_000)
"""The seeds development runs outside the bench drew their worlds from."""


def test_two_confirmatory_runs_share_each_block_and_read_no_world_of_the_other():
    """The first confirmatory run's worlds are each block's first :data:`SECOND_RUN`, the second's
    the rest, in order: together the block whole, apart from each other, each long enough for the
    largest run the claim declares (520 worlds a family, all of them in one environment)."""
    for family in FAMILY_OBJECTS:
        first, second = confirmatory_of(family), confirmatory_of(family, second=True)
        assert tuple(first) == tuple(second) == tuple(family.scored)
        for environment, block in first.items():
            rest = second[environment]
            assert (len(block), len(rest)) == (SECOND_RUN, WORLDS - SECOND_RUN)
            assert block.stop == rest.start and rest.step == block.step == 1
            assert len(rest) >= 520
    assert confirmatory_of(DRIFT, second=True)["walk"] == range(230_300, 231_000)
    assert confirmatory_of(FAMILY_0, second=True) == {
        "drawn": range(12_300, 13_000),
        "reference": range(22_300, 23_000),
    }


def test_no_world_of_the_second_confirmatory_run_was_drawn_before():
    """No world of the second confirmatory run is a world of another block, pilot or scored, or of
    the first confirmatory run, which read at most 240 worlds an environment; nor a seed a committed
    run drew, nor one development runs drew from; nor is its tests' noise, 1000 s + c, a committed
    seed's."""
    # family 17's two environments are twins of one world, drawn from one block
    blocks = {b for family in FAMILY_OBJECTS for b in confirmatory_of(family, second=True).values()}
    seconds = [s for block in blocks for s in block]
    assert len(seconds) == len(set(seconds)) == len(blocks) * (WORLDS - SECOND_RUN)
    firsts = [s for family in FAMILY_OBJECTS for b in confirmatory_of(family).values() for s in b]
    assert SECOND_RUN > 240
    others = [*_blocks(), *FAMILY_0.pilots.values(), *FAMILY_0.scored.values()]
    others = [b for b in others if b not in _confirmatory_blocks()]
    drawn = {s for block in others for s in block} | set(firsts)
    used = _used(scorecard=True)
    ints = {words[0] for words in used if words[1:] == (0, 0, 0)}
    assert not set(seconds) & (drawn | ints | set(DEVELOPMENT))
    assert not {s // 1000 for s in ints} & set(seconds)


def test_a_second_confirmatory_world_is_found_in_its_own_order():
    block = confirmatory_of(DRIFT, second=True)["walk"]
    assert index(DRIFT, "walk", block[0]) == 0
    assert index(DRIFT, "walk", block[17]) == 17
    assert index(FAMILY_0, "drawn", 12_300) == 0
    assert index(FAMILY_0, "drawn", 12_299) == 299
    with pytest.raises(ValueError, match="draws no pilot world"):
        index(DRIFT, "walk", block[0], pilot=True)

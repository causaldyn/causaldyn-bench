"""The scorecard's seeds: every family's block and stream apart from one another and from every seed
and stream a committed run drew from, numpy's padding included."""

import numpy as np
import pytest

from causaldyn_bench.scorecard.seeds import (
    BLOCK,
    FAMILIES,
    FIRST,
    PILOTS,
    TRACK_M2,
    USED_SEEDS,
    USED_STREAMS,
    WORLDS,
    Role,
    pilot,
    rng,
    scored,
    stream,
)


def _canonical(seed: int | tuple[int, ...]) -> tuple[int, ...]:
    """The words numpy hashes: an int is one word, and a seed of fewer than four is padded with
    zeros to four."""
    words = (seed,) if isinstance(seed, int) else tuple(seed)
    return words + (0,) * (4 - len(words))


def _used() -> set[tuple[int, ...]]:
    ints = {s for ranges in USED_SEEDS.values() for r in ranges for s in r}
    streams = {seed for seeds in USED_STREAMS.values() for seed in seeds}
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

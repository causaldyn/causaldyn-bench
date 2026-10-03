"""Where the scorecard's worlds and their variates are drawn from: blocks of seeds and streams of
their own, apart from every seed and stream a committed run of the bench drew from.

Family ``f``, 1 to 14, owns the stream ``100 + f`` and the block of world seeds from
``200 000 + 20 000 f``: environment ``e``'s pilots from ``200 000 + 20 000 f + 100 e``, in the
block's first thousand, and its scored worlds from ``200 000 + 20 000 f + 1 000 (e + 1)``. Family 0,
Track M v2 itself, keeps the committed runs' seeds and owns the stream 100 for the variates the
scorecard adds to its worlds. Within an environment the worlds are taken in the order of their
seeds, so a run's sample, and each look at it, is a prefix of that order.

Every variate a family adds is drawn from ``numpy.random.default_rng((stream, role, seed))``, the
role one of :class:`Role`. Two rules keep a family built on Track M v2's world a Track M v2 world
where its own mechanism is switched off:

* the world's processes continue past its history from Track M v2's stream, 100, whichever family
  drew the world (:mod:`~causaldyn_bench.scorecard.continuation`);
* its lift tests' noise is drawn as Track M v2's is, from ``1000 seed + c`` for the ``c``-th channel
  (:mod:`~causaldyn_bench.scorecard.ladder`).

numpy pads a seed's words with zeros to four before it hashes them, so ``(a, b, c)`` and
``(a, b, c, 0)`` seed one stream, and an int seed ``s`` is ``(s, 0, 0, 0)``: a further index, where
a role needs several streams, starts at 1 (:func:`rng` refuses 0), and no world seed is 0.

:data:`USED_SEEDS` and :data:`USED_STREAMS` list what the committed runs drew from; a test holds
every family's block apart from them.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

FAMILIES = 14  # the families with blocks of their own; family 0 keeps Track M v2's seeds
FIRST = 200_000  # family f's block starts at FIRST + BLOCK f
BLOCK = 20_000
PILOTS = 100  # pilot worlds an environment's pilot block holds
WORLDS = 1_000  # scored worlds an environment's block holds
TRACK_M2 = 100  # Track M v2's stream: its worlds' continuation, whichever family drew them


class Role(IntEnum):
    """What a family's variate is for; each role is a stream of its own."""

    PARAMETERS = 0
    PATH = 1
    TESTS = 2
    CONTINUATION = 3
    MEASUREMENT = 4


def stream(family: int) -> int:
    if not 0 <= family <= FAMILIES:
        raise ValueError(f"family {family} is not one of 0 to {FAMILIES}")
    return TRACK_M2 + family


def _block(family: int) -> int:
    if not 1 <= family <= FAMILIES:
        raise ValueError(f"family {family} has no block: only 1 to {FAMILIES} do")
    return FIRST + BLOCK * family


def pilot(family: int, environment: int) -> range:
    """The pilot worlds of the family's environment ``environment``, counted from 0, in order."""
    if not 0 <= environment < WORLDS // PILOTS:
        raise ValueError(f"a family's pilots hold {WORLDS // PILOTS} environments")
    first = _block(family) + PILOTS * environment
    return range(first, first + PILOTS)


def scored(family: int, environment: int) -> range:
    """The scored worlds of the family's environment ``environment``, counted from 0, in order."""
    if not 0 <= environment < BLOCK // WORLDS - 1:
        raise ValueError(f"a family's block holds {BLOCK // WORLDS - 1} scored environments")
    first = _block(family) + WORLDS * (environment + 1)
    return range(first, first + WORLDS)


def rng(family: int, role: Role, seed: int, *index: int) -> np.random.Generator:
    """The generator of the family's variates for ``role`` in world ``seed``; ``index``, each 1 or
    more, picks one of several streams a role needs."""
    if any(i < 1 for i in index):
        raise ValueError(f"a further index starts at 1, since numpy pads seeds with zeros: {index}")
    return np.random.default_rng((stream(family), int(role), seed, *index))


# every int seed a committed run drew a world, a test's noise or a resample from
USED_SEEDS: dict[str, tuple[range, ...]] = {
    "the lift track's histories, and the check's and the families' runs' seed 0": (range(0, 500),),
    "the pilots of the budgets, curve-family, check and geo runs": (range(900, 1_000),),
    "budgets and external arms, drawn": (range(10_000, 10_200),),
    "budgets and external arms, reference": (range(20_000, 20_100),),
    "curve families": (range(30_000, 30_100),),
    "observational check": (range(40_000, 40_500),),
    "geo selection": (range(50_000, 50_500),),
    "the library's geo dynamic-linear-model run and its pilot": (
        range(0, 20),
        range(20_261_002, 20_261_102),
    ),
    "the lift track's tests": (range(10_000, 10_500),),
    # 1000 s + c for the c-th channel of every world whose tests the budgets, curve-family and check
    # runs drew, pilots included
    "every channel's tests": tuple(
        range(1000 * s, 1000 * s + 3)
        for block in (
            range(900, 1_000),
            range(10_000, 10_200),
            range(20_000, 20_100),
            range(30_000, 30_100),
            range(40_000, 40_500),
        )
        for s in block
    ),
}
# the tuple seeds committed runs drew from, each (first word, the seeds it was paired with): a drawn
# world's channels (7, s); the geo run's panel, pool and prior (11, s), (12, s), (12, s, 1) and
# (13, s); and the library's geo run's hierarchy (s, 1), whose first word is the world's seed
USED_STREAMS: dict[str, tuple[tuple[int, ...], ...]] = {
    "a drawn world's channels": tuple(
        (7, s)
        for block in (
            range(900, 1_000),
            range(10_000, 10_200),
            range(30_000, 30_100),
            range(50_000, 50_500),
        )
        for s in block
    ),
    "the geo run's panel, pool and prior": tuple(
        seed
        for s in (*range(900, 940), *range(50_000, 50_500))
        for seed in ((11, s), (12, s), (12, s, 1), (13, s))
    ),
    "the library's geo run": tuple(
        (s, 1) for block in (range(0, 20), range(20_261_002, 20_261_102)) for s in block
    ),
}

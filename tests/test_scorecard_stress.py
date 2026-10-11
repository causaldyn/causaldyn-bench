"""Families 2 to 7 through the scorecard, as an arm reads them: each world drawn from its seed
alone, from the blocks its family owns; at the ladder's rungs 0 and 4, or the deepest a short
series holds, exported and read back bit for bit; scored beside the built-in arms, the oracle's own
plan at nought; and beneath its stress, Track M v2's drawn world."""

import numpy as np
import pytest

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.endogenous_mmm import MediaMixWorld
from causaldyn_bench.lift_calibration import STARTS, TEST
from causaldyn_bench.scorecard.collinear import COLLINEAR
from causaldyn_bench.scorecard.confounded import CONFOUNDED
from causaldyn_bench.scorecard.continuation import Shadow
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.observe import VERSION, export, read
from causaldyn_bench.scorecard.outside import OUTSIDE
from causaldyn_bench.scorecard.performance import PERFORMANCE
from causaldyn_bench.scorecard.s_curves import S_CURVES
from causaldyn_bench.scorecard.scoring import score_world
from causaldyn_bench.scorecard.seeds import pilot, scored
from causaldyn_bench.scorecard.short import DEEPEST, SHORT
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

FAMILIES = {2: CONFOUNDED, 3: COLLINEAR, 4: S_CURVES, 5: PERFORMANCE, 6: OUTSIDE, 7: SHORT}
KEPT = {6, 7}  # the families whose stress leaves the history alone, and moves what reads past it


def _top(f: int, name: str) -> int:
    """The ladder's top rung for the environment: every test, or as many as a short series holds."""
    return DEEPEST[name] if f == 7 else len(STARTS)


def _first(f: int, e: int, *, pilots: bool = True) -> tuple[str, int]:
    family = FAMILIES[f]
    name = list(family.scored)[e]
    return name, (family.pilots if pilots else family.scored)[name][0]


@pytest.mark.parametrize("f", FAMILIES)
def test_a_family_draws_its_three_environments_from_its_own_blocks(f):
    family = FAMILIES[f]
    assert number(family) == f and tuple(family.pilots) == tuple(family.scored)
    assert len(family.scored) == 3
    for e, name in enumerate(family.scored):
        assert family.pilots[name] == pilot(f, e) and family.scored[name] == scored(f, e)
    assert all(label.endswith(".") for label in family.labels)
    with pytest.raises(ValueError, match="no environment 'drawn'"):
        family.world("drawn", scored(f, 0)[0])


@pytest.mark.parametrize("f", FAMILIES)
def test_a_world_is_its_seed_alone(f):
    family = FAMILIES[f]
    name, seed = _first(f, 1, pilots=False)
    first, again, other = (family.world(name, s) for s in (seed, seed, seed + 1))
    for k in (0, _top(f, name)):
        assert family.observe(first, k).digest() == family.observe(again, k).digest()
        assert family.observe(first, k).digest() != family.observe(other, k).digest()
    assert family.truth(first).best.worth == family.truth(again).best.worth
    np.testing.assert_array_equal(family.truth(first).best.weekly, family.truth(again).best.weekly)
    others = [family.observe(family.world(n, seed), 0).digest() for n in family.scored]
    assert len(set(others)) == 3  # each environment moves the world


@pytest.mark.parametrize("f", FAMILIES)
def test_an_arm_reads_its_rungs_tests_and_its_export_reads_back(f, tmp_path):
    family = FAMILIES[f]
    name, seed = _first(f, 0)
    world = family.world(name, seed)
    k = _top(f, name)
    none, top = family.observe(world, 0), family.observe(world, k)
    cut = world.history.week.size - top.sales.size  # the weeks a short series leaves out
    assert cut == 0 or f == 7
    assert len(none.lift.channel) == none.lift.dropped == 0
    assert len(top.lift.channel) + top.lift.dropped == k * len(top.channels)
    assert (top.family, top.environment, top.seed, top.k, top.version) == (
        f,
        name,
        seed,
        k,
        VERSION,
    )
    assert {start + cut for start in top.lift.start} <= set(STARTS)  # each row's first dark week
    for row, (channel, start) in enumerate(zip(top.lift.channel, top.lift.start, strict=True)):
        dark = world.history.spend[cut + start - 1 : cut + start - 1 + TEST]
        mean = float(np.mean(dark[:, top.channels.index(channel)]))
        assert top.lift.x[row] == pytest.approx(mean, rel=1e-12, abs=0.0)
    for observation in (none, top):
        back = read(export(observation, tmp_path / str(observation.k)))
        assert back.digest() == observation.digest()
        np.testing.assert_array_equal(back.spend, world.history.spend[cut:])
        np.testing.assert_array_equal(back.sales, world.history.sales[cut:])


@pytest.mark.parametrize("f", FAMILIES)
def test_a_world_is_scored_at_both_rungs_and_the_best_plan_scores_nought(f):
    family = FAMILIES[f]
    name, seed = _first(f, 2, pilots=False)
    world = family.world(name, seed)
    best = family.truth(world).best
    for k in (0, _top(f, name)):
        observation = family.observe(world, k)
        record = {
            "digest": observation.digest(),
            "version": VERSION,
            "error": None,
            "weekly": best.weekly.tolist(),
        }
        scores = score_world(family, name, seed, k, {"oracle": record})
        assert (scores.family, scores.environment, scores.index, scores.k) == (f, name, 0, k)
        assert abs(scores.arms["oracle"].regret) <= TIE
        for arm in scores.arms.values():
            assert arm.regret >= -TIE and arm.realised is None and arm.failure is None
        assert scores.arms["status quo"].regret > TIE


@pytest.mark.parametrize("f", FAMILIES)
def test_beneath_its_stress_a_world_is_track_m2s_drawn_world(f):
    """What the stress leaves alone is Track M v2's for the seed, bit for bit: the sales before
    media and every process beneath them, the channels' parameters, the measurement layer, and the
    quarter's processes; the quarter's effect path is each channel's coefficient."""
    family = FAMILIES[f]
    name, seed = _first(f, 2)
    world, base = family.world(name, seed), TRACK_M2.world("drawn", seed)
    kept = ("premedia", "baseline", "sentiment", "budget", "bidding", "observed_promotion")
    kept += ("observed_price", "retention", "saturation", "effect", "kernel_length", "week")
    for field in kept:
        assert np.array_equal(getattr(world.history, field), getattr(base.history, field)), field
    assert isinstance(world.history, MediaMixWorld)
    assert np.array_equal(world.history.sales, base.history.sales) == (f in KEPT)
    for field in ("sentiment", "baseline", "premedia", "observed_promotion", "observed_price"):
        assert np.array_equal(getattr(world.shadow, field), getattr(base.shadow, field)), field
    assert isinstance(world.shadow, Shadow)
    assert np.array_equal(world.expected, world.realised)
    effect = np.asarray(world.history.effect)[:, None]
    assert np.array_equal(world.expected, np.broadcast_to(effect, world.expected.shape))
    assert family.truth(world).realised is None

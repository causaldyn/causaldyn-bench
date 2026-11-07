"""Family 19: one channel's spend never moves, before the history or in it, so neither do its media;
every other channel and all the business read before media is Track M v2's; the truth states the
flat channel's return as every channel's, which its go-dark tests read; and a flat channel worth
half again as much moves the oracle's plan where its box lets it."""

import dataclasses

import numpy as np
import pytest

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.flat import ENVIRONMENTS, FLAT, Flat
from causaldyn_bench.scorecard.observe import VERSION, export, read
from causaldyn_bench.scorecard.scoring import score_world
from causaldyn_bench.scorecard.seeds import pilot, scored
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, still, truth, window
from causaldyn_bench.scorecard.truth import worth


def test_the_family_draws_its_environments_from_its_own_blocks():
    assert number(FLAT) == 19 and tuple(FLAT.pilots) == tuple(FLAT.scored) == ENVIRONMENTS
    for e, name in enumerate(ENVIRONMENTS):
        assert FLAT.pilots[name] == pilot(19, e) and FLAT.scored[name] == scored(19, e)
    assert all(label.endswith(".") for label in FLAT.labels)
    with pytest.raises(ValueError, match="no environment 'drawn'"):
        FLAT.world("drawn", FLAT.pilots["tv"][0])


@pytest.mark.parametrize("name", ENVIRONMENTS)
def test_the_flat_channel_never_moves_and_nothing_else_does(name):
    for seed in FLAT.pilots[name][:2]:
        world = FLAT.world(name, seed).history
        base = TRACK_M2.world("drawn", seed).history
        assert isinstance(world, Flat)
        c = world.channels.index(name)
        assert world.flat == c and world.level == float(np.mean(base.spend[:, c]))
        np.testing.assert_array_equal(world.spend[:, c], world.level)
        assert np.ptp(world.media[:, c]) <= 1e-12 * world.media[0, c]
        others = [o for o in range(3) if o != c]
        np.testing.assert_array_equal(world.spend[:, others], base.spend[:, others])
        np.testing.assert_array_equal(world.media[:, others], base.media[:, others])
        for field in ("premedia", "budget", "bidding", "plan_noise", "sentiment", "effect"):
            assert np.array_equal(getattr(world, field), getattr(base, field)), field
        np.testing.assert_array_equal(world.sales, world.premedia + world.media.sum(axis=1))


def test_the_truth_states_the_flat_channels_return_and_its_tests_read_its_level():
    """Its ROI on last year: what the world's own media lose, over the history and the kernel's
    tail, with the window's spend removed, over what the window spent, the same number by another
    path than :mod:`.returns`. Each of its go-dark tests takes the level dark."""
    name = "tv"
    seed = FLAT.pilots[name][0]
    world = FLAT.world(name, seed)
    history = world.history
    assert isinstance(history, Flat)
    c = history.channels.index(name)
    first, last = window(history)
    tail = np.zeros(history.kernel_length - 1)
    spend = np.concatenate([history.spend[:, c], tail])
    without = spend.copy()
    without[first - 1 : last] = 0.0
    lost = history._effect_of(c, spend) - history._effect_of(c, without)
    expected = float(lost.sum()) / float(spend[first - 1 : last].sum())
    returns = FLAT.truth(world).returns
    assert returns.roi[c] == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert returns.roi[c] > 0.0 and returns.marginal[c] > 0.0
    lift = FLAT.observe(world, 4).lift
    rows = [r for r, channel in enumerate(lift.channel) if channel == name]
    assert rows
    np.testing.assert_array_equal(lift.x[rows], history.level)
    np.testing.assert_array_equal(lift.delta_x[rows], -history.level)


def test_a_flat_channel_worth_half_again_as_much_moves_the_oracle():
    """Television flat on the first pilot world: half again its coefficient lifts its ROI by half
    and its spend in the best plan off its floor, and the plan made for it before leaves 0.062 of a
    euro of the budget (asserted: more than 0.05)."""
    seed = FLAT.pilots["tv"][0]
    world = FLAT.world("tv", seed)
    c = world.history.channels.index("tv")
    effect = list(world.history.effect)
    effect[c] *= 1.5
    history = dataclasses.replace(world.history, effect=tuple(effect))
    swapped = dataclasses.replace(world, history=history, expected=still(history))
    swapped = dataclasses.replace(swapped, realised=swapped.expected)
    before, after = FLAT.truth(world), truth(swapped)
    ratio = after.returns.roi / before.returns.roi
    np.testing.assert_allclose(ratio, [1.5 if o == c else 1.0 for o in range(3)], rtol=1e-12)
    assert before.best.weekly[c] == pytest.approx(before.quarter.lower[c], rel=1e-9, abs=0.0)
    assert after.best.weekly[c] > before.best.weekly[c] + 10.0
    left = (after.best.worth - worth(after.cells, before.best.weekly)) / after.quarter.budget
    assert left > 0.05


def test_an_arm_reads_a_flat_world_and_its_best_plan_scores_nought(tmp_path):
    name = "meta"
    seed = FLAT.pilots[name][0]
    world = FLAT.world(name, seed)
    observation = FLAT.observe(world, 4)
    back = read(export(observation, tmp_path))
    assert back.digest() == observation.digest()
    np.testing.assert_array_equal(back.spend, world.history.spend)
    record = {
        "digest": observation.digest(),
        "version": VERSION,
        "error": None,
        "weekly": FLAT.truth(world).best.weekly.tolist(),
    }
    scores = score_world(FLAT, name, seed, 4, {"oracle": record}, pilot=True)
    assert abs(scores.arms["oracle"].regret) <= TIE
    assert scores.arms["status quo"].regret > TIE

"""Family 18: every world reads a brand measure, the branded searches, which the measurement layer
does not; in ``unread`` television builds them and sales do not read them, in ``unbuilt`` sales read
them and television builds none, so the long-term share of every channel's return is 0 and the
searches move no channel's return and no plan; with both links on, the share is television's and
above 0, which the truth reader reports."""

import math

import numpy as np
import pytest

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.long_term import (
    BUILT,
    CUTS,
    LINK,
    LONG_TERM,
    Branded,
    branded,
    carried,
    long_term,
)
from causaldyn_bench.scorecard.observe import VERSION, export, read
from causaldyn_bench.scorecard.scoring import score_world
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored
from causaldyn_bench.scorecard.track_m2 import TRACK_M2


def test_the_family_draws_its_cuts_from_its_own_blocks():
    assert number(LONG_TERM) == 18 and tuple(LONG_TERM.pilots) == tuple(LONG_TERM.scored)
    assert tuple(LONG_TERM.scored) == ("unread", "unbuilt")
    for e, name in enumerate(CUTS):
        assert LONG_TERM.pilots[name] == pilot(18, e) and LONG_TERM.scored[name] == scored(18, e)
    assert all(label.endswith(".") for label in LONG_TERM.labels)
    with pytest.raises(ValueError, match="no environment 'drawn'"):
        LONG_TERM.world("drawn", LONG_TERM.pilots["unread"][0])


@pytest.mark.parametrize("name", CUTS)
def test_every_arm_reads_the_searches_the_cut_builds(name):
    """The searches are 500 a week, a yearly wave of 50, what television builds, and a normal noise
    of sd 15 from the family's measurement stream over the history and its continuation stream over
    the quarter, at the status quo; the sales read them at the cut's link."""
    built, link = CUTS[name]
    for seed in LONG_TERM.pilots[name][:2]:
        world = LONG_TERM.world(name, seed)
        base = TRACK_M2.world("drawn", seed)
        history = world.history
        assert isinstance(world, Branded) and (world.built, world.link) == (built, link)
        tv = history.spend[:, history.channels.index("tv")]
        quo = Quarter.after(base.history).status_quo[history.channels.index("tv")]
        whole = np.concatenate([tv, np.full(PLANNED, quo)])
        scale = float(np.mean(tv[tv > 0.0]))
        h = carried(whole, scale)
        weeks = tv.size
        t = np.arange(1, weeks + PLANNED + 1)
        noise = np.concatenate(
            [
                rng(18, Role.MEASUREMENT, seed).standard_normal(weeks),
                rng(18, Role.CONTINUATION, seed).standard_normal(PLANNED),
            ]
        )
        per = tv.sum() / h[:weeks].sum()
        searches = 500.0 + 50.0 * np.sin(2.0 * math.pi * t / 52) + built * per * h + 15.0 * noise
        observation = LONG_TERM.observe(world, 0)
        assert observation.future_controls is not None
        np.testing.assert_allclose(observation.controls["searches"], searches[:weeks], rtol=1e-14)
        np.testing.assert_allclose(
            observation.future_controls["searches"], searches[weeks:], rtol=1e-14
        )
        # television builds ``built`` searches a unit of its spend over the history
        built_over = world.built_searches(whole)[:weeks].sum()
        assert built_over == pytest.approx(built * tv.sum(), rel=1e-12, abs=1e-9)
        np.testing.assert_allclose(
            history.sales, base.history.sales + link * (searches[:weeks] - 500.0), rtol=1e-14
        )
        np.testing.assert_allclose(
            world.shadow.sales,
            base.shadow.sales + link * (searches[weeks:] - 500.0),
            rtol=1e-14,
        )


@pytest.mark.parametrize("name", CUTS)
def test_the_long_term_share_is_nought_and_the_searches_move_no_plan(name):
    for seed in LONG_TERM.pilots[name][:2]:
        world = LONG_TERM.world(name, seed)
        truth = LONG_TERM.truth(world)
        base = TRACK_M2.truth(TRACK_M2.world("drawn", seed))
        assert truth.long_term is not None
        np.testing.assert_array_equal(truth.long_term, 0.0)
        np.testing.assert_array_equal(truth.returns.roi, base.returns.roi)
        np.testing.assert_array_equal(truth.best.weekly, base.best.weekly)


def test_with_both_links_on_the_share_is_televisions_and_above_nought():
    """The fixture the cuts guard against: television builds the searches and sales read them. The
    truth reader then reports a share above 0 for television alone: 0.52 on the first pilot world
    of ``unread`` (asserted: between 0.1 and 0.9)."""
    world = LONG_TERM.world("unread", LONG_TERM.pilots["unread"][0])
    both = branded(world, BUILT, LINK)
    roi = LONG_TERM.truth(world).returns.roi
    share = long_term(both, roi)
    tv = world.history.channels.index("tv")
    assert 0.1 < share[tv] < 0.9
    np.testing.assert_array_equal(np.delete(share, tv), 0.0)
    np.testing.assert_array_equal(long_term(world, roi), 0.0)


def test_an_arm_reads_the_searches_back_and_the_best_plan_scores_nought(tmp_path):
    name = "unbuilt"
    seed = LONG_TERM.pilots[name][0]
    world = LONG_TERM.world(name, seed)
    observation = LONG_TERM.observe(world, 4)
    back = read(export(observation, tmp_path))
    assert back.digest() == observation.digest()
    np.testing.assert_array_equal(back.controls["searches"], observation.controls["searches"])
    record = {
        "digest": observation.digest(),
        "version": VERSION,
        "error": None,
        "weekly": LONG_TERM.truth(world).best.weekly.tolist(),
    }
    scores = score_world(LONG_TERM, name, seed, 4, {"oracle": record}, pilot=True)
    assert abs(scores.arms["oracle"].regret) <= TIE
    assert scores.arms["status quo"].regret > TIE

"""Family 20: each week is a promotion week by a draw of its own, at the environment's chances in
television's burst weeks and in the others, so the promotions follow television in ``tv-0.5`` and
``tv-0.8`` and not in ``independent``; everything that reads the calendar reads the drawn one; the
price lever's contribution against its stated level is the sales before media less the baseline at
that level, which replaying the generator's draws recomputes; a stated level of another price moves
it, and another calendar on the same draws moves the oracle."""

import numpy as np
import pytest

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.endogenous_mmm import ELASTICITY, _ar1, _calendar
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.observe import VERSION, export, read
from causaldyn_bench.scorecard.promotions import (
    CHANCES,
    CUT,
    LEVEL,
    PROMOTIONS,
    Promoted,
    bursting,
    contribution,
)
from causaldyn_bench.scorecard.scoring import score_world
from causaldyn_bench.scorecard.seeds import pilot, scored
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, window


def test_the_family_draws_its_environments_from_its_own_blocks():
    assert number(PROMOTIONS) == 20 and tuple(PROMOTIONS.pilots) == tuple(PROMOTIONS.scored)
    assert tuple(PROMOTIONS.scored) == ("independent", "tv-0.5", "tv-0.8")
    for e, name in enumerate(CHANCES):
        assert PROMOTIONS.pilots[name] == pilot(20, e) and PROMOTIONS.scored[name] == scored(20, e)
    assert all(label.endswith(".") for label in PROMOTIONS.labels)
    with pytest.raises(ValueError, match="no environment 'drawn'"):
        PROMOTIONS.world("drawn", PROMOTIONS.pilots["tv-0.5"][0])


def test_the_generator_on_his_calendar_is_track_m2_bit_for_bit():
    seed = PROMOTIONS.pilots["independent"][0]
    world = TRACK_M2.world("drawn", seed)
    week = world.history.week
    for got, want in zip(world.generator.calendar(week), _calendar(week), strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("name", CHANCES)
def test_the_promotions_fall_at_the_environments_chances(name):
    """Over the 100 pilot worlds' 156 weeks, the share of television's burst weeks that hold a
    promotion, and of the other weeks, are each within four standard errors of the environment's
    chances, and the promotion weeks a year within four of 8.4 (each a binomial's error)."""
    heavy, light = CHANCES[name]
    hits = np.zeros(2)
    weeks = np.zeros(2)
    for seed in PROMOTIONS.pilots[name]:
        history = PROMOTIONS.world(name, seed).history
        on = history.discount > 0.0
        burst = bursting(history.week)
        hits += (on[burst].sum(), on[~burst].sum())
        weeks += (burst.sum(), (~burst).sum())
    for chance, hit, n in zip((heavy, light), hits, weeks, strict=True):
        assert abs(hit / n - chance) < 4.0 * np.sqrt(chance * (1.0 - chance) / n)
    years = 3 * len(PROMOTIONS.pilots[name])
    per_year = hits.sum() / years
    rate = (heavy * weeks[0] + light * weeks[1]) / weeks.sum()
    assert abs(per_year - 8.4) < 0.2 + 4.0 * 52 * np.sqrt(rate * (1.0 - rate) / weeks.sum())


def test_the_promotions_follow_television_where_the_environment_says():
    """Over the first 20 pilot worlds of each environment, the mean correlation of the promotion
    weeks with television's spend: 0.005 in ``independent``, 0.38 in ``tv-0.5`` and 0.70 in
    ``tv-0.8`` (asserted: below 0.1, above 0.25 and above 0.55)."""
    means = {}
    for name in CHANCES:
        correlations = []
        for seed in PROMOTIONS.pilots[name][:20]:
            history = PROMOTIONS.world(name, seed).history
            tv = history.spend[:, history.channels.index("tv")]
            correlations.append(np.corrcoef((history.discount > 0.0).astype(float), tv)[0, 1])
        means[name] = float(np.mean(correlations))
    assert abs(means["independent"]) < 0.1
    assert means["tv-0.5"] > 0.25 and means["tv-0.8"] > 0.55


def test_everything_that_reads_the_calendar_reads_the_drawn_one():
    seed = PROMOTIONS.pilots["tv-0.8"][0]
    world = PROMOTIONS.world("tv-0.8", seed)
    generator, history = world.generator, world.history
    assert isinstance(generator, Promoted)
    on = np.isin(history.week, generator.promoted)
    np.testing.assert_array_equal(history.discount, np.where(on, CUT, 0.0))
    seen = history.observed_promotion.astype(bool)
    assert seen[on].mean() > 0.5 > seen[~on].mean()  # caught at 0.7, read falsely at 0.15
    uplift = (1.0 - CUT) ** ELASTICITY - 1.0
    np.testing.assert_allclose(history.promotion, np.where(on, history.baseline * uplift, 0.0))
    quarter = world.shadow
    later = history.week[-1] + np.arange(1, quarter.discount.size + 1)
    np.testing.assert_array_equal(quarter.discount > 0.0, np.isin(later, generator.promoted))
    assert not np.array_equal(history.discount, TRACK_M2.world("drawn", seed).history.discount)


def test_the_price_levers_contribution_is_the_sales_before_media_above_its_level():
    """Replay the generator's draws for the seed, in its order, to rebuild the baseline at the
    stated level, undiscounted: the sales before media over last year less it is the truth's
    lever, to rounding. A level of 0.9 or 1.1 moves it."""
    for name in ("independent", "tv-0.8"):
        seed = PROMOTIONS.pilots[name][0]
        world = PROMOTIONS.world(name, seed)
        history, base = world.history, world.generator.base
        draws = np.random.default_rng(seed)
        weeks = history.week.size
        quality = _ar1(draws, weeks, 0.95, 0.0, 0.02, start=0.0)
        _ar1(draws, weeks, 0.9, 0.1, 0.03, start=1.0)
        sentiment = _ar1(draws, weeks, 0.7, 0.0, 0.03, start=float(draws.normal(0.0, 0.05)))
        noise = draws.normal(0.0, 0.02 * base, weeks)
        at_level = (
            base * (1 + history.seasonality) * (1 + quality) * LEVEL**ELASTICITY
            + base * sentiment
            + noise
        )
        at_level = np.maximum(at_level, 0.5 * base)
        first, last = window(history)
        inside = slice(first - 1, last)
        expected = float(np.sum(history.premedia[inside] - at_level[inside]))
        truth = PROMOTIONS.truth(world)
        assert truth.levers is not None
        assert truth.levers["price"] == pytest.approx(expected, rel=1e-12, abs=0.0)
        assert contribution(world, 0.9) < truth.levers["price"] < contribution(world, 1.1)


def test_another_calendar_on_the_same_draws_moves_the_oracle():
    """The same seed in ``independent`` and in ``tv-0.8``: the calendar moves the spend raised ahead
    of each promotion and the quarter's promotions, so the quarter and its best plan move."""
    seed = PROMOTIONS.pilots["tv-0.8"][0]
    plans = [PROMOTIONS.truth(PROMOTIONS.world(name, seed)).best.weekly for name in CHANCES]
    assert not np.allclose(plans[0], plans[2], rtol=1e-6)


def test_an_arm_reads_a_promotions_world_and_its_best_plan_scores_nought(tmp_path):
    name = "tv-0.5"
    seed = PROMOTIONS.pilots[name][0]
    world = PROMOTIONS.world(name, seed)
    observation = PROMOTIONS.observe(world, 4)
    back = read(export(observation, tmp_path))
    assert back.digest() == observation.digest()
    np.testing.assert_array_equal(back.controls["promotion"], observation.controls["promotion"])
    record = {
        "digest": observation.digest(),
        "version": VERSION,
        "error": None,
        "weekly": PROMOTIONS.truth(world).best.weekly.tolist(),
    }
    scores = score_world(PROMOTIONS, name, seed, 4, {"oracle": record}, pilot=True)
    assert abs(scores.arms["oracle"].regret) <= TIE
    assert scores.arms["status quo"].regret > TIE

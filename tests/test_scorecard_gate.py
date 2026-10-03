"""The gate: its boundaries against the published four looks, Genz's integration of the looks'
normal law, Maxima's integral of two looks and Brownian paths; a look's verdict on scored worlds;
and the reading's size where each look estimates its variance.
"""

import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm
from scipy.stats import t as student

from causaldyn_bench.scorecard import gate
from causaldyn_bench.scorecard.gate import ALPHA, Gate, boundaries, spent
from causaldyn_bench.scorecard.scoring import ArmScore, WorldRecord

QUARTERS = (25, 50, 75, 100)
DESIGNS = [
    ((0.25, 0.5, 0.75, 1.0), 0.025),
    ((0.2, 0.45, 0.7, 0.85, 1.0), 0.025),
    ((0.5, 1.0), 0.025),
    ((0.3, 0.6, 1.0), 0.05),
    ((0.5, 0.52, 1.0), 0.025),  # a step far shorter than the one before it
]


def _cumulative(fractions, bounds):
    """The chance some look among the first k crosses its boundary, by Genz's integration."""
    t = np.asarray(fractions)
    out = []
    for k in range(1, t.size + 1):
        shares = t[:k]
        correlation = np.sqrt(np.minimum.outer(shares, shares) / np.maximum.outer(shares, shares))
        below = multivariate_normal.cdf(
            np.asarray(bounds[:k]),
            mean=np.zeros(k),
            cov=correlation,
            maxpts=2_000_000,
            abseps=1e-10,
            releps=1e-10,
            rng=np.random.default_rng(k),
        )
        out.append(1.0 - float(below))
    return out


@pytest.mark.parametrize(("fractions", "alpha"), DESIGNS)
def test_the_first_look_spends_its_share_in_closed_form(fractions, alpha):
    first = boundaries(fractions, alpha)[0]
    assert first == pytest.approx(norm.isf(spent(fractions[0], alpha)), abs=1e-9)


def test_four_equal_looks_reach_the_published_boundaries():
    # Lan and DeMets' O'Brien-Fleming type at one-sided 0.025, four equal looks
    assert Gate(QUARTERS).bounds == pytest.approx((4.3326, 2.9631, 2.3590, 2.0141), abs=5e-5)


@pytest.mark.parametrize(("fractions", "alpha"), DESIGNS)
def test_the_looks_spend_alpha_as_genz_integrates_their_normal_law(fractions, alpha):
    spending = [spent(t, alpha) for t in fractions]
    # Genz's quasi-Monte Carlo misses by up to 3.6e-7 here at this many points: the precision is
    # held by Maxima's two looks and by refining the grid
    assert _cumulative(fractions, boundaries(fractions, alpha)) == pytest.approx(spending, abs=5e-7)


@pytest.mark.parametrize(("fractions", "alpha"), DESIGNS)
def test_a_grid_four_times_finer_moves_no_boundary(fractions, alpha, monkeypatch):
    coarse = boundaries(fractions, alpha)
    monkeypatch.setattr(gate, "PER_SD", 4 * gate.PER_SD)
    assert boundaries(fractions, alpha) == pytest.approx(coarse, abs=1e-7)


def test_two_looks_reach_maxima_s_integral():
    # Maxima 5.50, find_root over quad_qags: P(Z1 >= c1) = alpha(1/2), and
    # alpha(1/2) + integral of phi(z) (1 - Phi((c2 - z / sqrt(2)) sqrt(2))) dz over z < c1 = 0.025
    assert boundaries((0.5, 1.0)) == pytest.approx(
        (2.9625880427275764, 1.9685956406374354), abs=1e-8
    )


def test_a_look_that_spends_nothing_a_double_holds_cannot_show_and_leaves_the_rest_to_the_last():
    first, last = boundaries((1e-4, 1.0))
    assert first == math.inf
    assert last == pytest.approx(norm.isf(ALPHA), abs=1e-7)


def test_under_the_null_the_crossings_spend_alpha_look_by_look():
    looks = (10, 25, 30, 48, 60)
    crossings = Gate(looks).crossings(0.0)
    spending = [spent(n / looks[-1]) for n in looks]
    assert np.cumsum(crossings.shown) == pytest.approx(spending, abs=1e-9)
    assert crossings.futile == (0.0,) * len(looks)


def test_a_futility_threshold_moves_no_boundary_and_spends_less_than_alpha():
    bound, free = Gate(QUARTERS, futility=0.0), Gate(QUARTERS)
    assert bound.bounds == free.bounds
    crossings = bound.crossings(0.0)
    assert sum(crossings.shown) < ALPHA - 1e-3
    assert crossings.futile[0] == pytest.approx(0.5, abs=1e-9)
    assert crossings.futile[-1] == 0.0


def _paths(fractions, drift, paths, seed):
    """Brownian paths with drift ``drift`` read at the looks, as Z statistics, one row a path."""
    t = np.asarray(fractions)
    steps = np.diff(t, prepend=0.0)
    rng = np.random.default_rng(seed)
    score = np.cumsum(rng.normal(drift * steps, np.sqrt(steps), size=(paths, t.size)), axis=1)
    return score / np.sqrt(t)


@pytest.mark.parametrize(
    ("drift", "futility"), [(0.0, None), (2.8, None), (2.8, 0.0), (-1.0, 0.5), (6.0, 0.0)]
)
def test_the_crossings_follow_brownian_paths(drift, futility):
    gate_ = Gate(QUARTERS, futility=futility)
    paths = 400_000
    z = _paths(gate_.fractions, drift, paths, seed=17)
    last = len(QUARTERS) - 1
    going = np.ones(paths, dtype=bool)
    shown, futile, read = [], [], np.zeros(paths)
    for k, (share, bound) in enumerate(zip(gate_.fractions, gate_.bounds, strict=True)):
        up = going & (z[:, k] >= bound)
        floor = futility if futility is not None and k < last else -np.inf
        down = going & ~up & (z[:, k] < floor)
        shown.append(up.mean())
        futile.append(down.mean())
        read[going] = share
        going &= ~(up | down)
    expected = gate_.crossings(drift)
    for chance, frequency in zip(expected.shown + expected.futile, shown + futile, strict=True):
        assert abs(frequency - chance) <= 4.5 * math.sqrt(max(chance * (1 - chance), 1e-12) / paths)
    assert read.mean() == pytest.approx(expected.expected, abs=4.5 * read.std() / math.sqrt(paths))


def _seen(differences, sizes, looks):
    """Each look's (worlds, n, mean, sd) over the first ``n`` of each row's differences."""
    for worlds, n in zip(looks, sizes, strict=True):
        head = differences[:n]
        yield worlds, n, float(head.mean()), float(head.std(ddof=1))


def test_the_reading_holds_its_alpha_where_each_look_estimates_its_variance():
    # 20 to 80 differences a look: a boundary left on the normal scale shows 0.028 here
    looks, environments, margin, replicates = (10, 20, 30, 40), 2, 0.01, 60_000
    sizes = [worlds * environments for worlds in looks]
    rng = np.random.default_rng(23)
    draws = rng.normal(margin, 0.08, size=(replicates, sizes[-1]))
    bounds = Gate(looks).bounds
    shown = sum(
        gate._reading(_seen(row, sizes, looks), bounds, margin, None)[-1].verdict == "shown"
        for row in draws
    )
    assert abs(shown / replicates - ALPHA) <= 3 * math.sqrt(ALPHA * (1 - ALPHA) / replicates)


def _records(differences, environments=("drawn", "reference")):
    """Scored worlds whose arm ``a`` less arm ``b`` is ``differences[e][i]``."""
    records = []
    for environment, values in zip(environments, differences, strict=True):
        for i, value in enumerate(values):
            arms = {
                arm: ArmScore(regret, None, None, 0, None, None, None, None, {})
                for arm, regret in (("a", 0.1 + float(value)), ("b", 0.1))
            }
            records.append(
                WorldRecord(0, environment, 10 * i, i, False, 4, 1.0, 1.0, 3, 0, {}, arms)
            )
    return records


def _noise(environments, worlds, seed):
    return np.random.default_rng(seed).normal(0.0, 0.08, size=(environments, worlds))


def test_a_plain_claim_is_shown_at_the_first_look_with_the_numbers_it_was_read_on():
    differences = -0.06 + _noise(2, 100, 1)
    gate_ = Gate(QUARTERS)
    (look,) = gate_.read(_records(differences), "a", "b", margin=0.01)
    first = differences[:, :25].ravel()
    mean, sd = first.mean(), first.std(ddof=1)
    assert (look.worlds, look.n, look.verdict) == (25, 50, "shown")
    assert (look.mean, look.sd) == pytest.approx((mean, sd), abs=1e-12)
    assert look.statistic == pytest.approx((0.01 - mean) / (sd / math.sqrt(50)), rel=1e-9)
    assert look.boundary == pytest.approx(student.isf(norm.sf(gate_.bounds[0]), 49), rel=1e-12)


def test_a_comparison_that_is_not_plain_is_read_to_the_last_look_and_not_shown():
    looks = Gate(QUARTERS).read(_records(0.01 + _noise(2, 100, 2)), "a", "b", margin=0.01)
    assert [(look.worlds, look.n, look.verdict) for look in looks] == [
        (25, 50, "continue"),
        (50, 100, "continue"),
        (75, 150, "continue"),
        (100, 200, "not shown"),
    ]


def test_a_futility_threshold_stops_a_comparison_the_first_arm_plainly_loses():
    records = _records(0.05 + _noise(2, 100, 3))
    assert [
        look.verdict for look in Gate(QUARTERS, futility=0.0).read(records, "a", "b", margin=0.01)
    ] == ["futile"]
    assert len(Gate(QUARTERS).read(records, "a", "b", margin=0.01)) == 4


def test_the_futility_threshold_is_read_on_students_scale_and_never_at_the_last_look():
    bounds = Gate(QUARTERS).bounds
    moved = float(student.isf(norm.sf(-0.5), 19))  # -0.507: a statistic of -0.503 lies between
    sd = 0.08
    mean = 0.01 + 0.503 * sd / math.sqrt(20)
    (look,) = gate._reading(iter([(10, 20, mean, sd)]), bounds, 0.01, -0.5)
    assert moved < look.statistic < -0.5
    assert look.verdict == "continue"
    last = gate._reading(
        iter([(10, 20, 0.0, sd), (20, 40, 0.0, sd), (30, 60, 0.0, sd), (40, 80, 0.5, sd)]),
        bounds,
        0.01,
        -0.5,
    )
    assert [look.verdict for look in last] == ["continue", "continue", "continue", "not shown"]


def test_only_the_looks_the_records_hold_whole_are_read():
    differences = 0.01 + _noise(2, 100, 4)
    records = _records(differences)
    held = [r for r in records if r.environment == "drawn" or r.index < 50]
    looks = Gate(QUARTERS).read(held, "a", "b", margin=0.01)
    assert [(look.worlds, look.verdict) for look in looks] == [(25, "continue"), (50, "continue")]
    gappy = [r for r in records if r.index != 30]
    assert [look.worlds for look in Gate(QUARTERS).read(gappy, "a", "b", margin=0.01)] == [25]
    assert Gate(QUARTERS).read(records[:24], "a", "b", margin=0.01) == ()
    assert Gate(QUARTERS).read([], "a", "b", margin=0.01) == ()


def test_differences_that_do_not_spread_show_a_gap_plainly_and_no_gap_not_at_all():
    records = _records(np.zeros((2, 100)))
    (inside,) = Gate(QUARTERS).read(records, "a", "b", margin=0.01)
    assert (inside.statistic, inside.verdict) == (math.inf, "shown")
    level = Gate(QUARTERS).read(records, "a", "b", margin=0.0)
    assert [(look.statistic, look.verdict) for look in level][-1] == (0.0, "not shown")
    (outside,) = Gate(QUARTERS, futility=0.0).read(records, "a", "b", margin=-0.01)
    assert (outside.statistic, outside.verdict) == (-math.inf, "futile")


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Gate(()), "at least 2 worlds"),
        (lambda: Gate((1, 4)), "at least 2 worlds"),
        (lambda: Gate((10, 10, 20)), "increase strictly"),
        (lambda: Gate((20, 10)), "increase strictly"),
        (lambda: Gate(QUARTERS, alpha=0.0), "alpha lies in"),
        (lambda: Gate(QUARTERS, alpha=0.5), "alpha lies in"),
        (lambda: Gate(QUARTERS, futility=2.97), "stops every look it does not show"),
        (lambda: boundaries((0.5, 0.9)), "increase strictly"),
        (lambda: boundaries((0.0, 1.0)), "increase strictly"),
        (lambda: spent(0.0), "share of the worlds"),
        (lambda: spent(1.5), "share of the worlds"),
    ],
)
def test_a_gate_refuses_a_design_it_cannot_read(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_a_look_refuses_records_it_cannot_pair():
    records = _records(_noise(2, 30, 5))
    with pytest.raises(ValueError, match="paired difference reads one of"):
        Gate(QUARTERS).read(records, "a", "b", margin=0.01, axis="profit")
    with pytest.raises(ValueError, match="regret is not scored"):
        Gate(QUARTERS).read(records, "a", "c", margin=0.01)

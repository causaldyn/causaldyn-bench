"""Track N: the fold-design head-to-head, and the two facts that make it readable."""

import numpy as np
import pytest

from causaldyn_bench.fold_design import (
    fold_design_report,
    random_regular_graph,
    track_fold_design,
)
from causaldyn_bench.paper_two import RatioCI, arm_table


def test_random_regular_graph_is_simple_and_regular() -> None:
    adjacency = np.asarray(random_regular_graph(12, 3, 5))
    assert np.array_equal(adjacency, adjacency.T)
    assert np.all(np.diag(adjacency) == 0)
    assert np.all(adjacency.sum(axis=1) == 3)
    assert set(np.unique(adjacency).tolist()) <= {0, 1}


def test_a_regular_graph_that_cannot_exist_is_refused() -> None:
    with pytest.raises(ValueError, match="degree sum is odd"):
        random_regular_graph(5, 3, 0)
    with pytest.raises(ValueError, match="cannot have degree"):
        random_regular_graph(4, 4, 0)


def test_the_obvious_graph_aware_split_is_the_expensive_one() -> None:
    """The track's claim is an ORDERING: designed ~ random units << contiguous < exclusion.

    Asserted through paired bootstrap intervals, not point estimates. A TIE is an interval that
    fails to convict, and a COST is an interval that clears 1; a tolerance around a point estimate
    can state neither.

    The earlier version asserted ``abs(designed - units) < 0.2`` at 60 draws and failed under
    ``JAX_ENABLE_X64=1`` with 0.219. Not because x64 changes the estimator: threefry spends a
    different number of bits per float64 element, so x64 draws a DIFFERENT sample, and the two
    dtypes put the designed arm at 1.04 and 0.78 on the direct coefficient. The module docstring
    already says 120 draws does not resolve designed against random units, so a 0.2 tolerance on
    that pair was a coin flip on which sample the run happened to get.

    What survives at both dtypes and both draw counts, and is what is asserted:

    * the designed arm is never CONVICTED of costing anything -- its interval reaches below 1;
    * neighbour exclusion always is -- its interval clears 1 on both coefficients, worst lower
      bound 1.166 over the four configurations measured;
    * and the two are cleanly separated, ``designed.hi < exclusion.lo``, worst margin 0.070.

    Deliberately NOT asserted: that CONTIGUOUS clears 1. At 120 draws it does on the spillover
    coefficient at both dtypes (lower bounds 1.147 and 1.139) and on the direct coefficient it is
    sample-dependent -- [1.103, 1.735] under float32 against [0.934, 1.599] under float64, at the
    4000 resamples this test uses; the paper table quotes [1.101, 1.732] at 10 000. The +37%
    headline is a point estimate whose interval does not separate from the baseline at g = 2 on
    that coefficient, and the honest claim there is the ordering, not the separation.
    """
    table = arm_table("cycle", clusters=2, draws=120, n_boot=4000)
    for coefficient in ("direct", "spillover"):
        units = table["random units"][coefficient]
        designed = table["designed folds"][coefficient]
        contiguous = table["contiguous folds"][coefficient]
        exclusion = table["neighbour exclusion"][coefficient]
        assert units == RatioCI(1.0, 1.0, 1.0)  # the pairing, pinned: an arm against itself
        assert designed.lo < 1.0
        assert exclusion.lo > 1.0
        assert designed.hi < exclusion.lo
        assert contiguous.ratio > designed.ratio + 0.25
        assert exclusion.ratio > contiguous.ratio


def test_the_leaderboard_arms_are_wired_to_the_layouts_they_name() -> None:
    """``track_fold_design`` is what the leaderboard calls, so its wiring needs its own check.

    Cheap and non-stochastic on purpose: the ordering is asserted above with intervals, and
    repeating it here on six draws would only re-test draw noise.
    """
    results = track_fold_design(clusters=2, seeds=6)
    assert {r.track for r in results} == {"N-fold-direct", "N-fold-spillover"}
    assert {r.method for r in results} == {
        "random rows",
        "random units",
        "contiguous folds",
        "neighbour exclusion",
        "designed folds (chc)",
    }
    baseline = [r.value for r in results if r.method == "random units"]
    assert baseline == pytest.approx([1.0, 1.0])
    assert all(np.isfinite(r.value) for r in results)  # exclusion can run on the cycle


def test_the_predicted_gap_is_reported_beside_the_realised_one() -> None:
    """A design law that cannot be checked against an estimator is a claim, not a result."""
    report = fold_design_report(cluster_grid=(2,), seeds=20)
    assert set(report) == {"cycle", "torus", "cubic"}
    # the cycle is where blocks are catastrophic and the functional says so; the other two are
    # where it says there is nothing to win, and the measurement agrees.
    assert report["cycle"]["predicted_ratio"] < 0.75
    assert report["torus"]["predicted_ratio"] > 0.95
    assert report["cubic"]["predicted_ratio"] > 0.95


def test_neighbour_exclusion_is_infeasible_on_the_denser_topologies() -> None:
    """At K = 2 the test fold's hop-1 neighbourhood covers the training fold and empties it."""
    report = fold_design_report(cluster_grid=(2,), seeds=8)
    for topology in ("torus", "cubic"):
        assert report[topology]["exclusion/direct/g2"] == float("inf")
    assert np.isfinite(report["cycle"]["exclusion/direct/g2"])

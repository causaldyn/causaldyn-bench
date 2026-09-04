"""Track N: the fold-design head-to-head, and the two facts that make it readable."""

import numpy as np
import pytest

from causaldyn_bench.fold_design import (
    fold_design_report,
    random_regular_graph,
    track_fold_design,
)


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

    Designed and random-unit folds tie, and the design law says why: on ``C_12`` they keep 50% and
    46% of the edges inside a fold, while contiguous blocks keep 83%. The value of the law here is
    that it convicts the split a practitioner would reach for first, before any simulation.
    """
    results = {(r.track, r.method): r.value for r in track_fold_design(clusters=2, seeds=60)}
    for coefficient in ("direct", "spillover"):
        track = f"N-fold-{coefficient}"
        designed = results[(track, "designed folds (chc)")]
        units = results[(track, "random units")]
        contiguous = results[(track, "contiguous folds")]
        exclusion = results[(track, "neighbour exclusion")]
        assert units == pytest.approx(1.0)
        assert abs(designed - units) < 0.2  # a tie, and the edge fractions above say why
        assert contiguous > units + 0.2
        assert exclusion > contiguous


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

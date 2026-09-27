"""P2's table generator: the pairing that makes its intervals mean anything, and one convention."""

import numpy as np
import pytest

from causaldyn_bench.fold_design import arm_errors
from causaldyn_bench.paper_two import RatioCI, _quadrature_markdown, paired_ratio_ci


def test_pairing_is_what_makes_a_ratio_interval_narrow() -> None:
    """A split compared with ITSELF must return exactly 1, with no width.

    This is the whole reason the bootstrap resamples one index set and applies it to both arms.
    Resample the two arms independently and the same comparison returns an interval of finite
    width -- draw noise that cancels in the ratio, reported as uncertainty about the split. The
    designed-vs-random row of Table 1 is a near-tie, so that difference decides whether the table
    says "these tie" or "we cannot tell".
    """
    rng = np.random.default_rng(0)
    errors = rng.standard_normal(80)
    same = paired_ratio_ci(errors, errors, n_boot=2000, seed=1)
    assert same == RatioCI(1.0, 1.0, 1.0)

    # the unpaired comparison of the same array against itself, for contrast
    idx_a = rng.integers(0, errors.size, size=(2000, errors.size))
    idx_b = rng.integers(0, errors.size, size=(2000, errors.size))
    unpaired = (errors[idx_a] ** 2).mean(axis=1) / (errors[idx_b] ** 2).mean(axis=1)
    lo, hi = np.quantile(unpaired, [0.025, 0.975])
    assert hi - lo > 0.5


def test_the_interval_brackets_the_point_estimate_and_tracks_a_real_gap() -> None:
    rng = np.random.default_rng(1)
    base = rng.standard_normal(200)
    worse = base * 1.5
    ci = paired_ratio_ci(worse, base, n_boot=2000, seed=2)
    assert ci.lo <= ci.ratio <= ci.hi
    assert abs(ci.ratio - 2.25) < 1e-12  # a deterministic scaling: (1.5)^2, exactly
    assert ci.hi - ci.lo < 1e-9  # and no width, because the pairing removes the draw entirely
    assert ci.cell().startswith("2.250 [")


def test_mismatched_arms_are_refused_rather_than_broadcast() -> None:
    with pytest.raises(ValueError, match="paired arms need equal draws"):
        paired_ratio_ci(np.zeros(10), np.zeros(11))


def test_arms_are_paired_across_splits_in_the_measurement_itself() -> None:
    """``arm_errors`` must give draw ``i`` the same panel under every split, or the pairing above
    is comparing two different datasets. Two arms that differ only in the split must therefore
    disagree; two calls with the SAME split must agree bit for bit."""
    units = np.arange(12)
    blocks = np.arange(12) * 2 // 12
    first = arm_errors(None, units, False, 2, 3)
    again = arm_errors(None, units, False, 2, 3)
    other = arm_errors(None, blocks, False, 2, 3)
    assert first is not None and again is not None and other is not None
    for coefficient in ("direct", "spillover"):
        assert np.array_equal(first[coefficient], again[coefficient])
        assert not np.allclose(first[coefficient], other[coefficient])


def test_geometric_errors_are_the_one_case_where_residual_steps_recover_the_rate() -> None:
    """Errors ``e_k = 2^-k`` refine with residual ``e_{k-1} - e_k``, so residual / true is
    ``r - 1 = 1`` (Result 63 (e)), and every derived number the text quotes is known exactly: the
    rate and the residual step ratio both read 2 at every refinement, and six digits sit
    ``log2(e_9 * 1e6) = 10.93`` nodes past 9, so at 20 nodes."""
    errors = {k: 2.0**-k for k in range(4, 10)}
    steps = {k: 1.0 for k in errors if k - 1 in errors}
    four = {
        "rel_n5": errors,
        "rel_n7": errors,
        "residual_over_true_n5": steps,
        "residual_over_true_n7": steps,
    }
    text = "\n".join(_quadrature_markdown(four))
    assert "| 9 | 531,441 | 1.953e-03 | 2.000 | 1.00 | 1.953e-03 | 2.000 | 1.00 |" in text
    assert "geometric-mean rate 2.00 per node over nodes 4-9, and 2.71 digits at 9" in text
    assert "six digits at that rate need 20 nodes, 6.4e+07 points" in text
    assert "2.00 at 6, 2.00 at 7, 2.00 at 8, 2.00 at 9." in text


def test_a_single_grid_prints_its_row_and_derives_nothing() -> None:
    """The 64-bit recipe runs one grid; a rate needs two, so there is no rate to print."""
    four = {
        "rel_n5": {4: 0.29},
        "rel_n7": {4: 0.29},
        "residual_over_true_n5": {},
        "residual_over_true_n7": {},
    }
    text = "\n".join(_quadrature_markdown(four))
    assert "| 4 | 4,096 | 2.900e-01 | -- | -- | 2.900e-01 | -- | -- |" in text
    assert "geometric-mean" not in text

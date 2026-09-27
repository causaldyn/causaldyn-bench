"""P4's table generator: the counts a reader checks the claim against, on fabricated sweeps."""

import dataclasses

from chc.deep_galerkin import BlindnessSweep

from causaldyn_bench.paper_four import Configuration, summarise


def _sweep(rank_residual: float, rank_dual: float, discrepancy: float) -> BlindnessSweep:
    one = (1.0,)
    return BlindnessSweep(
        horizons=one,
        denominators=one,
        exact_slopes=one,
        fitted_slopes=one,
        errors=one,
        control_errors=one,
        residuals=one,
        conditioned_residuals=one,
        dual_weighted=one,
        rank_residual=rank_residual,
        rank_conditioned=0.0,
        rank_dual_weighted=rank_dual,
        worst_dual_discrepancy=discrepancy,
    )


def test_a_zero_rank_is_not_counted_as_the_wrong_sign() -> None:
    """``wrong_sign`` backs "ranks it backwards" and ``not_positive`` backs "cannot rank it"; a
    residual uncorrelated with the error supports the second claim and not the first."""
    cells = [
        Configuration("adam", 32, 0, _sweep(-0.5, 1.0, 0.01)),
        Configuration("adam", 32, 1, _sweep(0.0, 1.0, 0.02)),
        Configuration("lbfgs", 32, 0, _sweep(0.3, 0.9, 0.05)),
    ]
    summary = summarise(cells)
    assert (summary.wrong_sign, summary.not_positive, summary.dual_perfect) == (1, 2, 2)
    assert summary.worst_dual_discrepancy == 0.05


def test_groups_split_by_optimiser_and_width_and_spread_over_seeds() -> None:
    cells = [
        Configuration("adam", width, seed, _sweep(-0.1 * seed, 1.0, 0.0))
        for width in (64, 32)
        for seed in range(3)
    ]
    summary = summarise(cells)
    assert [(g.optimizer, g.width, g.seeds) for g in summary.groups] == [
        ("adam", 32, 3),
        ("adam", 64, 3),
    ]
    spread = summary.groups[0].ranks["rank_residual"]
    assert (spread.low, spread.middle, spread.high) == (-0.2, -0.1, 0.0)


def test_the_largest_fitted_slope_is_an_absolute_maximum_over_seeds_and_horizons() -> None:
    """Theorem 1's hypothesis bounds ``|S_hat(0)|``, so a negative slope counts by its size."""
    wide = dataclasses.replace(_sweep(-0.5, 1.0, 0.0), fitted_slopes=(2.0, -7.0))
    cells = [
        Configuration("lbfgs", 32, 0, wide),
        Configuration("lbfgs", 32, 1, _sweep(0.5, 1.0, 0.0)),
    ]
    assert summarise(cells).groups[0].largest_fitted_slope == 7.0

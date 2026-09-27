"""P3's table generator: the gates that make its tables checkable rather than transcribed."""

from itertools import pairwise

import numpy as np
import pytest
from chc.regret import capped_exploration_policy

from causaldyn_bench import paper_three
from causaldyn_bench.paper_three import (
    _schedules,
    exact_mass,
    explore_then_commit,
    plant_constants,
    result_56_mass,
    table_five,
)


def test_the_plant_constants_are_checked_against_the_library_not_copied_from_it() -> None:
    """``A``, ``K``, ``c`` are not exposed by any certificate, so this module reconstructs them --
    and a reconstruction nobody checks is a second copy that drifts. Both identities the public
    surface does expose must hold at machine zero, and the gate must actually fire."""
    constants = plant_constants()
    assert constants.c_causal_residual < 1e-12
    assert constants.floor_residual < 1e-9

    # the gate itself: move one knob and the reconstruction stops reproducing the library
    original = paper_three._B
    try:
        paper_three._B = 1.5
        with pytest.raises(RuntimeError, match="no longer reproduce"):
            plant_constants()
    finally:
        paper_three._B = original


def test_the_closed_form_root_solves_the_balance_the_library_iterates_to() -> None:
    """Table 4 compares the library's fixed point against an INDEPENDENT root, so the root has to
    be a root. Under a constant cap the stopping round is ``S/cap`` and the balance is a quadratic
    in ``w = I0 + c S``; the fixed point is the same equation discretised on integer rounds, so it
    may differ by the mass one round delivers and by no more than that."""
    k = plant_constants()
    for horizon in (10**4, 10**5, 10**6):
        for cap in (0.1, 0.03, 0.01):
            mass = exact_mass(horizon, cap, k)
            w = k.prior_info + k.info_rate * mass
            residual = (
                k.curvature * w * w
                + (k.numerator / cap) * w
                - k.numerator * k.info_rate * horizon
                - k.numerator * k.prior_info / cap
            )
            assert abs(residual) < 1e-8 * abs(k.numerator * k.info_rate * horizon)
            policy = capped_exploration_policy(horizon=horizon, cap=cap)
            assert abs(policy.predicted_mass - mass) < cap


def test_the_leading_form_over_states_the_mass_and_never_by_more_than_the_ceiling() -> None:
    """Result 66 (f), which is the content of Table 4: the gap is positive at every finite horizon
    and is bounded by ``K/(2 A c cap)``. A gap that crossed either way would make the table's
    "rises to the ceiling and stops" a coincidence of the grid."""
    k = plant_constants()
    for cap in (0.3, 0.1, 0.03, 0.01, 0.003):
        ceiling = k.numerator / (2.0 * k.curvature * k.info_rate * cap)
        gaps = [
            result_56_mass(horizon, k) - exact_mass(horizon, k=k, cap=cap)
            for horizon in (10**4, 10**5, 10**6, 10**7, 10**8)
        ]
        assert all(0.0 < gap < ceiling for gap in gaps)
        assert all(earlier < later for earlier, later in pairwise(gaps))


def test_the_schedules_in_table_three_really_do_differ_in_length() -> None:
    """Table 3's claim is that block lengths spanning nearly an order of magnitude all stop at the
    same mass. If the schedules produced similar blocks the table would be vacuous, so the spread
    is a property of the fixture and is asserted here rather than read off the output."""
    horizon = 4000
    lengths = [
        capped_exploration_policy(horizon=horizon, cap=[float(c) for c in caps]).block_rounds
        for caps in _schedules(horizon).values()
    ]
    assert max(lengths) / min(lengths) > 5.0


def test_the_aligned_factor_is_an_instance_and_the_bracket_is_the_result() -> None:
    """Table 5 reports a range because the headline factor is seed-dependent and the bracket is
    not. The certificate draws its 2x2 effect matrix from the seed, so how much the optimal action
    leans on the cut direction is a property of that draw; a single number read off one seed states
    the bracket's content and carries the draw's. Both halves are asserted here."""
    single = table_five((11,))["spread"]
    # one seed cannot tell an exact column from a lucky one, so it must decline to mark them
    assert all(values["seed_invariant"] is None for values in single.values())

    spread = table_five((11, 12, 13))["spread"]
    assert all(isinstance(values["seed_invariant"], bool) for values in spread.values())

    # the factor MOVES, and by more than either Monte-Carlo estimator column
    assert not spread["aligned_ratio"]["seed_invariant"]
    assert spread["aligned_ratio"]["span"] > 0.1
    assert spread["aligned_ratio"]["span"] > spread["plugin_ratio"]["span"]
    assert spread["aligned_ratio"]["span"] > spread["hodges_bayes_ratio"]["span"]

    # ... while the bracket, the exact zero and the closed form do not
    assert 1.0 < spread["aligned_ratio"]["lo"] <= spread["aligned_ratio"]["hi"] < 4.0
    for edge in ("lo", "hi"):
        assert np.isclose(
            spread["worst_single_direction"][edge], spread["aligned_ratio"][edge], rtol=1e-9
        )
    assert np.isclose(spread["orthogonal_ratio"]["lo"], 1.0, atol=1e-12)
    assert np.isclose(spread["orthogonal_ratio"]["hi"], 1.0, atol=1e-12)


def _ratio(horizon: int, rounds: int, probe: str, seed: int, reps: int = 50_000) -> float:
    k = plant_constants()
    c_causal = 2.0 * np.sqrt(k.curvature * k.numerator / k.info_rate)
    outcome = explore_then_commit(
        horizon, rounds, probe, paper_three._B, k, np.random.default_rng(seed), reps
    )
    return outcome.regret / (c_causal * np.sqrt(horizon))


def test_a_constant_magnitude_probe_attains_the_floor_with_a_real_estimator() -> None:
    """Tables 1-5 charge every round the van Trees floor, so "the constant is sharp" there means a
    SCHEDULE attains it. Table 6's claim is that a POLICY does: one probe of magnitude
    ``sqrt(M)``, least squares, commit. Its regret must approach ``c_causal sqrt(T)`` and grow at
    rate ``1/2``; a wrong budget, a random-magnitude probe or a mis-scaled estimate each break one
    of the two."""
    ratios = [_ratio(horizon, 1, "constant", seed) for seed, horizon in enumerate((10**5, 10**6))]
    assert abs(ratios[-1] - 1.0) < 0.03
    horizons = (10**4, 10**5, 10**6, 10**7)
    k = plant_constants()
    means = [
        explore_then_commit(
            t, 1, "constant", paper_three._B, k, np.random.default_rng(100 + i), 50_000
        ).regret
        for i, t in enumerate(horizons)
    ]
    slope = np.polyfit(np.log(horizons), np.log(means), 1)[0]
    assert abs(slope - 0.5) < 0.03


def test_gaussian_dither_of_the_same_budget_is_not_the_same_design() -> None:
    """The reduced objective sees a probe through its variance, least squares through its realised
    energy, and ``E[1/chi2_n] = 1/(n - 2)``. So from three rounds the price is the factor
    ``(n - 1)/(n - 2)``, and one round loses the RATE: the bounded ``u*`` clips an estimate whose
    variance is infinite, which leaves regret of order ``T^(3/4)``."""
    for rounds in (10, 30):
        assert (
            abs(_ratio(10**6, rounds, "gaussian", seed=rounds) - (rounds - 1) / (rounds - 2)) < 0.05
        )

    horizons = (10**3, 10**4, 10**5, 10**6)
    k = plant_constants()
    means = [
        explore_then_commit(
            t, 1, "gaussian", paper_three._B, k, np.random.default_rng(200 + i), 50_000
        ).regret
        for i, t in enumerate(horizons)
    ]
    slope = np.polyfit(np.log(horizons), np.log(means), 1)[0]
    assert 0.70 < slope < 0.80


def test_an_unknown_probe_is_refused() -> None:
    with pytest.raises(ValueError, match="probe must be"):
        explore_then_commit(
            10**3, 1, "uniform", 1.0, plant_constants(), np.random.default_rng(0), 10
        )


def test_the_budget_is_where_the_commit_phase_costs_what_the_probe_did() -> None:
    """The regret is flat in the budget at its optimum -- ten per cent too much costs half a per
    cent -- so the ratio to the floor cannot pin the budget. Validation STEP 10's identity can: at
    ``M = sqrt(K (T - n)/(A c))`` the committed rounds cost exactly what the probe did. The plant's
    bounded ``u*`` shaves the commit phase slightly, so the balance holds to a few per cent."""
    k = plant_constants()
    outcome = explore_then_commit(
        10**6, 1, "constant", paper_three._B, k, np.random.default_rng(7), 50_000
    )
    assert abs(outcome.commit_cost / outcome.probe_cost - 1.0) < 0.06

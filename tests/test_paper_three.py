"""P3's table generator: the gates that make its tables checkable rather than transcribed."""

from itertools import pairwise, product

import numpy as np
import pytest
from chc.regret import capped_exploration_policy
from scipy.optimize import minimize

from causaldyn_bench import paper_three
from causaldyn_bench.paper_three import (
    PlantConstants,
    _schedules,
    explore_then_commit,
    leading_mass,
    plant_constants,
    rounding,
    smooth_mass,
    table_five,
    table_four,
)

# A = b^2 + rr = 1, K = A (du*/db)^2 = 4 with du*/db = -xt (rr - b^2)/(rr + b^2)^2 = -2,
# c = eta/sigma^2 = 1, I0 = 1: a plant where one round at a loose cap outweighs the prior, so the
# relaxation's error shows at T = 3
UNIT_KNOBS = {"b": 0.5, "rr": 0.75, "xt": 4.0, "sigma": 1.0, "eta": 1.0, "i0": 1.0}
UNIT = PlantConstants(1.0, 4.0, 1.0, 1.0, 0.0, 0.0)


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


def test_the_smooth_root_solves_the_relaxed_balance_and_leads_the_optimum_by_under_a_cap() -> None:
    """Table 4 sets the library's optimum beside an INDEPENDENT root, so the root has to be a root:
    of ``A w^2 + (K/cap) w = K c T + K I0/cap``, the balance with the stopping round read as
    ``S/cap``. The optimum charges the integer round instead, so it is that root at a horizon
    shorter by ``rho_T`` in ``[0, 1]`` and lies below it by less than one cap -- on a round
    boundary too, where the optimum is a whole block (``cap = 0.001`` at ``T = 4000``)."""
    k = plant_constants()
    cases = [(h, cap) for h in (10**4, 10**5, 10**6) for cap in (0.1, 0.03, 0.01)]
    cases += [(4000, 10 ** (-j / 2)) for j in range(1, 7)]
    # short horizons, from the first at which exploring can pay, A I0^2 <= K c T
    cases += [(h, cap) for h in range(17, 400, 13) for cap in (0.3, 0.1, 0.03, 0.01, 0.003)]
    boundaries = 0
    for horizon, cap in cases:
        smooth = smooth_mass(horizon, cap, k)
        w = k.prior_info + k.info_rate * smooth
        residual = (
            k.curvature * w * w
            + (k.numerator / cap) * w
            - k.numerator * k.info_rate * horizon
            - k.numerator * k.prior_info / cap
        )
        assert abs(residual) < 1e-8 * abs(k.numerator * k.info_rate * horizon)
        policy = capped_exploration_policy(horizon=horizon, cap=cap)
        exact = policy.exploration_mass
        if exact > 0.0 and policy.last_round == cap:
            boundaries += 1
        rho = rounding(horizon, cap, exact, k)
        assert -1e-9 <= rho <= 1.0 + 1e-9
        assert smooth_mass(horizon - rho, cap, k) == pytest.approx(exact, rel=1e-10, abs=1e-12)
        assert 0.0 <= smooth - exact < cap
    assert boundaries > 0


def test_the_leading_form_over_states_both_masses_and_never_by_more_than_the_ceiling() -> None:
    """The content of Table 4: the leading form's gap to the smooth mass is positive at every
    finite horizon, rises, and is bounded by ``K/(2 A c cap)``. A gap that crossed either way would
    make the table's "rises to the ceiling and stops" a coincidence of the grid. The exact mass
    lies lower still, and under the same ceiling wherever ``c cap <= I0``, which all these caps
    meet."""
    k = plant_constants()
    for cap in (0.3, 0.1, 0.03, 0.01, 0.003):
        ceiling = k.numerator / (2.0 * k.curvature * k.info_rate * cap)
        gaps = [
            leading_mass(horizon, k) - smooth_mass(horizon, k=k, cap=cap)
            for horizon in (10**4, 10**5, 10**6, 10**7, 10**8)
        ]
        assert all(0.0 < gap < ceiling for gap in gaps)
        assert all(earlier < later for earlier, later in pairwise(gaps))

        assert k.info_rate * cap <= k.prior_info
        for horizon in (10**4, 10**5, 10**6):
            exact = capped_exploration_policy(horizon=horizon, cap=cap).exploration_mass
            assert leading_mass(horizon, k) - smooth_mass(horizon, cap, k) < (
                leading_mass(horizon, k) - exact
            )
            assert leading_mass(horizon, k) - exact < ceiling


def _regret(v: np.ndarray, k: PlantConstants) -> np.ndarray:
    """``A sum_t v_t + K sum_t 1/(I0 + c S_(t-1))`` for each row of ``v``."""
    before = np.cumsum(v, axis=-1) - v
    information = k.prior_info + k.info_rate * before
    return k.curvature * v.sum(axis=-1) + (k.numerator / information).sum(axis=-1)


def _regret_gradient(v: np.ndarray, k: PlantConstants) -> np.ndarray:
    # round t's weight; v_j moves the information of every round after j
    weights = k.numerator / (k.prior_info + k.info_rate * (np.cumsum(v) - v)) ** 2
    later = np.concatenate([np.cumsum(weights[::-1])[::-1][1:], [0.0]])
    return k.curvature - k.info_rate * later


def _box_minimum(horizon: int, cap: float, k: PlantConstants) -> tuple[float, float]:
    """The least regret over ``0 <= v_t <= cap`` and the mass that attains it, by brute force.

    Up to four rounds every point of a grid of nine levels a round is priced; past that the corners
    and random points of the box stand in. The best starts are polished by a bounded quasi-Newton
    descent on the full program, which assumes neither a greedy fill nor a stopping round.
    """
    if horizon <= 4:
        starts = np.array(list(product(np.linspace(0.0, cap, 9), repeat=horizon)))
    else:
        rng = np.random.default_rng(horizon)
        starts = np.vstack(
            [np.zeros(horizon), np.full(horizon, cap), rng.uniform(0.0, cap, (8, horizon))]
        )
    priced = _regret(starts, k)
    polished = [
        minimize(
            lambda v: float(_regret(v, k)),
            start,
            jac=lambda v: _regret_gradient(v, k),
            method="L-BFGS-B",
            bounds=[(0.0, cap)] * horizon,
            options={"ftol": 1e-15, "gtol": 1e-13, "maxiter": 10_000},
        )
        for start in starts[np.argsort(priced)[:5]]
    ]
    best = min(polished, key=lambda result: result.fun)
    assert best.fun <= priced.min()
    return float(best.fun), float(np.sum(best.x))


@pytest.mark.parametrize(
    ("knobs", "k", "horizons", "caps"),
    [
        (UNIT_KNOBS, UNIT, (2, 3, 4), (0.05, 0.3, 0.8, 1.5, 4.0, 100.0)),
        ({}, None, (40, 200), (0.01, 0.03, 0.1, 0.3)),
    ],
    ids=["unit plant", "the tables' plant"],
)
def test_the_exact_mass_is_the_minimum_over_the_box(
    knobs: dict[str, float],
    k: PlantConstants | None,
    horizons: tuple[int, ...],
    caps: tuple[float, ...],
) -> None:
    """Table 4's exact column is the library's ``exploration_mass``, so it is checked against the
    program it claims to solve: no schedule in the box costs less than the library's optimum, the
    brute force reaches it, and at the mass the library names."""
    k = k or plant_constants()
    for horizon in horizons:
        for cap in caps:
            policy = capped_exploration_policy(horizon=horizon, cap=cap, **knobs)
            least, mass = _box_minimum(horizon, cap, k)
            assert policy.cost <= least * (1.0 + 1e-12)
            assert policy.cost == pytest.approx(least, rel=1e-12, abs=0.0)
            assert policy.exploration_mass == pytest.approx(mass, abs=1e-6)


def test_the_relaxation_misses_the_optimum_where_one_round_outweighs_the_prior() -> None:
    """At ``A = 1``, ``K = 4``, ``c = 1``, ``I0 = 1``, ``T = 3`` and ``cap = 100`` one round at the
    cap carries a hundred times the prior's information. The optimum is one partial round of
    ``sqrt(8) - 1``; the smooth root reads ``2.449928`` and the leading form ``sqrt(12) - 1``, which
    over-states the optimum by ``0.6357`` against ``K/(2 A c cap) = 0.02``: the ceiling is the
    relaxation's, not the optimum's. At ``cap = 4/5`` the optimum is the schedule
    ``(4/5, 1/5, 0)`` at cost ``83/9``, and the smooth root reaches its mass at ``T = 9/4``."""
    loose = capped_exploration_policy(horizon=3, cap=100.0, **UNIT_KNOBS)
    assert loose.exploration_mass == pytest.approx(np.sqrt(8.0) - 1.0, rel=1e-14, abs=0.0)
    assert loose.cost == pytest.approx(3.0 + 4.0 * np.sqrt(2.0), rel=1e-14, abs=0.0)
    assert smooth_mass(3, 100.0, UNIT) == pytest.approx(2.449928, abs=5e-7)
    over = leading_mass(3, UNIT) - loose.exploration_mass
    assert leading_mass(3, UNIT) == pytest.approx(np.sqrt(12.0) - 1.0, rel=1e-14, abs=0.0)
    assert over == pytest.approx(0.6357, abs=5e-5)
    assert UNIT.numerator / (2.0 * UNIT.curvature * UNIT.info_rate * 100.0) == 0.02
    rho = rounding(3, 100.0, loose.exploration_mass, UNIT)
    assert 0.0 <= rho <= 1.0
    assert smooth_mass(3 - rho, 100.0, UNIT) == pytest.approx(
        loose.exploration_mass, rel=1e-12, abs=0.0
    )

    tight = capped_exploration_policy(horizon=3, cap=0.8, **UNIT_KNOBS)
    assert tight.block_rounds == 2
    assert tight.last_round == pytest.approx(0.2, rel=1e-14, abs=0.0)
    assert tight.exploration_mass == pytest.approx(1.0, rel=1e-14, abs=0.0)
    assert tight.cost == pytest.approx(83.0 / 9.0, rel=1e-14, abs=0.0)
    assert rounding(3, 0.8, tight.exploration_mass, UNIT) == pytest.approx(0.75, abs=1e-14)
    assert smooth_mass(2.25, 0.8, UNIT) == pytest.approx(1.0, rel=1e-14, abs=0.0)


def _prints_as(value: float, text: str) -> bool:
    """Whether ``value`` rounds to ``text`` at the digits ``text`` shows."""
    mantissa, _, exponent = text.partition("e")
    decimals = len(mantissa.partition(".")[2])
    if exponent:
        return f"{value:.{decimals}e}" == f"{float(text):.{decimals}e}"
    return f"{value:.{decimals}f}" == text


def test_table_four_prints_the_papers_digits() -> None:
    """Every cell of Table 4 is an exact function of the model, so the digits the paper prints are
    pinned: its constant-cap mass table, its ``T = 4000`` cap ladder and its ``C_cap``.
    ``T = 1e7`` and ``1e8`` are left to ``scripts/paper_three_mass.gp``, which reaches every digit
    at 77: the library builds arrays of length ``T``."""
    four = table_four(
        (10**4, 10**5, 10**6),
        (0.1, 0.03, 0.01),
        tuple(10 ** (-j / 2) for j in range(1, 7)),
        4000,
        plant_constants(),
    )
    assert _prints_as(four["ceiling"], "0.672154")
    assert _prints_as(four["coefficient"], "3.85826")
    assert _prints_as(four["band"][0], "-3.8583")
    assert _prints_as(four["band"][1], "-3.7578")

    columns = (
        "smooth",
        "smooth_minus_exact",
        "rounding",
        "smooth_over_ceiling",
        "exact_over_ceiling",
        "smooth_residual",
        "exact_residual",
    )
    printed = {
        10**4: ("18.6319", "9.7e-4", "0.970", "0.942654", "0.944100", "-3.8546", "-3.7573"),
        10**5: ("62.0287", "1.2e-4", "0.382", "0.981850", "0.982030", "-3.8579", "-3.8196"),
        10**6: ("199.3364", "4.6e-5", "0.455", "0.994260", "0.994328", "-3.8582", "-3.8125"),
    }
    for row in four["ladder"]:
        for column, text in zip(columns, printed[int(row["horizon"])], strict=True):
            assert _prints_as(row[column], text), (row["horizon"], column, row[column], text)

    shares = ("0.51", "1.60", "5.15", "17.05", "61.03", "236.6")
    for row, text in zip(four["by_cap"], shares, strict=True):
        assert _prints_as(100.0 * row["relative"], text), (row["cap"], row["relative"])
        assert abs(row["relative"] - row["smooth_relative"]) < 3e-4

    for row in four["agreement"]:
        assert row["fixed_point"] == pytest.approx(row["optimum"], rel=1e-12, abs=0.0)
        assert 0.0 <= row["smooth_lead_in_caps"] < 1.0

    constants = {
        0.1: ("-2.640763", "0.020286", "2.005734"),
        0.03: ("-6.032919", "0.049435", "4.675019"),
        0.01: ("-15.723437", "-0.033778", "-5.921131"),
    }
    for row in four["constant"]:
        constant, residual, slope = constants[row["cap"]]
        assert _prints_as(row["constant"], constant)
        assert _prints_as(row["scaled_residuals"][0] / 100.0, residual)
        assert _prints_as(row["slope"], slope)


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

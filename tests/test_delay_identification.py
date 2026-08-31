"""Track K: the confounder relocates the delay's peak, and the payoff is a bifurcation."""

import numpy as np
import pytest
from chc.delay import STABILISING_RATIO_FLOOR, delay_margin
from chc.irf import delay_estimate

from causaldyn_bench.delay_identification import (
    _CHANNEL,
    _CONFOUND,
    _DT,
    _ETA,
    _EVERY,
    _KAPPA,
    _TAU_U,
    _TAU_Z,
    _best_gain,
    _confounded_delay_log,
    crosscorrelation_delay,
    track_delay_identification,
)


def _by_method(results: list, track: str) -> dict[str, float]:
    return {r.method: r.value for r in results if r.track == track}


def test_the_confounder_moves_the_peak_not_just_the_amplitude() -> None:
    """The failure this track exists for: the estimate is wrong about *when*, not how much."""
    results = track_delay_identification(seed=0)
    delay = _by_method(results, "K-delay")
    # the unadjusted peak lands on the CONFOUNDER's lag, a whole observation sample early
    assert delay["cross-correlation"] == pytest.approx(abs(_TAU_Z - _TAU_U), abs=1e-9)
    assert delay["adjusted-LP"] < 0.5 * delay["cross-correlation"]
    assert delay["adjusted-LP-refined"] < delay["adjusted-LP"]


def test_the_classical_baseline_is_not_a_strawman() -> None:
    """Cross-correlation and an *unadjusted* local projection are the same argmax, so the board's
    gap is adjustment, not the estimator family."""
    data = _confounded_delay_log(0)
    unadjusted = delay_estimate(
        data, horizon=12, dt=_DT * _EVERY, adjust_for=(), refine=False, seed=0
    ).delay
    assert crosscorrelation_delay(data) == pytest.approx(unadjusted, abs=1e-9)


def test_ignoring_the_delay_bifurcates_while_mis_estimating_it_only_costs() -> None:
    """Three tiers, and the middle one is the point: a bad estimate still survives."""
    payoff = _by_method(track_delay_identification(seed=0), "K-payoff")
    assert payoff["oracle"] == 0.0
    assert payoff["delay-blind"] > 1e3  # past pi/2 -- a divergence, not a gap
    assert payoff["cross-correlation"] < 1.0  # ...but a 40% delay error still stabilises
    assert payoff["adjusted-LP-refined"] < 0.05 * payoff["cross-correlation"]


def test_the_survival_of_the_biased_arm_is_the_half_line_not_luck() -> None:
    """`chc.delay` says under-estimating is survivable down to 2/(pi e); this arm is well inside."""
    assert _TAU_Z / _TAU_U > STABILISING_RATIO_FLOOR
    boundary_gain = np.pi / (2.0 * _CHANNEL * _TAU_U)
    assert delay_margin(0.0, _CHANNEL * boundary_gain) == pytest.approx(_TAU_U, rel=1e-12)
    assert _best_gain(_TAU_Z) * _CHANNEL * _TAU_U < boundary_gain  # the biased design stabilises
    assert _best_gain(0.0) * _CHANNEL * _TAU_U > boundary_gain  # the blind one does not


def test_the_regret_is_convex_in_the_delay_error_not_proportional_to_it() -> None:
    """A 4x smaller delay error buys far more than 4x less regret -- why adjustment pays."""
    results = track_delay_identification(seed=0)
    delay, payoff = _by_method(results, "K-delay"), _by_method(results, "K-payoff")
    delay_ratio = delay["cross-correlation"] / delay["adjusted-LP"]
    regret_ratio = payoff["cross-correlation"] / payoff["adjusted-LP"]
    assert regret_ratio > 4.0 * delay_ratio


def test_enough_exploration_restores_the_delay_without_any_adjustment() -> None:
    """The bias is a property of a thin log, not of cross-correlation -- and the threshold is exact.

    The incentive's window straddles a block boundary and splits 2:1 across lags 3 and 4 while the
    confounder's lands wholly in lag 2, so the peak relocates iff the exploration variance falls
    below ``1.5|c*kappa|/|b| - kappa**2``.
    """
    threshold = float(np.sqrt(1.5 * abs(_CONFOUND * _KAPPA) - _KAPPA**2))
    assert threshold > _ETA  # the shipped log sits deliberately below it

    def unadjusted(eta: float, seed: int) -> float:
        data = _confounded_delay_log(seed, eta=eta)
        return delay_estimate(
            data, horizon=12, dt=_DT * _EVERY, adjust_for=(), refine=False, seed=seed
        ).delay

    assert all(unadjusted(0.95 * threshold, seed) == pytest.approx(_TAU_Z) for seed in range(3))
    assert all(unadjusted(1.15 * threshold, seed) > 0.75 * _TAU_U for seed in range(3))

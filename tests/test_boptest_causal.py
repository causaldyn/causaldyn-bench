"""Track D-causal: the parts that need no emulator, plus one live 2x2 when BOPTEST_URL is set.

Every offline test here is constructed from a *synthetic* log with a known channel, so it checks the
harness rather than the building: unit conversion, the two-stage fit, the drift's weather term, and
the overlap accounting that decides whether anything is identified at all.
"""

import itertools
import os
import re

import jax.numpy as jnp
import numpy as np
import pytest

from causaldyn_bench.boptest import is_available
from causaldyn_bench.boptest_causal import (
    CASES,
    HEAT_PUMP,
    BOPTestClient,
    LoggedEpisode,
    ThermalFit,
    _mpc_solver,
    finite_difference_step_response,
    fit_neural,
    fit_thermal,
    log_episode,
    overlap_report,
    predictive_comparison,
    track_boptest_causal,
)

_URL = os.environ.get("BOPTEST_URL", "")  # "" rather than None: the live test needs a `str`


def _synthetic_log(
    *,
    weather_gain: float,
    channel: float = 1.2,
    pole: float = -0.05,
    n: int = 800,
    seed: int = 0,
) -> LoggedEpisode:
    """A confounded log with a KNOWN channel: outdoor temperature drives both action and rate.

    ``weather_gain`` is the outdoor-reset slope. At 0 the action is exogenous and OLS is
    unbiased; at a positive value the naive fit must be biased and adjustment must repair it.
    """
    rng = np.random.default_rng(seed)
    outdoor = 5.0 + 8.0 * np.sin(np.arange(n) / 24.0) + rng.normal(0.0, 1.0, n)
    zone = np.empty(n)
    action = np.empty(n)
    zone_next = np.empty(n)
    temp = 21.0
    dt = 0.5
    for i in range(n):
        raw = 0.3 + weather_gain * (5.0 - outdoor[i]) + rng.normal(0.0, 0.15)
        action[i] = float(np.clip(raw, 0.0, 1.0))
        zone[i] = temp
        rate = pole * temp + 0.9 + 0.04 * (outdoor[i] - 5.0) + channel * action[i]
        temp = temp + dt * rate + rng.normal(0.0, 0.01)
        zone_next[i] = temp
    covariates = np.column_stack([outdoor, np.zeros(n), np.full(n, 21.0)])
    centre, scale = (
        covariates.mean(axis=0),
        np.where(covariates.std(axis=0) > 1e-9, covariates.std(axis=0), 1.0),
    )
    return LoggedEpisode(
        zone=jnp.asarray(zone).reshape(-1, 1),
        action=jnp.asarray(action).reshape(-1, 1),
        zone_next=jnp.asarray(zone_next).reshape(-1, 1),
        weather=jnp.asarray((covariates - centre) / scale),
        weather_mean=tuple(centre),
        weather_scale=tuple(scale),
        dt_hours=dt,
        at_bound=0.0,
        policy="reset",
    )


def test_adjustment_repairs_a_confounded_channel_and_leaves_an_exogenous_one_alone() -> None:
    """The 2x2 in miniature: both directions, so a distorting estimator would fail too."""
    confounded = _synthetic_log(weather_gain=0.05)
    exogenous = _synthetic_log(weather_gain=0.0)

    naive_bias = abs(fit_thermal(confounded, adjusted=False).channel[0] - 1.2)
    adjusted_bias = abs(fit_thermal(confounded, adjusted=True).channel[0] - 1.2)
    assert adjusted_bias < 0.5 * naive_bias  # de-confounding must actually pay

    before = fit_thermal(exogenous, adjusted=False).channel[0]
    after = fit_thermal(exogenous, adjusted=True).channel[0]
    assert (
        abs(after - before) < 0.35 * naive_bias
    )  # and must not move a design that was already fine


def test_the_drift_recovers_a_stable_pole_only_with_the_weather_term() -> None:
    """The finding that forced the two-stage fit: a trending confounder buys a positive pole."""
    log = _synthetic_log(weather_gain=0.05)
    fit = fit_thermal(log, adjusted=True)
    assert fit.stable
    assert fit.drift[1] < -0.01  # decays on the timescale the data can see
    assert abs(fit.weather_drift[0]) > 0.05  # the outdoor column is load-bearing, not decoration


def test_step_response_is_reported_where_the_dc_gain_is_not_identified() -> None:
    """A near-unit-root fit must not turn into a giant steady-state gain via the reported metric."""
    unstable = ThermalFit(
        drift=(0.0, -3e-4),
        weather_drift=(0.0, 0.0, 0.0),
        channel=(1.0, 0.0),
        method="observational",
        identified=False,
        adjusted=False,
        policy="prbs",
        action_residual_variance=0.02,
        nuisance_r2_action=0.0,
        channel_error=None,
    )
    assert unstable.step_response(hours=8.0) < 9.0  # -b/a would be 3333
    assert unstable.stable  # nominally stable, and still useless as a steady-state gain


def test_overlap_collapses_when_the_logging_policy_is_deterministic() -> None:
    """No exploration noise, no identification -- the assumption the whole track rests on."""
    rng = np.random.default_rng(0)
    n = 400
    outdoor = 5.0 + 8.0 * np.sin(np.arange(n) / 24.0)
    action = np.clip(0.3 + 0.05 * (5.0 - outdoor), 0.0, 1.0)  # an exact function of the covariate
    zone = 21.0 + rng.normal(0.0, 0.2, n)
    covariates = np.column_stack([outdoor, np.zeros(n), np.full(n, 21.0)])
    scale = np.where(covariates.std(axis=0) > 1e-9, covariates.std(axis=0), 1.0)
    log = LoggedEpisode(
        zone=jnp.asarray(zone).reshape(-1, 1),
        action=jnp.asarray(action).reshape(-1, 1),
        zone_next=jnp.asarray(zone + 0.1).reshape(-1, 1),
        weather=jnp.asarray((covariates - covariates.mean(axis=0)) / scale),
        weather_mean=tuple(covariates.mean(axis=0)),
        weather_scale=tuple(scale),
        dt_hours=0.5,
        at_bound=0.0,
        policy="reset",
    )
    assert overlap_report(log)["residual_share"] < 0.02


def test_every_case_maps_its_action_into_the_actuator_range() -> None:
    for name, case in CASES.items():
        lo = case.to_raw(case.action_lo - 100.0)
        hi = case.to_raw(case.action_hi + 100.0)
        assert lo == pytest.approx(case.action_lo + case.action_offset), name
        assert hi == pytest.approx(case.action_hi + case.action_offset), name
        assert case.overwrite(case.action_lo)[f"{case.action_point[:-2]}_activate"] == 1, name


def test_the_exploration_band_sits_inside_the_actuator_box() -> None:
    """The randomised arm excites around the operating point; outside the box it gets clipped."""
    for name, case in CASES.items():
        assert case.action_lo <= case.explore_lo < case.explore_hi <= case.action_hi, name


def test_pessimism_shrinks_the_channel_toward_a_weaker_plant() -> None:
    """The sign is the claim: under a comfort constraint, believing the plant strong costs most."""
    log = _synthetic_log(weather_gain=0.05)
    fit = fit_thermal(log, adjusted=True)

    shrunk = fit.pessimistic(0.1)
    assert shrunk.channel[0] == pytest.approx(fit.channel[0] - 0.1)
    assert shrunk.channel[1] == fit.channel[1]  # only the gain moves, not the interaction
    assert shrunk.step_response() < fit.step_response()  # less believed authority, always
    assert fit.pessimistic(0.0).channel == fit.channel  # zero shrink is exactly the point estimate

    with pytest.raises(ValueError, match="cannot be negative"):
        fit.pessimistic(-1.0)


def test_the_black_box_matches_the_structured_fit_on_prediction_and_misses_the_channel() -> None:
    """The physics-off ablation, and the reason a rollout-error leaderboard cannot see it.

    Two-sided on purpose. The upper bound on held-out error earns the comparison -- a black box
    that predicted *worse* would be a straw man and its channel error would prove nothing. The
    lower bound on the channel is the finding: the same forecast, a different derivative.

    Aggregated over seeds rather than asserted per seed. Both arms' per-seed authority errors are
    the same order as their spreads, so a single-seed ratio is a coin flip dressed as a threshold;
    the RMSE over several seeds is the quantity that separates them. Measured over six seeds at
    ``n=1600``: authority RMSE 0.0469 against 0.0179 about a truth of 1.200 (bias 3.4% against
    0.07%), with held-out MSE 4.20e-4 against 4.13e-4 -- under half a seed standard deviation, both
    sitting on the 4.0e-4 noise floor.
    """
    errors = {"affine-adjusted": [], "neural": []}
    scores = {"affine-adjusted": [], "neural": []}
    for seed in (0, 1, 2):
        log = _synthetic_log(weather_gain=0.05, n=1200, seed=seed)
        predicted = predictive_comparison(log, seed=seed)
        fits = {
            "affine-adjusted": fit_thermal(log, adjusted=True),
            "neural": fit_neural(log, seed=seed),
        }
        for arm, fit in fits.items():
            errors[arm].append(fit.authority() - 1.2)
            scores[arm].append(predicted[arm])

    def rmse(arm: str) -> float:
        return float(np.sqrt(np.mean(np.square(errors[arm]))))

    assert np.mean(scores["neural"]) < 1.25 * np.mean(scores["affine-adjusted"])
    assert rmse("neural") > 1.5 * rmse("affine-adjusted")


def test_omitting_the_confounder_is_the_one_failure_prediction_does_catch() -> None:
    """The other side of the same claim, so it does not read as "prediction is useless".

    The naive affine arm leaves the confounder out of the channel stage entirely, and that shows up
    as an order-of-magnitude worse forecast *as well as* a badly attenuated channel. Held-out error
    catches omitted variables; what it cannot catch is a model that conditions on everything and
    still misattributes the derivative. Only the second failure needs a moment condition.
    """
    log = _synthetic_log(weather_gain=0.05, n=1200, seed=0)
    scores = predictive_comparison(log)

    assert scores["affine-naive"] > 5.0 * scores["affine-adjusted"]
    naive, adjusted = fit_thermal(log, adjusted=False), fit_thermal(log, adjusted=True)
    assert naive.authority() < 0.5 * adjusted.authority()  # the truth is +1.2


def test_both_arms_are_scored_on_one_step_response_definition() -> None:
    """The generic finite difference must reproduce the closed form, or the ablation is a metric."""
    log = _synthetic_log(weather_gain=0.05, n=800, seed=0)
    fit = fit_thermal(log, adjusted=True)
    assert finite_difference_step_response(fit) == pytest.approx(fit.step_response(), rel=1e-5)


def test_the_black_box_reports_the_pole_and_authority_the_structured_arm_does() -> None:
    """Substitutability is what lets one MPC serve both arms; check it on the accessors too."""
    log = _synthetic_log(weather_gain=0.0, n=800, seed=0)  # exogenous action: both should be close
    structured = fit_thermal(log, adjusted=True)
    black_box = fit_neural(log, seed=0)

    assert black_box.pole < 0.0  # a building that returns to ambient, not one that runs away
    assert abs(black_box.pole - structured.pole) < 0.5 * abs(structured.pole)
    assert abs(black_box.authority() - structured.authority()) < 0.5  # nothing left to differ on


def test_the_affine_authority_is_the_channel_at_the_operating_point_not_its_intercept() -> None:
    """``b0`` is the channel extrapolated to 0 C, which is why it is not the reported estimand."""
    fit = ThermalFit(
        drift=(0.0, -0.05),
        weather_drift=(0.0, 0.0, 0.0),
        channel=(0.4, 0.03),
        method="orthogonal",
        identified=True,
        adjusted=True,
        policy="reset",
        action_residual_variance=0.02,
        nuisance_r2_action=0.85,
        channel_error=0.1,
    )
    assert fit.authority(21.0) == pytest.approx(0.4 + 0.03 * 21.0)
    assert fit.authority(21.0) > 2.5 * fit.channel[0]  # 21 K of extrapolation is not a detail


def _plant(authority: float, pole: float = -0.05) -> ThermalFit:
    """A temperature-independent channel, so the MPC optimum is easy to reason about."""
    return ThermalFit(
        drift=(-pole * 21.0, pole),
        weather_drift=(0.0, 0.0, 0.0),
        channel=(authority, 0.0),
        method="ols",
        identified=True,
        adjusted=True,
        policy="reset",
        action_residual_variance=0.02,
        nuisance_r2_action=0.0,
        channel_error=0.0,
    )


@pytest.mark.parametrize("authority", [0.2, 0.5, 1.0])
def test_the_commanded_action_falls_as_the_room_starts_warmer(authority: float) -> None:
    """A warmer start needs no more heat, at any believed authority.

    This is the property a constant step size broke. The comfort curvature scales with
    ``w_comfort * (dt * authority)**2``, so the largest stable step is a function of the authority
    the *model* believes in, and one shared ``lr`` therefore sits further past the stability limit
    for the arms that identify a larger channel. The iterate limit-cycled instead of converging, and
    the commanded action came back non-monotone in the starting temperature -- 0.430 at 23.5 C
    against 0.704 at 24.0 C on the heat-pump log, which no optimum of this objective can be.

    The grid walks the *interior*, between the temperature where the plan stops saturating and the
    one where the shortfall vanishes, because that is the only band where the command is a decision
    rather than a bound. Checked against the old solver: it reads 0.229 then 0.580 at authority 0.5
    and 0.246 then 0.729 at authority 1.0, and is monotone at 0.2 -- the one arm whose curvature
    kept it inside the old step's stability limit. It therefore discriminates on two of three cases,
    and the parametrisation is what keeps that visible instead of averaging it away.
    """
    solve = _mpc_solver(
        _plant(authority), HEAT_PUMP, 0.5, w_comfort=800.0, margin=1.5, iterations=600
    )
    bounds, covariates = jnp.full(16, 21.0), jnp.zeros((16, 3))
    grid = (22.2, 22.4, 22.6, 22.8, 23.0, 23.2, 23.4)
    commanded = [solve(temp, bounds, covariates) for temp in grid]
    assert all(a >= b - 1e-6 for a, b in itertools.pairwise(commanded)), commanded


def test_the_commanded_action_stops_moving_once_the_budget_is_spent() -> None:
    """A command that keeps moving with the budget is a transient, not a decision.

    The old constant step never reached a fixed point: it clipped to the actuator limit, decayed by
    ``1 - 2*lr`` per iteration until the predicted temperature recrossed the comfort hinge, and
    clipped again -- a period-8 cycle. The commanded action was therefore a function of where that
    cycle happened to be at the last iteration, reading 0.729 / 0.900 / 0.590 / 0.674 at budgets of
    60 / 240 / 1000 / 4000 on the same input -- a span of 0.31 on a [0, 1] actuator.

    The tolerance is the measured tail, not an aspiration. At the harness default the command still
    moves by up to 0.012 (worst at authority 1.0, whose curvature is largest and whose step is
    therefore smallest); by 3000 iterations it is settled to 5e-4. So the default buys roughly 1% of
    actuator range, which is the accuracy the closed-loop numbers should be read at, and this test
    fails if the default is ever moved back below convergence.
    """
    bounds, covariates = jnp.full(16, 21.0), jnp.zeros((16, 3))
    for authority in (0.2, 0.5, 1.0):
        commanded = [
            _mpc_solver(
                _plant(authority), HEAT_PUMP, 0.5, w_comfort=800.0, margin=1.5, iterations=budget
            )(23.0, bounds, covariates)
            for budget in (600, 3000)
        ]
        assert commanded[0] == pytest.approx(commanded[1], abs=0.02), (authority, commanded)


def test_the_track_refuses_to_run_without_an_emulator_rather_than_scoring_nothing() -> None:
    """A silently skipped track is indistinguishable from a track that found nothing."""
    with pytest.raises(RuntimeError, match=re.escape("127.0.0.1:1")):
        track_boptest_causal("http://127.0.0.1:1")


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        log_episode(
            BOPTestClient("http://127.0.0.1:1"), HEAT_PUMP, policy="greedy", seed=0, steps=1
        )


@pytest.mark.skipif(not _URL, reason="set BOPTEST_URL to a running BOPTEST-Service")
def test_live_reset_log_is_confounded_and_adjustable() -> None:
    if not is_available(_URL):
        pytest.skip("BOPTEST_URL is set but the service is unreachable")
    client = BOPTestClient(_URL)
    log = log_episode(client, HEAT_PUMP, policy="reset", seed=0, steps=96)
    assert log.zone.shape == log.action.shape == (96, 1)
    assert log.weather.shape == (96, 3)
    assert 0.0 <= log.at_bound <= 1.0
    assert fit_thermal(log, adjusted=True).method == "orthogonal"
    assert fit_thermal(log, adjusted=False).method == "observational"

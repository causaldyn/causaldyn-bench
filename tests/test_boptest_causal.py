"""Track D-causal: the parts that need no emulator, plus one live 2x2 when BOPTEST_URL is set.

Every offline test here is constructed from a *synthetic* log with a known channel, so it checks the
harness rather than the building: unit conversion, the two-stage fit, the drift's weather term, and
the overlap accounting that decides whether anything is identified at all.
"""

import dataclasses
import itertools
import math
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
    HorizonPlan,
    LoggedEpisode,
    RunawayDriftError,
    SafetyAudit,
    StepAudit,
    ThermalFit,
    _audit_plan,
    _horizon_dynamics,
    _mpc_solver,
    finite_difference_step_response,
    fit_neural,
    fit_thermal,
    log_episode,
    overlap_report,
    predictive_comparison,
    run_control_episode,
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


def test_the_reported_pole_survives_a_change_of_actuator_units() -> None:
    """``drift[1]`` is the decay at ``action = 0``, which is a choice of units, not of building.

    Maxima gives the exact law for this model class under ``u = alpha v + beta``: it is closed, and
    ``a -> a + beta b1``, ``b0 -> alpha b0``, ``b1 -> alpha b1`` (closure residual 0). So refitting
    the same log with the action re-expressed on a ``[15, 25]`` scale -- what the two setpoint
    BOPTEST cases report -- must move ``drift[1]`` by exactly ``-(lo/span) b1`` and must leave
    :meth:`ThermalFit.decay` alone. Measured here: ``drift[1]`` moves 0.0113 while ``decay`` agrees
    to 2.6e-7, a separation of four orders, and the emulator reproduces the same invariance at
    ``-1.3969`` against ``-1.3970`` on hydronic.

    The tolerances are not symmetric on purpose. ``decay`` is invariant algebraically and is held to
    it; ``b1`` scales only up to the nuisance ridge, which leaks 0.15% of the shift because the
    penalty sees a column whose scale the substitution changed.
    """
    log = _synthetic_log(weather_gain=0.05)
    lo, span = 15.0, 10.0
    native = fit_thermal(log, adjusted=True)
    rescaled = fit_thermal(dataclasses.replace(log, action=log.action * span + lo), adjusted=True)

    assert rescaled.drift[1] - native.drift[1] == pytest.approx(
        -(lo / span) * native.channel[1], abs=1e-4
    )
    assert rescaled.channel[1] == pytest.approx(native.channel[1] / span, rel=2e-3)
    assert rescaled.authority() * span == pytest.approx(native.authority(), rel=1e-3)
    assert rescaled.decay() == pytest.approx(native.decay(), abs=1e-5)
    assert abs(rescaled.drift[1] - native.drift[1]) > 100 * abs(rescaled.decay() - native.decay())


def test_a_setpoint_actuator_reports_a_runaway_pole_for_a_decaying_building() -> None:
    """The emulator's hydronic fit, as algebra: the sign flip needs no estimator to reproduce.

    Coefficients as fitted on 960 half-hour rows of ``bestest_hydronic``. The building decays at
    -1.40/h at the setpoint it actually held, and the same fit reports +6.42/h at a setpoint of
    0 C -- 15 K below anything the actuator can command. Pinned because the first version of
    :class:`RunawayDriftError` refused this fit, and every such refusal was a false positive.
    """
    fit = dataclasses.replace(_plant(0.0), drift=(0.0, 6.4197), channel=(9.2965, -0.3779))
    hydronic = dataclasses.replace(fit, action_mean=20.685)

    assert hydronic.drift[1] > 0.0
    assert hydronic.decay() == pytest.approx(-1.397, abs=1e-3)
    assert hydronic.stable
    assert not hydronic.stable_over(15.0, 16.0)  # the low end of the box does not decay
    assert hydronic.stable_over(20.0, 25.0)


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
        action_mean=0.5,
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
        action_mean=0.5,
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
        action_mean=0.5,
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
    commanded = [float(solve(temp, bounds, covariates).actions[0]) for temp in grid]
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
            )(23.0, bounds, covariates).actions[0]
            for budget in (600, 3000)
        ]
        assert commanded[0] == pytest.approx(commanded[1], abs=0.02), (authority, commanded)


def test_the_plan_does_not_depend_on_the_units_the_actuator_reports_in() -> None:
    """Reparametrising the actuator must move the command and nothing else.

    The heat pump modulates on ``[0, 1]``; the two setpoint cases command Celsius on ``[15, 25]``.
    An effort term written as ``(action - lo)**2`` is therefore 100x larger on those cases for the
    same fraction of travel, so one ``w_comfort`` would buy an energy-first controller on one case
    and a comfort-first controller on another while the table claims a fixed objective.

    The check maps the plant through the same affine change of variable as the actuator -- the
    channel divides by the span and the displaced intercept moves into the drift -- so the two
    problems are the same problem in different units and the commands must agree to the map.
    Measured against the unnormalised effort term the two disagree by 0.037 / 0.009 / 0.012 of
    actuator travel at authority 0.2 / 0.5 / 1.0 -- an order above the tolerance, and an
    under-statement of what the setpoint cases would have carried, because this operating point
    sits near the low bound where the effort term has least leverage.
    """
    lo, span = 15.0, 10.0
    rescaled_case = dataclasses.replace(HEAT_PUMP, action_lo=lo, action_hi=lo + span)
    bounds, covariates = jnp.full(16, 21.0), jnp.zeros((16, 3))
    for authority in (0.2, 0.5, 1.0):
        native = _plant(authority)
        rescaled = dataclasses.replace(
            native,
            drift=(native.drift[0] - native.channel[0] * lo / span, native.drift[1]),
            channel=(native.channel[0] / span, native.channel[1]),
        )
        kwargs = {"w_comfort": 800.0, "margin": 1.5, "iterations": 600}
        commanded = _mpc_solver(native, HEAT_PUMP, 0.5, **kwargs)(23.0, bounds, covariates).actions[
            0
        ]
        in_setpoint_units = _mpc_solver(rescaled, rescaled_case, 0.5, **kwargs)(
            23.0, bounds, covariates
        ).actions[0]
        assert (in_setpoint_units - lo) / span == pytest.approx(commanded, abs=2e-3), (
            authority,
            commanded,
            in_setpoint_units,
        )


@pytest.mark.parametrize("pole", [0.0, 1.8807, 8.3633])
def test_a_runaway_fit_is_refused_before_the_emulator_is_touched(pole: float) -> None:
    """A horizon planned on a non-decaying drift is an artefact, and an expensive one.

    The client points at a closed port, so the only way this raises ``RunawayDriftError`` instead of
    a connection error is if the check runs *before* ``select``. That ordering is the point: on the
    two setpoint-actuated cases the fitted pole came back between ``+1.9`` and ``+8.4`` on every
    policy and both arms, and the resulting commands stalled BOPTEST's own solver past the client
    timeout -- an hour of emulator time and an orphaned worker child for a number that could not
    have meant anything. ``0.0`` is in the grid because marginal stability is already unusable: the
    homogeneous solution stops decaying at exactly zero, not somewhere past it.
    """
    fit = _plant(0.5, pole=pole)
    log = _synthetic_log(weather_gain=0.05, n=64)
    with pytest.raises(RunawayDriftError, match="does not decay"):
        run_control_episode(BOPTestClient("http://127.0.0.1:1"), HEAT_PUMP, fit, log, steps=1)


def test_the_track_refuses_to_run_without_an_emulator_rather_than_scoring_nothing() -> None:
    """A silently skipped track is indistinguishable from a track that found nothing."""
    with pytest.raises(RuntimeError, match=re.escape("127.0.0.1:1")):
        track_boptest_causal("http://127.0.0.1:1")


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        log_episode(
            BOPTestClient("http://127.0.0.1:1"), HEAT_PUMP, policy="greedy", seed=0, steps=1
        )


def _audited(
    fit: ThermalFit,
    temp: float,
    floor: float,
    *,
    w_comfort: float = 800.0,
    radius: float = 0.073,
    alpha: float = 1.0,
) -> tuple[HorizonPlan, StepAudit]:
    """One MPC solve on flat boundary conditions, plus the §40 price of the plan it returned."""
    dt = 0.5
    bounds, covariates = jnp.full(16, floor), jnp.zeros((16, 3))
    plan = _mpc_solver(fit, HEAT_PUMP, dt, w_comfort=w_comfort, margin=1.5, iterations=600)(
        temp, bounds, covariates
    )
    audit = SafetyAudit(radius=radius, alpha=alpha)
    return plan, _audit_plan(fit, HEAT_PUMP, plan, covariates, floor, dt, audit)


def test_the_planned_trajectory_is_the_one_the_planner_integrated() -> None:
    """The audit's whole premise: it prices the plan the solver optimised, not a copy of it.

    ``HorizonPlan.states`` is carried out of the solver's own scan rather than recomputed, so this
    checks the carrying rather than the integrator -- and it is what makes the certificate's drift
    and channel evaluations land on the same states the comfort hinge was evaluated at.
    """
    fit = _plant(0.5)
    plan, _ = _audited(fit, 20.0, 21.0)
    covariate = jnp.zeros(3)
    for k in range(plan.actions.shape[0]):
        stepped = plan.states[k] + 0.5 * fit.rate(plan.states[k], covariate, plan.actions[k])
        assert float(plan.states[k + 1]) == pytest.approx(float(stepped), rel=1e-12, abs=1e-12)


def test_the_certificate_prices_the_channel_the_fit_reports_at_the_radius_asked_for() -> None:
    """``guaranteed = drift + channel*u - radius*|u|``, recomputed from the fit alone.

    Three conventions have to be right at once for this to hold and each has a plausible wrong
    version: the barrier's gradient must be 1 (an augmented state would charge ``sqrt(2)``), the
    ``(gamma, cvar_gap)`` pair must realise ``radius`` (the natural reading of a unit gap is the
    other way round), and the ``Dynamics`` adapter must hand ``certify_safety`` the same rate the
    planner used. Checked against arithmetic that shares none of that code.
    """
    fit, radius = _plant(0.5), 0.073
    plan, priced = _audited(fit, 20.0, 21.0, radius=radius)
    temp, action = float(plan.states[0]), float(plan.actions[0])
    covariate = jnp.zeros(3)
    drift = float(fit.rate(jnp.asarray(temp), covariate, jnp.zeros(())))
    channel = float(fit.rate(jnp.asarray(temp), covariate, jnp.ones(()))) - drift
    assert priced.guaranteed == pytest.approx(drift + channel * action - radius * abs(action))
    assert priced.required == pytest.approx(-(temp - 21.0))
    assert priced.certified is (priced.guaranteed >= priced.required)


def test_the_audit_reads_the_forecast_row_belonging_to_the_step_it_prices() -> None:
    """A time-varying plant audited against row 0 everywhere would be a silent wrong answer.

    The weather enters the drift as a regressor, so a horizon whose boundary conditions move is the
    normal case rather than the exotic one -- and an adapter that ignored ``t`` would still return a
    plausible certificate. Distinct rows are the only way to see the difference.
    """
    fit = dataclasses.replace(_plant(0.5), weather_drift=(1.0, 0.0, 0.0))
    covariates = jnp.arange(16.0).reshape(16, 1) * jnp.asarray([[1.0, 0.0, 0.0]])
    field = _horizon_dynamics(fit, covariates, 0.5)
    for k in (0, 7, 15):
        priced = float(field(0.5 * k, jnp.asarray([21.0]), jnp.asarray([0.3]))[0])
        expected = float(fit.rate(jnp.asarray(21.0), covariates[k], jnp.asarray(0.3)))
        assert priced == pytest.approx(expected, rel=1e-12)


def test_the_sensitivity_level_and_the_radius_are_one_convention_not_two() -> None:
    """``gamma`` exists so ``(G-1)/(G+1)`` reads the radius back; illegal radii do not exist."""
    for radius in (0.0, 0.073, 0.5, 0.9):
        gamma = SafetyAudit(radius=radius).gamma
        assert (gamma - 1.0) / (gamma + 1.0) == pytest.approx(radius)
    with pytest.raises(ValueError, match="radius must lie"):
        SafetyAudit(radius=1.0)
    with pytest.raises(ValueError, match="class-K gain"):
        SafetyAudit(alpha=0.0)


def test_the_filter_leaves_a_command_alone_where_the_barrier_has_slack() -> None:
    """Least-restrictive means exactly this: a certified plan is executed unchanged.

    At 24 C against a 21 C floor the barrier allows 3 K/h of cooling and the building does far less,
    so every action in the box is admissible and the filter has nothing to clip. A filter that moved
    the command here would be a controller, not a safety layer.
    """
    _, priced = _audited(_plant(0.5), 24.0, 21.0)
    assert priced.margin == pytest.approx(3.0)
    assert priced.certified
    assert priced.filtered == pytest.approx(priced.nominal)


def test_the_filter_raises_a_command_a_planner_traded_away_for_effort() -> None:
    """In deficit the filter clips *up*, to the left endpoint of the admissible interval.

    The planner has to be willing to accept a comfort shortfall for this to have anything to say,
    which is what the low ``w_comfort`` buys: at the harness default the comfort term outweighs
    effort by four orders and the command is already saturated whenever the zone is below its bound,
    so the filter would have nothing to raise. That is itself worth knowing -- on this plant the
    certificate can only bind where the *objective*, not the actuator, left the margin unspent.

    The endpoint is checked against the closed form ``deficit / (channel - radius)`` rather than by
    an inequality: "the action went up" would also pass if the filter had saturated, which is a
    different behaviour with a different energy bill.
    """
    fit, radius = _plant(0.5), 0.073
    plan, priced = _audited(fit, 20.6, 21.0, w_comfort=0.1, radius=radius)
    # The literal 20.6 is not the state the audit saw: at JAX's default float32 the trajectory holds
    # 20.600000381, and reading the deficit off the literal instead leaves an 8.9e-07 gap that looks
    # like a tolerance question and is a "which number is this" question.
    temp = float(plan.states[0])
    drift = float(fit.rate(jnp.asarray(temp), jnp.zeros(3), jnp.zeros(())))
    deficit = (21.0 - temp) - drift  # -alpha*h - drift at alpha = 1
    assert not priced.certified
    assert float(plan.actions[0]) == pytest.approx(0.438, abs=1e-3)
    assert priced.filtered == pytest.approx(deficit / (0.5 - radius))


def test_a_channel_the_radius_cannot_sign_switches_the_actuator_off() -> None:
    """The mechanism the confounded fit is expected to hit, on a plant built to hit it.

    Near where a bilinear channel crosses zero the identified effect is smaller than the radius, so
    its *sign* is not identified: no action has a guaranteed margin and the admissible interval is
    empty. The library's filter answers that with the margin-maximising extreme, and intersected
    with a heat pump's one-sided box that is the pump switched off -- the honest answer, since a
    model that cannot say whether heating warms the room has no business commanding heat.

    Both temperatures are uncertified and both have a deficit no authority can cover, so what
    separates them is only whether the channel can be signed at 0.073. Reading one without the other
    would confuse "the certificate refuses" with "the filter shuts down".
    """
    fit = dataclasses.replace(_plant(0.5), channel=(0.5 * 20.5, -0.5))  # channel zero at 20.5 C
    signed, unsigned = (_audited(fit, temp, 21.0)[1] for temp in (20.3, 20.4))

    assert fit.authority(20.3) == pytest.approx(0.100)  # above the radius: the sign is identified
    assert fit.authority(20.4) == pytest.approx(0.050)  # below it: the sign is not
    assert (signed.nominal, unsigned.nominal) == (1.0, 1.0)  # the planner asks for full heat twice
    assert not signed.certified and not unsigned.certified
    assert math.isnan(signed.gamma_star) and math.isnan(unsigned.gamma_star)
    assert signed.filtered == pytest.approx(1.0)  # sign known: push as hard as the box allows
    assert unsigned.filtered == pytest.approx(0.0)  # sign unknown: stop


@pytest.mark.skipif(not _URL, reason="set BOPTEST_URL to a running BOPTEST-Service")
def test_the_audit_alone_moves_no_command() -> None:
    """The certificate-off arm has to be the un-audited loop, or the ablation compares three things.

    ``certify_safety`` is read-only by construction, but "by construction" is what a reader has to
    take on trust; two short live episodes make it a measurement. BOPTEST replays the same weather
    from the same initialisation and the MPC is deterministic, so identical KPIs are the right bar
    rather than a tolerance.

    The plant is the synthetic one and the log only supplies the standardiser, because what is under
    test is the loop and not the estimator -- and two days of emulator, which is all this test can
    afford, fit a building that grows rather than decays and would be refused before the loop ran.
    """
    if not is_available(_URL):
        pytest.skip("BOPTEST_URL is set but the service is unreachable")
    client, fit = BOPTestClient(_URL), _plant(0.5)
    log = _synthetic_log(weather_gain=0.05, n=64)
    plain = run_control_episode(client, HEAT_PUMP, fit, log, steps=6)
    audited = run_control_episode(client, HEAT_PUMP, fit, log, steps=6, audit=SafetyAudit())
    assert "cert_uncertified" in audited and "cert_uncertified" not in plain  # the audit did run
    # `wall_s` and BOPTEST's `time_rat` measure how long the controller took, and the audit is not
    # free -- it adds a `certify_safety` trace per step. Every KPI that describes the *building*
    # matches to every digit, which is the claim.
    for key, value in plain.items():
        if key not in ("wall_s", "time_rat"):
            assert audited[key] == value, key


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

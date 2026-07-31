"""Track J: the pendulum harness, checked against the environment it wraps.

Everything here runs offline -- ``Pendulum-v1`` is a few lines of arithmetic and needs no service --
so unlike the BOPTEST tests these are not gated. The ladder starts from the one check that makes all
the others meaningful: that the fitted model class, integrated the way the estimator inverts it,
reproduces Gymnasium's own step exactly.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("gymnasium", reason="Track J needs the `gym` extra")

from causaldyn_bench.pendulum_causal import (
    ActuatorClipError,
    UnusablePlanError,
    _mpc_solver,
    default_spec,
    extrapolation_error,
    fit_pendulum,
    holdout_rate_error,
    log_episode,
    make_env,
    oracle_fit,
    predicted_naive_gain,
    run_control_episode,
    run_overlap_ablation,
)

STEPS = 4000  # the exogenous arms need it: at 1200 their sampling error is 8-15%, not 2-3%

# The exactness claims are float64 claims, and JAX defaults to float32 -- which is how the ordinary
# CI job runs this file. At float32 the bounds below are noise floors rather than claims: a
# least-squares channel over STEPS rows accumulates roughly sqrt(STEPS) * eps, and one Euler step of
# the oracle carries about |state| * eps. Measured at float32: 1.7e-06 on the gain, 1.3e-05 on the
# two rows' relative agreement, 2.2e-07 on the step. The seven-digit figure `results/pendulum_causal
# .md` reports is gated by the `test-x64` CI job, which runs this file under JAX_ENABLE_X64=1.
X64 = jnp.zeros(()).dtype == jnp.float64
CHANNEL_TOL = 1e-6 if X64 else 1e-4
STEP_TOL = 1e-12 if X64 else 1e-5


def _log(**kwargs):
    spec = default_spec()
    return log_episode(spec, seed=kwargs.pop("seed", 0), steps=kwargs.pop("steps", STEPS), **kwargs)


def test_the_spec_is_the_environments_own_constants() -> None:
    spec = default_spec()
    assert spec.gravity == pytest.approx(15.0)
    assert spec.gain == pytest.approx(3.0)
    assert spec.dt == pytest.approx(0.05)
    assert (spec.max_torque, spec.max_speed) == (2.0, 8.0)


def test_the_oracle_reproduces_the_environments_own_step() -> None:
    """The load-bearing test: ``f_known`` plus an Euler step *is* Gymnasium's semi-implicit update.

    If this fails, every fitted coefficient below is measuring the integrator rather than the plant.
    """
    spec = default_spec()
    oracle = oracle_fit(spec)
    env = make_env()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        env.state = np.array([rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0)])
        action = float(rng.uniform(-1.5, 1.5))
        predicted = np.asarray(oracle.next_state(env.state[0], env.state[1], action))
        env.step(np.array([action]))
        worst = max(worst, float(np.abs(predicted - env.state).max()))
    assert worst < STEP_TOL


def test_adjustment_recovers_the_known_gain_and_omitting_the_wind_does_not() -> None:
    log = _log(policy="reactive")
    naive = fit_pendulum(log, adjusted=False)
    adjusted = fit_pendulum(log, adjusted=True)
    assert adjusted.gain == pytest.approx(log.spec.gain, abs=CHANNEL_TOL)
    assert adjusted.identified and not naive.identified
    assert naive.gain_error > 1.0


def test_adjustment_leaves_a_randomised_log_where_it_was() -> None:
    """The falsifier for the estimator itself: on an exogenous action it must change nothing."""
    log = _log(policy="random")
    naive = fit_pendulum(log, adjusted=False)
    adjusted = fit_pendulum(log, adjusted=True)
    assert adjusted.gain == pytest.approx(log.spec.gain, abs=CHANNEL_TOL)
    assert naive.gain == pytest.approx(log.spec.gain, rel=0.05)


def test_the_channels_two_rows_agree_up_to_the_integrators_factor() -> None:
    log = _log(policy="reactive")
    for adjusted in (False, True):
        fit = fit_pendulum(log, adjusted=adjusted)
        assert fit.coupling == pytest.approx(fit.gain * log.spec.dt, rel=CHANNEL_TOL)


@pytest.mark.parametrize("compensation", [0.0, 0.5, 1.0])
def test_the_naive_gain_follows_the_omitted_variable_closed_form(compensation: float) -> None:
    log = _log(policy="reactive", compensation=compensation)
    naive = fit_pendulum(log, adjusted=False)
    predicted = predicted_naive_gain(log)
    assert naive.gain == pytest.approx(predicted, abs=0.12)


def test_partial_compensation_flips_the_sign_where_full_compensation_does_not() -> None:
    """The non-monotone claim, which is the part of the closed form that could have been wrong."""
    partial = fit_pendulum(_log(policy="reactive", compensation=0.5), adjusted=False)
    full = fit_pendulum(_log(policy="reactive", compensation=1.0), adjusted=False)
    assert partial.gain < 0.0
    assert 0.0 < full.gain < partial.spec.gain


def test_overlap_collapses_when_the_operator_stops_exploring() -> None:
    rows = run_overlap_ablation(steps=STEPS, explores=(0.12, 0.0))
    explored, deterministic = rows[0], rows[1]
    assert explored["overlap"] > 1e-4
    assert deterministic["overlap"] < 1e-12
    assert explored["adjusted_error"] < CHANNEL_TOL
    assert deterministic["adjusted_error"] > 1.0


def test_dropping_the_physics_prior_costs_extrapolation() -> None:
    """The two axes are independent, which is the claim a single metric would hide.

    The threshold is the *seed-0* ratio (34x: 2.39 against 0.070 rad/s^2) rather than the five-seed
    mean, which is 520x. Writing the mean into a single-seed test would be writing a number the test
    cannot see.
    """
    log = _log(policy="reactive")
    structured = fit_pendulum(log, adjusted=True, physics=True)
    flexible = fit_pendulum(log, adjusted=True, physics=False)
    assert extrapolation_error(flexible) > 20.0 * extrapolation_error(structured)


def test_the_physics_prior_buys_an_exact_channel_rather_than_a_correct_sign() -> None:
    """What structure costs the channel is precision, not direction.

    The flexible arm's gain is unbiased in the mean -- the logged action is exogenous to the angle,
    so the polynomial's misfit of gravity lands in the error term -- but it carries a seed spread of
    about 0.37 where the structured arm returns the environment's constant on every seed. One seed
    is therefore allowed two standard deviations here, and pinned away from exactness.
    """
    log = _log(policy="reactive")
    structured = fit_pendulum(log, adjusted=True, physics=True)
    flexible = fit_pendulum(log, adjusted=True, physics=False)
    assert structured.gain == pytest.approx(log.spec.gain, abs=CHANNEL_TOL)
    assert flexible.gain == pytest.approx(log.spec.gain, abs=0.75)
    assert flexible.gain_error > 1e-3


def test_held_out_prediction_prefers_the_fit_with_the_wrong_sign() -> None:
    """Prediction does not rank causal models -- here it actively prefers the unusable one.

    The unadjusted fit is by construction the best linear predictor of the rate given the state and
    the *commanded* action, and the holdout is drawn from the same confounded policy. Reporting a
    held-out error as evidence about a control channel is therefore reporting the wrong number.
    """
    log = _log(policy="reactive", steps=2000)
    train, holdout = log.split()
    naive = fit_pendulum(train, adjusted=False)
    adjusted = fit_pendulum(train, adjusted=True)
    assert holdout_rate_error(naive, holdout) < holdout_rate_error(adjusted, holdout)
    assert naive.gain < 0.0 < adjusted.gain


def test_the_solver_reaches_the_box_constrained_optimum() -> None:
    """Projected Adam against L-BFGS-B on the same objective: is the fixed budget enough?"""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    spec = default_spec()
    fit = oracle_fit(spec)
    horizon, bound = 25, spec.max_torque - 0.5

    def rollout(actions, theta0, theta_dot0):
        def step(state, action):
            nxt = fit.next_state(state[0], state[1], action)
            return nxt, nxt[0] ** 2 + 0.1 * nxt[1] ** 2 + 0.001 * action**2

        _, costs = jax.lax.scan(step, jnp.stack([theta0, theta_dot0]), actions)
        return jnp.sum(costs)

    value = jax.jit(rollout)
    grad = jax.jit(jax.grad(rollout))
    origin = (jnp.asarray(0.15), jnp.asarray(0.0))
    reference = minimize(
        lambda a: (
            float(value(jnp.asarray(a), *origin)),
            np.asarray(grad(jnp.asarray(a), *origin), dtype=np.float64),
        ),
        np.zeros(horizon),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-bound, bound)] * horizon,
        options={"maxiter": 500},
    )
    solve = _mpc_solver(
        fit,
        horizon=horizon,
        iterations=400,
        bound=bound,
        weight_speed=0.1,
        weight_effort=0.001,
        learning_rate=0.3,
    )
    plan = solve(0.15, 0.0)
    optimum = float(value(jnp.asarray(reference.x), *origin))
    assert float(value(plan, *origin)) < 1.2 * optimum
    assert optimum < 0.01 * float(value(jnp.zeros(horizon), *origin))


def test_a_wrong_signed_channel_is_what_the_controller_acts_on() -> None:
    log = _log(policy="reactive")
    oracle = run_control_episode(oracle_fit(log.spec), seed=0, steps=20)
    adjusted = run_control_episode(fit_pendulum(log, adjusted=True), seed=0, steps=20)
    naive = run_control_episode(fit_pendulum(log, adjusted=False), seed=0, steps=20)
    assert adjusted["cost"] == pytest.approx(oracle["cost"], rel=0.05)
    assert naive["cost"] > 10.0 * oracle["cost"]
    assert not naive["sign_agrees"]


def test_a_drift_that_extrapolates_does_not_deliver_a_controller() -> None:
    """Refusal or catastrophe, not a near miss -- the claim is that the arm has no plan to offer.

    Which of the two happens is a property of the draw: on the 3200-row training splits
    :func:`run_closed_loop` uses, the degree-5 drift diverges over the horizon and the plan comes
    back non-finite on all five seeds. Asserting only the refusal would make the test a hostage to
    that, so the alternative is admitted and bounded.
    """
    train, _ = _log(policy="reactive").split()
    flexible = fit_pendulum(train, adjusted=True, physics=False)
    oracle = run_control_episode(oracle_fit(train.spec), seed=0, steps=20)
    try:
        episode = run_control_episode(flexible, seed=0, steps=20)
    except UnusablePlanError as refusal:
        assert "non-finite" in str(refusal)
    else:
        assert episode["cost"] > 100.0 * oracle["cost"]


def test_a_design_that_would_clip_the_actuator_is_refused() -> None:
    spec = default_spec()
    with pytest.raises(ActuatorClipError, match="clip"):
        log_episode(spec, policy="random", seed=0, steps=200, explore=1.5, wind_cap=1.5)


def test_unknown_policy_is_rejected() -> None:
    spec = default_spec()
    with pytest.raises(ValueError, match="unknown logging policy"):
        log_episode(spec, policy="reset", seed=0, steps=10)


def test_the_log_stays_inside_the_environments_own_limits() -> None:
    log = _log(policy="reactive")
    assert np.abs(log.action + log.wind).max() < log.spec.max_torque
    assert np.abs(log.theta_dot_next).max() < log.spec.max_speed
    assert log.swing > 1.0  # the physics ablation needs angles where sin is not its argument
    assert math.isclose(float(log.wind.std()), 0.175, rel_tol=0.3)

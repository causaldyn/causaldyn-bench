"""Track O: the harness, checked against the environments it steps.

Both environments are a few lines of arithmetic, so none of this is gated on a service. The first
check is the one every other number rests on: that the linearisation the plans and the control arm
use is the Jacobian of the environment's own ``step``.
"""

import inspect
import json
import math

import numpy as np
import pytest
from scipy.stats import binomtest

pytest.importorskip("gymnasium", reason="Track O needs the `gym` extra")

from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

from causaldyn_bench.ope_calibration import (
    DAMPING_RATIO,
    DITHER,
    ENVIRONMENTS,
    MOUNTAIN_CAR_GRAVITY,
    STRESS_DITHER,
    Arm,
    Interval,
    Scores,
    _markdown,
    _record,
    clopper_pearson,
    design,
    gate,
    online_truth,
    run_environment,
    run_policy,
    score,
)


@pytest.mark.parametrize(
    ("key", "step", "atol"), [("pendulum", 1e-4, 1e-8), ("mountain_car", 1e-2, 1e-5)]
)
def test_the_linearisation_is_the_jacobian_of_the_environments_step(key, step, atol):
    operating = ENVIRONMENTS[key]()
    env = operating.make()

    def advance(x, u):
        operating.place(env, x)
        env.step(np.array([u]))
        return operating.read(env)

    at_rest = advance(np.zeros(2), 0.0)
    a = np.column_stack(
        [(advance(step * e, 0.0) - advance(-step * e, 0.0)) / (2 * step) for e in np.eye(2)]
    )
    b = (advance(np.zeros(2), step) - advance(np.zeros(2), -step)) / (2 * step)
    np.testing.assert_allclose(at_rest, 0.0, atol=1e-7)
    np.testing.assert_allclose(a, operating.a, atol=atol)
    np.testing.assert_allclose(b, operating.b[:, 0], atol=atol)


def test_the_mountain_cars_gravity_is_still_the_literal_the_linearisation_assumes():
    assert f"{MOUNTAIN_CAR_GRAVITY} * math.cos(3 * position)" in inspect.getsource(
        Continuous_MountainCarEnv.step
    )


@pytest.mark.parametrize("key", list(ENVIRONMENTS))
def test_the_logger_damps_the_linearisation_to_the_stated_ratio(key):
    operating = ENVIRONMENTS[key]()
    spec = design(operating)
    # the continuous-time poles, up to the step length, which the ratio does not see
    poles = np.log(np.linalg.eigvals(operating.a + operating.b @ spec.logger.gain).astype(complex))
    ratio = -poles.real / np.abs(poles)
    # the ratio is set in continuous time; the discrete step moves it by ~(omega dt)^2
    np.testing.assert_allclose(ratio, DAMPING_RATIO, rtol=0.02)


@pytest.mark.parametrize("key", list(ENVIRONMENTS))
@pytest.mark.parametrize(
    ("dither", "low", "high"), [(DITHER, 0.0, 0.002), (STRESS_DITHER, 0.015, 0.025)]
)
def test_the_logger_clips_as_rarely_as_the_design_says(key, dither, low, high):
    operating = ENVIRONMENTS[key]()
    spec = design(operating, dither)
    _, _, clips = run_policy(
        operating, operating.make(), spec.logger, spec.disturbance, 40_000, np.random.default_rng(1)
    )
    assert low <= clips / 40_000 <= high


@pytest.mark.parametrize("key", list(ENVIRONMENTS))
def test_the_online_truth_is_the_linearisations_value_when_nothing_clips(key):
    spec = design(ENVIRONMENTS[key]())
    truth = online_truth(spec, "moderate", (3,), steps=100_000)
    assert truth.clip_share == 0.0
    assert abs(truth.value - truth.linearised) < 4.0 * truth.se


def test_clopper_pearson_is_the_exact_binomial_interval():
    for covered, n in [(475, 500), (500, 500), (0, 20), (13, 17)]:
        expected = binomtest(covered, n).proportion_ci(confidence_level=0.95, method="exact")
        np.testing.assert_allclose(clopper_pearson(covered, n), (expected.low, expected.high))


def test_score_counts_what_each_side_of_the_truth_missed_and_what_was_refused():
    estimates = [
        Interval(1.0, 0.5, 1.5, 0.1, 30.0),
        Interval(3.0, 2.0, 4.0, 0.2, 20.0),  # wholly above the truth
        Interval(0.2, 0.0, 0.4, 0.3, None),  # wholly below
        Interval(1.1, 0.9, 1.3, 0.4, 10.0),
        "refused: too few effective samples",
        "refused: too few effective samples",
    ]
    scored, refusals, scores = score(estimates, truth=1.0)
    assert scored == 4
    assert refusals == {"refused: too few effective samples": 2}
    assert scores is not None
    assert scores.coverage == 0.5
    assert (scores.above_truth, scores.below_truth) == (0.25, 0.25)
    assert scores.mean_error == pytest.approx(0.325)
    assert scores.median_degrees_of_freedom == 20.0


def _arm(design_name, model, model_error, coverage):
    scores = Scores(coverage, (0.0, 1.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)
    return Arm(design_name, "moderate", model, "dr", model_error, 500, {}, scores)


def test_the_gate_reads_the_nominal_fitted_arms_and_nothing_else():
    assert gate(_arm("nominal", "fitted", 0.0, 0.93)) is True
    assert gate(_arm("nominal", "fitted", 0.0, 0.975)) is False
    assert gate(_arm("nominal", "fitted", 1.0, 1.0)) is True
    assert gate(_arm("nominal", "fitted", 1.0, 0.92)) is False
    assert gate(_arm("stress", "fitted", 0.0, 0.5)) is None
    assert gate(_arm("nominal", "linearisation", 0.0, 0.5)) is None
    refused = Arm("nominal", "moderate", "fitted", "mis", 0.0, 0, {"refused": 500}, None)
    assert gate(refused) is False


@pytest.mark.parametrize("key", list(ENVIRONMENTS))
def test_a_small_run_scores_every_arm_and_writes_its_report(key):
    run = run_environment(key, replicates=3, truth_steps=4_000)
    # nominal: two plans x two models x five arms; stress: the moderate plan only
    assert len(run.arms) == 30
    assert all(arm.scored + sum(arm.refusals.values()) == 3 for arm in run.arms)
    text = _markdown([run], replicates=3, x64=True)
    assert "## The gate" in text
    assert run.name in text
    record = json.loads(json.dumps(_record([run], replicates=3, x64=True)))
    assert len(record["environments"][0]["arms"]) == 30
    assert math.isfinite(record["environments"][0]["truths"]["moderate"]["value"])

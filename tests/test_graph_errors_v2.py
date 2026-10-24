"""Track Q, version 2: the a priori tolerance, the margin and the exceedance, each against the law
it is read from, and a small run.

The worlds' plants are checked against the panels they draw in ``test_graph_errors.py``; here the
radii and the exceedances are checked against draws of those plants.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.stats import chi2, norm

from causaldyn_bench.graph_errors import HORIZON, STATES, WORLDS, mean_path, plant
from causaldyn_bench.graph_errors_v2 import (
    MARGIN,
    RADII,
    READ,
    THRESHOLD,
    TOLERANCE,
    WIDEST,
    _markdown,
    _record,
    claims,
    exceedance,
    predictive_covariances,
    predictive_radius,
    run_world,
)


def _realised(name: str, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``draws`` realised (supply, wait) paths of ``name``'s true plant from a known start under an
    open-loop schedule, and the plant's mean path."""
    system = plant(WORLDS[name])
    start = np.zeros(system.a.shape[0])
    start[: len(STATES)] = (0.05, -0.02)
    actions = np.array([0.9, 0.5, 0.25])
    rng = np.random.default_rng(seed)
    root = np.linalg.cholesky(system.noise)
    x = np.tile(start, (draws, 1))
    paths = [x[:, : len(STATES)]]
    for action in actions:
        x = x @ system.a.T + system.b * action + rng.standard_normal(x.shape) @ root.T
        paths.append(x[:, : len(STATES)])
    return np.stack(paths, axis=1), mean_path(system, start, actions)[:, : len(STATES)]


@pytest.mark.parametrize(
    ("name", "radius"),
    [  # the pre-registered radii, to four figures (Maxima, from the generator's parameters)
        ("none", 0.4813),
        ("confounder", 0.7218),
        ("confounder AR", 0.6741),
        ("mediator", 0.7283),
        ("collider latent", 0.5779),
    ],
)
def test_the_a_priori_tolerance_is_the_widest_true_plants_own_spread(name, radius):
    """Over 400 000 draws of each true plant, the realised path is outside the closed-form radius
    at the plan's last step on 0.05 of them, within four standard errors; the tolerance is 1.1 of
    the widest, the mediator world's."""
    draws = 400_000
    paths, mean = _realised(name, draws, seed=list(WORLDS).index(name))
    gap = np.linalg.norm(paths[:, HORIZON] - mean[HORIZON], axis=1)
    assert RADII[name] == pytest.approx(radius, abs=5e-5)
    outside = float(np.mean(gap > RADII[name]))
    assert abs(outside - 0.05) < 4.0 * math.sqrt(0.05 * 0.95 / draws)
    assert WIDEST == "mediator"
    assert max(RADII.values()) == RADII["mediator"]
    assert pytest.approx(0.8012, abs=5e-5) == TOLERANCE == 1.1 * RADII["mediator"]


def test_the_margin_is_the_slack_of_a_tube_read_off_one_log():
    """The noise's read scale over 4400 transitions alone leaves a correctly specified 0.95 tube
    crossed with probability 0.05 + 0.00416 at its 5% quantile, exactly; with the fitted
    persistence's error, read from the base worlds' panel law, 0.05 + 0.00476 at the third step,
    the largest: the margin, to two figures."""
    assert READ == 4400
    z = norm.ppf(0.975)
    scale = math.sqrt(chi2.ppf(0.05, READ) / READ)
    assert 2.0 * norm.sf(z * scale) - 0.05 == pytest.approx(0.004161, abs=5e-7)
    # supply's variance given wait over the panel's 11 transitions, from its law at the start
    system, push = plant(WORLDS["none"]), np.array([0.08, -0.04])
    covariance, given = 0.04 * np.eye(2), []
    for _ in range(11):
        given.append(covariance[0, 0] - covariance[0, 1] ** 2 / covariance[1, 1])
        covariance = system.a @ covariance @ system.a.T + 1.5**2 * np.outer(push, push)
        covariance += system.noise
    error = math.sqrt(system.noise[0, 0] / (READ * np.mean(given)))
    assert error == pytest.approx(0.005961, abs=5e-7)
    a = system.a[0, 0]
    slopes = [0.0, a / (1 + a**2), (a + 2 * a**3) / (1 + a**2 + a**4)]
    slack = [
        2.0 * norm.sf(z * (1.0 - norm.ppf(0.95) * math.hypot(1 / math.sqrt(2 * READ), g * error)))
        - 0.05
        for g in slopes
    ]
    assert slack == pytest.approx([0.004155, 0.004319, 0.004756], abs=5e-7)
    assert float(f"{max(slack):.2g}") == MARGIN
    assert pytest.approx(0.0548, abs=1e-15) == THRESHOLD


@pytest.mark.parametrize(
    ("name", "step", "offset", "radius"),
    [
        ("mediator", 3, (0.0, 0.0), None),  # the radius itself
        ("confounder", 3, (0.10, -0.01), 0.72),  # a wrong graph's mean offset at a tube's radius
        ("confounder", 1, (0.05, 0.0), 0.44),
        ("confounder AR", 2, (-0.30, 0.02), 0.66),
        ("none", 1, (0.25, 0.03), 0.30),  # most of the mass outside
        ("collider latent", 2, (0.0, 0.0), 0.10),
    ],
)
def test_the_exceedance_is_the_share_of_the_true_plants_draws_outside_the_tube(
    name, step, offset, radius
):
    """Against 400 000 draws of the Gaussian, within four standard errors, on the plants' own
    covariances, where one state carries nearly all of the noise, and on one rotated off the
    axes."""
    draws = 400_000
    covariance = predictive_covariances(plant(WORLDS[name]))[step]
    radius = predictive_radius(covariance) if radius is None else radius
    exact = exceedance(offset, covariance, radius)
    rng = np.random.default_rng(step)
    e = rng.multivariate_normal(np.zeros(2), covariance, size=draws, method="cholesky")
    share = float(np.mean(np.linalg.norm(e + np.asarray(offset), axis=1) > radius))
    assert abs(share - exact) < 4.0 * math.sqrt(max(exact * (1.0 - exact), 1e-6) / draws)
    if offset == (0.0, 0.0) and name == "mediator":
        assert exact == pytest.approx(0.05, abs=1e-11)
    turn = np.array([[0.6, -0.8], [0.8, 0.6]])
    tilted = turn @ np.diag([0.04, 0.01]) @ turn.T
    e = rng.multivariate_normal(np.zeros(2), tilted, size=draws, method="cholesky")
    share = float(np.mean(np.linalg.norm(e + np.asarray(offset), axis=1) > radius))
    exact = exceedance(offset, tilted, radius)
    assert abs(share - exact) < 4.0 * math.sqrt(max(exact * (1.0 - exact), 1e-6) / draws)


def test_two_replicates_score_every_arm_on_the_tubes_claim_and_write_the_report():
    """On a seed no scored run reads: the tolerance arms the plan leg, each plan's exceedance is
    read at every step, and a plan leaves exactly where one at a trusted step passes the
    threshold."""
    with ThreadPoolExecutor(1) as pool:
        run = run_world("confounder AR", 2, pool, seed=7)
    for arm_runs in run.runs.values():
        for arm_run in arm_runs:
            assert arm_run["identified"]
            assert len(arm_run["exceedance"]) == HORIZON
            assert all(0.0 <= e <= 1.0 for e in arm_run["exceedance"])
            read = claims(arm_run, "as is")
            trusted = arm_run["exceedance"][: arm_run["trustworthy_steps"]]
            assert read.left == (read.acted and any(e > THRESHOLD for e in trusted))
    assert all(arm_run["trustworthy_steps"] >= 1 for arm_run in run.runs["oracle"])
    text = _markdown([run], 2, 7, True, None, None)
    assert "## The calibration check" in text
    assert "## confounder AR" in text
    record = json.loads(json.dumps(_record([run], 2, 7, True, None, None)))
    assert record["rule"]["tolerance"] == TOLERANCE
    assert len(record["worlds"][0]["replicates"]["wrong"]["exceedance"]) == 2
    check = record["calibration"]["worlds"]["confounder AR"]
    assert check["plans"]["n"] == 2
    assert check["steps"]["n"] == sum(r["trustworthy_steps"] for r in run.runs["oracle"])

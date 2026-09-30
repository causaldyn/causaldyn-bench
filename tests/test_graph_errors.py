"""Track Q: the harness, checked against the worlds it draws.

Every failure the track counts is read against the true plant, so the first check is that the plant
the truth is computed on is the law the panels are drawn from.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor

import chc
import numpy as np
import pytest
from scipy.optimize import minimize

from causaldyn_bench.graph_errors import (
    DIRECT,
    HORIZON,
    STATES,
    TOLERANCE,
    WORLDS,
    _markdown,
    _record,
    best_cost,
    by_fit,
    claims,
    draw,
    gate,
    planning_cost,
    plant,
    regret,
    run_world,
    stop_effect,
)


def _transitions(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Over 20 000 units: a design of an intercept, the state, the incentive and the confounder
    where there is one, and the next period's state and confounder."""
    world = WORLDS[name]
    columns = draw(world, (7, 0), units=20_000).columns
    order = np.lexsort((columns["time"], columns["unit"]))
    same = columns["unit"][order][1:] == columns["unit"][order][:-1]
    now, then = order[:-1][same], order[1:][same]
    extra = [world.confounder.column] if world.confounder is not None else []
    design = np.column_stack(
        [np.ones(now.size), *(columns[name][now] for name in (*STATES, "incentive", *extra))]
    )
    following = np.column_stack([columns[name][then] for name in (*STATES, *extra)])
    return design, following


@pytest.mark.parametrize(
    "name", ["none", "confounder AR", "mediator", "collider latent", "collider observed"]
)
def test_the_plant_the_truth_is_read_on_is_the_law_the_panels_are_drawn_from(name):
    """Least squares over 20 000 units recovers the plant's mean map and its noise. The incentive is
    drawn afresh, so its coefficient is its total effect, through the orders where there are any."""
    system = plant(WORLDS[name])
    design, following = _transitions(name)
    coefficients, *_ = np.linalg.lstsq(design, following, rcond=None)
    residual = following - design @ coefficients
    # rows: the intercept, supply, wait, the incentive, then the confounder where there is one
    rows = [1, 2, *range(4, 2 + system.a.shape[0])]
    np.testing.assert_allclose(coefficients[rows].T, system.a, atol=0.004)
    np.testing.assert_allclose(coefficients[3], system.b, atol=0.002)
    covariance = np.cov(residual, rowvar=False)
    np.testing.assert_allclose(np.diag(covariance), np.diag(system.noise), rtol=0.03)
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    np.testing.assert_allclose(covariance / scale, system.noise / scale, atol=0.01)


def test_the_mediator_carries_most_of_the_incentives_push_and_the_total_is_unchanged():
    mediator = WORLDS["mediator"].mediator
    assert mediator is not None
    np.testing.assert_allclose(plant(WORLDS["mediator"]).b, DIRECT)
    assert mediator.through > DIRECT[0] - mediator.through > 0


def test_the_best_schedule_is_the_optimum_over_the_box_and_idling_buys_nothing():
    system = plant(WORLDS["none"])
    start = np.array([0.05, -0.02])
    found = minimize(
        lambda actions: planning_cost(system, start, actions),
        np.zeros(HORIZON),
        bounds=[(-2.0, 2.0)] * HORIZON,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-12},
    )
    assert best_cost(system, start) <= found.fun + 1e-12
    assert best_cost(system, start) == pytest.approx(found.fun, abs=1e-9)
    lost, stakes = regret(system, start, np.zeros(HORIZON))
    assert lost == pytest.approx(stakes)
    assert regret(system, start, found.x)[0] == pytest.approx(0.0, abs=1e-9)
    assert stakes > 0.0


@pytest.mark.parametrize(
    ("name", "chosen"),
    [
        ("none", ("demand",)),
        ("confounder", ("demand", "promo")),
        ("mediator", ("demand", "orders")),
        ("collider latent", ("demand", "sessions")),
    ],
)
def test_the_fit_arm_adjusts_for_whatever_explains_supply_best(name, chosen):
    """A confounder and a confounded mediator both fit better, and only one of them belongs."""
    assert by_fit(draw(WORLDS[name], (11, 0)).columns) == chosen


def test_each_wrong_graph_changes_the_adjustment_set_where_its_world_says_it_does():
    sets = {}
    for name, world in WORLDS.items():
        columns = draw(world, (3, 0), units=4).columns
        true = chc.CausalGraph.from_edges(world.true_edges, latent=world.true_latent)
        wrong = chc.CausalGraph.from_edges(world.wrong_edges)
        for graph in (true, wrong):
            graph.require_columns(columns)
        sets[name] = tuple(
            set(graph.adjustment_set(treatment="incentive", outcome="supply").covariates)
            for graph in (true, wrong)
        )
    assert sets["confounder"] == ({"demand", "promo"}, {"demand"})
    assert sets["mediator"] == ({"demand"}, {"demand", "orders"})
    assert sets["collider latent"] == ({"demand"}, {"demand", "sessions"})
    assert sets["collider observed"] == ({"demand"}, {"demand", "sessions"})
    for name in ("none", "chase", "sticky", "non-ancestor"):
        assert sets[name] == ({"demand"}, {"demand"})


def _run(**changes):
    run = {
        "identified": True,
        "covariates": ("demand",),
        "trustworthy_steps": 3,
        "fit_flag": False,
        "channel": 0.08,
        "deviation": (0.0, 0.01, 0.02, 0.03),
        "regret": 0.0,
        "stakes": 1.0,
        "regret_bound": 0.0,
        "evaluation": "evaluated",
        "covers": True,
        "later_flag": False,
    }
    return run | changes


def test_a_claim_fails_only_where_the_certificate_trusted_it_and_the_stop_only_withdraws():
    # the path leaves the tolerance at the third step, which only a three-step trust covers
    beyond = (0.0, 0.01, 0.02, TOLERANCE + 0.01)
    assert claims(_run(deviation=beyond), "as is").left
    assert not claims(_run(deviation=beyond, trustworthy_steps=2), "as is").left
    assert not claims(_run(deviation=beyond, trustworthy_steps=0), "as is").acted
    # the regret passes its bound by more than a tenth of the stakes
    assert claims(_run(regret=0.2, regret_bound=0.05), "as is").lost
    assert not claims(_run(regret=0.12, regret_bound=0.05), "as is").lost
    assert claims(_run(covers=False), "as is").missed
    # a flag on the fitted panel withdraws the plan and its evaluation; on the later panel, the
    # evaluation alone
    stopped = claims(_run(deviation=beyond, fit_flag=True), "stop")
    assert not stopped.acted and not stopped.evaluated and not stopped.failed
    later = claims(_run(covers=False, later_flag=True), "stop")
    assert later.acted and not later.evaluated and not later.failed

    runs = [
        _run(deviation=beyond, fit_flag=True),  # a failure the stop removes
        _run(fit_flag=True),  # a sound claim it refuses
        _run(covers=False),  # a failure it cannot see
        _run(),
    ]
    effect = stop_effect(runs)
    assert (effect.failed, effect.removed, effect.sound, effect.refused_sound) == (2, 1, 2, 1)
    # a stop blind to the failures that withdraws two of the four removes a failure 5 times in 6
    assert effect.p_value == pytest.approx(5 / 6)


def test_a_flag_on_every_panel_of_the_wrong_graph_earns_its_stop_only_beside_the_true_graph():
    beyond = (0.0, 0.01, 0.02, TOLERANCE + 0.01)
    wrong = [_run(deviation=beyond, fit_flag=True) for _ in range(9)] + [_run(fit_flag=True)]
    true = [_run() for _ in range(10)]
    assert stop_effect(wrong).p_value == pytest.approx(1.0)
    assert stop_effect(wrong + true).p_value == pytest.approx(11 / math.comb(20, 10))


def test_a_small_run_scores_every_arm_and_writes_its_report():
    with ThreadPoolExecutor(1) as pool:
        runs = [run_world(name, 1, pool) for name in ("none", "collider observed")]
    for run in runs:
        assert {(s.arm, s.mode) for s in run.scores} == {
            (arm, mode) for arm in ("oracle", "wrong", "by fit") for mode in ("as is", "stop")
        }
        for arm_runs in run.runs.values():
            (only,) = arm_runs
            assert only["identified"]
            assert math.isfinite(only["regret"])
    control = runs[0].runs["oracle"][0]
    assert control["evaluation"] == "evaluated"
    assert max(control["deviation"]) < TOLERANCE
    # the collider's other parent is adjusted for too, so the fit keeps its channel, and the check
    # sees the dependence the collider opens
    collider = runs[1].runs["wrong"][0]
    assert collider["fit_flag"] is True
    assert collider["channel"] == pytest.approx(DIRECT[0], abs=0.005)
    text = _markdown(runs, 1, x64=True)
    assert "## The gate" in text
    assert "## collider observed" in text
    record = json.loads(json.dumps(_record(runs, 1, x64=True)))
    assert record["gate"]["size"]["n"] == gate(runs).size.n == 2
    assert record["worlds"][1]["replicates"]["wrong"]["fit_flag"] == [True]
    assert record["worlds"][1]["stop"]["failed"] == 0

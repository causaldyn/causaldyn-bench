"""P3.3's protocol offline, on a surrogate plant; the live run is the experiment, not a test."""

import json
import math
import os
import sys
from dataclasses import replace

import jax
import numpy as np
import pytest

from causaldyn_bench import boptest_capped
from causaldyn_bench.boptest import BOPTestClient, is_available
from causaldyn_bench.boptest_capped import (
    Arm,
    Design,
    Reference,
    SurrogatePlant,
    ZoneModel,
    _probe_energy,
    markdown,
    plan_budget,
    run_episode,
    run_experiment,
    verdict,
)

_URL = os.environ.get("BOPTEST_URL")
_EFFECT = 0.6  # the surrogate's true heating rate per unit modulation, K/h
_SMALL = Design(
    rounds=24,
    window_days=(1, 3),
    seeds=4,
    nuisance_rounds=240,
    reference_rounds=240,
    power_seeds=2,
    bootstrap=500,
)


def _surrogate(noise_sd: float = 0.1, seed: int = 0) -> SurrogatePlant:
    """Twenty days of two-hour rounds: a daily outdoor cycle, midday sun, a constant bound."""
    rounds = 240
    hours = 2.0 * np.arange(rounds)
    outdoor = 3.0 + 4.0 * np.sin(2.0 * np.pi * hours / 24.0)
    solar = 0.3 * np.clip(np.sin(2.0 * np.pi * (hours - 6.0) / 24.0), 0.0, None)
    covariates = np.column_stack([outdoor, solar, np.full(rounds, 21.0)])
    truth = ZoneModel(
        bias=0.45,
        pole=-0.03,
        weather=(0.02, 0.5, 0.0),
        weather_mean=(0.0, 0.0, 0.0),
        weather_scale=(1.0, 1.0, 1.0),
        authority=_EFFECT,
        residual_sd=noise_sd,
    )
    return SurrogatePlant(
        model=truth,
        effect=_EFFECT,
        covariates=covariates,
        temps=np.full(rounds, 21.0),
        noise=np.random.default_rng(seed).normal(0.0, noise_sd, rounds),
        step_s=7200.0,
    )


def test_the_block_fills_the_cap_and_the_taper_is_clipped_by_it() -> None:
    """The two schedules differ only in the envelope under the cap; both stop at the budget."""
    assert _probe_energy("block", 1, 0.09, 1.0, 0.5) == 0.09
    assert _probe_energy("block", 7, 0.09, 0.05, 0.5) == 0.05  # the budget binds before the cap
    assert _probe_energy("taper", 1, 0.09, 1.0, 0.05) == 0.05  # its own envelope binds
    assert _probe_energy("taper", 4, 0.01, 1.0, 0.05) == 0.01  # the cap binds
    assert math.isclose(_probe_energy("taper", 4, 0.09, 1.0, 0.05), 0.025)
    assert _probe_energy("block", 1, 0.09, 0.0, 0.5) == 0.0
    assert _probe_energy("none", 1, 0.09, 1.0, 0.5) == 0.0


def test_the_plan_is_gated_on_the_library_and_predicts_the_block_cheaper() -> None:
    """The library takes a scalar plant, not ``(A, K, c, I0)``; the knobs that map one onto the
    other are checked against its ``uncapped_floor`` identity, and at the SAME budget the library's
    block must come out no dearer than its taper -- Theorem 5 read off the prediction itself."""
    reference = Reference(
        effect=_EFFECT,
        noise_sd=0.16,
        energy=40.0,
        lag1=0.0,
        first_half=_EFFECT,
        second_half=_EFFECT,
        rounds=900,
    )
    rng = np.random.default_rng(3)
    caps = rng.uniform(0.0, 0.2, size=(3, 288))
    demand = rng.uniform(0.0, 0.8, size=(3, 288))
    plan = plan_budget(reference, caps, demand, Design())
    assert plan.floor_residual < 1e-9
    assert plan.mass > 0.0
    assert plan.block_cost <= plan.taper_cost
    assert math.isclose(plan.curvature, (2.0 * _EFFECT) ** 2)


def test_the_plan_refuses_when_the_library_disagrees_with_the_knobs(monkeypatch) -> None:
    """The gate has to fire, not merely exist: perturb the library's floor and the plan refuses."""
    real = boptest_capped.capped_exploration_policy

    def drifted(**kwargs):
        policy = real(**kwargs)
        return replace(policy, uncapped_floor=policy.uncapped_floor + 1.0)

    monkeypatch.setattr(boptest_capped, "capped_exploration_policy", drifted)
    reference = Reference(_EFFECT, 0.16, 40.0, 0.0, _EFFECT, _EFFECT, 900)
    with pytest.raises(RuntimeError, match="no longer reproduce"):
        plan_budget(reference, np.full((2, 288), 0.1), np.full((2, 288), 0.4), Design())


def _pairs(gaps: list[float], block: float = 1.0) -> list[dict]:
    return [
        {
            "window": k % 2,
            "seed": k,
            "taper": {"regret": block + gap, "spent": 1.0},
            "block": {"regret": block, "spent": 1.0},
        }
        for k, gap in enumerate(gaps)
    ]


@pytest.mark.parametrize(
    ("gaps", "decision"),
    [
        ([0.5, 0.6, 0.4, 0.55, 0.45, 0.5], "CONFIRMED"),
        ([-0.5, -0.6, -0.4, -0.55, -0.45, -0.5], "REFUTED"),
        ([1.0, -1.0, 0.8, -0.9, 0.1, -0.05], "INCONCLUSIVE"),
    ],
)
def test_the_gate_reads_the_interval(gaps: list[float], decision: str) -> None:
    lazy = [{"regret": 10.0}, {"regret": 12.0}]
    result = verdict(_pairs(gaps), lazy, replace(Design(), bootstrap=2000))
    assert result["decision"] == decision
    assert result["stands"]


def test_a_lazy_arm_that_competes_voids_the_verdict() -> None:
    """R12: if never exploring is as good as the block, the experiment is not testing exploration,
    and a CONFIRMED read off it would be the lazy arm's win, not the schedule's."""
    result = verdict(
        _pairs([0.5, 0.6, 0.4, 0.55]), [{"regret": 0.9}], replace(Design(), bootstrap=500)
    )
    assert result["decision"] == "CONFIRMED"
    assert not result["v1_exploring_matters"]
    assert not result["stands"]


def test_a_block_short_of_the_taper_budget_voids_the_verdict() -> None:
    pairs = _pairs([0.5, 0.6, 0.4, 0.55])
    for pair in pairs:
        pair["block"]["spent"] = 0.8
    result = verdict(pairs, [{"regret": 10.0}], replace(Design(), bootstrap=500))
    assert not result["v2_budgets_match"]
    assert not result["stands"]


def test_the_surrogate_answers_as_its_own_model() -> None:
    """A round on the surrogate is exactly ``drift + effect u + noise`` -- what the power figure and
    every offline test rely on, so it is pinned rather than trusted."""
    plant = _surrogate(noise_sd=0.0)
    first = plant.initialize("s", 10 * 7200.0, 0.0)
    temp = first["reaTZon_y"] - 273.15
    after = plant.advance("s", {"oveHeaPumY_u": 0.5, "oveHeaPumY_activate": 1})
    expected = temp + 2.0 * (plant.model.drift(temp, plant.covariates[10]) + _EFFECT * 0.5)
    assert math.isclose(after["reaTZon_y"] - 273.15, expected, rel_tol=0, abs_tol=1e-12)


def test_an_oracle_episode_holds_the_target_on_a_noiseless_surrogate() -> None:
    """Told the true effect and the true drift, the deadbeat controller lands on the target every
    round the actuator is not saturated: the controller is right before anything is learned."""
    plant = _surrogate(noise_sd=0.0)
    arm = Arm("none", 0.0, 0.0, _EFFECT, 1.0, 0.1, 0.0, fixed=_EFFECT)
    trace = run_episode(
        plant, "s", plant.model, _SMALL, arm, start_day=1, signs=np.ones(_SMALL.rounds)
    )
    inside = (trace.planned > 0.0) & (trace.planned < 1.0)
    assert inside.any()
    assert np.allclose(trace.next_temp[inside], _SMALL.target, atol=1e-9)


def test_the_running_estimate_is_the_probe_only_moment() -> None:
    """The planned action never identifies: after one round the estimate is the prior pooled with
    ``e (obs - theta_hat u0)`` and with nothing else -- in particular not with ``e obs``, which is
    also unbiased and would pass any recovery check while carrying the planned action's variance."""
    plant = _surrogate(noise_sd=0.0)
    prior, info, sd = 0.9, 25.0, 0.1
    arm = Arm("block", 1.0, 0.0, prior, info, sd, 0.0)
    trace = run_episode(
        plant, "s", plant.model, _SMALL, arm, start_day=1, signs=np.ones(_SMALL.rounds), rounds=2
    )
    e, planned = trace.probe[0], trace.planned[0]
    obs = (trace.next_temp[0] - trace.temp[0]) / _SMALL.dt - trace.drift[0]
    expected = (info * prior + e * (obs - prior * planned) / sd**2) / (info + e * e / sd**2)
    assert e != 0.0
    assert math.isclose(trace.estimate[1], expected, rel_tol=1e-12)


@pytest.fixture
def x64():
    """Stage 0 fits in JAX, and the live run refuses anything but float64; so do these tests."""
    with jax.enable_x64(True):
        yield


def test_the_protocol_end_to_end_on_the_surrogate(x64) -> None:
    """Every stage on a plant whose effect is known: stage 1 recovers it, every pair spends one
    budget on both arms, and the results serialise and render."""
    results = run_experiment(_surrogate(), _SMALL, say=lambda _line: None)
    assert results["precision"] == "float64"
    assert abs(results["reference"]["effect"] - _EFFECT) < 0.1 * _EFFECT
    assert results["verdict"]["pairs"] == len(_SMALL.window_days) * _SMALL.seeds
    for pair in results["pairs"]:
        assert pair["taper"]["spent"] > 0.0
        assert math.isclose(pair["block"]["spent"], pair["taper"]["spent"], rel_tol=1e-9)
        # the block front-loads: half its energy lands no later than the taper's half
        assert pair["block"]["half_spent_round"] <= pair["taper"]["half_spent_round"]
    assert len(results["lazy"]) == 2 * len(_SMALL.window_days)
    json.dumps(results)
    assert "decision" in markdown(results)


def test_a_resumed_run_reuses_the_journal_and_refuses_a_different_plan(x64, tmp_path) -> None:
    journal = tmp_path / "journal.jsonl"
    first = run_experiment(_surrogate(), _SMALL, journal=journal, say=lambda _line: None)
    lines = journal.read_text().splitlines()
    second = run_experiment(_surrogate(), _SMALL, journal=journal, say=lambda _line: None)
    assert journal.read_text().splitlines() == lines  # nothing re-run, nothing appended
    assert second["verdict"] == first["verdict"]
    with pytest.raises(RuntimeError, match="not reproducing"):
        run_experiment(_surrogate(seed=1), _SMALL, journal=journal, say=lambda _line: None)


def test_main_refuses_without_a_service(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["boptest_capped", "--url", "http://127.0.0.1:1"])
    with pytest.raises((RuntimeError, SystemExit)):
        boptest_capped.main()


@pytest.mark.skipif(not _URL, reason="set BOPTEST_URL to a running BOPTEST-Service")
def test_live_oracle_rounds_reach_the_emulator() -> None:
    if not _URL or not is_available(_URL):
        pytest.skip("BOPTEST_URL is set but the service is unreachable")
    client = BOPTestClient(_URL, timeout=300.0)
    model = ZoneModel(0.43, -0.027, (0.05, 0.1, 0.0), (1.0, 0.1, 18.0), (4.0, 0.1, 3.0), 0.6, 0.2)
    arm = Arm("none", 0.0, 0.0, 0.6, 1.0, 0.2, 0.0, fixed=0.6)
    testid = client.select("bestest_hydronic_heat_pump")
    try:
        client.set_step(testid, 7200.0)
        trace = run_episode(
            client, testid, model, Design(), arm, start_day=1, signs=np.ones(3), rounds=3
        )
    finally:
        client.stop(testid)
    assert trace.next_temp.shape == (3,)
    assert np.all((trace.planned >= 0.0) & (trace.planned <= 1.0))

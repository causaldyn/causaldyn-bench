"""P3.3's post-hoc questions offline: the anchoring, and a curve centred where it should be."""

import json
import math
import sys
from dataclasses import asdict

import numpy as np
import pytest

from causaldyn_bench import boptest_capped_posthoc
from causaldyn_bench.boptest_capped import Arm, Design, SurrogatePlant, ZoneModel, loss, run_episode
from causaldyn_bench.boptest_capped_posthoc import (
    by_prior_sign,
    cost_summary,
    curve_summary,
    fixed_estimate_curve,
    markdown,
    paired_estimates,
    probe_cost,
)

_EFFECT = 0.6
_SMALL = Design(rounds=24, window_days=(1, 3), bootstrap=500)


def _plant() -> SurrogatePlant:
    """A noiseless surrogate: the controller's round model is the plant, exactly."""
    rounds = 240
    hours = 2.0 * np.arange(rounds)
    outdoor = 3.0 + 4.0 * np.sin(2.0 * np.pi * hours / 24.0)
    solar = 0.3 * np.clip(np.sin(2.0 * np.pi * (hours - 6.0) / 24.0), 0.0, None)
    truth = ZoneModel(0.45, -0.03, (0.02, 0.5, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.5, 0.1)
    return SurrogatePlant(
        model=truth,
        effect=_EFFECT,
        covariates=np.column_stack([outdoor, solar, np.full(rounds, 21.0)]),
        temps=np.full(rounds, 20.0),
        noise=np.zeros(rounds),
        step_s=7200.0,
    )


def _fixed_loss(plant: SurrogatePlant, value: float, day: int) -> float:
    arm = Arm("none", 0.0, 0.0, value, 1.0, 0.1, 0.0, fixed=value)
    ones = np.ones(_SMALL.rounds)
    return loss(
        run_episode(plant, "s", plant.model, _SMALL, arm, start_day=day, signs=ones), _SMALL
    )


def _scored(plant: SurrogatePlant) -> dict:
    """A results file whose oracle and lazy episodes are this plant's own, as the scored run's are
    the emulator's; the lazy arms' estimates are computed the way ``prior_for`` computes them."""
    delta = _SMALL.prior_spread * _EFFECT
    oracle = [_fixed_loss(plant, _EFFECT, day) for day in _SMALL.window_days]
    lazy = []
    for w, day in enumerate(_SMALL.window_days):
        for sign in (1, -1):
            value = _fixed_loss(plant, _EFFECT + sign * delta, day)
            lazy.append({"window": w, "sign": sign, "loss": value, "regret": value - oracle[w]})
    return {
        "precision": "float64",
        "verdict": {"decision": "REFUTED"},
        "model": asdict(plant.model),
        "reference": {"effect": _EFFECT, "noise_sd": 0.1},
        "oracle": [
            {"window": w, "day": day, "loss": value}
            for w, (day, value) in enumerate(zip(_SMALL.window_days, oracle, strict=True))
        ],
        "lazy": lazy,
        "pairs": [],
        "plan": {"mass": 0.2, "scale": 0.05, "curvature": (_SMALL.dt * _EFFECT) ** 2},
    }


def _pair(window: int, seed: int, taper: tuple[float, float], block: tuple[float, float]) -> dict:
    return {
        "window": window,
        "seed": seed,
        "sign": 1 if seed % 2 == 0 else -1,
        "taper": {"regret": taper[0], "estimate": taper[1]},
        "block": {"regret": block[0], "estimate": block[1]},
    }


def test_the_paired_estimate_is_taper_minus_block_and_splits_by_prior_side() -> None:
    results = {
        "reference": {"effect": 0.6},
        "pairs": [
            _pair(0, 0, (1.0, 0.55), (3.0, 0.65)),
            _pair(0, 1, (2.0, 0.50), (1.0, 0.52)),
            _pair(1, 0, (1.5, 0.60), (2.5, 0.70)),
            _pair(1, 1, (0.5, 0.58), (0.6, 0.57)),
        ],
        "lazy": [{"sign": 1, "regret": 9.0}, {"sign": -1, "regret": 4.0}],
    }
    paired = paired_estimates(results, _SMALL)
    assert paired["pairs"] == 4
    assert math.isclose(paired["mean"], (-0.10 - 0.02 - 0.10 + 0.01) / 4)
    assert paired["taper_lower"] == 3
    above, below = by_prior_sign(results)
    assert (above["sign"], above["pairs"], above["taper_wins"]) == (1, 2, 2)
    assert math.isclose(above["gap"], -1.5) and math.isclose(below["gap"], 0.45)
    assert (above["lazy_regret"], below["lazy_regret"]) == (9.0, 4.0)


def test_the_curve_is_centred_on_the_truth_when_the_plant_is_the_model() -> None:
    """On a plant that IS the controller's model, the fixed-estimate regret is zero at the true
    effect and positive on both sides of it -- so a live curve lowest elsewhere is the plant's
    doing, not this routine's."""
    plant = _plant()
    rows = fixed_estimate_curve(
        plant, _scored(plant), _SMALL, (0.5, 0.8, 1.0, 1.25, 1.5), say=lambda _line: None
    )
    summary = curve_summary(rows)
    assert summary["lowest_mean"] == 1.0
    assert summary["lowest_by_window"] == [1.0, 1.0]
    assert all(row["regret"] > 0.0 for row in rows if row["multiplier"] != 1.0)
    assert all(row["regret"] == 0.0 for row in rows if row["multiplier"] == 1.0)
    assert [row["anchored"] for row in rows[:5]] == [True, False, True, False, True]


def test_a_curve_that_does_not_reproduce_the_scored_episodes_refuses() -> None:
    plant = _plant()
    results = _scored(plant)
    results["lazy"][1]["loss"] += 1e-6  # window 0, prior below: the m = 0.5 anchor
    with pytest.raises(RuntimeError, match="does not reproduce"):
        fixed_estimate_curve(plant, results, _SMALL, (0.5, 1.0), say=lambda _line: None)
    with pytest.raises(ValueError, match="m = 1"):
        fixed_estimate_curve(plant, results, _SMALL, (0.5, 1.5), say=lambda _line: None)


def test_a_probe_on_the_true_effect_costs_the_model_a_when_nothing_is_learned() -> None:
    """With the estimate held at the truth the deadbeat cancels a probe's deviation the round
    after, so on a plant that IS the model the block's cost per unit energy is exactly
    ``A = (dt theta)^2``; the estimate never moves off the value it was held at; and each fixed
    estimate is charged against its OWN scored no-probe episode, the lazy arm's for a prior."""
    plant = _plant()
    results = _scored(plant)
    say = lambda _line: None  # noqa: E731
    rows = probe_cost(plant, results, _SMALL, say=say)
    model_a = results["plan"]["curvature"]
    for row in rows:
        assert row["drift"] < 1e-9
        if row["multiplier"] == 1.0 and row["schedule"] == "block":
            assert math.isclose(row["per_energy"], model_a, rel_tol=1e-9)
    summary = cost_summary(rows, model_a)
    assert [row["multiplier"] for row in summary] == [0.5, 1.0, 1.5]
    assert all(row["episodes"] == 4 and row["model"] == model_a for row in summary)
    results["lazy"][0]["loss"] += 1.0  # window 0, prior above: the m = 1.5 episode's baseline
    shifted = probe_cost(plant, results, _SMALL, (1.5,), (0,), say=say)
    before = [r for r in rows if r["multiplier"] == 1.5 and r["window"] == 0 and r["seed"] == 0]
    assert [a["cost"] - b["cost"] for a, b in zip(before, shifted[:2], strict=True)] == [
        pytest.approx(1.0),
        pytest.approx(1.0),
    ]
    with pytest.raises(ValueError, match="no scored no-probe episode"):
        probe_cost(plant, results, _SMALL, (0.8,), say=say)


def test_the_post_hoc_file_renders_and_serialises() -> None:
    plant = _plant()
    results = _scored(plant)
    results["pairs"] = [_pair(0, 0, (1.0, 0.55), (3.0, 0.65)), _pair(1, 1, (2.0, 0.5), (1.0, 0.6))]
    post = boptest_capped_posthoc.analyse(plant, results, _SMALL, say=lambda _line: None)
    text = markdown(post)
    assert "POST HOC" in text and "| 1 |" in text and "one constant A" in text
    assert [row["window"] for row in post["by_window"]] == [0, 1]
    assert post["by_window"][0]["taper_wins"] == 1 and post["by_window"][1]["taper_wins"] == 0
    json.dumps(post)
    offline = boptest_capped_posthoc.analyse(None, results, _SMALL)
    assert offline["summary"] is None and offline["cost_summary"] is None
    assert "regret lowest" not in markdown(offline)


def test_main_refuses_without_a_service(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["boptest_capped_posthoc", "--url", "http://127.0.0.1:1"])
    with pytest.raises((RuntimeError, SystemExit)):
        boptest_capped_posthoc.main()

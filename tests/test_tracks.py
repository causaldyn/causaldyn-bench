"""Smoke tests for the five tracks (small budgets; full run_all is a deliberate heavier call)."""

import pytest

from causaldyn_bench.adaptive_cv import AdaptiveCVTask
from causaldyn_bench.interference import ZoneIncentiveGame
from causaldyn_bench.leaderboard import format_leaderboard, save_results, to_frame, to_markdown
from causaldyn_bench.tracks import (
    TrackResult,
    fit_dynamics_models,
    track_a_onestep,
    track_c_effect,
    track_d_control,
    track_e_systems,
)


def test_track_c_effect_ranks_causal_above_naive() -> None:
    results = {r.method: r for r in track_c_effect(n=5000)}
    assert results["backdoor-CHC"].value < results["naive"].value  # closer to the true effect
    assert results["double-ml-CHC"].value < results["naive"].value


def test_dynamics_tracks_run_on_a_small_budget() -> None:
    models = fit_dynamics_models(steps=50, n=400)
    a_methods = {r.method for r in track_a_onestep(models)}
    assert a_methods >= {"known-only", "hybrid-CHC", "dlm"}  # DLM baseline is always present
    latencies = track_e_systems(models, steps=5)
    assert all(r.value > 0.0 for r in latencies)  # a positive solve time


def test_save_results_writes_a_snapshot(tmp_path) -> None:
    rows = [
        TrackResult("A-onestep", "m1", "rmse", 0.1),
        TrackResult("A-onestep", "m2", "rmse", 0.2),
    ]
    assert "# causaldyn-bench leaderboard" in to_markdown(rows)
    out = save_results(rows, out_dir=str(tmp_path / "results"))
    assert (out / "leaderboard.md").exists()
    assert (out / "leaderboard.json").exists()


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_to_frame_carries_the_same_records_into_either_backend(backend) -> None:
    rows = [
        TrackResult("A-onestep", "m1", "rmse", 0.1),
        TrackResult("C-effect", "m2", "abs-error", 0.2, lower_is_better=False),
    ]
    frame = to_frame(rows, backend=backend)
    assert list(frame.columns) == ["track", "method", "metric", "value", "lower_is_better"]
    assert [str(v) for v in frame["method"]] == ["m1", "m2"]
    assert float(frame["value"][0]) == 0.1


def test_to_frame_rejects_an_unknown_backend() -> None:
    with pytest.raises(ValueError, match="got 'duckdb'"):
        to_frame([], backend="duckdb")  # type: ignore[arg-type]


def test_adaptive_cv_mpc_beats_priority_blind_myopic() -> None:
    results = {r.method: r for r in AdaptiveCVTask(horizon=14).run(mpc_steps=150)}
    assert (
        results["CHC-MPC"].value < results["myopic"].value
    )  # planning beats load-proportional myopic
    assert results["CHC-MPC"].value <= results["uniform"].value + 1e-6


def test_interference_naive_uplift_loses_to_equilibrium_aware() -> None:
    results = {r.method: r for r in ZoneIncentiveGame(n_zones=6).run(steps=200)}
    # under interference the SUTVA-naive uplift over-allocates and is even worse than doing nothing:
    assert results["equilibrium-CHC"].value < results["naive-uplift"].value
    assert results["naive-uplift"].value > results["no-incentive"].value


def test_track_d_control_and_leaderboard_render() -> None:
    results = track_d_control()
    assert any(r.method.startswith("pricing/") for r in results)
    text = format_leaderboard(results)
    assert "D-control" in text


def test_track_i_scores_the_assumption_and_pessimism_is_not_free() -> None:
    from causaldyn_bench.sensitivity import track_sensitivity

    scores = {r.method: r.value for r in track_sensitivity(n_steps=20)}
    ce = scores["certainty-equivalence"]
    calibrated = scores["robust-CHC (Gamma=2.5)"]
    assert calibrated < ce  # hedging the identification radius bounds the downside
    assert scores["robust-CHC (Gamma=1.3, under)"] > calibrated  # too little hedge leaves cost
    assert scores["robust-CHC (Gamma=6.0, over)"] > calibrated  # too much hedge costs a premium

"""Smoke tests for the five tracks (small budgets; full run_all is a deliberate heavier call)."""

from causaldyn_bench.adaptive_cv import AdaptiveCVTask
from causaldyn_bench.interference import ZoneIncentiveGame
from causaldyn_bench.leaderboard import format_leaderboard, save_results, to_markdown
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

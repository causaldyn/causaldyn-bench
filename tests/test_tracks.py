"""Smoke tests for the five tracks (small budgets; full run_all is a deliberate heavier call)."""

from causaldyn_bench.leaderboard import format_leaderboard
from causaldyn_bench.tracks import (
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
    assert a_methods >= {"known-only", "hybrid-CHC"}
    latencies = track_e_systems(models, steps=5)
    assert all(r.value > 0.0 for r in latencies)  # a positive solve time


def test_track_d_control_and_leaderboard_render() -> None:
    results = track_d_control()
    assert any(r.method.startswith("pricing/") for r in results)
    text = format_leaderboard(results)
    assert "D-control" in text

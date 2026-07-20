"""Run all tracks and render a per-track leaderboard (each track sorted on its own axis)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from causaldyn_bench.adaptive_cv import track_adaptive_cv
from causaldyn_bench.dynamic_effect import track_dynamic_effect
from causaldyn_bench.interference import track_interference
from causaldyn_bench.marketplace import track_marketplace
from causaldyn_bench.structure import track_structure
from causaldyn_bench.tracks import (
    TrackResult,
    fit_dynamics_models,
    track_a_onestep,
    track_b_rollout,
    track_c_effect,
    track_d_control,
    track_e_systems,
)


def run_all(seed: int = 0, steps: int = 1500) -> list[TrackResult]:
    """Run every track once; dynamics models (A/B/E) are fit a single time and shared."""
    models = fit_dynamics_models(seed, steps)
    return [
        *track_a_onestep(models),
        *track_b_rollout(models),
        *track_c_effect(seed),
        *track_d_control(),
        *track_adaptive_cv(),
        *track_interference(),
        *track_marketplace(seed),
        *track_structure(seed),
        *track_dynamic_effect(seed),
        *track_e_systems(models),
    ]


def to_frame(results: list[TrackResult]) -> pd.DataFrame:
    """Tidy DataFrame of results (one row per method/track)."""
    return pd.DataFrame(
        [
            {
                "track": r.track,
                "method": r.method,
                "metric": r.metric,
                "value": r.value,
                "lower_is_better": r.lower_is_better,
            }
            for r in results
        ]
    )


def format_leaderboard(results: list[TrackResult]) -> str:
    """Human-readable leaderboard, grouped by track and sorted on each track's own metric."""
    frame = to_frame(results)
    lines: list[str] = []
    for track in frame["track"].unique():
        sub = frame[frame["track"] == track]
        ascending = bool(sub["lower_is_better"].iloc[0])
        sub = sub.sort_values("value", ascending=ascending)
        direction = "lower" if ascending else "higher"
        lines.append(f"\n[{track}]  (metric: {sub['metric'].iloc[0]}, {direction} is better)")
        for i, (_, row) in enumerate(sub.iterrows()):
            mark = "  <-- best" if i == 0 else ""
            lines.append(f"  {row['method']:<28}{row['value']:>12.4f}{mark}")
    return "\n".join(lines)


def to_markdown(results: list[TrackResult]) -> str:
    """Render the leaderboard as grouped markdown tables (a committable results snapshot)."""
    frame = to_frame(results)
    lines = ["# causaldyn-bench leaderboard", ""]
    for track in frame["track"].unique():
        sub = frame[frame["track"] == track]
        ascending = bool(sub["lower_is_better"].iloc[0])
        sub = sub.sort_values("value", ascending=ascending)
        direction = "lower" if ascending else "higher"
        lines += [f"## {track}  (metric: {sub['metric'].iloc[0]}, {direction} is better)", ""]
        lines += ["| rank | method | value |", "|---|---|---|"]
        for i, (_, row) in enumerate(sub.iterrows()):
            best = " **(best)**" if i == 0 else ""
            lines.append(f"| {i + 1} | {row['method']}{best} | {row['value']:.4f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def save_results(results: list[TrackResult], out_dir: str = "results") -> Path:
    """Write leaderboard.md and leaderboard.json to ``out_dir`` (the committable snapshot)."""
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "leaderboard.md").write_text(to_markdown(results))
    rows = to_frame(results).to_dict(orient="records")
    (out / "leaderboard.json").write_text(json.dumps(rows, indent=2) + "\n")
    return out

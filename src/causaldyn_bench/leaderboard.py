"""Run all tracks and render a per-track leaderboard (each track sorted on its own axis)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Literal

from causaldyn_bench.adaptive_cv import track_adaptive_cv
from causaldyn_bench.delay_identification import track_delay_identification
from causaldyn_bench.dynamic_effect import track_dynamic_effect
from causaldyn_bench.interference import track_interference
from causaldyn_bench.marketplace import track_marketplace
from causaldyn_bench.sensitivity import track_sensitivity
from causaldyn_bench.shooting import track_planner
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

Backend = Literal["pandas", "polars"]


def run_all(seed: int = 0, steps: int = 1500) -> list[TrackResult]:
    """Run every track once; dynamics models (A/B/E) are fit a single time and shared."""
    models = fit_dynamics_models(seed, steps)
    return [
        *track_a_onestep(models),
        *track_b_rollout(models),
        *track_c_effect(seed),
        *track_d_control(),
        *track_planner(models),
        *track_adaptive_cv(),
        *track_interference(),
        *track_marketplace(seed),
        *track_sensitivity(seed),
        *track_structure(seed),
        *track_dynamic_effect(seed),
        *track_delay_identification(seed),
        *track_e_systems(models),
    ]


def _rows(results: list[TrackResult]) -> list[dict[str, Any]]:
    """One flat record per result: the tidy shape both frame backends and the JSON snapshot take."""
    return [
        {
            "track": r.track,
            "method": r.method,
            "metric": r.metric,
            "value": r.value,
            "lower_is_better": r.lower_is_better,
        }
        for r in results
    ]


def _by_track(results: list[TrackResult]) -> list[tuple[str, list[TrackResult]]]:
    """Tracks in first-appearance order, each sorted on the direction its own metric is read in.

    Plain Python rather than a group-by, because the renderers below are the only consumers and a
    frame here would have to commit to one of the two libraries this module supports: polars'
    ``unique`` does not preserve order, so the two would not even agree on how tracks are laid out.
    """
    grouped: dict[str, list[TrackResult]] = {}
    for r in results:
        grouped.setdefault(r.track, []).append(r)
    return [
        (track, sorted(rows, key=lambda r: r.value, reverse=not rows[0].lower_is_better))
        for track, rows in grouped.items()
    ]


def to_frame(results: list[TrackResult], backend: Backend = "pandas") -> Any:
    """Tidy frame of results (one row per method/track), built with ``backend``.

    The import is lazy and by name because both libraries construct the same way from records:
    rendering a leaderboard needs neither installed, and asking for polars does not impose it on
    callers who only wanted markdown.
    """
    if backend not in ("pandas", "polars"):
        msg = f"backend must be 'pandas' or 'polars', got {backend!r}"
        raise ValueError(msg)
    try:
        frames = importlib.import_module(backend)
    except ImportError as exc:
        msg = f"to_frame(backend={backend!r}) needs the {backend} package installed"
        raise ImportError(msg) from exc
    return frames.DataFrame(_rows(results))


def format_leaderboard(results: list[TrackResult]) -> str:
    """Human-readable leaderboard, grouped by track and sorted on each track's own metric."""
    lines: list[str] = []
    for track, rows in _by_track(results):
        direction = "lower" if rows[0].lower_is_better else "higher"
        lines.append(f"\n[{track}]  (metric: {rows[0].metric}, {direction} is better)")
        for i, r in enumerate(rows):
            mark = "  <-- best" if i == 0 else ""
            lines.append(f"  {r.method:<28}{r.value:>12.4f}{mark}")
    return "\n".join(lines)


def to_markdown(results: list[TrackResult]) -> str:
    """Render the leaderboard as grouped markdown tables (a committable results snapshot)."""
    lines = ["# causaldyn-bench leaderboard", ""]
    for track, rows in _by_track(results):
        direction = "lower" if rows[0].lower_is_better else "higher"
        lines += [f"## {track}  (metric: {rows[0].metric}, {direction} is better)", ""]
        lines += ["| rank | method | value |", "|---|---|---|"]
        for i, r in enumerate(rows):
            best = " **(best)**" if i == 0 else ""
            lines.append(f"| {i + 1} | {r.method}{best} | {r.value:.4f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def save_results(results: list[TrackResult], out_dir: str = "results") -> Path:
    """Write leaderboard.md and leaderboard.json to ``out_dir`` (the committable snapshot)."""
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "leaderboard.md").write_text(to_markdown(results))
    (out / "leaderboard.json").write_text(json.dumps(_rows(results), indent=2) + "\n")
    return out

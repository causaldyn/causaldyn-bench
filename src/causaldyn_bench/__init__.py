"""causaldyn-bench: a 5-track benchmark for causal, constrained, dynamical decision-making.

Tracks: A one-step prediction, B long-horizon rollout, C counterfactual effect, D closed-loop
control, E systems performance. Each competitor is scored on the axis it deserves -- trees win A,
the causal hybrid controller wins the decision tracks. Built on ``chc``.
"""

from __future__ import annotations

from causaldyn_bench.adaptive_cv import AdaptiveCVTask, track_adaptive_cv
from causaldyn_bench.dynamic_effect import track_dynamic_effect
from causaldyn_bench.interference import (
    ZoneIncentiveGame,
    interference_report,
    track_interference,
)
from causaldyn_bench.leaderboard import (
    format_leaderboard,
    run_all,
    save_results,
    to_frame,
    to_markdown,
)
from causaldyn_bench.shooting import cross_entropy_control, planner_gap, track_planner
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

__version__ = "0.0.1"

__all__ = [
    "AdaptiveCVTask",
    "TrackResult",
    "ZoneIncentiveGame",
    "__version__",
    "cross_entropy_control",
    "fit_dynamics_models",
    "format_leaderboard",
    "interference_report",
    "planner_gap",
    "run_all",
    "save_results",
    "to_frame",
    "to_markdown",
    "track_a_onestep",
    "track_adaptive_cv",
    "track_b_rollout",
    "track_c_effect",
    "track_d_control",
    "track_dynamic_effect",
    "track_e_systems",
    "track_interference",
    "track_planner",
    "track_structure",
]

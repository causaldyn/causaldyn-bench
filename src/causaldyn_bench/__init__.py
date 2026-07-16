"""causaldyn-bench: a 5-track benchmark for causal, constrained, dynamical decision-making.

Tracks: A one-step prediction, B long-horizon rollout, C counterfactual effect, D closed-loop
control, E systems performance. Each competitor is scored on the axis it deserves -- trees win A,
the causal hybrid controller wins the decision tracks. Built on ``chc``.
"""

from __future__ import annotations

from causaldyn_bench.leaderboard import format_leaderboard, run_all, to_frame
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
    "TrackResult",
    "__version__",
    "fit_dynamics_models",
    "format_leaderboard",
    "run_all",
    "to_frame",
    "track_a_onestep",
    "track_b_rollout",
    "track_c_effect",
    "track_d_control",
    "track_e_systems",
]

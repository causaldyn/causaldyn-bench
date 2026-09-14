"""Track M: allocation over time -- which axis of the media-planning problem actually pays.

The MMM case study (``chc.mmm``) as a scored track. Four rules read one confounded weekly log and
are audited on the true plant at matched budget, so the comparison is about allocation and not
about how much was spent:

* ``CHC-adjusted`` -- the season adjusted for, the whole horizon optimised;
* ``myopic-greedy`` -- the SAME identified fit, spent on this week's return alone;
* ``equal-split``  -- a flat allocation, neither identified nor forward-looking;
* ``naive-MMM``    -- the whole horizon optimised against an observational fit.

That is a 2x2 in (identified?) x (forward-looking?), which is what separates the two claims the
case study makes into two measurements. What it measured is not what the track was built to show.

**On the shipped plant the identification axis pays and the horizon axis does not.** Over eight
seeds, mean lift over doing nothing: ``CHC-adjusted 46.56``, ``myopic-greedy 46.88``,
``equal-split 44.11``, ``naive-MMM 36.70``. Adjusting for the season is worth ``+9.9`` and wins at
**8 of 8** seeds; looking past this week is worth ``-0.3`` and its sign flips **5/3**, with
``|difference| <= 6.2%`` of the mean lift. Both identified rules beat the equal split at **8 of 8**.

**And the design that flips it is one line, which is what makes the null meaningful.** A myopic rule
loses when the carryover ordering CONTRADICTS the immediate one -- not merely because carryover
exists. On the shipped plant ``beta_c/theta_c`` ranks the channels ``(1.29, 1.50, 1.60)`` against
``gamma_c``'s ``(0.50, 0.20, 0.35)``: the two orderings disagree about the top channel but AGREE
about which to drop, and dropping ``social`` is most of the available gain. Re-parameterise so the
best immediate channel is the worst carryover channel -- ``beta/theta`` of ``(0.07, 8.00, 1.60)`` at
unchanged ``gamma``, so the myopic rule's ordering is untouched by construction -- and
``CHC-adjusted`` beats ``myopic-greedy`` at **6 of 6** seeds by ``2.11..3.40`` (``+8.8%`` on the
mean). The equal split then catches the optimiser, because concentration is now the error.

So the transferable reading is: looking ahead buys nothing on its own; it buys the difference
between two orderings, and only when there is one.
"""

from __future__ import annotations

import numpy as np
from chc.mmm import MarketingMixSystem, MmmReport, run_marketing_mix

from causaldyn_bench.tracks import TrackResult

# the library's arm names, and what they are as decision rules
_ARMS: tuple[tuple[str, str], ...] = (
    ("adjusted", "CHC-adjusted"),
    ("myopic", "myopic-greedy"),
    ("flat", "equal-split"),
    ("confounded", "naive-MMM"),
)

# Same gammas as the default, so the myopic rule's channel ordering is unchanged; only the
# carryover is moved, and moved until it contradicts that ordering rather than merely differing.
CARRYOVER_DOMINANT = MarketingMixSystem(beta=(0.1, 1.6, 0.8), theta=(1.5, 0.2, 0.5))


def track_allocation(seed: int = 0, horizon: int = 12) -> list[TrackResult]:
    """Score lift over doing nothing for each allocation rule, audited on the true plant.

    Lift rather than cumulative sales, because cumulative sales is dominated by the do-nothing
    baseline the plant produces on its own and would compress every arm into the last digit; and
    lift rather than lift per unit spend, because under diminishing returns that metric rewards
    under-investment -- the confounded arm scores HIGHER on it while earning less, which is the
    trap ``chc.mmm`` pinned with a test.
    """
    report = run_marketing_mix(horizon=horizon, seed=seed)
    return [
        TrackResult("M-allocation", method, "lift", report.lift(arm), lower_is_better=False)
        for arm, method in _ARMS
    ]


def allocation_report(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
    *,
    horizon: int = 12,
    system: MarketingMixSystem | None = None,
) -> dict[str, dict[str, float]]:
    """Per-arm lift across seeds, plus the head-to-heads the single-seed track cannot resolve.

    ``adjusted_over_myopic`` is the horizon axis and ``adjusted_over_confounded`` the identification
    axis; ``wins`` counts seeds, because a mean over eight draws hides a sign that flips.
    """
    lifts: dict[str, list[float]] = {arm: [] for arm, _ in _ARMS}
    for seed in seeds:
        report: MmmReport = run_marketing_mix(system, horizon=horizon, seed=seed)
        for arm, _ in _ARMS:
            lifts[arm].append(report.lift(arm))

    def head_to_head(left: str, right: str) -> dict[str, float]:
        gap = np.asarray(lifts[left]) - np.asarray(lifts[right])
        return {
            "mean": float(gap.mean()),
            "min": float(gap.min()),
            "max": float(gap.max()),
            "wins": float((gap > 0.0).sum()),
            "seeds": float(gap.size),
        }

    out: dict[str, dict[str, float]] = {
        method: {
            "mean": float(np.mean(lifts[arm])),
            "min": float(np.min(lifts[arm])),
            "max": float(np.max(lifts[arm])),
        }
        for arm, method in _ARMS
    }
    out["adjusted_over_myopic"] = head_to_head("adjusted", "myopic")
    out["adjusted_over_confounded"] = head_to_head("adjusted", "confounded")
    out["myopic_over_flat"] = head_to_head("myopic", "flat")
    out["adjusted_over_flat"] = head_to_head("adjusted", "flat")
    return out

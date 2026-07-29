"""Track I: closed-loop control with no adjustment set, where the assumed sensitivity is the knob.

Every other decision track can be won by estimating better. This one cannot: the confounder is
absent from the log, so no backdoor set, no instrument, and no estimator recovers the effect. The
only remaining lever is the *assumed* sensitivity level ``Gamma`` -- which makes this the one track
that scores a **modelling assumption** rather than a method.

That is also why the axis is the **worst case** over a sweep of true (unknown) confounding
strengths, not the mean. A sensitivity-robust controller does not claim to be cheapest on average;
it claims to bound the downside. Scoring it on the mean would credit it for something it never
promised and hide what it actually buys.

Both directions of miscalibration are on the board on purpose, because the honest failure mode of
this line is a mis-set ``Gamma``, not a mis-fit model: ``Gamma = 1`` is point identification and
reduces to certainty-equivalence exactly, an under-assumed ``Gamma`` hedges too little, and an
over-assumed one pays a premium that shows up as a *worse* worst case than the calibrated setting.
The score is therefore non-monotone in ``Gamma`` -- "more pessimism is better" is a claim this track
is built to refute.

The method lives in ``chc.regret`` (Results 32-38); this track scores it.
"""

from __future__ import annotations

from chc.regret import confounding_robust_tracking_benchmark

from causaldyn_bench.tracks import TrackResult

TRACK = "I-sensitivity"
METRIC = "worst-case closed-loop cost"

# (label, assumed sensitivity level). 2.5 is the calibrated setting; the neighbours are the two
# ways to get it wrong, kept in the leaderboard rather than in prose.
ASSUMPTIONS: tuple[tuple[str, float], ...] = (
    ("robust-CHC (Gamma=1.3, under)", 1.3),
    ("robust-CHC (Gamma=2.5)", 2.5),
    ("robust-CHC (Gamma=6.0, over)", 6.0),
)


def track_sensitivity(seed: int = 0, n_steps: int = 30) -> list[TrackResult]:
    """Score worst-case closed-loop cost for certainty-equivalence and three assumed ``Gamma``."""
    curves = {
        label: confounding_robust_tracking_benchmark(
            sensitivity_gamma=gamma, n_steps=n_steps, seed=seed
        )
        for label, gamma in ASSUMPTIONS
    }
    # the CE arm is the same in every curve (it ignores Gamma), so take it from any of them
    reference = next(iter(curves.values()))
    return [
        TrackResult(TRACK, "certainty-equivalence", METRIC, float(reference.ce_worst_case)),
        *(
            TrackResult(TRACK, label, METRIC, float(curve.robust_worst_case))
            for label, curve in curves.items()
        ),
    ]

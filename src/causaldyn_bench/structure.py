"""Track F: causal-structure recovery under confounding + autocorrelation.

`chc.discovery` vs naive marginal screening, on the two axes plans/17 cares about: **F-structure**
-- F1 of recovering a confounded VAR's true lagged edges (naive over-links via indirect paths); and
**F-payoff** -- the decision that structure *enables*, i.e. the ATE error when the *discovered*
adjustment set is used versus the unadjusted (confounded) estimate. Scoring discovery on the
decision, not just the graph. Both data-generating processes match the gated `chc` tests.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from chc import discover_lagged_parents, estimate_control_effect, partial_corr_test

from causaldyn_bench.tracks import TrackResult

# A sparse VAR(2) over 4 variables with known edges (target, source, lag).
_COEF = {
    (0, 0, 1): 0.5,
    (0, 1, 1): 0.3,
    (1, 1, 1): 0.5,
    (1, 2, 2): 0.4,
    (2, 2, 1): 0.6,
    (3, 3, 1): 0.5,
    (3, 0, 1): 0.35,
}
_TRUE_EDGES = set(_COEF)


def _simulate_var(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 4))
    for t in range(2, n):
        for j in range(4):
            x[t, j] = sum(c * x[t - lag, i] for (jj, i, lag), c in _COEF.items() if jj == j)
            x[t, j] += 0.3 * rng.standard_normal()
    return x


def _confounded_trajectory(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b_true, c, kappa = 0.5, 1.0, 2.0, -1.5
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.7 * z[t - 1] + rng.standard_normal()
    u = kappa * z + 0.5 * rng.standard_normal(n)
    x = np.zeros(n)
    noise = 0.1 * rng.standard_normal(n)
    for t in range(1, n):
        x[t] = a * x[t - 1] + b_true * u[t - 1] + c * z[t - 1] + noise[t]
    return x, u, z


def _f1(found: set, truth: set) -> float:
    true_positive = len(found & truth)
    precision = true_positive / len(found) if found else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _naive_edges(x: np.ndarray, max_lag: int, alpha: float) -> set:
    """Mark an edge whenever the marginal (unconditioned) lagged correlation is significant."""
    n, d = x.shape
    edges = set()
    for j in range(d):
        target = x[max_lag:n, j]
        for i in range(d):
            for lag in range(1, max_lag + 1):
                if float(partial_corr_test(x[max_lag - lag : n - lag, i], target)[1]) < alpha:
                    edges.add((j, i, lag))
    return edges


def track_structure(seed: int = 0) -> list[TrackResult]:
    """Score edge recovery (F1) and the downstream payoff of the discovered adjustment set."""
    max_lag, alpha = 3, 0.01
    series = _simulate_var(3000, seed)
    var_graph = discover_lagged_parents(series, max_lag=max_lag)
    discovered = {(t, s, lag) for t, s, lag, _ in var_graph.edges()}
    naive = _naive_edges(series, max_lag, alpha)
    discovered_f1, naive_f1 = _f1(discovered, _TRUE_EDGES), _f1(naive, _TRUE_EDGES)
    results = [
        TrackResult("F-structure", "naive-correlation", "edge_f1", naive_f1, False),
        TrackResult("F-structure", "chc-discovery", "edge_f1", discovered_f1, False),
    ]

    x, u, z = _confounded_trajectory(4000, seed)
    conf_graph = discover_lagged_parents(np.column_stack([x, z]), np.column_stack([u]), max_lag=1)
    names = ["x", "z"]
    edges = conf_graph.edges()
    parents = {names[s] for target, s, _lag, kind in edges if target == 0 and kind == "state"}
    adjust = tuple(p for p in sorted(parents) if p != "x")  # confounders = state parents minus x
    columns = {"x": x[:-1], "u": u[:-1], "z": z[:-1], "x_next": x[1:]}
    data = {name: jnp.asarray(value) for name, value in columns.items()}
    b_true = 1.0
    naive_error = abs(float(estimate_control_effect(data, adjust_for=())) - b_true)
    adjusted_error = abs(float(estimate_control_effect(data, adjust_for=adjust)) - b_true)
    results += [
        TrackResult("F-payoff", "naive-unadjusted", "ate_error", naive_error),
        TrackResult("F-payoff", "chc-discovery-adjusted", "ate_error", adjusted_error),
    ]
    return results

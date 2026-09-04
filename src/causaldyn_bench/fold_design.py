"""Track N: which cross-fitting FOLDS to use under network interference, and when it matters.

Every other track asks which *method* to run. This one holds the method fixed -- the same
cross-fitted DML for a direct and a spillover effect, `chc.network_causal.estimate_network_effects`
-- and varies only the split. Five arms: random rows, random whole units, contiguous graph blocks,
Emmenegger-style neighbour exclusion, and the variance-optimal design of Result 52
(`chc.regret.optimal_fold_partition`).

**The result is an ordering, and the useful half of it is negative.** Measured on `C_12` over 120
draws at `g = 2` clusters, MSE relative to the random-unit split:

    arm                  direct   spillover
    random rows          0.861    0.976
    random units         1.000    1.000     <- baseline
    designed (Result 52) 0.906    0.995
    contiguous blocks    1.370    1.358
    neighbour exclusion  1.666    1.612

The designed split and the graph-blind unit split **tie**. What separates is the two arms a
practitioner would actually reach for: keeping neighbours together costs **+37%** MSE, and dropping
them from the training set costs **+66%**. The design law explains all three positions from one
number -- the fraction of edges left inside a fold: `0.50` designed, `0.46` random units, `0.83`
contiguous. Its value here is that it **convicts the obvious split before any simulation**, from a
trace computation on the graph.

Three further measurements.

1. *The law forecasts WHICH topology has a design effect, and it is right on all three.* Its
   cross-sectional mass ratio designed/contiguous is `0.720` on the cycle, `0.974` on the 3x4
   torus and `0.966` on a random cubic graph -- a real gap on one, almost none on the other two.
   Measured over 120 draws at each of `g = 2, 4, 8, 20`: on the cycle every arm separates in the
   predicted order; on the torus and the cubic graph every arm scatters inside `0.85-1.16` with no
   pattern in `g` or in the arm. The size is right too: the panel functional puts the designed
   split at `0.588` of the contiguous split's variance and the realised MSE ratio at `g = 2` is
   `0.66` (direct) and `0.73` (spillover) -- same sign, same order, functional optimistic by 7-14
   points. **The law tells you when fold design is worth doing, and on two of these three
   topologies the honest answer is "it is not".**

2. *It is an `O(1/g)` effect in the number of independent clusters* (Result 60), visible at
   estimator level rather than only in the functional. Contiguous-vs-random-units on `C_12` across
   `g = 2, 4, 8, 20`: `1.370, 1.139, 1.083, 1.047` (direct) and `1.357, 1.067, 1.089, 1.073`
   (spillover); neighbour exclusion `1.666, 1.240, 1.241, 1.078`. A bad split costs 37% more MSE
   with two clusters and 5% with twenty, and the ORDERING never changes -- which is Result 60's
   `ordering_is_cluster_invariant` showing up in a real fit. Fold design is a
   **small-cluster-count** instrument: the realistic regime for an experiment run over a handful
   of cities, and exactly not the regime of a simulation with twenty independent replicas.

3. *Neighbour exclusion is dominated where it runs and infeasible where it does not.* It is the
   worst arm on the cycle at every one of the four cluster counts, and on the torus and the cubic
   graph at `K = 2` it cannot run at all: the test fold's hop-1 neighbourhood covers the training
   fold, so the rule empties it.
   Buying validity by discarding data needs a split that is already graph-aware -- that is, it
   needs the design it was meant to replace.

Result 54's exact moment was checked against the plug law on this geometry and **moves the level
without moving the decision**: the exact `E[X/Y^2]` is `2.3x` the plug-in value for both designs,
and their ratio moves from `0.588` to `0.597`. The Jensen gap nearly cancels in the design ratio
here -- a statement about this operator pair, not a general one (Result 51 (m)).

Standard errors are cluster-robust on `cid`, and they have to be: clustered / i.i.d. is `1.87` on
the spillover coefficient and `0.93` on the direct one, because the exposure is a shell sum while
the treatment's exogenous part is i.i.d. across units. One i.i.d. SE for both would understate
exactly the coefficient interference is about.

Scope: synthetic outcomes throughout, on `chc.network_causal.DelayedNetworkPanel`, `K = 2`,
`m = 12`, `p = 16`, `phi = 0.6`, `lag = 1`, one propagated disturbance in the outcome (without it
the score is white and no split can move anything). The three topologies are a cycle, a torus and a
random cubic graph; a real road or power-grid topology is a data-licensing decision and is
deliberately absent. See plans/24 P2.2.
"""

from __future__ import annotations

import jax
import numpy as np
from chc.network_causal import (
    DelayedNetworkPanel,
    cycle_shells,
    estimate_network_effects,
    graph_shells,
    torus_adjacency,
)
from chc.regret import optimal_fold_partition

from causaldyn_bench.tracks import TrackResult

_M = 12  # units per cluster
_K = 2  # folds: r = K/(K-1) is largest here, so the design has the most leverage
_TIMES = 16
_PHI, _LAG, _GAMMAS = 0.6, 1, (1.0, 0.7, 0.4)
_DISTURBANCE = 2.0  # a propagated shock in the outcome: without it the score is white
_TRUE = {"direct": 1.0, "spillover": 0.6}


def random_regular_graph(m: int, degree: int, seed: int) -> tuple[tuple[int, ...], ...]:
    """A random ``degree``-regular simple graph on ``m`` vertices, by configuration-model retry.

    Regular because the panel's ``neighbours`` column is rectangular; random because the cycle and
    the torus are both vertex-transitive and a third topology that is not is what tests whether the
    design law needs the symmetry it is stated with.
    """
    if degree >= m:
        raise ValueError(f"a simple graph on {m} vertices cannot have degree {degree}")
    if (m * degree) % 2:
        raise ValueError(f"no {degree}-regular graph on {m} vertices: the degree sum is odd")
    rng = np.random.default_rng(seed)
    for _ in range(4000):
        stubs = np.repeat(np.arange(m), degree)
        rng.shuffle(stubs)
        adjacency = np.zeros((m, m), dtype=int)
        for i in range(0, stubs.size, 2):
            u, v = int(stubs[i]), int(stubs[i + 1])
            if u == v or adjacency[u, v]:
                break
            adjacency[u, v] = adjacency[v, u] = 1
        else:
            if np.all(adjacency.sum(axis=1) == degree):
                return tuple(tuple(int(x) for x in row) for row in adjacency)
    raise RuntimeError(f"no {degree}-regular graph on {m} vertices after 4000 draws")


def _topologies() -> dict[str, tuple[tuple[tuple[int, ...], ...] | None, np.ndarray]]:
    cubic = random_regular_graph(_M, 3, 5)
    torus = torus_adjacency(3, 4)
    return {
        "cycle": (None, np.rint(cycle_shells(_M, 1)[1]).astype(int)),
        "torus": (torus, np.asarray(torus)),
        "cubic": (cubic, np.asarray(cubic)),
    }


def _same_fold_mass(shells: list[np.ndarray], fold: np.ndarray) -> float:
    """The partition-dependent part of Psi -- Result 52's objective, and Result 61's max-cut."""
    x = _PHI**_LAG
    q = sum(
        _GAMMAS[d] * _GAMMAS[e] * x ** abs(d - e) * (shells[d] @ shells[e])
        for d in range(3)
        for e in range(3)
    )
    same = fold[:, None] == fold[None, :]
    return float(q[same].sum())


def _errors(
    graph: tuple[tuple[int, ...], ...] | None,
    fold: np.ndarray | None,
    exclude: bool,
    clusters: int,
    seeds: int,
) -> dict[str, np.ndarray] | None:
    """Signed errors of both coefficients over ``seeds`` draws, or None if the arm cannot run."""
    panel = DelayedNetworkPanel(
        n_clusters=clusters,
        cluster_size=_M,
        n_times=_TIMES,
        graph=graph,
        lag=_LAG,
        phi=_PHI,
        gammas=_GAMMAS,
        disturbance_scale=_DISTURBANCE,
    )
    out: dict[str, list[float]] = {name: [] for name in _TRUE}
    for seed in range(seeds):
        data = panel.sample(jax.random.key(seed))
        groups = None if fold is None else fold[np.asarray(data["unit"])]
        try:
            fit = estimate_network_effects(
                data, folds=_K, fold_groups=groups, exclude_neighbours=exclude
            )
        except ValueError:
            return None
        for name, truth in _TRUE.items():
            out[name].append(fit[name] - truth)
    return {name: np.asarray(values) for name, values in out.items()}


def track_fold_design(clusters: int = 2, seeds: int = 120) -> list[TrackResult]:
    """Score the four fold schemes on the cycle, in the small-cluster regime where design bites.

    The baseline is the graph-blind split that keeps each unit intact -- ``random units`` -- rather
    than the row permutation ``estimate_network_effects`` defaults to, because a row permutation
    puts a unit's own history on both sides of the split and the question here is which *valid*
    split to use. (Measured, the two are within 2% of each other on this DGP at ``g >= 8``, so the
    choice of baseline changes the table's zero and not its ordering.)

    ``clusters = 2`` is deliberate: the design effect is ``O(1/g)`` (Result 60), so scoring at
    ``g = 20`` would report near-ties and call the question settled.

    120 draws is what the two large gaps need to be stable; it does NOT resolve designed against
    random units, which differ by 5-10%. The leaderboard prints a rank because every track does,
    and those two rows should be read as the tie the module docstring reports, not as an ordering.
    """
    graph, adjacency = _topologies()["cycle"]
    shells = graph_shells(adjacency.astype(float), 2)
    designed = optimal_fold_partition(shells, _GAMMAS, _PHI, lag=_LAG, k_folds=_K).fold
    blocks = np.arange(_M) * _K // _M

    arms: dict[str, tuple[str, bool]] = {
        "random rows": ("rows", False),
        "random units": ("units", False),
        "contiguous folds": ("blocks", False),
        "neighbour exclusion": ("blocks", True),
        "designed folds (chc)": ("designed", False),
    }
    layouts = {"rows": None, "units": np.arange(_M), "blocks": blocks, "designed": designed}
    errors = {
        name: _errors(graph, layouts[layout], exclude, clusters, seeds)
        for name, (layout, exclude) in arms.items()
    }
    baseline = errors["random units"]
    assert baseline is not None

    results: list[TrackResult] = []
    for name, err in errors.items():
        for coefficient in _TRUE:
            if err is None:
                value = float("inf")  # the arm could not run: exclusion emptied a training fold
            else:
                value = float((err[coefficient] ** 2).mean() / (baseline[coefficient] ** 2).mean())
            results.append(
                TrackResult(
                    track=f"N-fold-{coefficient}",
                    method=name,
                    metric=f"MSE / random unit folds ({clusters} clusters)",
                    value=value,
                )
            )
    return results


def fold_design_report(
    cluster_grid: tuple[int, ...] = (2, 4, 8, 20), seeds: int = 120
) -> dict[str, dict[str, float]]:
    """The full three-topology, four-cluster-count sweep behind this module's docstring.

    Not part of the leaderboard -- it takes minutes -- but it is what the claims are measured on,
    and it reports the design law's own predicted gap beside the realised one so the two can be
    compared rather than asserted.
    """
    report: dict[str, dict[str, float]] = {}
    for topology, (graph, adjacency) in _topologies().items():
        shells = graph_shells(adjacency.astype(float), 2)
        designed = optimal_fold_partition(shells, _GAMMAS, _PHI, lag=_LAG, k_folds=_K).fold
        blocks = np.arange(_M) * _K // _M
        entry = {
            "predicted_ratio": _same_fold_mass(shells, designed) / _same_fold_mass(shells, blocks)
        }
        for clusters in cluster_grid:
            base = _errors(graph, np.arange(_M), False, clusters, seeds)
            assert base is not None
            for label, fold, exclude in (
                ("rows", None, False),
                ("contiguous", blocks, False),
                ("exclusion", blocks, True),
                ("designed", designed, False),
            ):
                err = _errors(graph, fold, exclude, clusters, seeds)
                for coefficient in _TRUE:
                    key = f"{label}/{coefficient}/g{clusters}"
                    entry[key] = (
                        float("inf")
                        if err is None
                        else float((err[coefficient] ** 2).mean() / (base[coefficient] ** 2).mean())
                    )
        report[topology] = entry
    return report

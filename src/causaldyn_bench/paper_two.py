"""Every table in paper P2, "Fold design for cross-fitting on networks and panels".

One command produces all four.

    uv run python -m causaldyn_bench.paper_two --out results/paper2

The measurements themselves live in :mod:`causaldyn_bench.fold_design` (Track N) and in
``chc.regret``; nothing is recomputed here with a second copy of the DGP. What this module adds is
the two things a manuscript needs and a leaderboard does not:

1. **Intervals.** Every ratio in the paper is a ratio of mean squared errors between two splits of
   the SAME draws, so the interval is a PAIRED percentile bootstrap: resample draw indices once and
   apply the same index set to numerator and denominator. Resampling the two arms independently
   would widen every interval by draw noise that cancels in the ratio, and would turn the
   designed-vs-random tie into an unfalsifiable one.
2. **One convention, stated.** The quadrature table reports RELATIVE max-entry error against the
   exact anchor, and says so in the header. The shipped docstring's table is absolute at ``n = 7``
   and both at ``n = 5``, which is the mistake Result 63 (j) retracted; a paper table that repeats
   it would repeat the retraction.

Draw counts are seeds: draw ``i`` is ``jax.random.key(i)``. The default 120 is what the two large
gaps need to be stable and is deliberately more than the ten a leaderboard row uses -- at ten the
paired interval on the designed-vs-random row spans a factor of two and reports nothing.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from chc.network_causal import graph_shells
from chc.regret import exact_matrix_ratio_moment, optimal_fold_partition

from causaldyn_bench.fold_design import (
    _GAMMAS,
    _K,
    _LAG,
    _M,
    _PHI,
    _TRUE,
    _same_fold_mass,
    _topologies,
    arm_errors,
)

_ARMS: tuple[tuple[str, str, bool], ...] = (
    ("random rows", "rows", False),
    ("random units", "units", False),
    ("contiguous folds", "blocks", False),
    ("neighbour exclusion", "blocks", True),
    ("designed folds", "designed", False),
)


@dataclass(frozen=True)
class RatioCI:
    """An MSE ratio against the random-unit baseline, with a paired bootstrap interval."""

    ratio: float
    lo: float
    hi: float

    def cell(self) -> str:
        if not np.isfinite(self.ratio):
            return "--"
        return f"{self.ratio:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def paired_ratio_ci(
    numerator: np.ndarray, denominator: np.ndarray, *, n_boot: int = 10_000, seed: int = 0
) -> RatioCI:
    """Percentile bootstrap for ``mean(num^2)/mean(den^2)`` under a SHARED resample of draws."""
    if numerator.shape != denominator.shape:
        raise ValueError(
            f"paired arms need equal draws, got {numerator.shape} and {denominator.shape}"
        )
    point = float((numerator**2).mean() / (denominator**2).mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, numerator.size, size=(n_boot, numerator.size))
    ratios = (numerator[idx] ** 2).mean(axis=1) / (denominator[idx] ** 2).mean(axis=1)
    lo, hi = (float(v) for v in np.quantile(ratios, [0.025, 0.975]))
    return RatioCI(point, lo, hi)


def _layouts(adjacency: np.ndarray) -> dict[str, np.ndarray | None]:
    shells = graph_shells(adjacency.astype(float), 2)
    designed = optimal_fold_partition(shells, _GAMMAS, _PHI, lag=_LAG, k_folds=_K).fold
    return {
        "rows": None,
        "units": np.arange(_M),
        "blocks": np.arange(_M) * _K // _M,
        "designed": designed,
    }


def arm_table(
    topology: str, clusters: int, draws: int, n_boot: int
) -> dict[str, dict[str, RatioCI]]:
    """Every arm's MSE ratio against the random-unit split, with paired intervals.

    The body of Table 1, and the honest form of Track N's ordering claim: a TIE is an interval
    covering 1, a COST is an interval strictly above it. A point estimate with a hand-picked
    tolerance can state neither, because the pair the claim calls a tie is not resolved at any
    draw count this benchmark runs.
    """
    graph, adjacency = _topologies()[topology]
    layouts = _layouts(adjacency)
    errors = {
        name: arm_errors(graph, layouts[layout], exclude, clusters, draws)
        for name, layout, exclude in _ARMS
    }
    baseline = errors["random units"]
    if baseline is None:
        raise RuntimeError("the random-unit baseline could not run; every ratio would be undefined")
    out: dict[str, dict[str, RatioCI]] = {}
    for arm, (name, _, _) in enumerate(_ARMS):
        err = errors[name]
        out[name] = {
            coefficient: (
                RatioCI(float("inf"), float("inf"), float("inf"))
                if err is None
                else paired_ratio_ci(
                    err[coefficient], baseline[coefficient], n_boot=n_boot, seed=arm
                )
            )
            for coefficient in _TRUE
        }
    return out


def table_one(draws: int, n_boot: int) -> dict[str, dict[str, RatioCI]]:
    """The arm ordering on the cycle at ``g = 2``, where the design effect is largest."""
    return arm_table("cycle", 2, draws, n_boot)


def table_two(
    cluster_grid: tuple[int, ...], draws: int, n_boot: int
) -> dict[int, dict[str, dict[str, RatioCI]]]:
    """The same arms across cluster counts: the ``O(1/g)`` decay, with intervals on every cell."""
    return {g: arm_table("cycle", g, draws, n_boot) for g in cluster_grid}


def table_three(draws: int, n_boot: int) -> dict[str, dict[str, float | RatioCI]]:
    """The design law as a FORECAST: predicted mass ratio against the realised MSE ratio.

    The predicted column is a trace computation on the graph and carries no draw noise, so it has
    no interval; the realised columns do. The forecast is falsifiable exactly because the predicted
    number is fixed before any panel is drawn.
    """
    out: dict[str, dict[str, float | RatioCI]] = {}
    for topology, (graph, adjacency) in _topologies().items():
        shells = graph_shells(adjacency.astype(float), 2)
        layouts = _layouts(adjacency)
        designed, blocks = layouts["designed"], layouts["blocks"]
        assert designed is not None and blocks is not None
        entry: dict[str, float | RatioCI] = {
            "predicted": _same_fold_mass(shells, designed) / _same_fold_mass(shells, blocks)
        }
        contiguous = arm_errors(graph, blocks, False, 2, draws)
        design = arm_errors(graph, designed, False, 2, draws)
        if contiguous is None or design is None:
            raise RuntimeError(f"an arm failed to run on {topology}; the forecast needs both")
        for coefficient in _TRUE:
            entry[f"realised/{coefficient}"] = paired_ratio_ci(
                design[coefficient], contiguous[coefficient], n_boot=n_boot, seed=7
            )
        out[topology] = entry
    return out


def table_four(node_grid: tuple[int, ...]) -> dict[str, dict[int, float]]:
    """The matrix-moment quadrature, in ONE convention: relative max-entry error at ``q = 3``.

    Two cells, both against an exact anchor. ``n = 5`` is the existence boundary ``n = q + 2``,
    where the answer is ``I``; ``n = 7`` is margin 2, where it is ``I/3`` -- so an absolute column
    would make the second look three times better than it is. The refinement residual of
    ``matrix_ratio_certificate`` is reported as a ratio to the true error in the same convention,
    which is what says whether it majorises (Result 63 (e): always once the per-node rate reaches
    2, and below that only where the error changes sign or moves its largest entry).

    That residual is ``max |value(nodes) - value(nodes - 1)|``, so it is read off the grids this
    table computes anyway rather than by the certificate's second quadrature per row -- at
    ``nodes = 9`` that second pass alone is ``262 144`` points. It needs ``nodes - 1`` in the grid;
    a row without it prints ``--``.
    """
    out: dict[str, dict[int, float]] = {
        "rel_n5": {},
        "rel_n7": {},
        "residual_over_true_n5": {},
        "residual_over_true_n7": {},
    }
    for n in (5, 7):
        want = np.eye(3) / (n - 3 - 1)
        scale = float(np.max(np.abs(want)))
        values = {
            nodes: exact_matrix_ratio_moment(np.eye(n), np.eye(n), np.eye(3 * n), nodes=nodes)
            for nodes in node_grid
        }
        for nodes, got in values.items():
            true = float(np.max(np.abs(got - want)))
            out[f"rel_n{n}"][nodes] = true / scale
            if nodes - 1 in values:
                # both sides are ABSOLUTE max-entry quantities, so the normalisation cancels and
                # the ratio is the same in either convention
                residual = float(np.max(np.abs(got - values[nodes - 1])))
                out[f"residual_over_true_n{n}"][nodes] = residual / true if true else float("inf")
    return out


def _markdown(
    one: dict[str, dict[str, RatioCI]],
    two: dict[int, dict[str, dict[str, RatioCI]]],
    three: dict[str, dict[str, float | RatioCI]],
    four: dict[str, dict[int, float]],
    draws: int,
    n_boot: int,
) -> str:
    lines = [
        "# P2 tables -- fold design for cross-fitting on networks and panels",
        "",
        f"{draws} paired draws per arm, {n_boot} paired bootstrap resamples, 95% percentile "
        "intervals. Ratios are mean squared error against the random-unit split on the same draws.",
        "",
        "## Table 1 -- fold schemes on the cycle, g = 2",
        "",
        "| split | direct | spillover |",
        "|---|---|---|",
    ]
    for name, _, _ in _ARMS:
        lines.append(f"| {name} | {one[name]['direct'].cell()} | {one[name]['spillover'].cell()} |")
    lines += [
        "",
        "## Table 2 -- the design effect is O(1/g)",
        "",
        "| split | " + " | ".join(f"g = {g}" for g in two) + " |",
        "|---|" + "---|" * len(two),
    ]
    for name, _, _ in _ARMS:
        if name == "random units":
            continue
        cells = " | ".join(two[g][name]["direct"].cell() for g in two)
        lines.append(f"| {name} | {cells} |")
    lines += [
        "",
        "Direct coefficient; the spillover column is in the JSON. The ordering is invariant in `g`"
        " (Result 60) while the size decays.",
        "",
        "## Table 3 -- the law forecasts WHICH topology has a design effect",
        "",
        "| topology | predicted mass ratio | realised MSE, direct | realised MSE, spillover |",
        "|---|---|---|---|",
    ]
    for topology, entry in three.items():
        predicted = entry["predicted"]
        assert isinstance(predicted, float)
        direct, spillover = entry["realised/direct"], entry["realised/spillover"]
        assert isinstance(direct, RatioCI) and isinstance(spillover, RatioCI)
        lines.append(f"| {topology} | {predicted:.3f} | {direct.cell()} | {spillover.cell()} |")
    lines += [
        "",
        "Designed against contiguous, so a ratio below 1 is a win for the design.",
        "",
    ]
    return "\n".join(lines + _quadrature_markdown(four))


def _quadrature_markdown(four: dict[str, dict[int, float]]) -> list[str]:
    """Table 4, and every number the text derives from it: rates, digits, residual step ratios."""
    lines = [
        "## Table 4 -- the q = 3 matrix moment: RELATIVE max-entry error against the exact anchor",
        "",
        "| nodes | points | n = 5 (boundary) | rate | residual / true "
        "| n = 7 (margin 2) | rate | residual / true |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rates = {n: _per_node_rate(four[f"rel_n{n}"]) for n in (5, 7)}
    for nodes in sorted(four["rel_n5"]):
        cells = [str(nodes), f"{nodes**6:,}"]
        for n in (5, 7):
            rate = rates[n].get(nodes)
            residual = four[f"residual_over_true_n{n}"].get(nodes)
            cells += [
                f"{four[f'rel_n{n}'][nodes]:.3e}",
                "--" if rate is None else f"{rate:.3f}",
                "--" if residual is None else f"{residual:.2f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Relative throughout: the exact answer is `I` at `n = 5` and `I/3` at `n = 7`, so an"
        " absolute column would flatter the second by 3x (Result 63 (j)). The rate is the per-node"
        " decay `e(nodes - 1) / e(nodes)`. By the reverse triangle inequality residual / true is at"
        " least `|rate - 1|`, with equality when the error keeps its sign and its largest entry: a"
        " rate of 2 always makes the residual majorise the error, and below 2 only a row where that"
        " premise fails can (Result 63 (e)).",
        "",
    ]
    for n in (5, 7):
        errors = four[f"rel_n{n}"]
        first, last = min(errors), max(errors)
        if first == last:
            continue
        mean_rate = (errors[first] / errors[last]) ** (1 / (last - first))
        line = (
            f"- n = {n}: geometric-mean rate {mean_rate:.2f} per node over nodes {first}-{last},"
            f" and {-math.log10(errors[last]):.2f} digits at {last}"
        )
        if mean_rate > 1:
            need = math.ceil(last + math.log10(errors[last] * 1e6) / math.log10(mean_rate))
            line += f"; six digits at that rate need {need} nodes, {need**6:.1e} points"
        residuals = {k: r * errors[k] for k, r in four[f"residual_over_true_n{n}"].items()}
        steps = _per_node_rate(residuals)
        if steps:
            line += (
                ". Residual step ratio `residual(nodes - 1) / residual(nodes)`, which equals the"
                " rate only if the errors are geometric: "
                + ", ".join(f"{v:.2f} at {k}" for k, v in sorted(steps.items()))
            )
        lines += [line + ".", ""]
    return lines


def _per_node_rate(errors: dict[int, float]) -> dict[int, float]:
    """``e(nodes - 1) / e(nodes)`` at every grid whose next-coarser grid was also computed."""
    return {k: errors[k - 1] / e for k, e in errors.items() if k - 1 in errors}


def _jsonable(obj: object) -> object:
    if isinstance(obj, RatioCI):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=120, help="paired draws per arm")
    parser.add_argument("--boot", type=int, default=10_000, help="paired bootstrap resamples")
    parser.add_argument("--clusters", type=int, nargs="+", default=[2, 4, 8, 20])
    parser.add_argument("--nodes", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--out", type=Path, default=Path("results/paper2"))
    args = parser.parse_args()

    one = table_one(args.draws, args.boot)
    two = table_two(tuple(args.clusters), args.draws, args.boot)
    three = table_three(args.draws, args.boot)
    four = table_four(tuple(args.nodes))

    args.out.mkdir(parents=True, exist_ok=True)
    text = _markdown(one, two, three, four, args.draws, args.boot)
    (args.out / "tables.md").write_text(text)
    (args.out / "tables.json").write_text(
        json.dumps(
            {
                "draws": args.draws,
                "boot": args.boot,
                "table1": _jsonable(one),
                "table2": _jsonable(two),
                "table3": _jsonable(three),
                "table4": _jsonable(four),
            },
            indent=2,
        )
    )
    print(text)
    print(f"written to {args.out}/tables.md and {args.out}/tables.json")


if __name__ == "__main__":
    main()

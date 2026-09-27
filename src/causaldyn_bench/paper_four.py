"""Every table in paper P4, "Residuals go blind near degeneracy".

One command produces both.

    uv run python -m causaldyn_bench.paper_four --out results/paper4

Nothing here trains a model with a second copy of the solver: every configuration is one call to
:func:`chc.deep_galerkin.residual_blindness_sweep`, which solves the anti-monotone LQ mean-field
game at horizons approaching the obstruction ``T* = 0.8036`` in one fixed box and scores each solve
three ways -- the raw interior residual, the residual conditioned by ``1/|den(T)|``, and the
dual-weighted estimate built from the model alone. Two decisions a manuscript needs and a
certificate does not:

1. **Table 1 is Result 55 (d) from a committed routine.** The eight-horizon table in the research
   log was first printed by a scratch script with the same solver, seed and box. This regenerates
   it at the routine's defaults, so the paper quotes nothing a reader cannot rerun.
2. **Table 2 reports a distribution, not a best case.** Result 55 measured one seed, one width and
   one optimiser. Here the rank correlations are read over every seed, width and optimiser on the
   command line -- Adam on fresh collocation points every step and L-BFGS on one fixed draw, the
   two a PINN practitioner actually reaches for -- and the table prints the minimum, median and
   maximum of each, and how many configurations put the raw residual on the wrong side of zero.

What would falsify the paper's reading is in the table rather than argued around it: a
configuration where the raw residual ranks the error correctly, or one where the dual-weighted
estimate does not. The proved part is narrower than the measured part, and the tables keep them
apart: a model whose reduced state stays bounded has a residual bounded independently of the
horizon while its error diverges (``proofs/mean_field_dwr.v``, ``bounded_approximator_is_blind``),
so the residual CANNOT rank the error near ``T*``. Whether it ranks it BACKWARDS is the
measurement, and the answer is the optimiser's: Table 2 splits it by optimiser, and prints the
largest ``|S_hat(0)|`` each row reached, which is the theorem's hypothesis read off the models.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

from chc.deep_galerkin import BlindnessSweep, MeanFieldOptimizer, residual_blindness_sweep

_HORIZONS = (0.30, 0.40, 0.50, 0.60, 0.68, 0.72, 0.76, 0.79)

_BUDGETS: dict[MeanFieldOptimizer, int] = {"adam": 2500, "lbfgs": 1000}
"""Iterations per solve. An L-BFGS iteration is a line search over several loss evaluations, and
100 of them already fit the monotone instance inside the certificate's 2% gate."""

_SCORES = ("rank_residual", "rank_conditioned", "rank_dual_weighted")


@dataclass(frozen=True)
class Configuration:
    """One cell of Table 2: which solver produced the sweep, and the sweep."""

    optimizer: MeanFieldOptimizer
    width: int
    seed: int
    sweep: BlindnessSweep

    def rank(self, score: str) -> float:
        return float(getattr(self.sweep, score))


@dataclass(frozen=True)
class Spread:
    low: float
    middle: float
    high: float

    @classmethod
    def of(cls, values: list[float]) -> Spread:
        return cls(min(values), median(values), max(values))

    def cell(self) -> str:
        return f"{self.low:+.3f} / {self.middle:+.3f} / {self.high:+.3f}"


@dataclass(frozen=True)
class Group:
    """Table 2's row: one optimiser at one width, spread over seeds."""

    optimizer: MeanFieldOptimizer
    width: int
    seeds: int
    ranks: dict[str, Spread]
    worst_dual_discrepancy: float
    largest_fitted_slope: float  # max |S_hat(0)| over seeds and horizons: Theorem 1's bound B


@dataclass(frozen=True)
class Summary:
    """``wrong_sign`` counts configurations whose raw residual correlates NEGATIVELY with the error,
    ``not_positive`` those where it fails to correlate positively at all. The second backs "cannot
    rank it", which is also what Theorem 1 proves near ``T*``; the first backs "ranks it
    backwards", which nothing proves and the optimiser decides."""

    groups: list[Group]
    configurations: int
    pooled: dict[str, Spread]
    wrong_sign: int
    not_positive: int
    dual_perfect: int
    worst_dual_discrepancy: float


def table_one(horizons: tuple[float, ...]) -> BlindnessSweep:
    """Result 55 (d) at the routine's defaults: seed 0, width 32, Adam for 2500 steps."""
    return residual_blindness_sweep(horizons)


def table_two(
    horizons: tuple[float, ...], seeds: tuple[int, ...], widths: tuple[int, ...]
) -> list[Configuration]:
    """One sweep per optimiser, width and seed."""
    cells = []
    for optimizer, steps in _BUDGETS.items():
        for width in widths:
            for seed in seeds:
                sweep = residual_blindness_sweep(
                    horizons, width=width, optimizer=optimizer, steps=steps, seed=seed
                )
                cells.append(Configuration(optimizer, width, seed, sweep))
                print(
                    f"{optimizer} width={width} seed={seed}: residual {sweep.rank_residual:+.3f}, "
                    f"conditioned {sweep.rank_conditioned:+.3f}, "
                    f"dual-weighted {sweep.rank_dual_weighted:+.3f}",
                    flush=True,
                )
    return cells


def summarise(cells: list[Configuration]) -> Summary:
    """Minimum, median and maximum of each rank per optimiser and width, and the pooled counts."""
    keys = sorted({(c.optimizer, c.width) for c in cells})
    groups = []
    for optimizer, width in keys:
        members = [c for c in cells if (c.optimizer, c.width) == (optimizer, width)]
        groups.append(
            Group(
                optimizer=optimizer,
                width=width,
                seeds=len(members),
                ranks={s: Spread.of([c.rank(s) for c in members]) for s in _SCORES},
                worst_dual_discrepancy=max(c.sweep.worst_dual_discrepancy for c in members),
                largest_fitted_slope=max(abs(s) for c in members for s in c.sweep.fitted_slopes),
            )
        )
    raw = [c.sweep.rank_residual for c in cells]
    return Summary(
        groups=groups,
        configurations=len(cells),
        pooled={s: Spread.of([c.rank(s) for c in cells]) for s in _SCORES},
        wrong_sign=sum(r < 0.0 for r in raw),
        not_positive=sum(r <= 0.0 for r in raw),
        dual_perfect=sum(c.sweep.rank_dual_weighted == 1.0 for c in cells),
        worst_dual_discrepancy=max(c.sweep.worst_dual_discrepancy for c in cells),
    )


def _markdown(one: BlindnessSweep, summary: Summary) -> str:
    lines = [
        "# P4 tables -- residuals go blind near degeneracy",
        "",
        "The anti-monotone LQ mean-field game (coupling 3, obstruction `T* = 0.8036`), one fixed "
        "box of half-width 10, one Deep Galerkin solve per horizon. Ranks are Spearman's, each "
        "score against `|S_hat(0) - S(0)|` over the horizons.",
        "",
        "## Table 1 -- one solve per horizon at the defaults (seed 0, width 32, Adam 2500)",
        "",
        "| T | den(T) | S(0) exact | S_hat(0) | error | control error | raw residual | "
        "residual / abs(den) | dual-weighted |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, horizon in enumerate(one.horizons):
        lines.append(
            f"| {horizon:.2f} | {one.denominators[i]:.4f} | {one.exact_slopes[i]:.4f} | "
            f"{one.fitted_slopes[i]:.4f} | {one.errors[i]:.4f} | "
            f"{100.0 * one.control_errors[i]:.1f}% | {one.residuals[i]:.3e} | "
            f"{one.conditioned_residuals[i]:.3e} | {one.dual_weighted[i]:.4f} |"
        )
    lines += [
        "",
        f"Rank correlation with the error: raw residual **{one.rank_residual:+.3f}**, conditioned "
        f"**{one.rank_conditioned:+.3f}**, dual-weighted **{one.rank_dual_weighted:+.3f}**; worst "
        f"relative discrepancy of the dual-weighted estimate `{one.worst_dual_discrepancy:.4f}`.",
        "",
        "## Table 2 -- the same ranks over seeds, widths and optimisers (min / median / max)",
        "",
        "| optimiser | width | seeds | raw residual | residual / abs(den) | dual-weighted | "
        "worst discrepancy | largest abs(S_hat(0)) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for group in summary.groups:
        lines.append(
            f"| {group.optimizer} | {group.width} | {group.seeds} | "
            + " | ".join(group.ranks[s].cell() for s in _SCORES)
            + f" | {group.worst_dual_discrepancy:.4f} | {group.largest_fitted_slope:.2f} |"
        )
    pooled, n = summary.pooled, summary.configurations
    lines += [
        "",
        f"Pooled over all {n} configurations: raw residual {pooled['rank_residual'].cell()}, "
        f"conditioned {pooled['rank_conditioned'].cell()}, dual-weighted "
        f"{pooled['rank_dual_weighted'].cell()}. The raw residual correlates negatively with the "
        f"error in {summary.wrong_sign} of {n} and fails to correlate positively in "
        f"{summary.not_positive}; the dual-weighted estimate ranks it perfectly in "
        f"{summary.dual_perfect}, and is never further than "
        f"`{summary.worst_dual_discrepancy:.4f}` from it in relative terms.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=float, nargs="+", default=list(_HORIZONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--out", type=Path, default=Path("results/paper4"))
    args = parser.parse_args()

    one = table_one(tuple(args.horizons))
    cells = table_two(tuple(args.horizons), tuple(args.seeds), tuple(args.widths))
    summary = summarise(cells)

    args.out.mkdir(parents=True, exist_ok=True)
    text = _markdown(one, summary)
    (args.out / "tables.md").write_text(text)
    (args.out / "tables.json").write_text(
        json.dumps(
            {
                "table1": asdict(one),
                "table2": {"cells": [asdict(c) for c in cells], "summary": asdict(summary)},
            },
            indent=2,
        )
    )
    print(text)
    print(f"written to {args.out}/tables.md and {args.out}/tables.json")


if __name__ == "__main__":
    main()

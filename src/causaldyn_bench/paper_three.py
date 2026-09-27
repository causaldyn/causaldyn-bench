"""Every table in paper P3, "Information-exploration duality".

One command produces all six.

    uv run python -m causaldyn_bench.paper_three --out results/paper3

Nothing here re-derives a model. Tables 1-5 are read off certificates in ``chc.regret`` --
:func:`minimax_exploration_certificate` (Result 56's floor and the policies that meet it),
:func:`capped_exploration_policy` (Results 56 and 66) and
:func:`multivariate_van_trees_certificate` (Result 67) -- so a table and the library cannot drift.
Table 6 is the one simulation in this module: every certificate charges a round the van Trees
floor, so none of them can say whether an actual estimator attains it, and :func:`table_six` runs
one on the certificate's own plant.

Three decisions a manuscript needs and a leaderboard does not, each falsifiable:

1. **No intervals where the quantity is exact.** P2's tables are ratios of Monte-Carlo errors and
   every cell carries a paired bootstrap. Four of the six tables here are *closed-form or exact-
   quadrature functions of the model* -- a minimax floor, a digamma sum, the root of a quadratic --
   and putting a band around them would invent uncertainty that does not exist. Tables 5 and 6
   are the exceptions and say so: Table 5's estimator arms are averages over prior draws, reported
   as a range across seeds; Table 6 is a Monte-Carlo and every cell carries its half-width.
2. **A range where the number is an instance.** Table 5's headline factor turned out to depend on
   the effect matrix the certificate draws from its seed -- it moves `2.83 .. 3.64` over five seeds,
   a `22%` spread against the Monte-Carlo columns' `0.9%` and `2.2%`. The table therefore reports
   ranges and a
   relative-spread column, and the prose quotes the bracket `1 < ratio < k`, which does not move.
3. **The plant constants are RECOVERED, not copied.** ``A``, ``K``, ``c`` are the certificates' own
   and are not exposed; recomputing them here would be a second copy that drifts silently. They are
   instead reconstructed from the shared knobs and *gated* against two identities the public surface
   does expose -- ``c_causal = 2 sqrt(A K/c)`` and ``uncapped_floor = c_causal sqrt(T) - A I0/c`` --
   and :func:`plant_constants` raises if either misses. Both hold to machine zero today; a change to
   either formula on the library side fails this module rather than silently re-scaling Table 4.
4. **Table 4 compares the library against an INDEPENDENT root.** ``predicted_mass`` is a fixed point
   of an integer map; the column beside it is the closed-form positive root of the same balance.
   Comparing the fixed point against itself would test nothing, so the gate is stated in the units
   that bound it -- the mass one round delivers, i.e. one cap.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
from chc.regret import (
    capped_exploration_policy,
    minimax_exploration_certificate,
    multivariate_van_trees_certificate,
)

# The knobs Result 56's family shares. These are the defaults of every certificate below; naming
# them once here is what lets `plant_constants` check itself against the library.
_B, _RR, _XT, _SIGMA, _I0, _ETA = 1.0, 0.5, 1.0, 0.7, 1.0, 0.6


@dataclass(frozen=True)
class PlantConstants:
    """``A``, ``K``, ``c``, ``I0`` of the exploration objective, with the checks that pin them."""

    curvature: float  # A = b^2 + rr
    numerator: float  # K = A (du*/db)^2; sigma^2 enters through c, never through K
    info_rate: float  # c = eta/sigma^2
    prior_info: float  # I0
    c_causal_residual: float  # |2 sqrt(A K/c) - certificate's c_causal|
    floor_residual: float  # |c_causal sqrt(T) - A I0/c - policy's uncapped_floor|


def plant_constants(*, horizon: int = 10**5, cap: float = 0.03) -> PlantConstants:
    """Reconstruct the objective's constants and refuse to return them if the library disagrees."""
    curvature = _B * _B + _RR
    sensitivity = -_XT * (_RR - _B * _B) / (_RR + _B * _B) ** 2
    numerator = curvature * sensitivity * sensitivity
    info_rate = _ETA / _SIGMA**2

    c_causal = 2.0 * np.sqrt(curvature * numerator / info_rate)
    first = abs(c_causal - minimax_exploration_certificate().c_causal)
    predicted_floor = c_causal * np.sqrt(horizon) - curvature * _I0 / info_rate
    second = abs(
        predicted_floor - capped_exploration_policy(horizon=horizon, cap=cap).uncapped_floor
    )
    if first > 1e-12 or second > 1e-9:
        raise RuntimeError(
            "the reconstructed constants no longer reproduce the library's own: "
            f"c_causal off by {first:.3e}, uncapped floor off by {second:.3e}"
        )
    return PlantConstants(curvature, numerator, info_rate, _I0, float(first), float(second))


def exact_mass(horizon: float, cap: float, k: PlantConstants) -> float:
    """The positive root of ``A w^2 + (K/cap) w = K c T + K I0/cap``, ``w = I0 + c S``.

    Under a constant cap the stopping round is ``S/cap``, so the remaining horizon in the balance
    is a function of the mass and the root is a quadratic rather than a square root. This is the
    closed form Result 66 (f) derives; the library reaches the same number as a fixed point.
    """
    q = k.numerator / cap
    w = (
        -q
        + np.sqrt(
            q * q + 4.0 * k.curvature * (k.numerator * k.info_rate * horizon + q * k.prior_info)
        )
    ) / (2.0 * k.curvature)
    return float((w - k.prior_info) / k.info_rate)


def result_56_mass(horizon: float, k: PlantConstants) -> float:
    """``sqrt(K T/(A c)) - I0/c``: the same balance with the horizon held at the FULL ``T``."""
    return float(
        np.sqrt(k.numerator * horizon / (k.curvature * k.info_rate)) - k.prior_info / k.info_rate
    )


# The tables are plain dicts because `main` writes each to tables.json as it stands; the
# TypedDicts below only name their schema.
class TableOne(TypedDict):
    horizons: list[int]
    floor: list[float]
    ratios: dict[str, list[float]]
    c_causal: float
    min_policy_ratio: float
    burst_over_floor: float
    taper_over_floor: float
    eta_slope: float
    sqrt_two: float


class TableTwo(TypedDict):
    horizons: list[int]
    caps: list[float]
    rows: dict[str, dict[int, dict[str, float]]]


class TableFour(TypedDict):
    cap: float
    cap_horizon: int
    ceiling: float
    coefficient: float
    ladder: list[dict[str, float]]
    by_cap: list[dict[str, float]]
    agreement: list[dict[str, float]]


class SeedSpread(TypedDict):
    lo: float
    hi: float
    span: float
    seed_invariant: bool | None


class TableFive(TypedDict):
    seeds: list[int]
    spread: dict[str, SeedSpread]


class EdgeRow(TypedDict):
    horizon: int
    radius: float
    cells: dict[str, dict[str, float]]
    local_constant_high: float


class DitherRow(TypedDict):
    rounds: int
    cells: list[dict[str, float]]
    slope: float
    predicted: float | None


class TableSix(TypedDict):
    horizons: list[int]
    reps: int
    seed: int
    c_causal: float
    edges: list[EdgeRow]
    dither: list[DitherRow]


def table_one(horizons: tuple[int, ...]) -> TableOne:
    """The sequential minimax floor, and which designs attain it.

    Every cell is a ratio to the floor, so a cell below ``1`` falsifies the lower bound outright.
    The claim is not a rate: burst approaches ``1`` and taper approaches ``sqrt(2)``, so the
    constant separates them and no amount of horizon closes the gap.
    """
    curve = minimax_exploration_certificate(horizons=horizons)
    floor = np.asarray(curve.floor)
    return {
        "horizons": [int(t) for t in curve.horizons],
        "floor": [float(v) for v in floor],
        "ratios": {
            "burst": [float(v) for v in np.asarray(curve.burst_regret) / floor],
            "taper": [float(v) for v in np.asarray(curve.taper_regret) / floor],
            "constant": [float(v) for v in np.asarray(curve.constant_regret) / floor],
            "greedy": [float(v) for v in np.asarray(curve.greedy_regret) / floor],
        },
        "c_causal": float(curve.c_causal),
        "min_policy_ratio": float(curve.min_policy_ratio),
        "burst_over_floor": float(curve.burst_over_floor),
        "taper_over_floor": float(curve.taper_over_floor),
        "eta_slope": float(curve.eta_slope),
        "sqrt_two": float(np.sqrt(2.0)),
    }


def table_two(horizons: tuple[int, ...], caps: tuple[float, ...]) -> TableTwo:
    """A per-round cap costs an ADDITIVE logarithm, so its ratio to the uncapped floor tends to 1.

    This is the table that separates a cap from a taper. The taper's ``sqrt(2)`` in Table 1 is a
    constant FACTOR and does not close; the cap's price is a constant ADDED to a growing floor, so
    every column here decreases, and the block length grows like ``sqrt(T)`` -- a tighter actuator
    explores for longer, not more gently.
    """
    rows: dict[str, dict[int, dict[str, float]]] = {}
    for cap in caps:
        cells: dict[int, dict[str, float]] = {}
        for horizon in horizons:
            policy = capped_exploration_policy(horizon=horizon, cap=cap)
            cells[horizon] = {
                "cost_over_floor": policy.cost / policy.uncapped_floor,
                "block_rounds": float(policy.block_rounds),
                "excess": policy.excess,
                "predicted_excess": policy.predicted_excess,
                "taper_over_block": policy.taper_cost / policy.cost,
            }
        rows[f"{cap:g}"] = cells
    return {"horizons": [int(t) for t in horizons], "caps": [float(c) for c in caps], "rows": rows}


def _schedules(horizon: int) -> dict[str, np.ndarray]:
    """Five cap schedules whose block lengths differ by nearly an order of magnitude."""
    ramp = np.linspace(0.01, 0.05, horizon)
    return {
        "constant 0.03": np.full(horizon, 0.03),
        "ramp up 0.01->0.05": ramp,
        "ramp down 0.05->0.01": ramp[::-1].copy(),
        "dead first third": np.concatenate(
            [np.zeros(horizon // 3), np.full(horizon - horizon // 3, 0.03)]
        ),
        "uniform noise U(0, 0.06)": np.random.default_rng(0).uniform(0.0, 0.06, horizon),
    }


def table_three(horizon: int, k: PlantConstants) -> dict[str, dict[str, float]]:
    """The stopping MASS is the invariant; the stopping ROUND is whatever prefix sum reaches it.

    Five schedules, one horizon. The block lengths differ by ``6.9x`` and every delivered mass sits
    within a fraction of a cap of the same fixed point -- while Result 56's closed form, which does
    not know when the caps open, is far outside on the schedules that open late.
    """
    leading = result_56_mass(horizon, k)
    out: dict[str, dict[str, float]] = {}
    for name, caps in _schedules(horizon).items():
        policy = capped_exploration_policy(horizon=horizon, cap=[float(c) for c in caps])
        landing = float(caps[min(policy.block_rounds, caps.size) - 1])
        out[name] = {
            "block_rounds": float(policy.block_rounds),
            "mass": policy.exploration_mass,
            "predicted_mass": policy.predicted_mass,
            "gap_in_caps": abs(policy.exploration_mass - policy.predicted_mass)
            / max(landing, 1e-12),
            "result_56_over_delivered": leading / policy.exploration_mass,
        }
    return out


def table_four(
    horizons: tuple[int, ...],
    caps: tuple[float, ...],
    ladder_caps: tuple[float, ...],
    cap_horizon: int,
    k: PlantConstants,
) -> TableFour:
    """What Result 56's closed form drops is a CONSTANT, and the cap decides how big it is.

    Panel (a) is the horizon ladder at one cap: the gap to the closed form rises to the ceiling
    ``K/(2 A c cap)`` and stops, and the residual after subtracting it decays exactly as the
    expansion says. Panel (b) is the cap ladder at one horizon, which is the operational reading --
    a constant that never vanishes is invisible at a loose cap and dominant at a tight one. It takes
    its own ladder, wider than the three caps Table 2 shares with panel (a).
    """
    cap = caps[len(caps) // 2]
    ceiling = k.numerator / (2.0 * k.curvature * k.info_rate * cap)
    coefficient = (
        (k.numerator + 4.0 * k.curvature * k.prior_info * cap)
        * np.sqrt(k.numerator / (k.curvature * k.info_rate))
        / (8.0 * k.curvature * k.info_rate * cap**2)
    )
    ladder: list[dict[str, float]] = []
    for horizon in horizons:
        exact = exact_mass(horizon, cap, k)
        gap = result_56_mass(horizon, k) - exact
        ladder.append(
            {
                "horizon": int(horizon),
                "exact": exact,
                "gap": gap,
                "gap_over_ceiling": gap / ceiling,
                "scaled_residual": (gap - ceiling) * np.sqrt(horizon),
            }
        )
    by_cap: list[dict[str, float]] = []
    for level in ladder_caps:
        exact = exact_mass(cap_horizon, level, k)
        gap = result_56_mass(cap_horizon, k) - exact
        by_cap.append(
            {
                "cap": float(level),
                "exact": exact,
                "gap": gap,
                "ceiling": k.numerator / (2.0 * k.curvature * k.info_rate * level),
                "relative": gap / exact,
            }
        )
    agreement: list[dict[str, float]] = []
    for horizon in (h for h in horizons if h <= 10**6):
        policy = capped_exploration_policy(horizon=horizon, cap=cap)
        agreement.append(
            {
                "horizon": int(horizon),
                "fixed_point": policy.predicted_mass,
                "root": exact_mass(horizon, cap, k),
                "gap_in_caps": abs(policy.predicted_mass - exact_mass(horizon, cap, k)) / cap,
            }
        )
    return {
        "cap": float(cap),
        "cap_horizon": int(cap_horizon),
        "ceiling": float(ceiling),
        "coefficient": float(coefficient),
        "ladder": ladder,
        "by_cap": by_cap,
        "agreement": agreement,
    }


def table_five(seeds: tuple[int, ...]) -> TableFive:
    """The matrix floor is a TRACE, so confounding is priced by ALIGNMENT rather than by a ratio.

    Reported as a range over seeds, and the range is the finding. The certificate's alignment arm
    draws its 2x2 effect matrix from the seed, so ``aligned_ratio`` is a property of THAT matrix and
    not of the model family: it moves by 22% across five seeds, more than either estimator column.
    What does not move is the shape of the claim -- ``orthogonal_ratio`` is 1 to floating-point zero
    at every seed, ``worst_single_direction`` reproduces ``aligned_ratio`` exactly, and the ratio
    stays strictly inside ``(1, k)``. So the headline factor is an instance and the bracket is the
    result; a single number quoted from one seed reads as the second and is the first.
    """
    curves = [multivariate_van_trees_certificate(seed=seed) for seed in seeds]
    fields = (
        "aligned_ratio",
        "orthogonal_ratio",
        "worst_single_direction",
        "information_loss",
        "live_channel_weight",
        "knife_edge_weight",
        "floor",
        "plugin_ratio",
        "hodges_pointwise_ratio",
        "hodges_bayes_ratio",
    )
    spread: dict[str, SeedSpread] = {}
    for field in fields:
        values = np.array([float(getattr(curve, field)) for curve in curves])
        lo, hi = float(values.min()), float(values.max())
        scale = max(abs(lo), abs(hi))
        # relative, not exact: a quantity assembled by an eigensolve can carry a last-bit wiggle
        # and still be the same number, and `orthogonal_ratio == 1` is exactly that case
        span = (hi - lo) / scale if scale else 0.0
        spread[field] = {
            "lo": lo,
            "hi": hi,
            "span": span,
            # one seed cannot distinguish an exact column from a lucky one, so it says nothing
            # rather than saying "invariant" about a quantity it never varied
            "seed_invariant": bool(span < 1e-12) if len(curves) > 1 else None,
        }
    return {"seeds": [int(s) for s in seeds], "spread": spread}


def _oracle_action(effect: np.ndarray | float) -> np.ndarray:
    """``u*(b) = -b x/(b^2 + rr)``: the one-step plant's optimal action, bounded in ``b``."""
    effect = np.asarray(effect, dtype=np.float64)
    return -effect * _XT / (effect * effect + _RR)


@dataclass(frozen=True)
class CommitOutcome:
    """The realised regret of explore-then-commit, averaged over replications, and its two parts."""

    regret: float
    standard_error: float  # of the regret's mean
    probe_cost: float  # mean cost of the probe rounds
    commit_cost: float  # mean cost of the committed rounds


def explore_then_commit(
    horizon: int,
    probe_rounds: int,
    probe: str,
    effect: float,
    k: PlantConstants,
    rng: np.random.Generator,
    reps: int,
) -> CommitOutcome:
    """The REALISED regret of explore-then-commit with least squares, over ``reps`` replications.

    ``probe_rounds`` rounds play the prior centre's action plus a probe, the rest commit to
    ``u*(bhat)``. Only the probe identifies the effect -- ``y = b sqrt(eta) e + noise`` -- which is
    the objective's own information model. The budget is the constant-magnitude optimum
    ``sqrt(K (T - n)/(A c))`` of validation STEP 10, for both probes: ``"constant"`` spends it as
    ``+-sqrt(M/n)``, ``"gaussian"`` as ``N(0, M/n)`` dither. The cost is the plant's exact
    ``(b^2 + rr)(u - u*(b))^2`` at the TRUE effect, so nothing here is linearised.
    """
    budget = np.sqrt(k.numerator * (horizon - probe_rounds) / (k.curvature * k.info_rate))
    scale = np.sqrt(budget / probe_rounds)
    if probe == "constant":
        e = rng.choice(np.array([-1.0, 1.0]), size=(reps, probe_rounds)) * scale
    elif probe == "gaussian":
        e = rng.normal(0.0, scale, size=(reps, probe_rounds))
    else:
        raise ValueError(f"probe must be 'constant' or 'gaussian', got {probe!r}")
    y = effect * np.sqrt(_ETA) * e + rng.normal(0.0, _SIGMA, size=(reps, probe_rounds))
    estimate = (y * e).sum(axis=1) / (np.sqrt(_ETA) * (e * e).sum(axis=1))
    curvature = effect * effect + _RR
    target = _oracle_action(effect)
    explore = curvature * ((_oracle_action(_B) - target + e) ** 2).sum(axis=1)
    commit = (horizon - probe_rounds) * curvature * (_oracle_action(estimate) - target) ** 2
    regret = explore + commit
    return CommitOutcome(
        float(regret.mean()),
        float(regret.std(ddof=1) / np.sqrt(reps)),
        float(explore.mean()),
        float(commit.mean()),
    )


def table_six(
    horizons: tuple[int, ...],
    rounds: tuple[int, ...],
    reps: int,
    seed: int,
    k: PlantConstants,
) -> TableSix:
    """The floor against a REAL estimator: attained by a constant-magnitude probe, not by dither.

    Tables 1-5 charge every round the van Trees floor, so they can say which SCHEDULE attains
    ``c_causal sqrt(T)`` and nothing about whether a policy does. This table runs one. Panel (a):
    explore-then-commit with a ``+-`` probe and least squares, at the prior centre and at the edges
    of the ``T^(-1/4)`` neighbourhood the minimax bound ranges over. Panel (b): the same budget as
    Gaussian dither over ``n`` rounds, which least squares sees through ``E[1/chi2_n] = 1/(n - 2)``
    -- a factor ``(n - 1)/(n - 2)`` where finite, and for ``n <= 2`` a lost RATE, because the
    plant's bounded ``u*`` clips an estimate that would otherwise have infinite variance.

    Monte-Carlo, so every cell carries a 95% half-width; each cell draws from its own child of
    ``seed``, so a cell does not depend on which cells ran before it.
    """
    c_causal = 2.0 * np.sqrt(k.curvature * k.numerator / k.info_rate)
    children = iter(np.random.SeedSequence(seed).spawn(len(horizons) * (3 + len(rounds))))

    def c_at(effect: float) -> float:
        sensitivity = -_XT * (_RR - effect * effect) / (_RR + effect * effect) ** 2
        return 2.0 * (effect * effect + _RR) * abs(sensitivity) * _SIGMA / np.sqrt(_ETA)

    edges: list[EdgeRow] = []
    for horizon in horizons:
        radius = horizon**-0.25
        cells = {}
        for name, effect in (("low", _B - radius), ("centre", _B), ("high", _B + radius)):
            outcome = explore_then_commit(
                horizon, 1, "constant", effect, k, np.random.default_rng(next(children)), reps
            )
            floor = c_causal * np.sqrt(horizon)
            cells[name] = {
                "ratio": outcome.regret / floor,
                "half_width": 1.96 * outcome.standard_error / floor,
                "commit_over_probe": outcome.commit_cost / outcome.probe_cost,
            }
        edges.append(
            {
                "horizon": int(horizon),
                "radius": float(radius),
                "cells": cells,
                "local_constant_high": float(c_at(_B + radius) / c_causal),
            }
        )

    dither: list[DitherRow] = []
    for n in rounds:
        means = []
        cells = []
        for horizon in horizons:
            outcome = explore_then_commit(
                horizon, n, "gaussian", _B, k, np.random.default_rng(next(children)), reps
            )
            floor = c_causal * np.sqrt(horizon)
            means.append(outcome.regret)
            cells.append(
                {
                    "ratio": outcome.regret / floor,
                    "half_width": 1.96 * outcome.standard_error / floor,
                }
            )
        slope = float(np.polyfit(np.log(horizons), np.log(means), 1)[0])
        dither.append(
            {
                "rounds": int(n),
                "cells": cells,
                "slope": slope,
                "predicted": (n - 1) / (n - 2) if n > 2 else None,
            }
        )
    return {
        "horizons": [int(t) for t in horizons],
        "reps": int(reps),
        "seed": int(seed),
        "c_causal": float(c_causal),
        "edges": edges,
        "dither": dither,
    }


def _cell(lo: float, hi: float) -> str:
    if lo == hi:
        return f"{lo:.4g}"
    return f"{lo:.4g} .. {hi:.4g}"


def _markdown(
    one: TableOne,
    two: TableTwo,
    three: dict[str, dict[str, float]],
    four: TableFour,
    five: TableFive,
    six: TableSix,
    k: PlantConstants,
) -> str:
    horizons = one["horizons"]
    ratios = one["ratios"]
    lines = [
        "# P3 tables -- information-exploration duality",
        "",
        f"Plant `A = {k.curvature:g}`, `K = {k.numerator:.6g}`, `c = {k.info_rate:.6g}`, "
        f"`I0 = {k.prior_info:g}`, recovered from the certificates and checked against them "
        f"(`c_causal` residual `{k.c_causal_residual:.1e}`, floor residual "
        f"`{k.floor_residual:.1e}`). Tables 1-4 are exact functions of the model and carry no "
        "intervals; Table 5 is a range over seeds, and says which of its columns moved; Table 6 "
        "runs an actual estimator and carries Monte-Carlo half-widths.",
        "",
        "## Table 1 -- the sequential minimax floor, and who attains it",
        "",
        "| T | floor | burst | taper | constant v | greedy |",
        "|---|---|---|---|---|---|",
    ]
    floors = one["floor"]
    for index, horizon in enumerate(horizons):
        lines.append(
            f"| {horizon:,} | {floors[index]:.4g} | "
            + " | ".join(
                f"{ratios[arm][index]:.4g}" for arm in ("burst", "taper", "constant", "greedy")
            )
            + " |"
        )
    lines += [
        "",
        f"Ratios to the floor, so a cell below 1 falsifies the bound; the minimum over all "
        f"policies and horizons is `{one['min_policy_ratio']:.6f}`. The constant is SHARP, not a "
        f"rate: burst reaches `{one['burst_over_floor']:.6f}` while taper sits at "
        f"`{one['taper_over_floor']:.4f}` against `sqrt(2) = {one['sqrt_two']:.4f}`. "
        f"`c_causal = {one['c_causal']:.6g}`. Every column here charges each round the van Trees "
        "floor, so this table says which SCHEDULE attains the constant; Table 6 runs a policy. "
        f"The certificate's log-log slope in `eta`, `{one['eta_slope']:.4f}`, evaluates the "
        "closed form `c_causal ~ 1/sqrt(eta)` and checks its transcription, nothing more.",
        "",
        "## Table 2 -- a cap costs an additive logarithm, so its ratio to the floor decreases",
        "",
        "| cap | " + " | ".join(f"T = {t:,}" for t in horizons) + " |",
        "|---|" + "---|" * len(horizons),
    ]
    rows = two["rows"]
    for cap, cells in rows.items():
        lines.append(
            f"| {cap} | "
            + " | ".join(f"{cells[t]['cost_over_floor']:.4f}" for t in horizons)
            + " |"
        )
    lines += [
        "",
        "Cost over the uncapped floor. Every row decreases: unlike the taper's `sqrt(2)` factor, a "
        "cap adds a constant to a growing floor. Block lengths, same cells:",
        "",
        "| cap | " + " | ".join(f"T = {t:,}" for t in horizons) + " |",
        "|---|" + "---|" * len(horizons),
    ]
    for cap, cells in rows.items():
        lines.append(
            f"| {cap} | "
            + " | ".join(f"{int(cells[t]['block_rounds']):,}" for t in horizons)
            + " |"
        )
    lines += [
        "",
        "## Table 3 -- the stopping MASS is the invariant, not the block length",
        "",
        "| schedule | block rounds | delivered mass | fixed point | gap, in caps | "
        "Result 56 / delivered |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in three.items():
        lines.append(
            f"| {name} | {int(row['block_rounds']):,} | {row['mass']:.4f} | "
            f"{row['predicted_mass']:.4f} | {row['gap_in_caps']:.3f} | "
            f"{row['result_56_over_delivered']:.3f} |"
        )
    ceiling = four["ceiling"]
    coefficient = four["coefficient"]
    ladder = four["ladder"]
    by_cap = four["by_cap"]
    agreement = four["agreement"]
    lines += [
        "",
        "The block lengths span nearly an order of magnitude and every delivered mass lands on the "
        "same fixed point. The last column is why the leading form was replaced: it does not know "
        "when the caps open.",
        "",
        f"## Table 4 -- what the closed form drops is a constant, `K/(2 A c cap) = {ceiling:.6g}`",
        "",
        f"(a) horizon ladder at `cap = {four['cap']}`:",
        "",
        "| T | exact mass | Result 56 - exact | over the ceiling | residual x sqrt(T) |",
        "|---|---|---|---|---|",
    ]
    for row in ladder:
        lines.append(
            f"| {row['horizon']:,} | {row['exact']:.4f} | {row['gap']:.6f} | "
            f"{row['gap_over_ceiling']:.6f} | {row['scaled_residual']:.4f} |"
        )
    lines += [
        "",
        f"The gap rises to the ceiling and stops -- it never reaches it, and never exceeds it. The "
        f"last column converges to `-{coefficient:.6g}`, the coefficient the expansion predicts.",
        "",
        f"(b) cap ladder at `T = {four['cap_horizon']:,}`:",
        "",
        "| cap | exact mass | Result 56 - exact | ceiling | as a share of the mass |",
        "|---|---|---|---|---|",
    ]
    for row in by_cap:
        lines.append(
            f"| {row['cap']:g} | {row['exact']:.4f} | {row['gap']:.4f} | {row['ceiling']:.4f} | "
            f"{100.0 * row['relative']:.2f}% |"
        )
    lines += [
        "",
        "(c) the library's fixed point against the closed-form root, in units of the cap:",
        "",
        "| T | fixed point | root | gap, in caps |",
        "|---|---|---|---|",
    ]
    for row in agreement:
        lines.append(
            f"| {row['horizon']:,} | {row['fixed_point']:.6f} | {row['root']:.6f} | "
            f"{row['gap_in_caps']:.4f} |"
        )
    spread = five["spread"]
    seeds = five["seeds"]
    lines += [
        "",
        "## Table 5 -- the matrix floor is a trace, so confounding is priced by ALIGNMENT",
        "",
        f"Range over {len(seeds)} seed{'' if len(seeds) == 1 else 's'}. A column marked exact did "
        "not move at all; with a single seed the column is `?`, because one draw cannot tell an "
        "exact quantity from a lucky one.",
        "",
        "| quantity | range | relative spread | seed-invariant |",
        "|---|---|---|---|",
    ]
    for field, values in spread.items():
        invariant = values["seed_invariant"]
        mark = "?" if invariant is None else ("yes" if invariant else "--")
        lines.append(
            f"| {field} | {_cell(values['lo'], values['hi'])} | {values['span']:.1e} | {mark} |"
        )
    lines += [
        "",
        "The same information loss costs a factor in the direction the optimal action leans on and "
        "exactly nothing in a direction in the kernel of `Psi'` -- which is the whole content of "
        "the trace form, and is invisible to a scalar plant that has only one direction.",
        "",
        "The spread column is the reason this table is a range. The certificate draws the 2x2 "
        "effect matrix from its seed, so the SIZE of the aligned factor is a property of that draw "
        "and moves more across seeds than either estimator column does. The BRACKET does not: "
        "`orthogonal_ratio` is 1 to floating-point zero at every seed, `worst_single_direction` "
        "reproduces `aligned_ratio` exactly, and the factor stays strictly inside `(1, k)`. Quote "
        "the bracket; the factor is an instance of it.",
    ]
    six_horizons = six["horizons"]
    edges = six["edges"]
    dither = six["dither"]
    lines += [
        "",
        "## Table 6 -- the floor against a real estimator: a constant-magnitude probe attains it, "
        "dither does not",
        "",
        f"Explore-then-commit with least squares on the one-step plant, `{six['reps']:,}` "
        f"replications a cell (seed `{six['seed']}`), the budget of validation STEP 10. Cells are "
        "the realised regret over `c_causal sqrt(T)`, with a 95% Monte-Carlo half-width.",
        "",
        "(a) a `+-sqrt(M)` probe in one round, at the prior centre and at the edges of the "
        "`T^(-1/4)` neighbourhood the minimax bound ranges over:",
        "",
        "| T | b0 - T^(-1/4) | b0 | b0 + T^(-1/4) | c(b0 + T^(-1/4)) / c(b0) |",
        "|---|---|---|---|---|",
    ]
    for row in edges:
        cells = row["cells"]
        lines.append(
            f"| {row['horizon']:,} | "
            + " | ".join(
                f"{cells[name]['ratio']:.4f} ± {cells[name]['half_width']:.4f}"
                for name in ("low", "centre", "high")
            )
            + f" | {row['local_constant_high']:.4f} |"
        )
    lines += [
        "",
        "At the centre the ratio tends to 1: the constant is attained by a policy, not only by a "
        "schedule. The edges approach it more slowly because the local constant itself moves "
        "across the neighbourhood at first order -- the last column -- and the neighbourhood "
        "shrinks only like `T^(-1/4)`.",
        "",
        "(b) the same budget as Gaussian dither over `n` rounds, at the prior centre:",
        "",
        "| n | "
        + " | ".join(f"T = {t:,}" for t in six_horizons)
        + " | log-log slope | (n-1)/(n-2) |",
        "|---|" + "---|" * (len(six_horizons) + 2),
    ]
    for row in dither:
        predicted = row["predicted"]
        lines.append(
            f"| {row['rounds']} | "
            + " | ".join(f"{cell['ratio']:.3f} ± {cell['half_width']:.3f}" for cell in row["cells"])
            + f" | {row['slope']:.3f} | "
            + ("inf" if predicted is None else f"{predicted:.4f}")
            + " |"
        )
    lines += [
        "",
        "Least squares sees a probe through its realised energy, not its variance. One Gaussian "
        "round loses the RATE -- the slope is `3/4`, not `1/2` -- because a near-zero draw leaves "
        "an estimate the bounded `u*` can only clip; two rounds lose a logarithm; from three the "
        "rate returns with the factor `(n-1)/(n-2)`. A cap forces `n = M/cap`, of order "
        "`sqrt(T)`, which is why capped blocks never see this.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10**3, 10**4, 10**5, 10**6])
    parser.add_argument("--caps", type=float, nargs="+", default=[0.1, 0.03, 0.01])
    parser.add_argument(
        "--mass-horizons", type=int, nargs="+", default=[10**4, 10**5, 10**6, 10**7, 10**8]
    )
    parser.add_argument("--schedule-horizon", type=int, default=4000)
    parser.add_argument("--cap-horizon", type=int, default=4000)
    parser.add_argument(
        "--ladder-caps",
        type=float,
        nargs="+",
        default=[10 ** (-k / 2) for k in range(1, 7)],
        help="Table 4(b)'s caps: half-decades from 10^-0.5 to 10^-3",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    parser.add_argument(
        "--estimator-horizons", type=int, nargs="+", default=[10**3, 10**4, 10**5, 10**6, 10**7]
    )
    parser.add_argument("--probe-rounds", type=int, nargs="+", default=[1, 2, 3, 10, 30])
    parser.add_argument("--estimator-reps", type=int, default=200_000)
    parser.add_argument("--estimator-seed", type=int, default=20260926)
    parser.add_argument("--out", type=Path, default=Path("results/paper3"))
    args = parser.parse_args()

    constants = plant_constants()
    one = table_one(tuple(args.horizons))
    two = table_two(tuple(args.horizons), tuple(args.caps))
    three = table_three(args.schedule_horizon, constants)
    four = table_four(
        tuple(args.mass_horizons),
        tuple(args.caps),
        tuple(args.ladder_caps),
        args.cap_horizon,
        constants,
    )
    five = table_five(tuple(args.seeds))
    six = table_six(
        tuple(args.estimator_horizons),
        tuple(args.probe_rounds),
        args.estimator_reps,
        args.estimator_seed,
        constants,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    text = _markdown(one, two, three, four, five, six, constants)
    (args.out / "tables.md").write_text(text)
    (args.out / "tables.json").write_text(
        json.dumps(
            {
                "plant": {
                    "curvature": constants.curvature,
                    "numerator": constants.numerator,
                    "info_rate": constants.info_rate,
                    "prior_info": constants.prior_info,
                    "c_causal_residual": constants.c_causal_residual,
                    "floor_residual": constants.floor_residual,
                },
                "table1": one,
                "table2": two,
                "table3": three,
                "table4": four,
                "table5": five,
                "table6": six,
            },
            indent=2,
        )
    )
    print(text)
    print(f"written to {args.out}/tables.md and {args.out}/tables.json")


if __name__ == "__main__":
    main()

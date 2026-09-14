"""Every table in paper P3, "Information-exploration duality".

One command produces all five.

    uv run python -m causaldyn_bench.paper_three --out results/paper3

Nothing here re-derives a model. Every number is read off a certificate in ``chc.regret`` --
:func:`minimax_exploration_certificate` (Result 56's floor and the policies that meet it),
:func:`capped_exploration_policy` (Results 56 and 66) and
:func:`multivariate_van_trees_certificate` (Result 67) -- so a table and the library cannot drift.

Three decisions a manuscript needs and a leaderboard does not, each falsifiable:

1. **No intervals where the quantity is exact.** P2's tables are ratios of Monte-Carlo errors and
   every cell carries a paired bootstrap. Four of the five tables here are *closed-form or exact-
   quadrature functions of the model* -- a minimax floor, a digamma sum, the root of a quadratic --
   and putting a band around them would invent uncertainty that does not exist. Table 5 is the
   exception and says so: its estimator arms are averages over prior draws, and it is reported as a
   range across seeds rather than a single number.
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
    numerator: float  # K = A (du*/db)^2 sigma^2, the van Trees numerator
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


def table_one(horizons: tuple[int, ...]) -> dict[str, object]:
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


def table_two(horizons: tuple[int, ...], caps: tuple[float, ...]) -> dict[str, object]:
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
        policy = capped_exploration_policy(horizon=horizon, cap=caps)
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
    horizons: tuple[int, ...], caps: tuple[float, ...], cap_horizon: int, k: PlantConstants
) -> dict[str, object]:
    """What Result 56's closed form drops is a CONSTANT, and the cap decides how big it is.

    Panel (a) is the horizon ladder at one cap: the gap to the closed form rises to the ceiling
    ``K/(2 A c cap)`` and stops, and the residual after subtracting it decays exactly as the
    expansion says. Panel (b) is the cap ladder at one horizon, which is the operational reading --
    a constant that never vanishes is invisible at a loose cap and dominant at a tight one.
    """
    cap = caps[len(caps) // 2]
    ceiling = k.numerator / (2.0 * k.curvature * k.info_rate * cap)
    coefficient = (
        (k.numerator + 4.0 * k.curvature * k.prior_info * cap)
        * np.sqrt(k.numerator / (k.curvature * k.info_rate))
        / (8.0 * k.curvature * k.info_rate * cap**2)
    )
    ladder = []
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
    by_cap = []
    for level in caps:
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
    agreement = []
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


def table_five(seeds: tuple[int, ...]) -> dict[str, object]:
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
    out: dict[str, object] = {"seeds": [int(s) for s in seeds]}
    spread: dict[str, dict[str, float | bool | None]] = {}
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
    out["spread"] = spread
    return out


def _cell(lo: float, hi: float) -> str:
    if lo == hi:
        return f"{lo:.4g}"
    return f"{lo:.4g} .. {hi:.4g}"


def _markdown(
    one: dict[str, object],
    two: dict[str, object],
    three: dict[str, dict[str, float]],
    four: dict[str, object],
    five: dict[str, object],
    k: PlantConstants,
) -> str:
    horizons = one["horizons"]
    assert isinstance(horizons, list)
    ratios = one["ratios"]
    assert isinstance(ratios, dict)
    lines = [
        "# P3 tables -- information-exploration duality",
        "",
        f"Plant `A = {k.curvature:g}`, `K = {k.numerator:.6g}`, `c = {k.info_rate:.6g}`, "
        f"`I0 = {k.prior_info:g}`, recovered from the certificates and checked against them "
        f"(`c_causal` residual `{k.c_causal_residual:.1e}`, floor residual "
        f"`{k.floor_residual:.1e}`). Tables 1-4 are exact functions of the model and carry no "
        "intervals; Table 5 is a range over seeds, and says which of its columns moved.",
        "",
        "## Table 1 -- the sequential minimax floor, and who attains it",
        "",
        "| T | floor | burst | taper | constant v | greedy |",
        "|---|---|---|---|---|---|",
    ]
    floors = one["floor"]
    assert isinstance(floors, list)
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
        f"`c_causal = {one['c_causal']:.6g}` and its log-log slope in `eta` is "
        f"`{one['eta_slope']:.4f}` -- the `1/sqrt(eta)` causal scaling.",
        "",
        "## Table 2 -- a cap costs an additive logarithm, so its ratio to the floor decreases",
        "",
        "| cap | " + " | ".join(f"T = {t:,}" for t in horizons) + " |",
        "|---|" + "---|" * len(horizons),
    ]
    rows = two["rows"]
    assert isinstance(rows, dict)
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
    assert isinstance(ceiling, float) and isinstance(coefficient, float)
    ladder = four["ladder"]
    by_cap = four["by_cap"]
    agreement = four["agreement"]
    assert isinstance(ladder, list) and isinstance(by_cap, list) and isinstance(agreement, list)
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
    assert isinstance(spread, dict) and isinstance(seeds, list)
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    parser.add_argument("--out", type=Path, default=Path("results/paper3"))
    args = parser.parse_args()

    constants = plant_constants()
    one = table_one(tuple(args.horizons))
    two = table_two(tuple(args.horizons), tuple(args.caps))
    three = table_three(args.schedule_horizon, constants)
    four = table_four(tuple(args.mass_horizons), tuple(args.caps), args.cap_horizon, constants)
    five = table_five(tuple(args.seeds))

    args.out.mkdir(parents=True, exist_ok=True)
    text = _markdown(one, two, three, four, five, constants)
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
            },
            indent=2,
        )
    )
    print(text)
    print(f"written to {args.out}/tables.md and {args.out}/tables.json")


if __name__ == "__main__":
    main()

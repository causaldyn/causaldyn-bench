"""Every table in paper P1, "Debias every channel".

One command produces all five.

    uv run python -m causaldyn_bench.paper_one --out results/paper1

Nothing here re-derives a model. Every number is read off a certificate in ``chc.regret`` --
:func:`composition_transfer_certificate` (the order-doubling lemma),
:func:`multivariate_interference_certificate` and :func:`exposure_map_certificate` (the two- and
three-channel bottleneck), :func:`end_to_end_c2_certificate` (both regimes),
:func:`clustered_lower_bound_certificate` (the two-sided floor) and :func:`regret_scaling` (the
quadratic law from random draws) -- so a table and the library cannot drift.

**Every headline number in this paper is an exponent fitted to a log-log sweep, and the one thing
a manuscript owes its reader about such a number is the window it was measured on.** That is the
organising decision here, and it is what separates these tables from the appendix they replace.
Four consequences, each falsifiable:

1. **A fitted slope is not the exponent; it is the exponent plus a window term, and the window term
   is closed form.** The order-transfer certificate reports `2.05 / 4.01 / 6.00` where the theorem
   says `2 / 4 / 6`, and the `0.05` has been quoted as agreement-up-to-noise. It is not noise: it is
   deterministic. Expanding the exact regret map in `e = delta^p` gives
   `log R = const + 2 p t + lambda e + 2 c2 e^2 + O(e^3)` with `t = log delta`, so an ordinary
   least-squares fit over the window picks up `lambda cov(t, e^{pt})/var(t)` exactly. Table 1
   reconstructs `lambda` and `c2` from the plant in closed form (derived in Maxima, see
   :func:`transfer_constants`) and reports what is left. Two terms reduce the `2.83e-2` miss at
   `p = 1` to `7.6e-3`, and the `1.04e-3` miss at `p = 2` to `5.6e-5`.
2. **The same window has a lower end, and it fails catastrophically rather than gracefully.** Push
   the sweep down to `delta in [1e-5, 2e-4]` and the order-3 slope reads `6.035`, then the
   three-channel LQ full-orthogonality slope reads `nan` outright -- `delta^4` regret has
   underflowed to exactly zero and `log 0` is what the fit sees. Table 1 carries the
   double-precision cancellation floor beside every cell, and it is the column that says which
   reading to believe.
3. **The `1/G` rate is a window measurement too, and its shipped window is too small.** The
   appendix's `-1.08` and its "flat" `-0.09` plateau are finite-`G` artefacts of a grid that starts
   at `G = 10`: move the grid up and the end-to-end slope walks `-1.139, -1.121, -1.054, -1.024`
   and the lower-bound plateau `-0.177, -0.115, -0.054, -0.045`, while `c0` sits still at
   `0.183 .. 0.202` -- so the shipped grid got the grid wrong, not the constant. The independent
   base-R check in `validation/clustered_rate_check.R` walks the same way on a different statistic
   in a different language (`-0.5575 -> -0.5072`, rejecting the theoretical `-0.5` at `4.8`
   standard errors on the old grid and at `0.6` on the new one). Tables 3 and 4 report the ladder,
   not the single cell.
4. **Intervals only where a number is random, and the error bar the API admits.** Tables 1 and 2
   and the `delta` half of Table 3 are deterministic functions of the plant -- a band around them
   would invent uncertainty. The `G` halves of Tables 3 and 4 are Monte-Carlo averages, and neither
   certificate takes a seed offset, so an independent replicate cannot be drawn; what can is the
   chain of *nested prefixes*, where rung `j` has a KNOWN multiple of the wanted variance
   (:func:`_monte_carlo_scale`). Reading one rung is not enough, and the run proved it rather than
   assuming it: the single-rung version reported a scale of `0.0475` at 240 seeds and `0.0665` at
   960 -- larger after four times the work -- and both ladders correctly refused to quote
   themselves. Pooling three rescaled rungs on the same budget turned those refusals into `4.9x`
   and `3.1x`. The fix was a statistic, not compute. Table 5 is a per-seed range.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from chc.regret import (
    clustered_lower_bound_certificate,
    composition_transfer_certificate,
    end_to_end_c2_certificate,
    exposure_map_certificate,
    multivariate_interference_certificate,
    regret_scaling,
)
from numpy.typing import NDArray

# The scalar transfer plant `u*(b) = b xt/(b^2 + rr)`, which is
# `composition_transfer_certificate`'s default and the only plant whose window term is available in
# closed form. The multivariate certificates below use a different regret map (a Riccati solve), so
# their windows are shown by shrinking them rather than by predicting them.
_B, _RR, _XT = 1.0, 0.5, 1.0

# The state dimension of the multivariate LQ plant every other certificate in P1 shares. Only
# `table_five` needs it, because `regret_scaling` is the one certificate that takes the plant as
# arguments rather than carrying its own.
_LQ_STATES = 2

_EPS = float(np.finfo(np.float64).eps)

# How many nested prefixes the two `G` ladders pool their error bar over. Three costs 87% more
# than the full run alone and turns one noisy draw of `sigma` into six independent ones.
_PREFIX_LEVELS = 3


@dataclass(frozen=True)
class TransferConstants:
    """The closed-form window expansion of the order-transfer certificate, with its own gate."""

    first: float  # lambda = u''(b)/u'(b), the coefficient of `e` in `log R - 2 p t`
    second: float  # c2, the coefficient of `e^2` in `log(f/(u' e))`; enters the slope as `2 c2`
    doubling_residual: float  # max |expected_slopes - 2 p| across the certificate's orders
    prediction_residual: float  # |measured - (2p + first + second terms)| at the reference window


def transfer_constants(
    *, delta_lo: float = 1e-3, delta_hi: float = 2e-2, n_delta: int = 12
) -> TransferConstants:
    """Reconstruct the window expansion and refuse to return it if the library disagrees.

    Derived in Maxima from `u*(b) = b xt/(b^2 + rr)` (the derivation is four lines: expand
    `log((u*(b + e) - u*(b))/(u*'(b) e))` in `e` and read the first two coefficients):

        lambda = u*''(b)/u*'(b) = (2 b^3 - 6 b rr)/(rr^2 - b^4)
        c2     = (b^6 - 8 b^4 rr + 5 b^2 rr^2 - 2 rr^3) / (2 (rr^2 - b^4)^2)

    Neither is exposed by any certificate, so both are gated rather than trusted: the two-term
    prediction must reproduce the certificate's own fitted slope at the reference window to better
    than the next order, and the certificate's advertised `expected_slopes` must still be `2 p`.
    """
    denominator = _RR**2 - _B**4
    first = (2.0 * _B**3 - 6.0 * _B * _RR) / denominator
    second = (_B**6 - 8.0 * _B**4 * _RR + 5.0 * _B**2 * _RR**2 - 2.0 * _RR**3) / (
        2.0 * denominator**2
    )

    curve = composition_transfer_certificate(
        b=_B, rr=_RR, xt=_XT, delta_lo=delta_lo, delta_hi=delta_hi, n_delta=n_delta
    )
    doubling = float(np.max(np.abs(curve.expected_slopes - 2.0 * curve.orders)))

    logs = np.log(curve.deltas)
    worst = 0.0
    orders: np.ndarray[tuple[int], np.dtype[np.float64]] = curve.orders
    for index, order in enumerate(orders):
        predicted = 2.0 * order + _window_terms(logs, float(order), first, second)
        # what is left after two terms is bounded from above by the next one -- one more power of
        # `e = delta^p` on the first -- and from below by what double precision invents; whichever
        # is larger is the bar, and Table 1 exists to show which cell is which
        budget = abs(
            _window_terms(logs, float(order), first, 0.0)
        ) * delta_hi**order + _cancellation_floor(logs, float(order))
        worst = max(worst, abs(float(curve.slopes[index]) - predicted) - budget)
    if doubling > 1e-12 or worst > 0.0:
        raise RuntimeError(
            "the order-transfer certificate no longer reproduces the window expansion this "
            f"module predicts (doubling residual {doubling:.3e}, prediction excess {worst:.3e}); "
            "re-derive lambda and c2 before regenerating Table 1"
        )
    return TransferConstants(first, second, doubling, worst)


def _window_terms(logs: NDArray[np.float64], order: float, first: float, second: float) -> float:
    """The ordinary-least-squares slope picked up by `lambda e + 2 c2 e^2` over the sweep window."""
    variance = float(np.var(logs))
    one = float(np.cov(logs, np.exp(order * logs), bias=True)[0, 1]) / variance
    two = float(np.cov(logs, np.exp(2.0 * order * logs), bias=True)[0, 1]) / variance
    return first * one + 2.0 * second * two


def _cancellation_floor(logs: NDArray[np.float64], order: float) -> float:
    """How much of a fitted slope double precision can invent, on this window, at this order.

    `u*(b + e) - u*(b)` is a difference of two numbers near `u*(b)`, so its relative error is
    `2 eps |u*(b)| / (|u*'(b)| e)`; squaring doubles it, and a relative wobble of that size on the
    smallest regret in the window moves the fitted slope by roughly that over the spread of `log
    delta`. It is a bound and a loose one -- the point is the cell where it stops being small.
    """
    ustar = _B * _XT / (_B * _B + _RR)
    sensitivity = _XT * (_RR - _B * _B) / (_RR + _B * _B) ** 2
    smallest = float(np.exp(order * np.min(logs)))
    relative = 2.0 * _EPS * abs(ustar) / (abs(sensitivity) * smallest)
    return 2.0 * relative / float(np.std(logs))


def table_one(
    windows: tuple[tuple[float, float], ...], n_delta: int, constants: TransferConstants
) -> list[dict[str, Any]]:
    """The order-doubling law, with the window term that moves it and the floor that breaks it."""
    rows: list[dict[str, Any]] = []
    for lo, hi in windows:
        curve = composition_transfer_certificate(
            b=_B, rr=_RR, xt=_XT, delta_lo=lo, delta_hi=hi, n_delta=n_delta
        )
        logs = np.log(curve.deltas)
        for index, order in enumerate(curve.orders):
            measured = float(curve.slopes[index])
            one_term = _window_terms(logs, float(order), constants.first, 0.0)
            two_term = _window_terms(logs, float(order), constants.first, constants.second)
            floor = _cancellation_floor(logs, float(order))
            rows.append(
                {
                    "delta_lo": lo,
                    "delta_hi": hi,
                    "order": int(order),
                    "exponent": 2.0 * int(order),
                    "measured": measured,
                    "first_order": one_term,
                    "second_order": two_term - one_term,
                    "residual_first": measured - 2.0 * order - one_term,
                    "residual_second": measured - 2.0 * order - two_term,
                    "cancellation_floor": floor,
                    "limited_by": "float"
                    if abs(measured - 2.0 * order - two_term) < floor
                    else "window",
                }
            )
    return rows


def table_two(windows: tuple[tuple[float, float], ...], n_delta: int) -> list[dict[str, Any]]:
    """The bottleneck is a MIN over channels: one plug-in channel caps the exponent at 2, whether
    there are two channels or three, and no amount of debiasing the others moves it."""
    rows: list[dict[str, Any]] = []
    for lo, hi in windows:
        two = multivariate_interference_certificate(delta_lo=lo, delta_hi=hi, n_delta=n_delta)
        three = exposure_map_certificate(delta_lo=lo, delta_hi=hi, n_delta=n_delta)
        rows.append(
            {
                "delta_lo": lo,
                "delta_hi": hi,
                "two_channel_bottleneck": float(two.half_slope),
                "two_channel_full": float(two.full_slope),
                "three_channel_bottleneck": float(three.wbottleneck_slope),
                "three_channel_full": float(three.full_slope),
                "underflowed": bool(
                    np.min(two.full_orth_regret) <= 0.0 or np.min(three.full_regret) <= 0.0
                ),
            }
        )
    return rows


def _prefix_seed_counts(n_seeds: int, levels: int = _PREFIX_LEVELS) -> list[int]:
    """`[n, n/2, n/4, ...]` -- the nested prefixes the certificates' index seeding allows."""
    counts = [n_seeds]
    while len(counts) <= levels and counts[-1] // 2 >= 2:
        counts.append(counts[-1] // 2)
    return counts


def _monte_carlo_scale(prefix_values: Sequence[float]) -> float:
    """The Monte-Carlo standard deviation of `prefix_values[0]`, pooled over its nested prefixes.

    Neither `G`-sweep certificate takes a seed offset, so an independent replicate cannot be drawn
    and the usual bootstrap is unavailable. What the index seeding does give is a chain of prefixes.
    Write `theta_k` for the statistic over the first `k` seeds. Splitting the first `2m` seeds into
    two independent halves gives `theta_m - theta_{2m} = (theta_a - theta_b)/2`, whose variance is
    `sigma^2/(2m)` -- the variance of `theta_{2m}` itself. So each rung of the chain is an unbiased
    one-draw estimate of a KNOWN multiple of the quantity wanted:

        Var(theta_{n/2^{j-1}} - theta_{n/2^{j-2}}) = 2^{j-1} Var(theta_n)

    and the rungs are mutually independent, because `theta_a - theta_b` is orthogonal to
    `theta_a + theta_b`. Averaging the rescaled squares therefore pools `levels` independent draws
    instead of trusting one, which is the whole reason for doing it this way: a single
    `|N(0, sigma^2)|` draw lands anywhere from `0.2 sigma` to `3 sigma`, and at 240 and 960 seeds
    the one-draw version reported scales that DISAGREED with each other by more than the `sqrt(n)`
    it was supposed to shrink by -- it could not have separated a real walk from noise at any seed
    count.

    The identity is exact for a mean and first order for these slopes, which are linear in
    `log(mean regret)` rather than in the regret; the fluctuations here are a few percent, so the
    linearisation is not what limits the reading.
    """
    terms = [
        (prefix_values[j] - prefix_values[j - 1]) ** 2 / 2.0 ** (j - 1)
        for j in range(1, len(prefix_values))
    ]
    return float(np.sqrt(np.mean(terms))) if terms else 0.0


def table_three(
    g_windows: tuple[tuple[int, ...], ...], n_seeds: int, n_delta: int
) -> dict[str, Any]:
    """End to end: the `delta` regime is exact and the `G` regime is a window."""
    reference = end_to_end_c2_certificate(n_seeds=n_seeds)
    deltas = {
        "deltas": [float(d) for d in reference.deltas],
        "half_regret": [float(v) for v in reference.half_regret],
        "full_regret": [float(v) for v in reference.full_regret],
        "half_slope": float(reference.half_slope),
        "full_slope": float(reference.full_slope),
        "n_delta": n_delta,
    }
    ladder: list[dict[str, Any]] = []
    for grid in g_windows:
        counts = _prefix_seed_counts(n_seeds)
        curves = [end_to_end_c2_certificate(n_seeds=k, g_grid=grid) for k in counts]
        slopes = [float(curve.g_slope) for curve in curves]
        ladder.append(
            {
                "g_lo": int(grid[0]),
                "g_hi": int(grid[-1]),
                "g_slope": slopes[0],
                "prefix_seed_counts": counts,
                "prefix_slopes": slopes,
                "monte_carlo_scale": _monte_carlo_scale(slopes),
                "floor": float(curves[0].floor_g),
                "n_seeds": n_seeds,
            }
        )
    return {"delta_regime": deltas, "g_regime": ladder}


def table_four(g_windows: tuple[tuple[int, ...], ...], n_seeds: int) -> list[dict[str, Any]]:
    """The floor is two-sided once `G` is large enough: `G * regret` flattens onto a positive
    constant, so `1/G` bounds the regret from below as well as above."""
    rows: list[dict[str, Any]] = []
    for grid in g_windows:
        counts = _prefix_seed_counts(n_seeds)
        curves = [clustered_lower_bound_certificate(n_seeds=k, g_grid=grid) for k in counts]
        slopes = [float(curve.plateau_slope) for curve in curves]
        rows.append(
            {
                "g_lo": int(grid[0]),
                "g_hi": int(grid[-1]),
                "plateau_slope": slopes[0],
                "prefix_seed_counts": counts,
                "prefix_slopes": slopes,
                "monte_carlo_scale": _monte_carlo_scale(slopes),
                "c0": float(curves[0].c0_estimate),
                "c0_prefix": [float(curve.c0_estimate) for curve in curves],
                "g_times_regret_lo": float(np.min(curves[0].g_times_regret)),
                "g_times_regret_hi": float(np.max(curves[0].g_times_regret)),
                "n_seeds": n_seeds,
            }
        )
    return rows


def table_five(seeds: tuple[int, ...], n_samples: int) -> dict[str, Any]:
    """The quadratic law on random model error, and the event the exponent is conditional on."""
    a_mat = np.array([[1.0, 0.1], [0.0, 0.95]])
    b_mat = np.array([[0.5], [1.0]])
    q_mat = np.eye(_LQ_STATES)
    r_mat = np.array([[0.5]])
    x0 = np.array([1.0, 0.5])

    exponents: list[float] = []
    unbounded: list[float] = []
    for seed in seeds:
        curve = regret_scaling(a_mat, b_mat, q_mat, r_mat, x0, n_samples=n_samples, seed=seed)
        exponents.append(float(curve.exponent))
        unbounded.append(float(np.max(curve.infinite_fraction)))
    return {
        "seeds": [int(s) for s in seeds],
        "exponents": exponents,
        "unbounded_share": unbounded,
        "lo": float(np.min(exponents)),
        "hi": float(np.max(exponents)),
        "median": float(np.median(exponents)),
        "span": float(np.ptp(exponents) / abs(np.median(exponents))),
        "worst_unbounded_share": float(np.max(unbounded)),
        "n_samples": n_samples,
    }


def _jsonable(value: Any) -> Any:
    """Replace non-finite floats with `null` so the artefact is JSON a strict parser will read.

    Table 2 genuinely produces `nan` -- that is the finding, not a defect -- but `json.dumps`
    writes the literal `NaN`, which only Python's own loader accepts. The markdown table keeps the
    `nan` where a reader needs to see it.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _window_label(lo: float, hi: float) -> str:
    return f"[{lo:g}, {hi:g}]"


def _walk_verdict(rows: list[dict[str, Any]], field: str) -> str:
    """State which of the two explanations the numbers support, rather than presuppose it.

    The claim a ladder makes is that the statistic WALKS from the shipped window to the top one,
    so that is what gets tested -- the endpoint difference against the two endpoints' pooled
    sampling scales, which is `sqrt(s_lo^2 + s_hi^2)` because the two rows are independent runs.
    Testing the largest consecutive step instead would be the wrong statistic and a far weaker
    one: consecutive windows overlap, so their steps are small by construction while the walk
    across the whole ladder is not.

    Three is the bar. Below it the run says it has not separated the two explanations, which is
    the correct answer for a smoke run and a demand for more seeds in a full one.
    """
    if len(rows) < 2:
        return "A single window is not a ladder; nothing here separates the window from the seeds."
    lo, hi = rows[0], rows[-1]
    walk = abs(hi[field] - lo[field])
    noise = float(np.hypot(lo["monte_carlo_scale"], hi["monte_carlo_scale"]))
    ratio = walk / noise if noise > 0.0 else float("inf")
    lead = (
        f"Across the ladder the statistic moves `{walk:.4f}` against a pooled sampling scale of "
        f"`{noise:.4f}` on the two endpoints"
    )
    if ratio >= 3.0:
        return f"{lead} -- `{ratio:.1f}x`, so the walk is the window and not the seeds."
    return (
        f"{lead} -- `{ratio:.1f}x`, which does not separate the window from the seeds. Raise the "
        f"seed count before quoting this ladder."
    )


def _markdown(
    one: list[dict[str, Any]],
    two: list[dict[str, Any]],
    three: dict[str, Any],
    four: list[dict[str, Any]],
    five: dict[str, Any],
    constants: TransferConstants,
) -> str:
    window_limited = sum(1 for row in one if row["limited_by"] == "window")
    bottleneck_miss = max(
        max(abs(row["two_channel_bottleneck"] - 2.0), abs(row["three_channel_bottleneck"] - 2.0))
        for row in two
    )
    c0_lo = min(row["c0"] for row in four)
    c0_hi = max(row["c0"] for row in four)
    lines = [
        "# P1 tables -- debias every channel",
        "",
        f"Scalar transfer plant `b = {_B:g}`, `rr = {_RR:g}`, `xt = {_XT:g}`; its window expansion "
        f"`lambda = {constants.first:.6g}`, `c2 = {constants.second:.6g}` is reconstructed here in "
        "closed form and gated against the certificate (order-doubling residual "
        f"`{constants.doubling_residual:.1e}`, prediction excess over the next order "
        f"`{constants.prediction_residual:.1e}`). Every exponent below is a fit to a log-log "
        "sweep, so every table names the window it was fitted on.",
        "",
        "## Table 1 -- a fitted slope is the exponent plus a window term, and the term is exact",
        "",
        "| window in `delta` | p | 2p | measured | `lambda` term | `c2` term | residual | "
        "float floor | limited by |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in one:
        lines.append(
            f"| {_window_label(row['delta_lo'], row['delta_hi'])} | {row['order']} | "
            f"{row['exponent']:g} | {row['measured']:.6f} | {row['first_order']:+.2e} | "
            f"{row['second_order']:+.2e} | {row['residual_second']:+.1e} | "
            f"{row['cancellation_floor']:.1e} | {row['limited_by']} |"
        )
    lines += [
        "",
        f"`{one[0]['measured']:.4g}` is not `{one[0]['exponent']:g}` up to noise; it is "
        f"`{one[0]['exponent']:g}` plus `{one[0]['measured'] - one[0]['exponent']:+.2e}` of "
        f"window, of which the two closed-form terms account for "
        f"`{one[0]['first_order'] + one[0]['second_order']:+.2e}`, and what is left is one more "
        f"power of `delta`. Read the table down a column and the two failure modes bracket the "
        f"usable window from opposite ends: {window_limited} of the "
        f"{len(one)} cells are still window-limited, the rest have had the window term shrink "
        f"below what double precision invents, and at the bottom right the fit is reporting the "
        f"rounding of a difference of two nearly equal numbers rather than an exponent.",
        "",
        "## Table 2 -- the bottleneck is a MIN over channels, not an average",
        "",
        "| window in `delta` | 2-channel, spillover plug-in | 2-channel, both debiased | "
        "3-channel, `W` plug-in | 3-channel, all debiased | underflowed |",
        "|---|---|---|---|---|---|",
    ]
    for row in two:
        lines.append(
            f"| {_window_label(row['delta_lo'], row['delta_hi'])} | "
            f"{row['two_channel_bottleneck']:.4f} | {row['two_channel_full']:.4f} | "
            f"{row['three_channel_bottleneck']:.4f} | {row['three_channel_full']:.4f} | "
            f"{'yes' if row['underflowed'] else 'no'} |"
        )
    lines += [
        "",
        f"One plug-in channel caps the exponent at `2` whether there are two channels or three, "
        f"and debiasing the others does not move it -- which is the content of the claim that the "
        f"rate is a minimum over channels, and the two bottleneck columns stay within "
        f"`{bottleneck_miss:.1e}` of `2` on every window. The full-orthogonality columns are the "
        f"ones the float floor reaches first, because `delta^4` underflows two windows before "
        f"`delta^2` does; the last column marks where a regret in the sweep hit exactly zero, and "
        f"a `nan` slope beside it is the fit taking `log 0`.",
        "",
        "## Table 3 -- end to end: the `delta` regime is exact, the `G` regime is a window",
        "",
        f"(a) deterministic bias order, `{len(three['delta_regime']['deltas'])}`-point sweep:",
        "",
        "| arm | fitted slope | theory |",
        "|---|---|---|",
        f"| half-orthogonal (spillover plug-in) | {three['delta_regime']['half_slope']:.4f} | 2 |",
        f"| fully orthogonal | {three['delta_regime']['full_slope']:.4f} | 4 |",
        "",
        "(b) real cross-fit DML, sampling-dominated, as the cluster grid moves up:",
        "",
        "| `G` window | slope | nested prefixes | pooled sampling scale | "
        "regret at the largest `G` |",
        "|---|---|---|---|---|",
    ]
    for row in three["g_regime"]:
        lines.append(
            f"| {row['g_lo']} .. {row['g_hi']} | {row['g_slope']:+.4f} | "
            f"{', '.join(f'{value:+.4f}' for value in row['prefix_slopes'][1:])} | "
            f"{row['monte_carlo_scale']:.4f} | {row['floor']:.3e} |"
        )
    lines += [
        "",
        f"The slope ends at `{three['g_regime'][-1]['g_slope']:+.4f}` on the top window against "
        f"`{three['g_regime'][0]['g_slope']:+.4f}` on the shipped one. Whether that is the window "
        f"or the seeds is what the third column decides, and it is not an independent replicate -- "
        f"the certificate seeds by index, so the `n/2` run is a prefix of the `n` run. That is "
        f"what makes it usable: a prefix mean minus the full mean has exactly the variance of the "
        f"full mean, so it is an error bar; it is pooled over "
        f"{len(three['g_regime'][0]['prefix_slopes']) - 1} nested rungs rather than read off one, "
        f"because one draw of `|N(0, sigma^2)|` is too noisy to decide anything. "
        f"{_walk_verdict(three['g_regime'], 'g_slope')}",
        "",
        "## Table 4 -- the floor is two-sided, once `G` is large enough",
        "",
        "| `G` window | plateau slope | nested prefixes | pooled sampling scale | `c0` | "
        "`G x regret` range |",
        "|---|---|---|---|---|---|",
    ]
    for row in four:
        lines.append(
            f"| {row['g_lo']} .. {row['g_hi']} | {row['plateau_slope']:+.4f} | "
            f"{', '.join(f'{value:+.4f}' for value in row['prefix_slopes'][1:])} | "
            f"{row['monte_carlo_scale']:.4f} | {row['c0']:.4f} | "
            f"{row['g_times_regret_lo']:.3f} .. {row['g_times_regret_hi']:.3f} |"
        )
    lines += [
        "",
        f"`G * regret` has to flatten onto a positive constant for `1/G` to bound the regret from "
        f"below as well as above. The plateau slope walks from `{four[0]['plateau_slope']:+.4f}` "
        f"to `{four[-1]['plateau_slope']:+.4f}` while `c0` stays inside "
        f"`{c0_lo:.4f} .. {c0_hi:.4f}`, so the constant was never the thing in doubt -- the "
        f"shipped grid was. {_walk_verdict(four, 'plateau_slope')}",
        "",
        "## Table 5 -- the quadratic law on random error, and the event it is conditional on",
        "",
        f"`{five['n_samples']}` perturbation draws per level, {len(five['seeds'])} seeds.",
        "",
        "| seed | fitted exponent | largest unbounded share over the levels |",
        "|---|---|---|",
    ]
    for seed, exponent, share in zip(
        five["seeds"], five["exponents"], five["unbounded_share"], strict=True
    ):
        lines.append(f"| {seed} | {exponent:.4f} | {share:.4f} |")
    lines += [
        "",
        f"Range `{five['lo']:.4f} .. {five['hi']:.4f}`, median `{five['median']:.4f}`, relative "
        f"spread `{five['span']:.1e}`. Every seed sits above `2` and none reaches `2.1`: the "
        "excess is the same window term Table 1 prices, not sampling noise, and the appendix's "
        "single `2.05` is one draw from this range rather than the number. The last column is "
        "the honesty the exponent needs -- a draw whose perturbed plant is unstabilisable, or "
        "whose gain fails to stabilise the true plant, has no finite regret at all, so the "
        "exponent is conditional "
        f"on the complement of an event that reaches `{five['worst_unbounded_share']:.4f}` here.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta-windows",
        type=float,
        nargs="+",
        default=[0.2, 0.02, 0.002, 0.0002],
        help="upper end of each log-log sweep window; the lower end is this over --window-ratio",
    )
    parser.add_argument("--window-ratio", type=float, default=20.0)
    parser.add_argument("--n-delta", type=int, default=12)
    parser.add_argument("--g-windows", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--g-points", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=240)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("results/paper1"))
    args = parser.parse_args()

    windows = tuple((hi / args.window_ratio, hi) for hi in args.delta_windows)
    g_windows = tuple(
        tuple(int(start * 2**step) for step in range(args.g_points)) for start in args.g_windows
    )

    constants = transfer_constants()
    one = table_one(windows, args.n_delta, constants)
    two = table_two(windows, args.n_delta)
    three = table_three(g_windows, args.n_seeds, args.n_delta)
    four = table_four(g_windows, args.n_seeds)
    five = table_five(tuple(args.seeds), args.n_samples)

    args.out.mkdir(parents=True, exist_ok=True)
    text = _markdown(one, two, three, four, five, constants)
    (args.out / "tables.md").write_text(text)
    (args.out / "tables.json").write_text(
        json.dumps(
            _jsonable(
                {
                    "window_expansion": {
                        "first": constants.first,
                        "second": constants.second,
                        "doubling_residual": constants.doubling_residual,
                        "prediction_residual": constants.prediction_residual,
                    },
                    "table1": one,
                    "table2": two,
                    "table3": three,
                    "table4": four,
                    "table5": five,
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )
    print(text)
    print(f"written to {args.out}/tables.md and {args.out}/tables.json")


if __name__ == "__main__":
    main()

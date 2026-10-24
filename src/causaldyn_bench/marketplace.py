"""Track H: offline causal control under equilibrium interference (the marketplace moat).

Learn an incentive policy from *confounded, switchback-logged* marketplace data where the logging
policy chased demand (so demand confounds treatment and outcome) and drivers are mobile (SUTVA fails
through a shared congestion equilibrium -- incentivising a zone cannibalises its neighbours). The
method lives in ``chc.marketplace`` (grounded in Munro-Wager-Xu market equilibrium, shared-state DML
arXiv:2504.08836, pessimism-as-DR-MDP); this track scores it.

The leaderboard scores one market, ``SharedStateMarket(seed=0)``. ``python -m
causaldyn_bench.marketplace`` (``just track-h``) scores it with seeds 1-29 into
``results/track_h_markets.{md,json}``, and again under ``JAX_ENABLE_X64=1``, where each seed draws
another market, into ``results/track_h_markets_x64.{md,json}``. On every one of those 60 markets
the equilibrium-aware, pessimistic CHC allocation has the least regret, and it wins
0.74-1.01 of the oracle's lift over doing nothing, a median of 0.93 and 0.92 at the two precisions
(the oracle's ascent is local, so a plan can pass it). The predictive (MOPO) and naive-causal
planners assume SUTVA and leave large regret: the naive one wins a median of 0.30 and 0.36 of the
lift at the two precisions, at most 0.84, and the predictive one 0.04 and 0.00, at most 0.36; the
naive one realises less than doing nothing on 8 and 7 of the 30 markets, the predictive one on 13
and 15. :func:`marketplace_report` reads what the logs' confounding does to the responses the two
plan on.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from chc.marketplace import (
    SharedStateMarket,
    calibrate_naive_causal,
    calibrate_predictive,
    calibrate_shared_state,
    fit_equilibrium,
    interference_bias,
    pessimistic_equilibrium_allocation,
    sutva_allocation,
)

from causaldyn_bench.tracks import TrackResult

TRACK = "H-marketplace"
SEEDS = tuple(range(30))
BLOCKS = 400
ORACLE_STEPS = 2500
NAMES = {False: "track_h_markets", True: "track_h_markets_x64"}  # a run's file name, by x64
# each planner's name on the leaderboard, and the prefix of its keys in a report
PLANNERS = {"predictive-MOPO": "predictive", "naive-causal": "naive", "equilibrium-CHC": "chc"}


def _allocations(market: SharedStateMarket, logs: dict, radius: float = 1.0) -> dict:
    """The three offline planners' allocations from the same confounded logs."""
    return {
        "predictive-MOPO": sutva_allocation(market, calibrate_predictive(logs), radius),
        "naive-causal": sutva_allocation(market, calibrate_naive_causal(logs), radius),
        "equilibrium-CHC": pessimistic_equilibrium_allocation(
            market, calibrate_shared_state(logs), radius=radius, equilibrium=fit_equilibrium(logs)
        ),
    }


def chase(market: SharedStateMarket, demand: jax.Array, total: jax.Array) -> jax.Array:
    """The logging policy of :meth:`SharedStateMarket.generate_logs`: each block's total incentive
    ``total``, shape ``(n_blocks, 1)``, split across the zones by a softmax of the demand it
    chases, ``(n_blocks, n_zones)``."""
    return total * jax.nn.softmax(market.confounding * demand, axis=-1)


def rerun(market: SharedStateMarket, logs: dict, u: jax.Array) -> dict[str, jax.Array]:
    """The logged blocks under the incentives ``u``: each block keeps its demand, the
    attractiveness that moves with it and its completions noise, and its drivers settle anew. The
    shared exposure aggregate is the new incentives' total, as the logs record it."""
    base_demand, base_attract = market._base()
    attract = base_attract * logs["demand"] / base_demand
    settle = jax.vmap(market._equilibrium)
    noise = logs["y"] - jnp.minimum(logs["demand"], settle(attract, logs["u"]))
    y = jnp.minimum(logs["demand"], settle(attract, u)) + noise
    aggregate = jnp.broadcast_to(jnp.sum(u, axis=1, keepdims=True), u.shape)
    return {"u": u, "demand": logs["demand"], "aggregate": aggregate, "y": y}


def _completely_adjusted(market: SharedStateMarket, logs: dict, order: np.ndarray) -> jax.Array:
    """Each zone's naive-causal slope on the logged blocks rerun with that zone's incentive chasing
    its own logged demand and, in every other zone, the demand of the block ``order`` puts in its
    place. Its own demand is then the zone's only confounder, and adjusting for it is complete."""
    total = logs["aggregate"][:, :1]
    slopes = []
    for zone in range(market.n_zones):
        chased = logs["demand"][order].at[:, zone].set(logs["demand"][:, zone])
        rerun_logs = rerun(market, logs, chase(market, chased, total))
        slopes.append(calibrate_naive_causal(rerun_logs).marginal[zone])
    return jnp.stack(slopes)


def _correlation(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.corrcoef(a, b)[0, 1])


def marketplace_report(seed: int = 0, n_blocks: int = BLOCKS) -> dict[str, float]:
    """One market: the confounding in the calibrated responses, and each planner's realised lift,
    regret and the naive planner's interference over-count, against the no-incentive baseline and
    the oracle.

    The confounding is read against references the logs cannot give, built from the market. The
    *randomised twin* reruns the logged blocks with each block's incentives another block's (a
    fixed permutation), so the incentives no longer follow the block's demand, and reads it with
    the naive-causal regression. ``predictive_confounding_bias`` and ``naive_confounding_bias`` are
    the predictive and the naive-causal response's mean excess over the twin's, across the zones:
    how far the logging policy's chase of demand moves each. ``naive_cross_zone_confounding`` is the
    naive-causal response's mean excess over the slope its regression reads where adjusting each
    zone for its own demand is complete (:func:`_completely_adjusted`): the confounding that runs
    through the other zones' demand, which the logging policy also chases.
    ``chc_confounding_bias`` is equilibrium-CHC's response, adjusted for each zone's own demand and
    the block's total incentive, less its own regression on the twin. At a fixed total and own
    demand a zone's incentive moves only with the other zones' demand, so holding the total takes
    out the randomised budget and keeps the confounding; the planner reads the response only
    through its pessimism tilt.

    ``randomised_demand_corr`` and ``naive_demand_corr`` are the twin's and the naive-causal
    response's correlation with the zones' base demand. The first is why a correlation is no test
    of de-confounding here: with the confounding removed, the response still follows demand.
    """
    market = SharedStateMarket(seed=seed)
    demand, _ = market._base()
    logs = market.generate_logs(n_blocks, jax.random.key(seed + 1))
    base = market.value(jnp.zeros(market.n_zones))
    oracle = market.value(market.oracle_allocation(steps=ORACLE_STEPS))
    allocations = _allocations(market, logs)
    values = {name: market.value(u) for name, u in allocations.items()}
    predictive, naive = calibrate_predictive(logs), calibrate_naive_causal(logs)
    shared = calibrate_shared_state(logs)
    order = np.random.default_rng(seed).permutation(n_blocks)
    twin = rerun(market, logs, logs["u"][order])
    randomised = calibrate_naive_causal(twin).marginal
    complete = _completely_adjusted(market, logs, order)
    return {
        "oracle_lift": oracle - base,
        "predictive_confounding_bias": float(jnp.mean(predictive.marginal - randomised)),
        "naive_confounding_bias": float(jnp.mean(naive.marginal - randomised)),
        "naive_cross_zone_confounding": float(jnp.mean(naive.marginal - complete)),
        "chc_confounding_bias": float(
            jnp.mean(shared.marginal - calibrate_shared_state(twin).marginal)
        ),
        "randomised_demand_corr": _correlation(randomised, demand),
        "naive_demand_corr": _correlation(naive.marginal, demand),
        **{f"{key}_realised_lift": values[name] - base for name, key in PLANNERS.items()},
        **{f"{key}_regret": oracle - values[name] for name, key in PLANNERS.items()},
        "naive_interference_overcount": interference_bias(
            market, naive, allocations["naive-causal"]
        ),
    }


def track_marketplace(seed: int = 0, n_blocks: int = BLOCKS) -> list[TrackResult]:
    """Score realised-completions regret vs the equilibrium-aware oracle for each planner."""
    report = marketplace_report(seed, n_blocks)
    return [
        TrackResult(TRACK, name, "regret", report[f"{key}_regret"])
        for name, key in PLANNERS.items()
    ]


def readings(report: dict[str, float]) -> dict[str, float]:
    """A report's readings the 30-market tables summarise: the lifts as shares of the oracle's,
    the naive-causal bias as a share of the predictive one, and the rest as reported."""
    lift = report["oracle_lift"]
    return {
        "chc_share": report["chc_realised_lift"] / lift,
        "naive_share": report["naive_realised_lift"] / lift,
        "predictive_share": report["predictive_realised_lift"] / lift,
        "chc_regret": report["chc_regret"],
        "naive_regret": report["naive_regret"],
        "predictive_regret": report["predictive_regret"],
        "predictive_confounding_bias": report["predictive_confounding_bias"],
        "naive_confounding_bias": report["naive_confounding_bias"],
        "naive_kept_share": report["naive_confounding_bias"]
        / report["predictive_confounding_bias"],
        "naive_cross_zone_confounding": report["naive_cross_zone_confounding"],
        "chc_confounding_bias": report["chc_confounding_bias"],
        "randomised_demand_corr": report["randomised_demand_corr"],
        "naive_demand_corr": report["naive_demand_corr"],
        "naive_interference_overcount": report["naive_interference_overcount"],
        "oracle_lift": lift,
    }


LABELS = {
    "chc_share": "equilibrium-CHC's share of the oracle's lift",
    "naive_share": "naive-causal's share of the oracle's lift",
    "predictive_share": "predictive-MOPO's share of the oracle's lift",
    "chc_regret": "equilibrium-CHC's regret",
    "naive_regret": "naive-causal's regret",
    "predictive_regret": "predictive-MOPO's regret",
    "predictive_confounding_bias": "predictive response less the randomised twin's",
    "naive_confounding_bias": "naive-causal response less the randomised twin's",
    "naive_kept_share": "naive-causal bias as a share of the predictive one",
    "naive_cross_zone_confounding": "naive-causal response less the completely adjusted one",
    "chc_confounding_bias": "equilibrium-CHC's response less its own on the randomised twin",
    "randomised_demand_corr": "randomised twin's response, correlation with demand",
    "naive_demand_corr": "naive-causal response, correlation with demand",
    "naive_interference_overcount": "naive-causal interference over-count",
    "oracle_lift": "oracle's lift over doing nothing",
}


def summary(values: Sequence[float]) -> dict[str, float]:
    """Quantiles, mean and sample SD of one reading over the markets."""
    a = np.asarray(values, dtype=float)
    q25, median, q75 = np.quantile(a, [0.25, 0.5, 0.75])
    return {
        "min": float(a.min()),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)),
    }


def counts(markets: Sequence[dict[str, Any]]) -> dict[str, int]:
    """The orderings and signs the prose quotes, as counts of markets."""
    read = [readings(m["report"]) for m in markets]
    return {
        "markets": len(read),
        "chc_least_regret": sum(
            r["chc_regret"] < min(r["naive_regret"], r["predictive_regret"]) for r in read
        ),
        "predictive_confounding_bias_positive": sum(
            r["predictive_confounding_bias"] > 0.0 for r in read
        ),
        "naive_bias_below_predictive": sum(
            r["naive_confounding_bias"] < r["predictive_confounding_bias"] for r in read
        ),
        "naive_confounding_bias_positive": sum(r["naive_confounding_bias"] > 0.0 for r in read),
        "naive_cross_zone_confounding_positive": sum(
            r["naive_cross_zone_confounding"] > 0.0 for r in read
        ),
        "chc_bias_above_naive": sum(
            r["chc_confounding_bias"] > r["naive_confounding_bias"] for r in read
        ),
        "naive_interference_overcount_positive": sum(
            r["naive_interference_overcount"] > 0.0 for r in read
        ),
        "naive_below_doing_nothing": sum(r["naive_share"] < 0.0 for r in read),
        "predictive_below_doing_nothing": sum(r["predictive_share"] < 0.0 for r in read),
    }


def record(markets: Sequence[dict[str, Any]], n_blocks: int, x64: bool) -> dict[str, Any]:
    """A run: its markets' reports, and the summary and counts they read."""
    read = [readings(m["report"]) for m in markets]
    return {
        "track": TRACK,
        "x64": x64,
        "blocks": n_blocks,
        "oracle_steps": ORACLE_STEPS,
        "twin": "numpy default_rng(seed).permutation(blocks)",
        "summary": {key: summary([r[key] for r in read]) for key in LABELS},
        "counts": counts(markets),
        "markets": list(markets),
    }


def _markdown(result: dict[str, Any]) -> str:
    c = result["counts"]
    n = c["markets"]
    other = NAMES[not result["x64"]]
    precision = "float64, under `JAX_ENABLE_X64=1`" if result["x64"] else "float32, JAX's default"
    seed0 = next((m["report"] for m in result["markets"] if m["seed"] == 0), None)
    if result["x64"]:
        leaderboard = " The leaderboard, in float32, scores the other run's seed 0."
    elif seed0 is not None:
        leaderboard = (
            " The leaderboard scores seed 0, one of these: equilibrium-CHC's regret on it reads "
            f"{seed0['chc_regret']:.4f}, naive-causal's {seed0['naive_regret']:.4f} and "
            f"predictive-MOPO's {seed0['predictive_regret']:.4f}."
        )
    else:
        leaderboard = ""
    lines = [
        f"# Track H over {n} markets",
        "",
        f"`SharedStateMarket(seed=s)` for {n} seeds, each scored on {result['blocks']} logged "
        f"blocks drawn from `jax.random.key(s + 1)`, in {precision} "
        "(`causaldyn_bench.marketplace`, `just track-h`). The other precision is "
        f"`{other}.md`: a seed draws another market there, so the two runs are {2 * n} markets."
        + leaderboard,
        "",
        "A planner's share is its lift over doing nothing as a share of the oracle's "
        f"(`oracle_allocation(steps={result['oracle_steps']})`), its regret the oracle's "
        "completions less its own; the oracle's ascent is local, so a plan can pass it. The "
        "naive-causal planner's interference over-count is the lift its per-zone response predicts "
        "for its allocation less the lift the market realises.",
        "",
        "The confounding is read against two reruns of the logged blocks, which no log gives: each "
        "block keeps its demand, the attractiveness that moves with it and its completions noise, "
        "and its drivers settle anew under other incentives. In the *randomised twin* each block "
        "takes another block's incentives (`numpy.random.default_rng(s).permutation`), which no "
        "longer follow its demand; the naive-causal regression on it is the reference, and a "
        "response's bias is its mean over the zones less the reference's. In the *complete "
        "adjustment* a zone's incentive chases its own logged demand and, in every other zone, the "
        "permuted block's, so its own demand is its only confounder and adjusting for it is "
        "complete. The logging policy splits one budget by a softmax of every zone's demand, so "
        "the per-zone adjustment cannot block the other zones' demand, which moves a zone's "
        "incentive and draws its drivers: the naive-causal response's excess over the complete "
        "adjustment is that confounding. Equilibrium-CHC's response, adjusted for each zone's own "
        "demand and the block's total incentive, is read against its own regression on the twin: "
        "at a fixed total and own demand a zone's incentive moves only with the other zones' "
        "demand, so holding the total takes out the randomised budget and keeps the confounding. "
        "The planner reads that response only through its pessimism tilt. A response's correlation "
        "with the zones' demand is no test of de-confounding: the randomised twin's response "
        "follows demand too.",
        "",
        "| reading | min | 25% | median | 75% | max | mean | SD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, label in LABELS.items():
        s = result["summary"][key]
        lines.append(
            f"| {label} | {s['min']:.3f} | {s['q25']:.3f} | {s['median']:.3f} | {s['q75']:.3f} "
            f"| {s['max']:.3f} | {s['mean']:.3f} | {s['sd']:.3f} |"
        )
    lines += [
        "",
        f"- equilibrium-CHC has the least regret on {c['chc_least_regret']} of the {n} markets.",
        "- The predictive response exceeds the randomised twin's on "
        f"{c['predictive_confounding_bias_positive']}. The naive-causal response exceeds it on "
        f"{c['naive_confounding_bias_positive']}, by less than the predictive one on "
        f"{c['naive_bias_below_predictive']}, and exceeds the complete adjustment on "
        f"{c['naive_cross_zone_confounding_positive']}. Equilibrium-CHC's response exceeds its "
        f"own on the twin by more than the naive-causal one does on {c['chc_bias_above_naive']}.",
        "- The naive-causal planner predicts more lift than it realises on "
        f"{c['naive_interference_overcount_positive']}, and realises less than doing nothing on "
        f"{c['naive_below_doing_nothing']}; the predictive planner on "
        f"{c['predictive_below_doing_nothing']}.",
        "",
        "`tests/test_marketplace.py` asserts on seeds 0-2 only what every market of the two runs "
        "supports: the orderings and signs above that hold on all of them, and bounds at the "
        "markets' mean less or plus three SDs, at whichever precision reads wider, rounded outward "
        "to two decimals.",
    ]
    return "\n".join(lines) + "\n"


def leaderboard_note(results: Path = Path("results")) -> str | None:
    """The leaderboard's paragraph under Track H: where its one market, seed 0 in float32, sits
    among the committed runs' markets. ``None`` when a run is not there to read."""
    paths = [results / f"{NAMES[x64]}.json" for x64 in (False, True)]
    if not all(path.exists() for path in paths):
        return None
    run, run_x64 = (json.loads(path.read_text()) for path in paths)
    n = run["counts"]["markets"]
    regrets = {
        key: [m["report"][f"{key}_regret"] for m in run["markets"]] for key in PLANNERS.values()
    }
    seed0 = next(i for i, m in enumerate(run["markets"]) if m["seed"] == 0)
    largest = all(regrets[key][seed0] == max(regrets[key]) for key in ("naive", "predictive"))
    where = (
        ", and the one of them on which both SUTVA planners leave their largest regret"
        if largest
        else ""
    )

    def on(counts: dict[str, int]) -> str:
        least = counts["chc_least_regret"]
        return "every one" if least == counts["markets"] else f"{least} of them"

    spans = ", ".join(
        f"{name} from {min(regrets[key]):.3f} to {max(regrets[key]):.3f} "
        f"(median {float(np.median(regrets[key])):.3f})"
        for name, key in reversed(PLANNERS.items())
    )
    return (
        f"The market is `SharedStateMarket(seed=0)`, one of the {n} that "
        f"`results/{NAMES[False]}.md` scores in float32 (`just track-h`){where}. Over the {n}, "
        f"equilibrium-CHC has the least regret on {on(run['counts'])}; the regrets run {spans}. "
        f"On {run_x64['counts']['markets']} markets drawn under `JAX_ENABLE_X64=1` "
        f"(`results/{NAMES[True]}.md`) equilibrium-CHC has the least regret on "
        f"{on(run_x64['counts'])}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--blocks", type=int, default=BLOCKS)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    x64 = bool(jnp.zeros(()).dtype == jnp.float64)
    markets = []
    for seed in args.seeds:
        markets.append({"seed": seed, "report": marketplace_report(seed, args.blocks)})
        print(f"seed {seed}: {json.dumps(markets[-1]['report'])}", flush=True)
    result = record(markets, args.blocks, x64)
    name = NAMES[x64]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.md").write_text(_markdown(result))
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"written to {args.out}/{name}.md and {args.out}/{name}.json")


if __name__ == "__main__":
    main()

"""Track H: offline causal control under equilibrium interference (the marketplace moat, plans/20).

Learn an incentive policy from *confounded, switchback-logged* marketplace data where the logging
policy chased demand (so demand confounds treatment and outcome) and drivers are mobile (SUTVA fails
through a shared congestion equilibrium -- incentivising a zone cannibalises its neighbours). The
predictive (MOPO) and naive-causal planners assume SUTVA and leave large regret -- often below doing
nothing; the de-confounded, equilibrium-aware, pessimistic CHC allocation recovers the oracle. The
method lives in ``chc.marketplace`` (grounded in Munro-Wager-Xu market equilibrium, shared-state DML
arXiv:2504.08836, pessimism-as-DR-MDP); this track scores it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from chc.marketplace import (
    SharedStateMarket,
    calibrate_naive_causal,
    calibrate_predictive,
    calibrate_shared_state,
    interference_bias,
    pessimistic_equilibrium_allocation,
    sutva_allocation,
)

from causaldyn_bench.tracks import TrackResult


def _allocations(market: SharedStateMarket, logs: dict, radius: float = 1.0) -> dict:
    """The three offline planners' allocations from the same confounded logs."""
    return {
        "predictive-MOPO": sutva_allocation(market, calibrate_predictive(logs), radius),
        "naive-causal": sutva_allocation(market, calibrate_naive_causal(logs), radius),
        "equilibrium-CHC": pessimistic_equilibrium_allocation(
            market, calibrate_shared_state(logs), radius=radius
        ),
    }


def track_marketplace(seed: int = 0, n_blocks: int = 400) -> list[TrackResult]:
    """Score realised-completions regret vs the equilibrium-aware oracle for each planner."""
    market = SharedStateMarket(seed=seed)
    logs = market.generate_logs(n_blocks, jax.random.key(seed + 1))
    oracle = market.value(market.oracle_allocation(steps=2500))
    return [
        TrackResult("H-marketplace", name, "regret", oracle - market.value(u))
        for name, u in _allocations(market, logs).items()
    ]


def marketplace_report(seed: int = 0, n_blocks: int = 400) -> dict[str, float]:
    """Diagnostics: the confounding bias in the calibrated response and each planner's realised lift
    and interference over-count, against the no-incentive baseline and the oracle.
    """
    market = SharedStateMarket(seed=seed)
    demand, _ = market._base()
    logs = market.generate_logs(n_blocks, jax.random.key(seed + 1))
    base = market.value(jnp.zeros(market.n_zones))
    oracle = market.value(market.oracle_allocation(steps=2500))
    pred_resp, naive_resp = calibrate_predictive(logs), calibrate_naive_causal(logs)
    allocs = _allocations(market, logs)
    naive_alloc = allocs["naive-causal"]
    return {
        "oracle_lift": oracle - base,
        "predictive_confounding_corr": float(jnp.corrcoef(pred_resp.marginal, demand)[0, 1]),
        "naive_deconfounded_corr": float(jnp.corrcoef(naive_resp.marginal, demand)[0, 1]),
        "predictive_realised_lift": market.value(allocs["predictive-MOPO"]) - base,
        "naive_realised_lift": market.value(naive_alloc) - base,
        "chc_realised_lift": market.value(allocs["equilibrium-CHC"]) - base,
        "naive_interference_overcount": interference_bias(market, naive_resp, naive_alloc),
    }

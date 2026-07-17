"""Interference flagship: differentiable Stackelberg zone-incentive game (SUTVA breaks on markets).

Naive causal uplift assumes one zone's incentive does not change other zones' state (SUTVA). In a
marketplace it does -- incentivising a zone pulls drivers *from neighbours*, so a per-zone uplift
over-states the effect and over-allocates budget ("the incentive eats its own advantage"). Here the
platform is a Stackelberg leader over a congestion game of mobile drivers; the equilibrium is a
softmax congestion fixed point, and the leader's allocation is optimised by differentiable bilevel
gradient ascent through it (cf. Stackelberg Congestion Games, arXiv 2209.07618). Equilibrium-aware
CHC realises more rides than the SUTVA-naive one -- which predicts a lift it never realises.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from chc.games import softmax_congestion_equilibrium, stackelberg_allocation

from causaldyn_bench.tracks import TrackResult


@dataclass(frozen=True)
class ZoneIncentiveGame:
    """Mobile drivers best-respond to per-zone incentives; the platform allocates a fixed budget."""

    n_zones: int = 8
    driver_mass: float = 8.0
    budget: float = 4.0
    beta: float = 2.5  # driver price-sensitivity (sharpness of the migration response)
    congestion: float = 2.0  # crowding penalty (more drivers in a zone -> lower per-driver value)
    seed: int = 0

    def _demand_attract(self) -> tuple[jax.Array, jax.Array]:
        """Per-zone demand and base attractiveness, anti-correlated to bait the naive allocator."""
        k = jax.random.split(jax.random.key(self.seed), 2)
        demand = 0.4 + 1.6 * jax.random.uniform(k[0], (self.n_zones,))
        # zones drivers already flock to (high attractiveness) tend to be the *already-served* ones,
        # so pulling more drivers there via incentives mostly cannibalises supply.
        attract = 1.5 * demand + 0.3 * jax.random.normal(k[1], (self.n_zones,))
        return demand, attract

    def equilibrium(self, u: jax.Array, iters: int = 120) -> jax.Array:
        """Follower equilibrium via ``chc.games`` (softmax congestion fixed point) given ``u``."""
        _, attract = self._demand_attract()
        return softmax_congestion_equilibrium(
            attract, u, self.congestion, self.driver_mass, self.beta, iters
        )

    def completions(self, u: jax.Array) -> jax.Array:
        """Total completed rides = sum of min(demand, drivers) over zones at the equilibrium."""
        demand, _ = self._demand_attract()
        return jnp.sum(jnp.minimum(demand, self.equilibrium(u)))

    def naive_allocation(self) -> jax.Array:
        """SUTVA allocation: budget split by each zone's local uplift (d compl_i / d u_i)."""
        zero = jnp.zeros(self.n_zones)
        demand, _ = self._demand_attract()

        def local_completion(u: jax.Array, i: int) -> jax.Array:
            return jnp.minimum(demand[i], self.equilibrium(u)[i])

        grads = jnp.stack([jax.grad(local_completion)(zero, i)[i] for i in range(self.n_zones)])
        weights = jnp.maximum(grads, 0.0)
        return self.budget * weights / (jnp.sum(weights) + 1e-9)

    def optimal_allocation(self, steps: int = 400) -> jax.Array:
        """Equilibrium-aware allocation: bilevel ascent on completions through chc.games."""
        return stackelberg_allocation(self.completions, self.n_zones, self.budget, steps=steps)

    def run(self, steps: int = 400) -> list[TrackResult]:
        """Score no-incentive / naive / equilibrium-CHC by realised-completions regret."""
        base = float(self.completions(jnp.zeros(self.n_zones)))
        u_naive = self.naive_allocation()
        u_chc = self.optimal_allocation(steps=steps)
        u_oracle = self.optimal_allocation(steps=steps * 3)
        realised = {
            "no-incentive": base,
            "naive-uplift": float(self.completions(u_naive)),
            "equilibrium-CHC": float(self.completions(u_chc)),
        }
        oracle = float(self.completions(u_oracle))
        return [
            TrackResult("D-interference", name, "regret", oracle - value)
            for name, value in realised.items()
        ]


def interference_report(steps: int = 400) -> dict[str, float]:
    """Diagnostics: what the naive allocator predicts vs realises (the interference bias)."""
    game = ZoneIncentiveGame()
    base = float(game.completions(jnp.zeros(game.n_zones)))
    u_naive = game.naive_allocation()
    zero = jnp.zeros(game.n_zones)
    demand, _ = game._demand_attract()
    local_grads = jnp.stack(
        [
            jax.grad(lambda u, i=i: jnp.minimum(demand[i], game.equilibrium(u)[i]))(zero)[i]
            for i in range(game.n_zones)
        ]
    )
    predicted_lift = float(jnp.sum(local_grads * u_naive))  # the naive SUTVA belief
    realised_lift = float(game.completions(u_naive)) - base
    chc_lift = float(game.completions(game.optimal_allocation(steps=steps))) - base
    return {
        "naive_predicted_lift": predicted_lift,
        "naive_realised_lift": realised_lift,
        "chc_realised_lift": chc_lift,
        "interference_bias": predicted_lift - realised_lift,  # SUTVA over-count (completions)
    }


def track_interference(steps: int = 400) -> list[TrackResult]:
    """Track-D task: zone-incentive allocation under driver migration (naive vs equilibrium)."""
    return ZoneIncentiveGame().run(steps=steps)

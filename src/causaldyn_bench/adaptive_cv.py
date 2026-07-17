"""Adaptive video-inference compute control: split a shared GPU budget across streams over time.

Each camera has a known, bursty arrival schedule and a priority (some critical, some cheap). Every
tick the controller splits a fixed compute budget ``B`` across streams; compute drives throughput
(frames processed), unprocessed frames queue up (latency) and eventually drop. The queue dynamics
are known, the budget is a hard constraint, and the objective is a weighted decision cost.

The point is *priority-aware, constrained* planning: a naive split proportional to current backlog
is blind to which streams matter, so big cheap bursts crowd out small critical ones. An MPC that
minimises the weighted cost under the budget serves the critical streams and wins -- the user-unique
benchmark from ``plans/15`` (adaptive-CV-compute), fully synthetic: the queueing model is the task.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from causaldyn_bench.tracks import TrackResult


def _project_simplex(v: jax.Array, z: float) -> jax.Array:
    """Euclidean projection of ``v`` onto ``{u >= 0, sum u = z}`` (Duchi et al. 2008)."""
    n = v.shape[0]
    sorted_v = jnp.sort(v)[::-1]
    cssv = jnp.cumsum(sorted_v) - z
    ind = jnp.arange(n) + 1
    rho = jnp.count_nonzero(sorted_v - cssv / ind > 0)
    theta = cssv[rho - 1] / rho
    return jnp.maximum(v - theta, 0.0)


@dataclass(frozen=True)
class AdaptiveCVTask:
    """Dynamic GPU-budget allocation across video streams under bursty, known arrivals."""

    n_streams: int = 4
    horizon: int = 24
    budget: float = 4.0  # total compute per tick (tight: < sum of peak demand)
    throughput: float = 6.0  # frames processed per unit compute
    q_max: float = 25.0  # queue cap; overflow is dropped frames
    c_latency: float = 1.0  # cost per queued frame per tick
    c_drop: float = 8.0  # cost per dropped frame
    base_arrivals: float = 1.5
    burst_height: float = 10.0
    crit_weight: float = 6.0  # latency/drop cost multiplier for the critical (even-indexed) streams
    seed: int = 0

    def weights(self) -> jax.Array:
        """Per-stream priority: even-indexed streams are critical (high cost), odd are cheap."""
        idx = jnp.arange(self.n_streams)
        return jnp.where(idx % 2 == 0, self.crit_weight, 1.0)

    def arrivals(self) -> jax.Array:
        """Known schedule ``(horizon, n_streams)``: base load + overlapping bursts sized *inversely*
        to priority, so a size-greedy split serves cheap bursts over small critical streams."""
        t = jnp.arange(self.horizon)[:, None]
        idx = jnp.arange(self.n_streams)
        centers = (self.horizon / 2 + 1.5 * jnp.where(idx % 2 == 0, -1.0, 1.0))[
            None, :
        ]  # overlapping
        size = jnp.where(
            idx % 2 == 0, 0.5, 1.7
        )  # critical streams burst small, cheap streams burst big
        bursts = self.burst_height * size * jnp.exp(-0.5 * ((t - centers) / 1.6) ** 2)
        return self.base_arrivals + bursts

    def _cost_of_open_loop(self, allocations: jax.Array) -> jax.Array:
        """Cumulative cost of a full ``(horizon, n_streams)`` allocation plan."""
        arrivals = self.arrivals()
        weights = self.weights()

        def step(queue: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
            load = queue + arrivals[t]
            processed = jnp.minimum(load, self.throughput * allocations[t])
            remaining = load - processed
            dropped = jnp.maximum(remaining - self.q_max, 0.0)
            queue_next = jnp.minimum(remaining, self.q_max)
            cost = jnp.sum(weights * (self.c_latency * queue_next + self.c_drop * dropped))
            return queue_next, cost

        _, costs = jax.lax.scan(step, jnp.zeros(self.n_streams), jnp.arange(self.horizon))
        return jnp.sum(costs)

    def _cost_of_policy(self, policy: str) -> float:
        """Simulate a closed-loop myopic/uniform policy that sees only the current state."""
        arrivals = self.arrivals()
        weights = self.weights()

        def step(queue: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
            load = queue + arrivals[t]
            if policy == "uniform":
                alloc = jnp.full(self.n_streams, self.budget / self.n_streams)
            else:  # myopic: split the budget proportional to imminent load (blind to priority)
                alloc = self.budget * load / (jnp.sum(load) + 1e-8)
            processed = jnp.minimum(load, self.throughput * alloc)
            remaining = load - processed
            dropped = jnp.maximum(remaining - self.q_max, 0.0)
            queue_next = jnp.minimum(remaining, self.q_max)
            cost = jnp.sum(weights * (self.c_latency * queue_next + self.c_drop * dropped))
            return queue_next, cost

        _, costs = jax.lax.scan(step, jnp.zeros(self.n_streams), jnp.arange(self.horizon))
        return float(jnp.sum(costs))

    def plan_mpc(self, steps: int = 400, lr: float = 0.1) -> jax.Array:
        """Projected-gradient MPC on the known dynamics: plan the allocation under the budget.

        Adam (scale-invariant) handles the large gradients; the plan is re-projected onto the
        per-tick budget simplex after each step, and the best feasible plan seen is returned.
        """
        alloc = jnp.full((self.horizon, self.n_streams), self.budget / self.n_streams)
        value_and_grad = jax.jit(jax.value_and_grad(self._cost_of_open_loop))
        cost_of = jax.jit(self._cost_of_open_loop)
        project = jax.vmap(lambda row: _project_simplex(row, self.budget))
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(alloc)
        best_alloc, best_cost = alloc, float(cost_of(alloc))
        for _ in range(steps):
            _, grad = value_and_grad(alloc)
            updates, opt_state = optimizer.update(grad, opt_state)
            alloc = project(optax.apply_updates(alloc, updates))
            cost = float(cost_of(alloc))
            if cost < best_cost:  # Adam is not monotone; keep the best feasible plan seen
                best_alloc, best_cost = alloc, cost
        return best_alloc

    def run(self, mpc_steps: int = 300) -> list[TrackResult]:
        """Score uniform / myopic / CHC-MPC against a well-planned oracle; report regret."""
        oracle_cost = float(self._cost_of_open_loop(self.plan_mpc(steps=mpc_steps * 3)))
        controllers = {
            "uniform": self._cost_of_policy("uniform"),
            "myopic": self._cost_of_policy("myopic"),
            "CHC-MPC": float(self._cost_of_open_loop(self.plan_mpc(steps=mpc_steps))),
        }
        return [
            TrackResult("D-adaptive-cv", name, "regret", cost - oracle_cost)
            for name, cost in controllers.items()
        ]


def track_adaptive_cv(mpc_steps: int = 300) -> list[TrackResult]:
    """Track-D task: sequential GPU-budget allocation across video streams (CHC-MPC vs myopic)."""
    return AdaptiveCVTask().run(mpc_steps=mpc_steps)

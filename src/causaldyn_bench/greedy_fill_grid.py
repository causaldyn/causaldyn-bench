"""Exhaustive grid check of P3's greedy-fill theorem at T = 3, with non-convex step functions.

The theorem: for F(v) = A sum_t v_t + sum_t phi_t(S_{t-1}) with every phi_t non-increasing on
[0, inf) and caps 0 <= v_t <= kappa_t, the greedy fill of a total M -- each round at its cap until M
is spent, g_t = max(0, min(kappa_t, M - C_{t-1})) -- costs no more than any feasible schedule of
the same total. Rocq proves it at every horizon (causal-hybrid-control's ``proofs/greedy_fill.v``);
this is the independent finite check the paper quotes, and it uses no convexity.

The grid. Everything is counted in units of h = 1/20, so prefix sums are exact integers:

- T = 3 rounds, A = 1;
- 300 draws from numpy's PCG64 at seed 20260926;
- per draw, each cap kappa_t is uniform on {0, h, ..., 1} (0..20 units);
- every schedule with v_t in {0, h, ..., kappa_t} is enumerated, so a draw has
  prod_t (kappa_t/h + 1) feasible schedules;
- per draw and round, phi_t is a random non-increasing step function on [0, 3]: a value of 1 at
  zero, then 1 to 8 drops of Exp(1) height at breakpoints uniform on (0, 3), evaluated on the grid.
  A step function with a drop inside the range is not convex, and the check counts the draws whose
  phi_2 or phi_3 is non-convex on the grid points (a negative second difference).

The check is cost(v) - cost(g(M)) over every feasible v, with g built from the formula above, not
from the prefix-sum identity the proof uses. Identical prefix sums give identical floats, so a tie
is exactly 0. As a negative control, the same enumeration is compared against the BACK-loaded fill
(last round first), which the theorem does not protect; it must lose somewhere, or the check has no
power.

    python -m causaldyn_bench.greedy_fill_grid
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

SEED = 20260926
DRAWS = 300
UNITS = 20  # grid points per unit of exploration: h = 1/20
CAP_UNITS = 20  # caps up to 1
HORIZON = 3
CURVATURE = 1.0


class Check(NamedTuple):
    """What the enumeration found: ``worst`` is the least cost(v) - cost(greedy fill), which the
    theorem says is never negative; ``control_worst`` the same against the back-loaded fill, which
    must be negative somewhere."""

    draws: int
    nonconvex_draws: int
    schedules: int
    ties: int
    worst: float
    control_worst: float


def step_function(rng: np.random.Generator, top: int) -> np.ndarray:
    """A random non-increasing step function, tabulated at the grid points 0..top (in units)."""
    drops = int(rng.integers(1, 9))
    breaks = rng.uniform(0.0, top, size=drops)
    heights = rng.exponential(1.0, size=drops)
    grid = np.arange(top + 1)
    return 1.0 - (heights[None, :] * (breaks[None, :] <= grid[:, None])).sum(axis=1)


def fill(caps: np.ndarray, total: np.ndarray, order: list[int]) -> list[np.ndarray]:
    """Each round of ``order`` at its cap until ``total`` is spent; returned in round order."""
    left = total.copy()
    rounds: list[np.ndarray] = [np.zeros_like(total)] * len(caps)
    for t in order:
        rounds[t] = np.minimum(caps[t], left)
        left = left - rounds[t]
    return rounds


def cost(phis: list[np.ndarray], rounds: list[np.ndarray]) -> np.ndarray:
    """F(v) of each schedule, ``rounds[t]`` holding round t's spend of every schedule in units."""
    prefix = np.zeros_like(rounds[0])
    total = np.zeros(rounds[0].shape)
    for phi, v in zip(phis, rounds, strict=True):
        total = total + phi[prefix]
        prefix = prefix + v
    return CURVATURE * prefix / UNITS + total


def check(draws: int = DRAWS, seed: int = SEED) -> Check:
    """Enumerate every feasible schedule of ``draws`` random instances against both fills."""
    rng = np.random.default_rng(seed)
    top = HORIZON * CAP_UNITS
    schedules = ties = nonconvex_draws = 0
    worst = control_worst = np.inf
    for _ in range(draws):
        caps = rng.integers(0, CAP_UNITS + 1, size=HORIZON)
        phis = [step_function(rng, top) for _ in range(HORIZON)]
        if any((np.diff(phi, 2) < 0).any() for phi in phis[1:]):
            nonconvex_draws += 1
        axes = np.meshgrid(*[np.arange(k + 1) for k in caps], indexing="ij")
        rounds = [a.ravel() for a in axes]
        total = np.sum(np.stack(rounds), axis=0)
        mine = cost(phis, rounds)
        greedy = cost(phis, fill(caps, total, list(range(HORIZON))))
        back = cost(phis, fill(caps, total, list(reversed(range(HORIZON)))))
        gap = mine - greedy
        schedules += gap.size
        ties += int((gap == 0.0).sum())
        worst = min(worst, float(gap.min()))
        control_worst = min(control_worst, float((mine - back).min()))
    return Check(draws, nonconvex_draws, schedules, ties, float(worst), float(control_worst))


def main() -> None:
    found = check()
    print(f"seed {SEED}, {found.draws} draws, T = {HORIZON}, grid h = 1/{UNITS}, caps in [0, 1]")
    print(f"non-convex phi_2 or phi_3 on the grid: {found.nonconvex_draws} of {found.draws} draws")
    print(f"feasible schedules: {found.schedules}")
    print(f"minimum of cost(v) - cost(greedy fill of the same total): {found.worst!r}")
    print(f"schedules tied with the greedy fill: {found.ties}")
    print(f"negative control, minimum of cost(v) - cost(back-loaded fill): {found.control_worst!r}")
    if found.worst < 0.0:
        raise SystemExit("FAIL: a feasible schedule beats the greedy fill of its own total")
    if found.control_worst >= 0.0:
        raise SystemExit("FAIL: the negative control never lost, so the check has no power")
    print("PASS")


if __name__ == "__main__":
    main()

"""The three synthetic dynamic SCMs of DCBO, and the regret of an intervention sequence on them.

Source: V. Aglietti, N. Dhir, J. Gonzalez, T. Damoulas, "Dynamic Causal Bayesian Optimization",
NeurIPS 2021, arXiv:2110.13891 -- Appendix E.1 (``stat``), E.5 (``ind``) and E.6, Eq. (20)
(``nonstat``); and the reference code, github.com/neildhir/DCBO at commit ``85a9bdf``:
``dcbo/utils/sem_utils/toy_sems.py`` (the equations), ``dcbo/examples/example_setups.py`` (the
domain and the exploration sets) and ``dcbo/experimental/experiments.py``
(``optimal_sequence_of_interventions``, the oracle).

**No code is copied.** The repository's ``LICENSE`` file is MIT, but its README's licence section
and ``setup.py`` both say GPL-3.0-or-later, so the licence is ambiguous and the port is
re-implemented from the equations alone. The reference runs only in an isolated environment
(``scripts/dcbo_reference.py``) and enters the bench as data.

Where the paper and the code disagree, **the code is ported**, because it is what produced the
reference's printed optima and what the DCBO arm runs on. The four divergences:

* **Domain.** Appendix E gives ``D(X) = [-5, 5]`` and ``D(Z) = [-5, 20]``; the code uses
  ``X in [-4, 1]`` and ``Z in [-3, 3]`` for all three. Ported: the code's.
* **Exploration sets.** E.1 gives ``{{X}, {Z}}`` for ``stat``; the code explores the power set,
  ``{X}, {Z}, {X, Z}``, for all three. On ``stat`` the pair ties ``{Z}``: ``Y`` has no ``X``
  parent.
* **``ind``'s target.** E.5 prints E.4's (MULTIV) equation for ``Y_t`` -- with a ``Z_T^2`` term
  and ``-Y_{t-1}`` -- which is unbounded below on the paper's own ``D(Z)``. The code's
  ``StationaryIndependentSEM`` is the two-bump surface below, and its oracle is the one the
  reference notebook ``ind_scm.ipynb`` prints.
* **``nonstat``'s last slice.** Eq. (20)'s ``h`` has ``Y_t = Z_t - Y_{t-1} - Z_{t-1}``; the code has
  ``|Z_t|``, and only ``|Z_t|`` reproduces the notebook's ``y*_2 = 12.387``.

**The objective is the noise-free skeleton.** Eq. (1) writes ``E[Y_t | do(X_s,t = x), I_{0:t-1}]``,
but the reference evaluates every explorative intervention and its oracle with a zero noise model
(``evaluate_target_function``, ``optimal_sequence_of_interventions``). The bench follows the code,
so the oracle here reproduces the reference's printed optima; noise enters only the observational
log, as ``N(0, 1)`` on every variable (E.1, E.5, E.6).

**Costs never enter the score.** Section 4 sets a unit intervention cost for every variable (the
code's ``cost_type=1``, ``cost_fix_equal``). It divides DCBO's causal expected improvement
(Section 3.3), so it steers which trials the reference runs; the regret here is in ``Y_t`` alone.

**Regret is conditional on the arm's own history.** DCGO, Eq. (1), minimises ``Y_t`` given the
interventions already implemented, so an arm that deviated at ``t - 1`` is scored at ``t`` against
the best response to *its* past, not to the oracle's. On ``nonstat`` the two differ in sign: ``Y_1``
carries ``-Y_0``, so a worse first step makes a better second one available.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Variable = Literal["X", "Z"]
ExplorationSet = tuple[Variable, ...]

HORIZON = 3  # T in E.1, E.5, E.6
N_OBSERVATIONS = 10  # N in E.1, E.5, E.6: observational time series in the log
GRID = 100  # size_intervention_grid of the reference oracle, per variable
EXPLORATION_SETS: tuple[ExplorationSet, ...] = (("X",), ("Z",), ("X", "Z"))
DOMAIN: dict[Variable, tuple[float, float]] = {"X": (-4.0, 1.0), "Z": (-3.0, 3.0)}

# Zoom rounds after the grid: each re-grids +-one spacing around the incumbent at a tenth of it.
# The reference's grid alone is not an oracle for a continuous arm -- its best `z` on `nonstat`'s
# last slice is 0.0303 away from 0, which a continuous optimiser beats by that much.
_REFINE_POINTS = 21
_REFINE_ROUNDS = 6


@dataclass(frozen=True)
class Slice:
    """``(X_t, Z_t, Y_t)``. Arrays broadcast over samples or over candidate interventions."""

    x: Array
    z: Array
    y: Array


_ZERO = np.zeros(())
ORIGIN = Slice(_ZERO, _ZERO, _ZERO)
"""The slice before ``t = 0``: every ``1_{t>0}`` lag term of Appendix E vanishes against it."""


@dataclass(frozen=True)
class DynamicSCM:
    """One of the three SCMs: a structural equation per variable, applied in the order X, Z, Y.

    ``edges`` is the DAG within one time slice. The lags each variable carries into the next slice
    live in the equations, and so do ``nonstat``'s two edges across slices.
    """

    name: str
    edges: tuple[tuple[str, str], ...]
    x: Callable[[int, Slice, Array], Array]
    z: Callable[[int, Slice, Array, Array], Array]
    y: Callable[[int, Slice, Array, Array, Array], Array]

    def step(
        self,
        t: int,
        prev: Slice,
        noise: Slice = ORIGIN,
        *,
        do_x: Array | None = None,
        do_z: Array | None = None,
    ) -> Slice:
        """Slice ``t`` from slice ``t - 1``. An intervened variable ignores its own equation."""
        x = self.x(t, prev, noise.x) if do_x is None else do_x
        z = self.z(t, prev, x, noise.z) if do_z is None else do_z
        return Slice(np.asarray(x), np.asarray(z), np.asarray(self.y(t, prev, x, z, noise.y)))


STAT = DynamicSCM(
    "stat",
    edges=(("X", "Z"), ("Z", "Y")),
    x=lambda t, prev, e: prev.x + e,
    z=lambda t, prev, x, e: np.exp(-x) + prev.z + e,
    y=lambda t, prev, x, z, e: np.cos(z) - np.exp(-z / 20.0) + prev.y + e,
)
"""E.1 / ``StationaryDependentSEM``: the DAG of Fig. 1(a), ``X_t -> Z_t -> Y_t``."""

IND = DynamicSCM(
    "ind",
    edges=(("X", "Y"), ("Z", "Y")),
    x=lambda t, prev, e: -prev.x + e,
    z=lambda t, prev, x, e: -prev.z + e,
    y=lambda t, prev, x, z, e: (
        -2.0 * np.exp(-((x - 1.0) ** 2) - (z - 1.0) ** 2)
        - np.exp(-((x + 1.0) ** 2) - z**2)
        + prev.y
        + e
    ),
)
"""E.5 / ``StationaryIndependentSEM``: Fig. 3(b), ``X_t -> Y_t <- Z_t``, no edge between the two."""


def _nonstat_y(t: int, prev: Slice, x: Array, z: Array, e: Array) -> Array:
    if t == 0:
        return np.sqrt(np.abs(36.0 - (z - 1.0) ** 2)) + 1.0 + e
    if t == 1:
        return z * np.cos(np.pi * z) - prev.y + e
    return np.abs(z) - prev.y - prev.z + e


NONSTAT = DynamicSCM(
    "nonstat",
    edges=(("X", "Z"), ("Z", "Y")),
    x=lambda t, prev, e: prev.x + e,
    z=lambda t, prev, x, e: (-x / prev.x if t == 1 else x) + prev.z + e,
    y=_nonstat_y,
)
"""E.6, Eq. (20) / ``NonStationaryDependentSEM(change_point=1)``: ``f`` at ``t = 0``, ``g`` at the
change point ``t = 1``, ``h`` after it. Fig. 3(c) adds ``X_0 -> Z_1`` and ``Z_1 -> Y_2``, which is
where ``g`` divides by ``X_{t-1}`` and ``h`` subtracts ``Z_{t-1}``. At ``t = 0`` the lags are
``ORIGIN``'s zeros, so ``f``'s ``Z_0 = X_0 + e`` is ``h``'s equation and needs no branch of its
own."""

SCMS: tuple[DynamicSCM, ...] = (STAT, IND, NONSTAT)


@dataclass(frozen=True)
class Decision:
    """``do(X_s,t = x_s,t)``: an exploration set and a level for each of its variables."""

    variables: ExplorationSet
    levels: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.variables not in EXPLORATION_SETS:
            raise ValueError(f"{self.variables} is not one of {EXPLORATION_SETS}")
        if len(self.levels) != len(self.variables):
            raise ValueError(f"{self.variables} needs {len(self.variables)} levels: {self.levels}")
        for name, level in zip(self.variables, self.levels, strict=True):
            lo, hi = DOMAIN[name]
            if not lo <= level <= hi:
                raise ValueError(f"do({name} = {level}) is outside D({name}) = [{lo}, {hi}]")

    def level(self, name: Variable) -> Array | None:
        return (
            np.asarray(self.levels[self.variables.index(name)]) if name in self.variables else None
        )


def apply(scm: DynamicSCM, t: int, prev: Slice, decision: Decision) -> Slice:
    """The noise-free slice ``t`` under ``decision``: the quantity the reference scores."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return scm.step(t, prev, do_x=decision.level("X"), do_z=decision.level("Z"))


@dataclass(frozen=True)
class BestResponse:
    decision: Decision
    value: float


def best_response(
    scm: DynamicSCM, t: int, prev: Slice, *, grid: int = GRID, refine: bool = True
) -> BestResponse:
    """The lowest noise-free ``Y_t`` over every exploration set, given slice ``t - 1``.

    ``refine=False`` is the reference oracle exactly: a ``grid``-point grid per variable, the grid's
    minimum, ties to the earlier set. ``refine=True`` then zooms around each set's incumbent.

    A candidate whose outcome is not finite is not an intervention the SCM defines, and is skipped:
    ``nonstat``'s ``g`` divides by ``X_0``, so a history that left ``X_0`` at its noise-free 0 has
    no ``do(X_1)`` at all.
    """
    best: BestResponse | None = None
    for variables in EXPLORATION_SETS:
        axes = [np.linspace(*DOMAIN[name], grid) for name in variables]
        spacing = np.array([axis[1] - axis[0] for axis in axes])
        levels, value = _grid_minimum(scm, t, prev, variables, axes)
        for _ in range(_REFINE_ROUNDS if refine else 0):
            axes = [
                np.linspace(
                    max(DOMAIN[name][0], centre - step),
                    min(DOMAIN[name][1], centre + step),
                    _REFINE_POINTS,
                )
                for name, centre, step in zip(variables, levels, spacing, strict=True)
            ]
            levels, value = _grid_minimum(scm, t, prev, variables, axes)
            spacing = spacing / 10.0
        if best is None or value < best.value:
            best = BestResponse(Decision(variables, tuple(float(v) for v in levels)), value)
    if best is None or not np.isfinite(best.value):
        raise ValueError(f"no exploration set of {scm.name} has a finite outcome at t = {t}")
    return best


def _grid_minimum(
    scm: DynamicSCM, t: int, prev: Slice, variables: ExplorationSet, axes: list[Array]
) -> tuple[Array, float]:
    points = np.stack([mesh.ravel() for mesh in np.meshgrid(*axes, indexing="ij")], axis=1)
    columns = dict(zip(variables, points.T, strict=True))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        outcome = scm.step(t, prev, do_x=columns.get("X"), do_z=columns.get("Z")).y
    outcome = np.where(np.isfinite(outcome), outcome, np.inf)
    index = int(np.argmin(outcome))
    return points[index], float(outcome[index])


@dataclass(frozen=True)
class Scored:
    """An intervention sequence played on the noise-free SCM, next to the best response per step."""

    decisions: tuple[Decision, ...]
    outcomes: tuple[float, ...]  # Y_t along the arm's own path
    best: tuple[float, ...]  # the best Y_t available given that same path up to t - 1

    @property
    def regret(self) -> tuple[float, ...]:
        return tuple(y - b for y, b in zip(self.outcomes, self.best, strict=True))

    @property
    def total_regret(self) -> float:
        return float(sum(self.regret))


def score(scm: DynamicSCM, decisions: Sequence[Decision]) -> Scored:
    """Conditional per-step regret: each decision against the best response to the arm's past."""
    prev, outcomes, best = ORIGIN, [], []
    for t, decision in enumerate(decisions):
        best.append(best_response(scm, t, prev).value)
        prev = apply(scm, t, prev, decision)
        outcomes.append(float(prev.y))
    return Scored(tuple(decisions), tuple(outcomes), tuple(best))


def oracle(
    scm: DynamicSCM, horizon: int = HORIZON, *, grid: int = GRID, refine: bool = True
) -> tuple[BestResponse, ...]:
    """The best intervention per time step, each given the ones before it (DCGO, Eq. (1))."""
    prev, path = ORIGIN, []
    for t in range(horizon):
        response = best_response(scm, t, prev, grid=grid, refine=refine)
        path.append(response)
        prev = apply(scm, t, prev, response.decision)
    return tuple(path)


def sample_log(
    scm: DynamicSCM, seed: int, n: int = N_OBSERVATIONS, horizon: int = HORIZON
) -> dict[str, Array]:
    """``n`` observational series of length ``horizon``, no intervention, noise ``N(0, 1)``.

    Returned as the reference's ``D_O``: ``{"X", "Z", "Y"} -> (n, horizon)``. Every draw of a seed
    is taken up front in one ``(horizon, 3, n)`` block, so the log does not depend on evaluation
    order.
    """
    noise = np.random.default_rng(seed).standard_normal((horizon, 3, n))
    prev, slices = ORIGIN, []
    for t in range(horizon):
        prev = scm.step(t, prev, Slice(*noise[t]))
        slices.append(prev)
    return {
        "X": np.stack([s.x for s in slices], axis=1),
        "Z": np.stack([s.z for s in slices], axis=1),
        "Y": np.stack([s.y for s in slices], axis=1),
    }

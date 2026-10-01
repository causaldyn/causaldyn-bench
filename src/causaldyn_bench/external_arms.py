"""Track M v2, budgets, the external arms: Meridian's and Robyn's plans for the quarter, scored on
the pre-registered run's worlds beside its arms.

**Pre-registered 2026-10-01, before any scored world ran.** The budgets run
(:mod:`causaldyn_bench.budget_regret`, ``results/track_m2_budgets.md``) scored CHC against
PyMC-Marketing and named two tools it left out. Here they plan the same quarter on the same worlds,
each fitted outside the bench in an environment of its own, from the same export, and planned by
its own optimiser:

* *Meridian* 2.1.0, Google's (``scripts/meridian_arm.py``, pinned by ``scripts/meridian_arm.txt``):
  a national model at ``ModelSpec``'s defaults, the geo tests as its ROI prior, NUTS as its Getting
  Started notebook runs it, the quarter planned by its ``BudgetOptimizer``;
* *Robyn* 3.12.1, Meta's, the package itself rather than a ridge written to resemble it
  (``scripts/robyn_arm.R``, on the scratch library ``scripts/robyn_arm.lock.txt`` records): ridge
  regression whose hyperparameters Nevergrad searches over the demo's ranges, set as the pilot
  chose, the geo tests as calibration rows, Robyn's own model selection, the quarter planned by
  ``robyn_allocator``.

Each script's header names its choices. The arms were piloted on seeds no score reads (the
``-pilot`` recipes, scored into ``results/track_m2_external_pilot_<ranges>.md``, one for each
setting of Robyn's ranges); this docstring, the code and the pilot's results are committed
together, and only then do the scored fits run.

The pilot ran Robyn three times. The first, on worlds 900 to 904, took the demo's 5 trials, and
every fit drew Robyn's warning that a model calibrated by experiments wants at least 10: the demo
runs 5 on its dummy data, uncalibrated. The arm now runs 10. Its ranges are a choice the demo
leaves open. Set by each channel's genre (``genre``), they hold the digital channels' decay to 0 to
0.3, which rules out four fifths of the range the worlds draw paid shopping's and Meta's retention
from; their union over the demo's genres (``union``) holds all of it, over a wider search. The
second and third pilots ran each at 10 trials on worlds 900 to 909, and the scored run fits Robyn
over the setting whose pilot regret was the lower, the better for Robyn: the genre's, 0.118 [0.028,
0.208] per euro against the union's 0.218 [0.095, 0.340]
(``results/track_m2_external_pilot_genre.md`` and ``_union.md``). Over the union Robyn planned
worse than the status quo, +0.102 [+0.034, +0.171], higher in 9 of the 10 worlds.

**Scored as PyMC-Marketing's plans were**: a record is scored only on the world its digest names;
a fit that raised plays the status quo, and is counted; a plan off the budget by more than a
millionth of it, or outside the box, is moved into the box (``Quarter.project``), and the move
recorded.

**The other arms are read, not refitted**: CHC's, PyMC-Marketing's and the status quo's regrets are
the pre-registered run's, read from ``results/track_m2_budgets.json`` for the same worlds. Each
world first reproduces that run's oracle worth and status-quo regret to ``TIE`` per euro, so both
sides of a comparison are scored against one oracle.

**The samples.** Meridian plans every scored world, the 200 drawn from seed 10 000 and the 100
reference from 20 000. Robyn plans the first 100 drawn worlds, from seed 10 000, for its cost:
20 000 ridge fits a world, each scored on three objectives. The number is the pilot's rule: the
fewest worlds, in tens, at which the pilot's spread would give CHC's difference from Robyn, and
PyMC-Marketing's, a 95 % interval whose half-width is half the pilot's difference; at least 30, and
at most 100, the run's budget. On the pilot CHC's difference asked for 50 worlds and
PyMC-Marketing's for 100. Robyn's comparisons read those worlds alone, paired with the other arms
on the same worlds.

**The analysis**, the budgets run's: each arm's mean regret per euro with its 95 % Student-t
interval, and its median; paired differences, the first arm's regret less the second's over the
worlds both planned, with the worlds where the first's is lower, tied (within ``TIE``) or higher and
a Clopper-Pearson interval on the lower share. Compared: CHC and PyMC-Marketing against each
external arm, each external arm against the status quo, and Meridian against Robyn. And how many
channels each arm's plans hold on a bound of the box, against the oracle's on the same worlds.

**No gate, a reading rule.** The budgets run's gate was its one test, and it has been read. These
arms place CHC among the tools: on the drawn worlds CHC's plans are said to beat an arm's only if
CHC's regret less the arm's has its 95 % interval under nought, to lose to it only if the interval
lies over nought, and to tie it otherwise; PyMC-Marketing's the same. The reference worlds are
reported, as the budgets run reported them, and not read.

**Predictions**, from the pilot, before the run:

* **CHC beats Meridian, and so does PyMC-Marketing.** On the pilot's 40 drawn worlds CHC's regret
  less Meridian's was -0.0565 [-0.0868, -0.0263] per euro, CHC's lower in 29; PyMC-Marketing's
  -0.0591 [-0.0875, -0.0308], lower in 33. The two arms' scored regrets are published (0.050 and
  0.048), so this predicts Meridian's: above both, as the pilot's 0.098 [0.075, 0.122] stood;
* Meridian beats the status quo: -0.0518 [-0.0779, -0.0258] on the pilot, lower in 30 of 40;
* Meridian's plans stay inside the box where the best plan goes to its edges: 4 of its 120
  channel-plans on the pilot sat on a bound, against the oracle's 66;
* every Meridian plan is moved into the box, by nothing that matters: its grid search spends the
  budget in whole units, and the pilot's furthest move was 0.00015 of a week's budget;
* on the reference worlds, reported: Meridian's regret above CHC's and PyMC-Marketing's, each
  interval under nought over 100 worlds (pilot, 20 worlds: Meridian 0.088; CHC less Meridian -0.040
  [-0.089, +0.010], PyMC-Marketing less Meridian -0.055 [-0.097, -0.013]);
* **CHC beats Robyn, and so does PyMC-Marketing.** On the pilot's 10 worlds, over the genre
  ranges, CHC's regret less Robyn's was -0.0859 [-0.1859, +0.0141], CHC's lower in 7 and higher in
  2; PyMC-Marketing's -0.0640 [-0.1748, +0.0468], lower in 6. Ten worlds read neither; at the
  pilot's spread 100 put both intervals under nought, PyMC-Marketing's the nearer. Robyn's regret
  above both, as the pilot's 0.118 [0.028, 0.208] stood;
* Robyn ties the status quo: +0.0026 [-0.0659, +0.0710] on the pilot, lower in 5 of 10;
* Meridian beats Robyn, reported: -0.0413 [-0.1142, +0.0315] on the pilot, lower in 6 of 10;
* Robyn's plans sit on a bound of the box about as often as the best plan's: 13 of its 30
  channel-plans on the pilot, against the oracle's 17;
* no Robyn fit raises and no Robyn plan is moved into the box: its allocator spends the budget
  inside the bounds it is given, and stopped on no error in the pilot;
* Robyn's own convergence check fails on most fits: on the pilot DECOMP.RSSD and MAPE converged in
  none of 10 and NRMSE in 5, each model taken from Robyn's clusters.

**By construction** (R19). The worlds, the tests and the oracle are the budgets run's, so what that
run's docstring says of them holds here. Each external arm plans as its tool does, and two tools
count less of a plan's carryover than the oracle: Meridian's optimiser counts nothing the quarter's
spend returns after the quarter, and Robyn's plans one steady week whose carryover is the window's
mean. Each is named in its script, and the oracle scores every plan with the carryover in and out.
Not in this run: Robyn on the reference worlds or past its sample; a geo-level Meridian model, which
one national series cannot feed; an LLM planner.

Precision: ``JAX_ENABLE_X64=1``. The worlds are dealt to ``--workers`` processes; each is scored
apart from the others, so the split moves no number.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from causaldyn_bench.budget_regret import (
    ENVIRONMENTS,
    OUT_OF_PLAN,
    PILOTS,
    PRIMARY,
    PYMC,
    TIE,
    Comparison,
    Environment,
    Mean,
    digest,
    experiments,
    lift_rows,
)
from causaldyn_bench.endogenous_mmm import MediaMixWorld, Series
from causaldyn_bench.mmm_decision import PLANNED, Plan, Quarter, oracle, regret

TRACK = "M2-external"
MERIDIAN, ROBYN = "Meridian", "Robyn"
EXTERNAL = (MERIDIAN, ROBYN)
READ = ("CHC", PYMC, "status quo")  # the pre-registered run's arms, read beside the external ones
LEVEL = 0.95
EDGE = 0.01  # a channel planned within this share of its box's width of a bound is on the bound
ROBYN_RANGES = {  # scripts/robyn_arm.R's two settings of the demo's ranges, as --ranges names them
    "genre": {
        "pla": {"alphas": [0.5, 3], "gammas": [0.3, 1], "thetas": [0, 0.3]},
        "meta": {"alphas": [0.5, 3], "gammas": [0.3, 1], "thetas": [0, 0.3]},
        "tv": {"alphas": [0.5, 1], "gammas": [0.3, 1], "thetas": [0.3, 0.8]},
    },
    "union": {"alphas": [0.5, 3], "gammas": [0.3, 1], "thetas": [0, 0.8]},
}
ROBYN_SETTING = "genre"  # the setting the pilot found the better for Robyn, the one scored
SAMPLES: dict[str, tuple[Environment, ...]] = {
    MERIDIAN: ENVIRONMENTS,
    ROBYN: (Environment("drawn", 10_000, 100),),
}
PILOT_SAMPLES: dict[str, tuple[Environment, ...]] = {
    MERIDIAN: PILOTS,
    ROBYN: (Environment("drawn", 900, 10),),
}


def design(setting: str) -> dict[str, dict[str, Any]]:
    """What each arm's record must say it ran, as its script and this docstring fix it, Robyn over
    the ``setting`` of its ranges."""
    return {
        MERIDIAN: {"float_dtype": "float64"},
        ROBYN: {"iterations": 2000, "trials": 10, "cores": 4, "ranges": ROBYN_RANGES[setting]},
    }


def on_bounds(quarter: Quarter, weekly: Series) -> int:
    """The channels ``weekly`` plans on a bound of the box, within ``EDGE`` of its width."""
    where = (weekly - quarter.lower) / (quarter.upper - quarter.lower)
    return int(np.sum((where <= EDGE) | (where >= 1.0 - EDGE)))


@dataclass(frozen=True)
class ArmScore:
    """One external arm's plan on one world, scored as the budgets run scored PyMC-Marketing's."""

    regret: float  # per euro of the budget; a fit that raised plays the status quo
    failure: str | None  # why the fit raised
    moved: float | None  # how far the plan was moved into the box, if it left it
    bounds: int  # channels planned on a bound of the box
    record: dict[str, Any]  # the arm's own record, its plan, traceback and versions aside


def score_arm(
    world: MediaMixWorld, quarter: Quarter, best: Plan, record: Mapping[str, Any]
) -> ArmScore:
    """``record``'s plan on ``world`` against the oracle's ``best``.

    Raises:
        RuntimeError: the plan beat the oracle, which only a broken oracle allows.
    """
    failure, moved = record["error"], None
    if failure:
        plan = quarter.status_quo
    else:
        plan = np.asarray(record["weekly"], dtype=float)
        if not quarter.feasible(plan, tolerance=OUT_OF_PLAN):
            inside = quarter.project(plan)
            moved = float(np.max(np.abs(inside - plan)))
            plan = inside
    loss = regret(world, quarter, plan, best) / quarter.budget
    if loss < -TIE:
        raise RuntimeError(f"a plan beat the oracle by {-loss:.3g} per euro")
    aside = ("seed", "digest", "weekly", "quarter_spend", "traceback", "python", "versions")
    kept = {key: value for key, value in record.items() if key not in aside}
    return ArmScore(loss, failure, moved, on_bounds(quarter, plan), kept)


@dataclass(frozen=True)
class WorldScore:
    """One world: the oracle, the pre-registered run's regrets read for it, and the external arms
    that planned it."""

    seed: int
    budget: float
    best: float  # the oracle's worth
    channels: int
    oracle_bounds: int  # channels the oracle plans on a bound of the box
    committed: dict[str, float]  # the pre-registered run's regret per euro for each of READ
    arms: dict[str, ArmScore]


def score_world(
    job: tuple[str, int, Mapping[str, Mapping[str, Any]], Mapping[str, Any]],
) -> WorldScore:
    """Environment ``job[0]``'s world from seed ``job[1]``: the external arms' records ``job[2]``
    scored, beside the pre-registered run's score of the world ``job[3]``.

    Raises:
        RuntimeError: a record was fitted to other data, or the oracle no longer reproduces the
            pre-registered run's, so the two sides of a comparison would be scored apart.
    """
    name, seed, records, committed = job
    environment = next(e for e in ENVIRONMENTS if e.name == name)
    world = environment.generator(seed).simulate(seed)
    fitted = digest(world, lift_rows(experiments(world, seed)))
    for arm, record in records.items():
        if record["digest"] != fitted:
            raise RuntimeError(f"{name} world {seed}: the {arm} record was fitted to other data")
    quarter = Quarter.after(world)
    best = oracle(world, quarter)
    status_quo = regret(world, quarter, quarter.status_quo, best) / quarter.budget
    if (
        abs(best.worth - committed["best"]) > TIE * quarter.budget
        or abs(status_quo - committed["regret"]["status quo"]) > TIE
    ):
        raise RuntimeError(f"{name} world {seed}: the oracle is not the pre-registered run's")
    arms = {arm: score_arm(world, quarter, best, record) for arm, record in records.items()}
    return WorldScore(
        seed,
        quarter.budget,
        best.worth,
        len(world.channels),
        on_bounds(quarter, best.weekly),
        {arm: float(committed["regret"][arm]) for arm in READ},
        arms,
    )


def verdict(comparison: Comparison) -> str:
    """The reading rule: the first arm beats the second, loses to it, or ties it, by where the
    paired difference's interval lies against nought."""
    low, high = comparison.difference.interval
    return "beats" if high < 0.0 else "loses to" if low > 0.0 else "ties"


@dataclass(frozen=True)
class EnvironmentRun:
    environment: Environment
    scores: tuple[WorldScore, ...]
    seconds: float

    @property
    def arms(self) -> tuple[str, ...]:
        """The external arms that planned some world here."""
        return tuple(a for a in EXTERNAL if any(a in s.arms for s in self.scores))

    def worlds(self, *arms: str) -> tuple[WorldScore, ...]:
        """The worlds every one of ``arms`` was scored on; the read arms were scored on all."""
        return tuple(s for s in self.scores if all(a in READ or a in s.arms for a in arms))

    def regrets(self, arm: str, worlds: Sequence[WorldScore]) -> Series:
        return np.array([s.committed[arm] if arm in READ else s.arms[arm].regret for s in worlds])

    @property
    def means(self) -> dict[str, Mean]:
        return {arm: Mean.of(self.regrets(arm, self.worlds(arm))) for arm in (*READ, *self.arms)}

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        """The comparisons reported, each the first arm's regret less the second's."""
        pairs = [(first, arm) for arm in self.arms for first in ("CHC", PYMC)]
        pairs += [(arm, "status quo") for arm in self.arms]
        if self.arms == EXTERNAL:
            pairs.append(EXTERNAL)
        return tuple(pairs)

    @property
    def comparisons(self) -> dict[tuple[str, str], Comparison]:
        compared = {}
        for first, second in self.pairs:
            both = self.worlds(first, second)
            compared[first, second] = Comparison.of(
                second, self.regrets(first, both), self.regrets(second, both)
            )
        return compared

    @property
    def claims(self) -> dict[tuple[str, str], str]:
        """CHC's and PyMC-Marketing's standing against each external arm, by the reading rule."""
        return {
            (first, second): verdict(c)
            for (first, second), c in self.comparisons.items()
            if first in ("CHC", PYMC)
        }


def run_environment(
    environment: Environment,
    records: Mapping[int, Mapping[str, Mapping[str, Any]]],
    committed: Mapping[int, Mapping[str, Any]],
    pool: Executor,
) -> EnvironmentRun:
    began = time.perf_counter()
    jobs = [(environment.name, seed, records[seed], committed[seed]) for seed in sorted(records)]
    scores = tuple(pool.map(score_world, jobs, chunksize=2))
    return EnvironmentRun(environment, scores, time.perf_counter() - began)


def _count(values: Sequence[bool]) -> int:
    return int(sum(values))


def _moved(run: EnvironmentRun, arm: str) -> str:
    """How many of ``arm``'s plans were moved into the box, and the furthest move, in a channel's
    weekly spend, as a share of the week's budget."""
    moves = [
        moved / (s.budget / PLANNED)
        for s in run.worlds(arm)
        if (moved := s.arms[arm].moved) is not None
    ]
    if not moves:
        return "no plan moved into the box"
    furthest = f"{max(moves):.2g} of a week's budget"
    return f"{len(moves)} plans moved into the box, the furthest by {furthest}"


def _meridian_line(run: EnvironmentRun) -> str:
    scores = [s.arms[MERIDIAN] for s in run.worlds(MERIDIAN)]
    fits = [a.record for a in scores if a.failure is None]
    return (
        f"{MERIDIAN}'s fits: {len(scores) - len(fits)} raised; of the rest, "
        f"{_count([f['divergences'] > 0 for f in fits])} had divergent transitions, "
        f"{_count([f['max_rhat'] > 1.01 for f in fits])} an R-hat over 1.01, "
        f"{_count([f['review']['overall'] != 'PASS' for f in fits])} did not pass "
        f"{MERIDIAN}'s own review; {_moved(run, MERIDIAN)}."
    )


def _robyn_line(run: EnvironmentRun, setting: str) -> str:
    scores = [s.arms[ROBYN] for s in run.worlds(ROBYN)]
    fits = [a.record for a in scores if a.failure is None]

    def converged(metric: str) -> int:
        return _count(
            [not any(f"{metric} NOT converged" in m for m in f["convergence"]) for f in fits]
        )

    return (
        f"{ROBYN}'s fits over the {setting} ranges: {len(scores) - len(fits)} raised; of the "
        f"rest, its own convergence check passed NRMSE in {converged('NRMSE')}, DECOMP.RSSD in "
        f"{converged('DECOMP.RSSD')} and MAPE in {converged('MAPE')}; the model came from the "
        "clusters in "
        f"{_count([f['selection'] == 'clusters' for f in fits])}; its allocator stopped on an "
        f"error in {_count([f['allocator']['status'] < 0 for f in fits])}; {_moved(run, ROBYN)}."
    )


def _bounds_line(run: EnvironmentRun) -> str:
    parts = []
    for arm in run.arms:
        worlds = run.worlds(arm)
        parts.append(
            f"on {arm}'s {len(worlds)} worlds, {arm}'s {sum(s.arms[arm].bounds for s in worlds)} "
            f"of {sum(s.channels for s in worlds)} against the oracle's "
            f"{sum(s.oracle_bounds for s in worlds)}"
        )
    return (
        f"Channels planned on a bound of the box (within {EDGE:.0%} of its width): "
        + "; ".join(parts)
        + "."
    )


def _markdown(
    runs: Sequence[EnvironmentRun], x64: bool, setting: str, *, pilot: bool = False
) -> str:
    lines = [
        f"# Track M v2, budgets, the external arms{', pilot' if pilot else ''} "
        f"({'float64' if x64 else 'float32'})",
        "",
        "Regret per euro of the quarter's budget, scored as `results/track_m2_budgets.md` scores "
        f"it; {', '.join(READ[:-1])} and the {READ[-1]} are that run's, read for the same worlds. "
        "Mean with its 95 % Student-t interval, the median, the fits that raised (their arm then "
        "played the status quo) and the plans moved into the box.",
    ]
    if pilot:
        lines += [
            "",
            "The pilot: the seeds the design was set on, apart from every scored one. No reading "
            "rule reads it.",
        ]
    for run in runs:
        env = run.environment
        lines += [
            "",
            f"## {env.name}: {len(run.scores)} worlds from seed {run.scores[0].seed}",
            "",
            "| arm | worlds | mean [95 %] | median | failed | moved |",
            "|---|---|---|---|---|---|",
        ]
        for arm, m in run.means.items():
            worlds = run.worlds(arm)
            if arm in READ:
                failed = moved = "-"
            else:
                failed = str(sum(s.arms[arm].failure is not None for s in worlds))
                moved = str(sum(s.arms[arm].moved is not None for s in worlds))
            lines.append(
                f"| {arm} | {len(worlds)} | {m.mean:.4f} [{m.interval[0]:.4f}, "
                f"{m.interval[1]:.4f}] | {m.median:.4f} | {failed} | {moved} |"
            )
        lines += [
            "",
            "Each comparison: the first arm's regret less the second's over the worlds both "
            "planned, and the worlds where the first's was lower, tied or higher (Clopper-Pearson "
            "95 % on the lower share).",
            "",
            "| first | second | worlds | difference [95 %] | lower | tied | higher "
            "| lower share [95 %] |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for (first, second), c in run.comparisons.items():
            d, (low, high) = c.difference, c.lower_share
            n = c.lower + c.tied + c.higher
            lines.append(
                f"| {first} | {second} | {n} | {d.mean:+.4f} [{d.interval[0]:+.4f}, "
                f"{d.interval[1]:+.4f}] | {c.lower} | {c.tied} | {c.higher} | "
                f"{c.lower / n:.2f} [{low:.2f}, {high:.2f}] |"
            )
        lines += ["", _bounds_line(run)]
        if MERIDIAN in run.arms:
            lines += ["", _meridian_line(run)]
        if ROBYN in run.arms:
            lines += ["", _robyn_line(run, setting)]
        if env == PRIMARY:
            read = "; ".join(f"{f} {v} {s}" for (f, s), v in run.claims.items())
            lines += [
                "",
                f"**Reading** (on the {env.name} worlds, by the paired difference's 95 % "
                f"interval): {read}.",
            ]
    return "\n".join(lines) + "\n"


def _jsonable(value: Any) -> Any:
    """Non-finite floats as ``null``, so a strict parser reads the artefact."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _record(
    runs: Sequence[EnvironmentRun],
    x64: bool,
    tools: Mapping[str, Any],
    setting: str,
    *,
    pilot: bool = False,
) -> dict[str, Any]:
    samples = PILOT_SAMPLES if pilot else SAMPLES
    return _jsonable(
        {
            "track": TRACK,
            "pilot": pilot,
            "x64": x64,
            "design": {
                "read": READ,
                "external": EXTERNAL,
                "samples": {arm: [asdict(e) for e in envs] for arm, envs in samples.items()},
                "fits": design(setting),
                "robyn_ranges": setting,
                "edge": EDGE,
                "level": LEVEL,
            },
            "arm_environments": tools,
            "wall_clock": "environments[*].seconds and each record's seconds; recorded, not "
            "quoted: not measured on a clean machine",
            "environments": [
                {
                    "environment": asdict(run.environment),
                    "seconds": run.seconds,
                    "means": {arm: asdict(m) for arm, m in run.means.items()},
                    "comparisons": [
                        {"first": first, "second": second, **asdict(c)}
                        for (first, second), c in run.comparisons.items()
                    ],
                    "claims": (
                        {f"{f} - {s}": v for (f, s), v in run.claims.items()}
                        if run.environment == PRIMARY
                        else None
                    ),
                    "worlds": [asdict(s) for s in run.scores],
                }
                for run in runs
            ],
        }
    )


def _records(
    work: Path, samples: Mapping[str, Sequence[Environment]], setting: str
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    """Every sampled world's records, by environment, seed and arm, Robyn's those over the
    ``setting`` of its ranges. A run missing a record, or holding one whose fit departs from the
    design, is not scored."""
    fits = design(setting)
    out: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for arm, environments in samples.items():
        folder = f"{arm.lower()}-{setting}" if arm == ROBYN else arm.lower()
        for environment in environments:
            directory = work / environment.name / folder
            paths = {seed: directory / f"world_{seed}.json" for seed in environment.seeds}
            missing = [seed for seed, path in paths.items() if not path.exists()]
            if missing:
                raise SystemExit(
                    f"{environment.name}: no {arm} record for seeds {missing[:10]} ..."
                )
            for seed, path in paths.items():
                record = json.loads(path.read_text())
                ran = {key: record.get(key) for key in fits[arm]}
                if not record["error"] and ran != fits[arm]:
                    raise SystemExit(f"{environment.name} world {seed}: {arm} ran {ran}")
                out.setdefault(environment.name, {}).setdefault(seed, {})[arm] = record
    return out


def _tools(records: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]]) -> dict[str, Any]:
    """The one environment each arm's records ran on; records from two are not scored."""
    seen: dict[str, set[str]] = {}
    for by_seed in records.values():
        for by_arm in by_seed.values():
            for arm, record in by_arm.items():
                tool = {"python": record.get("python"), "versions": record.get("versions")}
                seen.setdefault(arm, set()).add(json.dumps(tool, sort_keys=True))
    for arm, tools in seen.items():
        if len(tools) != 1:
            raise SystemExit(f"the {arm} records ran on {len(tools)} environments: {tools}")
    return {arm: json.loads(tools.pop()) for arm, tools in seen.items()}


def _name(setting: str, *, pilot: bool) -> str:
    """The results a run writes: one pilot for each setting of Robyn's ranges, one scored run."""
    return f"track_m2_external_pilot_{setting}" if pilot else "track_m2_external"


def _committed(path: Path, *, pilot: bool) -> dict[str, dict[int, dict[str, Any]]]:
    """The pre-registered run's score of every world, by environment and seed."""
    data = json.loads(path.read_text())
    if data["pilot"] != pilot:
        raise SystemExit(f"{path} is {'' if data['pilot'] else 'not '}a pilot")
    return {
        e["environment"]["name"]: {w["seed"]: w for w in e["worlds"]} for e in data["environments"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="the design's seeds, not the scored")
    parser.add_argument(
        "--work", type=Path, help="the arms' records: default outputs/<the budgets run's name>"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--robyn-ranges",
        choices=sorted(ROBYN_RANGES),
        default=ROBYN_SETTING,
        help=f"the setting of Robyn's ranges whose records are read; the pilot's alone may be "
        f"other than {ROBYN_SETTING}, the scored run's",
    )
    parser.add_argument(
        "--name", help="default track_m2_external, the pilot's track_m2_external_pilot_<ranges>"
    )
    args = parser.parse_args()
    setting = args.robyn_ranges
    if setting != ROBYN_SETTING and not args.pilot:
        parser.error(f"the scored run fits Robyn over the {ROBYN_SETTING} ranges alone")
    budgets = "track_m2_budgets_pilot" if args.pilot else "track_m2_budgets"
    name = args.name or _name(setting, pilot=args.pilot)
    work = args.work or Path("outputs") / budgets
    x64 = bool(jnp.zeros(()).dtype == jnp.float64)
    if not x64:
        raise SystemExit("JAX_ENABLE_X64=1 is required: the pre-registered run scored at float64")
    # the pool is the parallelism: the workers inherit one BLAS thread each, whatever the shell set
    os.environ["OMP_NUM_THREADS"] = "1"
    records = _records(work, PILOT_SAMPLES if args.pilot else SAMPLES, setting)
    tools = _tools(records)
    committed = _committed(Path("results") / f"{budgets}.json", pilot=args.pilot)
    environments = [e for e in (PILOTS if args.pilot else ENVIRONMENTS) if e.name in records]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        runs = []
        for environment in environments:
            runs.append(
                run_environment(
                    environment, records[environment.name], committed[environment.name], pool
                )
            )
            print(f"{environment.name}: {runs[-1].seconds:.0f} s", flush=True)
    text = _markdown(runs, x64, setting, pilot=args.pilot)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.md").write_text(text)
    (args.out / f"{name}.json").write_text(
        json.dumps(
            _record(runs, x64, tools, setting, pilot=args.pilot),
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    print(text)
    print(f"written to {args.out}/{name}.md and {args.out}/{name}.json")


if __name__ == "__main__":
    main()

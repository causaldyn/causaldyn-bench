"""Track M v2, budgets: does CHC's plan for next quarter's media budget beat PyMC-Marketing's?

**Pre-registered 2026-09-30, before any scored world ran.** The arms were written and piloted on
seeds no score reads (``PILOTS``: ``just track-m2-budgets-pilot``, into
``results/track_m2_budgets_pilot.md``). This docstring, the code and the pilot's results are
committed together, and only then does ``just track-m2-budgets`` run.

**The decision** (:mod:`causaldyn_bench.mmm_decision`). The quarter after the world's 156 weeks: a
weekly spend for each channel, held for 13 weeks, at the budget of last year's rate, each channel
within half and twice its own mean over that year. A plan's regret is what the best plan in that
box returns over it on the world's own channels, carryover in and out included, per euro of the
budget.

**The worlds** (:mod:`causaldyn_bench.endogenous_mmm`, Heusch's generator written from his paper):

* *drawn*, the primary: each channel's retention, saturation and effect drawn from the range his
  three channels span; 200 worlds from seed 10 000;
* *reference*: his parameters; 100 worlds from seed 20 000. Its best plan is a corner, so an arm
  that ranks paid shopping first scores nought whether its reading is right or not.

**What every arm is given**: the history (sales, spend, the promotion and price the measurement
layer read), the quarter (budget, box, status quo), and his four go-dark geo tests on every
channel, each channel's in a treated universe of its own, the noise 1 % of mean weekly sales in
each universe.

**The arms**:

* *status quo*, last year's mix; *equal split*, the budget split evenly and moved into the box;
* *observational*: his realistic specification fitted to the history by least squares
  (:mod:`causaldyn_bench.observational_mmm`), the tests unused, planned by
  :func:`chc.allocation.allocate`;
* *CHC*: each channel read from its own tests alone (:func:`chc.lift.fit_lift`), planned by
  :func:`chc.allocation.allocate` with carryover counted both ways;
* *myopic*: CHC's channels as a planner reads them who counts a week's spend in that week alone,
  planned without carryover;
* *PyMC-Marketing* 1.2.0: its default model fitted to the history, every test entered as a lift
  measurement (:func:`lift_rows`), NUTS at its defaults, the quarter planned by its own optimiser
  (``scripts/pymc_marketing_arm.py``, in the environment ``scripts/pymc_marketing_arm.txt`` pins).

An arm whose fit raises plays the status quo, and is counted. A plan that leaves the box or misses
the budget by more than a millionth of it is moved into the box (``Quarter.project``), and the move
recorded. A PyMC-Marketing plan is scored only on the world its digest names.

**The analysis.** Each arm's mean regret per euro with its 95 % Student-t interval, and its
median. CHC against every other arm: the paired difference in regret per euro over the same worlds,
CHC's less the arm's, with its 95 % t interval, and the worlds where CHC's is lower, tied (within
``TIE``) or higher, with a Clopper-Pearson interval on the lower share. The gate reads one
comparison; the others are reported, not tested.

**The gate**, the media-mix release's, and its kill: on the drawn worlds CHC's regret is below
PyMC-Marketing's, the paired difference's 95 % interval under nought. If it is not, media mix stays
an example in the library and not a release theme.

**The sample.** On the pilot's 40 drawn worlds the paired difference with PyMC-Marketing had a
standard deviation of 0.050 per euro, so 200 worlds put its interval's half-width near 0.007, a
sixth of either arm's mean regret there. The reference worlds, 100, are reported, not gated.

**Predictions**, from the pilot, before the run:

* **the gate is not met.** On the pilot CHC's regret less PyMC-Marketing's was +0.003 per euro
  [-0.013, +0.019], CHC's lower in 16 worlds of 40 and higher in 15: a tie;
* CHC's regret is below the status quo's, the equal split's and the observational fit's, each
  interval under nought (pilot: -0.108, -0.138 and -0.176), and below the myopic plan's (pilot:
  -0.051 [-0.094, -0.009]): counting the carryover pays;
* the observational fit's regret is above the status quo's (pilot: 0.218 against 0.150): read
  from the history alone, the channels steer the budget worse than last year's mix does;
* on the reference worlds the myopic plan scores nought in every world, the corner, and CHC's
  median is nought with a tail (pilot mean 0.049), so the myopic plan beats CHC there: a lucky
  reading's win, which one instance cannot tell from a right one;
* the tests leave most channels' bend unread (pilot: 108 of 120 drawn channels' scale intervals
  open above, as ``results/track_m2_lift.md`` found for his channels), so CHC's plan reads the
  curve's shape past the tested spend, where PyMC-Marketing also has the history, whose spend
  ranges wider.

PyMC-Marketing's plans repeat bit for bit under the recipe: the pinned environment, one BLAS
thread, the world's seed. Another thread count draws other chains, and the plan moves by the
posterior's Monte Carlo error.

**By construction** (R19). The world is a reading of Heusch's paper, and its geo test is two
universes, so his differencing equation holds exactly and the tests carry no noise between regions
beyond what is drawn: no placebo floor. Every channel is tested. With his own design, paid
shopping's tests alone, no CHC arm is built: CHC reads only what was tested, and a plan across the
untested channels needs the history, which its arm does not read. Not in this run: Meridian, a
Robyn-style ridge, an LLM planner, and a CHC arm robust to the bend the tests leave unread.

Precision: ``JAX_ENABLE_X64=1``. The worlds are dealt to ``--workers`` processes; each is scored
apart from the others, so the split moves no number.
"""

from __future__ import annotations

import argparse
import hashlib
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
from chc.allocation import allocate
from chc.lift import LiftFit, LiftTest, fit_lift
from chc.response import Channel, GeometricAdstock, Tanh
from scipy.stats import t as student

from causaldyn_bench.endogenous_mmm import EndogenousMediaMix, MediaMixWorld, Series
from causaldyn_bench.lift_calibration import (
    COOLDOWN,
    NOISE,
    PRE,
    STARTS,
    TEST,
    Setting,
    clopper_pearson,
    lift_tests,
    template,
)
from causaldyn_bench.mmm_decision import BOX, PLANNED, Quarter, drawn, oracle, regret
from causaldyn_bench.observational_mmm import Observed, fit_observational

TRACK = "M2-budgets"
PYMC = "PyMC-Marketing"
ARMS = ("status quo", "equal split", "observational", "myopic", "CHC", PYMC)
LEVEL = 0.95
TIE = 1e-9  # regrets per euro closer than this are a tie
OUT_OF_PLAN = 1e-6  # a plan further than this, over the budget, from the box and the budget moves


@dataclass(frozen=True)
class Environment:
    """Which worlds a row of the table draws: the generator and the seeds it runs from."""

    name: str
    first: int  # the first seed scored
    worlds: int

    def generator(self, seed: int) -> EndogenousMediaMix:
        return drawn(seed) if self.name == "drawn" else EndogenousMediaMix()

    @property
    def seeds(self) -> range:
        return range(self.first, self.first + self.worlds)


ENVIRONMENTS = (Environment("drawn", 10_000, 200), Environment("reference", 20_000, 100))
PRIMARY = ENVIRONMENTS[0]
PILOTS = (Environment("drawn", 900, 40), Environment("reference", 900, 20))  # the design's seeds


def experiments(world: MediaMixWorld, seed: int) -> dict[str, tuple[LiftTest, ...]]:
    """His four go-dark tests on every channel, each channel's in a treated universe of its own,
    with noise drawn apart for each."""
    return {
        name: lift_tests(world, Setting(name), 1000 * seed + k)
        for k, name in enumerate(world.channels)
    }


def planned(channels: Sequence[Channel], quarter: Quarter, *, carryover: bool = True) -> Series:
    """The quarter's plan on ``channels``, through :func:`chc.allocation.allocate`."""
    return allocate(
        channels,
        quarter.budget,
        PLANNED,
        lower=quarter.lower,
        upper=quarter.upper,
        history=quarter.history if carryover else None,
    ).spend


def myopic(channels: Sequence[Channel]) -> tuple[Channel, ...]:
    """Each channel as a planner sees it who reads a week's spend in that week alone: through the
    kernel's first weight, with nothing carried over."""
    return tuple(
        Channel(
            GeometricAdstock(0.0, length=1, normalized=False),
            Tanh(float(channel.curve.scale) / float(channel.kernel.weights()[0])),
            channel.coefficient,
        )
        for channel in channels
    )


def identified(tests: dict[str, tuple[LiftTest, ...]], length: int) -> dict[str, LiftFit]:
    """CHC's reading of each channel from its own geo tests alone, :func:`chc.lift.fit_lift`."""
    return {name: fit_lift(t, template(t, length)) for name, t in tests.items()}


def reading(fit: LiftFit) -> dict[str, float]:
    """What a score keeps of CHC's fit to one channel: the estimates, and the scale's upper end,
    infinite where the tests did not read the curve's bend."""
    kept = dict(zip(fit.parameters, map(float, fit.estimate), strict=True))
    kept["curve.scale.upper"] = fit.interval("curve.scale")[1]
    return kept


def plans(
    world: MediaMixWorld, tests: dict[str, tuple[LiftTest, ...]]
) -> tuple[dict[str, Series | str], dict[str, LiftFit]]:
    """Every in-process arm's plan, or the error that stopped its fit, and CHC's readings."""
    quarter = Quarter.after(world)
    out: dict[str, Series | str] = {
        "status quo": quarter.status_quo,
        "equal split": quarter.equal_split(),
    }
    try:
        observational = fit_observational(Observed.of(world), length=world.kernel_length)
    except RuntimeError as error:
        out["observational"] = str(error)
    else:
        out["observational"] = planned(observational, quarter)
    fits: dict[str, LiftFit] = {}
    try:
        fits = identified(tests, world.kernel_length)
    except RuntimeError as error:
        out["CHC"] = out["myopic"] = str(error)
    else:
        channels = [fit.channel for fit in fits.values()]
        out["CHC"] = planned(channels, quarter)
        out["myopic"] = planned(myopic(channels), quarter, carryover=False)
    return out, fits


@dataclass(frozen=True)
class LiftRows:
    """Geo tests as PyMC-Marketing's lift measurements take them: a weekly spend ``x``, its change
    ``delta_x`` and the weekly change in sales ``delta_y`` with its standard error ``sigma``; and
    each test's first dark week ``start``, numbered from 1 as the world's weeks are, which dates it
    for a tool that calibrates on a test's window."""

    channel: tuple[str, ...]
    start: tuple[int, ...]
    x: Series
    delta_x: Series
    delta_y: Series
    sigma: Series
    dropped: int  # tests whose sales did not fall, which the measurement cannot take


def lift_rows(tests: dict[str, tuple[LiftTest, ...]]) -> LiftRows:
    """Each go-dark test reduced as an analyst reads one for a steady-state lift likelihood.

    ``x`` is the control's mean weekly spend over the dark weeks and ``delta_x = -x``. ``delta_y``
    is the gap summed over the dark weeks and the cooldown after them, so the carryover the
    darkness cost is counted, over the dark weeks. ``sigma`` scales the weekly gap's noise, read
    from the pre-test weeks of every test of the channel, to that sum. PyMC-Marketing's likelihood
    needs a fall in sales for a cut in spend, so a test whose sales did not fall is dropped and
    counted.
    """
    columns: dict[str, list] = {
        "channel": [],
        "start": [],
        "x": [],
        "delta_x": [],
        "delta_y": [],
        "sigma": [],
    }
    dropped = 0
    after = slice(PRE, PRE + TEST + COOLDOWN)
    for name, channel_tests in tests.items():
        pre = [t.difference[:PRE] for t in channel_tests]
        noise = math.sqrt(float(np.mean([np.var(g, ddof=1) for g in pre])))
        for test in channel_tests:
            readout = test.control.spend[test.control.history :]
            level = float(np.mean(readout[PRE : PRE + TEST]))
            change = float(np.sum(test.difference[after])) / TEST
            if not change < 0.0:
                dropped += 1
                continue
            columns["channel"].append(name)
            # the readout opens PRE weeks before the first dark week
            columns["start"].append(test.control.history + PRE + 1)
            columns["x"].append(level)
            columns["delta_x"].append(-level)
            columns["delta_y"].append(change)
            columns["sigma"].append(noise * math.sqrt(TEST + COOLDOWN) / TEST)
    return LiftRows(
        channel=tuple(columns["channel"]),
        start=tuple(columns["start"]),
        x=np.array(columns["x"]),
        delta_x=np.array(columns["delta_x"]),
        delta_y=np.array(columns["delta_y"]),
        sigma=np.array(columns["sigma"]),
        dropped=dropped,
    )


def digest(world: MediaMixWorld, rows: LiftRows) -> str:
    """What a fit outside the bench reads of a world, hashed, so its plan is scored only on the
    world it was fitted to."""
    blob = hashlib.sha256()
    for array in (
        world.sales,
        world.spend,
        world.observed_promotion.astype(float),
        world.observed_price,
        rows.x,
        rows.delta_y,
        rows.sigma,
    ):
        blob.update(np.ascontiguousarray(array, dtype=float).tobytes())
    blob.update("|".join(world.channels + rows.channel).encode())
    return blob.hexdigest()[:16]


def export(world: MediaMixWorld, seed: int, rows: LiftRows, directory: Path) -> Path:
    """What an arm outside the bench reads of one world, for a process in its own environment:
    PyMC-Marketing's lift rows, and each row's first dark week and the dark weeks' count, which
    date a test for Robyn's calibration."""
    quarter = Quarter.after(world)
    path = directory / f"world_{seed}.npz"
    np.savez(
        path,
        seed=seed,
        digest=digest(world, rows),
        channels=np.array(world.channels),
        sales=world.sales,
        spend=world.spend,
        promotion=world.observed_promotion.astype(float),
        price=world.observed_price,
        kernel_length=world.kernel_length,
        planned=PLANNED,
        budget=quarter.budget,
        lower=quarter.lower,
        upper=quarter.upper,
        lift_channel=np.array(rows.channel),
        lift_x=rows.x,
        lift_delta_x=rows.delta_x,
        lift_delta_y=rows.delta_y,
        lift_sigma=rows.sigma,
        lift_dropped=rows.dropped,
        lift_start=np.array(rows.start, dtype=np.int64),
        lift_weeks=TEST,
    )
    return path


@dataclass(frozen=True)
class WorldScore:
    """One world's plans, scored: each arm's regret per euro of the quarter's budget."""

    seed: int
    budget: float
    best: float  # the oracle's worth
    regret: dict[str, float]  # an arm that failed is scored on the status quo it then plays
    failure: dict[str, str]  # the arms that failed, and why
    moved: dict[str, float]  # the arms whose plan left the box or missed the budget: how far
    pymc: dict[str, Any]  # the PyMC-Marketing run's own record, its plan and versions aside
    chc: dict[str, dict[str, float]]  # CHC's reading of each channel; empty if its fit failed

    @property
    def bend_unread(self) -> bool:
        """Whether the tests left some channel's curve unbent: its scale's interval open above."""
        return any(math.isinf(r["curve.scale.upper"]) for r in self.chc.values())


def score(
    world: MediaMixWorld,
    seed: int,
    arms: Mapping[str, Series | str],
    *,
    pymc: Mapping[str, Any],
    chc: Mapping[str, LiftFit],
) -> WorldScore:
    """Every arm's plan on ``world`` against the oracle, with the PyMC-Marketing record and CHC's
    readings kept beside the scores.

    Raises:
        RuntimeError: a plan beat the oracle, which only a broken oracle allows.
    """
    quarter = Quarter.after(world)
    best = oracle(world, quarter)
    losses: dict[str, float] = {}
    failure: dict[str, str] = {}
    moved: dict[str, float] = {}
    for arm in ARMS:
        plan = arms[arm]
        if isinstance(plan, str):
            failure[arm] = plan
            plan = quarter.status_quo
        elif not quarter.feasible(plan, tolerance=OUT_OF_PLAN):
            inside = quarter.project(plan)
            moved[arm] = float(np.max(np.abs(inside - plan)))
            plan = inside
        losses[arm] = regret(world, quarter, plan, best) / quarter.budget
        if losses[arm] < -TIE:
            raise RuntimeError(f"world {seed}: {arm} beat the oracle by {-losses[arm]:.3g}")
    readings = {name: reading(fit) for name, fit in chc.items()}
    return WorldScore(
        seed, quarter.budget, best.worth, losses, failure, moved, dict(pymc), readings
    )


def score_world(job: tuple[str, int, Mapping[str, Any]]) -> WorldScore:
    """One world of environment ``job[0]`` from seed ``job[1]``: its arms' plans, the
    PyMC-Marketing record ``job[2]`` among them, scored.

    Raises:
        RuntimeError: the record was fitted to other data.
    """
    name, seed, record = job
    environment = next(e for e in ENVIRONMENTS if e.name == name)
    world = environment.generator(seed).simulate(seed)
    tests = experiments(world, seed)
    if record["digest"] != digest(world, lift_rows(tests)):
        raise RuntimeError(f"{name} world {seed}: the {PYMC} record was fitted to other data")
    arms, fits = plans(world, tests)
    arms[PYMC] = record["error"] or np.asarray(record["weekly"], dtype=float)
    aside = ("weekly", "versions", "python", "digest", "traceback")
    pymc = {k: v for k, v in record.items() if k not in aside}
    return score(world, seed, arms, pymc=pymc, chc=fits)


@dataclass(frozen=True)
class Mean:
    """A sample's mean with its Student-t interval, and its median."""

    mean: float
    interval: tuple[float, float]
    median: float
    sd: float

    @classmethod
    def of(cls, values: Series, level: float = LEVEL) -> Mean:
        n = values.size
        centre, sd = float(np.mean(values)), float(np.std(values, ddof=1))
        half = float(student.ppf(0.5 + level / 2.0, n - 1)) * sd / math.sqrt(n)
        return cls(centre, (centre - half, centre + half), float(np.median(values)), sd)


@dataclass(frozen=True)
class Comparison:
    """CHC against one arm over the same worlds: CHC's regret minus the arm's."""

    arm: str
    difference: Mean
    lower: int  # worlds where CHC's regret is the lower
    tied: int
    higher: int
    lower_share: tuple[float, float]  # Clopper-Pearson, of all worlds

    @classmethod
    def of(cls, arm: str, chc: Series, other: Series) -> Comparison:
        difference = chc - other
        lower = int(np.sum(difference < -TIE))
        higher = int(np.sum(difference > TIE))
        return cls(
            arm,
            Mean.of(difference),
            lower,
            difference.size - lower - higher,
            higher,
            clopper_pearson(lower, difference.size),
        )


@dataclass(frozen=True)
class EnvironmentRun:
    environment: Environment
    scores: tuple[WorldScore, ...]
    seconds: float

    def regrets(self, arm: str) -> Series:
        return np.array([s.regret[arm] for s in self.scores])

    @property
    def means(self) -> dict[str, Mean]:
        return {arm: Mean.of(self.regrets(arm)) for arm in ARMS}

    @property
    def comparisons(self) -> tuple[Comparison, ...]:
        chc = self.regrets("CHC")
        return tuple(Comparison.of(a, chc, self.regrets(a)) for a in ARMS if a != "CHC")

    @property
    def gate(self) -> Comparison:
        return next(c for c in self.comparisons if c.arm == PYMC)

    @property
    def passes(self) -> bool:
        return self.gate.difference.interval[1] < 0.0


def run_environment(
    environment: Environment, records: Mapping[int, Mapping[str, Any]], pool: Executor
) -> EnvironmentRun:
    began = time.perf_counter()
    jobs = [(environment.name, seed, records[seed]) for seed in environment.seeds]
    scores = tuple(pool.map(score_world, jobs, chunksize=2))
    return EnvironmentRun(environment, scores, time.perf_counter() - began)


def _pymc_line(run: EnvironmentRun) -> str:
    fits = [s.pymc for s in run.scores if s.pymc.get("error") is None]
    return (
        f"{PYMC}'s fits: {len(run.scores) - len(fits)} raised; of the rest, "
        f"{sum(f['divergences'] > 0 for f in fits)} had divergent transitions, "
        f"{sum(f['max_rhat'] > 1.01 for f in fits)} an r-hat over 1.01, "
        f"{sum(not f['optimiser_success'] for f in fits)} an optimiser that reported failure; "
        f"{sum(f['lift_dropped'] for f in fits)} of "
        f"{sum(f['lift_rows'] + f['lift_dropped'] for f in fits)} geo tests dropped, their "
        f"sales not having fallen; {sum(PYMC in s.moved for s in run.scores)} plans moved into "
        "the box."
    )


def _chc_line(run: EnvironmentRun) -> str:
    readings = [r for s in run.scores for r in s.chc.values()]
    unread = sum(math.isinf(r["curve.scale.upper"]) for r in readings)
    return (
        f"CHC's readings: {unread} of {len(readings)} channels' scale intervals open above, the "
        "tests not having read the curve's bend; the plan reads the family's shape past the "
        "tested spend."
    )


def _markdown(runs: Sequence[EnvironmentRun], x64: bool, *, pilot: bool = False) -> str:
    lines = [
        f"# Track M v2, budgets{', pilot' if pilot else ''} ({'float64' if x64 else 'float32'})",
        "",
        "Regret per euro of the quarter's budget: what the best plan in the box returns over each "
        "arm's, on the world's own channels, carryover included. Mean with its 95 % Student-t "
        "interval, the median, and the fits that raised, whose arm then played the status quo.",
    ]
    if pilot:
        lines += [
            "",
            "The pilot: the seeds the design was set on, apart from every scored one. No gate "
            "reads it.",
        ]
    for run in runs:
        env = run.environment
        lines += [
            "",
            f"## {env.name}: {env.worlds} worlds from seed {env.first}",
            "",
            "| arm | mean [95 %] | median | failed |",
            "|---|---|---|---|",
        ]
        for arm, m in run.means.items():
            failed = sum(arm in s.failure for s in run.scores)
            lines.append(
                f"| {arm} | {m.mean:.4f} [{m.interval[0]:.4f}, {m.interval[1]:.4f}] | "
                f"{m.median:.4f} | {failed} |"
            )
        lines += [
            "",
            "CHC against each arm: CHC's regret minus the arm's over the same worlds, and the "
            "worlds where CHC's was lower, tied or higher (Clopper-Pearson 95 % on the lower "
            "share).",
            "",
            "| against | difference [95 %] | lower | tied | higher | lower share [95 %] |",
            "|---|---|---|---|---|---|",
        ]
        for c in run.comparisons:
            d, (low, high) = c.difference, c.lower_share
            lines.append(
                f"| {c.arm} | {d.mean:+.4f} [{d.interval[0]:+.4f}, {d.interval[1]:+.4f}] | "
                f"{c.lower} | {c.tied} | {c.higher} | "
                f"{c.lower / len(run.scores):.2f} [{low:.2f}, {high:.2f}] |"
            )
        lines += ["", _pymc_line(run), "", _chc_line(run)]
        if env == PRIMARY:
            verdict = "met" if run.passes else "NOT met"
            lines += [
                "",
                f"**Gate** (CHC's regret below {PYMC}'s on the {env.name} worlds, the paired "
                f"difference's 95 % interval under nought): {verdict}.",
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
    runs: Sequence[EnvironmentRun], x64: bool, pymc: Mapping[str, Any], *, pilot: bool = False
) -> dict[str, Any]:
    return _jsonable(
        {
            "track": TRACK,
            "pilot": pilot,
            "x64": x64,
            "design": {
                "planned": PLANNED,
                "box": BOX,
                "tests": {"starts": STARTS, "pre": PRE, "test": TEST, "cooldown": COOLDOWN},
                "noise_share": NOISE,
                "arms": ARMS,
                "level": LEVEL,
            },
            "pymc_environment": pymc,
            "wall_clock": "environments[*].seconds; recorded, not quoted: not measured on a "
            "clean machine",
            "environments": [
                {
                    "environment": asdict(run.environment),
                    "seconds": run.seconds,
                    "means": {arm: asdict(m) for arm, m in run.means.items()},
                    "comparisons": [asdict(c) for c in run.comparisons],
                    "passes": run.passes if run.environment == PRIMARY else None,
                    "worlds": [asdict(s) for s in run.scores],
                }
                for run in runs
            ],
        }
    )


def _export(work: Path, environments: Sequence[Environment]) -> None:
    for environment in environments:
        directory = work / environment.name
        directory.mkdir(parents=True, exist_ok=True)
        for seed in environment.seeds:
            world = environment.generator(seed).simulate(seed)
            export(world, seed, lift_rows(experiments(world, seed)), directory)
        print(f"{environment.name}: {environment.worlds} worlds in {directory}", flush=True)


def _records(work: Path, environment: Environment) -> dict[int, dict[str, Any]]:
    """Every world's PyMC-Marketing record; a run missing one is not scored."""
    directory = work / environment.name / "pymc"
    paths = {seed: directory / f"world_{seed}.json" for seed in environment.seeds}
    missing = [seed for seed, path in paths.items() if not path.exists()]
    if missing:
        raise SystemExit(f"{environment.name}: no {PYMC} record for seeds {missing[:10]} ...")
    return {seed: json.loads(path.read_text()) for seed, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "score"))
    parser.add_argument("--pilot", action="store_true", help="the design's seeds, not the scored")
    parser.add_argument("--work", type=Path, help="default outputs/<name>")
    parser.add_argument("--environments", nargs="+", default=[e.name for e in ENVIRONMENTS])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--name", help="default track_m2_budgets, with _pilot for the pilot")
    args = parser.parse_args()
    name = args.name or ("track_m2_budgets_pilot" if args.pilot else "track_m2_budgets")
    work = args.work or Path("outputs") / name
    x64 = bool(jnp.zeros(()).dtype == jnp.float64)
    if not x64:
        raise SystemExit("JAX_ENABLE_X64=1 is required: the arms were piloted at float64")
    # the pool is the parallelism: the workers inherit one BLAS thread each, whatever the shell set
    os.environ["OMP_NUM_THREADS"] = "1"
    environments = [
        e for e in (PILOTS if args.pilot else ENVIRONMENTS) if e.name in args.environments
    ]
    if args.stage == "export":
        _export(work, environments)
        return
    records = {e.name: _records(work, e) for e in environments}
    versions = {
        json.dumps({"python": r.get("python"), **r.get("versions", {})}, sort_keys=True)
        for by_seed in records.values()
        for r in by_seed.values()
    }
    if len(versions) != 1:
        raise SystemExit(f"the {PYMC} records ran on {len(versions)} environments: {versions}")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        runs = []
        for environment in environments:
            runs.append(run_environment(environment, records[environment.name], pool))
            print(f"{environment.name}: {runs[-1].seconds:.0f} s", flush=True)
    text = _markdown(runs, x64, pilot=args.pilot)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.md").write_text(text)
    record = _record(runs, x64, json.loads(versions.pop()), pilot=args.pilot)
    (args.out / f"{name}.json").write_text(
        json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(text)
    print(f"written to {args.out}/{name}.md and {args.out}/{name}.json")


if __name__ == "__main__":
    main()

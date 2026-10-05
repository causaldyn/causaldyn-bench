"""Track M v2, lift tests: does a structural fit's interval cover a channel's carryover?

A geo test's weekly gap between its groups identifies a channel's carryover, its curve and its
size over the spend it covered (Heusch 2026a). :func:`chc.lift.fit_lift` fits that gap by least
squares and gives each parameter a profile-likelihood interval. This track measures those intervals
on the world of :mod:`causaldyn_bench.endogenous_mmm`, Heusch's generator written from his paper,
where spend follows the business and every parameter is known.

The design is his 2026a paper's: go-dark tests of one channel from weeks 20, 55, 100 and 140, each
read over four weeks before, its four dark weeks and eight after, with the two universes' noise
one percent of mean weekly sales each. Every history is a fresh draw of the whole world, the
channel's parameters his reference ones. The fit starts from a retention of 0.5, a scale at the
mean spend the control group's readouts saw, and the coefficient's least-squares value there.

**The gate** (plans/27, MM5): over 500 histories, the retention's 95 % interval covers the truth
within two points of 95 %, for paid shopping (PLA), the channel his tests ran on. Reported beside
it: the scale's and the coefficient's coverage, how often each interval closes, the median width,
and the median estimate against the truth.

**The kill**, measured rather than argued (Lewis and Rao 2015): the same with the noise raised to
three and ten percent, and on the other two channels, whose tests move sales less; a retention
whose interval runs to both bounds is one the tests do not identify. A fit that raises rather than
converge is kept as a reading that covers nothing.

By construction, the world is a reading of Heusch's paper (R19): the track measures whether the
intervals are honest where the equation holds exactly, not whether real geo tests follow it.

Precision: run with ``JAX_ENABLE_X64=1`` (``just track-m2-lift``). The histories are dealt to
``--workers`` processes; each history's fit is independent of the others, so the split does not
move a number.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import time
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from chc.lift import GeoArm, LiftFit, LiftTest, fit_lift
from chc.response import Channel, GeometricAdstock, Tanh
from scipy.stats import beta

from causaldyn_bench.endogenous_mmm import EndogenousMediaMix, Market

TRACK = "M2-lift-calibration"
NOMINAL = 0.95
TOLERANCE = 0.02
STARTS = (20, 55, 100, 140)  # Heusch 2026a's four tests
PRE, TEST, COOLDOWN = 4, 4, 8
HISTORIES = 500
NOISE = 0.01
PARAMETERS = ("kernel.retention", "curve.scale", "coefficient")


@dataclass(frozen=True)
class Setting:
    """One row of the table: a channel, its tests and the noise they are read under."""

    channel: str
    starts: tuple[int, ...] = STARTS
    noise_share: float = NOISE

    @property
    def label(self) -> str:
        return f"{self.channel}, {len(self.starts)} tests, noise {self.noise_share:.0%}"


SETTINGS: tuple[Setting, ...] = (
    Setting("pla"),
    Setting("pla", starts=STARTS[:2]),
    Setting("pla", noise_share=0.03),
    Setting("pla", noise_share=0.10),
    Setting("meta"),
    Setting("tv"),
)


def lift_tests(world: Market, setting: Setting, seed: int) -> tuple[LiftTest, ...]:
    """The setting's go-dark tests on ``world``, each cut to its readout, spend history included."""
    experiment = world.geo_test(
        setting.channel, setting.starts, test=TEST, noise_share=setting.noise_share, seed=seed
    )
    tests = []
    for start in setting.starts:
        window = experiment.readout(start, pre=PRE, cooldown=COOLDOWN)
        tests.append(
            LiftTest(
                treated=GeoArm(
                    experiment.spend_treated[: window.stop], experiment.sales_treated[window]
                ),
                control=GeoArm(
                    experiment.spend_control[: window.stop], experiment.sales_control[window]
                ),
            )
        )
    return tuple(tests)


def template(tests: Sequence[LiftTest], length: int) -> Channel:
    """Where the fit starts: a retention of 0.5, a scale at the control readouts' mean spend."""
    spend = np.concatenate([test.control.spend[test.control.history :] for test in tests])
    return Channel(
        GeometricAdstock(0.5, length=length, normalized=True), Tanh(float(np.mean(spend))), 1.0
    )


def truth(generator: EndogenousMediaMix, channel: str) -> dict[str, float]:
    """The channel's parameters in the fit's names: his ``lambda`` is the scale ``2 / lambda``."""
    column = generator.channels.index(channel)
    return {
        "kernel.retention": generator.retention[column],
        "curve.scale": 2.0 / generator.saturation[column],
        "coefficient": generator.effect[column],
    }


@dataclass(frozen=True)
class Reading:
    """One history's fit: each parameter's estimate and interval, or why there is none."""

    seed: int
    estimate: dict[str, float]
    lower: dict[str, float]
    upper: dict[str, float]
    noise_sd: float
    tested_adstock: tuple[float, float]
    failure: str | None = None  # the fit's own RuntimeError; every number is then nan

    @classmethod
    def failed(cls, seed: int, failure: str) -> Reading:
        missing = dict.fromkeys(PARAMETERS, math.nan)
        return cls(seed, missing, missing, missing, math.nan, (math.nan, math.nan), failure)


def read(fit: LiftFit, seed: int) -> Reading:
    names = fit.parameters
    return Reading(
        seed=seed,
        estimate=dict(zip(names, map(float, fit.estimate), strict=True)),
        lower=dict(zip(names, map(float, fit.lower), strict=True)),
        upper=dict(zip(names, map(float, fit.upper), strict=True)),
        noise_sd=fit.noise_sd,
        tested_adstock=fit.tested_adstock,
    )


def fit_history(job: tuple[int, int]) -> Reading:
    """One history of setting ``job[0]``: a fresh world from seed ``job[1]``, its tests, the fit."""
    index, seed = job
    setting = SETTINGS[index]
    world = EndogenousMediaMix().simulate(seed)
    tests = lift_tests(world, setting, seed=10_000 + seed)
    try:
        fit = fit_lift(tests, template(tests, world.kernel_length))
    except RuntimeError as error:  # the fit says it did not converge: a reading, not a crash
        return Reading.failed(seed, str(error))
    return read(fit, seed)


def clopper_pearson(covered: int, n: int, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    low = float(beta.ppf(tail, covered, n - covered + 1)) if covered > 0 else 0.0
    high = float(beta.ppf(1.0 - tail, covered + 1, n - covered)) if covered < n else 1.0
    return low, high


@dataclass(frozen=True)
class Score:
    """One parameter's intervals over the histories."""

    coverage: float  # a history whose fit failed covers nothing
    coverage_interval: tuple[float, float]  # Clopper-Pearson, 95 %
    closed: float  # the share of intervals with both ends inside the parameter's range
    median_width: float  # over the closed intervals; nan when none closed
    median_estimate: float  # over the fits that converged


def score(readings: Sequence[Reading], parameter: str, true: float) -> Score:
    lower = np.array([r.lower[parameter] for r in readings])
    upper = np.array([r.upper[parameter] for r in readings])
    covered = int(np.sum((lower <= true) & (true <= upper)))
    floor, ceiling = (0.0, 1.0) if parameter == "kernel.retention" else (0.0, math.inf)
    if parameter == "coefficient":
        floor = -math.inf
    closed = (lower > floor) & (upper < ceiling)
    widths = (upper - lower)[closed]
    return Score(
        coverage=covered / len(readings),
        coverage_interval=clopper_pearson(covered, len(readings)),
        closed=float(np.mean(closed)),
        median_width=float(np.median(widths)) if widths.size else math.nan,
        median_estimate=float(np.nanmedian([r.estimate[parameter] for r in readings])),
    )


@dataclass(frozen=True)
class SettingRun:
    setting: Setting
    readings: tuple[Reading, ...]
    truth: dict[str, float]
    scores: dict[str, Score]
    seconds: float

    @property
    def gated(self) -> bool:
        return self.setting == SETTINGS[0]

    @property
    def passes(self) -> bool:
        return abs(self.scores["kernel.retention"].coverage - NOMINAL) <= TOLERANCE


def run_setting(index: int, histories: int, pool: Executor) -> SettingRun:
    setting = SETTINGS[index]
    began = time.perf_counter()
    jobs = [(index, seed) for seed in range(histories)]
    readings = tuple(pool.map(fit_history, jobs, chunksize=4))
    true = truth(EndogenousMediaMix(), setting.channel)
    scores = {name: score(readings, name, value) for name, value in true.items()}
    return SettingRun(setting, readings, true, scores, time.perf_counter() - began)


def _markdown(runs: Sequence[SettingRun], histories: int, x64: bool) -> str:
    lines = [
        f"# Track M v2, lift tests ({histories} histories a row, "
        f"{'float64' if x64 else 'float32'})",
        "",
        "Coverage of each parameter's 95 % profile-likelihood interval (Clopper-Pearson 95 % in "
        "brackets), the share of intervals closed on both sides, the median width of those, and "
        "the fits that raised rather than converge, which cover nothing.",
        "",
        "| setting | retention | closed | width | scale | closed | coefficient | closed | failed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        r, k, b = (run.scores[name] for name in PARAMETERS)
        failed = sum(reading.failure is not None for reading in run.readings)
        lines.append(
            f"| {run.setting.label} | {r.coverage:.3f} [{r.coverage_interval[0]:.3f}, "
            f"{r.coverage_interval[1]:.3f}] | {r.closed:.2f} | {r.median_width:.3f} | "
            f"{k.coverage:.3f} | {k.closed:.2f} | {b.coverage:.3f} | {b.closed:.2f} | {failed} |"
        )
    lines += [
        "",
        "Median estimate against the truth, and the median of the adstock range the readouts "
        "covered (the curve is read over that range only).",
        "",
        "| setting | retention | scale | coefficient | tested adstock |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        cells = [
            f"{run.scores[name].median_estimate:.4g} ({run.truth[name]:.4g})" for name in PARAMETERS
        ]
        low, high = np.nanmedian([r.tested_adstock for r in run.readings], axis=0)
        lines.append(f"| {run.setting.label} | {' | '.join(cells)} | {low:.1f} to {high:.1f} |")
    gate = next((run for run in runs if run.gated), None)
    if gate is not None:
        verdict = "met" if gate.passes else "NOT met"
        lines += [
            "",
            f"**Gate** ({gate.setting.label}, retention within {TOLERANCE:.2f} of {NOMINAL}): "
            f"{verdict}.",
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


def _columns(readings: Sequence[Reading]) -> dict[str, Any]:
    """Each history's reading, column by column, to six significant digits."""

    def rounded(value: float) -> float:
        return float(f"{value:.6g}")

    columns: dict[str, Any] = {"seed": [r.seed for r in readings]}
    for field in ("estimate", "lower", "upper"):
        columns[field] = {
            name: [rounded(getattr(r, field)[name]) for r in readings] for name in PARAMETERS
        }
    columns["noise_sd"] = [rounded(r.noise_sd) for r in readings]
    columns["tested_adstock"] = [[rounded(end) for end in r.tested_adstock] for r in readings]
    columns["failure"] = [r.failure for r in readings]
    return columns


def _record(runs: Sequence[SettingRun], histories: int, x64: bool) -> dict[str, Any]:
    return _jsonable(
        {
            "track": TRACK,
            "histories": histories,
            "x64": x64,
            "design": {"pre": PRE, "test": TEST, "cooldown": COOLDOWN, "nominal": NOMINAL},
            "null": "an interval end that runs off its side: -inf in `lower`, +inf in `upper`",
            "wall_clock": "settings[*].seconds; recorded, not quoted: not measured on a clean "
            "machine",
            "settings": [
                {
                    "setting": asdict(run.setting),
                    "truth": run.truth,
                    "scores": {name: asdict(value) for name, value in run.scores.items()},
                    "seconds": run.seconds,
                    "readings": _columns(run.readings),
                }
                for run in runs
            ],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, default=HISTORIES)
    parser.add_argument(
        "--settings",
        type=int,
        nargs="+",
        default=list(range(len(SETTINGS))),
        help="indices into SETTINGS",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--name", default="track_m2_lift")
    args = parser.parse_args()
    # the pool is the parallelism: the workers inherit one BLAS thread each, whatever the shell set
    os.environ["OMP_NUM_THREADS"] = "1"

    x64 = bool(jnp.zeros(()).dtype == jnp.float64)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        runs = []
        for index in args.settings:
            runs.append(run_setting(index, args.histories, pool))
            print(f"{runs[-1].setting.label}: {runs[-1].seconds:.0f} s", flush=True)
    text = _markdown(runs, args.histories, x64)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.name}.md").write_text(text)
    (args.out / f"{args.name}.json").write_text(
        json.dumps(_record(runs, args.histories, x64), separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    print(text)
    print(f"written to {args.out}/{args.name}.md and {args.out}/{args.name}.json")


if __name__ == "__main__":
    main()

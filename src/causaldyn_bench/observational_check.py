"""Track M v2, the observational check: do lift tests tell an observational channel from the
truth, and is the factor they read of it honest?

**Pre-registered 2026-10-01, before any scored history ran.** The check was piloted on seeds no
score reads (``PILOT``: ``just track-m2-check-pilot``, into ``results/track_m2_check_pilot.md``).
This docstring, the code and the pilot's results are committed together, and only then does
``just track-m2-check`` run. The gate was changed once, before that pilot and after one of 20
histories on the same seeds, which rejected the truth in none: it was written two-sided, and now
reads validity one-sided with a floor on power (below).

Where spend follows the business an observational media-mix fit reads the business's decisions as
the channel's effect (Heusch 2026a). :func:`chc.lift.check_observational` reads a channel so
fitted against the geo tests of it: an ``F`` test of whether the tests' gaps could be the
channel's, and the factor of the lift it predicts that the tests read, with a ``t`` interval.
:meth:`chc.lift.ObservationalCheck.least_gamma` turns the factor into the least
marginal-sensitivity ``Gamma`` the tests leave the channel. This track measures both where the
truth is known.

**The worlds** are Heusch's reference world
(:class:`causaldyn_bench.endogenous_mmm.EndogenousMediaMix`), a fresh history from each seed, with
his four go-dark tests on every channel, the noise a percent of mean weekly sales in each group
(:func:`causaldyn_bench.budget_regret.experiments`). Each channel's tests are fitted by
:func:`chc.lift.fit_lift` from Track M v2's template.

**Two channels are checked against each fit:**

* *the truth*, the world's own channel: the ``F`` should reject it at its nominal five percent,
  and the factor's interval cover 1;
* *the observational*, the channel his realistic specification reads off the history
  (:func:`causaldyn_bench.observational_mmm.fit_observational`): how often the ``F`` rejects it, and
  whether the factor's interval covers the factor the tests would read of it without noise,
  ``g.m / g.g``, ``g`` the gaps it predicts and ``m`` the world's own, from the generator.

``least_gamma`` is read at an effect-scale gap of 1, the channel's own predicted lift, as
``chc.sensitivity``'s example sets it. It is a floor under the ``Gamma`` the observational channel
needs: it lies at or under the one the noise-free factor needs whenever the factor's interval
covers that factor.

**The gate** (the library's MM5, second half): on paid shopping, the channel his tests ran on,
over 500 histories, the check is valid and has power. The ``F`` rejects the truth in at most 7
percent, five and two points of Monte Carlo error; the factor's interval covers in at least 93
percent, on both channels; and the ``F`` rejects the observational channel in at least 90
percent. Validity is read one-sided: an ``F`` that rejects the truth less often than its level
has lost power, not validity, which the observational channel's rejections measure. Meta and
television are reported.

**The sample**: 500 histories from seed 40 000. A rate near five percent then carries a Monte
Carlo error near 0.010, and one near 95 percent the same, so each threshold sits two errors from
its nominal value.

**Predictions**, from the pilot's 100 histories, before the run:

* **the gate is met.** On paid shopping the pilot rejected the truth in 0.030 [0.006, 0.085], its
  factor's interval covered 1 in 0.970 and the observational factor in 0.959, and the ``F``
  rejected the observational channel in all 100;
* the ``F`` is conservative: the truth's p-value fell under 0.05 in 3 histories and under 0.10 in
  7, its mean 0.55;
* the observational channel overstates paid shopping's lift about twofold: the tests read a median
  factor of 0.51 of it, 0.52 without noise, and a median least ``Gamma`` of 2.6 at an effect-scale
  gap of 1; Meta's about fivefold, a factor of 0.20 and a ``Gamma`` of 7;
* the observational fit reads no television in most histories (pilot: 90 of 100), so its factor
  is read over the few that remain.

By construction (R19): the world is a reading of Heusch's paper, and his geo test's two universes
differ in the channel alone, so the gap is the lift equation exactly, with the one noise scale the
check assumes. A lift fit or an observational fit that raises is counted, and its checks are
not read. An observational channel of coefficient nought predicts no gap: the ``F`` tests it, and
it has no factor (pilot: 2 of 100 on paid shopping).

Precision: ``JAX_ENABLE_X64=1``. The histories are dealt to ``--workers`` processes; each is read
apart from the others, so the split moves no number.
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
from chc.barrier import barrier_gamma_star
from chc.lift import LiftFit, LiftTest, ObservationalCheck, check_observational, fit_lift
from chc.response import Channel

from causaldyn_bench.budget_regret import experiments
from causaldyn_bench.endogenous_mmm import EndogenousMediaMix, MediaMixWorld, Series
from causaldyn_bench.lift_calibration import (
    COOLDOWN,
    PRE,
    STARTS,
    TEST,
    clopper_pearson,
    template,
    truth,
)
from causaldyn_bench.observational_mmm import Observed, channel, fit_observational

TRACK = "M2-observational-check"
LEVEL = 0.95  # fit_lift's, which the check reads at
SIZE = 1.0 - LEVEL
TOLERANCE = 0.02
POWER = 0.9  # the gate's least share of observational channels rejected
GAP = 1.0  # least_gamma's effect-scale gap: the channel's own predicted lift
HISTORIES = 500
FIRST = 40_000  # the scored seeds, apart from every other Track M v2 track's
PILOT = range(900, 1000)  # the seeds the design was set on
GATED = "pla"


@dataclass(frozen=True)
class Check:
    """One channel checked against one fit; every number nan when the check did not run."""

    p_value: float
    factor: float
    lower: float
    upper: float
    least_gamma: float
    failure: str | None = None  # the check's own ValueError

    @classmethod
    def of(cls, fit_check: ObservationalCheck) -> Check:
        lower, upper = fit_check.factor_interval
        return cls(fit_check.p_value, fit_check.factor, lower, upper, fit_check.least_gamma(GAP))

    @classmethod
    def missing(cls, failure: str) -> Check:
        return cls(math.nan, math.nan, math.nan, math.nan, math.nan, failure)

    def covers(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class Reading:
    """One channel of one history: the truth and the observational channel checked against its
    lift fit, and the factor the tests would read of the observational one without noise."""

    seed: int
    channel: str
    truth: Check
    observational: Check
    target: float
    failure: str | None = None  # the lift fit's or the observational fit's error


def gaps(read: Channel, tests: Sequence[LiftTest]) -> Series:
    """The gaps ``read`` predicts over the tests' readouts, stacked, at the market's scale."""
    return np.concatenate(
        [
            np.asarray(
                read(t.treated.spend / t.treated.share) - read(t.control.spend / t.control.share)
            )[t.treated.history :]
            for t in tests
        ]
    )


def true_gaps(world: MediaMixWorld, name: str) -> Series:
    """The gaps the world makes over the tests' readouts, before the noise: the generator's own."""
    experiment = world.geo_test(name, STARTS, test=TEST)
    return np.concatenate(
        [experiment.true_gap[experiment.readout(s, pre=PRE, cooldown=COOLDOWN)] for s in STARTS]
    )


def true_channel(generator: EndogenousMediaMix, name: str, length: int) -> Channel:
    """The world's own channel ``name``, in the lift fit's form."""
    parameters = truth(generator, name)
    return channel(
        parameters["kernel.retention"], parameters["curve.scale"], parameters["coefficient"], length
    )


def _check(fit: LiftFit, observed: Channel) -> Check:
    try:
        return Check.of(check_observational(fit, observed))
    except ValueError as error:  # the channel fits better than the fit: no least squares to test
        return Check.missing(str(error))


def check_history(seed: int) -> tuple[Reading, ...]:
    """One history from ``seed``: every channel's tests fitted, and both channels checked."""
    generator = EndogenousMediaMix()
    world = generator.simulate(seed)
    tests = experiments(world, seed)
    try:
        observational: tuple[Channel, ...] | str = fit_observational(
            Observed.of(world), length=world.kernel_length
        )
    except RuntimeError as error:
        observational = f"observational: {error}"
    readings = []
    for column, name in enumerate(world.channels):
        true = true_channel(generator, name, world.kernel_length)
        try:
            fit = fit_lift(tests[name], template(tests[name], world.kernel_length))
        except RuntimeError as error:
            failure = f"lift: {error}"
            missing = Check.missing(failure)
            readings.append(Reading(seed, name, missing, missing, math.nan, failure))
            continue
        if isinstance(observational, str):
            observed, target = Check.missing(observational), math.nan
        else:
            predicted = gaps(observational[column], tests[name])
            size = float(predicted @ predicted)
            target = float(predicted @ true_gaps(world, name)) / size if size > 0.0 else math.nan
            observed = _check(fit, observational[column])
        readings.append(
            Reading(
                seed,
                name,
                _check(fit, true),
                observed,
                target,
                observational if isinstance(observational, str) else None,
            )
        )
    return tuple(readings)


@dataclass(frozen=True)
class Rate:
    share: float
    interval: tuple[float, float]  # Clopper-Pearson, 95 %
    count: int  # the checks it is read over

    @classmethod
    def of(cls, hits: Sequence[bool]) -> Rate:
        count = len(hits)
        if not count:
            return cls(math.nan, (math.nan, math.nan), 0)
        return cls(sum(hits) / count, clopper_pearson(sum(hits), count), count)


@dataclass(frozen=True)
class Score:
    """One channel over the histories."""

    size: Rate  # the truth rejected
    covers_one: Rate  # the truth's factor interval covers 1
    rejected: Rate  # the observational channel rejected
    # over the observational channels that predict a gap: the factor's interval covers the
    # noise-free factor, and the least Gamma lies at or under the one that factor needs
    covers_target: Rate
    floor: Rate
    median_factor: float
    median_target: float
    median_gamma: float
    infinite_gamma: float  # the share of its least Gammas that are infinite
    no_factor: int  # observational channels that predict no gap, a coefficient of nought
    failed: int  # histories whose lift or observational fit raised
    unchecked: int  # checks that raised: a channel that fits better than the fit

    @property
    def passes(self) -> bool:
        return (
            self.size.share <= SIZE + TOLERANCE
            and self.covers_one.share >= LEVEL - TOLERANCE
            and self.covers_target.share >= LEVEL - TOLERANCE
            and self.rejected.share >= POWER
        )


def needed_gamma(target: float) -> float:
    """The least ``Gamma`` whose set, ``1 ± (Gamma-1)/(Gamma+1) GAP``, holds ``target``."""
    return barrier_gamma_star(abs(1.0 - target), GAP, 1.0)


def score(readings: Sequence[Reading]) -> Score:
    truth_read = [r.truth for r in readings if r.truth.failure is None and r.failure is None]
    observed = [
        (r.observational, r.target)
        for r in readings
        if r.failure is None and r.observational.failure is None
    ]
    pairs = [(check, target) for check, target in observed if not math.isnan(check.factor)]
    gammas = np.array([check.least_gamma for check, _ in pairs])
    unchecked = sum(
        c.failure is not None and r.failure is None
        for r in readings
        for c in (r.truth, r.observational)
    )
    return Score(
        size=Rate.of([c.p_value < SIZE for c in truth_read]),
        covers_one=Rate.of([c.covers(1.0) for c in truth_read]),
        rejected=Rate.of([c.p_value < SIZE for c, _ in observed]),
        covers_target=Rate.of([c.covers(t) for c, t in pairs]),
        floor=Rate.of([c.least_gamma <= needed_gamma(t) for c, t in pairs]),
        median_factor=float(np.median([c.factor for c, _ in pairs])) if pairs else math.nan,
        median_target=float(np.median([t for _, t in pairs])) if pairs else math.nan,
        median_gamma=float(np.median(gammas)) if pairs else math.nan,
        infinite_gamma=float(np.mean(np.isinf(gammas))) if pairs else math.nan,
        no_factor=len(observed) - len(pairs),
        failed=sum(r.failure is not None for r in readings),
        unchecked=unchecked,
    )


@dataclass(frozen=True)
class Run:
    seeds: tuple[int, ...]
    readings: tuple[Reading, ...]
    seconds: float

    def channel(self, name: str) -> tuple[Reading, ...]:
        return tuple(r for r in self.readings if r.channel == name)

    @property
    def scores(self) -> dict[str, Score]:
        names = dict.fromkeys(r.channel for r in self.readings)
        return {name: score(self.channel(name)) for name in names}


def run(seeds: Sequence[int], pool: Executor) -> Run:
    began = time.perf_counter()
    readings = tuple(r for rs in pool.map(check_history, seeds, chunksize=1) for r in rs)
    return Run(tuple(seeds), readings, time.perf_counter() - began)


def _rate(rate: Rate) -> str:
    return f"{rate.share:.3f} [{rate.interval[0]:.3f}, {rate.interval[1]:.3f}]"


def _markdown(result: Run, x64: bool, *, pilot: bool = False) -> str:
    lines = [
        f"# Track M v2, the observational check{', pilot' if pilot else ''} "
        f"({len(result.seeds)} histories, {'float64' if x64 else 'float32'})",
        "",
        "Each channel's lift fit checks two channels: the world's own, and the one his realistic "
        "observational specification reads off the history. Shares with their Clopper-Pearson "
        f"95 % intervals; a rejection is a p-value under {SIZE:.2f}.",
    ]
    if pilot:
        lines += ["", "The pilot: the seeds the design was set on. No gate reads it."]
    lines += [
        "",
        "| channel | truth rejected | covers 1 | observational rejected | covers its factor | "
        "floor holds | failed | unchecked |",
        "|---|---|---|---|---|---|---|---|",
    ]
    scores = result.scores
    for name, s in scores.items():
        lines.append(
            f"| {name} | {_rate(s.size)} | {_rate(s.covers_one)} | {_rate(s.rejected)} | "
            f"{_rate(s.covers_target)} | {_rate(s.floor)} | {s.failed} | {s.unchecked} |"
        )
    lines += [
        "",
        f"The observational channel: the median factor the tests read of it, the median factor "
        f"without noise, and its least `Gamma` at an effect-scale gap of {GAP:g}, median and the "
        "share infinite (no level reconciles it with the tests); over the channels that predict a "
        "gap, and the count that predict none.",
        "",
        "| channel | factor | noise-free | least Gamma | infinite | no factor |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in scores.items():
        lines.append(
            f"| {name} | {s.median_factor:.3f} | {s.median_target:.3f} | {s.median_gamma:.3g} | "
            f"{s.infinite_gamma:.2f} | {s.no_factor} |"
        )
    if not pilot and GATED in scores:
        verdict = "met" if scores[GATED].passes else "NOT met"
        lines += [
            "",
            f"**Gate** ({GATED}: the truth rejected in at most {SIZE + TOLERANCE:.2f}, both "
            f"factors' intervals covering in at least {LEVEL - TOLERANCE:.2f}, and the "
            f"observational channel rejected in at least {POWER:.2f}): {verdict}.",
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


def _record(result: Run, x64: bool, *, pilot: bool = False) -> dict[str, Any]:
    scores = result.scores
    return _jsonable(
        {
            "track": TRACK,
            "pilot": pilot,
            "x64": x64,
            "design": {
                "level": LEVEL,
                "gap": GAP,
                "tolerance": TOLERANCE,
                "power": POWER,
                "gated": GATED,
            },
            "gate": None if pilot or GATED not in scores else scores[GATED].passes,
            "wall_clock": "seconds; recorded, not quoted: not measured on a clean machine",
            "seconds": result.seconds,
            "seeds": list(result.seeds),
            "scores": {name: asdict(s) for name, s in scores.items()},
            "readings": [asdict(r) for r in result.readings],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="the design's seeds, not the scored")
    parser.add_argument("--histories", type=int, default=HISTORIES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--name", help="default track_m2_check, with _pilot for the pilot")
    args = parser.parse_args()
    name = args.name or ("track_m2_check_pilot" if args.pilot else "track_m2_check")
    if not bool(jnp.zeros(()).dtype == jnp.float64):
        raise SystemExit("JAX_ENABLE_X64=1 is required: the check was piloted at float64")
    # the pool is the parallelism: the workers inherit one BLAS thread each, whatever the shell set
    os.environ["OMP_NUM_THREADS"] = "1"
    seeds = list(PILOT) if args.pilot else list(range(FIRST, FIRST + args.histories))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        result = run(seeds, pool)
    text = _markdown(result, True, pilot=args.pilot)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.md").write_text(text)
    (args.out / f"{name}.json").write_text(
        json.dumps(_record(result, True, pilot=args.pilot), separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    print(text)
    print(f"written to {args.out}/{name}.md and {args.out}/{name}.json")


if __name__ == "__main__":
    main()

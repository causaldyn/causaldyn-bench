"""How much of a comparison's variance its design removes: common random numbers, which score the
arms on the same worlds, and control variates, the covariates the oracle computes for every world.

**Common random numbers** (:func:`pairing`). Every arm is scored on the same worlds, the same truth
and the same quarter (:func:`~causaldyn_bench.scorecard.scoring.paired`), so two arms' difference
reads each world's draws once. An unpaired comparison, each arm on worlds of its own, has the
variance ``var(a) + var(b)`` a world; the paired one has ``var(a - b) = var(a) + var(b) - 2 cov(a,
b)``. The share pairing removes, ``1 - var(a - b) / (var(a) + var(b))``, is the share of the worlds
an unpaired comparison needs for the same precision that the paired one saves. It is negative where
the two arms' regrets move against each other over the worlds: there pairing costs worlds.

**Control variates** (:func:`controlled`). A world's record keeps four covariates
(:data:`~causaldyn_bench.scorecard.scoring.COVARIATES`): the status quo's and the equal split's
regrets, the best plan's return per euro of the budget and the budget's shadow price. The oracle
computes them with no arm's fit, so their mean over an environment can be read on worlds no arm is
scored on (:func:`oracle_covariates`, :meth:`Centre.of`). An arm's regret, or a paired difference,
``y``, is adjusted world by world to ``z = y - beta' (x - m)``, ``x`` the world's covariates and
``m`` their mean. ``z``'s mean is ``y``'s, and its variance a world is ``var(y) - 2 beta' cov(x, y)
+ beta' var(x) beta``, least at the slope of the regression of ``y`` on ``x``, where it is
``var(y) (1 - R^2)``.

*The slope is fitted on pilot worlds and frozen* (:func:`fit`) before a scored world is read. A
slope fitted on the scored worlds themselves biases the interval: the adjusted values then depend
on one another through their slope, and their spread understates the estimate's. For ``n`` normal
worlds and ``p`` covariates the estimate's variance is ``(n - 2) / (n - p - 2)`` times
``sigma^2 (1 - R^2) / n`` (Lavenberg and Welch 1981), 5/3 at ``n = 12`` and ``p = 4``, where the
interval read off the fitted residuals, on ``n - p - 1`` degrees of freedom, holds the mean about
0.89 of the time, and read as :func:`controlled` reads a frozen slope's, 0.79, not 0.95. With the
slope frozen, the adjusted values are independent, and their Student-t interval is exact where they
are normal. Where ``y`` is not linear in ``x`` a fitted slope biases the mean too, by order ``1/n``;
a frozen one does not.

*The covariates' mean* is read on ``m`` worlds apart from the scored ones, so its own error adds:
``var(mean z) = var(z) / n + beta' var(x) beta / m``, and the interval reads Welch and
Satterthwaite's degrees of freedom. Control variates pay only where ``m`` is many times ``n``: at
the best slope and ``m = n`` they remove nothing. A scored environment's block holds 1 000 worlds
(:mod:`.seeds`), so the worlds of the block past a run's last look are its ``m``.

**The reduction** (:func:`adjustment`) is ``1 - var(adjusted) / var(raw)`` over a sample's worlds,
the share of the worlds a comparison needs that the covariates save where ``m`` is large. With a
frozen slope it is that ratio over the scored worlds. On the pilot that fits the slope, each world
is adjusted by the slope fitted on the other worlds, and its raw value read against the other
worlds' mean: the share is what a slope frozen on a pilot removes from a world it was not fitted
on, the cost of fitting it included. For normal worlds it is near ``1 - (1 - R^2) (n - 3) / (n - p
- 3)`` (Breiman and Freedman 1983, the error of a prediction by least squares, at ``n - 1``
worlds), not ``R^2``: at ``n = 40`` and ``p = 4`` the covariates must hold 11 % of the variance
before they save anything. ``scripts/control_variates.mac`` checks the closed forms the tests hold
these to.

Each share carries a 95 % percentile bootstrap interval, the worlds resampled :data:`DRAWS` times
(on the pilot, the worlds' left-out terms resampled with their fits held).

``python -m causaldyn_bench.scorecard.variance`` writes the table (:func:`section`) into Track M
v2's committed pilot reports, :data:`PILOTS`, after their last line: their arms' regrets as the
reports hold them, each world's covariates from the scorecard's oracle on the same world (family 0,
whose worlds are those runs' own), the pairs each report compares. It writes it too into each
scored run's report, :data:`SCORED`, out of sample: each pair's slope fitted on the run's pilot, the
same arms in the same environment on other worlds, frozen and applied to the scored worlds, beside
the share the pilot read left out on its own (:func:`scored_section`). A run that rewrites its
report drops the section, so the command runs after it.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.stats import t as student

from causaldyn_bench.endogenous_mmm import Series
from causaldyn_bench.scorecard.family import Family
from causaldyn_bench.scorecard.scoring import COVARIATES, WorldRecord, covariates, look
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

DRAWS = 10_000  # the bootstrap's resamples of the worlds
LEVEL = 0.95
ROOT = Path(__file__).resolve().parents[3]
# Track M v2's committed pilots whose reports compare arms world by world
PILOTS = (
    "track_m2_budgets_pilot",
    "track_m2_budgets_pilot_pymc130",
    "track_m2_external_pilot_genre",
    "track_m2_external_pilot_union",
)
# each committed scored run whose report compares arms world by world, and the pilot whose worlds
# freeze its slopes: the same arms, in the same environment
SCORED: Mapping[str, str] = MappingProxyType({"track_m2_budgets": "track_m2_budgets_pilot"})
HEADING = "## Variance reduction"


@dataclass(frozen=True)
class Reduction:
    """The share of a comparison's variance a design removes, with its bootstrap interval."""

    share: float
    interval: tuple[float, float]
    worlds: int


@dataclass(frozen=True)
class Slope:
    """A control variate's slope, fitted on ``worlds`` pilot worlds and frozen: ``beta`` on the
    covariates ``names``, in their order."""

    names: tuple[str, ...]
    beta: Series
    worlds: int


@dataclass(frozen=True)
class Centre:
    """The covariates' mean over ``worlds`` worlds no arm is scored on, and its covariance, the
    covariates' over the worlds divided by ``worlds``: 0 where the mean is known exactly."""

    mean: Series
    covariance: Series
    worlds: int

    @classmethod
    def of(cls, covariates: Series) -> Centre:
        """The mean of ``covariates``, ``(worlds, p)``, read on worlds apart from the scored ones.

        Raises:
            ValueError: on fewer than two worlds, or a covariate that is not a number.
        """
        x = _covariates(covariates)
        m = x.shape[0]
        if m < 2:
            raise ValueError(f"a centre is read on two worlds or more, not {m}")
        return cls(x.mean(axis=0), np.atleast_2d(np.cov(x, rowvar=False, ddof=1)) / m, m)

    @classmethod
    def known(cls, mean: Series) -> Centre:
        """A mean known exactly."""
        mean = np.asarray(mean, dtype=float)
        return cls(mean, np.zeros((mean.size, mean.size)), 0)


@dataclass(frozen=True)
class Estimate:
    """A mean adjusted by control variates, with its interval: ``se`` its standard error, ``df``
    the degrees of freedom its interval reads, ``worlds`` the scored worlds."""

    mean: float
    interval: tuple[float, float]
    se: float
    df: float
    worlds: int


def _covariates(covariates: Series) -> Series:
    x = np.asarray(covariates, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"covariates are (worlds, p), not {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("every covariate a control variate reads is a number")
    return x


def _resamples(n: int, draws: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def _interval(shares: Series) -> tuple[float, float]:
    low, high = np.quantile(shares, [(1.0 - LEVEL) / 2.0, (1.0 + LEVEL) / 2.0])
    return float(low), float(high)


def pairing(first: Series, second: Series, *, draws: int = DRAWS, seed: int = 0) -> Reduction:
    """The share of an unpaired comparison's variance that pairing ``first`` and ``second`` on the
    same worlds removes, ``1 - var(first - second) / (var(first) + var(second))``.

    Raises:
        ValueError: on fewer than three worlds, unequal lengths, or two arms that never move.
    """
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 3:
        raise ValueError(
            f"two arms' regrets on the same three worlds or more, not {a.shape}, {b.shape}"
        )
    spread = np.var(a, ddof=1) + np.var(b, ddof=1)
    if not spread > 0.0:
        raise ValueError("neither arm's regret moves over the worlds")
    share = 1.0 - float(np.var(a - b, ddof=1) / spread)
    pick = _resamples(a.size, draws, seed)
    ra, rb = a[pick], b[pick]
    shares = 1.0 - np.var(ra - rb, axis=1, ddof=1) / (
        np.var(ra, axis=1, ddof=1) + np.var(rb, axis=1, ddof=1)
    )
    return Reduction(share, _interval(shares), a.size)


def fit(y: Series, covariates: Series, names: Sequence[str] = COVARIATES) -> Slope:
    """The slope of the least-squares regression of ``y`` on ``covariates``, ``(worlds, p)``, with
    an intercept: fitted on pilot worlds, to be frozen.

    Raises:
        ValueError: on fewer than ``p + 3`` worlds, covariates not named one by one, or covariates
            collinear on the worlds.
    """
    y = np.asarray(y, dtype=float)
    x = _covariates(covariates)
    n, p = x.shape
    if len(names) != p or y.shape != (n,):
        raise ValueError(
            f"{n} worlds of {len(names)} named covariates, not {x.shape} and {y.shape}"
        )
    if n < p + 3:
        raise ValueError(f"a slope on {p} covariates is fitted on {p + 3} worlds or more, not {n}")
    centred = x - x.mean(axis=0)
    if np.linalg.matrix_rank(centred) < p:
        raise ValueError("the covariates are collinear on these worlds")
    beta, *_ = np.linalg.lstsq(centred, y - y.mean(), rcond=None)
    return Slope(tuple(names), beta, n)


def _left_out(y: Series, x: Series) -> tuple[Series, Series]:
    """Each world's raw and adjusted error, each read off the other worlds: the world less the
    others' mean, and less the others' least-squares fit, intercept and slope. The second is the
    fit's residual over one less the world's leverage, the identity of a least-squares fit left one
    world out."""
    n, p = x.shape
    if n < p + 4:
        raise ValueError(f"a slope on {p} covariates is read left out on {p + 4} worlds or more")
    design = np.column_stack([np.ones(n), x])
    q, r = np.linalg.qr(design)
    diagonal = np.abs(np.diag(r))
    if diagonal.min() <= 1e-10 * diagonal.max():
        raise ValueError("the covariates are collinear on these worlds")
    leverage = np.sum(q**2, axis=1)
    if leverage.max() >= 1.0 - 1e-9:
        raise ValueError("a world alone sets a covariate's slope: it cannot be left out")
    residual = y - q @ (q.T @ y)
    return (y - y.mean()) * n / (n - 1), residual / (1.0 - leverage)


def adjustment(
    y: Series,
    covariates: Series,
    slope: Slope | None = None,
    *,
    draws: int = DRAWS,
    seed: int = 0,
) -> Reduction:
    """The share of ``y``'s variance a world that control variates on ``covariates`` remove,
    ``1 - var(adjusted) / var(raw)``: by ``slope`` where it is given, the slope frozen on a pilot;
    otherwise each world by the slope fitted on the others, as a pilot reads it.

    Raises:
        ValueError: on too few worlds, covariates that are not numbers, covariates collinear on the
            worlds, a slope of other covariates, or a ``y`` that never moves.
    """
    y = np.asarray(y, dtype=float)
    x = _covariates(covariates)
    if y.shape != (x.shape[0],):
        raise ValueError(f"one value a world: {y.shape} against {x.shape[0]} worlds")
    if not np.var(y) > 0.0:
        raise ValueError("the value never moves over the worlds")
    pick = _resamples(y.size, draws, seed)
    if slope is None:
        raw, adjusted = _left_out(y, x)
        share = 1.0 - float(np.sum(adjusted**2) / np.sum(raw**2))
        shares = 1.0 - np.sum(adjusted[pick] ** 2, axis=1) / np.sum(raw[pick] ** 2, axis=1)
        return Reduction(share, _interval(shares), y.size)
    if slope.beta.shape != (x.shape[1],):
        raise ValueError(f"a slope on {slope.names} against {x.shape[1]} covariates")
    z = y - x @ slope.beta
    share = 1.0 - float(np.var(z, ddof=1) / np.var(y, ddof=1))
    shares = 1.0 - np.var(z[pick], axis=1, ddof=1) / np.var(y[pick], axis=1, ddof=1)
    return Reduction(share, _interval(shares), y.size)


def controlled(
    y: Series, covariates: Series, slope: Slope, centre: Centre, level: float = LEVEL
) -> Estimate:
    """The mean of ``y`` adjusted by control variates: ``mean(y - beta' (x - m))`` with ``beta``
    the frozen ``slope`` and ``m`` the ``centre``, its standard error
    ``sqrt(var(z) / n + beta' C beta)``, ``C`` the centre's covariance, and its Student-t interval
    at Welch and Satterthwaite's degrees of freedom, ``n - 1`` where the centre is known exactly.

    Raises:
        ValueError: on fewer than two worlds, or a slope or a centre of other covariates.
    """
    y = np.asarray(y, dtype=float)
    x = _covariates(covariates)
    n, p = x.shape
    if y.shape != (n,) or n < 2:
        raise ValueError(f"one value a world, on two worlds or more: {y.shape}, {n}")
    if slope.beta.shape != (p,) or centre.mean.shape != (p,):
        raise ValueError(f"the slope and the centre read {p} covariates")
    z = y - (x - centre.mean) @ slope.beta
    scored = float(np.var(z, ddof=1)) / n
    centred = float(slope.beta @ centre.covariance @ slope.beta)
    se = math.sqrt(scored + centred)
    if centred > 0.0:
        df = (scored + centred) ** 2 / (scored**2 / (n - 1) + centred**2 / (centre.worlds - 1))
    else:
        df = float(n - 1)
    half = float(student.ppf(0.5 + level / 2.0, df)) * se
    mean = float(np.mean(z))
    return Estimate(mean, (mean - half, mean + half), se, df, n)


def oracle_covariates(family: Family[Any], environment: str, seeds: Sequence[int]) -> Series:
    """``(worlds, 4)``: the covariates of the family's worlds ``seeds`` of ``environment``, in the
    order of :data:`COVARIATES`, from the oracle alone; the worlds a centre is read on."""
    rows = []
    for seed in seeds:
        held = covariates(family.truth(family.world(environment, seed)))
        rows.append([held[name] for name in COVARIATES])
    return np.array(rows, dtype=float)


@dataclass(frozen=True)
class Sample:
    """One environment's worlds as the reductions read them: each arm's regret a world, nan where
    the arm did not plan the world, and the world's covariates, ``(worlds, p)``, in one order."""

    environment: str
    regrets: Mapping[str, Series]
    covariates: Series

    def pair(self, first: str, second: str) -> tuple[Series, Series, Series]:
        """The two arms' regrets and the covariates on the worlds both planned."""
        a, b = self.regrets[first], self.regrets[second]
        both = np.isfinite(a) & np.isfinite(b)
        return a[both], b[both], self.covariates[both]


def samples(records: Sequence[WorldRecord], worlds: int) -> tuple[Sample, ...]:
    """Each environment of ``records``' look at ``worlds`` worlds as a :class:`Sample`, its
    covariates :data:`COVARIATES`.

    Raises:
        ValueError: as :func:`~causaldyn_bench.scorecard.scoring.look` does, or where a world's arms
            differ from the first world's.
    """
    out = []
    for environment in dict.fromkeys(r.environment for r in records):
        seen = look([r for r in records if r.environment == environment], worlds)
        arms = list(seen[0].arms)
        if any(list(r.arms) != arms for r in seen):
            raise ValueError(f"{environment}'s worlds do not score the same arms")
        regrets = {arm: np.array([r.arms[arm].regret for r in seen]) for arm in arms}
        x = np.array([[r.covariates[name] for name in COVARIATES] for r in seen])
        out.append(Sample(environment, regrets, x))
    return tuple(out)


@dataclass(frozen=True)
class Row:
    """One pair of arms on one environment: what pairing saves of an unpaired comparison, what the
    control variates save of the paired one, and both together."""

    environment: str
    first: str
    second: str
    worlds: int
    sd: float  # the paired difference's, a world
    pairing: Reduction
    adjustment: Reduction
    both: Reduction
    piloted: Reduction | None = None  # the pilot's own share, left out, where a slope was frozen


def row(
    sample: Sample,
    first: str,
    second: str,
    slope: Slope | None = None,
    *,
    draws: int = DRAWS,
    seed: int = 0,
) -> Row:
    """``first`` against ``second`` on ``sample``'s worlds both planned: the control variates read
    left out, as a pilot reads them, or by ``slope`` frozen on one."""
    a, b, x = sample.pair(first, second)
    paired = pairing(a, b, draws=draws, seed=seed)
    adjusted = adjustment(a - b, x, slope, draws=draws, seed=seed)
    # one resample serves both shares, so their product's interval keeps their dependence
    pick = _resamples(a.size, draws, seed)
    kept = _kept(a, b, x, slope, pick)
    both = Reduction(
        1.0 - (1.0 - paired.share) * (1.0 - adjusted.share), _interval(1.0 - kept), a.size
    )
    return Row(
        sample.environment,
        first,
        second,
        a.size,
        float(np.std(a - b, ddof=1)),
        paired,
        adjusted,
        both,
    )


def _kept(a: Series, b: Series, x: Series, slope: Slope | None, pick: np.ndarray) -> Series:
    """Each resample's share of an unpaired comparison's variance the paired and adjusted one
    keeps: one less pairing's share times one less the control variates'."""
    ra, rb = a[pick], b[pick]
    paired = np.var(ra - rb, axis=1, ddof=1) / (
        np.var(ra, axis=1, ddof=1) + np.var(rb, axis=1, ddof=1)
    )
    if slope is None:
        raw, adjusted = _left_out(a - b, x)
        return paired * np.sum(adjusted[pick] ** 2, axis=1) / np.sum(raw[pick] ** 2, axis=1)
    z = (a - b) - x @ slope.beta
    return paired * np.var(z[pick], axis=1, ddof=1) / np.var((a - b)[pick], axis=1, ddof=1)


def _share(reduction: Reduction) -> str:
    low, high = reduction.interval
    return f"{reduction.share:+.3f} [{low:+.3f}, {high:+.3f}]"


def section(rows: Sequence[Row], *, frozen_on: str | None = None) -> str:
    """The rows as a report's markdown section, one table an environment: read left out, as a
    pilot reads them, or by the slopes frozen on the pilot ``frozen_on``, beside its own shares."""
    how = (
        "each world by the slope fitted on the other worlds of its environment (left out), and "
        "its raw value against the others' mean, so the share is what a slope frozen on this "
        "pilot removes from a world it was not fitted on"
        if frozen_on is None
        else f"each world by the slope fitted on the worlds of `{frozen_on}`, the pilot of these "
        "arms in this environment, none of them a world here, and frozen; *on the pilot* is the "
        "share that pilot read on its own worlds, left out, the number this run checks out of "
        "sample"
    )
    extra = "" if frozen_on is None else " on the pilot [95 %] |"
    lines = [
        HEADING,
        "",
        "What each design removes of a comparison's variance a world, so of the worlds it needs "
        "for one precision: the share, with its 95 % percentile bootstrap interval over the worlds "
        f"({DRAWS} resamples). *Pairing*: the arms scored on the same worlds (common random "
        "numbers), against an unpaired comparison, `1 - var(a - b) / (var(a) + var(b))`; "
        "negative where the arms' regrets move against each other. *Control variates*: the "
        "paired difference adjusted by the world's covariates, the status quo's and the equal "
        "split's regrets, the best plan's return per euro and the budget's shadow price, "
        f"`1 - var(adjusted) / var(raw)`, {how}; it is the saving where the covariates' mean is "
        "read on many more worlds than the arms are scored on. *Both*: one less the product of "
        "what each keeps. `sd` is the paired difference's.",
    ]
    for environment in dict.fromkeys(r.environment for r in rows):
        lines += [
            "",
            f"### {environment}",
            "",
            "| first | second | worlds | sd | pairing [95 %] | control variates [95 %] "
            f"| both [95 %] |{extra}",
            "|---|---|---|---|---|---|---|" + ("---|" if extra else ""),
        ]
        lines += [
            f"| {r.first} | {r.second} | {r.worlds} | {r.sd:.4f} | {_share(r.pairing)} "
            f"| {_share(r.adjustment)} | {_share(r.both)} |"
            + ("" if r.piloted is None else f" {_share(r.piloted)} |")
            for r in rows
            if r.environment == environment
        ]
    return "\n".join(lines) + "\n"


def pilot(name: str, root: Path = ROOT) -> tuple[tuple[Sample, ...], tuple[tuple[str, str], ...]]:
    """A committed Track M v2 run, ``results/<name>.json``, a pilot or a scored run, as samples,
    each world's covariates from family 0's oracle, and the pairs its report compares, in its order.

    Raises:
        RuntimeError: a world's status quo is not the oracle's within the bench's tie, so the
            report and the oracle read different worlds.
    """
    data = json.loads((root / "results" / f"{name}.json").read_text())
    out, pairs = [], {}
    for environment in data["environments"]:
        label = environment["environment"]["name"]
        worlds = environment["worlds"]
        seeds = [w["seed"] for w in worlds]
        x = oracle_covariates(TRACK_M2, label, seeds)
        regrets: dict[str, list[float]] = {}
        for i, world in enumerate(worlds):
            held = dict(world.get("regret") or world["committed"])
            held |= {arm: score["regret"] for arm, score in world.get("arms", {}).items()}
            if abs(held["status quo"] - x[i, 0]) > 1e-9:
                raise RuntimeError(f"{label} world {world['seed']}: another world's status quo")
            for arm, value in held.items():
                regrets.setdefault(arm, [math.nan] * len(worlds))[i] = float(value)
        out.append(Sample(label, {a: np.array(v) for a, v in regrets.items()}, x))
        for c in environment["comparisons"]:
            pairs[c.get("first", "CHC"), c.get("second", c["arm"])] = None
    return tuple(out), tuple(pairs)


def pilot_section(name: str, root: Path = ROOT) -> str:
    """The section :func:`main` writes into the report of the pilot ``name``."""
    held, pairs = pilot(name, root)
    rows = [
        row(sample, first, second)
        for sample in held
        for first, second in pairs
        if first in sample.regrets and second in sample.regrets
    ]
    return section(rows)


def scored_section(name: str, root: Path = ROOT) -> str:
    """The section :func:`main` writes into the report of the scored run ``name``: each pair's
    control variates by the slope fitted on its pilot's worlds, :data:`SCORED`, and frozen, beside
    the share the pilot read left out.

    Raises:
        ValueError: the run and its pilot hold other environments or compare other pairs.
    """
    trial, pairs = pilot(SCORED[name], root)
    held, compared = pilot(name, root)
    if compared != pairs or [s.environment for s in held] != [s.environment for s in trial]:
        raise ValueError(f"{name} and its pilot {SCORED[name]} compare other pairs or environments")
    rows = []
    for before, sample in zip(trial, held, strict=True):
        for first, second in pairs:
            if not {first, second} <= set(sample.regrets) & set(before.regrets):
                continue
            a, b, x = before.pair(first, second)
            frozen = row(sample, first, second, fit(a - b, x))
            rows.append(replace(frozen, piloted=adjustment(a - b, x)))
    return section(rows, frozen_on=SCORED[name])


def written(report: str, part: str) -> str:
    """``report`` with ``part`` as its last section, in place of the one it held."""
    kept = report.split(f"\n{HEADING}\n", 1)[0].rstrip("\n")
    return f"{kept}\n\n{part}"


def main(argv: list[str] | None = None) -> None:
    """The section in each report ``argv`` names, every one of :data:`PILOTS` and
    :data:`SCORED` by default."""
    for name in (sys.argv[1:] if argv is None else argv) or (*PILOTS, *SCORED):
        path = ROOT / "results" / f"{name}.md"
        part = scored_section(name) if name in SCORED else pilot_section(name)
        path.write_text(written(path.read_text(), part))


if __name__ == "__main__":
    main()

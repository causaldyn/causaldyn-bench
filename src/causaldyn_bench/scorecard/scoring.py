"""Each arm's plan for a world scored against the world's truth, and the paired differences a look
at a run reads.

**A world's score** (:func:`score_world`) holds every arm's record of the world scored as Track M
v2's external arms were (:mod:`causaldyn_bench.external_arms`):

* a record is scored only on the export whose digest it echoes, by the rule of the export's version
  it names, version 1 where it names none, as the budgets run's records do;
* an arm whose fit raised plays the status quo, and is counted;
* a plan off the budget by more than a millionth of it, or outside the box, is moved into the box
  (``Quarter.project``), and the move recorded;
* the primary axis is regret per euro of the budget: what the best plan returns over the plan, on
  the effect path the best plan is made for;
* the channels the plan holds on a bound of the box are counted, beside the best plan's.

Every secondary axis a record carries is scored with it: where the quarter realised another effect
path, the plan's regret on that path, against the best plan in hindsight; where the record claims
the plan's ``gain`` over the status quo, in the outcome's units over the quarter and the kernel's
tail, the claim less the gain the plan realised, per euro of the budget, and, where it gives a
``gain_interval``, whether that holds the realised gain; where it gives a ``forecast``, draws of the
quarter's weekly sales at the status quo, their CRPS, averaged over the weeks and read in units of
the history's mean weekly sales; where it gives a ``response`` its tool's decomposition holds
(:mod:`.mapping`), each channel's ROI and mROI on the export's window (:mod:`.returns`) read off its
draws against the world's: the relative error of the draws' mean, averaged over the channels, and
the share of the channels whose truth the draws' central 90 % holds. A response that is missing,
unmapped, malformed or off its tool's decomposition is not read, and the score says why. A failed
fit claims, forecasts and maps nothing.

The status quo and the equal split are scored on every world, without a record. Beside the arms the
score keeps covariates of the world that the oracle computes anyway, for control variates: the
status quo's and the equal split's regrets, the best plan's return per euro of the budget, and the
budget's shadow price.

**Looks** (:func:`look`, :func:`paired`). Every arm is scored on the same worlds, the same truth
and the same quarter, so their differences are paired and their common noise cancels. A world's
``index`` is its place in its environment's seeded order, so a look at ``n`` reads the first ``n``
worlds of every environment a run holds: each look is a prefix of the run, the run a prefix of the
family's blocks, and a comparison may be read at any look, as a group-sequential reading does.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from causaldyn_bench.budget_regret import OUT_OF_PLAN, TIE, Comparison
from causaldyn_bench.endogenous_mmm import Series
from causaldyn_bench.external_arms import on_bounds
from causaldyn_bench.scorecard.family import Family, Truth, W, index, number
from causaldyn_bench.scorecard.mapping import TOLERANCE, residual
from causaldyn_bench.scorecard.observe import FIRST, Observation
from causaldyn_bench.scorecard.returns import Returns, drawn_returns
from causaldyn_bench.scorecard.truth import worth

BUILT_IN = ("status quo", "equal split")  # the arms every world scores without a record
AXES = ("regret", "realised", "uplift", "crps", "roi", "mroi")  # the axes a paired difference reads
LEVEL = 0.9  # the central share of a response's draws its returns' coverage reads
# what a score keeps of a record beside its numbers: neither the plan, nor the draws of its forecast
# and its response, nor where it ran
ASIDE = (
    "seed",
    "digest",
    "version",
    "weekly",
    "quarter_spend",
    "forecast",
    "response",
    "traceback",
    "python",
    "versions",
)


@dataclass(frozen=True)
class ArmScore:
    """One arm's plan for one world, scored."""

    regret: float  # per euro of the budget, on the expected path
    failure: str | None  # why the fit raised; the arm then played the status quo
    moved: float | None  # how far the plan was moved into the box, if it left it
    bounds: int  # channels planned on a bound of the box
    realised: float | None  # regret per euro on the realised path, where it is another
    uplift: float | None  # the claimed gain less the realised one, per euro
    covered: bool | None  # whether the claimed gain's interval holds the realised gain
    crps: float | None  # of the forecast, in units of the history's mean weekly sales
    record: dict[str, Any]  # the arm's own record, ``ASIDE`` set aside
    roi: float | None = None  # the draws' mean ROI's relative error, averaged over the channels
    mroi: float | None = None  # the same of the mROI
    roi_covered: float | None = None  # the channels whose ROI the draws' central 90 % holds
    mroi_covered: float | None = None  # the same of the mROI
    # each channel's ROI and mROI as the draws give them: their mean and their central 90 %'s ends
    returns: dict[str, dict[str, list[float]]] | None = None
    unread: str | None = None  # why a record's response was not read, where it was not


def crps(draws: Series, target: Series) -> float:
    """The CRPS of the draws' empirical distribution, ``(draws, weeks)``, at ``target``, averaged
    over the weeks: ``E|X - y| - E|X - X'| / 2``, the second term read off the sorted draws."""
    if draws.ndim != 2 or draws.shape[1] != target.size or draws.shape[0] < 1:
        raise ValueError(f"a forecast is draws of {target.size} weeks, not {draws.shape}")
    if not np.all(np.isfinite(draws)):
        raise ValueError("a forecast's draws are finite")
    n = draws.shape[0]
    spread = np.abs(draws - target).mean(axis=0)
    weights = (2.0 * np.arange(1, n + 1) - n - 1)[:, None]
    half = (weights * np.sort(draws, axis=0)).sum(axis=0) / n**2
    return float(np.mean(spread - half))


def _read(
    truth: Returns, observation: Observation, response: Mapping[str, Any] | None
) -> dict[str, Any]:
    """A record's ``response`` read against the world's returns, or why it is not read.

    A record is an arm's output, so a response that does not parse is a finding about the arm, kept
    with its reason, and not a fault of the score's.
    """
    if response is None:
        return {"unread": "no response"}
    if "unmapped" in response:
        return {"unread": f"unmapped: {response['unmapped']}"}
    try:
        gap = residual(response, observation)
    except ValueError as error:
        return {"unread": f"malformed: {error}"}
    if gap > TOLERANCE:
        return {"unread": f"{gap:.3g} off its tool's decomposition, beyond {TOLERANCE:g}"}
    drawn = drawn_returns(response, observation)
    read: dict[str, Any] = {"returns": {name: {} for name in observation.channels}}
    for key, true, draws in (
        ("roi", truth.roi, drawn.roi),
        ("mroi", truth.marginal, drawn.marginal),
    ):
        mean = draws.mean(axis=0)
        low, high = np.quantile(draws, [(1.0 - LEVEL) / 2.0, (1.0 + LEVEL) / 2.0], axis=0)
        read[key] = float(np.mean(np.abs(mean - true) / np.abs(true)))
        read[f"{key}_covered"] = float(np.mean((low <= true) & (true <= high)))
        for c, name in enumerate(observation.channels):
            read["returns"][name][key] = [float(mean[c]), float(low[c]), float(high[c])]
    return read


def _score(
    truth: Truth, record: Mapping[str, Any], observation: Observation | None = None
) -> ArmScore:
    """``record``'s plan scored against ``truth``, and its response read on ``observation``'s
    spend where that is given.

    Raises:
        RuntimeError: the plan beat the best plan, which only a broken oracle allows, or the export
            reads returns over another window than the truth's.
    """
    quarter = truth.quarter
    failure, moved = record["error"], None
    if failure:
        plan = quarter.status_quo
    else:
        plan = np.asarray(record["weekly"], dtype=float)
        if not quarter.feasible(plan, tolerance=OUT_OF_PLAN):
            inside = quarter.project(plan)
            moved = float(np.max(np.abs(inside - plan)))
            plan = inside
    loss = (truth.best.worth - worth(truth.cells, plan)) / quarter.budget
    realised = None
    if truth.realised is not None and truth.hindsight is not None:
        realised = (truth.hindsight.worth - worth(truth.realised, plan)) / quarter.budget
    for value in (loss, realised):
        if value is not None and value < -TIE:
            raise RuntimeError(f"a plan beat the best plan by {-value:.3g} per euro")
    uplift = covered = score = None
    if not failure and record.get("gain") is not None:
        on = truth.cells if truth.realised is None else truth.realised
        gain = worth(on, plan) - worth(on, quarter.status_quo)
        uplift = (float(record["gain"]) - gain) / quarter.budget
        if record.get("gain_interval") is not None:
            low, high = map(float, record["gain_interval"])
            if not low <= high:
                raise ValueError(f"a gain interval runs from low to high, not {low} to {high}")
            covered = low <= gain <= high
    if not failure and record.get("forecast") is not None:
        draws = np.asarray(record["forecast"], dtype=float)
        score = crps(draws, truth.target) / truth.scale
    read: dict[str, Any] = {}
    if not failure and observation is not None:
        if observation.roi_window != truth.returns.window:
            raise RuntimeError(
                f"the export reads returns over weeks {observation.roi_window}, the truth over "
                f"{truth.returns.window}"
            )
        read = _read(truth.returns, observation, record.get("response"))
    kept = {key: value for key, value in record.items() if key not in ASIDE}
    return ArmScore(
        loss,
        failure,
        moved,
        on_bounds(quarter, plan),
        realised,
        uplift,
        covered,
        score,
        kept,
        **read,
    )


def score_arm(truth: Truth, observation: Observation, record: Mapping[str, Any]) -> ArmScore:
    """``record``'s plan scored against ``truth``, once its digest is ``observation``'s.

    Raises:
        RuntimeError: the record was fitted to other data, or its plan beat the best plan.
    """
    version = int(record.get("version", FIRST))
    if record["digest"] != observation.digest(version):
        raise RuntimeError("the record was fitted to other data")
    return _score(truth, record, observation)


@dataclass(frozen=True)
class WorldRecord:
    """One world at one rung: every arm's score, paired, and the world's covariates."""

    family: int
    environment: str
    seed: int
    index: int  # the world's place in its environment's seeded order, from 0
    pilot: bool
    k: int
    budget: float
    best: float  # the best plan's worth on the expected path
    channels: int  # the cells a plan spends on
    oracle_bounds: int  # cells the best plan holds on a bound of the box
    covariates: dict[str, float]
    arms: dict[str, ArmScore]
    returns: dict[str, dict[str, float]] = field(default_factory=dict)  # each channel's ROI, mROI

    def as_json(self) -> dict[str, Any]:
        """The record as strict JSON holds it: a covariate that is not a number as null."""
        out = asdict(self)
        out["covariates"] = {
            name: value if math.isfinite(value) else None for name, value in self.covariates.items()
        }
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> WorldRecord:
        fields = dict(data)
        fields["covariates"] = {
            name: math.nan if value is None else float(value)
            for name, value in data["covariates"].items()
        }
        fields["arms"] = {name: ArmScore(**score) for name, score in data["arms"].items()}
        return cls(**fields)


def score_world(
    family: Family[W],
    environment: str,
    seed: int,
    k: int,
    records: Mapping[str, Mapping[str, Any]],
    *,
    pilot: bool = False,
    archived: Observation | None = None,
) -> WorldRecord:
    """The family's world ``seed`` of ``environment`` at rung ``k``: each arm's record in
    ``records`` scored, beside the status quo and the equal split. Each record must echo the digest
    of ``archived``, the export the arms ran on, or of the world's observation where none is given.

    Raises:
        ValueError: the family draws no such world, or a record takes a built-in arm's name.
        RuntimeError: the export is another world's, a record was fitted to other data, or a plan
            beat the best plan.
    """
    position = index(family, environment, seed, pilot=pilot)
    clash = sorted(set(records) & set(BUILT_IN))
    if clash:
        raise ValueError(f"{clash} are scored on every world, without a record")
    world = family.world(environment, seed)
    observation = family.observe(world, k) if archived is None else archived
    named = (observation.family, observation.environment, observation.seed, observation.k)
    if named != (number(family), environment, seed, k):
        raise RuntimeError(
            f"the export is family {named[0]}'s {named[1]} world {named[2]} at {named[3]}"
        )
    truth = family.truth(world)
    quarter = truth.quarter
    arms = {
        name: _score(truth, {"error": None, "weekly": plan})
        for name, plan in zip(BUILT_IN, (quarter.status_quo, quarter.equal_split()), strict=True)
    }
    arms |= {name: score_arm(truth, observation, record) for name, record in records.items()}
    covariates = {
        "status quo": arms["status quo"].regret,
        "equal split": arms["equal split"].regret,
        "return": truth.best.worth / quarter.budget,
        "price": truth.best.price,
    }
    return WorldRecord(
        family=number(family),
        environment=environment,
        seed=seed,
        index=position,
        pilot=pilot,
        k=k,
        budget=quarter.budget,
        best=truth.best.worth,
        channels=len(truth.cells),
        oracle_bounds=on_bounds(quarter, truth.best.weekly),
        covariates=covariates,
        arms=arms,
        returns={
            name: {"roi": float(truth.returns.roi[c]), "mroi": float(truth.returns.marginal[c])}
            for c, name in enumerate(observation.channels)
        },
    )


def look(records: Sequence[WorldRecord], worlds: int) -> tuple[WorldRecord, ...]:
    """The first ``worlds`` worlds of the seeded order of each environment ``records`` hold.

    Raises:
        ValueError: the records mix families, rungs or pilots with scored worlds, hold a world
            twice, or miss one of the look's worlds.
    """
    if worlds < 1:
        raise ValueError(f"a look reads at least one world, not {worlds}")
    kinds = {(r.family, r.k, r.pilot) for r in records}
    if len(kinds) != 1:
        raise ValueError(f"a look reads one family at one rung, pilots or scored: {sorted(kinds)}")
    held: dict[str, dict[int, WorldRecord]] = {}
    for record in records:
        places = held.setdefault(record.environment, {})
        if record.index in places:
            raise ValueError(f"{record.environment} world {record.index} is held twice")
        places[record.index] = record
    chosen = []
    for environment, places in sorted(held.items()):
        missing = [i for i in range(worlds) if i not in places]
        if missing:
            raise ValueError(f"the look at {worlds} misses {environment} worlds {missing[:10]}")
        chosen += [places[i] for i in range(worlds)]
    return tuple(chosen)


def _value(record: WorldRecord, arm: str, axis: str) -> float:
    score = record.arms.get(arm)
    value = None if score is None else getattr(score, axis)
    if value is None:
        raise ValueError(
            f"{arm}'s {axis} is not scored on {record.environment} world {record.seed}"
        )
    return float(value)


def paired(
    records: Sequence[WorldRecord], first: str, second: str, worlds: int, axis: str = "regret"
) -> Comparison:
    """``first``'s ``axis`` less ``second``'s over the look at ``worlds`` worlds of each
    environment, with its 95 % Student-t interval and the worlds where ``first``'s is lower, tied
    or higher.

    Raises:
        ValueError: on an axis that is not one of :data:`AXES`, a malformed look, or a world of
            the look where either arm's axis was not scored.
    """
    if axis not in AXES:
        raise ValueError(f"a paired difference reads one of {AXES}, not {axis!r}")
    seen = look(records, worlds)
    ours = np.array([_value(r, first, axis) for r in seen])
    theirs = np.array([_value(r, second, axis) for r in seen])
    return Comparison.of(second, ours, theirs)

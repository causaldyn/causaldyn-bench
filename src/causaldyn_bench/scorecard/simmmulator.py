"""Family 15: siMMMulator's own worlds, Meta's simulator run as its documentation's demo.

siMMMulator 1.1.0 (github.com/facebookexperimental/siMMMulator at commit ``da44d9a``, MIT) is
Meta's simulator of marketing-mix data, written to test models such as Robyn. The bench does not
write these worlds. ``scripts/simmmulator_worlds.R`` runs the demo of the simulator's documentation
(``siMMMulator-website/docs/demo_code.md`` at that commit) over four years, ``set.seed(seed)``
before its first step, in a scratch R library of its own (``scripts/simmmulator_worlds.lock.txt``),
and writes each world's daily record as JSON: each day's spend, cost per thousand impressions or
per click and conversion rate of each channel, and the baseline revenue; the demo's constants and
the inflexions the simulator computes; and what it returned, each day's impressions or clicks,
conversions and revenue. The bench never runs R. It reads a record only where its SHA-256 is the
one ``simmmulator_worlds.json`` holds for its seed (:data:`DIGESTS`), and holds its own map of the
record to the simulator's conversions and revenue, which it meets within 6e-16.

The map is the simulator's steps 3 to 7. For channel ``c`` on day ``d``,

    n_{c,d} = 1000 s_{c,d} / cpm_{c,d}  (Facebook, TV)  or  s_{c,d} / cpc_{c,d}  (Search),
    x_{c,d} = n_{c,d} + lambda_c x_{c,d-1},
    v_{c,d} = x_{c,d} x_{c,d}^a / (x_{c,d}^a + K_{c,d}^a),
    revenue_d = baseline_d + r sum_c cvr_{c,d} v_{c,d},

with the demo's decays ``lambda`` of 0.1, 0.2 and 0.3, shape ``a = 2`` and a revenue ``r`` of 1 a
conversion. Each day's spend is one campaign's, drawn normal and split between the channels in
shares uniform on fixed ranges, so the channels' spend moves together; each day draws its own
costs, normal, and its own rates, truncated normal. Three quirks of the simulator are kept as
written:

* its diminishing returns multiply the adstock by its Hill, so a channel's curve rises without
  bound, convex below ``sqrt(3) K`` and close to linear past the inflexion;
* the Hill's shape and inflexion are recycled over the days where the channels were meant: day
  ``d``, counted from 1, reads the ``((d - 1) mod 3)``-th of the demo's three gammas, so each
  channel's curve cycles through three inflexions every three days (R warns of the recycling);
* each inflexion is the gamma-quantile of 100 points evenly spaced over the channel's adstock
  range over the four years, rounded to four decimals: ``min + gamma (max - min)``.

What the bench reads of a record:

* **The history** is the first 156 weeks, days 1 to 1092, each week the seven days from its start,
  as the simulator's weekly table sums them (that table's last row, a partial week, lies past the
  quarter). An arm reads the weekly revenue and spend, not the impressions, clicks or costs, and no
  control: the simulator's table holds none. Money is in the simulator's dollars.
* **The curves are the world's.** The simulator computes the inflexions once, from its own spend
  over the four years; a geo test's universe and a plan read them unchanged (CHOICE). The
  simulator would recompute them from any spend it were given, so a test going dark would move the
  curve in every week, before the test too.
* **A plan's week is spent evenly over its seven days** (CHOICE), and a geo test scales each day of
  a test week by the week's multiplier.
* **The quarter** is days 1093 to 1183. Its cells read it and the 63 days after it, nothing spent
  after the quarter, by when a day's adstock keeps at most 0.3^63 of itself. The path the best plan
  is made for reads each day's cost and rate at its mean, the rate's that of the simulator's
  truncated normal (CHOICE: the mean cost in place of the mean of its reciprocal, which the cost's
  noise sets 1.5e-3 above it on Search, 5.6e-5 on TV and 1.1e-5 on Facebook). The realised path,
  which the hindsight plan reads, and the quarter's revenue at the status quo, which an arm
  forecasts, read the simulator's own draws for those days, and its baseline.
* **The returns** are read on last year, weeks 105 to 156, through the daily map, the carryover of
  the window's spend counted over the 63 days after the history at those days' own rates.

Every arm is told the kernel spans the history, as the simulator's adstock carries over every later
day. Its carryover is daily and short, so a test's dies out to the bit before the next test's
readout and the ladder's rungs nest exactly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import lfilter
from scipy.stats import truncnorm

from causaldyn_bench.budget_regret import lift_rows
from causaldyn_bench.endogenous_mmm import Curve, Market, Series
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard import ladder
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.returns import Returns
from causaldyn_bench.scorecard.seeds import pilot, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, window
from causaldyn_bench.scorecard.truth import Cell, oracle

FAMILY = 15
ENVIRONMENT = "demo"
SIMMMULATOR = "1.1.0"
COMMIT = "da44d9afdbf3012078deba7a98f4d2999dcccf93"
CHANNELS = ("Facebook", "TV", "Search")
YEARS = 4
DAYS = 7  # a week's
WEEKS = 156  # the history's
HISTORY = WEEKS * DAYS  # the history's days
PHASES = 3  # the demo's gammas, which the simulator recycles over the days
TAIL = 63  # the days after the quarter, or after the history, a cell or a return reads
HORIZON = PLANNED * DAYS + TAIL  # a cell's days, the quarter's and the tail's
TABLE = "simmmulator_worlds.json"  # beside this module: each recorded world's SHA-256
DIGESTS: Mapping[int, str] = MappingProxyType(
    {
        int(seed): digest
        for seed, digest in json.loads(files(__package__).joinpath(TABLE).read_text())[
            "worlds"
        ].items()
    }
)


def exposures(spend: ArrayLike, cost: ArrayLike, clicks: bool) -> Series:
    """The simulator's step 3: ``spend / cpm * 1000`` impressions, or ``spend / cpc`` clicks."""
    per = np.asarray(spend, dtype=float) / np.asarray(cost, dtype=float)
    return per if clicks else per * 1000.0


def adstocked(media: ArrayLike, decay: float) -> Series:
    """Its step 5b: ``x_d = n_d + decay x_{d-1}`` from ``x_1 = n_1``."""
    return lfilter([1.0], [1.0, -decay], np.asarray(media, dtype=float))


def hill(adstock: ArrayLike, shape: ArrayLike, inflexion: ArrayLike) -> Series:
    """``x^a / (x^a + K^a)``, its step 5c's s-curve."""
    rising = np.asarray(adstock, dtype=float) ** shape
    return rising / (rising + np.asarray(inflexion, dtype=float) ** shape)


def diminished(adstock: ArrayLike, shape: ArrayLike, inflexion: ArrayLike) -> Series:
    """Its step 5c's diminishing returns: the s-curve times the adstock."""
    return hill(adstock, shape, inflexion) * np.asarray(adstock, dtype=float)


def daily_curve(shape: Series, inflexion: Series) -> Curve:
    """A cell's diminishing returns over its days, each day at its own shape and inflexion, the
    inflexions read times the cell's scale. The slope is ``h + a h (1 - h)`` for the s-curve ``h``,
    0 at no adstock at the demo's shape of 2."""

    def value(adstock: Series, scale: float) -> Series:
        return diminished(adstock, shape, scale * inflexion)

    def slope(adstock: Series, scale: float) -> Series:
        h = hill(adstock, shape, scale * inflexion)
        return h + shape * h * (1.0 - h)

    return Curve(value, slope, concave=False)


def weekly(daily: Series) -> Series:
    """Each week's sum of seven days, a week the seven days from the first, along the first axis."""
    return daily.reshape(-1, DAYS, *daily.shape[1:]).sum(axis=1)


@dataclass(frozen=True, eq=False)
class Record:
    """One world as the simulator wrote it, every day of its four years: arrays over the days are
    ``(days, channels)``, the baseline's and the revenue's ``(days,)``."""

    seed: int
    clicks: NDArray[np.bool_]  # (channels,) whether a channel buys clicks, not impressions
    decay: Series  # (channels,)
    shape: Series  # (phases,) the s-curve's shape on each day of a three-day cycle
    inflexion: Series  # (channels, phases) and its inflexion
    revenue_per_conversion: float
    cost_mean: Series  # (channels,)
    cvr_mean: Series  # (channels,) the mean of a day's rate, its truncated normal's
    spend: Series
    cost: Series
    cvr: Series
    baseline: Series
    media: Series  # the simulator's own impressions or clicks
    conversions: Series  # the simulator's own
    revenue: Series  # the simulator's own

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> Record:
        """A record from the JSON ``scripts/simmmulator_worlds.R`` writes.

        Raises:
            ValueError: it is not a record of this family's demo, or breaks one of its invariants.
        """
        stamp = (data["simmmulator"], data["commit"], tuple(data["channels"]), data["years"])
        if stamp != (SIMMMULATOR, COMMIT, CHANNELS, YEARS):
            raise ValueError(f"not a record of this family's demo: {stamp}")
        constants, daily = data["constants"], data["daily"]

        def by_day(name: str) -> Series:
            return np.asarray(daily[name], dtype=float).T

        true, mean, sd = (
            np.asarray(constants[name], dtype=float)
            for name in ("true_cvr", "cvr_noise_mean", "cvr_noise_sd")
        )
        record = cls(
            seed=int(data["seed"]),
            clicks=np.array([kind == "clicks" for kind in data["kinds"]]),
            decay=np.asarray(constants["decay"], dtype=float),
            shape=np.asarray(constants["alpha"], dtype=float),
            inflexion=np.asarray(constants["inflexions"], dtype=float),
            revenue_per_conversion=float(constants["revenue_per_conv"]),
            cost_mean=np.asarray(constants["cost_mean"], dtype=float),
            # each day's rate is true_cvr plus a normal truncated below at -true_cvr
            cvr_mean=true + truncnorm.mean((-true - mean) / sd, np.inf, loc=mean, scale=sd),
            spend=by_day("spend"),
            cost=by_day("cost"),
            cvr=by_day("cvr"),
            baseline=np.asarray(daily["baseline"], dtype=float),
            media=by_day("media"),
            conversions=by_day("conversions"),
            revenue=np.asarray(daily["revenue"], dtype=float),
        )
        record.check()
        return record

    def check(self) -> None:
        """The invariants the bench's reading rests on.

        Raises:
            ValueError: one is broken.
        """
        days, channels = YEARS * 365, len(CHANNELS)
        shapes = {
            "spend": self.spend.shape,
            "cost": self.cost.shape,
            "cvr": self.cvr.shape,
            "media": self.media.shape,
            "conversions": self.conversions.shape,
        }
        if set(shapes.values()) != {(days, channels)} or {
            self.baseline.shape,
            self.revenue.shape,
        } != {(days,)}:
            raise ValueError(f"a record holds {days} days of {channels} channels: {shapes}")
        if (self.clicks.shape, self.decay.shape, self.shape.shape, self.inflexion.shape) != (
            (channels,),
            (channels,),
            (PHASES,),
            (channels, PHASES),
        ):
            raise ValueError(
                "a record holds a kind and a decay a channel, three shapes, and three "
                "inflexions a channel"
            )
        if not (np.all(self.cost > 0.0) and np.all(self.cvr >= 0.0) and np.all(self.spend >= 0.0)):
            raise ValueError(
                f"world {self.seed} has a cost of 0 or less, or a negative rate or spend"
            )
        if not np.all(weekly(self.spend[:HISTORY]) > 0.0):
            raise ValueError(
                f"world {self.seed} has a week of the history some channel spent nothing"
            )

    def returned(self, column: int, spend: Series) -> Series:
        """Channel ``column``'s revenue on each of the first ``spend.size`` days had it spent
        ``spend`` on them, at those days' own costs and rates."""
        days = spend.size
        media = exposures(spend, self.cost[:days, column], bool(self.clicks[column]))
        phase = np.arange(days) % PHASES
        curve = diminished(
            adstocked(media, self.decay[column]), self.shape[phase], self.inflexion[column, phase]
        )
        return curve * self.cvr[:days, column] * self.revenue_per_conversion


@dataclass(frozen=True, eq=False)
class SimmmulatorWorld(Market):
    """A record's history as an arm reads it, weekly: its revenue, each channel's spend and the
    revenue each channel's media bring, by the bench's map."""

    environment: str
    seed: int
    record: Record
    channels: tuple[str, ...]
    week: NDArray[np.int64]  # 1, 2, ..., 156
    sales: Series  # the simulator's own revenue
    media: Series  # (weeks, channels)
    spend: Series  # (weeks, channels)

    @classmethod
    def of(cls, environment: str, record: Record) -> SimmmulatorWorld:
        spend = record.spend[:HISTORY]
        media = np.column_stack(
            [weekly(record.returned(c, spend[:, c])) for c in range(len(CHANNELS))]
        )
        return cls(
            environment=environment,
            seed=record.seed,
            record=record,
            channels=CHANNELS,
            week=np.arange(1, WEEKS + 1),
            sales=weekly(record.revenue[:HISTORY]),
            media=media,
            spend=weekly(spend),
        )

    def _effect_of(self, column: int, spend: Series) -> Series:
        """The revenue channel ``column``'s media bring in each week had it spent ``spend``, each
        day of a week its spend there times the week's over the history's: 1.0, and the same
        bits, where the two agree."""
        ratio = np.asarray(spend, dtype=float) / self.spend[:, column]
        daily = self.record.spend[:HISTORY, column] * np.repeat(ratio, DAYS)
        return weekly(self.record.returned(column, daily))


def cells(world: SimmmulatorWorld, cost: Series, cvr: Series) -> tuple[Cell, ...]:
    """Each channel as a cell over the quarter and the tail, at each day's ``cost`` and ``cvr``,
    ``(HORIZON, channels)``: the history's carryover, the reach of a unit a week spent evenly over
    the quarter's days, and each day's curve and rate."""
    record = world.record
    phase = (HISTORY + np.arange(HORIZON)) % PHASES
    spent = np.arange(HORIZON) < PLANNED * DAYS
    lags = np.arange(1, HORIZON + 1)
    out = []
    for c in range(len(world.channels)):
        clicks = bool(record.clicks[c])
        decay = float(record.decay[c])
        before = exposures(record.spend[:HISTORY, c], record.cost[:HISTORY, c], clicks)
        carry = adstocked(before, decay)[-1] * decay**lags
        reach = adstocked(np.where(spent, exposures(1.0 / DAYS, cost[:, c], clicks), 0.0), decay)
        curve = daily_curve(record.shape[phase], record.inflexion[c, phase])
        out.append(Cell(carry, reach, curve, 1.0, record.revenue_per_conversion * cvr[:, c]))
    return tuple(out)


def true_returns(world: SimmmulatorWorld) -> Returns:
    """Each channel's ROI and mROI on last year: through the daily map, the window's spend removed
    or moved along itself, the carryover counted over the tail after the history."""
    record = world.record
    first, last = window(world)
    days = HISTORY + TAIL
    inside = np.zeros(days, dtype=bool)
    inside[(first - 1) * DAYS : last * DAYS] = True
    phase = np.arange(days) % PHASES
    roi, marginal = [], []
    for c in range(len(world.channels)):
        clicks, decay = bool(record.clicks[c]), float(record.decay[c])
        spend = np.zeros(days)
        spend[:HISTORY] = record.spend[:HISTORY, c]
        spent = float(spend[inside].sum())
        gained = record.returned(c, spend) - record.returned(c, np.where(inside, 0.0, spend))
        roi.append(float(np.sum(gained)) / spent)
        cost = record.cost[:days, c]
        adstock = adstocked(exposures(spend, cost, clicks), decay)
        own = adstocked(exposures(np.where(inside, spend, 0.0), cost, clicks), decay)
        slope = daily_curve(record.shape[phase], record.inflexion[c, phase]).slope(adstock, 1.0)
        rate = record.revenue_per_conversion * record.cvr[:days, c]
        marginal.append(float(np.sum(rate * slope * own)) / spent)
    return Returns((first, last), np.array(roi), np.array(marginal))


def read(path: Path, environment: str = ENVIRONMENT) -> SimmmulatorWorld:
    """The world recorded at ``path``, ``<seed>.json``.

    Raises:
        ValueError: no world of that seed is recorded, or the file is not the recorded one.
    """
    seed = int(path.stem)
    if seed not in DIGESTS:
        raise ValueError(f"no world {seed} is recorded in {TABLE}")
    data = path.read_bytes()
    found = hashlib.sha256(data).hexdigest()
    if found != DIGESTS[seed]:
        raise ValueError(f"{path} is not world {seed} as recorded: SHA-256 {found}")
    record = Record.parse(json.loads(data))
    if record.seed != seed:
        raise ValueError(f"{path} holds world {record.seed}")
    return SimmmulatorWorld.of(environment, record)


class Simmmulator:
    """Family 15, its worlds read from ``directory``."""

    name = "siMMMulator's own"
    stream = stream(FAMILY)
    labels = (
        "The world is siMMMulator 1.1.0's own, Meta's simulator, run as its documentation's demo "
        "over four years: each day's impressions or clicks are the spend over a noisy cost, "
        "carried by an unnormalised geometric adstock into every later day and through its "
        "diminishing returns, the adstock times a Hill of it, into conversions at a noisy rate.",
        "Its quirks are kept as written: the Hill's shape and inflexion recycled over the days "
        "rather than the channels, each inflexion a quantile of the channel's adstock range over "
        "the four years, and a curve that rises without bound.",
        "Every arm is told the kernel spans the history, as the simulator's adstock does; its "
        "carryover is daily, at most 0.3 of a day's adstock kept into the next.",
        "An arm reads the simulator's weekly revenue and spend, and no control; a plan's week is "
        "spent evenly over its seven days.",
        "The demo's largest channel by spend, Facebook, returns the most per dollar in every "
        "world, by its costs and rates: a rule that moves budget to the largest channel finds the "
        "best plan.",
        TRACK_M2.labels[2],
    )
    pilots = MappingProxyType({ENVIRONMENT: pilot(FAMILY, 0)})
    scored = MappingProxyType({ENVIRONMENT: scored(FAMILY, 0)})

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def world(self, environment: str, seed: int) -> SimmmulatorWorld:
        if environment not in self.scored:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        return read(self.directory / f"{seed}.json", environment)

    def observe(self, world: SimmmulatorWorld, k: int) -> Observation:
        quarter = Quarter.after(world)
        return Observation(
            family=FAMILY,
            environment=world.environment,
            seed=world.seed,
            k=k,
            channels=world.channels,
            sales=world.sales,
            spend=world.spend,
            controls={},
            future_controls={},
            kernel_length=WEEKS,
            planned=PLANNED,
            budget=quarter.budget,
            lower=quarter.lower,
            upper=quarter.upper,
            status_quo=quarter.status_quo,
            roi_window=window(world),
            lift=lift_rows(ladder.rung(world, world.seed, k)),
        )

    def truth(self, world: SimmmulatorWorld) -> Truth:
        record = world.record
        quarter = Quarter.after(world)
        span = slice(HISTORY, HISTORY + HORIZON)
        channels = len(world.channels)
        expected = cells(
            world,
            np.broadcast_to(record.cost_mean, (HORIZON, channels)),
            np.broadcast_to(record.cvr_mean, (HORIZON, channels)),
        )
        realised = cells(world, record.cost[span], record.cvr[span])
        days = PLANNED * DAYS
        media = np.sum(
            [
                cell.returns(float(w))[:days]
                for cell, w in zip(realised, quarter.status_quo, strict=True)
            ],
            axis=0,
        )
        return Truth(
            quarter=quarter,
            cells=expected,
            best=oracle(expected, quarter),
            realised=realised,
            hindsight=oracle(realised, quarter),
            target=weekly(record.baseline[HISTORY : HISTORY + days] + media),
            scale=float(np.mean(world.sales)),
            returns=true_returns(world),
        )

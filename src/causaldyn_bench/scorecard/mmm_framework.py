"""Family 21: mmm-framework's answer keys, a rival's worlds read as data.

mmm-framework 1.5.0 (github.com/redam94/mmm-framework, Apache-2.0) is a Bayesian media-mix tool
whose wheel ships two synthetic data sets, each a long "master flat file" (one row a period, a
geography and a variable) and a JSON of what its generator states the truth to be. The bench does
not write these worlds and holds none of the tool's code: the four files are copied unchanged from
the wheel into ``mmm_framework_worlds/`` beside this module, whose ``NOTICE`` names their licence
and the wheel's SHA-256, and a world is read only where each file's SHA-256 is the one
:data:`DIGESTS` holds.

* **national**: 104 weeks of sales, seven channels and thirteen controls, the tool's "realistic"
  scenario: two channels weak, two mediated through a brand-awareness measure the file holds, two
  confounders among the controls, and six controls of pure noise; the file names each control's
  role.
* **geo**: 91 weeks of eight geographies, four channels and a price, the tool's
  "geo_heterogeneous" scenario: each channel's effect differs by geography, 0.3 to 1.8 times, and
  the budgets chase performance.

**What the truth states, and what it does not.** Each channel's contribution over the file's weeks
and its return per unit of spend (ROAS), and on the geo set each of those by geography. It states
no response curve, no adstock and no future, so no plan can be scored against it: the family scores
an arm's reading of each channel's contribution and ROAS against the stated ones (:func:`score`),
and is not a :class:`~causaldyn_bench.scorecard.family.Family`. The two sets read the stated
numbers differently:

* on the geo set, each channel's stated ROAS is its stated contribution over its spend in the file,
  by geography and in total, within 4e-15 of it, and the geographies' contributions add up to the
  total within 2e-16 of it;
* on the national set they do not: a channel's stated contribution over its stated ROAS is 0.80 to
  0.90 of its spend in the file, the share differing by channel, and no run of the file's weeks
  brings every channel within 7 % of it. The spend the stated ROAS is read on is not in the files,
  so an arm's ROAS is scored against the stated ROAS as stated, and its contribution against the
  stated contribution, which needs no spend (:func:`denominator`).

**The generator's worlds** (:data:`MARKETS`). The tool's other synthetic worlds are not files:
its generator, ``mmm_framework.synth``, draws them. ``scripts/mmm_framework_markets.py`` ran it
once as a black box, offline and in an environment of its own, by the call its docstrings document,
``generate_mff(name)`` at the world's default seed and length, and the files it wrote are kept
unchanged in ``mmm_framework_markets/`` beside this module with their ``NOTICE``. A world is read
only where that directory's ``MANIFEST.json`` has the SHA-256 :data:`MANIFEST_SHA256` holds and each
file the SHA-256 the manifest names. The tool's pages count sixteen stress worlds, but none of the
wheel's documents names them, so the family holds every world of the generator's registry it can
read: 26, the 24 national ones of 156 weeks and two panels of four geographies and 130 weeks. Two
it cannot read: ``aurora_kitchen_sink``, which the generator refuses to draw outside the tool's
repository, and ``geo_product``, whose file fills the product dimension. On 24 of the 26 the stated
ROAS is the stated contribution over the file's spend; on ``spend_outliers`` and
``mixed_data_errors``, whose recorded spend carries the recording errors the tool plants, it is
over the true spend, which the key's notes state. The keys' notes state more of a world, each
channel's carry-over and the planted violation's own parameters, but no channel's whole response
curve, so here too an arm's reading is scored, not a plan.

**The bench's arm that reads these worlds** (:func:`observational`) is the least-squares media mix
of :mod:`causaldyn_bench.observational_mmm`, the observational arm of Track M v2's budgets run: each
channel's spend through a normalised six-week geometric kernel and ``tanh``, the coefficients kept
at or above zero, beside an intercept, the controls and three yearly harmonics. CHOICES: it reads
the controls the truth's roles do not exclude, every variable of the file but the sales and the
channels, less a mediator and the controls of pure noise, so it is handed the confounders the
national set's spend follows; it reads the geo set as one national series, the sales and the spend
summed over the geographies and the price their mean, so it reads each channel's total and not a
geography's. A channel's contribution is its fitted return over the file's weeks, its ROAS that
over its spend there. The bench's other arms read nothing here: the status quo and the equal split
are plans, and there is no plan to score; CHC and the myopic arm read lift tests, and the files hold
none.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import itertools
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
from numpy.typing import NDArray

from causaldyn_bench.endogenous_mmm import YEAR, Series, Vector
from causaldyn_bench.observational_mmm import HARMONICS, fit_observational

FAMILY = 21
MMM_FRAMEWORK = "1.5.0"
WHEEL = "mmm_framework-1.5.0-py3-none-any.whl"
WHEEL_SHA256 = "d9e28710deb382e861f88c05e030b69a51dd5ed0b0e31855ce774382cd5be366"
DATA = "mmm_framework_worlds"  # the directory beside this module that holds the files
MARKETS_DATA = "mmm_framework_markets"  # the one that holds the generator's worlds
MANIFEST = "MANIFEST.json"
MANIFEST_SHA256 = "7feddddcdfe7fca84c9099d45ad0d8f47444ae016ef5d0cc53ca060817bad028"
MARKETS = (  # in the order of the generator's registry, ``mmm_framework.synth.dgp.SCENARIOS``
    "realistic",
    "clean",
    "unobserved_confounding",
    "confounding_controlled",
    "reverse_causality",
    "multicollinearity",
    "adstock_misspec",
    "saturation_misspec",
    "time_varying_beta",
    "heavy_tailed_noise",
    "synergy",
    "spend_outliers",
    "mixed_data_errors",
    "negative_effect",
    "trend_break",
    "seasonality_misspec",
    "dense_controls",
    "economic_health",
    "reach_frequency",
    "promo_and_media",
    "promo_endogenous",
    "breakout_heterogeneous",
    "breakout_homogeneous",
    "breakout_collinear",
    "geo_clean",
    "geo_heterogeneous",
)
UNREAD_MARKETS = MappingProxyType(
    {
        "aurora_kitchen_sink": "the generator refuses to draw it outside the tool's repository",
        "geo_product": "its file fills the product dimension, which the bench does not read",
    }
)
ENVIRONMENTS = ("national", "geo")
KPI = "Sales"
DIGESTS: Mapping[str, str] = MappingProxyType(
    {
        "national_mff.csv": "84e113a7adf7bc254ee54ba14f5f307d84da8b463b196dda5c45fc075a3f7f11",
        "national_truth.json": "90b66439a1837c965f3afbbaed0b9120fcc82ae9335a37397b3f290829c3602f",
        "geo_mff.csv": "9f4fa9ae55bbaeda2d14ab60e55e8cb22e937c10ffa7d20852cd087785103ca0",
        "geo_truth.json": "7e6bc52a6cc34278a7ff015b22198785ed2cd7d4b53e975800ba2a559bfbaca5",
    }
)
COLUMNS = (
    "Period",
    "Geography",
    "Product",
    "Campaign",
    "Outlet",
    "Creative",
    "VariableName",
    "VariableValue",
)
UNREAD = ("Product", "Campaign", "Outlet", "Creative")  # dimensions both files leave empty
EXCLUDED = ("mediator", "irrelevant")  # the roles whose controls an arm does not read
LENGTH = 6  # the observational arm's kernel, in weeks: its own, Track M v2's
ROOT = Path(__file__).resolve().parents[3]  # the repository, whose results/ the pilot writes


@dataclass(frozen=True, eq=False)
class Panel:
    """A master flat file read as arrays: each variable's value in each week and geography."""

    periods: tuple[datetime.date, ...]  # a week apart
    geos: tuple[str, ...]  # ``("",)`` for a national file, whose rows name no geography
    variables: Mapping[str, Series]  # each ``(weeks, geos)``

    def __getitem__(self, name: str) -> Series:
        return self.variables[name]


def read_mff(path: Path) -> Panel:
    """The master flat file at ``path``, every cell of its periods, geographies and variables
    given once.

    Raises:
        ValueError: its header is not the eight columns, it fills a dimension the bench does not
            read, a cell is missing, given twice or not finite, or its periods are not a week apart.
    """
    with Path(path).open(newline="") as stream:
        rows = csv.reader(stream)
        header = tuple(next(rows))
        if header != COLUMNS:
            raise ValueError(f"{path} is not a master flat file: its header is {header}")
        cells: dict[tuple[datetime.date, str, str], float] = {}
        for row in rows:
            entry = dict(zip(COLUMNS, row, strict=True))
            if any(entry[name] for name in UNREAD):
                raise ValueError(f"{path} fills a dimension the bench does not read: {row}")
            period = datetime.date.fromisoformat(entry["Period"])
            key = (period, entry["Geography"], entry["VariableName"])
            if key in cells:
                raise ValueError(f"{path} gives {key} twice")
            value = float(entry["VariableValue"])
            if not math.isfinite(value):
                raise ValueError(f"{path} gives {key} as {value}")
            cells[key] = value
    periods = sorted({p for p, _, _ in cells})
    if any((b - a).days != 7 for a, b in itertools.pairwise(periods)):
        raise ValueError(f"{path}'s periods are not a week apart")
    geos = tuple(dict.fromkeys(g for _, g, _ in cells))
    names = tuple(dict.fromkeys(v for _, _, v in cells))
    if len(cells) != len(periods) * len(geos) * len(names):
        raise ValueError(f"{path} misses a cell of its periods, geographies and variables")
    variables = {
        name: np.array([[cells[(p, g, name)] for g in geos] for p in periods], dtype=float)
        for name in names
    }
    return Panel(tuple(periods), geos, MappingProxyType(variables))


@dataclass(frozen=True, eq=False)
class AnswerKey:
    """What a truth file states of its world: each channel's contribution over the file's weeks and
    its ROAS, by geography where it states them, and each control's role where it names one."""

    scenario: str
    description: str
    violates: str
    representable: bool  # whether the tool says its own model can represent the world
    channels: tuple[str, ...]
    contribution: Vector  # (channels,)
    roas: Vector  # (channels,)
    geos: tuple[str, ...] | None
    contribution_by_geo: Series | None  # (geos, channels)
    roas_by_geo: Series | None  # (geos, channels)
    roles: Mapping[str, str]  # each control's role, as the file names it

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> AnswerKey:
        """A key from a truth file's JSON.

        Raises:
            ValueError: it states a channel's contribution or ROAS but not both, or a geography's
                but not every channel's.
        """
        channels = tuple(data["channels"])
        if set(data["true_contribution"]) != set(channels) or set(data["true_roas"]) != set(
            channels
        ):
            raise ValueError("a truth file states each channel's contribution and ROAS")

        def by_channel(stated: Mapping[str, float]) -> Vector:
            return np.array([float(stated[c]) for c in channels])

        geos = data.get("geographies")
        by_geo: dict[str, Series | None] = {"contribution": None, "roas": None}
        if geos:
            for name in by_geo:
                stated = data[f"true_{name}_by_geo"]
                if set(stated) != set(geos) or any(set(stated[g]) != set(channels) for g in geos):
                    raise ValueError(f"a truth file states every channel's {name} in every geo")
                by_geo[name] = np.array([by_channel(stated[g]) for g in geos])
        roles = (data.get("notes") or {}).get("roles") or {}
        return cls(
            scenario=str(data["scenario"]),
            description=str(data["description"]),
            violates=str(data["violates"]),
            representable=bool(data["representable"]),
            channels=channels,
            contribution=by_channel(data["true_contribution"]),
            roas=by_channel(data["true_roas"]),
            geos=tuple(geos) if geos else None,
            contribution_by_geo=by_geo["contribution"],
            roas_by_geo=by_geo["roas"],
            roles=MappingProxyType({str(k): str(v) for k, v in roles.items()}),
        )


def read_truth(path: Path) -> AnswerKey:
    """The answer key at ``path``."""
    return AnswerKey.parse(json.loads(Path(path).read_text()))


@dataclass(frozen=True, eq=False)
class MffWorld:
    """One of the tool's data sets: its panel, read in the key's channels and geographies."""

    environment: str
    panel: Panel
    key: AnswerKey
    controls: tuple[str, ...]  # the variables an arm reads beside the sales and the spend

    @classmethod
    def of(cls, environment: str, panel: Panel, key: AnswerKey) -> MffWorld:
        """Raises:
        ValueError: the panel holds no sales or misses a channel, or its geographies are not
            the key's.
        """
        missing = {KPI, *key.channels} - set(panel.variables)
        if missing:
            raise ValueError(f"the panel holds no {sorted(missing)}")
        if key.geos is not None and set(panel.geos) != set(key.geos):
            raise ValueError(f"the panel's geographies {panel.geos} are not the key's {key.geos}")
        if key.geos is None and panel.geos != ("",):
            raise ValueError(f"a key with no geographies reads a national panel: {panel.geos}")
        if key.geos is not None:
            order = [panel.geos.index(g) for g in key.geos]
            panel = Panel(
                panel.periods,
                key.geos,
                MappingProxyType({n: v[:, order] for n, v in panel.variables.items()}),
            )
        controls = tuple(
            name
            for name in panel.variables
            if name != KPI and name not in key.channels and key.roles.get(name) not in EXCLUDED
        )
        return cls(environment, panel, key, controls)

    @property
    def sales(self) -> Series:
        """``(weeks, geos)``."""
        return self.panel[KPI]

    @property
    def spend(self) -> Series:
        """``(weeks, geos, channels)``."""
        return np.stack([self.panel[c] for c in self.key.channels], axis=-1)


def denominator(world: MffWorld) -> Vector:
    """Each channel's stated contribution over its stated ROAS, as a share of its spend in the
    file: 1 where the stated ROAS is read on the file's spend."""
    spent = world.spend.sum(axis=(0, 1))
    return world.key.contribution / world.key.roas / spent


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(directory: Path, environment: str) -> MffWorld:
    """The data set ``environment`` from ``directory``, its two files the recorded ones.

    Raises:
        ValueError: the family has no such data set, or a file is not the recorded one.
    """
    if environment not in ENVIRONMENTS:
        raise ValueError(f"mmm-framework's answer keys have no environment {environment!r}")
    paths = {kind: Path(directory) / f"{environment}_{kind}" for kind in ("mff.csv", "truth.json")}
    for path in paths.values():
        found = _digest(path)
        if found != DIGESTS[path.name]:
            raise ValueError(f"{path} is not mmm-framework {MMM_FRAMEWORK}'s: SHA-256 {found}")
    return MffWorld.of(environment, read_mff(paths["mff.csv"]), read_truth(paths["truth.json"]))


def read_market(directory: Path, name: str) -> MffWorld:
    """The generator's world ``name`` from ``directory``, its manifest the recorded one and its two
    files those the manifest names.

    Raises:
        ValueError: the family holds no such world, or the manifest or a file is not the recorded
            one.
    """
    if name not in MARKETS:
        raise ValueError(f"mmm-framework's generated worlds hold no {name!r}")
    manifest = Path(directory) / MANIFEST
    found = _digest(manifest)
    if found != MANIFEST_SHA256:
        raise ValueError(f"{manifest} is not the recorded manifest: SHA-256 {found}")
    recorded = json.loads(manifest.read_text())["worlds"][name]
    paths = {kind: Path(directory) / f"{name}_{kind}" for kind in ("mff.csv", "truth.json")}
    for kind, path in paths.items():
        found = _digest(path)
        if found != recorded[kind]["sha256"]:
            raise ValueError(f"{path} is not what mmm-framework's generator wrote: SHA-256 {found}")
    return MffWorld.of(name, read_mff(paths["mff.csv"]), read_truth(paths["truth.json"]))


class MmmFramework:
    """Family 21, its two shipped sets read from ``directory`` and the generator's worlds from
    ``markets``, by default the files the bench holds."""

    name = "mmm-framework's answer keys"
    environments = ENVIRONMENTS + MARKETS
    labels = (
        "The worlds are mmm-framework 1.5.0's own, the two data sets its wheel ships, and their "
        "truth is what its generator states: a rival's worlds, written to test its own model.",
        "So are the 26 the bench holds of its generator's: drawn once by the tool's own code at "
        "each world's default seed and length, read as files; which sixteen of them the tool's "
        "pages count, no document of its wheel says.",
        "The tool wrote them to test its own model: the national set's adstock and saturation "
        "are in that model's family, by the truth file's own account, which hands the tool's own "
        "fit its forms.",
        "The truth states each channel's contribution and ROAS, no curve: an arm's reading is "
        "scored, not a plan.",
        "On the national set the stated ROAS is not read on the file's spend: the spend it is "
        "read on is 0.80 to 0.90 of the file's, by channel; a contribution needs no spend.",
        "The bench's own arm is handed the controls the truth's roles keep, the confounders among "
        "them, and reads the geo set's total, not a geography's.",
    )

    def __init__(self, directory: Path | None = None, markets: Path | None = None) -> None:
        self.directory = Path(str(files("causaldyn_bench.scorecard").joinpath(DATA)))
        self.markets = Path(str(files("causaldyn_bench.scorecard").joinpath(MARKETS_DATA)))
        if directory is not None:
            self.directory = Path(directory)
        if markets is not None:
            self.markets = Path(markets)

    def world(self, environment: str) -> MffWorld:
        if environment in MARKETS:
            return read_market(self.markets, environment)
        return read(self.directory, environment)

    def truth(self, world: MffWorld) -> AnswerKey:
        return world.key


@dataclass(frozen=True, eq=False)
class Reading:
    """An arm's reading of a world's channels, in the key's order: each one's contribution over the
    file's weeks and its ROAS, with an interval of its ROAS where the arm gives one."""

    arm: str
    contribution: Vector
    roas: Vector
    roas_interval: Series | None = None  # (channels, 2)


@dataclass(frozen=True, eq=False)
class KeyScore:
    """A reading against the key: each channel's relative error, the reading over the stated less
    1, and the share of the channels whose stated ROAS its interval holds."""

    arm: str
    contribution_error: Vector
    roas_error: Vector
    covered: float | None

    @property
    def contribution(self) -> float:
        """The relative error's size, averaged over the channels."""
        return float(np.mean(np.abs(self.contribution_error)))

    @property
    def roas(self) -> float:
        """The relative error's size, averaged over the channels."""
        return float(np.mean(np.abs(self.roas_error)))


def score(key: AnswerKey, reading: Reading) -> KeyScore:
    """``reading`` against ``key``.

    Raises:
        ValueError: the reading is not one of each of the key's channels.
    """
    count = len(key.channels)
    shapes = (np.shape(reading.contribution), np.shape(reading.roas))
    if shapes != ((count,), (count,)) or (
        reading.roas_interval is not None and np.shape(reading.roas_interval) != (count, 2)
    ):
        raise ValueError(f"a reading gives each of the {count} channels one number: {shapes}")
    covered = None
    if reading.roas_interval is not None:
        low, high = np.asarray(reading.roas_interval).T
        covered = float(np.mean((low <= key.roas) & (key.roas <= high)))
    return KeyScore(
        reading.arm,
        np.asarray(reading.contribution) / key.contribution - 1.0,
        np.asarray(reading.roas) / key.roas - 1.0,
        covered,
    )


@dataclass(frozen=True)
class Totalled:
    """A world as the observational arm reads it: one weekly series of sales and of each channel's
    spend, and the controls it reads."""

    channels: tuple[str, ...]
    week: NDArray[np.int64]
    sales: Series
    spend: Series  # (weeks, channels)
    columns: Series  # (weeks, controls)

    @classmethod
    def of(cls, world: MffWorld) -> Totalled:
        """The sales and the spend summed over the geographies, each control their mean."""
        weeks = len(world.panel.periods)
        return cls(
            channels=world.key.channels,
            week=np.arange(1, weeks + 1),
            sales=world.sales.sum(axis=1),
            spend=world.spend.sum(axis=1),
            columns=np.column_stack(
                [world.panel[c].mean(axis=1) for c in world.controls] or [np.empty((weeks, 0))]
            ),
        )

    def controls(self) -> Series:
        """``(weeks, 1 + controls + 2 * HARMONICS)``: intercept, the controls, sine and cosine."""
        angle = 2.0 * math.pi * self.week / YEAR
        harmonics = [f(k * angle) for k in range(1, HARMONICS + 1) for f in (np.sin, np.cos)]
        return np.column_stack([np.ones(self.week.size), self.columns, *harmonics])


def observational(world: MffWorld) -> Reading:
    """The bench's least-squares arm on ``world`` (:func:`fit_observational`), read as each
    channel's fitted return over the file's weeks and that over its spend there."""
    totalled = Totalled.of(world)
    fitted = fit_observational(totalled, length=LENGTH)
    contribution = np.array(
        [float(np.sum(np.asarray(ch(totalled.spend[:, c])))) for c, ch in enumerate(fitted)]
    )
    return Reading("observational", contribution, contribution / totalled.spend.sum(axis=0))


def pilot_rows(
    worlds: Sequence[MffWorld], readings: Mapping[str, Sequence[Reading]]
) -> list[dict[str, Any]]:
    """Each world's key and every arm's reading of it with its score, as the pilot reports them."""
    rows = []
    for world in worlds:
        key = world.key
        arms = []
        for reading in readings.get(world.environment, ()):
            scored = score(key, reading)
            arms.append(
                {
                    "arm": reading.arm,
                    "contribution": reading.contribution.tolist(),
                    "roas": reading.roas.tolist(),
                    "roas_interval": None
                    if reading.roas_interval is None
                    else np.asarray(reading.roas_interval).tolist(),
                    "contribution_error": scored.contribution_error.tolist(),
                    "roas_error": scored.roas_error.tolist(),
                    "mean_contribution_error": scored.contribution,
                    "mean_roas_error": scored.roas,
                    "covered": scored.covered,
                }
            )
        rows.append(
            {
                "environment": world.environment,
                "scenario": key.scenario,
                "violates": key.violates,
                "representable": key.representable,
                "channels": list(key.channels),
                "controls": list(world.controls),
                "contribution": key.contribution.tolist(),
                "roas": key.roas.tolist(),
                "denominator": denominator(world).tolist(),
                "arms": arms,
            }
        )
    return rows


def tool_reading(record: Mapping[str, Any], world: MffWorld) -> Reading:
    """mmm-framework's own fit of ``world`` (``scripts/mmm_framework_arm.py``), as its analyser's
    ROI table reports it: each channel's contribution, its ROI, which is that over the channel's
    spend in the file, and the interval of its contribution over that spend.

    Raises:
        ValueError: the table is not one row a channel of the key, its spend is not the file's,
            or its ROI is not its contribution over that spend.
    """
    table = record["roi"]
    columns = list(table["columns"])
    rows = {row[columns.index("Channel")]: row for row in table["values"]}
    if sorted(rows) != sorted(world.key.channels):
        raise ValueError(f"the tool's table is not one row a channel: {sorted(rows)}")

    def column(name: str) -> Vector:
        return np.array([float(rows[c][columns.index(name)]) for c in world.key.channels])

    spend = column("Total Spend")
    spent = world.spend.sum(axis=(0, 1))
    if not np.allclose(spend, spent, rtol=1e-12, atol=0.0):
        raise ValueError(f"the tool's spend {spend} is not the file's {spent}")
    contribution, roas = column("Total Contribution"), column("ROI")
    if not np.allclose(roas, contribution / spend, rtol=1e-12, atol=0.0):
        raise ValueError("the tool's ROI is not its contribution over the spend")
    low, high = column("Contribution HDI Low"), column("Contribution HDI High")
    return Reading(
        "mmm-framework", contribution, roas, np.column_stack([low, high]) / spend[:, None]
    )


NO_CELL = MappingProxyType(
    {
        "status quo": "a plan, and the truth states no curve to score a plan on",
        "equal split": "a plan, and the truth states no curve to score a plan on",
        "CHC": "reads each channel from its lift tests, and the files hold none",
        "myopic": "reads each channel from its lift tests, and the files hold none",
    }
)


def pilot(
    fits: Path | None = None,
    directory: Path | None = None,
    environments: Sequence[str] = ENVIRONMENTS,
) -> dict[str, Any]:
    """Every world of ``environments`` read by the bench's arm and, where ``fits`` holds its record
    of the world, ``<environment>.json``, by the tool's own fit; with what each world's key
    states."""
    family = MmmFramework(directory)
    worlds = [family.world(e) for e in environments]
    readings: dict[str, list[Reading]] = {w.environment: [observational(w)] for w in worlds}
    records = {}
    for world in worlds:
        path = None if fits is None else Path(fits) / f"{world.environment}.json"
        if path is not None and path.exists():
            record = json.loads(path.read_text())
            records[world.environment] = {
                "version": record["version"],
                "settings": record["settings"],
                "fit_warnings": record["fit_warnings"],  # each distinct one, with how often
            }
            readings[world.environment].append(tool_reading(record, world))
    data = {
        "family": FAMILY,
        "mmm_framework": MMM_FRAMEWORK,
        "wheel": {"name": WHEEL, "sha256": WHEEL_SHA256},
        "x64": bool(jax.config.read("jax_enable_x64")),  # the arm's channels are chc's, JAX's
        "worlds": pilot_rows(worlds, readings),
        "tool": records,
        "no_cell": dict(NO_CELL),
    }
    if any(e in MARKETS for e in environments):
        data["generator"] = {
            "call": "mmm_framework.synth.generate_mff(name)",
            "manifest_sha256": MANIFEST_SHA256,
            "unread": dict(UNREAD_MARKETS),
        }
    return data


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _channel_table(world: Mapping[str, Any]) -> list[str]:
    arms = world["arms"]
    lines = [
        "| channel | stated contribution | "
        + " | ".join(f"{a['arm']}" for a in arms)
        + " | stated ROAS | "
        + " | ".join(f"{a['arm']}" for a in arms)
        + " | stated over the file's spend |",
        "|---" * (4 + 2 * len(arms)) + "|",
    ]
    for c, channel in enumerate(world["channels"]):
        contributions = [
            f"{a['contribution'][c]:.1f} [{_signed(a['contribution_error'][c])}]" for a in arms
        ]
        returns = []
        for a in arms:
            text = f"{a['roas'][c]:.3f} [{_signed(a['roas_error'][c])}]"
            if a["roas_interval"] is not None:
                low, high = a["roas_interval"][c]
                text += f" ({low:.3f}, {high:.3f})"
            returns.append(text)
        lines.append(
            f"| {channel} | {world['contribution'][c]:.1f} | "
            + " | ".join(contributions)
            + f" | {world['roas'][c]:.3f} | "
            + " | ".join(returns)
            + f" | {world['denominator'][c]:.4f} |"
        )
    return lines


def markdown(data: Mapping[str, Any]) -> str:
    """The pilot's report."""
    lines = [
        f"# Family {data['family']}: mmm-framework's answer keys, pilot "
        f"({'float64' if data['x64'] else 'float32'})",
        "",
        f"The two data sets mmm-framework {data['mmm_framework']} ships with its wheel, read as "
        "data: a rival's worlds, written to test its own model. Each arm's reading of each "
        "channel's contribution over the file's weeks and its ROAS, against what the tool's truth "
        "file states, with the relative error, the reading over the stated less 1, in brackets.",
    ]
    for world in data["worlds"]:
        arms = world["arms"]
        lines += [
            "",
            f"## {world['environment']}: {world['scenario']}",
            "",
            f"What it breaks, by the truth file: {world['violates']}. The tool's model can "
            f"represent it: {'yes' if world['representable'] else 'no'}, by the same file. The "
            f"bench's arm reads the controls {', '.join(world['controls'])}.",
            "",
            *_channel_table(world),
        ]
        lines.append("")
        for a in arms:
            text = (
                f"{a['arm']}: the relative error's size averaged over the channels, "
                f"{a['mean_contribution_error']:.3f} on the contribution and "
                f"{a['mean_roas_error']:.3f} on the ROAS"
            )
            if a["covered"] is not None:
                text += (
                    f"; its interval, in parentheses, holds the stated ROAS of "
                    f"{round(a['covered'] * len(world['channels']))} of "
                    f"{len(world['channels'])} channels"
                )
            lines.append(text + ".")
    lines += [
        "",
        "The last column is each channel's stated contribution over its stated ROAS, as a share of "
        "its spend in the file: 1 where the stated ROAS is read on the file's spend. On the "
        "national set it is not, and the spend it is read on is not in the files, so a ROAS read "
        "on the file's spend, as both arms read it, is scored against a number on another "
        "denominator; the contribution needs none.",
    ]
    if data["tool"]:
        lines += [
            "",
            "## The tool's own fit",
            "",
            "mmm-framework fitted as its README's Python example fits a model "
            "(`scripts/mmm_framework_arm.py`, in the environment `scripts/mmm_framework_arm.txt` "
            "pins): `BayesianMMM` on the file through `create_simple_mff_config` and `load_mff`, "
            "a linear trend, NumPyro's NUTS at the example's chains, draws and tuning, read off "
            "`MMMAnalyzer.compute_channel_roi`. Its interval is that table's HDI of the "
            "contribution over the channel's spend, at the level the analyser uses, which the "
            "table does not name. It reads the same controls as the bench's arm. The seed does "
            "not fix its numbers: four runs of the script on one machine moved each channel's "
            "contribution by at most 0.5 % of its mean.",
        ]
        for environment, record in data["tool"].items():
            settings = record["settings"]
            lines.append("")
            lines.append(
                f"{environment}: {settings.get('chains')} chains of {settings.get('tune')} tuning "
                f"and {settings.get('draws')} kept draws, seed {settings.get('seed')}."
            )
            warned = record["fit_warnings"]
            lines += [
                f"- {warning} ({'once' if count == 1 else f'{count} times'})"
                for warning, count in warned.items()
            ]
            if not any(w.startswith("ConvergenceWarning") for w in warned):
                lines.append(
                    "- No ConvergenceWarning, which the tool raises where its own check of a fit "
                    "fails."
                )
    lines += [
        "",
        "## Arms without a cell",
        "",
        *[f"- {arm}: {why}." for arm, why in data["no_cell"].items()],
        "- The worlds the tool's generator draws are read apart: "
        "`results/mmm_framework_markets.md`.",
    ]
    return "\n".join(lines) + "\n"


def _fit_check(warned: Mapping[str, int]) -> str:
    raised = any(w.startswith("ConvergenceWarning") for w in warned)
    return "a ConvergenceWarning" if raised else "silent"


def markets_markdown(data: Mapping[str, Any]) -> str:
    """The report of the pilot on the generator's worlds."""
    worlds = data["worlds"]
    count = len(worlds)
    lines = [
        f"# Family {data['family']}: mmm-framework's generated worlds, pilot "
        f"({'float64' if data['x64'] else 'float32'})",
        "",
        f"The {count} worlds of mmm-framework {data['mmm_framework']}'s generator the bench "
        "holds, each drawn once by the tool's own code at its default seed and length "
        f"(`{data['generator']['call']}`) and read as data: a rival's worlds, written to test its "
        "own model. The tool's pages count sixteen stress worlds, but none of its wheel's "
        "documents names them, so every world of its registry the bench can read is here. Each "
        "arm's reading of each channel's contribution over the file's weeks and its ROAS is "
        "scored against the world's answer key: the relative error, the reading over the stated "
        "less 1, its size averaged over the channels.",
        "",
        "| world | what it breaks, by its key | the tool's model can represent it | channels | "
        "observational: contribution, ROAS | mmm-framework: contribution, ROAS | "
        "mmm-framework's interval holds | the tool's own check |",
        "|---|---|---|---|---|---|---|---|",
    ]
    errors: dict[str, list[float]] = {}
    held = []
    for world in worlds:
        arms = {a["arm"]: a for a in world["arms"]}
        cells = []
        for name in ("observational", "mmm-framework"):
            arm = arms.get(name)
            if arm is None:
                cells.append("no cell")
                continue
            errors.setdefault(name, []).append(arm["mean_contribution_error"])
            cells.append(f"{arm['mean_contribution_error']:.3f}, {arm['mean_roas_error']:.3f}")
        tool = arms.get("mmm-framework")
        holds, check = "no cell", "no cell"
        if tool is not None and tool["covered"] is not None:
            channels = len(world["channels"])
            holds = f"{round(tool['covered'] * channels)} of {channels}"
            held.append(round(tool["covered"] * channels) == channels)
        if world["environment"] in data["tool"]:
            check = _fit_check(data["tool"][world["environment"]]["fit_warnings"])
        lines.append(
            f"| {world['environment']} | {world['violates'] or 'nothing, by its key'} | "
            f"{'yes' if world['representable'] else 'no'} | {len(world['channels'])} | "
            + " | ".join(cells)
            + f" | {holds} | {check} |"
        )
    lines.append("")
    for name, values in errors.items():
        lines.append(
            f"{name}: a cell on {len(values)} of {count} worlds; the contribution's error, "
            f"averaged over a world's channels, {np.median(values):.3f} at the median world, "
            f"{min(values):.3f} to {max(values):.3f}."
        )
    if len(errors.get("mmm-framework", ())) == count == len(errors.get("observational", ())):
        fewer = sum(
            t < o for t, o in zip(errors["mmm-framework"], errors["observational"], strict=True)
        )
        raised = sum(
            _fit_check(record["fit_warnings"]) != "silent" for record in data["tool"].values()
        )
        lines.append(
            f"mmm-framework's contribution error is the smaller of the two on {fewer} of "
            f"{count} worlds; its interval holds every channel's stated ROAS on {sum(held)} of "
            f"the {count}; its own check raised a ConvergenceWarning on {raised} of them."
        )
    lines += [
        "",
        "\"The tool's own check\" is whether the tool's fit raised a ConvergenceWarning, which it "
        'raises where its own check of a fit fails; "silent" where it raised none.',
        "",
        "## The worlds, channel by channel",
        "",
        "The last column is each channel's stated contribution over its stated ROAS, as a share "
        "of its spend in the file: 1 where the stated ROAS is read on the file's spend. Where the "
        "tool plants recording errors in the spend the file holds, the stated ROAS is read on "
        "the true spend, which the key's notes state; the contribution needs none.",
    ]
    for world in worlds:
        lines += [
            "",
            f"### {world['environment']}",
            "",
            f"What it breaks, by its key: {world['violates'] or 'nothing'}. The bench's arm "
            f"reads the controls {', '.join(world['controls'])}.",
            "",
            *_channel_table(world),
        ]
    if data["tool"]:
        lines += [
            "",
            "## The tool's own fit",
            "",
            "mmm-framework fitted as its README's Python example fits a model "
            "(`scripts/mmm_framework_arm.py`, in the environment `scripts/mmm_framework_arm.txt` "
            "pins), as on the two shipped sets, and given the same controls as the bench's arm. "
            "Its interval is its ROI table's HDI of the contribution over the channel's spend. "
            "These are the example's settings, not those of the tool's own stress harness, which "
            "no document of its wheel states in full. Each world was fitted once; the seed does "
            "not fix the tool's numbers, which four runs on each shipped set moved by at most "
            "0.5 % of a channel's contribution.",
            "",
        ]
        settings = Counter(
            "{chains} chains of {tune} tuning and {draws} kept draws, seed {seed}".format(
                **record["settings"]
            )
            for record in data["tool"].values()
        )
        lines += [f"- {text}: {n} of {count} worlds." for text, n in settings.items()]
        warned = Counter(
            w
            for record in data["tool"].values()
            for w in record["fit_warnings"]
            if not w.startswith("ConvergenceWarning")
        )
        lines += [f"- {warning} (on {n} of {count} worlds)" for warning, n in warned.items()]
        lines += [
            f"- {environment}: {warning}"
            for environment, record in data["tool"].items()
            for warning in record["fit_warnings"]
            if warning.startswith("ConvergenceWarning")
        ]
    lines += [
        "",
        "## Arms and worlds without a cell",
        "",
        *[f"- {arm}: {why}." for arm, why in data["no_cell"].items()],
        *[f"- The world {name}: {why}." for name, why in data["generator"]["unread"].items()],
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Write the pilot, ``results/mmm_framework_pilot.{json,md}``, or with ``--markets`` the
    pilot on the generator's worlds, ``results/mmm_framework_markets.{json,md}``, with the tool's
    records read from ``results/mmm_framework_fits/``."""
    parser = argparse.ArgumentParser(description="Family 21's pilot: mmm-framework's answer keys.")
    parser.add_argument("--markets", action="store_true", help="the generator's worlds")
    parser.add_argument("--fits", type=Path, default=ROOT / "results" / "mmm_framework_fits")
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args(argv)
    if arguments.markets:
        data = pilot(arguments.fits, environments=MARKETS)
        out, text = ROOT / "results" / "mmm_framework_markets", markets_markdown(data)
    else:
        data = pilot(arguments.fits)
        out, text = ROOT / "results" / "mmm_framework_pilot", markdown(data)
    out = arguments.out or out
    out.with_suffix(".json").write_text(json.dumps(data, indent=1) + "\n")
    out.with_suffix(".md").write_text(text)


if __name__ == "__main__":
    main()

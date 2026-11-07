"""PyMC-Marketing's arm at the version its pin file names, piloted on the scorecard's version-2
exports of each family's pilot worlds at the ladder's top rung.

The arm (``scripts/pymc_marketing_arm.py``, static media, in the environment
``scripts/pymc_marketing_arm.txt`` pins) reads every lift test an export holds, whichever way its
sales moved, on the tool's default likelihood, and counts the rows it refuses, ``lift_dropped``
(``scripts/pymc_marketing_lift.py``). The budgets run's exports, version 1, held only the tests
whose sales fell, which PyMC-Marketing 1.2.0's likelihood took, so a version-1 record's
``lift_dropped`` also counts the tests that export left out. A version-2 export holds every test,
and this pilot reads those exports alone: on them ``lift_dropped`` counts the rows the arm refused
and nothing else. Beside it each environment counts the rows whose sales did not fall, the rows the
version-1 export left out and 1.2.0 refused.

The worlds, in the order the pilot fits them (:data:`CELLS`): the national model on Track M v2's
pilot, family 0, 40 drawn worlds and 20 at the reference instance, the budgets run's pilot seeds,
and on family 13, PyMC-Marketing's home ground, every world of its pilot block; then the geo model
on families 8 and 10. Every world at rung 4 (:data:`RUNG`), four go-dark tests a channel, the rung
at which the lift likelihood reads anything.

``python -m causaldyn_bench.scorecard.pymc_marketing_pilot export --work DIR`` writes the exports,
one directory an environment (:func:`directory`); the arm fits each directory into its
``pymc-marketing`` subdirectory; ``score --work DIR`` scores every record against its world
(:func:`~causaldyn_bench.scorecard.scoring.score_world`) and writes
``results/pymc_marketing_pilot.{json,md}``, each family's report
(:func:`~causaldyn_bench.scorecard.report.report`) and the lift tests' count. ``just
pymc-marketing-pilot`` runs the three. Scoring refuses a record that names other versions than the
pin file's, a smoke run, or another setting or export version. An environment is reported over the
first worlds of its seeded order it holds whole, and the worlds it did not fit are counted. What a
fit cost is left out: the records were made under whatever load the machine carried.

Precision: ``JAX_ENABLE_X64=1``, as every scored run of the scorecard.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from causaldyn_bench.scorecard.family import Family, number
from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.observe import VERSION, export, read
from causaldyn_bench.scorecard.pymc_marketing import PYMC_MARKETING
from causaldyn_bench.scorecard.report import markdown as family_markdown
from causaldyn_bench.scorecard.report import report, whole
from causaldyn_bench.scorecard.scoring import WorldRecord, score_world
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

ROOT = Path(__file__).resolve().parents[3]
PINS = ROOT / "scripts" / "pymc_marketing_arm.txt"
OUT = ROOT / "results" / "pymc_marketing_pilot"
ARM = "pymc-marketing"
SETTING = "static"
RUNG = 4  # the ladder's top rung: four go-dark tests a channel
# the packages a record names (the arm's PACKAGES), each held to the pin file's version
PACKAGES = ("pymc-marketing", "pymc", "pytensor", "arviz", "numpy", "scipy", "pandas")
RHAT = 1.01  # the largest R-hat PyMC's own convergence check passes
CELLS: tuple[tuple[Family[Any], str], ...] = (
    (TRACK_M2, "drawn"),
    (TRACK_M2, "reference"),
    (PYMC_MARKETING, "priors"),
    (GEO, "policy-0"),
    (GEO, "policy-0.5"),
    (FEW_GEOS, "geos-3"),
    (FEW_GEOS, "geos-8"),
    (FEW_GEOS, "geos-50"),
)
CUT = "\n## Compute, as recorded\n"  # where a family report's markdown turns to what it cost


def pins(path: Path = PINS) -> dict[str, str]:
    """The version the pin file names of each of :data:`PACKAGES`."""
    found = {}
    for line in path.read_text().splitlines():
        name, _, version = line.partition("==")
        if version and name.strip() in PACKAGES:
            found[name.strip()] = version.strip()
    missing = sorted(set(PACKAGES) - set(found))
    if missing:
        raise ValueError(f"{path} pins no version of {missing}")
    return {name: found[name] for name in PACKAGES}


def directory(work: Path, family: Family[Any], environment: str) -> Path:
    """Where the environment's exports are written; the arm's records go to its :data:`ARM`."""
    return work / f"family-{number(family)}" / environment / f"k{RUNG}"


def export_pilot(work: Path) -> int:
    """Every pilot world of :data:`CELLS` exported at :data:`RUNG`; how many."""
    count = 0
    for family, environment in CELLS:
        for seed in family.pilots[environment]:
            export(
                family.observe(family.world(environment, seed), RUNG),
                directory(work, family, environment),
            )
            count += 1
    return count


def _checked(record: Mapping[str, Any], pinned: Mapping[str, str], where: Path) -> None:
    """Refuses a record the pilot does not read: another version of a package, a smoke run, or
    another setting or export version."""
    named = {name: (record.get("versions") or {}).get(name) for name in pinned}
    if named != dict(pinned):
        raise ValueError(f"{where} names {named}, not the pin file's {dict(pinned)}")
    if int(record.get("version", 1)) != VERSION:
        raise ValueError(f"{where} is a version-{record.get('version', 1)} record")
    settings = record.get("settings") or {}
    if "smoke" in record or settings.get("smoke"):
        raise ValueError(f"{where} is a smoke run, not a result")
    if settings.get("setting") != SETTING:
        raise ValueError(f"{where} sets {settings.get('setting')!r}, not {SETTING!r}")


def _world(record: WorldRecord, own: Mapping[str, Any], rows: np.ndarray) -> dict[str, Any]:
    """What the pilot keeps of one world: its lift rows as the arm read them, its fit's flags and
    each arm's regret."""
    flags = own.get("flags") or {}
    return {
        "family": record.family,
        "environment": record.environment,
        "seed": record.seed,
        "index": record.index,
        "digest": own["digest"],
        "tests": int(rows.size),
        "not_fallen": int(np.count_nonzero(rows >= 0.0)),
        "lift_rows": own.get("lift_rows"),
        "lift_dropped": own.get("lift_dropped"),
        "lift_tests": own.get("lift_tests"),
        "error": own.get("error"),
        "divergences": flags.get("divergences"),
        "max_rhat": flags.get("max_rhat"),
        "optimiser_success": own.get("optimiser_success"),
        "regret": {arm: score.regret for arm, score in record.arms.items()},
    }


def _environment(
    family: Family[Any], environment: str, worlds: Sequence[dict[str, Any]], held: int
) -> dict[str, Any]:
    """One environment's count of its worlds and their lift rows."""
    fitted = [w for w in worlds if w["error"] is None]
    tests = [w["lift_tests"] for w in fitted if w["lift_tests"] is not None]
    return {
        "family": number(family),
        "name": family.name,
        "environment": environment,
        "pilots": len(family.pilots[environment]),
        "fitted": len(worlds),
        "whole": held,
        "tests": sum(w["tests"] for w in worlds),
        "not_fallen": sum(w["not_fallen"] for w in worlds),
        "taken": sum(w["lift_rows"] for w in fitted),
        "dropped": sum(w["lift_dropped"] for w in fitted),
        "tests_dropped": sum(t["dropped"] for t in tests) if tests else None,
        "raised": len(worlds) - len(fitted),
        "divergent": sum(w["divergences"] > 0 for w in fitted),
        "rhat": sum(w["max_rhat"] > RHAT for w in fitted),
        "optimiser_failed": sum(not w["optimiser_success"] for w in fitted),
    }


def pilot(work: Path, pinned: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Every record under ``work`` scored against its world, as the pilot's JSON holds it.

    Raises:
        ValueError: a record names other versions than ``pinned``, the pin file's by default, is a
            smoke run, or reads another setting or export version.
        RuntimeError: a record was fitted to another export.
    """
    pinned = pins() if pinned is None else pinned
    worlds: list[dict[str, Any]] = []
    environments: list[dict[str, Any]] = []
    reports = []
    for family in dict.fromkeys(f for f, _ in CELLS):
        records: list[WorldRecord] = []
        for environment in (e for f, e in CELLS if f is family):
            here = directory(work, family, environment)
            kept = []
            for seed in family.pilots[environment]:
                path, made = here / f"world_{seed}.npz", here / ARM / f"world_{seed}.json"
                if not (path.exists() and made.exists()):
                    continue
                own = json.loads(made.read_text())
                _checked(own, pinned, made)
                archived = read(path)
                record = score_world(
                    family, environment, seed, RUNG, {ARM: own}, pilot=True, archived=archived
                )
                records.append(record)
                kept.append(_world(record, own, np.asarray(archived.lift.delta_y, dtype=float)))
            held = whole([r for r in records if r.environment == environment])
            environments.append(_environment(family, environment, kept, held))
            worlds += kept
        if records:
            found = report(family, records)
            del found["compute"]
            reports.append(found)
    return {
        "arm": ARM,
        "setting": SETTING,
        "versions": dict(pinned),
        "rung": RUNG,
        "x64": jnp.zeros(()).dtype == jnp.float64,
        "environments": environments,
        "reports": reports,
        "worlds": worlds,
    }


def _count(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def markdown(data: Mapping[str, Any]) -> str:
    """The pilot's markdown, written from its JSON (:func:`pilot`)."""
    versions = ", ".join(f"{name} {version}" for name, version in data["versions"].items())
    precision = "float64" if data["x64"] else "float32"
    lines = [
        f"# PyMC-Marketing {data['versions']['pymc-marketing']}'s arm, pilot ({precision})",
        "",
        "The arm as `scripts/pymc_marketing_arm.py` fits it, static media and every lift test "
        "on the tool's default likelihood, in the environment `scripts/pymc_marketing_arm.txt` "
        f"pins; every record names {versions}. Each world is the scorecard's version-2 export at "
        f"rung {data['rung']}, four go-dark tests a channel, every test whichever way its sales "
        "moved.",
        "",
        "The pilot: the seeds the design was set on, apart from every scored one. No gate reads "
        "it.",
        "",
        "## The lift tests",
        "",
        "*Tests*: the tests the exports hold, one row each. *Not fallen*: those whose sales did "
        "not fall, which PyMC-Marketing 1.2.0 refused and the budgets run's version-1 export left "
        "out. *Rows taken* and *refused*: the rows the arm gave the likelihood and those it did "
        "not, its records' `lift_rows` and `lift_dropped`; the geo model takes a test as one row "
        "a tested geo, so there they outnumber the tests, and *tests refused* counts tests. "
        "*Divergent*: fits with a divergent transition; *r-hat*: fits whose largest r-hat is over "
        f"{RHAT}.",
        "",
        "| family | environment | pilot worlds | fitted | tests | not fallen | rows taken "
        "| refused | tests refused | raised | divergent | r-hat | optimiser failed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in data["environments"]:
        lines.append(
            f"| {e['family']} | {e['environment']} | {e['pilots']} | {e['fitted']} "
            f"| {e['tests']} | {e['not_fallen']} | {e['taken']} | {e['dropped']} "
            f"| {_count(e['tests_dropped'])} | {e['raised']} | {e['divergent']} | {e['rhat']} "
            f"| {e['optimiser_failed']} |"
        )
    unfitted = [e for e in data["environments"] if e["fitted"] < e["pilots"]]
    if unfitted:
        lines += [
            "",
            "Not run: "
            + "; ".join(
                f"family {e['family']}'s {e['environment']}, {e['pilots'] - e['fitted']} of its "
                f"{e['pilots']} pilot worlds"
                for e in unfitted
            )
            + ". Each environment's cells below read the first worlds of its seeded order it "
            "holds whole.",
        ]
    for found in data["reports"]:
        text = family_markdown({**found, "compute": {}}).split(CUT)[0]
        lines += ["", *(f"#{line}" if line.startswith("#") else line for line in text.splitlines())]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PyMC-Marketing's arm piloted at its pin.")
    parser.add_argument("step", choices=("export", "score"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args(argv)
    if jnp.zeros(()).dtype != jnp.float64:
        raise SystemExit("the scorecard runs in double precision: set JAX_ENABLE_X64=1")
    if arguments.step == "export":
        print(f"exported {export_pilot(arguments.work)} worlds under {arguments.work}")
        return
    data = pilot(arguments.work)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.with_suffix(".json").write_text(json.dumps(data, indent=1) + "\n")
    arguments.out.with_suffix(".md").write_text(markdown(data))
    print(f"written to {arguments.out}.md and .json")


if __name__ == "__main__":
    main()

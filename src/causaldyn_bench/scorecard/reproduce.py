"""One command reproduces a cell: family 0's worlds at rungs 0, 1 and 2, regenerated from their
stored seeds through every arm the bench runs, scored, and held to the committed cells.

``scripts/reproduce.sh`` runs :func:`main`. The committed cells (:data:`REFERENCE`) name what they
hold: the family, its pilot worlds, how many of each environment's first worlds, the rungs, every
world's score (:class:`~causaldyn_bench.scorecard.scoring.WorldRecord`) and each world's export
digest at each rung by machine class. A reproduction regenerates exactly that, and prints its cells
as a family's report does (:mod:`~causaldyn_bench.scorecard.report`).

**The arms.** In the bench's own environment, the two arms every world scores, the status quo and
the equal split, against the best plan, which the bench solves. The tools' arms, PyMC-Marketing,
Meridian, Orbit and Robyn (:data:`RIVALS`), each need an environment of their own, which the image
the bench's ``Containerfile`` builds holds under the directory ``CAUSALDYN_RIVALS`` names: there
each tool's script fits each world's export at each rung, one thread a process, and its record is
scored beside the rest. Where a tool's environment is not found, its arm is skipped and the reason
printed. A tool's records are held to the committed ones where those hold the tool, and printed
beside them otherwise.

**The tolerance.** An export's digest hashes the world's exact bits, which move in their last bits
with numpy's float64 ``exp``, ``log`` and ``tanh`` kernels: the AVX-512 kernels are other code than
the baseline and AVX2 ones, which agree (:func:`machine_class`), and with the versions of numpy,
SciPy, JAX and the library. So the committed digests are kept by machine class, each with the
versions it was recorded under: on a class they hold, under its versions, every digest must be the
committed one; elsewhere the digests are printed and not compared, and ``--record`` adds that
class's once its numbers agree.
Every number is held to the committed one by ``math.isclose`` with :data:`TOLERANCE` as both its
relative and its absolute tolerance. 1e-9 per euro is the bench's tie between two regrets
(:data:`~causaldyn_bench.budget_regret.TIE`), so a reproduction within it moves no verdict; the
best plan's worth moves with the machine class by at most 1.6e-12 of it
(``tests/test_scorecard_truth.py``). What a record says of where and how fast it was made
(:data:`VARIES`) is not compared.

Precision: ``JAX_ENABLE_X64=1``, as every scored run of the scorecard.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.scorecard.observe import export, read
from causaldyn_bench.scorecard.report import markdown, report
from causaldyn_bench.scorecard.scoring import WorldRecord, score_world
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "results" / "scorecard_reproduce.json"
RUNGS = (0, 1, 2)
WORLDS = 5  # each environment's first pilot worlds the committed cells hold
TOLERANCE = TIE
# what a record says of where and how fast it was made, which no rerun repeats
VARIES = frozenset({"cost", "fit_seconds", "plan_seconds", "seconds", "provenance"})
ONE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PJRT_NPROC": "1",
}
EXPORTING = "x86-64 without AVX-512"  # the class whose baseline and AVX2 kernels agree
SECONDS_A_WORLD = 3600  # the most a tool's script may take a world before it is stopped


@dataclass(frozen=True)
class Rival:
    """A tool's arm: ``name`` in a world's score, its environment ``directory`` under
    ``CAUSALDYN_RIVALS``, and its ``script`` and options, run from the bench's tree; ``r`` where
    it runs in R, on the R library and nevergrad's environment its directory holds."""

    name: str
    directory: str
    script: tuple[str, ...]
    r: bool = False

    def command(self, root: Path) -> tuple[list[str], dict[str, str]] | str:
        """The script's command and what its process reads beyond the environment, or why it
        cannot run here."""
        home = root / self.directory
        if not self.r:
            python = home / "bin" / "python"
            if not python.exists():
                return f"no environment at {home}"
            return [str(python), *self.script], {}
        library, nevergrad = home / "rlib", home / "ngvenv" / "bin" / "python"
        rscript = shutil.which("Rscript")
        if rscript is None or not library.is_dir() or not nevergrad.exists():
            return f"no Rscript, R library {library} and nevergrad at {nevergrad}"
        paths = {"R_LIBS": str(library), "R_LIBS_USER": str(library)}
        return [rscript, "--vanilla", *self.script], paths | {"RETICULATE_PYTHON": str(nevergrad)}


RIVALS = (
    Rival(
        "pymc-marketing", "pymc-marketing", ("scripts/pymc_marketing_arm.py", "--setting", "static")
    ),
    Rival("meridian", "meridian", ("scripts/meridian_arm.py",)),
    Rival("orbit", "orbit", ("scripts/orbit_arm.py",)),
    Rival("robyn", "robyn", ("scripts/robyn_arm.R", "--ranges", "genre"), r=True),
)


def machine_class() -> str:
    """The class of numpy's float64 ``exp``, ``expm1``, ``log``, ``log1p`` and ``tanh`` kernels
    here: :data:`EXPORTING` where each is the baseline x86 or the AVX2 kernel, which return the
    same bits, and the kernels' names otherwise."""
    try:
        from numpy.lib.introspect import opt_func_info
    except ImportError:  # numpy < 2 does not name the kernel it runs
        return f"numpy {np.__version__}"
    found = opt_func_info(func_name="^(exp|expm1|log|log1p|tanh)$", signature="float64")
    running = sorted({kernels["dd"]["current"] for kernels in found.values()})
    if all(k == "X86_V3" or k.startswith("baseline(X86") for k in running):
        return EXPORTING
    return "+".join(running)


def differences(ours: Any, theirs: Any, where: str = "") -> list[str]:
    """Where ``ours`` departs from ``theirs``, two JSON values: a key or an item missing, another
    value of another type, or a number beyond :data:`TOLERANCE`; the keys of :data:`VARIES` are
    not read."""
    if isinstance(ours, Mapping) and isinstance(theirs, Mapping):
        keys = set(ours) - VARIES, set(theirs) - VARIES
        if keys[0] != keys[1]:
            return [f"{where}: keys {sorted(keys[0] ^ keys[1])} on one side only"]
        return [
            d
            for key in sorted(keys[0])
            for d in differences(ours[key], theirs[key], f"{where}/{key}")
        ]
    if isinstance(ours, list) and isinstance(theirs, list):
        if len(ours) != len(theirs):
            return [f"{where}: {len(ours)} items, not {len(theirs)}"]
        return [
            d
            for i, (a, b) in enumerate(zip(ours, theirs, strict=True))
            for d in differences(a, b, f"{where}/{i}")
        ]
    numbers = all(isinstance(v, int | float) and not isinstance(v, bool) for v in (ours, theirs))
    if numbers and math.isclose(ours, theirs, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
        return []
    if not numbers and type(ours) is type(theirs) and ours == theirs:
        return []
    return [f"{where}: {ours!r}, not {theirs!r}"]


def _key(environment: str, seed: int, k: int) -> str:
    return f"{environment}/{seed}/k{k}"


Exports = dict[tuple[str, int], list[tuple[int, Path]]]  # (environment, k) to each world's file


def _tool_records(
    rival: Rival, run: tuple[list[str], dict[str, str]], exports: Exports, work: Path
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """``rival``'s record of every world-rung ``exports`` holds, each directory's worlds fitted by
    one run of its script, ``run`` its command and the paths its process reads."""
    command, paths = run
    records = {}
    for (environment, k), files in exports.items():
        inputs = work / rival.directory / environment / f"k{k}"
        inputs.mkdir(parents=True, exist_ok=True)
        for _, path in files:
            os.link(path, inputs / path.name)
        argv = [*command, "--worlds", str(inputs), "--out", str(inputs)]
        limit = SECONDS_A_WORLD * len(files)
        print(f"  {rival.name} on {environment} at k = {k}: {len(files)} worlds", flush=True)
        try:
            subprocess.run(argv, cwd=ROOT, env=os.environ | ONE_THREAD | paths, timeout=limit)
        except subprocess.TimeoutExpired:
            print(f"  {rival.name} stopped after {limit} s", flush=True)
        for seed, path in files:
            made = inputs / f"{path.stem}.json"
            if made.exists():
                records[(environment, seed, k)] = json.loads(made.read_text())
    return records


@dataclass(frozen=True)
class Run:
    """A regeneration: every world-rung's score, its export's digest, and the tools that ran and
    those skipped, with why."""

    records: tuple[WorldRecord, ...]
    digests: dict[str, str]
    ran: tuple[str, ...]
    skipped: dict[str, str]


def regenerate(
    worlds: Mapping[str, int],
    rungs: Sequence[int] = RUNGS,
    *,
    rivals: Sequence[Rival] = RIVALS,
    root: str | None = None,
) -> Run:
    """Family 0's first ``worlds[e]`` pilot worlds of each environment ``e`` at each of
    ``rungs``, scored through the bench's own arms and every tool of ``rivals`` whose
    environment ``root`` holds, ``CAUSALDYN_RIVALS`` by default."""
    root = os.environ.get("CAUSALDYN_RIVALS") if root is None else root
    skipped: dict[str, str] = {}
    usable = []
    for rival in rivals:
        found = "CAUSALDYN_RIVALS is not set" if root is None else rival.command(Path(root))
        if isinstance(found, str):
            skipped[rival.name] = (
                f"{found}; the tools' arms run in the image the Containerfile builds"
            )
        else:
            usable.append((rival, found))
    with tempfile.TemporaryDirectory(prefix="reproduce-") as scratch:
        work = Path(scratch)
        exports: Exports = {}
        digests = {}
        for environment, n in worlds.items():
            for seed in TRACK_M2.pilots[environment][:n]:
                world = TRACK_M2.world(environment, seed)
                for k in rungs:
                    observation = TRACK_M2.observe(world, k)
                    digests[_key(environment, seed, k)] = observation.digest()
                    path = export(observation, work / "exports" / environment / f"k{k}")
                    exports.setdefault((environment, k), []).append((seed, path))
        made: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
        for rival, command in usable:
            for at, written in _tool_records(rival, command, exports, work).items():
                made.setdefault(at, {})[rival.name] = written
        records = tuple(
            score_world(
                TRACK_M2,
                environment,
                seed,
                k,
                made.get((environment, seed, k), {}),
                pilot=True,
                archived=read(path),
            )
            for (environment, k), files in exports.items()
            for seed, path in files
        )
    return Run(records, digests, tuple(rival.name for rival, _ in usable), skipped)


def _arms(record: Mapping[str, Any], arms: set[str]) -> dict[str, Any]:
    return {**record, "arms": {a: s for a, s in record["arms"].items() if a in arms}}


def compare(run: Run, reference: Mapping[str, Any], machine: str) -> dict[str, Any]:
    """``run`` against the committed ``reference`` on the machine class ``machine``: the digests,
    where the class is committed, and every number of every arm both hold in every world-rung;
    the arms one of them holds alone are named."""
    theirs = {_key(r["environment"], r["seed"], r["k"]): r for r in reference["records"]}
    ours = {_key(r.environment, r.seed, r.k): r.as_json() for r in run.records}
    found = []
    if set(ours) != set(theirs):
        found.append(f"world-rungs {sorted(set(ours) ^ set(theirs))} on one side only")
    alone: dict[str, set[str]] = {"run": set(), "committed": set()}
    for key in sorted(set(ours) & set(theirs)):
        mine, committed = set(ours[key]["arms"]), set(theirs[key]["arms"])
        alone["run"] |= mine - committed
        alone["committed"] |= committed - mine
        both = mine & committed
        found += [
            f"{key}{d}" for d in differences(_arms(ours[key], both), _arms(theirs[key], both))
        ]
    committed = reference["digests"].get(machine)
    missed, why = None, f"none is committed for {machine}"
    if committed is not None and committed["under"] != versions():
        why = f"{machine}'s were committed under {committed['under']}"
    elif committed is not None:
        exports = committed["exports"]
        missed = [key for key, value in run.digests.items() if exports.get(key) != value]
    return {
        "numbers": found,
        "digests": missed,
        "why": why,
        "uncommitted": sorted(alone["run"]),
        "not run": sorted(alone["committed"]),
    }


def versions() -> dict[str, str]:
    """The versions an export's bits move with, beside the machine class."""
    from importlib.metadata import version

    names = ("numpy", "scipy", "jax", "jaxlib", "causal-hybrid-control")
    return {name: version(name) for name in names}


def record(
    run: Run, worlds: Mapping[str, int], rungs: Sequence[int], previous: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The committed cells: ``run``'s where there are none yet, ``previous``'s otherwise, with this
    machine class's digests added once ``run``'s numbers are ``previous``'s within the tolerance.
    The numbers stay the first class's, so a class that adds its digests moves none of them.

    Raises:
        RuntimeError: the run's numbers or this class's digests are not the previous reference's.
    """
    machine = machine_class()
    held = {machine: {"under": versions(), "exports": dict(sorted(run.digests.items()))}}
    if previous is not None:
        found = compare(run, previous, machine)
        if found["numbers"] or found["digests"]:
            raise RuntimeError(f"the run is not the committed one: {found}")
        return {**previous, "digests": {**previous["digests"], **held}}
    return {
        "family": 0,
        "pilot": True,
        "worlds": dict(worlds),
        "rungs": list(rungs),
        "tolerance": TOLERANCE,
        "digests": held,
        "records": [r.as_json() for r in run.records],
        "report": report(TRACK_M2, run.records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce family 0's committed cells.")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path, help="where to write the regenerated cells' report")
    parser.add_argument("--record", action="store_true", help="write the reference from this run")
    args = parser.parse_args()
    if jnp.zeros(()).dtype != jnp.float64:
        raise SystemExit("the scorecard runs in double precision: set JAX_ENABLE_X64=1")
    previous = json.loads(args.reference.read_text()) if args.reference.exists() else None
    if previous is None and not args.record:
        raise SystemExit(f"no committed cells at {args.reference}")
    if previous is None:
        worlds, rungs = dict.fromkeys(TRACK_M2.pilots, WORLDS), RUNGS
    else:
        worlds, rungs = dict(previous["worlds"]), tuple(previous["rungs"])
    machine = machine_class()
    print(f"family 0, {TRACK_M2.name}: pilot worlds {worlds}, rungs {list(rungs)}; {machine}")
    run = regenerate(worlds, rungs)
    print(f"arms run here: status quo, equal split{''.join(', ' + n for n in run.ran)}")
    for name, why in run.skipped.items():
        print(f"skipped {name}: {why}")
    data = report(TRACK_M2, run.records)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "reproduce.md").write_text(markdown(data))
        (args.out / "reproduce.json").write_text(json.dumps(data, allow_nan=False) + "\n")
    print(markdown(data))
    if args.record:
        written = record(run, worlds, rungs, previous)
        args.reference.write_text(json.dumps(written, allow_nan=False, indent=1) + "\n")
        print(f"recorded {args.reference} on {machine}")
        return
    assert previous is not None
    found = compare(run, previous, machine)
    if found["digests"] is None:
        print(f"digests: {found['why']}, so they are printed, not compared")
        for key, value in sorted(run.digests.items()):
            print(f"  {key} {value}")
    else:
        same = len(run.digests) - len(found["digests"])
        print(f"digests: {same} of {len(run.digests)} are the committed ones")
    print(f"numbers: {len(found['numbers'])} beyond {TOLERANCE:g} of the committed ones")
    for line in found["numbers"][:20]:
        print(f"  {line}")
    for name in found["uncommitted"]:
        print(f"{name}: the committed cells hold no record of it; printed, not compared")
    for name in found["not run"]:
        print(f"{name}: committed, not run here; not compared")
    if found["numbers"] or found["digests"]:
        sys.exit("not reproduced")
    print("reproduced")


if __name__ == "__main__":
    main()

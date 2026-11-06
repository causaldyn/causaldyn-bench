"""One command reproduces a cell: the committed cells regenerate from their seeds within the stated
tolerance, their digests on the machine class and versions they were recorded under, a number or a
digest off them is caught, a tool's arm runs where its environment is found and is skipped with its
reason where not."""

import copy
import json
import os
import sys

import jax
import pytest

from causaldyn_bench.scorecard.reproduce import (
    EXPORTING,
    REFERENCE,
    RIVALS,
    TOLERANCE,
    Rival,
    compare,
    differences,
    machine_class,
    record,
    regenerate,
    versions,
)

STUB = """
import argparse, json
from pathlib import Path
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--worlds", type=Path)
parser.add_argument("--out", type=Path)
args = parser.parse_args()
for path in sorted(args.worlds.glob("world_*.npz")):
    with np.load(path) as saved:
        record = {
            "version": int(saved["version"]),
            "seed": int(saved["seed"]),
            "digest": str(saved["digest"]),
            "error": None,
            "weekly": saved["status_quo"].tolist(),
            "cost": {"wall_seconds": 1.0, "threads": "1"},
        }
    (args.out / f"{path.stem}.json").write_text(json.dumps(record))
"""


@pytest.fixture(scope="module")
def committed():
    return json.loads(REFERENCE.read_text())


@pytest.fixture(scope="module")
def run(committed, tmp_path_factory):
    with jax.enable_x64(True):
        return regenerate(
            committed["worlds"], committed["rungs"], root=str(tmp_path_factory.mktemp("none"))
        )


def test_the_committed_cells_reproduce_within_the_stated_tolerance(committed, run):
    assert committed["tolerance"] == TOLERANCE and committed["rungs"] == [0, 1, 2]
    found = compare(run, committed, machine_class())
    assert found["numbers"] == []
    held = committed["digests"].get(machine_class())
    if held is not None and held["under"] == versions():
        assert found["digests"] == []
    else:
        assert found["digests"] is None
    assert len(run.records) == len(committed["records"]) == 3 * sum(committed["worlds"].values())
    assert run.ran == () and set(run.skipped) == {r.name for r in RIVALS}
    assert all(why.startswith("no ") for why in run.skipped.values())


def test_a_number_or_a_digest_off_the_committed_ones_is_caught(committed, run):
    moved = copy.deepcopy(committed)
    moved["records"][0]["best"] *= 1 + 10 * TOLERANCE
    moved["records"][1]["arms"]["status quo"]["regret"] += 10 * TOLERANCE
    found = compare(run, moved, machine_class())
    assert [line.split(":")[0] for line in found["numbers"]] == [
        f"{r['environment']}/{r['seed']}/k{r['k']}{at}"
        for r, at in zip(moved["records"][:2], ("/best", "/arms/status quo/regret"), strict=True)
    ]
    other = copy.deepcopy(committed)
    other["digests"][machine_class()] = {
        "under": versions(),
        "exports": dict.fromkeys(run.digests, "0"),
    }
    assert sorted(compare(run, other, machine_class())["digests"]) == sorted(run.digests)
    other["digests"][machine_class()]["under"] = {"numpy": "0"}
    assert compare(run, other, machine_class())["digests"] is None


def test_differences_hold_numbers_to_the_tolerance_and_skip_where_a_record_was_made():
    ours = {"regret": 0.1, "best": 1e6, "cost": {"wall_seconds": 1.0}, "error": None}
    same = {"regret": 0.1 + TOLERANCE / 2, "best": 1e6 * (1 + TOLERANCE / 2), "error": None}
    assert differences(ours, same) == []
    assert differences(ours, same | {"best": 1e6 * (1 + 2 * TOLERANCE)}) == [
        f"/best: 1000000.0, not {1e6 * (1 + 2 * TOLERANCE)!r}"
    ]
    assert differences(ours, same | {"error": "x"}) == ["/error: None, not 'x'"]
    assert differences({"a": [1, 2]}, {"a": [1]}) == ["/a: 2 items, not 1"]
    assert differences({"a": 1}, {"b": 1}) == [": keys ['a', 'b'] on one side only"]
    assert differences({"a": True}, {"a": 1}) == ["/a: True, not 1"]


def test_the_machine_class_is_the_exporting_one_where_the_suites_kernels_are(other_kernels):
    assert (machine_class() == EXPORTING) == (other_kernels is None)


def test_a_tool_runs_where_its_environment_is_found_and_is_scored_beside_the_bench_arms(tmp_path):
    home = tmp_path / "stub" / "bin"
    home.mkdir(parents=True)
    (home / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    os.chmod(home / "python", 0o755)
    script = tmp_path / "stub_arm.py"
    script.write_text(STUB)
    stub = Rival("stub", "stub", (str(script),))
    with jax.enable_x64(True):
        made = regenerate({"drawn": 2}, (0, 1), rivals=(stub, RIVALS[0]), root=str(tmp_path))
    assert made.ran == ("stub",) and set(made.skipped) == {RIVALS[0].name}
    assert len(made.records) == 4
    for world in made.records:
        assert world.arms["stub"].regret == world.arms["status quo"].regret
        assert world.arms["stub"].record["cost"] == {"wall_seconds": 1.0, "threads": "1"}
    without = [r.as_json() for r in made.records]
    for r in without:
        del r["arms"]["stub"]
    found = compare(made, {"records": without, "digests": {}}, machine_class())
    assert (found["numbers"], found["uncommitted"], found["not run"]) == ([], ["stub"], [])


def test_a_class_adds_its_digests_and_moves_no_committed_number(committed, run):
    other = copy.deepcopy(committed)
    other["digests"] = {"another class": committed["digests"][next(iter(committed["digests"]))]}
    added = record(run, committed["worlds"], committed["rungs"], other)
    assert added["records"] == committed["records"] and added["report"] == committed["report"]
    assert added["digests"] == other["digests"] | {
        machine_class(): {"under": versions(), "exports": dict(sorted(run.digests.items()))}
    }
    moved = copy.deepcopy(other)
    moved["records"][0]["best"] *= 1 + 10 * TOLERANCE
    with pytest.raises(RuntimeError, match="not the committed one"):
        record(run, committed["worlds"], committed["rungs"], moved)
    fresh = record(run, committed["worlds"], committed["rungs"], None)
    assert compare(run, fresh, machine_class())["numbers"] == []

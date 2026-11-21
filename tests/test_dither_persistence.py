"""The dither persistence study: the settings, the lean against its exact value, the report, and a
small run on a seed the study does not read."""

import json

import pytest

from causaldyn_bench.dither_persistence import (
    BASE,
    PERSISTENCE,
    SEED,
    SETTINGS,
    cell,
    lean,
    markdown,
    replicate,
    world_of,
)
from causaldyn_bench.graph_errors import WORLDS
from causaldyn_bench.graph_errors_v3 import SEED as V3_SEED
from causaldyn_bench.graph_errors_v3 import has_dither


def test_each_size_setting_sets_the_persistence_and_nothing_else():
    base = WORLDS[BASE]
    assert base.confounder is not None
    for persistence in PERSISTENCE:
        world = world_of(SETTINGS[f"size {persistence:g}"])
        assert world.confounder is not None
        assert world.confounder.persistence == persistence
        assert (world.confounder.read, world.confounder.push) == (
            base.confounder.read,
            base.confounder.push,
        )
        assert (world.true_edges, world.wrong_edges) == (base.true_edges, base.wrong_edges)
    assert world_of(SETTINGS["power mediator"]) is WORLDS["mediator"]


def test_the_seed_is_new():
    assert SEED != V3_SEED


# scripts/dither_persistence_lean.mac, STEP 2, at eight digits
@pytest.mark.parametrize(
    ("persistence", "read", "expected"),
    [
        (0.5, 1.0, (0.58689516, 0.15906993)),
        (0.8, 1.0, (0.95575184, 0.26096138)),
        (0.95, 1.0, (1.1531368, 0.19152301)),
        (0.8, 0.0, (1.1343265, 0.37186365)),
    ],
)
def test_the_lean_is_the_exact_one(persistence, read, expected):
    supply, wait = lean(persistence, read=read)
    assert supply == pytest.approx(expected[0], rel=1e-7, abs=0.0)
    assert wait == pytest.approx(expected[1], rel=1e-7, abs=0.0)


def test_no_lean_where_the_promotion_is_fresh_or_pushes_nothing():
    assert lean(0.0) == (0.0, 0.0)
    assert lean(0.8, push=0.0) == (0.0, 0.0)


def _rows(p_values, void=0):
    rows = [{"status": "tested", "p_value": p, "cost": 0.0, "noise": 1e-5} for p in p_values]
    return rows + [{"status": "void", "p_value": None, "cost": None, "noise": None}] * void


def test_a_cell_counts_the_tested_panels_alone():
    measured = cell(SETTINGS["size 0.8"], _rows([0.01, 0.05, 0.2, 0.9], void=1))
    assert (measured.tested, measured.void, measured.rejected) == (4, 1, 2)
    assert measured.rate == 0.5
    assert (measured.rate_10, measured.rate_20) == (0.5, 0.75)
    assert measured.mc_error == pytest.approx(0.25, rel=1e-12, abs=0.0)
    assert measured.lean == lean(0.8)


def _record(label, commit, cells):
    rows = {name: _rows(p) for name, p in cells}
    return json.loads(
        json.dumps(
            {
                "label": label,
                "chc_version": "1.3.1",
                "chc_commit": commit,
                "x64": True,
                "rows": rows,
            }
        )
    )


def test_the_report_merges_a_label_s_records_and_refuses_two_libraries():
    records = [
        _record("fixed", "aaaaaaa", [("size 0.8", [0.5, 0.01])]),
        _record("fixed", "aaaaaaa", [("power mediator", [0.001])]),
        _record("base", "bbbbbbb", [("size 0.8", [0.01, 0.01])]),
    ]
    text = markdown(records)
    assert "**fixed** on causal-hybrid-control 1.3.1 at `aaaaaaa`" in text
    assert "| size 0.8 | confounder AR | 0.8 | (0.956, 0.261) | 0.500" in text
    assert "| 1.000 ± 0.000" in text
    assert "| size 0.8 | base | 1.000 | 1.000 | 0.010 |" in text
    with pytest.raises(ValueError, match="more than one library"):
        markdown([*records, _record("fixed", "ccccccc", [("size 0", [0.5])])])


@pytest.mark.skipif(not has_dither(), reason="needs a prescribe that reads dither=")
def test_a_replicate_runs_end_to_end_on_a_seed_the_study_does_not_read():
    row = replicate(("size 0.8", 0), seed=7)
    assert row["status"] == "tested"
    assert 0.0 <= row["p_value"] <= 1.0
    assert row["noise"] > 0.0

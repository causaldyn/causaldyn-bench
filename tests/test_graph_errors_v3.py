"""Track Q, version 3: the logged draw, the stop rule, and a small run on a seed no scored run
reads.

The run needs a causal-hybrid-control whose ``prescribe`` takes ``dither=``; on one without, it is
skipped and the rest still runs.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from causaldyn_bench.graph_errors import HORIZON, WORLDS, draw
from causaldyn_bench.graph_errors_v3 import (
    ALPHA,
    COST_SHARE,
    DITHER,
    SEED,
    _markdown,
    _record,
    has_dither,
    rejected,
    run_world,
    stop,
)


@pytest.mark.parametrize("name", list(WORLDS))
def test_logging_the_draw_moves_no_other_column(name):
    world = WORLDS[name]
    plain = draw(world, (7, 0, 0, 0), units=20)
    logged = draw(world, (7, 0, 0, 0), units=20, dither=DITHER)
    assert set(logged.columns) == {*plain.columns, DITHER}
    for column, values in plain.columns.items():
        np.testing.assert_array_equal(logged.columns[column], values)
    assert logged.confounder_last == plain.confounder_last
    assert np.std(logged.columns[DITHER]) > 0.5


def _check(status="tested", p=0.01, cost=0.02, gain=0.1):
    return {"status": status, "p_value": p, "cost": cost, "claimed_gain": gain}


def test_the_stop_needs_a_rejection_and_a_material_cost():
    assert stop(None) is None
    assert rejected(None) is None
    assert stop(_check()) is True
    assert stop(_check(p=ALPHA)) is True
    assert stop(_check(p=0.06)) is False
    assert rejected(_check(p=0.06)) is False
    # at the margin the cost is not above it
    assert stop(_check(cost=COST_SHARE * 0.1)) is False
    assert rejected(_check(cost=0.0)) is True
    void = {"status": "void", "p_value": None, "cost": None, "claimed_gain": 0.1}
    assert stop(void) is False
    assert rejected(void) is False


def test_the_seed_is_new():
    assert SEED == 20261026


@pytest.mark.skipif(not has_dither(), reason="this causal-hybrid-control has no prescribe(dither=)")
def test_two_replicates_read_the_draw_and_write_the_report():
    """On a seed no scored run reads: the true and the wrong graph read their check, the arm
    chosen by fit reads none, and the stop is the check's."""
    with ThreadPoolExecutor(1) as pool:
        run = run_world("mediator", 2, pool, seed=7)
    for arm, arm_runs in run.runs.items():
        for arm_run in arm_runs:
            assert len(arm_run["exceedance"]) == HORIZON
            if arm == "by fit":
                assert arm_run["dither"] is None
                assert arm_run["fit_flag"] is None
                assert DITHER not in arm_run["covariates"]
            else:
                check = arm_run["dither"]
                assert check["status"] in ("tested", "void")
                assert arm_run["fit_flag"] == stop(check)
            assert arm_run["later_flag"] is None
    text = _markdown([run], 2, 7, True, None, None)
    assert "## The gate" in text
    assert "## mediator" in text
    record = json.loads(json.dumps(_record([run], 2, 7, True, None, None)))
    assert record["rule"]["dither"] == DITHER
    assert len(record["worlds"][0]["replicates"]["wrong"]["dither"]) == 2

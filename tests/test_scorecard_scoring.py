"""The scorecard's scoring: the rules a record is scored by, the secondary axes it may carry, family
0 through the scorecard against every committed Track M v2 world, and the looks a run is read at.
"""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from causaldyn_bench.budget_regret import ENVIRONMENTS, PILOTS, TIE, Comparison
from causaldyn_bench.external_arms import on_bounds
from causaldyn_bench.lift_calibration import STARTS
from causaldyn_bench.mmm_decision import PLANNED, Plan
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.observe import FIRST, VERSION, read
from causaldyn_bench.scorecard.observe import export as archive
from causaldyn_bench.scorecard.scoring import (
    BUILT_IN,
    ArmScore,
    WorldRecord,
    crps,
    look,
    paired,
    score_arm,
    score_world,
)
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from causaldyn_bench.scorecard.truth import worth

ROOT = Path(__file__).resolve().parents[1]
TOP = len(STARTS)


@pytest.fixture(scope="module")
def world():
    return TRACK_M2.world("drawn", 905)


@pytest.fixture(scope="module")
def truth(world):
    return TRACK_M2.truth(world)


@pytest.fixture(scope="module")
def observed(world):
    return TRACK_M2.observe(world, TOP)


def _plan(observed, weekly, **extra):
    return {"digest": observed.digest(), "version": VERSION, "error": None, **extra} | {
        "weekly": list(map(float, weekly))
    }


def _status_quo_regret(truth):
    best, quarter = truth.best, truth.quarter
    return (best.worth - worth(truth.cells, quarter.status_quo)) / quarter.budget


def test_a_plan_inside_the_box_is_scored_on_the_worlds_own_channels(truth, observed):
    quarter = truth.quarter
    response = {"unmapped": "a time-varying multiplier"}
    record = _plan(observed, quarter.status_quo, divergences=0, response=response)
    arm = score_arm(truth, observed, record)
    assert arm.regret == pytest.approx(_status_quo_regret(truth), abs=1e-15)
    assert (arm.failure, arm.moved, arm.realised, arm.uplift, arm.crps, arm.roi) == (None,) * 6
    assert arm.unread == "unmapped: a time-varying multiplier"
    assert arm.record == {"error": None, "divergences": 0}
    assert arm.bounds == on_bounds(quarter, quarter.status_quo)


def test_a_record_is_scored_on_the_export_its_digest_names_version_one_where_it_names_none(
    truth, observed
):
    weekly = truth.quarter.status_quo
    legacy = {"digest": observed.digest(FIRST), "error": None, "weekly": list(weekly)}
    assert score_arm(truth, observed, legacy).regret == pytest.approx(_status_quo_regret(truth))
    for record in (
        {**legacy, "version": VERSION},
        _plan(observed, weekly) | {"version": FIRST},
        _plan(observed, weekly) | {"digest": "0" * 16},
    ):
        with pytest.raises(RuntimeError, match="fitted to other data"):
            score_arm(truth, observed, record)


def test_an_arm_that_failed_plays_the_status_quo_and_claims_nothing(truth, observed):
    record = _plan(observed, [], gain=5.0, forecast=[list(truth.target)]) | {
        "error": "RuntimeError: no start converged",
        "traceback": "...",
    }
    arm = score_arm(truth, observed, record)
    assert arm.failure == "RuntimeError: no start converged"
    assert arm.regret == pytest.approx(_status_quo_regret(truth))
    assert (arm.uplift, arm.covered, arm.crps) == (None, None, None)
    assert "traceback" not in arm.record and "forecast" not in arm.record


def test_a_plan_off_the_budget_is_moved_into_the_box_and_the_move_recorded(truth, observed):
    quarter = truth.quarter
    over = quarter.status_quo + 1.0 / PLANNED
    arm = score_arm(truth, observed, _plan(observed, over))
    inside = quarter.project(over)
    assert arm.moved == pytest.approx(float(np.max(np.abs(inside - over))))
    expected = (truth.best.worth - worth(truth.cells, inside)) / quarter.budget
    assert arm.regret == pytest.approx(expected)
    assert arm.bounds == on_bounds(quarter, inside)


def test_the_best_plan_scores_nought_and_a_plan_that_beats_it_is_refused(truth, observed):
    arm = score_arm(truth, observed, _plan(observed, truth.best.weekly))
    assert abs(arm.regret) <= TIE
    lower = Plan(truth.best.weekly, truth.best.worth - 1.0, truth.best.price)
    broken = dataclasses.replace(truth, best=lower)
    with pytest.raises(RuntimeError, match="beat the best plan"):
        score_arm(broken, observed, _plan(observed, truth.best.weekly))


def test_a_claimed_gain_is_read_against_the_gain_the_plan_realises(truth, observed):
    quarter, weekly = truth.quarter, truth.best.weekly
    gain = worth(truth.cells, weekly) - worth(truth.cells, quarter.status_quo)
    claimed = gain + 0.01 * quarter.budget
    arm = score_arm(truth, observed, _plan(observed, weekly, gain=claimed))
    assert arm.uplift == pytest.approx(0.01)
    assert arm.covered is None
    held = score_arm(
        truth, observed, _plan(observed, weekly, gain=claimed, gain_interval=[0, claimed])
    )
    assert held.covered is True
    above = [gain + 1.0, gain + 2.0]
    assert (
        score_arm(truth, observed, _plan(observed, weekly, gain=0, gain_interval=above)).covered
        is False
    )
    with pytest.raises(ValueError, match="low to high"):
        score_arm(truth, observed, _plan(observed, weekly, gain=0, gain_interval=[2.0, 1.0]))


def test_the_crps_of_one_draw_is_its_error_and_of_many_the_normals_closed_form():
    target = np.array([1.0, -2.0])
    assert crps(np.array([[1.5, -2.0]]), target) == pytest.approx(0.25)
    two = np.array([[0.0, 0.0], [2.0, 1.0]])
    expected = np.mean([(1 + 1) / 2 - 2 / 4, (2 + 3) / 2 - 1 / 4])
    assert crps(two, target) == pytest.approx(expected)
    y, z = 0.3, np.random.default_rng(0).standard_normal((200_000, 1))
    closed = y * (2 * norm.cdf(y) - 1) + 2 * norm.pdf(y) - 1 / math.sqrt(math.pi)
    assert crps(z, np.array([y])) == pytest.approx(closed, abs=3e-3)
    with pytest.raises(ValueError, match="draws of 2 weeks"):
        crps(np.zeros((3, 5)), target)


def test_a_forecast_is_scored_by_its_crps_in_units_of_mean_weekly_sales(truth, observed):
    weekly = truth.quarter.status_quo
    exact = score_arm(truth, observed, _plan(observed, weekly, forecast=[list(truth.target)]))
    assert exact.crps == 0.0
    off = [list(truth.target + 0.1 * truth.scale)]
    assert score_arm(truth, observed, _plan(observed, weekly, forecast=off)).crps == pytest.approx(
        0.1
    )


def test_where_the_quarter_realises_another_path_the_plan_is_scored_on_it_too(world, observed):
    realised = world.expected * np.linspace(0.8, 1.2, world.expected.shape[1])
    truth = TRACK_M2.truth(dataclasses.replace(world, realised=realised))
    assert truth.realised is not None and truth.hindsight is not None
    quarter = truth.quarter
    hindsight = score_arm(truth, observed, _plan(observed, truth.hindsight.weekly)).realised
    best = score_arm(truth, observed, _plan(observed, truth.best.weekly))
    assert hindsight is not None and best.realised is not None
    assert abs(hindsight) <= TIE
    # the box fixes this split, two channels at an end of it and the third spending the rest, so
    # the best plan is the hindsight plan and its realised regret is 0 up to rounding: -4.3e-16
    # where numpy's exp is its AVX-512 kernel
    assert best.realised >= -TIE and abs(best.regret) <= TIE
    status_quo = score_arm(truth, observed, _plan(observed, quarter.status_quo, gain=0.0))
    expected = (truth.hindsight.worth - worth(truth.realised, quarter.status_quo)) / quarter.budget
    assert status_quo.realised == pytest.approx(expected)
    assert status_quo.uplift == 0.0  # the status quo gains nothing over itself, on either path
    claim = score_arm(truth, observed, _plan(observed, truth.best.weekly, gain=0.0)).uplift
    gain = worth(truth.realised, truth.best.weekly) - worth(truth.realised, quarter.status_quo)
    assert claim == pytest.approx(-gain / quarter.budget)
    assert TRACK_M2.truth(world).realised is None


def test_a_world_scores_its_records_beside_the_built_in_arms(world, truth, observed):
    records = {"x": _plan(observed, truth.best.weekly)}
    scored = score_world(TRACK_M2, "drawn", 905, TOP, records, pilot=True)
    assert set(scored.arms) == {*BUILT_IN, "x"}
    assert (scored.family, scored.environment, scored.seed, scored.index) == (0, "drawn", 905, 5)
    assert (scored.pilot, scored.k, scored.channels) == (True, TOP, 3)
    assert scored.best == truth.best.worth
    assert scored.covariates == {
        "status quo": scored.arms["status quo"].regret,
        "equal split": scored.arms["equal split"].regret,
        "return": truth.best.worth / truth.quarter.budget,
        "price": truth.best.price,
    }
    assert scored.oracle_bounds == on_bounds(truth.quarter, truth.best.weekly)


def test_a_world_refuses_a_world_it_does_not_draw_a_built_in_name_and_another_worlds_export(
    world, observed, tmp_path
):
    with pytest.raises(ValueError, match="draws no scored world 905"):
        score_world(TRACK_M2, "drawn", 905, TOP, {})
    with pytest.raises(ValueError, match="scored on every world"):
        score_world(TRACK_M2, "drawn", 905, TOP, {"status quo": {}}, pilot=True)
    other = TRACK_M2.observe(TRACK_M2.world("drawn", 906), TOP)
    with pytest.raises(RuntimeError, match="world 906"):
        score_world(TRACK_M2, "drawn", 905, TOP, {}, pilot=True, archived=other)
    with pytest.raises(RuntimeError, match="world 905 at 0"):
        score_world(
            TRACK_M2, "drawn", 905, TOP, {}, pilot=True, archived=TRACK_M2.observe(world, 0)
        )
    with pytest.raises(RuntimeError, match="fitted to other data"):
        score_world(TRACK_M2, "drawn", 905, 0, {"x": _plan(observed, [1, 1, 1])}, pilot=True)
    kept = read(archive(observed, tmp_path))
    record = {"x": _plan(observed, observed.status_quo)}
    fresh = score_world(TRACK_M2, "drawn", 905, TOP, record, pilot=True)
    assert score_world(TRACK_M2, "drawn", 905, TOP, record, pilot=True, archived=kept) == fresh


def test_a_world_record_survives_strict_json(observed):
    record = {"x": _plan(observed, observed.status_quo, gain=1.0, gain_interval=[0.0, 2.0])}
    scored = score_world(TRACK_M2, "drawn", 905, TOP, record, pilot=True)
    scored = dataclasses.replace(scored, covariates={**scored.covariates, "price": math.nan})
    text = json.dumps(scored.as_json(), allow_nan=False)
    back = WorldRecord.from_json(json.loads(text))
    assert math.isnan(back.covariates["price"])
    assert dataclasses.replace(back, covariates={}) == dataclasses.replace(scored, covariates={})


def _committed(name):
    data = json.loads((ROOT / "results" / f"{name}.json").read_text())
    return data["pilot"], [
        (e["environment"]["name"], world) for e in data["environments"] for world in e["worlds"]
    ]


@pytest.fixture(scope="module")
def through():
    """Every Track M v2 world, pilot or scored, through the scorecard at the top rung."""
    out = {}
    for pilot, environments in ((True, PILOTS), (False, ENVIRONMENTS)):
        for environment in environments:
            for seed in environment.seeds:
                out[pilot, environment.name, seed] = score_world(
                    TRACK_M2, environment.name, seed, TOP, {}, pilot=pilot
                )
    return out


@pytest.mark.parametrize("name", ["track_m2_budgets", "track_m2_budgets_pilot"])
def test_family_nought_reproduces_the_budgets_runs_oracle_status_quo_and_equal_split(
    through, name, other_kernels
):
    pilot, worlds = _committed(name)
    for environment, committed in worlds:
        scored = through[pilot, environment, committed["seed"]]
        if other_kernels is None:
            assert scored.budget == committed["budget"]
        else:  # the budget is a drawn world's, and these kernels move its last bit
            assert scored.budget == pytest.approx(committed["budget"], rel=1e-12, abs=0.0)
        assert abs(scored.best - committed["best"]) <= TIE * scored.budget
        for arm in BUILT_IN:
            assert abs(scored.arms[arm].regret - committed["regret"][arm]) <= TIE


@pytest.mark.parametrize(
    "name", ["track_m2_external", "track_m2_external_pilot_genre", "track_m2_external_pilot_union"]
)
def test_family_nought_reproduces_the_external_runs_oracle_and_its_bounds(through, name):
    pilot, worlds = _committed(name)
    for environment, committed in worlds:
        scored = through[pilot, environment, committed["seed"]]
        assert abs(scored.best - committed["best"]) <= TIE * scored.budget
        assert abs(scored.arms["status quo"].regret - committed["committed"]["status quo"]) <= TIE
        assert (scored.channels, scored.oracle_bounds) == (
            committed["channels"],
            committed["oracle_bounds"],
        )


def test_family_noughts_worlds_are_the_budgets_runs():
    assert number(TRACK_M2) == 0
    assert dict(TRACK_M2.scored) == {e.name: e.seeds for e in ENVIRONMENTS}
    assert dict(TRACK_M2.pilots) == {e.name: e.seeds for e in PILOTS}
    assert all(label.endswith(".") for label in TRACK_M2.labels)


def _record(environment, i, regrets, *, k=TOP, pilot=False, missing=()):
    arms = {
        arm: ArmScore(regret, None, None, 0, None, None, None, None, {})
        for arm, regret in regrets.items()
        if arm not in missing
    }
    return WorldRecord(0, environment, 10 * i, i, pilot, k, 1.0, 1.0, 3, 0, {}, arms)


def _run(worlds=12):
    rng = np.random.default_rng(3)
    records = []
    for environment in ("drawn", "reference"):
        for i in range(worlds):
            other = float(rng.uniform(0.05, 0.15))
            records.append(_record(environment, i, {"a": other - 0.02, "b": other}))
    return records


def test_a_look_reads_the_first_worlds_of_each_environments_order_whatever_the_records_order():
    records = _run()
    seen = look(records[::-1], 5)
    assert sorted((r.environment, r.index) for r in seen) == [
        (e, i) for e in ("drawn", "reference") for i in range(5)
    ]
    first = [r for r in records if r.index < 5]
    expected = Comparison.of(
        "b",
        np.array([r.arms["a"].regret for r in first]),
        np.array([r.arms["b"].regret for r in first]),
    )
    shuffled = [records[i] for i in np.random.default_rng(0).permutation(len(records))]
    assert paired(shuffled, "a", "b", 5) == expected
    assert paired(records, "a", "b", 5).difference.mean == pytest.approx(-0.02)


def test_every_prefix_of_a_run_is_a_look_and_the_whole_run_is_the_last():
    records = _run()
    sizes = [paired(records, "a", "b", n).lower for n in range(2, 13)]
    assert sizes == list(range(4, 25, 2))
    with pytest.raises(ValueError, match="misses drawn worlds \\[12\\]"):
        look(records, 13)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda r: r[1:], "misses drawn worlds \\[0\\]"),
        (lambda r: [*r, r[0]], "held twice"),
        (lambda r: [*r[:-1], dataclasses.replace(r[-1], k=0)], "one family at one rung"),
        (lambda r: [*r[:-1], dataclasses.replace(r[-1], pilot=True)], "one family at one rung"),
    ],
)
def test_a_malformed_look_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        look(change(_run()), 12)


def test_a_paired_difference_refuses_an_arm_or_an_axis_a_world_of_its_look_lacks():
    records = _run()
    records[3] = _record("drawn", 3, {"a": 0.1, "b": 0.1}, missing=("b",))
    paired(records, "a", "b", 3)  # world 3 lies past the look
    with pytest.raises(ValueError, match="b's regret is not scored on drawn world 30"):
        paired(records, "a", "b", 4)
    with pytest.raises(ValueError, match="realised is not scored"):
        paired(_run(), "a", "b", 4, axis="realised")
    with pytest.raises(ValueError, match="one of"):
        paired(_run(), "a", "b", 4, axis="covered")

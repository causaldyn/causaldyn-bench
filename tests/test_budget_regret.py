"""Track M v2, budgets: the harness around the arms, checked apart from the arms themselves.

What PyMC-Marketing is handed of a geo test, the digest that ties its plan to the world it was
fitted on, the rules that score a failed or misplaced plan, and the statistics the gate reads.
"""

import dataclasses
import json
import sys

import jax
import numpy as np
import pytest
from chc.response import Channel, GeometricAdstock, Tanh
from scipy.stats import sem
from scipy.stats import t as student

from causaldyn_bench.budget_regret import (
    ARMS,
    PILOTS,
    PRIMARY,
    PYMC,
    TIE,
    Comparison,
    Environment,
    EnvironmentRun,
    Mean,
    WorldScore,
    _markdown,
    _record,
    _records,
    digest,
    experiments,
    export,
    lift_rows,
    main,
    myopic,
    score,
    score_world,
)
from causaldyn_bench.endogenous_mmm import EndogenousMediaMix
from causaldyn_bench.lift_calibration import (
    COOLDOWN,
    PRE,
    STARTS,
    TEST,
    Setting,
    clopper_pearson,
    lift_tests,
)
from causaldyn_bench.mmm_decision import Quarter, drawn, oracle, regret


@pytest.fixture(scope="module")
def world():
    return drawn(905).simulate(905)


def test_a_quiet_go_dark_test_reduces_to_its_gap_over_the_dark_weeks():
    world = EndogenousMediaMix().simulate(3)
    quiet = Setting("pla", noise_share=1e-12)
    rows = lift_rows({"pla": lift_tests(world, quiet, seed=1)})
    experiment = world.geo_test("pla", quiet.starts, noise_share=1e-12, seed=1)
    assert rows.dropped == 0
    assert rows.channel == ("pla",) * len(quiet.starts)
    assert rows.start == quiet.starts
    for k, start in enumerate(quiet.starts):
        gap = experiment.true_gap[experiment.readout(start)]
        assert rows.delta_y[k] == pytest.approx(np.sum(gap[PRE : PRE + TEST + COOLDOWN]) / TEST)
        assert rows.x[k] == pytest.approx(
            np.mean(experiment.spend_control[start - 1 : start - 1 + TEST])
        )
    np.testing.assert_array_equal(rows.delta_x, -rows.x)


def test_a_test_whose_sales_did_not_fall_is_dropped_and_counted():
    world = EndogenousMediaMix().simulate(4)
    tests = lift_tests(world, Setting("meta", noise_share=0.10), seed=2)
    rows = lift_rows({"meta": tests})
    rose = sum(float(np.sum(t.difference[PRE:])) >= 0.0 for t in tests)
    assert rose == 2
    assert rows.dropped == rose
    assert len(rows.channel) == len(tests) - rose
    assert np.all(rows.delta_y < 0.0)
    fell = [s for s, t in zip(STARTS, tests, strict=True) if float(np.sum(t.difference[PRE:])) < 0]
    assert rows.start == tuple(fell)


def test_the_digest_reads_the_world_and_its_tests_and_nothing_else(world):
    rows = lift_rows(experiments(world, 905))
    again = drawn(905).simulate(905)
    assert digest(again, lift_rows(experiments(again, 905))) == digest(world, rows)
    other = drawn(906).simulate(906)
    assert digest(other, lift_rows(experiments(other, 906))) != digest(world, rows)
    nudged = dataclasses.replace(rows, delta_y=rows.delta_y * (1.0 + 1e-12))
    assert digest(world, nudged) != digest(world, rows)


def test_the_digest_is_the_one_the_pre_registered_records_were_fitted_under(world):
    # computed at d6cf637, the commit the pre-registered PyMC-Marketing records ran from: a digest
    # that moves orphans them, since the scoring refuses a record fitted to other data
    assert digest(world, lift_rows(experiments(world, 905))) == "aba9555185e64d61"


def test_export_writes_what_the_pymc_arm_reads(world, tmp_path):
    rows = lift_rows(experiments(world, 905))
    quarter = Quarter.after(world)
    saved = np.load(export(world, 905, rows, tmp_path))
    assert str(saved["digest"]) == digest(world, rows)
    assert float(saved["budget"]) == quarter.budget
    np.testing.assert_array_equal(saved["lower"], quarter.lower)
    np.testing.assert_array_equal(saved["upper"], quarter.upper)
    np.testing.assert_array_equal(saved["sales"], world.sales)
    np.testing.assert_array_equal(saved["lift_delta_y"], rows.delta_y)
    assert tuple(saved["lift_channel"]) == rows.channel
    assert tuple(saved["channels"]) == world.channels


def test_export_dates_each_test_by_its_first_dark_week(world, tmp_path):
    rows = lift_rows(experiments(world, 905))
    saved = np.load(export(world, 905, rows, tmp_path))
    assert int(saved["lift_weeks"]) == TEST
    assert tuple(saved["lift_start"].tolist()) == rows.start
    assert set(rows.start) <= set(STARTS)
    for name, start, level in zip(rows.channel, rows.start, rows.x, strict=True):
        dark = world.spend[start - 1 : start - 1 + TEST, world.channels.index(name)]
        assert level == pytest.approx(float(np.mean(dark)))


def _arms(world, pymc):
    quarter = Quarter.after(world)
    arms = {arm: quarter.status_quo for arm in ARMS}
    arms[PYMC] = pymc
    return arms


def test_an_arm_that_failed_plays_the_status_quo_and_is_counted(world):
    scored = score(world, 905, _arms(world, "RuntimeError: no start converged"), pymc={}, chc={})
    assert scored.failure == {PYMC: "RuntimeError: no start converged"}
    assert scored.regret[PYMC] == scored.regret["status quo"]
    assert scored.regret["status quo"] > 0.0


def test_a_plan_off_the_budget_is_moved_into_the_box_and_the_move_recorded(world):
    quarter = Quarter.after(world)
    over = 1.01 * quarter.status_quo
    scored = score(world, 905, _arms(world, over), pymc={}, chc={})
    inside = quarter.project(over)
    assert scored.moved == {PYMC: pytest.approx(float(np.max(np.abs(inside - over))))}
    best = oracle(world, quarter)
    assert scored.regret[PYMC] == pytest.approx(
        regret(world, quarter, inside, best) / quarter.budget
    )


def test_the_oracles_plan_scores_nought(world):
    quarter = Quarter.after(world)
    scored = score(world, 905, _arms(world, oracle(world, quarter).weekly), pymc={}, chc={})
    assert abs(scored.regret[PYMC]) <= TIE
    assert scored.best == pytest.approx(oracle(world, quarter).worth)


def test_a_record_fitted_to_other_data_is_refused():
    record = {"digest": "0" * 16, "weekly": [1.0, 1.0, 1.0], "error": None}
    with pytest.raises(RuntimeError, match="other data"):
        score_world(("drawn", 905, record))


def test_a_world_scores_every_arm_with_the_pymc_record_kept_beside_it(world):
    rows = lift_rows(experiments(world, 905))
    quarter = Quarter.after(world)
    record = {
        "digest": digest(world, rows),
        "weekly": quarter.status_quo.tolist(),
        "error": None,
        "divergences": 0,
        "traceback": None,
    }
    scored = score_world(("drawn", 905, record))
    assert set(scored.regret) == set(ARMS)
    assert scored.failure == {}
    assert scored.regret[PYMC] == scored.regret["status quo"]
    assert scored.pymc == {"error": None, "divergences": 0}
    assert tuple(scored.chc) == world.channels
    for kept in scored.chc.values():
        assert set(kept) == {"kernel.retention", "curve.scale", "coefficient", "curve.scale.upper"}
        assert kept["curve.scale.upper"] >= kept["curve.scale"]


def test_the_mean_interval_is_students_t():
    values = np.array([0.1, 0.3, 0.2, 0.5, 0.05, 0.12])
    low, high = student.interval(0.95, values.size - 1, loc=values.mean(), scale=sem(values))
    mean = Mean.of(values)
    assert mean.interval == pytest.approx((low, high))
    assert mean.median == pytest.approx(0.16)


def test_a_comparison_counts_ties_apart_and_scores_the_lower_share():
    chc = np.array([0.1, 0.2, 0.3, 0.4, 0.0])
    other = np.array([0.2, 0.2 + TIE / 2, 0.1, 0.5, 0.3])
    compared = Comparison.of("x", chc, other)
    assert (compared.lower, compared.tied, compared.higher) == (3, 1, 1)
    assert compared.lower_share == clopper_pearson(3, 5)
    assert compared.difference.mean == pytest.approx(np.mean(chc - other))


def _run(shift: float) -> EnvironmentRun:
    rng = np.random.default_rng(0)
    fit = {"error": None, "divergences": 0, "max_rhat": 1.0, "optimiser_success": True}
    scores = []
    for seed in range(30):
        regrets = {arm: float(rng.uniform(0.1, 0.3)) for arm in ARMS}
        regrets["CHC"] = regrets[PYMC] + shift + float(rng.normal(0.0, 0.01))
        pymc = {**fit, "lift_rows": 11, "lift_dropped": 1}
        scores.append(WorldScore(seed, 1.0, 1.0, regrets, {}, {}, pymc, {}))
    return EnvironmentRun(PRIMARY, tuple(scores), 0.0)


def test_the_gate_reads_the_paired_interval_and_the_report_parses():
    met, missed = _run(-0.05), _run(+0.05)
    assert met.passes
    assert not missed.passes
    assert "): met." in _markdown([met], x64=True)
    assert "): NOT met." in _markdown([missed], x64=True)
    record = _record([met], True, {"python": "3.12.13"})
    assert json.loads(json.dumps(record, allow_nan=False))["environments"][0]["passes"] is True


def test_a_pilot_is_reported_without_a_gate():
    met = _run(-0.05)
    pilot = EnvironmentRun(PILOTS[0], met.scores, 0.0)
    text = _markdown([pilot], x64=True, pilot=True)
    assert "pilot" in text.splitlines()[0]
    assert "**Gate**" not in text
    assert _record([pilot], True, {}, pilot=True)["environments"][0]["passes"] is None


def test_the_myopic_reading_keeps_the_kernels_first_week_alone():
    channel = Channel(GeometricAdstock(0.5, length=6, normalized=True), Tanh(80.0), 900.0)
    (short,) = myopic([channel])
    spend = np.random.default_rng(0).uniform(0.0, 200.0, 30)
    first = float(channel.kernel.weights()[0])
    expected = 900.0 * np.tanh(first * spend / 80.0)
    np.testing.assert_allclose(np.asarray(short(spend)), expected, rtol=1e-5)


def test_a_run_missing_a_record_is_not_scored(tmp_path):
    folder = tmp_path / "drawn" / "pymc"
    folder.mkdir(parents=True)
    for seed in (900, 902):
        (folder / f"world_{seed}.json").write_text("{}")
    with pytest.raises(SystemExit, match="901"):
        _records(tmp_path, Environment("drawn", 900, 3))


def test_main_refuses_to_run_at_float32(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["budget_regret", "export"])
    with jax.enable_x64(False), pytest.raises(SystemExit, match="JAX_ENABLE_X64"):
        main()

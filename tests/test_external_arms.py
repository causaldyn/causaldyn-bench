"""Track M v2, budgets, the external arms: the harness that scores Meridian's and Robyn's records
beside the pre-registered run's arms, checked apart from the arms themselves.

The rules a record is scored by, the check that ties both sides of a comparison to one oracle, the
worlds each comparison reads, and the reading rule.
"""

import json
import math
import re
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.stats import t as student

from causaldyn_bench.budget_regret import (
    ENVIRONMENTS,
    PILOTS,
    PRIMARY,
    PYMC,
    TIE,
    Comparison,
    Environment,
    Mean,
    digest,
    experiments,
    lift_rows,
)
from causaldyn_bench.external_arms import (
    EDGE,
    EXTERNAL,
    MERIDIAN,
    PILOT_SAMPLES,
    READ,
    ROBYN,
    ROBYN_RANGES,
    ROBYN_SETTING,
    SAMPLES,
    ArmScore,
    EnvironmentRun,
    WorldScore,
    _committed,
    _markdown,
    _name,
    _record,
    _records,
    _tools,
    design,
    main,
    on_bounds,
    score_arm,
    score_world,
    verdict,
)
from causaldyn_bench.mmm_decision import Quarter, drawn, oracle, regret

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world():
    return drawn(905).simulate(905)


@pytest.fixture(scope="module")
def scored(world):
    quarter = Quarter.after(world)
    best = oracle(world, quarter)
    status_quo = regret(world, quarter, quarter.status_quo, best) / quarter.budget
    return quarter, best, status_quo


def _plan(weekly, **extra):
    return {"error": None, "weekly": list(map(float, weekly)), **extra}


def test_a_plan_inside_the_box_is_scored_on_the_worlds_own_channels(world, scored):
    quarter, best, status_quo = scored
    arm = score_arm(world, quarter, best, _plan(quarter.status_quo, divergences=0))
    assert arm.regret == pytest.approx(status_quo)
    assert (arm.failure, arm.moved) == (None, None)
    assert arm.record == {"error": None, "divergences": 0}


def test_an_arm_that_failed_plays_the_status_quo_and_is_counted(world, scored):
    quarter, best, status_quo = scored
    arm = score_arm(world, quarter, best, {"error": "ValueError: no draws", "traceback": "..."})
    assert arm.failure == "ValueError: no draws"
    assert arm.regret == pytest.approx(status_quo)
    assert "traceback" not in arm.record


def test_a_plan_off_the_budget_is_moved_into_the_box_and_the_move_recorded(world, scored):
    quarter, best, _ = scored
    over = quarter.status_quo + 1.0 / 13  # each channel a unit of the quarter's spend over
    arm = score_arm(world, quarter, best, _plan(over))
    inside = quarter.project(over)
    assert arm.moved == pytest.approx(float(np.max(np.abs(inside - over))))
    assert arm.regret == pytest.approx(regret(world, quarter, inside, best) / quarter.budget)


def test_a_moved_plans_bounds_are_read_where_it_was_moved_to(world, scored):
    quarter, best, _ = scored
    edge = quarter.status_quo.copy()
    edge[0] = quarter.upper[0] - 0.25 * EDGE * (quarter.upper[0] - quarter.lower[0])
    inside = quarter.project(edge)  # every channel shifted down: the first leaves its bound
    assert on_bounds(quarter, inside) != on_bounds(quarter, edge)
    assert score_arm(world, quarter, best, _plan(edge)).bounds == on_bounds(quarter, inside)


def test_the_oracles_plan_scores_nought_and_its_bounds_are_counted(world, scored):
    quarter, best, _ = scored
    arm = score_arm(world, quarter, best, _plan(best.weekly))
    assert abs(arm.regret) <= TIE
    assert arm.bounds == on_bounds(quarter, best.weekly)


def test_a_channel_is_on_a_bound_within_the_edge_of_its_box():
    quarter = Quarter(78.0, np.ones(3), np.full(3, 3.0), np.full(3, 2.0), np.zeros((1, 3)))
    width = 2.0
    inside = 0.5 * EDGE * width
    assert on_bounds(quarter, np.array([1.0 + inside, 2.0, 3.0 - inside])) == 2
    assert on_bounds(quarter, np.array([1.0 + 2 * EDGE * width, 2.0, 3.0 - 2 * EDGE * width])) == 0


def _job(scored, records):
    _, best, status_quo = scored
    known = {"best": best.worth, "regret": {"CHC": 0.01, PYMC: 0.02, "status quo": status_quo}}
    return ("drawn", 905, records, known)


def test_a_world_reads_the_committed_regrets_beside_the_arms_it_scores(world, scored):
    quarter, best, status_quo = scored
    fitted = digest(world, lift_rows(experiments(world, 905)))
    records = {MERIDIAN: {"digest": fitted, **_plan(quarter.status_quo)}}
    score = score_world(_job(scored, records))
    assert score.committed == {"CHC": 0.01, PYMC: 0.02, "status quo": pytest.approx(status_quo)}
    assert set(score.arms) == {MERIDIAN}
    assert score.arms[MERIDIAN].regret == pytest.approx(status_quo)
    assert (score.best, score.channels) == (pytest.approx(best.worth), len(world.channels))
    assert score.oracle_bounds == on_bounds(quarter, best.weekly)


def test_a_record_fitted_to_other_data_is_refused(world, scored):
    records = {ROBYN: {"digest": "0" * 16, **_plan(scored[0].status_quo)}}
    with pytest.raises(RuntimeError, match="Robyn record was fitted to other data"):
        score_world(_job(scored, records))


@pytest.mark.parametrize("moved", ["best", "status quo"])
def test_a_world_whose_oracle_is_not_the_pre_registered_runs_is_refused(world, scored, moved):
    quarter, best, status_quo = scored
    fitted = digest(world, lift_rows(experiments(world, 905)))
    records = {MERIDIAN: {"digest": fitted, **_plan(quarter.status_quo)}}
    job = _job(scored, records)
    if moved == "best":
        job[3]["best"] = best.worth + 10 * TIE * quarter.budget
    else:
        job[3]["regret"]["status quo"] = status_quo + 10 * TIE
    with pytest.raises(RuntimeError, match="not the pre-registered run's"):
        score_world(job)


def test_the_oracle_still_reproduces_the_committed_pilot(world):
    # the pre-registered run's own record of world 905, so a moved oracle fails here first
    committed = _committed(ROOT / "results/track_m2_budgets_pilot.json", pilot=True)
    known = committed["drawn"][905]
    fitted = digest(world, lift_rows(experiments(world, 905)))
    records = {MERIDIAN: {"digest": fitted, **_plan(Quarter.after(world).status_quo)}}
    score = score_world(("drawn", 905, records, known))
    assert score.committed == {arm: known["regret"][arm] for arm in READ}


def _score(seed, arms, committed):
    return WorldScore(seed, 1.0, 1.0, 3, 1, committed, arms)


def _arm(regret, **record):
    return ArmScore(regret, None, None, 0, record)


MERIDIAN_FIT = {"divergences": 0, "max_rhat": 1.0, "review": {"overall": "PASS"}}
ROBYN_FIT = {
    "convergence": ["NRMSE converged"],
    "selection": "clusters",
    "allocator": {"status": 4},
}


def _run(environment=PRIMARY, shift=-0.05, robyn=3):
    rng = np.random.default_rng(1)
    scores = []
    for seed in range(environment.first, environment.first + 12):
        meridian = float(rng.uniform(0.05, 0.15))
        committed = {"CHC": meridian + shift + float(rng.normal(0, 0.005)), PYMC: meridian}
        committed["status quo"] = meridian + 0.1
        arms = {MERIDIAN: _arm(meridian, **MERIDIAN_FIT)}
        if seed - environment.first < robyn:
            arms[ROBYN] = _arm(meridian - 0.01, **ROBYN_FIT)
        scores.append(_score(seed, arms, committed))
    return EnvironmentRun(environment, tuple(scores), 0.0)


def test_each_comparison_reads_the_worlds_both_arms_planned():
    run = _run()
    assert run.arms == EXTERNAL
    compared = run.comparisons
    assert list(compared) == [
        ("CHC", MERIDIAN),
        (PYMC, MERIDIAN),
        ("CHC", ROBYN),
        (PYMC, ROBYN),
        (MERIDIAN, "status quo"),
        (ROBYN, "status quo"),
        (MERIDIAN, ROBYN),
    ]
    sizes = {pair: c.lower + c.tied + c.higher for pair, c in compared.items()}
    assert sizes[("CHC", MERIDIAN)] == 12
    assert sizes[("CHC", ROBYN)] == sizes[(MERIDIAN, ROBYN)] == 3
    robyn = run.worlds(ROBYN)
    expected = Mean.of(run.regrets("CHC", robyn) - run.regrets(ROBYN, robyn))
    assert compared[("CHC", ROBYN)].difference == expected
    assert compared[(MERIDIAN, ROBYN)].difference.mean == pytest.approx(0.01)
    counts = {arm: len(run.worlds(arm)) for arm in run.means}
    assert counts == {"CHC": 12, PYMC: 12, "status quo": 12, MERIDIAN: 12, ROBYN: 3}


def test_a_run_without_robyn_compares_meridian_alone():
    run = _run(robyn=0)
    assert run.arms == (MERIDIAN,)
    assert all(ROBYN not in pair for pair in run.comparisons)


@pytest.mark.parametrize(
    ("interval", "read"),
    [((-0.03, -0.01), "beats"), ((0.01, 0.03), "loses to"), ((-0.01, 0.01), "ties")],
)
def test_the_reading_rule_reads_the_interval_against_nought(interval, read):
    difference = Mean(sum(interval) / 2, interval, 0.0, 0.01)
    assert verdict(Comparison("x", difference, 1, 0, 1, (0.0, 1.0))) == read


def test_the_reading_is_reported_for_the_scored_primary_alone():
    run = _run(shift=-0.05)
    assert set(run.claims) == {(f, s) for f in ("CHC", PYMC) for s in EXTERNAL}
    assert run.claims[("CHC", MERIDIAN)] == "beats"
    assert run.claims[(PYMC, MERIDIAN)] == "ties"  # the same regrets: the difference is nought
    text = _markdown([run], True, ROBYN_SETTING)
    assert "**Reading**" in text and "CHC beats Meridian" in text
    pilot = _markdown([_run(environment=PILOTS[0])], True, ROBYN_SETTING, pilot=True)
    assert "pilot" in pilot.splitlines()[0] and "**Reading**" not in pilot
    reference = _markdown([_run(environment=ENVIRONMENTS[1])], True, ROBYN_SETTING)
    assert "**Reading**" not in reference


@pytest.mark.parametrize("setting", sorted(ROBYN_RANGES))
def test_the_record_parses_strictly_and_keeps_every_world_and_the_setting(setting):
    run = _run()
    record = _record([run], True, {MERIDIAN: {"python": "3.12.13"}}, setting)
    again = json.loads(json.dumps(record, allow_nan=False))
    assert again["design"]["robyn_ranges"] == setting
    assert again["design"]["fits"][ROBYN]["ranges"] == ROBYN_RANGES[setting]
    environment = again["environments"][0]
    assert len(environment["worlds"]) == 12
    assert environment["claims"]["CHC - Meridian"] == "beats"
    assert {c["first"] for c in environment["comparisons"]} == {"CHC", PYMC, MERIDIAN, ROBYN}


def test_the_diagnostic_lines_count_from_the_records():
    (other,) = set(ROBYN_RANGES) - {ROBYN_SETTING}
    meridian = {"divergences": 3, "max_rhat": 1.02, "review": {"overall": "PASS"}}
    robyn = {
        "convergence": ["NRMSE converged: x", "DECOMP.RSSD NOT converged: y", "MAPE converged"],
        "selection": "clusters",
        "allocator": {"status": -1},
    }
    committed = {"CHC": 0.1, PYMC: 0.1, "status quo": 0.2}
    scores = [
        _score(900, {MERIDIAN: _arm(0.1, **meridian), ROBYN: _arm(0.1, **robyn)}, committed),
        _score(
            901,
            {
                MERIDIAN: ArmScore(0.2, "RuntimeError: x", None, 0, {}),
                ROBYN: ArmScore(0.1, None, 0.5, 1, {**robyn, "selection": "pareto"}),
            },
            committed,
        ),
    ]
    text = _markdown([EnvironmentRun(PILOTS[0], tuple(scores), 0.0)], True, other, pilot=True)
    assert (
        "Meridian's fits: 1 raised; of the rest, 1 had divergent transitions, 1 an R-hat over "
        "1.01, 0 did not pass Meridian's own review; no plan moved into the box." in text
    )
    assert (
        f"Robyn's fits over the {other} ranges: 0 raised; of the rest, its own convergence check "
        "passed NRMSE in 2, "
        "DECOMP.RSSD in 0 and MAPE in 2; the model came from the clusters in 1; its allocator "
        "stopped on an error in 2; 1 plans moved into the box, the furthest by 6.5 of a week's "
        "budget." in text
    )
    assert "Robyn's 1 of 6 against the oracle's 2" in text


def _write(folder, seed, record):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"world_{seed}.json").write_text(json.dumps({"error": None, **record}))


def test_a_run_missing_a_record_is_not_scored(tmp_path):
    for seed in (900, 902):
        _write(tmp_path / "drawn" / "meridian", seed, design(ROBYN_SETTING)[MERIDIAN])
    with pytest.raises(SystemExit, match="no Meridian record for seeds \\[901\\]"):
        _records(tmp_path, {MERIDIAN: (Environment("drawn", 900, 3),)}, ROBYN_SETTING)


def test_a_fit_that_departs_from_the_design_is_not_scored_unless_it_raised(tmp_path):
    folder = tmp_path / "drawn" / "robyn-genre"
    sample = {ROBYN: (Environment("drawn", 900, 1),)}
    for departure in ({"trials": 5}, {"ranges": ROBYN_RANGES["union"]}):
        _write(folder, 900, {**design("genre")[ROBYN], **departure})
        with pytest.raises(SystemExit, match="Robyn ran"):
            _records(tmp_path, sample, "genre")
    _write(folder, 900, {"error": "Error: no fit"})
    assert _records(tmp_path, sample, "genre")["drawn"][900][ROBYN]["error"] == "Error: no fit"


def test_robyns_records_are_read_from_their_settings_folder(tmp_path):
    sample = {MERIDIAN: (Environment("drawn", 900, 1),), ROBYN: (Environment("drawn", 900, 1),)}
    _write(tmp_path / "drawn" / "meridian", 900, design("genre")[MERIDIAN])
    for setting in ROBYN_RANGES:
        _write(tmp_path / "drawn" / f"robyn-{setting}", 900, design(setting)[ROBYN])
    for setting in ROBYN_RANGES:
        read = _records(tmp_path, sample, setting)["drawn"][900]
        assert read[ROBYN]["ranges"] == ROBYN_RANGES[setting]
        assert read[MERIDIAN] == {"error": None, **design(setting)[MERIDIAN]}


def test_records_from_two_environments_are_not_scored():
    one = {"python": "3.12.13", "versions": {"google-meridian": "2.1.0"}}
    two = {"python": "3.12.13", "versions": {"google-meridian": "2.1.1"}}
    records = {"drawn": {900: {MERIDIAN: one}, 901: {MERIDIAN: two}}}
    with pytest.raises(SystemExit, match="Meridian records ran on 2 environments"):
        _tools(records)
    assert _tools({"drawn": {900: {MERIDIAN: one}}}) == {MERIDIAN: one}


def test_the_committed_run_must_be_the_one_asked_for():
    with pytest.raises(SystemExit, match="is a pilot"):
        _committed(ROOT / "results/track_m2_budgets_pilot.json", pilot=False)


def test_each_sample_lies_inside_the_pre_registered_runs_worlds():
    for samples, environments in ((SAMPLES, ENVIRONMENTS), (PILOT_SAMPLES, PILOTS)):
        assert samples[MERIDIAN] == environments
        for sample in samples[ROBYN]:
            (budgets,) = [e for e in environments if e.name == sample.name]
            assert set(sample.seeds) <= set(budgets.seeds)


def _recipe(name):
    line = re.search(rf"^{name} .*$", (ROOT / "justfile").read_text(), flags=re.MULTILINE)
    assert line is not None
    return line.group(0)


def test_the_robyn_recipes_fit_the_samples_seeds_the_scored_one_over_the_chosen_ranges():
    for recipe, samples in (("track-m2-robyn", SAMPLES), ("track-m2-robyn-pilot", PILOT_SAMPLES)):
        (sample,) = samples[ROBYN]
        assert f'"{sample.first}:{sample.first + sample.worlds}"' in _recipe(recipe)
    assert f'"{ROBYN_SETTING}" rlib venv shards' in _recipe("track-m2-robyn")
    assert '"900:910" ranges rlib venv shards' in _recipe("track-m2-robyn-pilot ranges")


def test_robyns_records_are_written_where_the_scoring_reads_them():
    text = (ROOT / "justfile").read_text()
    body = text.split("\n_track-m2-robyn ", 1)[1].split("\n\n", 1)[0]
    assert "--ranges {{ranges}}" in body
    assert "--out {{work}}/drawn/robyn-{{ranges}}" in body


def _worlds_for(difference):
    """The fewest worlds, in tens, at whose pilot spread the interval's half-width is half the
    pilot's difference."""
    for worlds in range(10, 10_000, 10):
        half = student.ppf(0.975, worlds - 1) * difference["sd"] / math.sqrt(worlds)
        if half <= abs(difference["mean"]) / 2:
            return worlds
    return 10_000


def _pilot(setting):
    """The drawn worlds of the committed pilot over one setting of Robyn's ranges."""
    pilot = json.loads((ROOT / "results" / f"{_name(setting, pilot=True)}.json").read_text())
    assert pilot["pilot"]
    assert pilot["design"]["robyn_ranges"] == setting
    (drawn,) = [e for e in pilot["environments"] if e["environment"]["name"] == "drawn"]
    return drawn


def test_robyn_is_scored_over_the_setting_whose_pilot_regret_was_the_lower():
    regrets = {setting: _pilot(setting)["means"][ROBYN]["mean"] for setting in ROBYN_RANGES}
    assert min(regrets, key=regrets.__getitem__) == ROBYN_SETTING


def test_robyns_sample_is_the_pilots_rule():
    needed = [
        _worlds_for(c["difference"])
        for c in _pilot(ROBYN_SETTING)["comparisons"]
        if c["first"] in ("CHC", PYMC) and c["second"] == ROBYN
    ]
    assert len(needed) == 2
    (sample,) = SAMPLES[ROBYN]
    assert sample.worlds == min(max(*needed, 30), 100)


def test_main_reads_robyns_other_setting_in_the_pilot_alone(monkeypatch, capsys):
    (other,) = set(ROBYN_RANGES) - {ROBYN_SETTING}
    monkeypatch.setattr(sys, "argv", ["external_arms", "--robyn-ranges", other])
    with pytest.raises(SystemExit):
        main()
    assert f"over the {ROBYN_SETTING} ranges alone" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["external_arms", "--pilot", "--robyn-ranges", other])
    with jax.enable_x64(False), pytest.raises(SystemExit, match="JAX_ENABLE_X64"):
        main()


def test_main_writes_a_pilot_under_its_settings_name(monkeypatch, tmp_path):
    (other,) = set(ROBYN_RANGES) - {ROBYN_SETTING}
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr("causaldyn_bench.external_arms._records", lambda *_: {})
    argv = ["external_arms", "--pilot", "--robyn-ranges", other, "--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    with jax.enable_x64(True):
        main()
    stem = f"track_m2_external_pilot_{other}"
    assert {p.name for p in tmp_path.iterdir()} == {f"{stem}.md", f"{stem}.json"}


def test_main_refuses_to_run_at_float32(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["external_arms"])
    with jax.enable_x64(False), pytest.raises(SystemExit, match="JAX_ENABLE_X64"):
        main()

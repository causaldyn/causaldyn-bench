"""Track P: the harness, checked against the environments it steps and the alarm it reads.

The number every other rests on is the residual: the next state less the linearisation's prediction
at the action the environment applied. If it held the command instead, a clipped decision's
residual would move with its draw, and the monitor would be reading the harness, not the plant.
"""

import json
import math

import numpy as np
import pytest

pytest.importorskip("gymnasium", reason="Track P needs the `gym` extra")

from chc.gate import DriftAlarm

from causaldyn_bench.drift_calibration import (
    ARL,
    DISTURBANCE,
    HORIZON,
    PLANS,
    Arm,
    MoveScore,
    NullScore,
    _markdown,
    _move,
    _null,
    _record,
    first_alarm,
    gate,
    plan_at,
    run_environment,
    run_path,
    set_channel,
)
from causaldyn_bench.ope_calibration import ENVIRONMENTS


@pytest.mark.parametrize(
    ("key", "step", "atol"), [("pendulum", 1e-4, 1e-8), ("mountain_car", 1e-2, 1e-5)]
)
def test_setting_the_channel_scales_the_environments_step_in_the_action_and_nothing_else(
    key, step, atol
):
    operating = ENVIRONMENTS[key]()
    envs = {factor: operating.make() for factor in (1.0, 1.25)}
    set_channel(envs[1.25], key, 1.25)

    def advance(factor, x, u):
        operating.place(envs[factor], x)
        envs[factor].step(np.array([u]))
        return operating.read(envs[factor])

    off_rest = np.array([0.05, -0.01]) if key == "pendulum" else np.array([0.1, 0.002])
    np.testing.assert_array_equal(advance(1.25, off_rest, 0.0), advance(1.0, off_rest, 0.0))
    b = (advance(1.25, np.zeros(2), step) - advance(1.25, np.zeros(2), -step)) / (2 * step)
    np.testing.assert_allclose(b, 1.25 * operating.b[:, 0], atol=atol)


@pytest.mark.parametrize("key", list(ENVIRONMENTS))
@pytest.mark.parametrize("share", list(PLANS.values()))
def test_the_plan_averages_its_share_of_the_bound_at_rest_on_the_linearisation(key, share):
    operating = ENVIRONMENTS[key]()
    plan = plan_at(operating, share)
    a, b = operating.a, operating.b[:, 0]
    rest = np.linalg.solve(np.eye(2) - a - np.outer(b, plan.gain), b * plan.offset)
    assert plan.gain @ rest + plan.offset == pytest.approx(share * operating.bound, rel=1e-9)


@pytest.mark.parametrize("key", list(ENVIRONMENTS))
@pytest.mark.parametrize(("factor", "clipped"), [(1.0, True), (1.25, False)])
def test_a_path_logs_the_action_applied_and_its_residual_moves_with_it_only_through_the_channel(
    key, factor, clipped
):
    operating = ENVIRONMENTS[key]()
    env = operating.make()
    set_channel(env, key, factor)
    steps = 6000
    chunks = list(
        run_path(
            operating,
            env,
            plan_at(operating, PLANS["on the bound"]),
            steps,
            np.random.default_rng(5),
        )
    )
    applied = np.concatenate([log.action for log, _, _ in chunks])
    drawn = np.concatenate([log.dither for log, _, _ in chunks if log.dither is not None])
    saturated = np.concatenate([log.saturated for log, _, _ in chunks])
    residual = np.concatenate([r for _, r, _ in chunks])
    assert sum(clips for _, _, clips in chunks) == 0
    np.testing.assert_array_equal(saturated, np.abs(applied) == operating.bound)
    assert 0.1 < saturated.mean() < 0.35
    scale = np.abs(operating.b[:, 0]) * DISTURBANCE * operating.bound
    z = residual / scale
    # what is left of the residual once the channel's own share is taken out, draw by draw
    left = z - np.outer(applied, (factor - 1.0) * operating.b[:, 0] / scale)
    for rows in (saturated, ~saturated) if clipped else (np.ones(steps, dtype=bool),):
        n = int(rows.sum())
        for column in left[rows].T:
            corr = np.corrcoef(column, drawn[rows])[0, 1]
            assert abs(corr) * math.sqrt(n) < 4.0
    spread = z.std(axis=0)
    assert np.all((spread > 0.9) & (spread < 2.0))


def test_first_alarm_is_the_row_a_row_by_row_update_sounds_at():
    rows = np.ones((12, 2))
    rows[6] = 2000.0  # (6 + 1) * 2000 >= ARL
    alarm = DriftAlarm(ARL)
    assert first_alarm(alarm, rows[:4]) is None
    assert first_alarm(alarm, rows[4:]) == 2
    assert alarm.statistic == 0.0
    by_row = DriftAlarm(ARL)
    sounded = [by_row.update(row[None, :]) for row in rows]
    assert sounded.index(True) == 6


def test_a_null_is_scored_by_horizon_and_by_censored_run_length():
    score = _null(np.array([HORIZON, HORIZON + 1, np.inf, 2000.0]), cap=10_000)
    assert score.alarmed_by_horizon == 0.25
    assert score.bound == HORIZON / ARL
    expected = (2 * HORIZON + 1 + 10_000 + 2000) / 4 / ARL
    assert score.run_length_over_arl == pytest.approx(expected)
    assert score.censored == 0.25
    moved = _move(np.array([50.0, np.inf, 150.0]))
    assert moved.caught == pytest.approx(2 / 3)
    assert (moved.mean_delay, moved.median_delay) == (100.0, 100.0)
    assert _move(np.full(3, np.inf)) == MoveScore(0.0, None, None)


def _arm(channel, reading, low):
    null = NullScore(low + 0.05, (low, low + 0.1), HORIZON / ARL, 3.0, 0.4)
    move = MoveScore(1.0, 200.0, 180.0)
    moved = channel == "moved"
    return Arm(
        "on the bound",
        channel,
        reading,
        0.25,
        0,
        None if moved else null,
        move if moved else None,
        [HORIZON, None],
    )


def test_the_gate_reads_the_librarys_reading_on_the_nulls_and_nothing_else():
    assert gate(_arm("modelled", "draw", 0.0)) is True
    assert gate(_arm("on the edge", "draw", HORIZON / ARL)) is True
    assert gate(_arm("on the edge", "draw", HORIZON / ARL + 0.01)) is False
    assert gate(_arm("modelled", "clipped zeroed", 0.9)) is None
    assert gate(_arm("moved", "draw", 0.0)) is None


@pytest.mark.parametrize("key", list(ENVIRONMENTS))
def test_a_small_run_scores_every_arm_and_writes_its_report(key):
    run = run_environment(key, paths=3, cap=600, move_steps=600)
    # inside: three channels read one way; on the bound: three channels, each read two ways
    assert len(run.arms) == 9
    assert sum(gate(arm) is not None for arm in run.arms) == 4
    assert all(arm.state_clipped == 0 for arm in run.arms)
    for arm in run.arms:
        assert len(arm.first_alarms) == 3
        if arm.null is not None:
            censored = [t is None for t in arm.first_alarms]
            assert arm.null.censored == pytest.approx(np.mean(censored))
    text = _markdown([run], paths=3, cap=600, x64=True)
    assert "## The gate" in text
    assert run.name in text
    record = json.loads(json.dumps(_record([run], paths=3, cap=600, x64=True)))
    assert len(record["environments"][0]["arms"]) == 9

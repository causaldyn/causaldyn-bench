"""Track M v2, families: the rules that pick and keep a family, the readings the robust arm hedges
between, the gate's statistic, and the arms on one world against the budget track's."""

import dataclasses
import json
import math
import sys

import jax
import numpy as np
import pytest
from chc.lift import LiftFit
from chc.response import Channel, GeometricAdstock, Hill, MichaelisMenten, Tanh
from scipy import stats

from causaldyn_bench.budget_regret import experiments
from causaldyn_bench.budget_regret import plans as budget_plans
from causaldyn_bench.endogenous_mmm import CURVES
from causaldyn_bench.family_regret import (
    ARMS,
    CANDIDATES,
    ENVIRONMENTS,
    PILOTS,
    TRUE_CANDIDATE,
    CurveRun,
    Environment,
    WorldScore,
    _channel,
    _markdown,
    _record,
    distinct,
    gate,
    main,
    plans,
    select,
    template,
)
from causaldyn_bench.lift_calibration import template as lift_template
from causaldyn_bench.mmm_decision import Quarter

PERIODS = 64  # four tests of sixteen periods
KERNEL = GeometricAdstock(0.5, length=6, normalized=True)


def _fit(cost: float, parameters: int) -> LiftFit:
    """A fit as the rules read it: its least squares, periods and parameter count."""
    dof = PERIODS - parameters
    channel = Channel(KERNEL, Tanh(100.0), 1.0)
    empty = np.zeros(parameters)
    names = tuple(f"p{k}" for k in range(parameters))
    return LiftFit(
        channel, names, empty, empty, empty, 0.95, math.sqrt(cost / dof), dof, (0, 1), ()
    )


def test_aic_and_the_keep_rule_read_each_fit_s_least_squares():
    fits = {
        "tanh": _fit(100.0, 3),
        "exponential": _fit(100.0, 3),
        "michaelis-menten": _fit(130.0, 3),
        "hill": _fit(95.0, 4),
        "weibull": _fit(90.0, 4),
        "logistic": "ValueError: the gap's slope is not finite",
    }
    chosen = select(fits)
    cutoff = 90.0 / 60 * stats.f.ppf(0.95, 1, 60)  # the least's variance, one parameter's F
    assert 5.0 < cutoff < 10.0  # so Hill is kept and the concave families are not
    assert chosen.least == "weibull"
    assert chosen.kept == ("hill", "weibull")
    costs = {"tanh": (100.0, 3), "hill": (95.0, 4), "weibull": (90.0, 4)}
    aic = {f: PERIODS * math.log(c / PERIODS) + 2 * p for f, (c, p) in costs.items()}
    assert chosen.aic == min(aic, key=aic.__getitem__) == "weibull"


def test_a_tie_goes_to_the_first_candidate_and_no_fit_is_no_choice():
    tie = {family: _fit(100.0, 3) for family in ("michaelis-menten", "tanh")}
    assert select(tie).aic == "michaelis-menten"  # the fits' own order, which is CANDIDATES'
    assert select({f: _fit(100.0, 3) for f in CANDIDATES}).aic == "tanh"
    with pytest.raises(RuntimeError, match="no candidate"):
        select(dict.fromkeys(CANDIDATES, "RuntimeError: did not converge"))


def test_a_fit_that_reads_a_loss_is_planned_as_a_channel_that_does_nothing():
    """The allocation plans increasing channels, so a negative coefficient enters an arm as nought,
    and a fit of either other sign as it is."""
    loss = dataclasses.replace(_fit(100.0, 3), channel=Channel(KERNEL, Tanh(1e12), -2.5e12))
    planned = _channel(loss)
    assert float(planned.coefficient) == 0.0
    assert (planned.kernel, planned.curve) == (loss.channel.kernel, loss.channel.curve)
    gain = _fit(100.0, 3)
    assert _channel(gain) is gain.channel
    with pytest.raises(TypeError, match="no fit to plan on"):
        _channel("RuntimeError: did not converge")


def test_readings_that_return_alike_over_the_box_are_one():
    """Two concave families at their linear limit return the same over the box, whatever their
    scales; a curve that bends is a reading of its own."""
    world = Environment("tanh", 905, 1).generator(905).simulate(905)
    quarter = Quarter.after(world)
    kernel = GeometricAdstock(0.4, length=6, normalized=True)
    with jax.enable_x64(True):
        line = Channel(kernel, Tanh(1e9), 1e9 * 3.0)
        also = Channel(kernel, MichaelisMenten(1e12), 1e12 * 3.0)
        bent = Channel(kernel, Hill(80.0, 2.0), 400.0)
        assert distinct([line, also, bent, line], 0, quarter) == (0, 2)


def _scores(curve: str, shift: float, seeds=range(20)) -> tuple[WorldScore, ...]:
    rng = np.random.default_rng(len(curve))
    channel = {
        "fits": {f: {"cost": 1.0} for f in CANDIDATES},
        "least": "tanh",
        "aic": "tanh",
        "kept": ["tanh", "hill"],
        "distinct": ["tanh", "hill"],
    }
    hedge = {"readings": 8.0, "worst": 1.0, "bound": 1.0}
    out = []
    for seed in seeds:
        regret = {arm: float(rng.uniform(0.05, 0.1)) for arm in ARMS}
        regret["robust"] = regret["AIC"] + shift + float(rng.normal(0.0, 0.002))
        channels = (channel, channel, channel)
        out.append(WorldScore(seed, curve, 1.0, 1.0, regret, {}, {}, channels, hedge))
    return tuple(out)


def test_the_gate_reads_the_worst_family_with_the_seeds_paired():
    """The worst family's mean regret, the robust plan's against AIC's, and a bootstrap that
    draws the seeds once for every family."""
    runs = [
        CurveRun(Environment(curve, 0, 20), _scores(curve, -0.02 if k else -0.01), 0.0)
        for k, curve in enumerate(("tanh", "hill-2"))
    ]
    verdict = gate(runs)
    assert verdict.robust == max(r.regrets("robust").mean() for r in runs)
    assert verdict.aic == max(r.regrets("AIC").mean() for r in runs)
    assert verdict.difference == pytest.approx(verdict.robust - verdict.aic)
    assert verdict.interval[0] <= verdict.difference <= verdict.interval[1]
    assert verdict.passes
    lagging = _scores("exponential", 0.0, range(1, 21))
    with pytest.raises(ValueError, match="share their seeds"):
        gate([*runs, CurveRun(Environment("exponential", 1, 20), lagging, 0.0)])


def test_the_report_parses_and_a_pilot_has_no_gate():
    runs = [CurveRun(Environment(c, 0, 20), _scores(c, -0.02), 0.0) for c in ("tanh", "hill-2")]
    assert "): met." in _markdown(runs, x64=True)
    record = _record(runs, True)
    assert json.loads(json.dumps(record, allow_nan=False))["gate"]["passes"] is True
    pilot = _markdown(runs, x64=True, pilot=True)
    assert "pilot" in pilot.splitlines()[0]
    assert "**Gate**" not in pilot
    assert _record(runs, True, pilot=True)["gate"]["passes"] is None


def test_every_curve_has_its_candidate_and_the_scored_seeds_are_fresh():
    assert set(TRUE_CANDIDATE) == set(CURVES)
    assert set(TRUE_CANDIDATE.values()) <= set(CANDIDATES)
    assert {e.curve for e in ENVIRONMENTS} == set(CURVES)
    scored = {s for e in ENVIRONMENTS for s in e.seeds}
    assert not scored & {s for e in PILOTS for s in e.seeds}
    assert not scored & (set(range(10_000, 10_200)) | set(range(20_000, 20_100)))  # the budgets'


def test_on_his_curve_the_tanh_arm_is_the_budget_track_s_chc_arm():
    """His curve's world, one seed: the tanh candidate starts where the budget track's lift fit
    starts, so the tanh arm, and the true family's, are that track's CHC plan; the robust plan
    spends the budget in the box, and its hedge is certified."""
    with jax.enable_x64(True):
        world = Environment("tanh", 905, 1).generator(905).simulate(905)
        tests = experiments(world, 905)
        ours, theirs = template("tanh", tests["pla"], 6), lift_template(tests["pla"], 6)
        assert float(ours.curve.scale) == float(theirs.curve.scale)
        arms, channels, hedge = plans(world, tests)
        budget, _ = budget_plans(world, tests)
    quarter = Quarter.after(world)
    np.testing.assert_array_equal(arms["tanh"], budget["CHC"])
    np.testing.assert_array_equal(arms["true family"], arms["tanh"])
    robust = arms["robust"]
    assert not isinstance(robust, str), robust
    assert quarter.feasible(robust)
    assert hedge["worst"] - hedge["bound"] <= 1e-9 * abs(hedge["worst"]) + 1e-6
    assert all(set(c["distinct"]) <= set(c["kept"]) for c in channels)
    assert all(c["aic"] in c["kept"] or c["aic"] not in c["kept"] for c in channels)
    assert all("failure" not in c["fits"][c["aic"]] for c in channels)


def test_main_refuses_to_run_at_float32(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["family_regret"])
    with jax.enable_x64(False), pytest.raises(SystemExit, match="JAX_ENABLE_X64"):
        main()

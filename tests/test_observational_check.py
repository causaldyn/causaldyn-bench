"""Track M v2, the observational check: the gaps it scores against are the generator's own, and
its rates are read over the checks that ran."""

import dataclasses
import json
import math
import sys

import jax
import numpy as np
import pytest

from causaldyn_bench.budget_regret import experiments
from causaldyn_bench.endogenous_mmm import EndogenousMediaMix
from causaldyn_bench.observational_check import (
    FIRST,
    GAP,
    HISTORIES,
    PILOT,
    Check,
    Rate,
    Reading,
    Run,
    _markdown,
    _record,
    check_history,
    gaps,
    main,
    needed_gamma,
    score,
    true_channel,
    true_gaps,
)


def test_the_world_s_own_gaps_are_its_channel_s_and_the_tests_carry_them_with_noise():
    """The noise-free factor is scored against the generator's gap, so the truth's channel must
    predict exactly that gap, and each test's gap must be it with the groups' noise alone."""
    generator = EndogenousMediaMix()
    world = generator.simulate(7)
    tests = experiments(world, 7)
    for name in world.channels:
        made = true_gaps(world, name)
        with jax.enable_x64(True):
            own = true_channel(generator, name, world.kernel_length)
            np.testing.assert_allclose(gaps(own, tests[name]), made, rtol=1e-9, atol=1e-9)
        noise = np.concatenate([t.difference for t in tests[name]]) - made
        assert 0.0 < np.std(noise) < 0.1 * np.mean(world.sales)


def _check(p: float, factor: float, half: float = 0.1, failure: str | None = None) -> Check:
    lower, upper = factor - half, factor + half
    least = needed_gamma(min(max(1.0, lower), upper)) if not math.isnan(factor) else math.nan
    return Check(p, factor, lower, upper, least, failure)


def _readings() -> tuple[Reading, ...]:
    ok = _check(0.5, 1.0)
    return (
        Reading(0, "pla", ok, _check(1e-9, 0.5), 0.52),  # covered
        Reading(1, "pla", _check(0.01, 1.2), _check(1e-9, 0.5), 0.8),  # neither covers
        Reading(2, "pla", ok, _check(0.2, math.nan), math.nan),  # no gap predicted
        Reading(3, "pla", ok, Check.missing("fits better"), 0.5),  # the check raised
        Reading(4, "pla", Check.missing("lift: raised"), Check.missing("lift"), math.nan, "lift"),
    )


def test_a_score_reads_each_rate_over_the_checks_that_ran():
    s = score(_readings())
    assert (s.size.share, s.size.count) == (1 / 4, 4)  # the raised fit is not read
    assert (s.covers_one.share, s.covers_one.count) == (3 / 4, 4)
    assert (s.rejected.share, s.rejected.count) == (2 / 3, 3)  # a channel with no gap is tested
    assert (s.covers_target.share, s.covers_target.count) == (1 / 2, 2)  # but has no factor
    assert (s.no_factor, s.failed, s.unchecked) == (1, 1, 1)
    assert s.median_factor == 0.5
    assert not s.passes


def test_the_gate_reads_validity_one_sided_and_power_apart():
    s = score(_readings())
    conservative = dataclasses.replace(
        s,
        size=Rate.of([False] * 100),
        covers_one=Rate.of([True] * 100),
        covers_target=Rate.of([True] * 100),
        rejected=Rate.of([True] * 90 + [False] * 10),
    )
    assert conservative.passes
    assert not dataclasses.replace(conservative, size=Rate.of([True] * 8 + [False] * 92)).passes
    assert not dataclasses.replace(
        conservative, covers_one=Rate.of([True] * 92 + [False] * 8)
    ).passes
    assert not dataclasses.replace(conservative, covers_target=Rate.of([])).passes  # none read
    assert not dataclasses.replace(
        conservative, rejected=Rate.of([True] * 89 + [False] * 11)
    ).passes


def test_the_floor_is_read_against_the_gamma_the_noise_free_factor_needs():
    assert GAP == 1.0
    assert needed_gamma(1.0) == 1.0
    assert needed_gamma(0.5) == pytest.approx(3.0)  # (1 + 0.5) / (1 - 0.5)
    assert needed_gamma(1.5) == pytest.approx(3.0)
    assert needed_gamma(2.1) == math.inf  # no level reaches it
    covered = _check(1e-9, 0.5)
    assert covered.least_gamma <= needed_gamma(0.52)


def test_the_report_parses_and_a_pilot_has_no_gate():
    result = Run((0, 1, 2, 3, 4), _readings(), 1.0)
    assert "**Gate** (pla" in _markdown(result, x64=True)
    pilot = _markdown(result, x64=True, pilot=True)
    assert "pilot" in pilot.splitlines()[0]
    assert "**Gate**" not in pilot
    record = json.loads(json.dumps(_record(result, True), allow_nan=False))
    assert record["gate"] is False
    assert _record(result, True, pilot=True)["gate"] is None


def test_the_scored_seeds_are_fresh():
    scored = set(range(FIRST, FIRST + HISTORIES))
    assert not scored & set(PILOT)
    used = {*range(500), *range(10_000, 10_200), *range(20_000, 20_100), *range(30_000, 30_100)}
    assert not scored & used  # the lift calibration's, the budgets' and the families'


def test_one_history_checks_both_channels_of_every_channel_s_fit():
    with jax.enable_x64(True):
        readings = check_history(901)
    assert [r.channel for r in readings] == ["pla", "meta", "tv"]
    for r in readings:
        assert r.failure is None
        assert r.truth.failure is None and r.observational.failure is None
        assert 0.0 <= r.truth.p_value <= 1.0
        assert r.truth.lower < r.truth.factor < r.truth.upper
        # an observational coefficient of nought predicts no gap, so neither factor is read; on
        # television the fit lands on nought or just off it as the platform's solver steps fall
        assert math.isnan(r.target) == math.isnan(r.observational.factor)
    assert math.isfinite(readings[0].target)  # paid shopping, the channel the gate reads


def test_main_refuses_to_run_at_float32(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["observational_check"])
    with jax.enable_x64(False), pytest.raises(SystemExit, match="JAX_ENABLE_X64"):
        main()

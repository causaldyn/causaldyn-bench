"""D21's harness offline, on a surrogate plant; the live run is the experiment, not a test."""

import json
import math
import os
import signal
import sys
from dataclasses import replace
from typing import TypeVar

import jax
import numpy as np
import pytest

from causaldyn_bench import boptest_rerun
from causaldyn_bench.boptest import BOPTestClient, is_available
from causaldyn_bench.boptest_capped import SurrogatePlant, ZoneModel
from causaldyn_bench.boptest_causal import HEAT_PUMP, KELVIN, LOWER_SETP, log_episode
from causaldyn_bench.boptest_prescribe import ARMS, DAY_S, OUTDOOR, SOLAR, panel_from_log
from causaldyn_bench.boptest_rerun import (
    DRY_BULB,
    GLOBAL_HORIZONTAL,
    UPPER_SETP,
    Design,
    Forecast,
    Step,
    episode_summary,
    markdown,
    prescription,
    run_arm_episode,
    run_experiment,
    run_replicate,
    verdict,
)

_URL = os.environ.get("BOPTEST_URL")
_STEP_S = 1800.0
_SMALL = Design(
    log_steps=192,
    control_steps=6,
    replicates=2,
    stride_days=1,
    warmup_days=0.0,
    horizon=4,
    offsets=(-0.5, 0.5),
)


class _Building(SurrogatePlant):
    """L8.1's surrogate building with the band's upper edge and the weather in its forecast.

    ``tdis_tot`` integrates the zone's distance outside the band and ``ener_tot`` the modulation,
    both in hours from ``initialize`` on. A call without an overwrite gets a proportional
    controller aiming half a kelvin above the lower bound, standing in for BOPTEST's baseline.
    """

    upper = 24.0

    def initialize(self, testid, start_time, warmup_period):
        self._tdis = self._ener = 0.0
        return super().initialize(testid, start_time, warmup_period)

    def advance(self, testid, u):
        bound = float(self.covariates[min(self._round, self.covariates.shape[0] - 1), 2])
        if HEAT_PUMP.action_point not in u:
            u = HEAT_PUMP.overwrite(float(np.clip(0.3 + 2.0 * (bound + 0.5 - self._temp), 0, 1)))
        hours = self.step_s / 3600.0
        self._ener += hours * float(np.clip(u[HEAT_PUMP.action_point], 0.0, 1.0))
        measurements = super().advance(testid, u)
        self._tdis += hours * (max(0.0, bound - self._temp) + max(0.0, self._temp - self.upper))
        return measurements

    def forecast(self, testid, point_names, horizon, interval):
        rows = np.arange(self._round, self._round + round(horizon / interval) + 1)
        rows = np.minimum(rows, self.covariates.shape[0] - 1)
        columns = {
            LOWER_SETP: self.covariates[rows, 2] + KELVIN,
            UPPER_SETP: np.full(rows.size, self.upper + KELVIN),
            DRY_BULB: self.covariates[rows, 0] + KELVIN,
            GLOBAL_HORIZONTAL: self.covariates[rows, 1] * 1000.0,
        }
        return {name: [float(v) for v in columns[name]] for name in point_names}

    def kpi(self, testid):
        return {"tdis_tot": self._tdis, "ener_tot": self._ener, "time_rat": 1.0}


def _building(seed: int = 0) -> _Building:
    """A week of half-hour rounds: daily and slower outdoor swings, midday sun, and a bound that is
    high from midnight to noon, so a window starting at midnight starts occupied and too cold."""
    rounds = 336
    hours = 0.5 * np.arange(rounds)
    outdoor = 2.0 + 4.0 * np.sin(2 * np.pi * hours / 24.0) + 3.0 * np.sin(2 * np.pi * hours / 79.0)
    solar = 0.4 * np.clip(np.sin(2 * np.pi * (hours - 6.0) / 24.0), 0.0, None)
    bound = np.where(hours % 24 < 12, 21.0, 17.0)
    truth = ZoneModel(
        bias=0.45,
        pole=-0.03,
        weather=(0.03, 0.5, 0.0),
        weather_mean=(0.0, 0.0, 0.0),
        weather_scale=(1.0, 1.0, 1.0),
        authority=0.6,
        residual_sd=0.05,
    )
    return _Building(
        model=truth,
        effect=0.6,
        covariates=np.column_stack([outdoor, solar, bound]),
        temps=np.full(rounds, 19.5),
        noise=np.random.default_rng(seed).normal(0.0, 0.05, rounds),
        step_s=_STEP_S,
    )


_B = TypeVar("_B", bound=_Building)


def _like(cls: type[_B], plant: _Building) -> _B:  # noqa: UP047 -- CI runs 3.11
    """A plant of class ``cls`` with ``plant``'s model, weather and noise."""
    return cls(
        model=plant.model,
        effect=plant.effect,
        covariates=plant.covariates,
        temps=plant.temps,
        noise=plant.noise,
        step_s=_STEP_S,
    )


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _panel(plant: _Building):
    return panel_from_log(
        log_episode(plant, HEAT_PUMP, policy="reset", seed=0, steps=192, step_s=_STEP_S), seed=0
    )


def test_the_forecast_is_read_in_the_panels_units() -> None:
    """Kelvin to Celsius for the band and the air, W/m2 to kW/m2 for the sun, horizon + 1 points
    from now; the held edge is the lowest upper bound after now, and the target the lower bound
    after each step plus the offset."""
    plant = _building()
    plant.initialize("surrogate", 10 * _STEP_S, 0.0)
    forecast = Forecast.read(plant, "surrogate", _SMALL)
    rows = slice(10, 10 + _SMALL.horizon + 1)
    assert np.allclose(forecast.outdoor, plant.covariates[rows, 0])
    assert np.allclose(forecast.solar, plant.covariates[rows, 1])
    assert np.allclose(forecast.lower, plant.covariates[rows, 2])
    assert forecast.ceiling == pytest.approx(24.0)
    assert np.allclose(forecast.target(0.5), plant.covariates[11:15, 2] + 0.5)


def test_a_short_forecast_is_refused() -> None:
    class _Short(_Building):
        def forecast(self, testid, point_names, horizon, interval):
            return {name: [0.0] for name in point_names}

    with pytest.raises(RuntimeError, match="forecast 5 points"):
        Forecast.read(_like(_Short, _building()), "surrogate", _SMALL)


def test_the_two_arms_differ_only_in_what_they_adjust_for(x64) -> None:
    """Same log, same call. The graph's arm is identified; the other asserts its channel. Both
    carry the weather as drivers and the band's upper edge as a held constraint."""
    plant = _building()
    panel = _panel(plant)
    plant.initialize("surrogate", 0.0, 0.0)
    forecast = Forecast.read(plant, "surrogate", _SMALL)
    held = {
        arm: prescription(panel, arm, temp=20.5, forecast=forecast, offset=0.0, design=_SMALL)
        for arm in ARMS
    }
    assert held["adjusted"].certificate.identification == "identified"
    assert held["naive"].certificate.identification == "asserted"
    assert held["naive"].certificate.adjustment.covariates == ()
    for arm in ARMS:
        assert held[arm].plan is not None
        assert held[arm].model_fit.drivers == (OUTDOOR, SOLAR)
        assert held[arm].certificate.barrier_certified_steps is not None


class _Spy(_Building):
    """Records the zone temperature each action was chosen at, and the action applied."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.applied: list[tuple[int, float, float]] = []

    def advance(self, testid, u):
        self.applied.append((self._round, self._temp, float(u[HEAT_PUMP.action_point])))
        return super().advance(testid, u)


def test_each_step_applies_the_first_action_of_a_fresh_prescription(x64) -> None:
    """Receding horizon: the plant gets the first action of the call made from the state it is
    in, with the forecast from the round it is at."""
    plant = _building()
    panel = _panel(plant)
    spy = _like(_Spy, plant)
    design = replace(_SMALL, control_steps=3)
    run_arm_episode(spy, panel, "adjusted", -1.0, design, start_day=4.0)
    assert len(spy.applied) == 3
    reader = _building()
    for round_, temp, action in spy.applied:
        reader.initialize("surrogate", round_ * _STEP_S, 0.0)
        forecast = Forecast.read(reader, "surrogate", design)
        decided = prescription(
            panel, "adjusted", temp=temp, forecast=forecast, offset=-1.0, design=design
        )
        assert math.isclose(action, float(decided.schedule.magnitudes[0, 0]), rel_tol=1e-9)


def _step(
    temp: float = 20.0,
    ceiling: float = 24.0,
    realised: float = 21.1,
    status: str = "converged",
    tube: float | None = 0.2,
    bound: float = 1e-7,
) -> Step:
    return Step(
        temp=temp,
        target=21.0,
        ceiling=ceiling,
        action=0.5,
        predicted=21.0,
        realised=realised,
        tube=tube,
        trusted=2 if tube is not None else 0,
        barrier=4,
        solver_status=status,
        regret_bound=bound,
    )


def test_v2_reads_only_the_calls_made_from_inside_the_band() -> None:
    """A call from above the held edge cannot meet the barrier, so its budget stop is counted
    apart; one from inside the band that the budget stops is V2's."""
    above = [_step(temp=25.0, status="max_iterations") for _ in range(3)]
    inside = [_step() for _ in range(97)]
    summary = episode_summary(above + inside, _SMALL)
    assert summary["budget_stopped"] == 0.0
    assert summary["budget_stopped_outside"] == 3
    assert summary["calls_from_above_band"] == pytest.approx(0.03)
    stopped = episode_summary([*inside[:-1], _step(status="max_iterations")], _SMALL)
    assert stopped["budget_stopped"] == pytest.approx(1 / 97)


def test_the_band_is_read_on_the_plant() -> None:
    steps = [_step(realised=24.5), _step(realised=23.0), _step(realised=24.1), _step()]
    summary = episode_summary(steps, _SMALL)
    assert summary["ended_above_band"] == 0.5
    assert summary["barrier_min"] == 4
    assert summary["within_tolerance"] == 0.25


def _replicate(adjusted, naive, baseline=(2.0, 2.0), stopped=0.0) -> dict:
    def episodes(points):
        return [
            {"kpi": {"tdis_tot": d, "ener_tot": e}, "budget_stopped": stopped} for d, e in points
        ]

    return {
        "baseline": {"tdis_tot": baseline[0], "ener_tot": baseline[1]},
        "episodes": {"adjusted": episodes(adjusted), "naive": episodes(naive)},
    }


_ADJUSTED = [(40.0, 1.0), (10.0, 1.2), (1.0, 1.4), (0.0, 1.6)]


@pytest.mark.parametrize(
    ("factors", "decision"),
    [
        ([1.10, 1.12, 1.08, 1.11, 1.09, 1.10], "CONFIRMED"),
        ([0.90, 0.88, 0.92, 0.89, 0.91, 0.90], "REFUTED"),
        ([1.20, 0.80, 1.10, 0.90, 1.02, 0.97], "INCONCLUSIVE"),
    ],
)
def test_the_gate_is_l81s(factors: list[float], decision: str) -> None:
    """A naive front dearer by ``factor`` at every comfort has excess ``factor - 1``."""
    replicates = [_replicate(_ADJUSTED, [(d, e * f) for d, e in _ADJUSTED]) for f in factors]
    result = verdict(replicates, Design())
    assert result["decision"] == decision
    assert result["stands"]
    assert math.isclose(result["excess_mean"], float(np.mean(factors)) - 1.0, rel_tol=1e-12)


def test_a_budget_stopped_episode_voids_the_verdict() -> None:
    replicates = [_replicate(_ADJUSTED, [(d, 1.1 * e) for d, e in _ADJUSTED]) for _ in range(5)]
    replicates.append(_replicate(_ADJUSTED, _ADJUSTED, stopped=0.02))
    result = verdict(replicates, Design())
    assert not result["v2_optimiser_clean"]
    assert not result["stands"]


def test_the_protocol_end_to_end_on_the_surrogate(x64, tmp_path) -> None:
    """Every stage on a plant that answers as its model, each replicate into its own journal; the
    journals then collect without a plant, and the report renders identically from its JSON."""
    journal = tmp_path / "journals"
    results = run_experiment(_building(), _SMALL, journal=journal, say=lambda _line: None)
    assert sorted(p.name for p in journal.iterdir()) == ["replicate-0.jsonl", "replicate-1.jsonl"]
    assert results["precision"] == "float64"
    for replicate in results["replicates"]:
        assert replicate["fits"]["adjusted"]["identification"] == "identified"
        assert replicate["fits"]["naive"]["identification"] == "asserted"
        assert set(replicate["fits"]["naive"]["driver_gain"]) == {OUTDOOR, SOLAR}
        for arm in ARMS:
            episodes = replicate["episodes"][arm]
            assert [e["offset"] for e in episodes] == list(_SMALL.offsets)
            for episode in episodes:
                assert episode["steps"] == _SMALL.control_steps
                assert "time_rat" not in episode["kpi"]
    collected = run_experiment(None, _SMALL, journal=journal, say=lambda _line: None)
    assert json.dumps(collected) == json.dumps(results)
    text = markdown(results)
    assert "decision" in text
    assert "commit not recorded" in text
    assert markdown(json.loads(json.dumps(results, indent=2))) == text
    same = {**results, "replicates": results["replicates"]}
    assert "Against L8.1" in markdown(results, same)
    assert "at `3c75870`" in markdown({**results, "chc_commit": "3c75870"})


def test_a_resumed_replicate_refuses_a_different_log(x64, tmp_path) -> None:
    journal = tmp_path / "replicate-0.jsonl"
    run_replicate(_building(), 0, _SMALL, journal=journal, say=lambda _line: None)
    lines = journal.read_text().splitlines()
    run_replicate(_building(), 0, _SMALL, journal=journal, say=lambda _line: None)
    assert journal.read_text().splitlines() == lines  # nothing re-run, nothing appended
    journal.write_text("\n".join(lines[:-1]) + "\n")  # one episode short: the replicate re-logs
    with pytest.raises(RuntimeError, match="not reproducing"):
        run_replicate(_building(seed=1), 0, _SMALL, journal=journal, say=lambda _line: None)
    with pytest.raises(RuntimeError, match="no plant"):
        run_replicate(None, 0, _SMALL, journal=journal, say=lambda _line: None)


def test_main_refuses_float32(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["boptest_rerun", "--url", "http://127.0.0.1:1"])
    with jax.enable_x64(False), pytest.raises(SystemExit, match="JAX_ENABLE_X64"):
        boptest_rerun.main()


def test_main_refuses_to_run_replicates_without_a_service(monkeypatch, x64) -> None:
    argv = ["boptest_rerun", "--url", "http://127.0.0.1:1", "--replicates", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="no BOPTEST-Service"):
        boptest_rerun.main()


def test_main_takes_sigint_back_from_a_shell_that_ignored_it(monkeypatch, x64) -> None:
    """The recipe's workers are background jobs, which start with SIGINT ignored; the timeout's
    SIGINT must still raise, so that an episode's `finally` stops its test."""
    argv = ["boptest_rerun", "--url", "http://127.0.0.1:1", "--replicates", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with pytest.raises(RuntimeError, match="no BOPTEST-Service"):
            boptest_rerun.main()
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, previous)


@pytest.mark.skipif(not _URL, reason="set BOPTEST_URL to a running BOPTEST-Service")
def test_live_prescriptions_reach_the_emulator(x64) -> None:
    """Plumbing only, in an unscored part of the year: a short log, then two steps per arm."""
    if not _URL or not is_available(_URL):
        pytest.skip("BOPTEST_URL is set but the service is unreachable")
    client = BOPTestClient(_URL, timeout=300.0)
    log = log_episode(
        client, HEAT_PUMP, policy="reset", seed=0, steps=96, step_s=_STEP_S, start_time=300 * DAY_S
    )
    panel = panel_from_log(log, seed=0)
    design = replace(Design(), control_steps=2, horizon=4)
    for arm in ARMS:
        ran = run_arm_episode(client, panel, arm, 0.0, design, start_day=303.0)
        assert ran["steps"] == 2
        assert {"tdis_tot", "ener_tot"} <= set(ran["kpi"])
        assert "time_rat" not in ran["kpi"]

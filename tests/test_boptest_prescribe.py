"""L8.1's harness offline, on a surrogate plant; the live run is the experiment, not a test."""

import json
import math
import os
import sys
from dataclasses import replace

import jax
import numpy as np
import pytest

from causaldyn_bench import boptest_prescribe
from causaldyn_bench.boptest import BOPTestClient, is_available
from causaldyn_bench.boptest_capped import SurrogatePlant, ZoneModel
from causaldyn_bench.boptest_causal import HEAT_PUMP, log_episode
from causaldyn_bench.boptest_prescribe import (
    ARMS,
    BOUND,
    DAY_S,
    MODULATION,
    OUTDOOR,
    SOLAR,
    ZONE,
    Design,
    Step,
    adjustment,
    energy_at,
    episode_summary,
    front,
    graph,
    markdown,
    matched,
    panel_from_log,
    prescription,
    run_arm_episode,
    run_experiment,
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
    """The capped harness's surrogate, with the two KPIs the gate reads and a built-in controller.

    ``tdis_tot`` integrates the zone's shortfall below the comfort bound and ``ener_tot`` the
    modulation, both in hours from ``initialize`` on: BOPTEST's definitions up to the heat pump's
    rated power, which the gate's ratio cancels. A call without an overwrite gets a proportional
    controller aiming half a kelvin above the bound, standing in for BOPTEST's baseline. It reports
    ``time_rat`` too, so a test can see the harness drop it.
    """

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
        self._tdis += hours * max(0.0, bound - self._temp)
        return measurements

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


@pytest.fixture
def x64():
    """``prescribe`` fits in JAX, and the live run refuses anything but float64; so do these."""
    with jax.enable_x64(True):
        yield


def test_the_graph_resolves_to_the_reset_curves_inputs() -> None:
    """The adjusted arm's whole difference is this set; occupancy is latent and never in it."""
    resolved = graph().adjustment_set(treatment=MODULATION, outcome=ZONE)
    assert resolved.status == "identified"
    assert set(resolved.covariates) == {OUTDOOR, SOLAR, BOUND}
    assert adjustment("naive") == ()


def test_a_log_starts_where_it_is_told_and_the_panel_is_in_raw_units(x64) -> None:
    """``start_time`` reaches the plant, and the panel un-standardises what the log standardised."""
    plant = _building()
    log = log_episode(
        plant,  # ty: ignore[invalid-argument-type]
        HEAT_PUMP,
        policy="reset",
        seed=0,
        steps=48,
        step_s=_STEP_S,
        start_time=2 * DAY_S,
    )
    panel = panel_from_log(log, seed=0)
    start = round(2 * DAY_S / _STEP_S)
    assert np.allclose(np.asarray(panel[OUTDOOR]), plant.covariates[start : start + 48, 0])
    assert np.allclose(np.asarray(panel[BOUND]), plant.covariates[start : start + 48, 2])
    assert np.allclose(np.asarray(panel[ZONE]), np.asarray(log.zone).ravel())


def test_the_two_arms_differ_only_in_what_they_adjust_for(x64) -> None:
    """Same log, same call: the graph's arm is identified through the reset curve's inputs, the
    other asserts its channel with nothing adjusted, and both return a schedule."""
    log = log_episode(_building(), HEAT_PUMP, policy="reset", seed=0, steps=192, step_s=_STEP_S)  # ty: ignore[invalid-argument-type]
    panel = panel_from_log(log, seed=0)
    held = {arm: prescription(panel, arm, temp=20.5, target=21.0, design=_SMALL) for arm in ARMS}
    assert held["adjusted"].certificate.identification == "identified"
    assert set(held["adjusted"].certificate.adjustment.covariates) == {OUTDOOR, SOLAR, BOUND}
    assert held["naive"].certificate.identification == "asserted"
    assert held["naive"].certificate.adjustment.covariates == ()
    assert held["naive"].certificate.trustworthy_steps == 0  # nothing bounds an asserted channel
    for arm in ARMS:
        assert held[arm].plan is not None


class _Spy(_Building):
    """Records the zone temperature each action was chosen at, and the action applied."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.applied: list[tuple[float, float]] = []

    def advance(self, testid, u):
        self.applied.append((self._temp, float(u[HEAT_PUMP.action_point])))
        return super().advance(testid, u)


def test_each_step_applies_the_first_action_of_a_fresh_prescription(x64) -> None:
    """Receding horizon: the plant gets the first action of the call made from the state it is
    in. From 19.5 C towards 20 C the plan heats hard and then holds, so its first and last
    actions differ and applying any other than the first is visible."""
    plant = _building()
    log = log_episode(plant, HEAT_PUMP, policy="reset", seed=0, steps=192, step_s=_STEP_S)  # ty: ignore[invalid-argument-type]
    panel = panel_from_log(log, seed=0)
    spy = _Spy(
        model=plant.model,
        effect=plant.effect,
        covariates=plant.covariates,
        temps=plant.temps,
        noise=plant.noise,
        step_s=_STEP_S,
    )
    run_arm_episode(spy, panel, "adjusted", -1.0, replace(_SMALL, control_steps=3), start_day=4.0)
    assert len(spy.applied) == 3
    for temp, action in spy.applied:  # the bound is 21 C all morning, so the target is 20 C
        plan = prescription(panel, "adjusted", temp=temp, target=20.0, design=_SMALL)
        magnitudes = np.asarray(plan.schedule.magnitudes)[:, 0]
        assert math.isclose(action, float(magnitudes[0]), rel_tol=1e-9)  # K round trip only
    first = prescription(panel, "adjusted", temp=spy.applied[0][0], target=20.0, design=_SMALL)
    assert not np.isclose(first.schedule.magnitudes[0, 0], first.schedule.magnitudes[-1, 0])


def test_the_front_drops_dominated_points_and_reads_the_chord() -> None:
    points = [(4.0, 1.0), (0.0, 3.0), (3.0, 2.5), (2.0, 2.0)]
    assert front(points) == [(0.0, 3.0), (2.0, 2.0), (4.0, 1.0)]  # (3, 2.5) is dominated
    assert energy_at(points, 3.0) == pytest.approx(1.5)
    assert energy_at(points, 0.0) == 3.0
    assert energy_at(points, 4.0) == 1.0


@pytest.mark.parametrize("level", [-0.5, 4.5])
def test_no_energy_is_read_where_the_front_does_not_reach(level: float) -> None:
    """Outside the measured discomfort range there is no reading, not an extrapolation."""
    assert energy_at([(0.0, 3.0), (2.0, 2.0), (4.0, 1.0)], level) is None


def _replicate(adjusted, naive, baseline=(2.0, 2.0), stopped=0.0) -> dict:
    """One replicate's episodes from ``(tdis, ener)`` points, in offset order."""

    def episodes(points):
        return [
            {"kpi": {"tdis_tot": d, "ener_tot": e}, "budget_stopped": stopped} for d, e in points
        ]

    return {
        "baseline": {"tdis_tot": baseline[0], "ener_tot": baseline[1]},
        "episodes": {"adjusted": episodes(adjusted), "naive": episodes(naive)},
    }


_ADJUSTED = [(40.0, 1.0), (10.0, 1.2), (1.0, 1.4), (0.0, 1.6)]


def _scaled(factor: float) -> list[tuple[float, float]]:
    return [(d, e * factor) for d, e in _ADJUSTED]


@pytest.mark.parametrize(
    ("factors", "decision"),
    [
        ([1.10, 1.12, 1.08, 1.11, 1.09, 1.10], "CONFIRMED"),
        ([0.90, 0.88, 0.92, 0.89, 0.91, 0.90], "REFUTED"),
        ([1.20, 0.80, 1.10, 0.90, 1.02, 0.97], "INCONCLUSIVE"),
    ],
)
def test_the_gate_reads_the_interval(factors: list[float], decision: str) -> None:
    """A naive front dearer by ``factor`` at every comfort has excess ``factor - 1``."""
    replicates = [_replicate(_ADJUSTED, _scaled(f)) for f in factors]
    result = verdict(replicates, Design())
    assert result["decision"] == decision
    assert result["stands"]
    assert math.isclose(result["excess_mean"], float(np.mean(factors)) - 1.0, rel_tol=1e-12)


def test_heating_less_cannot_win_the_gate() -> None:
    """R12. The naive arm below spends less energy than the adjusted arm at EVERY offset, and
    only by being colder: read at the baseline's comfort it is the dearer one."""
    naive = [(60.0, 0.9), (30.0, 1.1), (5.0, 1.3), (0.5, 1.5)]
    assert all(n[1] < a[1] for n, a in zip(naive, _ADJUSTED, strict=True))
    row = matched(_replicate(_ADJUSTED, naive))
    assert math.isclose(row["energy"]["adjusted"], 1.4 - 0.2 / 9.0)
    assert math.isclose(row["energy"]["naive"], 1.5 - 0.2 / 3.0)
    assert row["excess"] > 0.0
    replicates = [_replicate(_ADJUSTED, naive) for _ in range(4)]
    assert verdict(replicates, Design())["decision"] == "CONFIRMED"


def test_a_front_that_never_gets_as_comfortable_voids_the_verdict() -> None:
    """V1: an arm cheaper than everything because it never heats enough has no reading, and a
    verdict without it does not stand -- whatever the other replicates say."""
    lazy = [(40.0, 0.2), (30.0, 0.3), (20.0, 0.4)]
    replicates = [_replicate(_ADJUSTED, _scaled(1.1)) for _ in range(5)]
    replicates.append(_replicate(_ADJUSTED, lazy))
    result = verdict(replicates, Design())
    assert result["per_replicate"][-1]["excess"] is None
    assert not result["v1_comfort_matched"]
    assert not result["stands"]


def test_a_front_read_at_zero_energy_has_no_ratio() -> None:
    """A window the arm needs no energy for at the baseline's comfort asks nothing of adjustment."""
    idle = [(2.0, 0.0), (0.0, 0.5)]
    row = matched(_replicate(idle, _ADJUSTED))
    assert row["energy"]["adjusted"] == 0.0
    assert row["excess"] is None
    assert not row["reached"]


def test_a_budget_stopped_episode_voids_the_verdict() -> None:
    """V2: more than 1% of one episode's solves stopped by the iteration budget."""
    replicates = [_replicate(_ADJUSTED, _scaled(1.1)) for _ in range(5)]
    replicates.append(_replicate(_ADJUSTED, _scaled(1.1), stopped=0.02))
    result = verdict(replicates, Design())
    assert result["decision"] == "CONFIRMED"
    assert not result["v2_optimiser_clean"]
    assert not result["stands"]


def test_the_saving_against_the_baseline_is_read_at_its_comfort() -> None:
    replicates = [_replicate(_ADJUSTED, _scaled(1.1), baseline=(1.0, 2.8)) for _ in range(3)]
    saving = verdict(replicates, Design())["saving_against_baseline"]
    assert math.isclose(saving["adjusted"]["mean"], 0.5)  # 1.4 at tdis 1.0, against 2.8
    assert math.isclose(saving["naive"]["mean"], 1.0 - 1.54 / 2.8)


def _step(predicted: float, realised: float, tube: float | None, bound: float) -> Step:
    return Step(
        temp=20.0,
        target=21.0,
        action=0.5,
        predicted=predicted,
        realised=realised,
        tube=tube,
        trusted=2 if tube is not None else 0,
        solver_status="converged",
        regret_bound=bound,
    )


def test_the_episode_summary_scores_the_plan_against_the_plant() -> None:
    """The one-step error is the plant against the plan's own next temperature; the tube and the
    tolerance are read against its size, and an infinite regret bound is counted, not maxed."""
    steps = [_step(21.0, 21.1, 0.2, 1e-7), _step(21.0, 20.7, 0.2, math.inf)]
    summary = episode_summary(steps, _SMALL)
    assert math.isclose(summary["error_rms"], math.sqrt((0.1**2 + 0.3**2) / 2))
    assert math.isclose(summary["error_mean"], -0.1)
    assert summary["within_tube"] == 0.5
    assert summary["within_tolerance"] == 1.0
    assert summary["regret_bound_max"] == 1e-7
    assert summary["regret_uncertified"] == 0.5
    unbounded = episode_summary([_step(21.0, 21.1, None, 1e-7)], _SMALL)
    assert unbounded["tube_mean"] is None
    assert unbounded["within_tube"] is None


def test_the_protocol_end_to_end_on_the_surrogate(x64) -> None:
    """Every stage on a plant that answers as its model: both arms over the grid in every window,
    the wall-clock KPI dropped, and a report that renders identically from its own JSON."""
    results = run_experiment(_building(), _SMALL, say=lambda _line: None)
    assert results["precision"] == "float64"
    assert len(results["replicates"]) == _SMALL.replicates
    for replicate in results["replicates"]:
        assert replicate["fits"]["adjusted"]["identification"] == "identified"
        assert replicate["fits"]["naive"]["identification"] == "asserted"
        assert set(replicate["baseline"]) == {"tdis_tot", "ener_tot"}
        for arm in ARMS:
            episodes = replicate["episodes"][arm]
            assert [e["offset"] for e in episodes] == list(_SMALL.offsets)
            for episode in episodes:
                assert episode["steps"] == _SMALL.control_steps
                assert "time_rat" not in episode["kpi"]
        assert all(e["trusted_max"] == 0 for e in replicate["episodes"]["naive"])
    text = markdown(results)
    assert "decision" in text
    assert markdown(json.loads(json.dumps(results, indent=2))) == text


def test_a_resumed_run_reuses_the_journal_and_refuses_a_different_log(x64, tmp_path) -> None:
    journal = tmp_path / "journal.jsonl"
    first = run_experiment(_building(), _SMALL, journal=journal, say=lambda _line: None)
    lines = journal.read_text().splitlines()
    second = run_experiment(_building(), _SMALL, journal=journal, say=lambda _line: None)
    assert journal.read_text().splitlines() == lines  # nothing re-run, nothing appended
    assert json.dumps(second) == json.dumps(first)  # NaN-safe, unlike ==
    journal.write_text("\n".join(lines[:-1]) + "\n")  # one episode short: its replicate re-logs
    with pytest.raises(RuntimeError, match="not reproducing"):
        run_experiment(_building(seed=1), _SMALL, journal=journal, say=lambda _line: None)


def test_main_refuses_without_a_service(monkeypatch, x64) -> None:
    monkeypatch.setattr(sys, "argv", ["boptest_prescribe", "--url", "http://127.0.0.1:1"])
    with pytest.raises(RuntimeError, match="no BOPTEST-Service"):
        boptest_prescribe.main()


def test_main_refuses_float32(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["boptest_prescribe", "--url", "http://127.0.0.1:1"])
    with jax.enable_x64(False), pytest.raises(SystemExit, match="JAX_ENABLE_X64"):
        boptest_prescribe.main()


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

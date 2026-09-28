"""L8.1 -- the BOPTEST heat pump through ``chc.prescribe``, end to end, pre-registered.

Handed the log a weather-compensated controller left behind and a graph of why it acted, does the
façade return a heating schedule that, run closed loop on ``bestest_hydronic_heat_pump``, heats for
less energy at the same comfort than the identical call told nothing about the confounding? This
docstring is the pre-registration. It was committed before any scored episode ran; the scored run
is :func:`main` at the defaults of :class:`Design`, and changing a default afterwards is a new
experiment.

**Design.** Replicate ``k = 0..5`` logs 960 half-hour steps of Track D-causal's ``reset`` policy
(outdoor-reset curve, comfort feedback, held exploration noise; seed ``k``) from day ``7k`` of the
year, with no warm-up -- replicate 0's log is the same call as Track D-causal's seed-0 reset log.
The log becomes a one-unit :class:`chc.Panel` of 959 transitions in Celsius and kW/m2. The scored
window is the 7 days (336 steps) that follow the log, from day ``7k + 20``, each episode started
after BOPTEST's own controller has run the day before it. In that window run BOPTEST's built-in
baseline once, then both arms at each of seven target offsets.

**Arms.** At every step both arms make the same call, from the zone temperature just measured::

    prescribe(panel, levers=[Lever("modulation", 0, 1, unit_cost=0)],
              target=Target("zone", value=target), horizon=16, dt=0.5, tolerance=0.5,
              x0=[zone], adjustment=<arm>)

and apply only the first action of the schedule (receding horizon). ``adjusted`` passes the
graph of :data:`EDGES` -- the reset curve reads the outdoor temperature, the comfort feedback reads
the bound, occupancy (unmeasured) sets the bound and the internal gains -- which the façade
resolves to ``{outdoor, solar, bound}``. ``naive`` passes ``()``. Nothing else differs. The target
is the highest lower comfort bound BOPTEST forecasts over the plan's 16 steps, plus the offset
(-0.5, -0.25, 0, +0.25, +0.5, +1, +1.5 K). The lever is free, so the offset is the only knob that
trades comfort for energy, as the margin sweep does in Track D-causal.

**Metric.** Per replicate, an arm's seven ``(tdis_tot, ener_tot)`` points -- BOPTEST's K h and
kWh/m2 -- are its operating points, and its front is their non-dominated subset. Its energy at the
built-in baseline's ``tdis_tot`` in the same window is read on that front, linearly between the two
points that bracket it: running each of the two settings for a share of the week buys,
approximately, that mixture of both KPIs, so the chord is a reading the arm can reach, if not the
least it could. ``excess = E_naive / E_adjusted - 1``.

**Gate.** The mean excess over replicates, with a two-sided 95% Student-t interval. CONFIRMED if
the interval lies above 0 (at the baseline's comfort, adjusting makes the schedule cheaper),
REFUTED if it lies below 0, INCONCLUSIVE if it contains 0. Two validity checks; either failing
voids the decision, which is then printed with DOES NOT STAND:

- V1: both arms' fronts bracket the baseline's discomfort in every replicate, at a positive
  energy. A front that never gets as comfortable as the baseline has no energy at its comfort,
  none is extrapolated, and a zero reading leaves the ratio undefined, so it counts as none.
- V2: no episode has more than 1% of its calls stopped by the solver's iteration budget.
  ``no_progress`` -- the zero guess not improved on -- is reported and not gated, because the
  certificate's regret bound prices those plans too.

**R12.** Energy alone is won by heating less. Here it cannot be: energy is read at the baseline's
discomfort, so an arm that saves energy by heating less is read further along its own front, and
one whose front stops short of the baseline's comfort has no reading at all and voids the verdict
(V1). The dual -- discomfort at the baseline's energy -- is reported beside it and not gated.

**Against the baseline** (secondary, descriptive, no decision): each arm's saving
``1 - E_arm / E_baseline`` at the baseline's comfort, with the same interval.

**Certificate against plant** (reported, not gated). Per call: the trustworthy prefix, the error
tube one step ahead, the solver status and the regret bound. Per step, from the plant: the one-step
error against the plan's own next temperature, and whether it stayed inside the tube and inside the
tolerance. Only the first step of a plan is ever applied, so it is the only step the plant checks.

**Size.** 6 replicates x 2 arms x 7 offsets = 84 controlled weeks, plus 6 baseline weeks and six
20-day logs, sized to the emulator time available and not for power: nothing here predicts the
effect's size, so no power figure is claimed. Neighbouring logs share 13 of their 20 days, so the
replicates are not independent draws of the log, and the interval is optimistic to that extent;
the windows do not overlap.

**Pilot, unscored.** One log (seed 99, days 320-340) and one window (days 340-347), neither in the
scored set, adjusted arm only, lever unit cost 0, offsets -1, 0 and +1 K. It fixed the unit cost
(no call stopped by the budget), the horizon, the tolerance and the grid: the baseline's discomfort
fell between the adjusted arm's -1 and 0, and +1 left none. The grid runs past +1 because the naive
arm was not run closed loop before this was committed, and a front that stops short voids the
verdict.

**Precision.** float64 only. :func:`main` refuses without ``JAX_ENABLE_X64=1``, and the precision is
written beside every number.

**What the façade cannot state here** -- findings, reported rather than worked around:

1. No bound on the steered state. ``prescribe`` refuses a constraint on its target column, so
   comfort gets no barrier and no gamma*: the certificate bounds the model's error, never the zone.
2. No exogenous driver in the drift. The fitted drift is a function of the zone temperature alone,
   so the weather the adjusted arm partials out never enters the plan's forecast.
3. One target per call. The time-varying comfort bound reaches the plan only through re-planning,
   so the target is the highest bound in view, which pre-heats up to one horizon early.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from chc import CausalGraph, Lever, Panel, Target, prescribe
from chc.decision import Prescription

from causaldyn_bench.boptest import DEFAULT_URL, BOPTestClient, baseline_controller, is_available
from causaldyn_bench.boptest import run_episode as run_baseline_episode
from causaldyn_bench.boptest_capped import Plant
from causaldyn_bench.boptest_causal import (
    HEAT_PUMP,
    KELVIN,
    LOWER_SETP,
    LoggedEpisode,
    ThermalFit,
    log_episode,
)

DAY_S = 86_400.0

# Student-t 0.975 quantiles by degrees of freedom, for the gate's two-sided 95% interval. Tabulated
# so the bench does not import scipy, which it does not declare; Maxima's quantile_student_t and
# scipy.stats.t.ppf agree on every entry to eight significant figures, and six decimals are kept.
T975 = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
    10: 2.228139,
}
# BOPTEST's KPIs, less ``time_rat``: that one is wall time per simulated time, which is neither
# deterministic nor measured faithfully on this machine.
KPIS = (
    "tdis_tot",
    "idis_tot",
    "ener_tot",
    "cost_tot",
    "emis_tot",
    "pele_tot",
    "pgas_tot",
    "pdih_tot",
)

Arm = Literal["adjusted", "naive"]
ARMS: tuple[Arm, ...] = ("adjusted", "naive")

ZONE, MODULATION = "zone", "modulation"
OUTDOOR, SOLAR, BOUND = "outdoor", "solar", "bound"
OCCUPANCY = "occupancy"  # latent: nothing in the log measures it

EDGES = (
    (OUTDOOR, MODULATION),  # the outdoor-reset curve: this is the confounding
    (OUTDOOR, ZONE),  # heat lost through the envelope
    (SOLAR, ZONE),  # solar gains
    (BOUND, MODULATION),  # the logging policy's comfort feedback aims at the bound
    (OCCUPANCY, BOUND),  # the schedule sets the bound ...
    (OCCUPANCY, ZONE),  # ... and the internal gains
    (MODULATION, ZONE),
)


def graph() -> CausalGraph:
    """What the logging policy and the building are assumed to do, as the façade reads it."""
    return CausalGraph.from_edges(EDGES, latent=(OCCUPANCY,))


def adjustment(arm: Arm) -> CausalGraph | tuple[str, ...]:
    """The one argument the two arms differ in: the graph, or nothing asserted."""
    return graph() if arm == "adjusted" else ()


@dataclass(frozen=True)
class Design:
    """The pre-registered constants. Changing a default after the scored run is a new experiment."""

    step_s: float = 1800.0
    log_steps: int = 960  # 20 days of the reset policy per replicate
    control_steps: int = 336  # a 7-day window right after the log
    replicates: int = 6
    stride_days: int = 7  # replicate k logs from day stride * k
    warmup_days: float = 1.0  # BOPTEST's own controller runs this long before a window starts
    horizon: int = 16
    offsets: tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5)
    unit_cost: float = 0.0
    tolerance: float = 0.5
    max_budget_share: float = 0.01

    @property
    def dt(self) -> float:
        return self.step_s / 3600.0

    def log_day(self, replicate: int) -> float:
        return float(self.stride_days * replicate)

    def window_day(self, replicate: int) -> float:
        return self.log_day(replicate) + self.log_steps * self.step_s / DAY_S


DESIGN = Design()


def panel_from_log(log: LoggedEpisode, *, seed: int) -> Panel:
    """The reset log as the long-format panel ``prescribe`` reads, in Celsius and kW/m2.

    One unit, one row per half-hour. The log's last ``zone_next`` has no row after it to sit in,
    so the panel carries ``log_steps - 1`` transitions rather than inventing a final action.
    """
    weather = np.asarray(log.weather) * np.asarray(log.weather_scale) + np.asarray(log.weather_mean)
    rows = int(np.asarray(log.zone).shape[0])
    return Panel.from_frame(
        {
            "log": np.zeros(rows, dtype=np.int64),
            "step": np.arange(rows, dtype=np.int64),
            ZONE: np.asarray(log.zone, dtype=np.float64).ravel(),
            MODULATION: np.asarray(log.action, dtype=np.float64).ravel(),
            OUTDOOR: weather[:, 0],
            SOLAR: weather[:, 1],
            BOUND: weather[:, 2],
        },
        unit="log",
        time="step",
        seed=seed,
    )


def prescription(
    panel: Panel, arm: Arm, *, temp: float, target: float, design: Design
) -> Prescription:
    """The one ``prescribe`` call both arms make, from the zone temperature measured now."""
    return prescribe(
        panel,
        levers=[
            Lever(MODULATION, HEAT_PUMP.action_lo, HEAT_PUMP.action_hi, unit_cost=design.unit_cost)
        ],
        target=Target(ZONE, value=target),
        horizon=design.horizon,
        adjustment=adjustment(arm),
        dt=design.dt,
        tolerance=design.tolerance,
        x0=jnp.asarray([temp]),
    )


def fit_summary(panel: Panel, arm: Arm, design: Design) -> dict[str, Any]:
    """What ``prescribe`` identified from the log, read through the harness's own definitions.

    ``ThermalFit`` carries the reporting conventions of Track D-causal -- the decay read at the
    log's operating action, the authority at 21 C, the 8-hour step response -- so the two tracks
    report one quantity by one definition. Its ``weather_drift`` is empty because the façade's
    drift has no weather term; nothing here evaluates the rate.
    """
    zone = np.asarray(panel[ZONE], dtype=np.float64)
    held = prescription(panel, arm, temp=float(zone[-1]), target=21.0, design=design)
    fit = held.model_fit
    residual = fit.residual
    drift = np.asarray(residual.drift)[0]
    channel = np.asarray(residual.channel)[0, 0]
    thermal = ThermalFit(
        drift=(float(drift[0]), float(drift[1])),
        weather_drift=(),
        channel=(float(channel[0]), float(channel[1])),
        method=fit.method,
        identified=fit.identified,
        adjusted=arm == "adjusted",
        policy="reset",
        action_mean=float(np.mean(np.asarray(panel[MODULATION], dtype=np.float64))),
        action_residual_variance=fit.action_residual_variance,
        nuisance_r2_action=fit.nuisance_r2_action,
        channel_error=fit.channel_error,
    )
    certificate = held.certificate
    return {
        "method": fit.method,
        "identification": certificate.identification,
        "adjusted_for": list(certificate.adjustment.covariates),
        "channel": list(thermal.channel),
        "drift": list(thermal.drift),
        "action_mean": thermal.action_mean,
        "decay": thermal.decay(),
        "decay_at_off": thermal.decay(HEAT_PUMP.action_lo),
        "decay_at_full": thermal.decay(HEAT_PUMP.action_hi),
        "stable_over_box": thermal.stable_over(HEAT_PUMP.action_lo, HEAT_PUMP.action_hi),
        "authority_21": thermal.authority(21.0),
        "step_response_8h": thermal.step_response(),
        "channel_error": fit.channel_error,
        "drift_error": fit.drift_error,
        "integrator_defect": fit.integrator_defect,
        "overlap": fit.action_residual_variance,
        "nuisance_r2_action": fit.nuisance_r2_action,
        "nuisance_r2_state": fit.nuisance_r2_state,
        "report": held.report(),
    }


@dataclass(frozen=True)
class Step:
    """One control step: what the prescription said, and what the plant then did."""

    temp: float
    target: float
    action: float
    predicted: float  # the plan's own next zone temperature
    realised: float
    tube: float | None  # the certificate's error radius one step ahead; None when not evaluated
    trusted: int
    solver_status: str | None
    regret_bound: float | None


def run_arm_episode(
    plant: Plant,
    panel: Panel,
    arm: Arm,
    offset: float,
    design: Design,
    *,
    start_day: float,
) -> dict[str, Any]:
    """One week of receding horizon: a fresh ``prescribe`` from every measured state.

    The target is the highest comfort bound BOPTEST forecasts over the plan's horizon plus
    ``offset``, so a plan starts heating when occupancy comes into view; only the first action of
    each schedule is applied.
    """
    testid = plant.select(HEAT_PUMP.testcase)
    steps: list[Step] = []
    try:
        plant.set_step(testid, design.step_s)
        measurements = plant.initialize(testid, start_day * DAY_S, design.warmup_days * DAY_S)
        for _ in range(design.control_steps):
            bounds = plant.forecast(
                testid, [LOWER_SETP], design.horizon * design.step_s, design.step_s
            )[LOWER_SETP]
            target = max(bounds[1 : design.horizon + 1]) - KELVIN + offset
            temp = measurements[HEAT_PUMP.zone_point] - KELVIN
            decided = prescription(panel, arm, temp=temp, target=target, design=design)
            plan = decided.plan
            if plan is None:  # the graph identifies the effect by construction; see the tests
                raise RuntimeError(f"{arm}: prescribe returned no schedule at {temp:.3f} C")
            action = float(decided.schedule.magnitudes[0, 0])
            measurements = plant.advance(testid, HEAT_PUMP.overwrite(action))
            tube = plan.uncertainty_tube
            steps.append(
                Step(
                    temp=float(temp),
                    target=float(target),
                    action=action,
                    predicted=float(plan.trajectory[1, 0]),
                    realised=float(measurements[HEAT_PUMP.zone_point] - KELVIN),
                    tube=None if tube is None else float(tube[1]),
                    trusted=decided.certificate.trustworthy_steps,
                    solver_status=decided.certificate.solver_status,
                    regret_bound=decided.certificate.regret_bound,
                )
            )
        kpi = plant.kpi(testid) or {}
    finally:
        plant.stop(testid)
    return {"kpi": _numeric(kpi), **episode_summary(steps, design)}


def _numeric(kpi: Mapping[str, Any]) -> dict[str, float]:
    return {k: float(kpi[k]) for k in KPIS if isinstance(kpi.get(k), int | float)}


def episode_summary(steps: Sequence[Step], design: Design) -> dict[str, Any]:
    """The certificate beside the plant, per episode. A pure function of the steps."""
    action = np.array([s.action for s in steps])
    error = np.array([s.realised - s.predicted for s in steps])
    tubes = [s.tube for s in steps]
    evaluated = all(t is not None for t in tubes)
    tube = np.array([t for t in tubes if t is not None])
    statuses = [s.solver_status for s in steps]
    bounds = [s.regret_bound for s in steps if s.regret_bound is not None]
    finite = [b for b in bounds if math.isfinite(b)]
    tol = 1e-6 * (HEAT_PUMP.action_hi - HEAT_PUMP.action_lo)
    return {
        "steps": len(steps),
        "saturated": float(
            np.mean((action <= HEAT_PUMP.action_lo + tol) | (action >= HEAT_PUMP.action_hi - tol))
        ),
        "mean_action": float(np.mean(action)),
        "budget_stopped": float(np.mean([s == "max_iterations" for s in statuses])),
        "no_progress": float(np.mean([s == "no_progress" for s in statuses])),
        "trusted_min": int(min(s.trusted for s in steps)),
        "trusted_max": int(max(s.trusted for s in steps)),
        "tube_mean": float(np.mean(tube)) if evaluated else None,
        "error_rms": float(np.sqrt(np.mean(error**2))),
        "error_max": float(np.max(np.abs(error))),
        "error_mean": float(np.mean(error)),
        "within_tube": float(np.mean(np.abs(error) <= tube)) if evaluated else None,
        "within_tolerance": float(np.mean(np.abs(error) <= design.tolerance)),
        "regret_bound_max": float(max(finite)) if finite else None,
        # inf is the certificate declining: the objective was not convex over the box
        "regret_uncertified": float(np.mean([not math.isfinite(b) for b in bounds]))
        if bounds
        else None,
    }


# ---- the gate ---------------------------------------------------------------------------------


def front(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """The non-dominated ``(discomfort, energy)`` points, sorted by discomfort (so energy falls)."""
    unique = sorted(set(points))
    return [p for p in unique if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in unique)]


def energy_at(points: Sequence[tuple[float, float]], discomfort: float) -> float | None:
    """Energy on the arm's front at ``discomfort``, or ``None`` where the front does not reach it.

    Linear between the two front points that bracket it: running each of the two settings for a
    share of the week buys, approximately, that mixture of both KPIs, so the chord is attainable.
    ``None`` rather than an extrapolation when every point is on one side -- a front that never gets
    as comfortable as the baseline has no energy at the baseline's comfort, and one that is always
    more comfortable has not been measured where it would be cheapest.
    """
    edge = front(points)
    if not edge or not edge[0][0] <= discomfort <= edge[-1][0]:
        return None
    for (d_lo, e_lo), (d_hi, e_hi) in itertools.pairwise(edge):
        if d_lo <= discomfort <= d_hi:
            return e_lo + (e_hi - e_lo) * (discomfort - d_lo) / (d_hi - d_lo)
    return edge[0][1]  # a one-point front, exactly at the level asked for


def discomfort_at(points: Sequence[tuple[float, float]], energy: float) -> float | None:
    """The dual: discomfort on the front at ``energy``. Descriptive, not gated."""
    return energy_at([(e, d) for d, e in points], energy)


def matched(replicate: Mapping[str, Any]) -> dict[str, Any]:
    """Each arm's energy at the baseline's discomfort, and the naive arm's excess over it."""
    baseline = replicate["baseline"]
    level = baseline["tdis_tot"]
    points = {
        arm: [(e["kpi"]["tdis_tot"], e["kpi"]["ener_tot"]) for e in replicate["episodes"][arm]]
        for arm in ARMS
    }
    energy = {arm: energy_at(points[arm], level) for arm in ARMS}
    comfort = {arm: discomfort_at(points[arm], baseline["ener_tot"]) for arm in ARMS}
    ratio = _ratio(energy["naive"], energy["adjusted"])
    return {
        "level": level,
        "energy": energy,
        "comfort_at_baseline_energy": comfort,
        "reached": ratio is not None,
        "excess": None if ratio is None else ratio - 1.0,
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """``None`` unless both readings exist and the denominator is positive: a zero energy makes
    the ratio undefined, so it counts as no reading rather than an infinite one."""
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def interval(values: Sequence[float]) -> tuple[float, float, float]:
    """Mean and two-sided 95% Student-t interval; unbounded below two values."""
    data = np.asarray(values, dtype=np.float64)
    n = int(data.size)
    if n < 2:
        return (float(data.mean()) if n else math.nan), -math.inf, math.inf
    mean = float(data.mean())
    half = T975[n - 1] * float(data.std(ddof=1)) / math.sqrt(n)
    return mean, mean - half, mean + half


def verdict(replicates: Sequence[Mapping[str, Any]], design: Design) -> dict[str, Any]:
    """The pre-registered gate, applied to whatever the episodes produced."""
    rows = [matched(r) for r in replicates]
    reached = all(row["reached"] for row in rows)
    values = [row["excess"] for row in rows if row["excess"] is not None]
    n = len(values)
    mean, low, high = interval(values)
    decision = "CONFIRMED" if low > 0.0 else "REFUTED" if high < 0.0 else "INCONCLUSIVE"
    saving = {}
    for arm in ARMS:
        ratios = [
            _ratio(row["energy"][arm], r["baseline"]["ener_tot"])
            for row, r in zip(rows, replicates, strict=True)
        ]
        shares = [1.0 - ratio for ratio in ratios if ratio is not None]
        centre, lo, hi = interval(shares)
        saving[arm] = {"n": len(shares), "mean": centre, "low": lo, "high": hi}
    worst_budget = max(
        (e["budget_stopped"] for r in replicates for arm in ARMS for e in r["episodes"][arm]),
        default=0.0,
    )
    optimiser_clean = worst_budget <= design.max_budget_share
    return {
        "replicates": len(rows),
        "matched": n,
        "excess_mean": mean,
        "excess_low": low,
        "excess_high": high,
        "adjusted_cheaper": sum(value > 0.0 for value in values),
        "decision": decision,
        "v1_comfort_matched": bool(reached),
        "worst_budget_share": float(worst_budget),
        "v2_optimiser_clean": bool(optimiser_clean),
        "stands": bool(reached and optimiser_clean),
        "saving_against_baseline": saving,
        "per_replicate": rows,
    }


# ---- the run ----------------------------------------------------------------------------------


def _read_journal(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path | None, record: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _precision() -> str:
    import jax

    return "float64" if jax.config.read("jax_enable_x64") else "float32"


def run_experiment(
    plant: Plant,
    design: Design = DESIGN,
    *,
    journal: Path | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Every replicate -- log, fits, baseline, both arms over the offset grid -- resumable.

    The journal holds one line per replicate's fits, per baseline and per episode. A replicate with
    anything left to run is logged again, and the run refuses to continue if the log's fingerprint
    differs from the journal's: the emulator is deterministic, so a different log means a different
    experiment.
    """
    done = _read_journal(journal)
    replicates: list[dict[str, Any]] = []
    for k in range(design.replicates):
        mine = [entry for entry in done if entry.get("replicate") == k]
        fitted: dict[str, Any] | None = next((e for e in mine if e["kind"] == "fits"), None)
        baseline = next((e for e in mine if e["kind"] == "baseline"), None)
        episodes = {(e["arm"], e["offset"]): e for e in mine if e["kind"] == "episode"}
        wanted = [(arm, offset) for arm in ARMS for offset in design.offsets]
        panel: Panel | None = None
        if fitted is None or any(key not in episodes for key in wanted):
            log = log_episode(
                plant,
                HEAT_PUMP,
                policy="reset",
                seed=k,
                steps=design.log_steps,
                step_s=design.step_s,
                start_time=design.log_day(k) * DAY_S,
            )
            panel = panel_from_log(log, seed=k)
            fingerprint = panel.provenance.data_sha256
            if fitted is not None and fitted["panel_sha256"] != fingerprint:
                raise RuntimeError(
                    f"replicate {k}: the log's fingerprint {fingerprint[:12]} disagrees with the "
                    f"journal's {fitted['panel_sha256'][:12]}; the stages are not reproducing -- "
                    "start a fresh journal"
                )
            if fitted is None:
                fitted = {
                    "kind": "fits",
                    "replicate": k,
                    "seed": k,
                    "log_day": design.log_day(k),
                    "window_day": design.window_day(k),
                    "panel_sha256": fingerprint,
                    "x64": panel.provenance.x64,
                    "chc_version": panel.provenance.chc_version,
                    "at_bound": log.at_bound,
                    "fits": {arm: fit_summary(panel, arm, design) for arm in ARMS},
                }
                _append(journal, fitted)
            say(
                f"replicate {k}: log from day {design.log_day(k):g}, authority at 21 C "
                + ", ".join(f"{arm} {fitted['fits'][arm]['authority_21']:.4f}" for arm in ARMS)
            )
        if baseline is None:
            kpi = run_baseline_episode(
                plant,
                HEAT_PUMP.testcase,
                baseline_controller(),
                start_time=design.window_day(k) * DAY_S,
                warmup_period=design.warmup_days * DAY_S,
                step_s=design.step_s,
                horizon_steps=design.control_steps,
            )
            baseline = {"kind": "baseline", "replicate": k, "kpi": _numeric(kpi)}
            _append(journal, baseline)
        for arm, offset in wanted:
            if (arm, offset) in episodes:
                continue
            assert panel is not None  # a missing episode is exactly what re-logged it above
            ran = run_arm_episode(plant, panel, arm, offset, design, start_day=design.window_day(k))
            entry = {"kind": "episode", "replicate": k, "arm": arm, "offset": offset, **ran}
            episodes[(arm, offset)] = entry
            _append(journal, entry)
            say(
                f"replicate {k} {arm} offset {offset:+.2f}: tdis {ran['kpi']['tdis_tot']:.4f} "
                f"ener {ran['kpi']['ener_tot']:.4f}"
            )
        replicates.append(
            {
                **{k: v for k, v in fitted.items() if k != "kind"},
                "baseline": baseline["kpi"],
                "episodes": {
                    arm: [
                        {k: v for k, v in episodes[(arm, offset)].items() if k != "kind"}
                        for offset in design.offsets
                    ]
                    for arm in ARMS
                },
            }
        )
    return {
        "design": asdict(design),
        "testcase": HEAT_PUMP.testcase,
        "precision": _precision(),
        "replicates": replicates,
        "verdict": verdict(replicates, design),
    }


# ---- the report -------------------------------------------------------------------------------


def _show(value: float | None, spec: str = ".4f") -> str:
    return "--" if value is None else format(value, spec)


def _certificate_rows(results: Mapping[str, Any]) -> list[str]:
    """Per arm, over every episode: what each call certified beside what the plant then did."""
    lines = []
    for arm in ARMS:
        runs = [e for r in results["replicates"] for e in r["episodes"][arm]]
        tube = [e["tube_mean"] for e in runs if e["tube_mean"] is not None]
        inside = [e["within_tube"] for e in runs if e["within_tube"] is not None]
        bounds = [e["regret_bound_max"] for e in runs if e["regret_bound_max"] is not None]
        declined = [e["regret_uncertified"] for e in runs if e["regret_uncertified"] is not None]
        lines.append(
            f"| {arm} | {min(e['trusted_min'] for e in runs)}-{max(e['trusted_max'] for e in runs)}"
            f" | {_show(float(np.mean(tube)) if tube else None)}"
            f" | {float(np.sqrt(np.mean([e['error_rms'] ** 2 for e in runs]))):.4f}"
            f" | {max(e['error_max'] for e in runs):.4f}"
            f" | {_show(float(np.mean(inside)) if inside else None, '.3f')}"
            f" | {float(np.mean([e['within_tolerance'] for e in runs])):.3f}"
            f" | {_show(max(bounds) if bounds else None, '.2e')}"
            f" | {_show(float(np.mean(declined)) if declined else None, '.3f')}"
            f" | {max(e['budget_stopped'] for e in runs):.3f}"
            f" | {max(e['no_progress'] for e in runs):.3f} |"
        )
    return lines


def markdown(results: Mapping[str, Any]) -> str:
    """The results file, every number in it read off ``results``."""
    design, gate = results["design"], results["verdict"]
    replicates = results["replicates"]
    lines = [
        "# L8.1 -- BOPTEST through `chc.prescribe`",
        "",
        "Produced by `causaldyn_bench.boptest_prescribe` at its pre-registered defaults; the",
        "pre-registration is that module's docstring, committed before this run.",
        "",
        f"- test case `{results['testcase']}`, steps of {design['step_s'] / 60.0:g} min; each of "
        f"{design['replicates']} replicates logs {design['log_steps']} steps of the reset policy, "
        f"then runs {design['control_steps']}-step windows",
        f"- target offsets {', '.join(f'{o:+g}' for o in design['offsets'])} K; horizon "
        f"{design['horizon']} steps, tolerance {design['tolerance']} K, lever unit cost "
        f"{design['unit_cost']}",
        f"- precision: **{results['precision']}** (`JAX_ENABLE_X64`); chc "
        f"{replicates[0]['chc_version']}",
        "",
        "## The gate",
        "",
        "`excess = E_naive / E_adjusted - 1`, each arm's `ener_tot` read off its own front at the "
        "built-in baseline's `tdis_tot` in the same window (relative, per replicate): mean "
        f"**{gate['excess_mean']:+.4f}**, 95% t-interval [{gate['excess_low']:+.4f}, "
        f"{gate['excess_high']:+.4f}] over {gate['matched']} of {gate['replicates']} replicates; "
        f"the adjusted arm was cheaper in {gate['adjusted_cheaper']}.",
        "",
        f"- decision: **{gate['decision']}**",
        f"- V1 (both fronts reach the baseline's comfort in every replicate): "
        f"{'pass' if gate['v1_comfort_matched'] else 'FAIL'}",
        f"- V2 (no episode had more than {design['max_budget_share']:g} of its solves stopped by "
        f"the iteration budget): worst {gate['worst_budget_share']:.3f} -> "
        f"{'pass' if gate['v2_optimiser_clean'] else 'FAIL'}",
        f"- the verdict {'STANDS' if gate['stands'] else 'DOES NOT STAND'}",
        "",
        "## Against the built-in baseline (descriptive, no decision)",
        "",
        "`saving = 1 - E_arm / E_baseline`, the arm's energy read at the baseline's discomfort.",
        "",
        "| arm | replicates | mean saving | 95% t-interval |",
        "|---|---|---|---|",
        *(
            f"| {arm} | {row['n']} | {row['mean']:+.4f} | [{row['low']:+.4f}, {row['high']:+.4f}] |"
            for arm, row in gate["saving_against_baseline"].items()
        ),
        "",
        "## Per replicate",
        "",
        "Energy in kWh/m2 and discomfort in K h, BOPTEST's `ener_tot` and `tdis_tot`. `--` where a "
        "front does not reach the level.",
        "",
        "| replicate | log from day | window from day | baseline tdis | baseline ener | adjusted "
        "ener at baseline tdis | naive ener at baseline tdis | excess | adjusted tdis at baseline "
        "ener | naive tdis at baseline ener |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for replicate, row in zip(replicates, gate["per_replicate"], strict=True):
        comfort = row["comfort_at_baseline_energy"]
        lines.append(
            f"| {replicate['replicate']} | {replicate['log_day']:g} | "
            f"{replicate['window_day']:g} | {row['level']:.4f} | "
            f"{replicate['baseline']['ener_tot']:.4f} | "
            f"{_show(row['energy']['adjusted'])} | {_show(row['energy']['naive'])} | "
            f"{_show(row['excess'], '+.4f')} | {_show(comfort['adjusted'])} | "
            f"{_show(comfort['naive'])} |"
        )
    lines += [
        "",
        "## What `prescribe` identified from each log",
        "",
        "Read through Track D-causal's definitions: the channel `b0 + b1 T` in K/h per unit "
        "modulation and the zone temperature where it changes sign, the decay at the log's mean "
        "action and at the two ends of the actuator, the authority at 21 C, the 8-hour step "
        "response in K.",
        "",
        "| replicate | arm | identification | adjusted for | channel | channel zero at C "
        "| authority at 21 C | 8 h step response | decay at mean action | decay off / full "
        "| channel standard error | overlap |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for replicate in replicates:
        for arm in ARMS:
            fit = replicate["fits"][arm]
            b0, b1 = fit["channel"]
            lines.append(
                f"| {replicate['replicate']} | {arm} | {fit['identification']} | "
                f"{', '.join(fit['adjusted_for']) or 'nothing'} | {b0:+.3f} {b1:+.4f} T | "
                f"{_show(-b0 / b1 if b1 else None, '.2f')} | {fit['authority_21']:.4f} | "
                f"{fit['step_response_8h']:.3f} | {fit['decay']:+.5f} | "
                f"{fit['decay_at_off']:+.5f} / {fit['decay_at_full']:+.5f} | "
                f"{_show(fit['channel_error'])} | {fit['overlap']:.5f} |"
            )
    lines += [
        "",
        "## Fronts",
        "",
        "| replicate | arm | offset K | tdis K h | ener kWh/m2 | cost | saturated | mean action |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for replicate in replicates:
        for arm in ARMS:
            for episode in replicate["episodes"][arm]:
                kpi = episode["kpi"]
                lines.append(
                    f"| {replicate['replicate']} | {arm} | {episode['offset']:+g} | "
                    f"{kpi['tdis_tot']:.4f} | {kpi['ener_tot']:.4f} | "
                    f"{_show(kpi.get('cost_tot'))} | {episode['saturated']:.3f} | "
                    f"{episode['mean_action']:.3f} |"
                )
    lines += [
        "",
        "## What the certificate said, and what the plant did",
        "",
        "Every control step is one `prescribe` call, and only its first action is applied, so the "
        "one step ahead is the part of each plan the plant can check. Pooled over every episode of "
        "an arm: the trustworthy prefix each call reported, the error tube's radius one step "
        "ahead, the plant's one-step error against the plan's own next temperature, the share of "
        "steps that error stayed inside the tube and inside the tolerance, the largest finite "
        "regret bound and the share of calls whose bound was infinite (the objective not convex "
        "over the box), and the largest share of calls any episode had stopped by the iteration "
        "budget or left at the zero guess. `--` where the certificate did not evaluate the "
        "quantity. The regret bound is a gap in the planning objective on the fitted model; the "
        "certificate carries no barrier, because the façade takes no constraint on its target.",
        "",
        "| arm | trustworthy steps | tube one step ahead K | one-step error rms K | largest "
        "error K | inside the tube | inside the tolerance | regret bound | bound infinite | "
        "stopped by budget | no progress |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *_certificate_rows(results),
        "",
        "## `report()` for replicate 0",
        "",
        "One call per arm from the log's last state towards 21 C: what the façade prints.",
        "",
    ]
    for arm in ARMS:
        lines += [f"### {arm}", "", "```text", replicates[0]["fits"][arm]["report"], "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=Path("results/boptest_prescribe"))
    args = parser.parse_args()
    if _precision() != "float64":
        raise SystemExit(
            "JAX_ENABLE_X64=1 is required: the harness's fits moved 10x in float32 once, and the "
            "precision is recorded next to every number"
        )
    if not is_available(args.url):
        raise RuntimeError(f"no BOPTEST-Service at {args.url}; bring it up and pass --url")
    client = BOPTestClient(args.url, timeout=600.0)
    results = run_experiment(client, Design(), journal=args.out / "journal.jsonl")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    text = markdown(results)
    (args.out / "results.md").write_text(text)
    print(text)
    print(f"written to {args.out}/results.md and {args.out}/results.json")


if __name__ == "__main__":
    main()

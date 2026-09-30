"""D21 -- L8.1 rerun: the BOPTEST heat pump through ``chc.prescribe``, now able to state the band,
the weather and the schedule. Pre-registered.

L8.1 (:mod:`causaldyn_bench.boptest_prescribe`) found three things the façade could not state on
this building, and its pre-registered verdict did not stand. Since then ``prescribe`` states all
three: a :class:`chc.Constraint` on the steered column held inside the solve, exogenous
:class:`chc.Driver` columns in the drift with their forecast in the plan, and a :class:`chc.Target`
whose level moves inside the horizon. This module asks L8.1's question again with the three stated.
This docstring is the pre-registration. It was committed before any scored episode ran; the scored
run is :func:`main` at the defaults of :class:`Design`, and changing a default afterwards is a new
experiment.

**Design.** L8.1's, unchanged: replicate ``k = 0..5`` logs 960 half-hour steps of Track D-causal's
``reset`` policy (seed ``k``) from day ``7k``, and the scored window is the 7 days (336 steps) that
follow it, each episode started after BOPTEST's own controller has run the day before. The emulator
is deterministic, so the logs are L8.1's, and each replicate's panel fingerprint is reported beside
L8.1's. In each window run BOPTEST's built-in baseline once, then both arms at each target offset;
the offsets are the pilot's, below.

**Arms.** At every step both arms make the same call, from the zone temperature just measured, with
BOPTEST's forecasts over the plan's 16 steps (17 points, the first one now)::

    prescribe(panel, levers=[Lever("modulation", 0, 1, unit_cost=0)],
              target=Target("zone", value=lower[1:] + offset),
              constraints=[Constraint("zone", hi=min(upper[1:]))], hold_constraints=True,
              drivers=[Driver("outdoor", outdoor), Driver("solar", solar)],
              horizon=16, dt=0.5, tolerance=0.5, x0=[zone], adjustment=<arm>)

and apply only the schedule's first action. ``lower`` and ``upper`` are the comfort band
(``LowerSetp[1]``, ``UpperSetp[1]``) in Celsius, ``outdoor`` and ``solar`` the forecast's
``TDryBul`` in Celsius and ``HGloHor`` in kW/m2, the panel's own units. ``adjusted`` passes L8.1's
graph, which the façade resolves to ``{outdoor, solar, bound}``; ``naive`` passes ``()``. Nothing
else differs.

- *The target* is the lower bound step by step, plus the offset, where L8.1 held the highest bound
  in view for the whole plan.
- *The band's upper edge* is held at the lowest upper bound in view. Its lower edge is not held:
  the offsets below it are how an arm trades comfort for energy.
- *The weather* enters the drift as drivers, with BOPTEST's forecast, which is the weather file
  itself: at day 300 the forecast's first point matched the measurement to 1e-9 K.

**What the drivers do to the naive arm.** :func:`chc.dynamics_id.fit_causal_residual` puts both
ends of every driver into the channel's nuisance covariates, whatever ``adjustment`` asserts. So
the naive arm is adjusted for the weather too, and the two arms differ only in ``bound``: the
comfort feedback of the logging policy, and the occupancy behind it. On an unscored log (days
300-320) the naive arm's channel came back ``0.767 + 0.0055 T`` against the adjusted
``1.138 - 0.0149 T``, with no sign change in the band, where L8.1's naive channel changed sign at
22.3-28.6 C. The gate therefore asks whether the graph's adjustment still pays once the weather is
a driver. No prediction of the effect's size or sign is made.

**Metric.** L8.1's. Per replicate, an arm's ``(tdis_tot, ener_tot)`` points over the offsets are
its operating points and its front their non-dominated subset. Its energy at the built-in
baseline's ``tdis_tot`` in the same window is read on the front, linearly between the two points
that bracket it. ``excess = E_naive / E_adjusted - 1``.

**Gate.** L8.1's. The mean excess over replicates, with a two-sided 95% Student-t interval.
CONFIRMED if the interval lies above 0, REFUTED if below, INCONCLUSIVE if it contains 0. Two
validity checks; either failing voids the decision, which is then printed with DOES NOT STAND:

- V1: both arms' fronts bracket the baseline's discomfort in every replicate, at a positive energy.
- V2: no episode has more than 1% of its calls **from inside the held band** stopped by the solver's
  iteration budget. A call from a zone already above the band's upper edge cannot meet the barrier.
  On the unscored log, from 25 C under a 24 C edge, both arms returned zero heating with the status
  ``max_iterations`` and no barrier-certified step. Such calls are counted and reported per episode,
  and not gated: they measure the plant's excursions, not the optimiser.

**R12.** As in L8.1: energy is read at the baseline's discomfort, so an arm cannot win by heating
less. The dual, discomfort at the baseline's energy, is reported beside it and not gated.

**Against the baseline** (secondary, descriptive, no decision): each arm's saving
``1 - E_arm / E_baseline`` at the baseline's comfort, with the same interval.

**The band on the plant** (reported, not gated): per episode, the share of steps the zone ended
above the held upper edge, the share of calls made from above it, and the barrier-certified prefix.
L8.1's runaway weeks overheated the zone to 522-1403 K h. Whether the held band keeps the plant out
of them is read here, not assumed.

**Certificate against plant** (reported, not gated): L8.1's. Per call, the trustworthy prefix, the
error tube one step ahead, the solver status and the regret bound. Per step, the one-step error
against the plan's own next temperature, and whether it stayed inside the tube and inside the
tolerance.

**Against L8.1** (descriptive): the same windows' baseline and each arm's saving beside L8.1's
published numbers (bench ``46bb2f9``). L8.1 ran chc 0.5.1 (``7796c69``), so this reads the
façade's change and the library's together.

**Size.** L8.1's: 6 replicates x 2 arms x the offsets, plus 6 baseline weeks and six 20-day logs,
sized to the emulator time available and not for power. Neighbouring logs share 13 of their 20
days, so the interval is optimistic to that extent. Replicates may run in parallel, one BOPTEST
worker each, each writing its own journal; every test is its own emulator instance, so this changes
no number.

**Pilot, unscored.** A log off the scored weeks (seed 99, days 320-340, panel ``0b64a769``) and
the window after it (days 340-347), each arm at offsets from -0.5 to +3 K.

- *The fits.* The adjusted channel came back ``0.891 - 0.0113 T`` and the naive ``1.331 -
  0.0291 T``, both positive through the band; the drivers' gains agreed (0.018 and 0.021 per K
  outdoors, 2.2 and 2.4 per kW/m2 of sun).
- *The fronts.* The baseline had ``tdis`` 2.51 K h at ``ener`` 1.89 kWh/m2. An arm tracks its
  target from both sides, and it under-heated until the offset was large: 57-61 K h at 0,
  20.5-20.8 at +1, 8.1-9.5 at +1.5, 2.1-2.3 at +2 and 0.22 at +2.5. At +3 the target met the upper
  edge, and 12% of the naive arm's calls were made from above it; the point, 0.74 K h at more
  energy, is dominated.
  The two arms' energies were within 1% of each other at every offset.
- *The solver.* No call was stopped by the budget from inside the band. A third of the calls ended
  with the solver's no-progress status, and 70% of the actions were at a bound of the box.
- *The offsets.* L8.1's, up to +1.5, stop short of the baseline's discomfort here. The scored grid
  is +1 to +2.5 in quarter steps: dense where L8.1's baseline weeks (3.4-5.9 K h) should cross,
  and ending at the pilot's least discomfort.

Episodes took 8-21 minutes each, four at a time beside other jobs; that is not a timing. The
pilot's processes also grew by about 4.5 MB a call, since every held prescription compiled its
barrier again. chc fixed that before the scored run (``f3f9eae``), which the run imports. On a
surrogate of this building the fix left all 144 schedules of a closed loop the same, bit for bit;
the pilot was not run again.

**Precision.** float64 only. :func:`main` refuses without ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from chc import Constraint, Driver, Lever, Panel, Target, prescribe
from chc.decision import Prescription

from causaldyn_bench.boptest import DEFAULT_URL, BOPTestClient, baseline_controller, is_available
from causaldyn_bench.boptest import run_episode as run_baseline_episode
from causaldyn_bench.boptest_capped import Plant
from causaldyn_bench.boptest_causal import HEAT_PUMP, KELVIN, LOWER_SETP, log_episode
from causaldyn_bench.boptest_prescribe import (
    ARMS,
    DAY_S,
    MODULATION,
    OUTDOOR,
    SOLAR,
    ZONE,
    Arm,
    adjustment,
    interval,
    matched,
    panel_from_log,
)
from causaldyn_bench.boptest_prescribe import _numeric as numeric_kpis
from causaldyn_bench.boptest_prescribe import _precision as precision
from causaldyn_bench.boptest_prescribe import _ratio as ratio

UPPER_SETP = "UpperSetp[1]"
DRY_BULB, GLOBAL_HORIZONTAL = "TDryBul", "HGloHor"
FORECAST_POINTS = [LOWER_SETP, UPPER_SETP, DRY_BULB, GLOBAL_HORIZONTAL]

# L8.1's published verdict for the same windows, read beside this run's (bench 46bb2f9)
L81_RESULTS = "results/boptest_prescribe/results.json"


@dataclass(frozen=True)
class Design:
    """The pre-registered constants. Changing a default after the scored run is a new experiment."""

    step_s: float = 1800.0
    log_steps: int = 960
    control_steps: int = 336
    replicates: int = 6
    stride_days: int = 7
    warmup_days: float = 1.0
    horizon: int = 16
    offsets: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
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


@dataclass(frozen=True)
class Forecast:
    """BOPTEST's boundary conditions over one plan, ``horizon + 1`` points from now, in the panel's
    units: Celsius for the band and the outdoor air, kW/m2 for the sun."""

    lower: np.ndarray
    upper: np.ndarray
    outdoor: np.ndarray
    solar: np.ndarray

    @classmethod
    def read(cls, plant: Plant, testid: str, design: Design) -> Forecast:
        raw = plant.forecast(testid, FORECAST_POINTS, design.horizon * design.step_s, design.step_s)
        points = design.horizon + 1

        def series(name: str) -> np.ndarray:
            values = np.asarray(raw[name], dtype=np.float64)
            if values.shape[0] < points:
                raise RuntimeError(
                    f"BOPTEST forecast {points} points of {name}, got {values.shape}"
                )
            return values[:points]

        return cls(
            lower=series(LOWER_SETP) - KELVIN,
            upper=series(UPPER_SETP) - KELVIN,
            outdoor=series(DRY_BULB) - KELVIN,
            solar=series(GLOBAL_HORIZONTAL) / 1000.0,
        )

    @property
    def ceiling(self) -> float:
        """The band's upper edge the plan holds: the lowest one in view."""
        return float(np.min(self.upper[1:]))

    def target(self, offset: float) -> np.ndarray:
        """The lower bound at the end of each step, plus ``offset``."""
        return self.lower[1:] + offset


def prescription(
    panel: Panel, arm: Arm, *, temp: float, forecast: Forecast, offset: float, design: Design
) -> Prescription:
    """The one ``prescribe`` call both arms make, from the zone temperature measured now."""
    return prescribe(
        panel,
        levers=[
            Lever(MODULATION, HEAT_PUMP.action_lo, HEAT_PUMP.action_hi, unit_cost=design.unit_cost)
        ],
        target=Target(ZONE, value=forecast.target(offset)),
        horizon=design.horizon,
        adjustment=adjustment(arm),
        constraints=[Constraint(ZONE, hi=forecast.ceiling)],
        hold_constraints=True,
        drivers=[Driver(OUTDOOR, forecast.outdoor), Driver(SOLAR, forecast.solar)],
        dt=design.dt,
        tolerance=design.tolerance,
        x0=jnp.asarray([temp]),
    )


def _held_forecast(panel: Panel, design: Design) -> Forecast:
    """The log's last conditions held over one plan: the fit does not read a forecast, but the call
    needs one, and a plan made from it is discarded."""
    last = {name: float(np.asarray(panel[name], dtype=np.float64)[-1]) for name in (OUTDOOR, SOLAR)}
    flat = np.ones(design.horizon + 1)
    return Forecast(
        lower=21.0 * flat, upper=24.0 * flat, outdoor=last[OUTDOOR] * flat, solar=last[SOLAR] * flat
    )


def fit_summary(panel: Panel, arm: Arm, design: Design) -> dict[str, Any]:
    """What the D21 call identified from the log: the channel ``b0 + b1 T`` in K/h per unit
    modulation and where it changes sign, the drift, the drivers' gains, and the fit's errors."""
    zone = np.asarray(panel[ZONE], dtype=np.float64)
    held = prescription(
        panel,
        arm,
        temp=float(zone[-1]),
        forecast=_held_forecast(panel, design),
        offset=0.0,
        design=design,
    )
    fit = held.model_fit
    drift = np.asarray(fit.residual.drift)[0]
    b0, b1 = (float(v) for v in np.asarray(fit.residual.channel)[0, 0])
    gain = None if fit.driver_gain is None else np.asarray(fit.driver_gain)[0]
    certificate = held.certificate
    return {
        "method": fit.method,
        "identification": certificate.identification,
        "adjusted_for": list(certificate.adjustment.covariates),
        "channel": [b0, b1],
        "channel_zero": -b0 / b1 if b1 else None,
        "authority_21": b0 + 21.0 * b1,
        "drift": [float(drift[0]), float(drift[1])],
        "driver_gain": None if gain is None else {OUTDOOR: float(gain[0]), SOLAR: float(gain[1])},
        "channel_error": fit.channel_error,
        "drift_error": fit.drift_error,
        "integrator_defect": fit.integrator_defect,
        "overlap": fit.action_residual_variance,
        "nuisance_r2_action": fit.nuisance_r2_action,
        "nuisance_r2_state": fit.nuisance_r2_state,
        "unmoved": int(np.shape(fit.unmoved)[1]) if fit.unmoved is not None else None,
        "report": held.report(),
    }


@dataclass(frozen=True)
class Step:
    """One control step: what the prescription said, and what the plant then did."""

    temp: float
    target: float  # the first step's level
    ceiling: float  # the band's upper edge the call held
    action: float
    predicted: float  # the plan's own next zone temperature
    realised: float
    tube: float | None  # the certificate's error radius one step ahead; None when not evaluated
    trusted: int
    barrier: int | None  # the barrier-certified prefix
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
    """One week of receding horizon: a fresh ``prescribe`` from every measured state, only the
    first action of each schedule applied."""
    testid = plant.select(HEAT_PUMP.testcase)
    steps: list[Step] = []
    try:
        plant.set_step(testid, design.step_s)
        measurements = plant.initialize(testid, start_day * DAY_S, design.warmup_days * DAY_S)
        for _ in range(design.control_steps):
            forecast = Forecast.read(plant, testid, design)
            temp = measurements[HEAT_PUMP.zone_point] - KELVIN
            decided = prescription(
                panel, arm, temp=temp, forecast=forecast, offset=offset, design=design
            )
            plan = decided.plan
            if plan is None:  # both arms' effects are identified or asserted; see the tests
                raise RuntimeError(f"{arm}: prescribe returned no schedule at {temp:.3f} C")
            action = float(decided.schedule.magnitudes[0, 0])
            measurements = plant.advance(testid, HEAT_PUMP.overwrite(action))
            tube = plan.uncertainty_tube
            certificate = decided.certificate
            steps.append(
                Step(
                    temp=float(temp),
                    target=float(forecast.target(offset)[0]),
                    ceiling=forecast.ceiling,
                    action=action,
                    predicted=float(plan.trajectory[1, 0]),
                    realised=float(measurements[HEAT_PUMP.zone_point] - KELVIN),
                    tube=None if tube is None else float(tube[1]),
                    trusted=certificate.trustworthy_steps,
                    barrier=certificate.barrier_certified_steps,
                    solver_status=certificate.solver_status,
                    regret_bound=certificate.regret_bound,
                )
            )
        kpi = plant.kpi(testid) or {}
    finally:
        plant.stop(testid)
    return {"kpi": numeric_kpis(kpi), **episode_summary(steps, design)}


def episode_summary(steps: Sequence[Step], design: Design) -> dict[str, Any]:
    """The certificate and the band beside the plant, per episode. A pure function of the steps.

    ``budget_stopped`` is V2's share: calls made from inside the held band that the iteration
    budget stopped. Calls from above the band's upper edge cannot meet the barrier, so their stops
    are reported apart, as ``budget_stopped_outside``.
    """
    action = np.array([s.action for s in steps])
    error = np.array([s.realised - s.predicted for s in steps])
    tubes = [s.tube for s in steps]
    evaluated = all(t is not None for t in tubes)
    tube = np.array([t for t in tubes if t is not None])
    outside = np.array([s.temp > s.ceiling for s in steps])
    stopped = np.array([s.solver_status == "max_iterations" for s in steps])
    bounds = [s.regret_bound for s in steps if s.regret_bound is not None]
    finite = [b for b in bounds if math.isfinite(b)]
    barriers = [s.barrier for s in steps if s.barrier is not None]
    tol = 1e-6 * (HEAT_PUMP.action_hi - HEAT_PUMP.action_lo)
    inside_calls = int(np.sum(~outside))
    return {
        "steps": len(steps),
        "saturated": float(
            np.mean((action <= HEAT_PUMP.action_lo + tol) | (action >= HEAT_PUMP.action_hi - tol))
        ),
        "mean_action": float(np.mean(action)),
        "budget_stopped": float(np.sum(stopped & ~outside) / inside_calls) if inside_calls else 0.0,
        "budget_stopped_outside": int(np.sum(stopped & outside)),
        "calls_from_above_band": float(np.mean(outside)),
        "ended_above_band": float(np.mean([s.realised > s.ceiling for s in steps])),
        "barrier_min": int(min(barriers)) if barriers else None,
        "no_progress": float(np.mean([s.solver_status == "no_progress" for s in steps])),
        "trusted_min": int(min(s.trusted for s in steps)),
        "trusted_max": int(max(s.trusted for s in steps)),
        "tube_mean": float(np.mean(tube)) if evaluated else None,
        "error_rms": float(np.sqrt(np.mean(error**2))),
        "error_max": float(np.max(np.abs(error))),
        "error_mean": float(np.mean(error)),
        "within_tube": float(np.mean(np.abs(error) <= tube)) if evaluated else None,
        "within_tolerance": float(np.mean(np.abs(error) <= design.tolerance)),
        "regret_bound_max": float(max(finite)) if finite else None,
        "regret_uncertified": float(np.mean([not math.isfinite(b) for b in bounds]))
        if bounds
        else None,
    }


def verdict(replicates: Sequence[Mapping[str, Any]], design: Design) -> dict[str, Any]:
    """The pre-registered gate: L8.1's, with V2 read over the calls made from inside the band."""
    rows = [matched(r) for r in replicates]
    reached = all(row["reached"] for row in rows)
    values = [row["excess"] for row in rows if row["excess"] is not None]
    mean, low, high = interval(values)
    decision = "CONFIRMED" if low > 0.0 else "REFUTED" if high < 0.0 else "INCONCLUSIVE"
    saving = {}
    for arm in ARMS:
        ratios = [
            ratio(row["energy"][arm], r["baseline"]["ener_tot"])
            for row, r in zip(rows, replicates, strict=True)
        ]
        shares = [1.0 - value for value in ratios if value is not None]
        centre, lo, hi = interval(shares)
        saving[arm] = {"n": len(shares), "mean": centre, "low": lo, "high": hi}
    worst_budget = max(
        (e["budget_stopped"] for r in replicates for arm in ARMS for e in r["episodes"][arm]),
        default=0.0,
    )
    optimiser_clean = worst_budget <= design.max_budget_share
    return {
        "replicates": len(rows),
        "matched": len(values),
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


def _journal(directory: Path | None, replicate: int) -> Path | None:
    return None if directory is None else directory / f"replicate-{replicate}.jsonl"


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


def run_replicate(
    plant: Plant | None,
    k: int,
    design: Design = DESIGN,
    *,
    journal: Path | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """One replicate -- log, fits, baseline, both arms over the offsets -- resumable from its own
    journal. ``plant`` may be ``None`` only when the journal already holds everything; the run
    refuses to continue if a re-logged panel's fingerprint differs from the journal's."""
    done = _read_journal(journal)
    fitted: dict[str, Any] | None = next((e for e in done if e["kind"] == "fits"), None)
    baseline = next((e for e in done if e["kind"] == "baseline"), None)
    episodes = {(e["arm"], e["offset"]): e for e in done if e["kind"] == "episode"}
    wanted = [(arm, offset) for arm in ARMS for offset in design.offsets]
    complete = fitted is not None and baseline is not None and all(k_ in episodes for k_ in wanted)
    if not complete and plant is None:
        raise RuntimeError(f"replicate {k}: the journal is incomplete and no plant was given")
    panel: Panel | None = None
    if fitted is None or any(key not in episodes for key in wanted):
        assert plant is not None
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
            f"replicate {k}: log from day {design.log_day(k):g}, channel zero at "
            + ", ".join(f"{arm} {fitted['fits'][arm]['channel_zero']:.2f} C" for arm in ARMS)
        )
    assert fitted is not None
    if baseline is None:
        assert plant is not None
        kpi = run_baseline_episode(
            plant,
            HEAT_PUMP.testcase,
            baseline_controller(),
            start_time=design.window_day(k) * DAY_S,
            warmup_period=design.warmup_days * DAY_S,
            step_s=design.step_s,
            horizon_steps=design.control_steps,
        )
        baseline = {"kind": "baseline", "replicate": k, "kpi": numeric_kpis(kpi)}
        _append(journal, baseline)
    for arm, offset in wanted:
        if (arm, offset) in episodes:
            continue
        assert plant is not None and panel is not None
        ran = run_arm_episode(plant, panel, arm, offset, design, start_day=design.window_day(k))
        entry = {"kind": "episode", "replicate": k, "arm": arm, "offset": offset, **ran}
        episodes[(arm, offset)] = entry
        _append(journal, entry)
        say(
            f"replicate {k} {arm} offset {offset:+.2f}: tdis {ran['kpi']['tdis_tot']:.4f} "
            f"ener {ran['kpi']['ener_tot']:.4f} above band {ran['ended_above_band']:.3f}"
        )
    return {
        **{key: value for key, value in fitted.items() if key != "kind"},
        "baseline": baseline["kpi"],
        "episodes": {
            arm: [
                {key: value for key, value in episodes[(arm, offset)].items() if key != "kind"}
                for offset in design.offsets
            ]
            for arm in ARMS
        },
    }


def run_experiment(
    plant: Plant | None,
    design: Design = DESIGN,
    *,
    journal: Path | None = None,
    chc_commit: str | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Every replicate in turn, each resumable from its own journal under ``journal``.
    ``chc_commit`` is the library commit the run imported, recorded beside the installed version,
    which names a release and not the code between releases."""
    replicates = [
        run_replicate(plant, k, design, journal=_journal(journal, k), say=say)
        for k in range(design.replicates)
    ]
    return {
        "design": asdict(design),
        "testcase": HEAT_PUMP.testcase,
        "precision": precision(),
        "chc_commit": chc_commit,
        "replicates": replicates,
        "verdict": verdict(replicates, design),
    }


def against_l81(results: Mapping[str, Any], l81: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per replicate: whether the panel is L8.1's, and the baseline beside L8.1's. Descriptive."""
    earlier = {r["replicate"]: r for r in l81["replicates"]}
    rows = []
    for replicate in results["replicates"]:
        then = earlier.get(replicate["replicate"])
        rows.append(
            {
                "replicate": replicate["replicate"],
                "same_panel": None
                if then is None
                else then["panel_sha256"] == replicate["panel_sha256"],
                "baseline_tdis": replicate["baseline"]["tdis_tot"],
                "l81_baseline_tdis": None if then is None else then["baseline"]["tdis_tot"],
            }
        )
    return rows


# ---- the report -------------------------------------------------------------------------------


def _show(value: float | None, spec: str = ".4f") -> str:
    return "--" if value is None else format(value, spec)


def markdown(results: Mapping[str, Any], l81: Mapping[str, Any] | None = None) -> str:
    """The results file, every number in it read off ``results`` (and L8.1's, when given)."""
    design, gate = results["design"], results["verdict"]
    replicates = results["replicates"]
    lines = [
        "# D21 -- L8.1 rerun: BOPTEST through `chc.prescribe`, the band held, the weather as "
        "drivers, the bound as a schedule",
        "",
        "Produced by `causaldyn_bench.boptest_rerun` at its pre-registered defaults; the",
        "pre-registration is that module's docstring, committed before this run.",
        "",
        f"- test case `{results['testcase']}`, steps of {design['step_s'] / 60.0:g} min; each of "
        f"{design['replicates']} replicates logs {design['log_steps']} steps of the reset policy, "
        f"then runs {design['control_steps']}-step windows",
        f"- target offsets {', '.join(f'{o:+g}' for o in design['offsets'])} K over the lower "
        f"bound; horizon {design['horizon']} steps, tolerance {design['tolerance']} K, lever unit "
        f"cost {design['unit_cost']}",
        f"- precision: **{results['precision']}** (`JAX_ENABLE_X64`); chc "
        f"{replicates[0]['chc_version']}"
        + (f" at `{results['chc_commit']}`" if results["chc_commit"] else ", commit not recorded"),
        "",
        "## The gate",
        "",
        "`excess = E_naive / E_adjusted - 1`, each arm's `ener_tot` read off its own front at the "
        "built-in baseline's `tdis_tot` in the same window: mean "
        f"**{gate['excess_mean']:+.4f}**, 95% t-interval [{gate['excess_low']:+.4f}, "
        f"{gate['excess_high']:+.4f}] over {gate['matched']} of {gate['replicates']} replicates; "
        f"the adjusted arm was cheaper in {gate['adjusted_cheaper']}.",
        "",
        f"- decision: **{gate['decision']}**",
        f"- V1 (both fronts reach the baseline's comfort in every replicate): "
        f"{'pass' if gate['v1_comfort_matched'] else 'FAIL'}",
        f"- V2 (no episode had more than {design['max_budget_share']:g} of its calls from inside "
        f"the band stopped by the iteration budget): worst {gate['worst_budget_share']:.3f} -> "
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
        "The channel `b0 + b1 T` in K/h per unit modulation and the zone temperature where it "
        "changes sign, the drift `a0 + a1 T`, and the drivers' gains in K/h per C of outdoor air "
        "and per kW/m2 of sun.",
        "",
        "| replicate | arm | identification | adjusted for | channel | channel zero at C "
        "| authority at 21 C | drift | outdoor gain | solar gain | channel standard error "
        "| overlap |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for replicate in replicates:
        for arm in ARMS:
            fit = replicate["fits"][arm]
            b0, b1 = fit["channel"]
            a0, a1 = fit["drift"]
            gain = fit["driver_gain"] or {}
            lines.append(
                f"| {replicate['replicate']} | {arm} | {fit['identification']} | "
                f"{', '.join(fit['adjusted_for']) or 'nothing'} | {b0:+.3f} {b1:+.4f} T | "
                f"{_show(fit['channel_zero'], '.2f')} | {fit['authority_21']:.4f} | "
                f"{a0:+.3f} {a1:+.4f} T | {_show(gain.get(OUTDOOR))} | {_show(gain.get(SOLAR))} | "
                f"{_show(fit['channel_error'])} | {fit['overlap']:.5f} |"
            )
    lines += [
        "",
        "## Fronts",
        "",
        "| replicate | arm | offset K | tdis K h | ener kWh/m2 | cost | saturated | mean action "
        "| ended above band | calls from above band |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for replicate in replicates:
        for arm in ARMS:
            for episode in replicate["episodes"][arm]:
                kpi = episode["kpi"]
                lines.append(
                    f"| {replicate['replicate']} | {arm} | {episode['offset']:+g} | "
                    f"{kpi['tdis_tot']:.4f} | {kpi['ener_tot']:.4f} | "
                    f"{_show(kpi.get('cost_tot'))} | {episode['saturated']:.3f} | "
                    f"{episode['mean_action']:.3f} | {episode['ended_above_band']:.3f} | "
                    f"{episode['calls_from_above_band']:.3f} |"
                )
    lines += [
        "",
        "## What the certificate said, and what the plant did",
        "",
        "Every control step is one `prescribe` call, and only its first action is applied, so the "
        "plant checks the plan's first step alone. Over every episode of each arm: the trustworthy "
        "prefix's range, the mean tube one step ahead, the one-step error's root mean square and "
        "largest size, the share of steps inside the tube and inside the tolerance, the largest "
        "finite regret bound, the share of calls whose bound was declined, the worst V2 share, and "
        "the calls from above the band that the budget stopped (not gated).",
        "",
        "| arm | trusted | tube | error rms | error max | within tube | within tolerance "
        "| regret bound max | declined | budget stopped (V2) | stopped from above the band "
        "| barrier prefix min |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        runs = [e for r in replicates for e in r["episodes"][arm]]
        tube = [e["tube_mean"] for e in runs if e["tube_mean"] is not None]
        inside = [e["within_tube"] for e in runs if e["within_tube"] is not None]
        bounds = [e["regret_bound_max"] for e in runs if e["regret_bound_max"] is not None]
        declined = [e["regret_uncertified"] for e in runs if e["regret_uncertified"] is not None]
        barriers = [e["barrier_min"] for e in runs if e["barrier_min"] is not None]
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
            f" | {sum(e['budget_stopped_outside'] for e in runs)}"
            f" | {min(barriers) if barriers else '--'} |"
        )
    if l81 is not None:
        lines += [
            "",
            "## Against L8.1 (descriptive)",
            "",
            "The same windows under L8.1's call and chc 0.5.1 (bench `46bb2f9`). Whether each "
            "replicate's panel is byte for byte L8.1's, and each baseline beside L8.1's.",
            "",
            "| replicate | same panel | baseline tdis | L8.1 baseline tdis |",
            "|---|---|---|---|",
            *(
                f"| {row['replicate']} | {row['same_panel']} | {row['baseline_tdis']:.4f} | "
                f"{_show(row['l81_baseline_tdis'])} |"
                for row in against_l81(results, l81)
            ),
            "",
            "| arm | L8.1 mean saving | D21 mean saving |",
            "|---|---|---|",
            *(
                f"| {arm} | {l81['verdict']['saving_against_baseline'][arm]['mean']:+.4f} | "
                f"{gate['saving_against_baseline'][arm]['mean']:+.4f} |"
                for arm in ARMS
            ),
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=Path("results/boptest_rerun"))
    parser.add_argument(
        "--replicates",
        type=int,
        nargs="+",
        help="run only these replicates, each into its own journal, and write no results; "
        "a run without it collects every journal and writes the results",
    )
    parser.add_argument("--chc-commit", help="the library commit this run imports, to record")
    args = parser.parse_args()
    # a non-interactive shell starts its background jobs with SIGINT ignored, and the recipe runs
    # the replicates as such; an episode's `finally` is what stops its BOPTEST test
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if precision() != "float64":
        raise SystemExit(
            "JAX_ENABLE_X64=1 is required: the harness's fits moved 10x in float32 once, and the "
            "precision is recorded next to every number"
        )
    design = Design()
    journal = args.out / "journals"
    if args.replicates is not None:
        if not is_available(args.url):
            raise RuntimeError(f"no BOPTEST-Service at {args.url}; bring it up and pass --url")
        client = BOPTestClient(args.url, timeout=600.0)
        for k in args.replicates:
            run_replicate(client, k, design, journal=_journal(journal, k))
        return
    plant = BOPTestClient(args.url, timeout=600.0) if is_available(args.url) else None
    results = run_experiment(plant, design, journal=journal, chc_commit=args.chc_commit)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    l81_path = Path(L81_RESULTS)
    l81 = json.loads(l81_path.read_text()) if l81_path.exists() else None
    text = markdown(results, l81)
    (args.out / "results.md").write_text(text)
    print(text)
    print(f"written to {args.out}/results.md and {args.out}/results.json")


if __name__ == "__main__":
    main()

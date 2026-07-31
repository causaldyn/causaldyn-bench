"""Track D-causal: does causal identification of the control channel pay off on a *real* emulator?

The existing BOPTEST harness (:mod:`causaldyn_bench.boptest_chc`) identifies its thermal model
from a **randomised** exploration episode -- slow PRBS on the actuator. That is a clean experiment,
and it is also the one thing a real building will never give you: production HVAC data comes from a
controller, and every sensible controller is *weather-compensated*. Outdoor-reset curves are the
textbook design.

That makes the logged action a function of the outdoor temperature, which is also what drives the
zone. Regress the temperature rate on ``(1, T, u)`` and the outdoor term lands in the error:

    u  <- outdoor reset      (colder outside => command more heat)
    T' <- outdoor loss       (colder outside => temperature falls faster)

so ``Cov(u, eps) < 0`` and the fitted control channel is biased **downward**: the estimator credits
the heating to a milder outdoors instead of to the actuator. That is the same identification failure
as ``chc.benchmark.CausalDynamicsTask``, except the plant is a Modelica building emulator rather
than a synthetic system and the confounder is real weather.

Which way that bias moves the *controller* is a property of the cost, not of the bias. Under a
quadratic tracking objective an attenuated channel under-actuates. Under BOPTEST's
comfort-dominated objective it does the opposite: believing the plant weak, the controller commands
more heat than a correctly identified model would, so the bias acts as an unintended safety margin
that buys comfort and pays energy. That is measured here, not assumed, and it is why the closed-loop
comparison is a frontier over the requested safety margin (:func:`run_pareto`) rather than a single
operating point -- at one fixed margin the *biased* arm looks better on discomfort.

The experiment is a 2x2, deliberately, so that both directions are falsifiable rather than only the
flattering one:

===================  ==================  ==================
logging policy       adjust for weather  expectation
===================  ==================  ==================
outdoor-reset        no                  attenuated channel
outdoor-reset        yes                 recovered
randomised (PRBS)    no                  unbiased already
randomised (PRBS)    yes                 unchanged
===================  ==================  ==================

If adjustment "helped" the randomised arm too, the estimator would be distorting rather than
de-confounding. All four fits use the same estimator, the same model class and the same MPC, so the
only thing varying across the first two rows is the adjustment set, and across the last two the
logging policy.

**No ground-truth channel exists here.** On a real emulator nothing can report "channel error", and
this module does not pretend otherwise: the randomised arms are the reference (identification by
design). BOPTEST's own KPIs are reported alongside, but they turned out **not** to rank the
arms -- see below.

What the 2x2 actually measured, so that reading the code does not require reading the results file:
de-confounding recovers the mean 8-hour authority on the confounded log to within 3% of the
randomised reference, and does so at roughly **2.4x the reference's seed-to-seed spread**, because
it estimates from the ~15% of action variance the covariates do not explain. A comfort-constrained
MPC is asymmetric in that spread -- believing the plant strong means commanding too little and
banking discomfort that cannot be recovered -- so the de-confounded *point* estimate was the worst
closed-loop arm despite the second-best estimate.

The obvious remedy, spending that radius pessimistically, was then tried and **did not work**, but
not for the reason this docstring used to give. It attributed the failure to the drift dominating
the closed-loop error, citing correlations of +0.79 / -0.79 against the channel's +0.43. Those were
measured with the planner defect described at the end of this docstring; re-derived after the fix
they read +0.67 / -0.64 against +0.11, and -- more to the point -- the de-confounded arm's
discomfort now varies over 0.045 K.h across seeds, so they are correlations against noise at the
comfort floor. The drift-dominates hypothesis is *untested* here, not confirmed.

What the sweep does establish, because it varies the channel deliberately rather than watching it
vary: shrinking the believed gain correlates -0.54 with the 8-hour rise and +0.49 with actuator
saturation over its twenty episodes. Zero shrink is the best setting on the mean and on the spread
at once: one standard error costs 0.9% of the mean, and past that the spread explodes from +-0.011
to +-5.036 K.h. Channel pessimism on this plant is indistinguishable from nothing until it is
catastrophic, which is a worse property than a smooth trade-off. See :func:`run_pessimism_sweep` and
``results/boptest_causal.md``.

Alongside the 2x2 sits the **physics-off** arm (:class:`NeuralFit`, :func:`run_structure_ablation`):
an MLP for the rate given the same ``(T, z, u)``, planning through the same MPC, differing from the
structured arms in the model and in nothing else. It exists to make one claim falsifiable --
*held-out predictive accuracy does not rank causal models*. On the synthetic fixture the black box
matches the structured causal fit's one-step forecast to under half a seed standard deviation --
4.20e-4 against 4.13e-4, both on the 4.0e-4 noise floor -- while its control authority carries 2.6x
the RMSE (0.047 against 0.018 about a truth of 1.200) and a 3.4% bias against 0.07%. A dynamics
benchmark scored in rollout error is blind to that gap by construction, which is why this track
scores the channel. On the emulator the failure is blunter than on the fixture: read at the action
the log sat at, the black box's fitted decay is *positive* on three of five seeds, so
:class:`RunawayDriftError` refuses those three plans offline and the arm has no closed-loop mean to
report. Of the two that plan, one reaches the comfort floor and the other spends 2.2x the
de-confounded arm's discomfort. Held-out error does penalise the black box on this plant (2.2x) and
still orders the arms differently from identification -- the two rankings agree on the winner and
swap the other two, because prediction calls the confounded affine fit second-best where the channel
calls it worst. Sharper still, *within* the black-box arm the ordering inverts: the best one-step
predictor of its five fits is one of the three the stability check refuses. See
``results/boptest_causal.md`` §6.

That section previously reported the opposite -- a black box unbiased in the mean and noisy per seed
-- and the retraction is worth carrying here because of how it was caught. Those numbers were
float32. The affine arms do not notice the precision (an accidental x32 run agreed with x64 to five
digits), so the flag looked optional; 3000 Adam steps do notice, because rounding compounds into the
*derivative* of the fitted surface. Pin ``JAX_ENABLE_X64=1`` before comparing anything to the
recorded numbers -- for this arm it is a requirement, not a recipe.

Scope limits worth stating before the numbers:

* ``bestest_hydronic_heat_pump`` actuates a **compressor modulation**, so control-affinity in the
  action is physically reasonable. The other two cases actuate a **temperature setpoint**, which
  reaches the plant through a local PI loop and saturation; control-affine is a local approximation
  there, and they are a robustness check, not the headline.
* Identification needs **overlap**. A deterministic outdoor-reset policy makes the action an exact
  function of the covariates, the Robinson action residual is identically zero, and nothing is
  identified at any sample size. :func:`overlap_report` measures it and
  :func:`run_overlap_ablation` drives the exploration noise to zero to show the collapse.

Requires a running BOPTEST-Service (see :mod:`causaldyn_bench.boptest`).

On precision, and on a claim this module got wrong for a while. It deliberately does **not** call
``jax.config.update`` at import scope: that is a *process-global* mutation, so merely importing the
module used to change the precision of every other benchmark track in the same pytest session, and
broke one. The docstring then justified dropping the flag by pointing at the synthetic fixture,
where float32 and float64 agree on the adjusted channel to 0.6%.

That justification was **false on the emulator**, and the emulator said so: fitted on 768 rows of a
real reset log, the adjusted arm returned ``nan`` in float32 and a perfectly reasonable channel in
float64. The cause was not precision but conditioning -- ``chc.dynamics_id`` built its degree-2
nuisance basis on the caller's raw covariates, so the zone entering in Celsius at ~21 beside
standardised weather columns produced a Gram at condition number 1.4e11. The fix is in the library
(standardise before the basis; see its CHANGELOG), after which float32 reproduces float64 to four
decimals on the same rows. The general lesson is the one worth keeping: *a precision claim measured
on a synthetic fixture does not transfer to real data*, because what actually failed was the
conditioning that the fixture's units happened to hide.

On equal compute, and on a second claim this module got wrong for longer. "Every arm sees the same
MPC and the same iteration count" was true of the code and false of the comparison. The planner used
a constant gradient step, and the largest *stable* step for this objective is a function of the
authority each arm believes in -- so the arms that identified a larger channel were pushed further
past the stability limit and handed a worse controller by the shared constant. The closed-loop
ordering that produced followed the believed authority rather than the model quality. Equal compute
needs an optimiser whose behaviour is invariant to the model's scale, not merely an equal iteration
count; see :func:`_mpc_solver` for the measurement and the replacement.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from causaldyn_bench.boptest import (
    DEFAULT_URL,
    BOPTestClient,
    baseline_controller,
    is_available,
    run_episode,
)
from causaldyn_bench.tracks import TrackResult

LOWER_SETP = "LowerSetp[1]"  # forecast of the lower comfort bound -- what `tdis_tot` scores against
KELVIN = 273.15
HOLDOUT = 0.2  # chronological tail withheld from every arm and scored by `holdout_mse`


class PlantModel(Protocol):
    """What the MPC needs from a fitted plant, and nothing else.

    The interface exists so the structured arm and the black-box arm plan through *the same*
    optimiser rather than through two solvers written to agree. An ablation whose two halves have
    separate control code cannot separate "the model is worse" from "the solver is different".
    """

    @property
    def policy(self) -> str:
        """Which exploration policy produced the log this was fitted on. Read-only on purpose:
        declaring it as a bare attribute demands read *and* write, which no frozen implementation
        can satisfy, so every arm in the ablation failed the protocol it was written for."""
        ...

    def authority(self, temp: float = 21.0) -> float:
        """``d(rate)/du`` at ``temp``, covariates at the log's mean -- K/h per unit of action.

        The comparable estimand. An affine fit's ``b0`` is *not*: it is the intercept of
        ``b0 + b1 T`` at ``T = 0``, some 21 K outside anything a heated building ever visits, so
        reading it off a nonlinear model extrapolates a local slope across the whole extrapolation
        and returns nonsense. Measured on the emulator over five seeds, the black box's implied
        ``b0`` came out between -0.045 and +0.254 -- a channel that does essentially nothing --
        against the structured fit's +1.25, a factor of 31, while their authorities at 21 C differ
        by 1.5x. The gap is the extrapolation, not the model.
        """
        ...

    @property
    def action_mean(self) -> float:
        """The action the log actually sat at, in the harness's action units.

        Carried because every *derivative* below has to be read somewhere, and ``0`` is not a
        defensible somewhere: it is ``off`` on a modulating heat pump and a setpoint of 0 C on the
        two cases whose actuator is a setpoint in ``[15, 25]``.
        """
        ...

    @property
    def pole(self) -> float:
        """``d(rate)/dT`` at :attr:`action_mean` -- negative on a building that returns to ambient.

        Read at the operating action rather than at ``u = 0`` because for a channel that depends on
        the state the two differ by a change of *units*, not of plant. See :meth:`ThermalFit.decay`.
        """
        ...

    def decay(self, action: float | None = None) -> float:
        """``d(rate)/dT`` at ``action``; :attr:`pole` is this at :attr:`action_mean`.

        On the protocol rather than on the affine fit alone so that the stability of a *plan* is
        scored by one definition for both arms, the way :func:`finite_difference_step_response`
        already scores the step response.
        """
        ...

    def rate(self, temp: Array, covariate: Array, action: Array) -> Array:
        """Predicted ``dT/dt`` in K/h at zone temperature ``temp`` under ``action``."""
        ...

    def step_response(self, hours: float = ..., dt: float = ..., temp: float = ...) -> float: ...


@dataclass(frozen=True)
class BoptestCase:
    """How one BOPTEST case is wired, plus the outdoor-reset gains its actuator units imply.

    The gains live here rather than in a separate policy object because they are per-unit: a gain of
    ``0.03`` means "modulation units per Kelvin" on the heat pump and "Kelvin per Kelvin" on a
    setpoint case, so a single number cannot be shared across actuators.
    """

    testcase: str
    action_point: str
    zone_point: str
    outdoor_point: str
    solar_point: str
    action_lo: float  # in the harness's action units (modulation, or Celsius for a setpoint)
    action_hi: float
    action_offset: float  # raw BOPTEST value = action + offset; 273.15 for a Kelvin setpoint
    reference_action: float  # the reset curve's value at ``outdoor_reference``
    zone_gain: float  # feedback on the comfort shortfall
    weather_gain: float  # THE CONFOUNDING CHANNEL: response to outdoor temperature
    outdoor_reference: float  # Celsius; reset curve pivots here
    explore_scale: float  # sd of the exploration noise that buys overlap
    # The randomised arm excites a BAND around the reset policy's operating point, not the whole
    # actuator box. Uniform on [0, 1] modulation parks a heat pump at 50% duty, which is a different
    # zone temperature and therefore a different local linearisation -- the first pass of this
    # harness did that and the two designs disagreed about the thermal pole by two orders of
    # magnitude. Excitation around the operating point keeps the estimand the same while staying
    # fully exogenous.
    explore_lo: float
    explore_hi: float
    control_affine_in_action: bool  # False where the action is a setpoint behind a local loop

    def to_raw(self, action: float) -> float:
        return float(np.clip(action, self.action_lo, self.action_hi)) + self.action_offset

    def overwrite(self, action: float) -> dict[str, Any]:
        return {self.action_point: self.to_raw(action), f"{self.action_point[:-2]}_activate": 1}


HEAT_PUMP = BoptestCase(
    testcase="bestest_hydronic_heat_pump",
    action_point="oveHeaPumY_u",
    zone_point="reaTZon_y",
    outdoor_point="weaSta_reaWeaTDryBul_y",
    solar_point="weaSta_reaWeaHGloHor_y",
    action_lo=0.0,
    action_hi=1.0,
    action_offset=0.0,
    # Calibrated on DESIGN metrics only -- actuator saturation and overlap -- never on the channel
    # estimate. The first pass commanded a mean modulation of 0.43 and spent 30% of steps clipped;
    # 20% remains, and it is the night setback driving the command to zero, which is what a real
    # controller does. `at_bound` is reported so a reader can see how much of the log carries no
    # action information at all.
    reference_action=0.15,
    zone_gain=0.10,
    # A real outdoor-reset curve takes the actuator from off to full across roughly a 20 K outdoor
    # range, i.e. ~0.05 modulation per Kelvin. The first pass used 0.02, which put the confounding
    # channel below the estimator's own noise floor -- the 2x2 came back unresolvable, not clean.
    # Raised on fidelity grounds, and the outcome is reported whichever way it lands.
    weather_gain=0.05,
    outdoor_reference=5.0,
    explore_scale=0.12,
    explore_lo=0.05,
    explore_hi=0.60,
    control_affine_in_action=True,
)

CASES: dict[str, BoptestCase] = {
    "heat_pump": HEAT_PUMP,
    "hydronic": BoptestCase(
        testcase="bestest_hydronic",
        action_point="oveTSetHea_u",
        zone_point="reaTRoo_y",
        outdoor_point="weaSta_reaWeaTDryBul_y",
        solar_point="weaSta_reaWeaHGloHor_y",
        action_lo=15.0,
        action_hi=25.0,
        action_offset=KELVIN,
        reference_action=21.0,
        zone_gain=0.5,
        weather_gain=0.15,
        outdoor_reference=5.0,
        explore_scale=0.8,
        explore_lo=18.0,
        explore_hi=24.0,
        control_affine_in_action=False,
    ),
    "air": BoptestCase(
        testcase="bestest_air",
        action_point="con_oveTSetHea_u",
        zone_point="zon_reaTRooAir_y",
        outdoor_point="zon_weaSta_reaWeaTDryBul_y",
        solar_point="zon_weaSta_reaWeaHGloHor_y",
        action_lo=15.0,
        action_hi=25.0,
        action_offset=KELVIN,
        reference_action=21.0,
        zone_gain=0.5,
        weather_gain=0.15,
        outdoor_reference=5.0,
        explore_scale=0.8,
        explore_lo=18.0,
        explore_hi=24.0,
        control_affine_in_action=False,
    ),
}


@dataclass(frozen=True)
class LoggedEpisode:
    """One identification episode, in Celsius and hours so the design matrix stays conditioned.

    Kelvin and seconds put the temperature rate at ~1e-4 and the squared solar term at ~1e6, which a
    degree-2 ridge-polynomial nuisance cannot survive: the ridge either dominates the signal or the
    Gram matrix loses its conditioning. Nothing here is a modelling choice -- it is unit hygiene,
    and it is the single most important difference between this harness and the synthetic task.
    """

    zone: Array  # (N, 1) Celsius
    action: Array  # (N, 1) harness action units
    zone_next: Array  # (N, 1) Celsius
    weather: Array  # (N, 3) standardised [outdoor C, solar kW/m2, comfort bound C]
    weather_mean: tuple[float, ...]  # kept so the controller can standardise a live forecast
    weather_scale: tuple[float, ...]
    dt_hours: float
    at_bound: float  # fraction of steps the policy spent clipped, where overlap is worst
    policy: str

    def as_data(self) -> dict[str, Array]:
        return {"x": self.zone, "u": self.action, "x_next": self.zone_next, "weather": self.weather}

    def standardise(self, raw: np.ndarray) -> Array:
        return jnp.asarray((raw - np.asarray(self.weather_mean)) / np.asarray(self.weather_scale))

    def head(self) -> LoggedEpisode:
        """The episode minus its :data:`HOLDOUT` tail -- what every arm is fitted on when the tail
        is going to be scored.

        Chronological, not a random subset: neighbouring half-hour samples of a building are almost
        the same row, so a shuffled split measures interpolation between them and is not a forecast
        error at all. ``at_bound`` is left describing the whole episode; it is a property of the
        logging policy, not of a slice.
        """
        cut = max(round(self.zone.shape[0] * (1.0 - HOLDOUT)), 1)
        return replace(
            self,
            zone=self.zone[:cut],
            action=self.action[:cut],
            zone_next=self.zone_next[:cut],
            weather=self.weather[:cut],
        )


def _standardise(columns: np.ndarray) -> tuple[Array, tuple[float, ...], tuple[float, ...]]:
    centre = columns.mean(axis=0)
    scale = np.where(columns.std(axis=0) > 1e-9, columns.std(axis=0), 1.0)
    return jnp.asarray((columns - centre) / scale), tuple(centre), tuple(scale)


def log_episode(
    client: BOPTestClient,
    case: BoptestCase,
    *,
    policy: str,
    seed: int,
    steps: int = 480,
    step_s: float = 1800.0,
    hold: int = 4,
    explore_scale: float | None = None,
) -> LoggedEpisode:
    """Drive one identification episode under ``policy`` and return the log.

    ``"reset"`` is the confounded production-like policy: an outdoor-reset curve plus comfort
    feedback plus exploration noise. ``"prbs"`` is the randomised reference: a level resampled every
    ``hold`` steps and independent of the plant. ``hold`` matters for both -- i.i.d. per-step action
    is low-passed by a building's slow thermal mode, which collapses the identified DC gain no
    matter how the action was chosen.
    """
    if policy not in ("reset", "prbs"):
        raise ValueError(f"unknown policy {policy!r}; expected 'reset' or 'prbs'")
    scale = case.explore_scale if explore_scale is None else explore_scale
    rng = np.random.default_rng(seed)
    testid = client.select(case.testcase)  # after validation: no worker spun up for a bad argument
    zone, action, zone_next, covariates = [], [], [], []
    clipped = 0
    try:
        client.set_step(testid, step_s)
        measurements = client.initialize(testid, 0.0, 0.0)
        level = case.reference_action
        for i in range(steps):
            forecast = client.forecast(testid, [LOWER_SETP], step_s, step_s)
            bound = forecast[LOWER_SETP][0] - KELVIN
            temp = measurements[case.zone_point] - KELVIN
            outdoor = measurements[case.outdoor_point] - KELVIN
            solar = measurements[case.solar_point] / 1000.0
            if policy == "prbs":
                if i % hold == 0:
                    level = float(rng.uniform(case.explore_lo, case.explore_hi))
                raw = level
            else:
                if i % hold == 0:  # hold the exploration draw, same reason as the PRBS hold
                    level = float(rng.normal(0.0, scale))
                raw = (
                    case.reference_action
                    + case.zone_gain * (bound + 1.0 - temp)
                    + case.weather_gain * (case.outdoor_reference - outdoor)
                    + level
                )
            applied = float(np.clip(raw, case.action_lo, case.action_hi))
            clipped += abs(applied - raw) > 1e-9
            measurements = client.advance(testid, case.overwrite(applied))
            zone.append(temp)
            action.append(applied)
            zone_next.append(measurements[case.zone_point] - KELVIN)
            covariates.append([outdoor, solar, bound])
    finally:
        client.stop(testid)
    standardised, centre, scale = _standardise(np.asarray(covariates))
    return LoggedEpisode(
        zone=jnp.asarray(zone, dtype=jnp.float64).reshape(-1, 1),
        action=jnp.asarray(action, dtype=jnp.float64).reshape(-1, 1),
        zone_next=jnp.asarray(zone_next, dtype=jnp.float64).reshape(-1, 1),
        weather=standardised,
        weather_mean=centre,
        weather_scale=scale,
        dt_hours=step_s / 3600.0,
        at_bound=clipped / steps,
        policy=policy,
    )


@dataclass(frozen=True)
class ThermalFit:
    """A fitted ``dT/dt = a*T + c'z + d + (b0 + b1*T) u`` together with what identified it.

    The ``c'z`` term -- the measured boundary conditions entering the drift as *regressors*, not
    only as nuisances -- is here because of what the first run of this harness produced without it:
    a **positive** thermal pole in three of four arms, i.e. a fitted building that heats itself
    without bound. That is not a bug in :func:`chc.dynamics_id.fit_causal_residual`; it is its
    documented scope showing up on real data. The estimator makes the *channel* interventional and
    lets ``a_theta`` mop up whatever is left of the rate at that channel, so the drift stays
    observational-conditional. On the synthetic task nothing was left to mop up. On twenty days of a
    real building the outdoor temperature trends, the zone follows it, and a drift with no weather
    term charges that common movement to a positive feedback in ``T``.

    An MPC uses the drift as well as the channel, so an unstable drift is fatal regardless of how
    good the channel is. The fix is not a constraint but the correct model: the weather is *observed
    and exogenous*, so it belongs in the drift, and the forecast needed to use it at control time is
    something BOPTEST serves. The Robinson moment still earns its keep -- it is what keeps the
    channel robust to the weather entering *nonlinearly*, which a linear ``c'z`` would miss.
    """

    drift: tuple[float, float]  # (d, a) -- bias first, matching control_affine_features
    weather_drift: tuple[float, ...]  # c, on the standardised covariate columns
    channel: tuple[float, float]  # (b0, b1)
    method: str
    identified: bool
    adjusted: bool
    policy: str
    action_mean: float  # the log's operating point; every derivative below is read there
    action_residual_variance: float
    nuisance_r2_action: float
    channel_error: float | None

    def rate(self, temp: Array, covariate: Array, action: Array) -> Array:
        bias, pole = self.drift
        exogenous = jnp.dot(jnp.asarray(self.weather_drift), covariate)
        return pole * temp + bias + exogenous + (self.channel[0] + self.channel[1] * temp) * action

    def decay(self, action: float | None = None) -> float:
        """``d(rate)/dT`` at ``action``, defaulting to the log's own operating point.

        A bilinear fit has no single pole. ``drift[1]`` is the decay at ``action = 0``, and that is
        a property of the *units the actuator reports in* rather than of the building. Under an
        affine change of actuator coordinates ``u = alpha v + beta`` the model class is closed and
        the coefficients map as ``a -> a + beta b1``, ``b0 -> alpha b0``, ``b1 -> alpha b1``
        (Maxima, closure residual 0), so ``a`` alone is not comparable across actuators while
        ``a + b1 u`` at a fixed physical actuator position is invariant.

        Measured on the emulator, reading it at ``0`` instead of here is the whole of the "runaway
        building" this harness reported for two of its three cases:

        ==========  =========  ============  ===========
        case        ``lo``     ``drift[1]``  ``decay()``
        ==========  =========  ============  ===========
        heat_pump   0.0        -0.034        -0.042
        hydronic    15.0 C     **+6.420**    -1.397
        air         15.0 C     **+5.878**    -0.512
        ==========  =========  ============  ===========

        Refitting the same logs on the action expressed as a fraction of travel reproduces
        ``decay()`` to four decimals (-1.3970 and -0.5120) while ``drift[1]`` moves to +0.75 and
        +1.16, which is the invariance above, measured rather than assumed. All three buildings are
        stable; only the two whose actuator is a setpoint in ``[15, 25] C`` had a reported pole 15 K
        outside the range the actuator can reach.
        """
        return self.drift[1] + self.channel[1] * (self.action_mean if action is None else action)

    def step_response(self, hours: float = 8.0, dt: float = 0.5, temp: float = 21.0) -> float:
        """Open-loop Kelvin gained over ``hours`` from a unit step in the action, at ``temp``.

        Reported instead of the steady-state gain ``-b/a`` on purpose. A five-day window at
        half-hour resolution does not identify a building's *infinite-horizon* gain: the first pass
        of this harness printed a randomised-design pole of ``-3e-4`` and hence a DC gain of 1493 K,
        which is not a finding about the building but a near-unit-root artefact of fitting one
        thermal time constant to a plant that has at least two. The finite-horizon step response
        over the MPC's own look-ahead *is* identifiable here, and is what the controller acts on.
        """
        pole, gain = self.decay(), self.channel[0] + self.channel[1] * temp
        state = 0.0
        for _ in range(max(round(hours / dt), 1)):
            state = state + dt * (pole * state + gain)
        return state

    def authority(self, temp: float = 21.0) -> float:
        return self.channel[0] + self.channel[1] * temp

    @property
    def pole(self) -> float:
        return self.decay()

    @property
    def stable(self) -> bool:
        """Whether the fitted drift decays. An unstable fit is reported, never silently repaired."""
        return self.decay() < 0.0

    def stable_over(self, lo: float, hi: float) -> bool:
        """Whether the drift decays at *every* action the controller may command.

        The condition an MPC needs, and stronger than :attr:`stable`, which only reads the log's
        operating point. ``decay`` is affine in the action, so the box maximum is at an endpoint --
        exact, not sampled.
        """
        return max(self.decay(lo), self.decay(hi)) < 0.0

    def pessimistic(self, shrink: float) -> ThermalFit:
        """Reduce the believed channel gain ``b0`` by ``shrink``: plan against a *weaker* plant.

        The direction is set by the cost, not by taste. Under a comfort constraint the expensive
        mistake is believing the plant *stronger* than it is: the controller then commands too
        little, the zone drifts below the band, and the discomfort is banked by the time the error
        shows. Believing it weaker only costs energy. A pure energy objective wants the opposite
        sign, which is why this is a signed decision and not a "robustness" dial.

        ``shrink`` is an **absolute** amount in ``b0``'s own units, and it is deliberately not
        expressed in standard errors, because this fit has no honest standard error for ``b0``.
        ``channel_error`` is the root-mean diagonal of the estimator's sandwich over the *whole*
        channel matrix, so it mixes ``b0`` with ``b1``, whose natural scale is smaller by a factor
        of the operating temperature. Using it as a 1-sigma radius on ``b0`` was measured here and
        over-states the radius by **6.9x**: ``channel_error`` came out ~0.51, while one seed-to-seed
        standard error of the quantity the controller actually uses -- the 8-hour step response,
        0.487 K over five seeds, at 6.64 K of rise per unit ``b0`` -- is a shrink of only 0.073. At
        ``shrink = channel_error`` the channel went negative and the controller gave up entirely:
        discomfort 76 -> 1475 K.h, energy to zero, saturated at a bound every single step.

        The ratio is data-dependent, so there is no offline regression test for it: on this
        module's synthetic fixture ``channel_error / |b0|`` is 0.08 and shrinking by it is
        harmless, while on the emulator it is 0.40 and fatal. That ratio is what to check before
        spending a radius, which is why ``channel_error`` is carried on the fit rather than hidden.

        So the radius is the caller's to supply and to justify. A defensible one here is a multiple
        of the *empirical* spread across logging seeds, which is what :func:`run_pessimism_sweep`
        sweeps.
        """
        if shrink < 0.0:
            raise ValueError(
                f"shrink reduces the believed gain and cannot be negative: {shrink}. "
                "Optimism about a control channel is not this function's job."
            )
        return replace(self, channel=(self.channel[0] - shrink, self.channel[1]))


def fit_thermal(log: LoggedEpisode, *, adjusted: bool, folds: int = 2, ridge: float = 1e-6):
    """Fit the channel with :func:`chc.dynamics_id.fit_causal_residual`, then the drift around it.

    Two stages on purpose, and the order matters. The channel comes from the orthogonal moment,
    which is the part that has to survive confounding. The drift is then least squares on what the
    channel leaves behind, ``rate - b(T) u`` regressed on ``(1, T, z)``: refitting the channel
    jointly with a linear weather term would hand identification back to OLS and throw away the
    reason for using a debiased moment at all.
    """
    from chc.dynamics import LinearDynamics
    from chc.dynamics_id import fit_causal_residual

    known = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))  # no physics prior: all residual
    fit = fit_causal_residual(
        known,
        log.as_data(),
        log.dt_hours,
        adjust_for=("weather",) if adjusted else (),
        degree=1,
        folds=folds,
        ridge=ridge,
    )
    residual = fit.residual
    channel = (float(residual.channel[0, 0, 0]), float(residual.channel[0, 0, 1]))
    drift, weather_drift = _fit_drift(log, channel)
    return ThermalFit(
        drift=drift,
        weather_drift=weather_drift,
        channel=channel,
        method=fit.method,
        identified=fit.identified,
        adjusted=adjusted,
        policy=log.policy,
        action_mean=float(np.mean(np.asarray(log.action))),
        action_residual_variance=fit.action_residual_variance,
        nuisance_r2_action=fit.nuisance_r2_action,
        channel_error=fit.channel_error,
    )


def _fit_drift(
    log: LoggedEpisode, channel: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, ...]]:
    """OLS of the rate net of the identified channel on ``(1, T, z)``; returns ``((d, a), c)``."""
    zone = np.asarray(log.zone).ravel()
    action = np.asarray(log.action).ravel()
    rate = (np.asarray(log.zone_next).ravel() - zone) / log.dt_hours
    target = rate - (channel[0] + channel[1] * zone) * action
    design = np.column_stack([np.ones_like(zone), zone, np.asarray(log.weather)])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return (float(coefficients[0]), float(coefficients[1])), tuple(
        float(value) for value in coefficients[2:]
    )


def finite_difference_step_response(
    model: PlantModel, *, hours: float = 8.0, dt: float = 0.5, temp: float = 21.0
) -> float:
    """:meth:`ThermalFit.step_response` for a model with no coefficients to read off.

    One definition for both arms, which is the point: authority is the rate gained per unit action,
    the drift is what the deviation does on its own, and the two are integrated together over the
    controller's own look-ahead. On an affine ``rate`` this reproduces the closed form exactly --
    pinned by an offline test, because an ablation that scored its two halves on two subtly
    different metrics would be measuring the metric.

    Authority is read at ``temp`` and held there, exactly as the closed form holds
    ``b0 + b1 * temp``, while the drift is followed along the deviation. Letting authority track
    the trajectory instead is defensible in isolation and wrong here: it folds each model's own
    curvature into the number, so the black box would be scored on a quantity the affine arm cannot
    express. Measured, that variant moved the two arms apart by 0.8% of the response for no reason
    but the definition.
    """
    covariate = jnp.zeros(3)  # the log's own mean: standardised covariates are centred
    # Both derivatives are read at the log's operating action, not at zero. Zero is "off" on a
    # modulating actuator but a setpoint of 0 C on the two setpoint cases, where the affine fit's
    # decay differs between the two points by 6.4 K/h -- a change of units, not of building.
    base, origin = jnp.asarray(model.action_mean), jnp.asarray(temp)
    authority = model.rate(origin, covariate, base + 1.0) - model.rate(origin, covariate, base)
    state = 0.0
    for _ in range(max(int(hours / dt), 1)):
        here = jnp.asarray(temp + state)
        decay = model.rate(here, covariate, base) - model.rate(origin, covariate, base)
        state = state + dt * float(authority + decay)
    return state


class NeuralFit(eqx.Module):
    """The physics-off, causality-off arm: an MLP for ``dT/dt`` given ``(T, z, u)``.

    This is the ablation that tests the *structured* half of "causal identification for
    physics-structured residual dynamics". It is deliberately not a straw man. It sees the same log,
    the same three boundary-condition columns and the same action; the confounder is *inside* its
    conditioning set, so plain back-door adjustment is available to it and it is free to fit any
    nonlinearity the affine model cannot. It also gets far more fitting compute than the
    closed-form arm, which is a handicap in the baseline's favour and is the point -- a black box
    that loses after being given every advantage loses for a structural reason.

    What it cannot do is orthogonalise. On six seeds of the synthetic fixture its *held-out one-step
    prediction is statistically indistinguishable* from the structured causal fit -- 4.20e-4 against
    4.13e-4, both on the 4.0e-4 irreducible noise floor, under half a seed-to-seed standard
    deviation -- while its control authority carries **2.6x the RMSE** (0.0469 against 0.0179 about
    a truth of 1.200) and a **3.4% bias against 0.07%**. Two models that agree on every forecast
    they will ever be scored on disagree measurably on the one number a controller consumes. That is
    the argument for identifying the channel rather than fitting the dynamics: **held-out predictive
    accuracy does not rank causal models**, so a dynamics benchmark reported in rollout error cannot
    see this failure at all.

    Worth not over-claiming: on that fixture the effect is real but modest, because the plant is
    well specified, stationary and generously sampled. The confounder is in the black box's
    conditioning set, so it is not biased by omission the way the ``naive`` affine arm is -- that
    arm attenuates the authority to 14% of truth, and prediction *does* catch it, at 19x the
    held-out error. What is left for the black box is regularisation bias plus variance under weak
    overlap: with 15% of the action's variance surviving partialling out, gradient descent has
    little signal to attribute to ``u`` and the fitted derivative wanders. On the emulator --
    nonstationary, 768 training rows, real weather -- it wanders much further; see
    ``results/boptest_causal.md``.

    The structure is what makes the correction expressible: it is only because
    ``dT/dt = a(T) + b(T) u`` names ``b`` as a separate object that there is a moment condition to
    make Neyman-orthogonal at all.
    """

    mlp: eqx.nn.MLP
    zone_mean: float = eqx.field(static=True)
    zone_scale: float = eqx.field(static=True)
    policy: str = eqx.field(static=True)
    action_mean: float = eqx.field(static=True)
    train_mse: float = eqx.field(static=True)

    def rate(self, temp: Array, covariate: Array, action: Array) -> Array:
        features = jnp.concatenate(
            [
                jnp.atleast_1d((temp - self.zone_mean) / self.zone_scale),
                jnp.atleast_1d(covariate).ravel(),
                jnp.atleast_1d(action),
            ]
        )
        return self.mlp(features)[0]

    def step_response(self, hours: float = 8.0, dt: float = 0.5, temp: float = 21.0) -> float:
        return finite_difference_step_response(self, hours=hours, dt=dt, temp=temp)

    @property
    def pole(self) -> float:
        """``d(rate)/dT`` at the operating point -- the black box's analogue of a thermal pole."""
        return self.decay()

    def decay(self, action: float | None = None) -> float:
        at = self.action_mean if action is None else action
        grad = jax.grad(lambda t: self.rate(t, jnp.zeros(3), jnp.asarray(at)))
        return float(grad(jnp.asarray(self.zone_mean)))

    def authority(self, temp: float = 21.0) -> float:
        gradient = jax.grad(lambda u, t: self.rate(t, jnp.zeros(3), u))
        return float(gradient(jnp.asarray(self.action_mean), jnp.asarray(temp)))


def fit_neural(
    log: LoggedEpisode,
    *,
    steps: int = 3000,
    width: int = 8,
    depth: int = 1,
    lr: float = 3e-3,
    seed: int = 0,
) -> NeuralFit:
    """Fit :class:`NeuralFit` by Adam on the one-step rate of every row it is given.

    The default box is small because it was **swept, not chosen**: 12 fits on one 768-row emulator
    head, width 8 and 64, depth 1 and 3, 1000/3000/12000 Adam steps, two inits each, all in
    float64. The small box wins at every step count -- held-out MSE 0.026 / 0.039 / 0.072 against
    the big box's 0.069 / 0.099 / 0.083 -- and for the small box the error rises monotonically with
    compute. The *derivatives* rot faster than the fit does: at width 64 and depth 3, two inits of
    the *same* configuration return fitted decays of **-7.99 and +1.12**, a factor of nine apart and
    of opposite sign, so they disagree about whether the building is stable at all, while their
    held-out errors differ by 4%. Even at the default the init alone moves the authority by 1.2x.
    So this is the kindest configuration found for this plant at this sample size, which is the
    point: the physics-off arm has to lose from its structure, not from a hyper-parameter. A longer
    log would justify a bigger box, and callers who want one pass it.

    Every number above is float64. The previous version of this sweep was not, and the difference
    was not cosmetic -- see ``results/boptest_causal.md`` §8, defect 7.

    No internal train/test split, deliberately. :func:`fit_thermal` has none either, so a split
    hidden inside one of them would fit the two arms on different data and then score them on the
    same tail -- leakage for the arm that saw it. :func:`predictive_comparison` owns the split for
    both, via :meth:`LoggedEpisode.head`, which is the only place it is defined.

    Explicit Euler on the rate, matching :func:`_fit_drift` and :func:`_mpc_solver` exactly. Fitting
    this arm through :func:`chc.train.fit_residual` would have been less code and would have put an
    RK4 integrator on one side of the ablation and Euler on the other -- a difference in the
    *solver*, reported as a difference in the *model*.
    """
    zone = np.asarray(log.zone).ravel()
    action = np.asarray(log.action).ravel()
    rate = (np.asarray(log.zone_next).ravel() - zone) / log.dt_hours
    centre, scale = float(zone.mean()), float(max(zone.std(), 1e-6))
    features = jnp.asarray(
        np.column_stack([(zone - centre) / scale, np.asarray(log.weather), action])
    )
    targets = jnp.asarray(rate)

    mlp = eqx.nn.MLP(
        in_size=features.shape[1],
        out_size=1,
        width_size=width,
        depth=depth,
        activation=jax.nn.tanh,
        key=jax.random.key(seed),
    )

    def mse(model: eqx.nn.MLP, xs: Array, ys: Array) -> Array:
        return jnp.mean((jax.vmap(model)(xs).ravel() - ys) ** 2)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(mlp, eqx.is_inexact_array))

    @eqx.filter_jit
    def update(
        model: eqx.nn.MLP, state: optax.OptState
    ) -> tuple[eqx.nn.MLP, optax.OptState, Array]:
        loss, grads = eqx.filter_value_and_grad(mse)(model, features, targets)
        updates, state = optimizer.update(grads, state, eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), state, loss

    loss = jnp.asarray(float("nan"))
    for _ in range(steps):
        mlp, opt_state, loss = update(mlp, opt_state)

    return NeuralFit(
        mlp=mlp,
        zone_mean=centre,
        zone_scale=scale,
        policy=log.policy,
        action_mean=float(np.mean(np.asarray(log.action))),
        train_mse=float(loss),
    )


def holdout_mse(fit: PlantModel, log: LoggedEpisode) -> float:
    """One-step rate MSE of ``fit`` on the :data:`HOLDOUT` tail of ``log``, same code for every arm.

    Only meaningful for a ``fit`` produced from ``log.head()``; :func:`predictive_comparison` is
    what arranges that. Scoring every arm through one function is not tidiness -- a second
    implementation on the black-box side is exactly how a 2% difference in held-out error turns out
    to be an artefact of two slightly different holdout definitions.
    """
    zone = np.asarray(log.zone).ravel()
    action = np.asarray(log.action).ravel()
    rate = (np.asarray(log.zone_next).ravel() - zone) / log.dt_hours
    weather = np.asarray(log.weather)
    cut = max(round(len(zone) * (1.0 - HOLDOUT)), 1)
    predicted = np.array(
        [
            float(fit.rate(jnp.asarray(zone[i]), jnp.asarray(weather[i]), jnp.asarray(action[i])))
            for i in range(cut, len(zone))
        ]
    )
    return float(np.mean((predicted - rate[cut:]) ** 2)) if predicted.size else float("nan")


def predictive_comparison(log: LoggedEpisode, *, seed: int = 0) -> dict[str, float]:
    """Held-out one-step error of all three arms, each refitted on ``log.head()``.

    The refit is the whole point. Every arm must be blind to the tail it is scored on, and the
    closed-loop arms are fitted on the *full* log because that is what a deployment would do -- so
    the two questions need two fits, and reusing the control fit here would have handed the
    closed-form arms the answer key while the black box worked from 80% of the data. The gap that
    would have produced looks exactly like a modelling result.

    Reported because it is the comparison that makes the ablation worth running: which arm predicts
    better, and whether that is the arm whose *control* channel is right. They are different
    questions and on this plant they have different answers.
    """
    head = log.head()
    arms: dict[str, PlantModel] = {
        "affine-naive": fit_thermal(head, adjusted=False),
        "affine-adjusted": fit_thermal(head, adjusted=True),
        "neural": fit_neural(head, seed=seed),
    }
    return {name: holdout_mse(fit, log) for name, fit in arms.items()}


@dataclass(frozen=True)
class HorizonPlan:
    """One MPC solve: the commanded sequence and the trajectory the *fitted model* predicts for it.

    The rollout is carried out of the solver rather than recomputed beside it. A certificate has to
    audit the plan against the same integration the planner optimised through, and a second Euler
    loop written elsewhere in this module would be a second definition of the plan that nothing
    keeps in step -- the failure mode of defects 2, 5 and 6 in the results doc, one more time.
    """

    actions: Array  # (horizon,) in the harness's action units
    states: Array  # (horizon + 1,) Celsius, starting at the measured temperature
    task_cost: float  # the objective the solver actually reached, comfort hinge plus effort


def _mpc_solver(
    fit: PlantModel,
    case: BoptestCase,
    dt: float,
    *,
    w_comfort: float,
    margin: float,
    iterations: int,
):
    """Accelerated projected-gradient comfort MPC on the fitted model.

    Plans against the *forecast* boundary conditions, which is the whole reason the drift carries a
    weather term: without it the controller cannot know a cold night is coming, and tracks the
    setback down instead of pre-heating through it.

    ``fit`` enters only through :meth:`PlantModel.rate`, so the structured arm and the black-box arm
    differ in the model and in nothing else -- same horizon, same objective, same iteration count,
    same projection. Everything below this line is shared by construction rather than by review.

    **The effort term is in fractions of actuator travel, not in actuator units.** ``(action - lo)``
    alone means something different per case: the heat pump modulates on [0, 1] while the two
    setpoint cases command Celsius on [15, 25], so the same ``w_comfort`` would price effort 100x
    higher against comfort there -- an energy-first controller on one case and a comfort-first
    controller on another, reported in the same table as though the objective were held fixed.
    Dividing by the span makes ``w_comfort`` mean one thing across cases; it is exactly a no-op on
    the heat pump, whose span is 1.0, so the recorded numbers are unaffected. Third instance of the
    same defect in this module, after the constant step size below and the unstandardised nuisance
    ridge upstream: a constant whose meaning silently depends on the scale of its input.

    **The step size has to come from the model, not from a constant.** This solver previously took a
    fixed ``lr``, and that quietly decided the benchmark. The comfort term's curvature scales with
    ``w_comfort * (dt * authority)**2``, so the largest stable step ``2/L`` is a function of the
    authority each arm *believes* it has. Measured on a 20-day heat-pump log, ``lr=0.05`` sat 75x
    past the stability limit for the confounded fit, 252x past it for the de-confounded one and 284x
    past it for the black box: the iterate entered a period-8 limit cycle (clip to ``hi``, decay by
    ``1 - 2*lr`` per step, recross the hinge, clip again) and the objective after 400 iterations was
    *worse* than after 60 -- 267.7 against 8.5 for the black box. A model that identifies a larger
    channel earns a larger Hessian and is punished harder by a shared constant step, so the arm with
    the better estimate was handed the worse controller. Reading a closed-loop KPI off that is
    measuring the optimiser.

    ``L`` is therefore taken per solve as the largest eigenvalue of the Hessian of the same
    objective with the hinge forced active, and the step is ``1/L``. For the affine arms that is a
    genuine bound on the curvature wherever the hinge actually is, since forcing it active only adds
    the terms the ``max`` would drop. For the black box it is an estimate rather than a bound -- the
    model's own second derivative in the action enters weighted by the shortfall, which the dense
    surrogate lets go negative -- so its adequacy is checked empirically, by the convergence test
    below rather than by the argument. That leaves conditioning: the action penalty carries
    curvature 2 against the comfort term's ~1e4, so plain gradient descent needs O(kappa) steps and
    reaches neither optimum in a realistic budget. Nesterov acceleration needs O(sqrt(kappa)), which
    is what makes the budget below sufficient. Both cost exactly one gradient evaluation per
    iteration for every arm, so "fixed iterations = equal compute" stays literally true rather than
    approximately.

    Rejected: Barzilai-Borwein, which stalled the black-box arm at its initialisation (the curvature
    pair ``s'y`` collapses where the hinge is inactive); and a box-QP solver, which would have been
    exact for the affine arms and unavailable for the black box, i.e. a different optimiser per arm.

    ``solve`` returns the **whole** horizon plan, not the action the loop applies. Receding horizon
    still executes only the first entry, but :func:`certify_safety` prices a finished plan, and a
    certificate read off the first step alone would be a one-step check rather than an audit.
    """
    lo, hi = case.action_lo, case.action_hi
    span = hi - lo

    def rollout(actions: Array, bounds: Array, covariates: Array, temp0: Array, hinged: bool):
        def step(
            temp: Array, triple: tuple[Array, Array, Array]
        ) -> tuple[Array, tuple[Array, Array]]:
            action, bound, covariate = triple
            nxt = temp + dt * fit.rate(temp, covariate, action)
            gap = bound + margin - nxt
            shortfall = jnp.maximum(gap, 0.0) if hinged else gap
            effort = (action - lo) / span
            return nxt, (nxt, w_comfort * shortfall**2 + effort**2)

        _, (states, costs) = jax.lax.scan(step, temp0, (actions, bounds, covariates))
        return states, jnp.sum(costs)

    def cost(actions: Array, bounds: Array, covariates: Array, temp0: Array) -> Array:
        return rollout(actions, bounds, covariates, temp0, hinged=True)[1]

    def dense(actions: Array, bounds: Array, covariates: Array, temp0: Array) -> Array:
        return rollout(actions, bounds, covariates, temp0, hinged=False)[1]

    grad = jax.jit(jax.grad(cost))
    curvature = jax.jit(jax.hessian(dense))

    def solve(temp0: float, bounds: Array, covariates: Array) -> HorizonPlan:
        start = jnp.full(bounds.shape[0], 0.5 * (lo + hi))
        origin = jnp.asarray(temp0)
        largest = jnp.linalg.eigvalsh(curvature(start, bounds, covariates, origin))[-1]
        step = 1.0 / jnp.maximum(largest, 1e-12)
        actions = lookahead = start
        momentum = 1.0
        for _ in range(iterations):
            moved = jnp.clip(lookahead - step * grad(lookahead, bounds, covariates, origin), lo, hi)
            nxt_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
            lookahead = moved + ((momentum - 1.0) / nxt_momentum) * (moved - actions)
            actions, momentum = moved, nxt_momentum
        reached, spent = rollout(actions, bounds, covariates, origin, hinged=True)
        return HorizonPlan(
            actions=actions,
            states=jnp.concatenate([origin[None], reached]),
            task_cost=float(spent),
        )

    return solve


class RunawayDriftError(RuntimeError):
    """A fitted plant whose drift does not decay, so no horizon planned on it means anything.

    An MPC integrates ``dT/dt = a*T + ...`` over its horizon. With ``a >= 0`` the homogeneous
    solution never decays, the predicted trajectory is set by the extrapolation rather than by the
    plant, and the optimum is an artefact.

    The decay is read at :attr:`PlantModel.action_mean`, and the first version of this check read it
    at ``a = drift[1]`` instead, which is the decay at ``action = 0``. That fired on both
    setpoint-actuated cases -- ``+1.9`` to ``+8.4`` across every policy and both arms -- and every
    one of those was a **false positive**: ``drift[1]`` shifts by ``beta * b1`` under an affine
    change of actuator coordinates, and those two cases report their action as a setpoint in
    ``[15, 25] C``, so ``beta = 15`` and the shift is 4.7 to 5.7 K/h of pure units. Refitting the
    same logs on fraction-of-travel returns the same :meth:`ThermalFit.decay` to four decimals and
    a ``drift[1]`` of ``+0.75`` and ``+1.16``. All three buildings decay: ``-0.042``, ``-1.397``,
    ``-0.512``. A refusal keyed on a quantity that a change of units can flip is not a safety check.

    Checked in :func:`run_control_episode` rather than in :func:`_mpc_solver`, because that is where
    the blast radius is: the solver would merely return a bad number, while the episode spends real
    emulator time and -- measured -- emits commands extreme enough to stall BOPTEST's own solver
    past the client timeout, leaving an orphaned worker child behind. Raised rather than clamped:
    clamping the pole to a small negative number would fabricate a stable plant the data never
    supported and report KPIs for it, and silently wrong is the worse failure.

    Scanned over the actuator box rather than read at the operating point, because the operating
    point is where the *log* sat and the horizon is planned wherever the optimiser wants to go. The
    hydronic fit is the case in point: it decays at -1.40 where it was logged and crosses zero at a
    setpoint of 17.0 C, well inside ``[15, 25]``, so a plan that reaches for the low end of the box
    is extrapolating against a growing model while the operating-point reading says it is fine. For
    an affine ``decay`` the box maximum is at an endpoint and the grid is exact; for the black-box
    arm it is a check and not a proof, which is the price of scoring both arms by one definition.

    Caught by the two sweeps that vary the *model* -- :func:`run_case` and
    :func:`run_structure_ablation` -- because there one runaway arm should not abort the arms beside
    it, and "identified but unplannable" is a result to record. Left to propagate out of the two
    that vary a *knob* on a single model, :func:`run_pareto` and :func:`run_pessimism_sweep`:
    neither the requested margin nor a channel shrink touches the drift, so a runaway there fails
    identically at every setting and the honest report is one exception, not six identical rows.
    """

    def __init__(self, testcase: str, pole: float, rise: float) -> None:
        super().__init__(
            f"{testcase}: fitted decay {pole:+.4f} at the log's operating action does not decay "
            f"(8h step response {rise:.3e} K); refusing to plan a horizon on it"
        )
        self.testcase = testcase
        self.pole = pole
        self.rise = rise


@dataclass(frozen=True)
class SafetyAudit:
    """Price the MPC's plan against a partially identified channel, and optionally act on the price.

    Three operations share the word "safety" in :mod:`chc.plan` -- *plan* (``causal_plan``),
    *audit* (``certify_safety``), *filter* (``robust_safety_filter``) -- and only the last changes
    an action. This carries the two a closed loop can use, and :attr:`enforce` is exactly the switch
    between them: ``False`` runs the audit and records what it found, ``True`` additionally clips
    the applied command into the certified interval. Both arms of the ablation therefore carry the
    same diagnostics and differ only in whether anything reads them, which is what makes them
    comparable -- and it makes the read-only claim checkable rather than asserted, since the
    ``False`` arm must reproduce an un-audited episode command for command.

    ``causal_plan`` is deliberately **not** the planner. It minimises a
    :class:`chc.plan.QuadraticCost` by projected gradient, while this harness plans against a hinge
    on a forecast comfort bound, which is not a quadratic in the state; swapping the objective would
    move every closed-loop number in ``results/boptest_causal.md`` and turn the ablation into "is a
    different controller better". ``certify_safety`` prices a *finished* plan by design, so the
    MPC's horizon is wrapped in a :class:`chc.plan.CausalPlan` and audited as it stands.

    Attributes:
        radius: the operator-norm radius on the control channel, in ``b0``'s units of K/h per unit
            of actuator travel. The default is the one number §5 of the results doc could defend:
            one *measured* standard error of the 8-hour step response across five seeds (0.487 K) at
            6.64 K of rise per unit of ``b0``. The estimator's own ``channel_error`` is **not**
            usable here -- it is a root-mean diagonal over the whole channel matrix, so it mixes
            ``b0`` with ``b1`` and overstates this radius by 6.9x.
        alpha: the class-K gain in ``h_dot >= -alpha*h``, per hour. It is what lets the zone coast
            down through the night rather than being held at the occupied bound: at a 15 C setback
            bound and an 18 C zone the barrier permits ``3*alpha`` K/h of cooling, an order more
            than this building does.
        enforce: whether :func:`chc.barrier.robust_safety_filter` may move the applied command.
    """

    radius: float = 0.073
    alpha: float = 1.0
    enforce: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.radius < 1.0:
            raise ValueError(f"radius must lie in [0, 1) at a unit gap, got {self.radius}")
        if self.alpha <= 0.0:
            raise ValueError(f"class-K gain must be positive, got {self.alpha}")

    @property
    def gamma(self) -> float:
        """The MSM sensitivity level that realises :attr:`radius` at a unit gap.

        ``certify_safety`` takes ``(gamma, cvar_gap)`` and uses them only through
        ``Delta = (G-1)/(G+1)*gap``, so the pair is a convention rather than two facts. Fixing the
        gap at 1 makes ``Delta`` the radius itself in the channel's own units and makes the
        certificate's ``gamma_star`` invertible by the same formula -- the threshold radius a step
        tolerates is ``(G*-1)/(G*+1)`` K/h per unit of travel, with no second convention to carry.
        """
        return (1.0 + self.radius) / (1.0 - self.radius)


@dataclass(frozen=True)
class StepAudit:
    """§40 priced on one MPC solve, and the command the filter would allow instead."""

    margin: float  # h(T) now: how far the zone sits above the bound it is audited against
    guaranteed: float  # worst-case barrier derivative at the step the plan applies
    required: float  # -alpha*h(T); certification is exactly `guaranteed >= required`
    certified: bool  # ...so this is that comparison, reported rather than left implicit
    horizon_share: float  # certified leading prefix of the plan, as a share of its length
    gamma_star: float  # weakest step's sensitivity ceiling; nan where no Gamma certifies it
    nominal: float  # what the MPC asked for
    filtered: float  # what the filter would allow, clipped into the actuator box


def _horizon_dynamics(fit: PlantModel, covariates: Array, dt: float):
    """The fitted plant as a :class:`chc.dynamics.Dynamics`, with the forecast indexed by time.

    ``certify_safety`` evaluates the model at ``t = dt*k`` along the plan, and this plant genuinely
    is time-varying: the boundary conditions enter the drift as regressors, and step ``k`` of the
    horizon sees forecast row ``k`` -- the same pairing the planner's scan uses. Recovering ``k``
    from ``t`` is exact on the grid the certificate builds, so this is the adapter and not an
    approximation of one; the clip only binds if a caller audits past the forecast it planned on.
    """
    last = covariates.shape[0] - 1

    def field(t: float | Array, x: Array, u: Array) -> Array:
        index = jnp.clip(jnp.round(jnp.asarray(t) / dt).astype(jnp.int32), 0, last)
        return jnp.atleast_1d(fit.rate(x[0], covariates[index], u[0]))

    return field


def _audit_plan(
    fit: PlantModel,
    case: BoptestCase,
    plan: HorizonPlan,
    covariates: Array,
    floor: float,
    dt: float,
    audit: SafetyAudit,
) -> StepAudit:
    """Wrap the MPC's horizon in a :class:`chc.plan.CausalPlan` and price it against §40.

    The barrier is ``h(T) = T - floor`` with ``floor`` the comfort bound in force **now**, held
    fixed across the horizon. BOPTEST's bound is a two-level step function -- 21 C occupied, 15 C
    setback, ten +-6 K jumps a week at half-hour resolution -- so a barrier that tracked it would
    demand 12 K/h of the zone at every transition, twenty times this heat pump's full authority, and
    the audit would be reporting the setback schedule rather than the channel. Freezing it is the
    standard CBF treatment of a moving safe set; it keeps ``||grad h|| = 1``, so the radius is
    ``Delta`` exactly rather than the ``sqrt(2)*Delta`` an augmented state would charge for a
    coordinate that carries no actuator; and it is the constraint the filter acts on, since
    enforcement only ever touches the first step, where "now" is unambiguous. Anticipating the
    *next* bound is the MPC's job, and the MPC does see the forecast.

    The planner's ``margin`` is deliberately absent. The certificate is about BOPTEST's comfort
    bound -- the line ``tdis_tot`` is billed against -- while the margin is the planner's
    conservatism dial, and folding one into the other would make the certificate a function of a
    knob that ``run_pareto`` sweeps.

    ``drift`` and ``channel`` for the filter are re-read off ``fit`` rather than recovered from
    the certificate, which reports only their combination. They agree with what ``certify_safety``
    used by construction, not by coincidence: with ``grad h = 1`` its drift is ``rate(T, z_0, 0)``
    and its channel the Jacobian of the same call, both at the same point.
    """
    from chc.barrier import robust_safety_filter
    from chc.plan import CausalPlan, certify_safety

    certificate = certify_safety(
        CausalPlan(
            actions=plan.actions[:, None],
            trajectory=plan.states[:, None],
            task_cost=plan.task_cost,
            uncertainty_tube=None,
            certified_horizon=None,
        ),
        _horizon_dynamics(fit, covariates, dt),
        lambda x: x[0] - floor,
        dt,
        alpha=audit.alpha,
        gamma=audit.gamma,
        cvar_gap=1.0,
        u_max=case.action_hi,
    )
    temp, nominal = jnp.asarray(plan.states[0]), float(plan.actions[0])
    zero = jnp.zeros(())
    drift = float(fit.rate(temp, covariates[0], zero))
    channel = float(jax.grad(lambda u: fit.rate(temp, covariates[0], u))(zero))
    allowed = robust_safety_filter(
        nominal, channel, audit.radius, case.action_hi, drift, audit.alpha * (float(temp) - floor)
    )
    return StepAudit(
        margin=float(temp) - floor,
        guaranteed=float(certificate.guaranteed_derivative[0]),
        required=float(certificate.required[0]),
        certified=bool(certificate.planned_certified[0]),
        horizon_share=certificate.certified_steps / plan.actions.shape[0],
        gamma_star=certificate.gamma_star,
        nominal=nominal,
        # The library's filter models a symmetric box ``|u| <= u_max``; a heat pump modulates on
        # [0, 1]. Intersecting with the physical box is what makes the returned action executable,
        # and where the intersection is empty it lands on the feasible endpoint that maximises the
        # guaranteed margin -- which above the confounded fit's 22.62 C sign flip is the pump off.
        filtered=float(np.clip(allowed, case.action_lo, case.action_hi)),
    )


def _audit_summary(audits: list[StepAudit]) -> dict[str, float]:
    """The episode's audit, as the numbers that let two arms of the ablation be compared."""
    moved = np.asarray([a.filtered - a.nominal for a in audits])
    ceilings = np.asarray([a.gamma_star for a in audits])
    undefined = bool(np.all(np.isnan(ceilings)))
    return {
        "cert_uncertified": float(np.mean([not a.certified for a in audits])),
        "cert_slack_mean": float(np.mean([a.guaranteed - a.required for a in audits])),
        "cert_horizon_share": float(np.mean([a.horizon_share for a in audits])),
        "cert_below_floor": float(np.mean([a.margin < 0.0 for a in audits])),
        "cert_gamma_star_median": float("nan") if undefined else float(np.nanmedian(ceilings)),
        "cert_gamma_star_undefined": float(np.mean(np.isnan(ceilings))),
        "cert_filter_share": float(np.mean(np.abs(moved) > 1e-9)),
        "cert_filter_mean_abs": float(np.mean(np.abs(moved))),
    }


def run_control_episode(
    client: BOPTestClient,
    case: BoptestCase,
    fit: PlantModel,
    log: LoggedEpisode,
    *,
    steps: int = 336,
    step_s: float = 1800.0,
    horizon: int = 16,
    w_comfort: float = 800.0,
    margin: float = 1.5,
    iterations: int = 600,
    audit: SafetyAudit | None = None,
) -> dict[str, float]:
    """Run the comfort MPC built on ``fit`` and return BOPTEST's KPIs, wall clock and saturation.

    ``at_bound`` is the share of steps whose commanded action sat on an actuator limit. It is the
    number that tells a reader how much of the episode the *model* actually decided: where the
    actuator is saturated the command is set by the box, not by the fitted channel, so a better
    channel cannot pay there. Without it, a closed-loop null result is unreadable -- it could mean
    identification does not matter, or it could mean this operating point never let it matter.

    ``audit`` runs the plan -> certificate half of the CHC spine on every solve and adds the
    ``cert_*`` keys; see :class:`SafetyAudit` for what it prices and when it is allowed to act. It
    defaults to ``None``, which is not a mode but the *absence* of one: the loop below is then
    byte-identical to the one every number in ``results/boptest_causal.md`` was measured on.

    Raises :class:`RunawayDriftError` rather than planning against a fit whose drift is unstable;
    see that class for why it is checked here and not in the solver.
    """
    span = case.action_hi - case.action_lo
    worst = max(fit.decay(case.action_lo + span * i / 8.0) for i in range(9))
    if worst >= 0.0:
        raise RunawayDriftError(case.testcase, worst, fit.step_response())
    dt = step_s / 3600.0
    solve = _mpc_solver(fit, case, dt, w_comfort=w_comfort, margin=margin, iterations=iterations)
    testid = client.select(case.testcase)
    started = time.perf_counter()
    clipped = 0
    audits: list[StepAudit] = []
    try:
        client.set_step(testid, step_s)
        measurements = client.initialize(testid, 0.0, 0.0)
        points = ["TDryBul", "HGloHor", LOWER_SETP]
        for _ in range(steps):
            forecast = client.forecast(testid, points, horizon * step_s, step_s)
            window = slice(1, horizon + 1)
            bounds = jnp.asarray(forecast[LOWER_SETP][window]) - KELVIN
            # the same three columns, in the same order, that `log_episode` standardised
            covariates = log.standardise(
                np.column_stack(
                    [
                        np.asarray(forecast["TDryBul"][window]) - KELVIN,
                        np.asarray(forecast["HGloHor"][window]) / 1000.0,
                        np.asarray(forecast[LOWER_SETP][window]) - KELVIN,
                    ]
                )
            )
            plan = solve(measurements[case.zone_point] - KELVIN, bounds, covariates)
            action = float(plan.actions[0])
            if audit is not None:
                priced = _audit_plan(
                    fit, case, plan, covariates, float(forecast[LOWER_SETP][0]) - KELVIN, dt, audit
                )
                audits.append(priced)
                if audit.enforce:
                    action = priced.filtered
            tolerance = 1e-6 * max(case.action_hi - case.action_lo, 1.0)
            clipped += bool(
                action <= case.action_lo + tolerance or action >= case.action_hi - tolerance
            )
            measurements = client.advance(testid, case.overwrite(action))
        kpis = {k: float(v) for k, v in client.kpi(testid).items() if isinstance(v, (int, float))}
    finally:
        client.stop(testid)
    kpis["wall_s"] = time.perf_counter() - started
    kpis["at_bound"] = clipped / steps
    if audits:
        kpis.update(_audit_summary(audits))
    return kpis


def run_pareto(
    client: BOPTestClient,
    case: BoptestCase,
    log: LoggedEpisode,
    *,
    margins: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
    control_steps: int = 336,
    step_s: float = 1800.0,
    progress: Callable[[str, dict[str, float]], None] | None = None,
    **control: Any,
) -> dict[str, list[dict[str, float]]]:
    """Trace each arm's comfort/energy frontier by sweeping the MPC's safety ``margin``.

    Comparing two controllers at a *single* operating point is a category error when the model error
    itself moves the operating point, and that is what happens here. The comfort term dominates the
    objective, so a controller that believes the plant has little authority commands more action
    than one that knows better: the attenuated channel acts as an unintended safety margin. At one
    fixed ``margin`` the biased arm therefore looks *better* on thermal discomfort and worse on
    energy -- which says nothing about whether the model is right.

    Sweeping the margin separates the two. Conservatism becomes an explicit, requested quantity, and
    the question becomes the one worth asking: for the same thermal discomfort, which model buys it
    with less energy? A frontier that dominates is a claim about the model. A single point is not.

    ``progress`` is called with ``(arm, point)`` after each episode. A full sweep is a dozen
    emulator episodes; without a hook the caller cannot see where it is, and loses every completed
    point if the run is killed.
    """
    frontier: dict[str, list[dict[str, float]]] = {}
    for adjusted in (False, True):
        fit = fit_thermal(log, adjusted=adjusted)
        name = "adjusted" if adjusted else "naive"
        for margin in margins:
            kpis = run_control_episode(
                client,
                case,
                fit,
                log,
                steps=control_steps,
                step_s=step_s,
                margin=margin,
                **control,
            )
            point = {"margin": margin, "step_response_8h": fit.step_response(), **kpis}
            frontier.setdefault(name, []).append(point)
            if progress is not None:
                progress(name, point)
    return frontier


def overlap_report(log: LoggedEpisode) -> dict[str, float]:
    """What survives of the action once the covariates are partialled out: the identifying budget.

    ``residual_share`` is the fraction of action variance surviving the projection on
    ``(T, weather)``. Identification rides entirely on it; at zero the moment has no regressor left
    and no sample size helps.
    """
    fit = fit_thermal(log, adjusted=True)
    total = float(jnp.var(log.action))
    return {
        "action_variance": total,
        "residual_variance": fit.action_residual_variance,
        "residual_share": fit.action_residual_variance / total if total > 0 else float("nan"),
        "nuisance_r2_action": fit.nuisance_r2_action,
        "at_bound": log.at_bound,
    }


def run_case(
    base_url: str = DEFAULT_URL,
    case_name: str = "heat_pump",
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    id_steps: int = 960,
    control_steps: int = 336,
    step_s: float = 1800.0,
    **control: Any,
) -> dict[str, Any]:
    """The full 2x2 plus BOPTEST's baseline, over ``seeds``, on one emulator case.

    Every arm sees the same model class, the same MPC and the same iteration count; the
    identification episodes are the same length for both policies, so no arm is bought extra data
    or extra compute.
    """
    case = CASES[case_name]
    client = BOPTestClient(base_url)
    baseline = run_episode(
        client,
        case.testcase,
        baseline_controller(),
        step_s=step_s,
        horizon_steps=control_steps,
    )
    arms: dict[str, list[dict[str, float]]] = {}
    fits: dict[str, list[ThermalFit]] = {}
    overlap: list[dict[str, float]] = []
    for seed in seeds:
        logs = {
            policy: log_episode(
                client, case, policy=policy, seed=seed, steps=id_steps, step_s=step_s
            )
            for policy in ("reset", "prbs")
        }
        overlap.append(overlap_report(logs["reset"]))
        for policy, adjusted in (
            ("reset", False),
            ("reset", True),
            ("prbs", False),
            ("prbs", True),
        ):
            name = f"{policy}-{'adjusted' if adjusted else 'naive'}"
            fit = fit_thermal(logs[policy], adjusted=adjusted)
            fits.setdefault(name, []).append(fit)
            # An arm can be identified and still be unplannable, which is a result rather than an
            # error: recorded per arm so one runaway fit does not abort the other three.
            try:
                kpis = run_control_episode(
                    client, case, fit, logs[policy], steps=control_steps, step_s=step_s, **control
                )
            except RunawayDriftError as unplannable:
                kpis = {"unplannable": 1.0, "pole": unplannable.pole, "rise_8h": unplannable.rise}
            arms.setdefault(name, []).append(kpis)
    return {
        "case": case_name,
        "testcase": case.testcase,
        "control_affine_in_action": case.control_affine_in_action,
        "seeds": list(seeds),
        "id_steps": id_steps,
        "control_steps": control_steps,
        "baseline": {k: float(v) for k, v in baseline.items() if isinstance(v, (int, float))},
        "arms": arms,
        "fits": fits,
        "overlap": overlap,
    }


def run_structure_ablation(
    base_url: str = DEFAULT_URL,
    case_name: str = "heat_pump",
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    id_steps: int = 960,
    control_steps: int = 336,
    step_s: float = 1800.0,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
    **control: Any,
) -> list[dict[str, Any]]:
    """Physics-off ablation: affine-naive, affine-adjusted and black-box, one confounded log each.

    Three arms on the *same* observational episode per seed, so nothing separates them but the model
    -- same MPC, same horizon, same iteration count, same objective, and the black box is given the
    same covariates plus far more fitting compute. One row per ``(seed, arm)``, carrying both the
    identification numbers and the closed-loop KPIs, because the interesting comparison spans them:
    on the synthetic fixture the black box matches the structured fit's held-out prediction to
    within a third of a seed standard deviation while missing the channel by 3x the RMSE.

    Deliberately separate from :func:`run_case` rather than a fifth arm inside it. This ablation
    needs only the reset log, so it costs three episodes a seed instead of six, and folding it in
    would have changed the shape of a result already recorded in ``results/boptest_causal.md``.
    """
    case = CASES[case_name]
    client = BOPTestClient(base_url)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        log = log_episode(client, case, policy="reset", seed=seed, steps=id_steps, step_s=step_s)
        models: dict[str, PlantModel] = {
            "affine-naive": fit_thermal(log, adjusted=False),
            "affine-adjusted": fit_thermal(log, adjusted=True),
            "neural": fit_neural(log, seed=seed),
        }
        scores = predictive_comparison(log, seed=seed)
        for name, model in models.items():
            try:
                kpis = run_control_episode(
                    client, case, model, log, steps=control_steps, step_s=step_s, **control
                )
            except RunawayDriftError as unplannable:
                kpis = {"unplannable": 1.0, "pole": unplannable.pole, "rise_8h": unplannable.rise}
            row = {
                "seed": seed,
                "arm": name,
                "step_response_8h": model.step_response(),
                "authority": model.authority(),
                "pole": model.pole,
                "holdout_mse": scores[name],
                **kpis,
            }
            rows.append(row)
            if progress is not None:
                progress(name, row)
    return rows


def run_overlap_ablation(
    base_url: str = DEFAULT_URL,
    case_name: str = "heat_pump",
    *,
    scales: tuple[float, ...] = (0.25, 0.12, 0.04, 0.01, 0.0),
    seed: int = 0,
    id_steps: int = 480,
    step_s: float = 1800.0,
) -> list[dict[str, float]]:
    """Shrink the exploration noise to zero and watch identification die with the overlap.

    The point is that adjustment is not magic: with a *deterministic* outdoor-reset policy the
    action is an exact function of the covariates, so partialling out leaves nothing to regress on.
    This is the assumption a reader should check before believing any of the numbers above, so it
    gets its own falsifiable curve rather than a sentence.
    """
    case = CASES[case_name]
    client = BOPTestClient(base_url)
    rows = []
    for scale in scales:
        log = log_episode(
            client,
            replace(case, explore_scale=scale),
            policy="reset",
            seed=seed,
            steps=id_steps,
            step_s=step_s,
        )
        fit = fit_thermal(log, adjusted=True)
        rows.append(
            {
                "explore_scale": scale,
                **overlap_report(log),
                "step_response_8h": fit.step_response(),
                "channel_b0": fit.channel[0],
            }
        )
    return rows


def run_certificate_ablation(
    base_url: str = DEFAULT_URL,
    case_name: str = "heat_pump",
    *,
    seed: int = 0,
    id_steps: int = 960,
    control_steps: int = 336,
    step_s: float = 1800.0,
    radius: float = 0.073,
    alpha: float = 1.0,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
    **control: Any,
) -> list[dict[str, Any]]:
    """Certificate off against certificate on, on the confounded and the de-confounded fit alike.

    The 2x2 the spine was missing an empirical answer for: ``fit -> plan -> certify -> act``, closed
    on a real emulator, with the certificate as the only thing that changes between arms.

    Falsifiable in both directions, which is the only reason to run it rather than assert it: the
    audit should find the confounded fit harder to certify, and one that separated the arms
    *equally* would be reading something other than the confounding. §40's threshold is undefined
    exactly where ``deficit > u_max * |channel|``, so what the diagnostic tracks is the authority
    the model *believes* at the operating point -- and attenuating that authority is what
    confounding does here. Measured on two seeds in §9 of the results doc, monotone over all four
    fits.

    Which seed is run matters for the sharpest version of it. On ``seed=1`` the confounded channel
    ``+2.989 - 0.1321*T`` crosses zero at 22.62 C -- inside the occupied band, and within 0.2 K of
    where this MPC targets -- so its believed authority collapses exactly where the controller
    lives; ``seed=0`` crosses at 28.6 C and 47.4 C, both outside it. Both are reported, because a
    diagnostic that only works on the seed it was designed against is not a diagnostic.

    One confounded 20-day log with both fits taken off it -- the same setup as the
    ``reset-adjusted`` and ``reset-naive`` rows of §4 -- so the certificate-off arms are comparable
    to numbers already published rather than to a fresh draw.
    """
    case = CASES[case_name]
    client = BOPTestClient(base_url)
    log = log_episode(client, case, policy="reset", seed=seed, steps=id_steps, step_s=step_s)
    rows: list[dict[str, Any]] = []
    for adjusted in (True, False):
        fit = fit_thermal(log, adjusted=adjusted)
        arm = "adjusted" if adjusted else "naive"
        # Where a bilinear channel changes sign the identified effect can no longer be signed, so
        # the certified interval is empty at any radius. Reported per arm because it is the
        # mechanism the ablation is meant to expose, not a derived statistic.
        flip = -fit.channel[0] / fit.channel[1] if fit.channel[1] < 0.0 else float("inf")
        for enforce in (False, True):
            try:
                kpis = run_control_episode(
                    client,
                    case,
                    fit,
                    log,
                    steps=control_steps,
                    step_s=step_s,
                    audit=SafetyAudit(radius=radius, alpha=alpha, enforce=enforce),
                    **control,
                )
            except RunawayDriftError as unplannable:
                kpis = {"unplannable": 1.0, "pole": unplannable.pole, "rise_8h": unplannable.rise}
            row = {
                "seed": seed,
                "arm": arm,
                "certificate": "on" if enforce else "off",
                "channel_b0": fit.channel[0],
                "channel_b1": fit.channel[1],
                "authority": fit.authority(),
                "sign_flip_c": flip,
                **kpis,
            }
            rows.append(row)
            if progress is not None:
                progress(f"{arm}-{'on' if enforce else 'off'}", row)
    return rows


def summarise(result: Mapping[str, Any]) -> dict[str, Any]:
    """Mean and bootstrap-free min/max across seeds for the KPIs that decide the comparison."""
    keys = ("tdis_tot", "ener_tot", "cost_tot", "emis_tot", "pele_tot", "at_bound", "wall_s")
    out: dict[str, Any] = {"baseline": {k: result["baseline"].get(k) for k in keys[:-1]}}
    for name, episodes in result["arms"].items():
        row: dict[str, Any] = {}
        for key in keys:
            values = [ep[key] for ep in episodes if key in ep]
            if values:
                row[key] = (float(np.mean(values)), float(np.min(values)), float(np.max(values)))
        rises = [fit.step_response() for fit in result["fits"][name]]
        row["step_response_8h"] = (
            float(np.mean(rises)),
            float(np.min(rises)),
            float(np.max(rises)),
        )
        poles = [fit.drift[1] for fit in result["fits"][name]]
        row["pole"] = (float(np.mean(poles)), float(np.min(poles)), float(np.max(poles)))
        out[name] = row
    return out


TRACK = "D-causal-identification"
METRIC = "step_response_error_8h"


def track_boptest_causal(
    base_url: str = DEFAULT_URL,
    case_name: str = "heat_pump",
    *,
    seeds: tuple[int, ...] = (0,),
    id_steps: int = 960,
    step_s: float = 1800.0,
) -> list[TrackResult]:
    """Score each arm by how far its 8-hour control authority sits from the randomised reference.

    No ground-truth channel exists on an emulator, so the reference is *identification by design*:
    the randomised (PRBS) log, fitted without adjustment because a randomised action needs none.
    Every other arm is scored by absolute disagreement with it, in Kelvin of 8-hour step response.
    Three things are falsifiable at once -- the confounded arm must land far away, adjustment must
    bring it back, and adjustment must leave the randomised arm where it was.

    The reference arm is not a row: it scores zero by construction, and a leaderboard ranks
    competitors, not the yardstick. Its value comes back from :func:`run_case`.

    Identification only, deliberately. Closed-loop KPIs are **not** a track metric here -- which way
    a channel error moves the controller depends on the cost (see the module docstring), so a single
    operating point cannot rank models. :func:`run_pareto` traces the frontier that can.

    This track needs a live emulator and says so loudly rather than returning an empty list: a
    silently skipped track reads exactly like a track with nothing to report.
    """
    if not is_available(base_url):
        raise RuntimeError(
            f"no BOPTEST-Service answered at {base_url}; this track scores a live emulator and "
            "cannot be run offline (see causaldyn_bench.boptest for the compose stack)"
        )
    case = CASES[case_name]
    client = BOPTestClient(base_url)
    rises: dict[str, list[float]] = {}
    for seed in seeds:
        for policy in ("reset", "prbs"):
            log = log_episode(client, case, policy=policy, seed=seed, steps=id_steps, step_s=step_s)
            for adjusted in (False, True):
                name = f"{policy}-{'adjusted' if adjusted else 'naive'}"
                rises.setdefault(name, []).append(
                    fit_thermal(log, adjusted=adjusted).step_response()
                )
    reference = float(np.mean(rises.pop("prbs-naive")))
    return [
        TrackResult(TRACK, method, METRIC, abs(float(np.mean(values)) - reference))
        for method, values in sorted(rises.items())
    ]


def run_pessimism_sweep(
    base_url: str = DEFAULT_URL,
    case_name: str = "heat_pump",
    *,
    shrinks: tuple[float, ...] = (0.0, 0.0733, 0.1466, 0.2199),
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    id_steps: int = 960,
    control_steps: int = 336,
    step_s: float = 1800.0,
    **control: Any,
) -> list[dict[str, float]]:
    """Shrink the de-confounded channel and re-run the closed loop, one row per ``(seed, shrink)``.

    This arm exists because of what the 2x2 measured, not because pessimism is fashionable.
    De-confounding recovered the *mean* 8-hour authority on the confounded log to within 3% of the
    randomised reference -- and did it with roughly 2.4x the reference's seed-to-seed spread. A
    comfort-constrained MPC is asymmetric in that spread: the seed whose channel came out high
    believes the plant strong, commands too little, and banks discomfort it cannot claw back. So the
    de-confounded *point* estimate was the worst closed-loop arm despite being the second-best
    estimate. Removing bias is necessary and not sufficient; the radius has to be spent.

    The default grid is 0, 1, 2 and 3 *measured* standard errors of the quantity the controller
    uses: the 8-hour step response varies by 0.487 K across the five logging seeds, and the fit
    gives 6.64 K of rise per unit ``b0``, so one standard error is a shrink of 0.073. It is
    emphatically **not** ``channel_error``, which over-states that by 6.9x on this plant -- see
    :meth:`ThermalFit.pessimistic` for the measurement that ruled it out.

    Measured on this plant the answer is that **zero is the best setting on the grid**, on the mean
    and on the spread at once -- see ``results/boptest_causal.md`` §5. The sweep is kept because a
    null result at a *correctly scaled* radius is the thing worth reporting; an earlier version read
    the radius as 0.055 and called one standard error a marginal improvement.
    """
    if not is_available(base_url):
        raise RuntimeError(
            f"no BOPTEST-Service answered at {base_url}; this sweep needs a live emulator"
        )
    case = CASES[case_name]
    client = BOPTestClient(base_url)
    rows: list[dict[str, float]] = []
    for seed in seeds:
        log = log_episode(client, case, policy="reset", seed=seed, steps=id_steps, step_s=step_s)
        fit = fit_thermal(log, adjusted=True)
        for shrink in shrinks:
            shrunk = fit.pessimistic(shrink)
            kpis = run_control_episode(
                client, case, shrunk, log, steps=control_steps, step_s=step_s, **control
            )
            rows.append(
                {
                    "seed": float(seed),
                    "shrink": shrink,
                    "channel_b0": shrunk.channel[0],
                    "channel_error": fit.channel_error or float("nan"),
                    "step_response_8h": shrunk.step_response(),
                    **kpis,
                }
            )
    return rows

"""Track M v2, geo selection: does choosing where a lift test runs by the plan's regret beat
choosing it as the literature does, by representativeness, by fit, or at random?

**Pre-registered 2026-10-01, before any scored world ran.** The design was piloted on seeds no
score reads (``PILOTS``: ``just track-m2-geo-pilot``, into ``results/track_m2_geo_pilot.md``).
This docstring, the code and the pilot's results are committed together, and only then does
``just track-m2-geo`` run. The pilot ran twice. The first read one placebo window for the choice
and for the prediction both, and the readings of the sets it chose erred within the placebo's band
in 0.887 of readout weeks, the least-RMSPE arm's in 0.806, a random set's in 0.936: a choice made
on a noisy placebo selects its luck. The screen, a window the prediction does not read, and the
kill's reading against the random arm's coverage in place of a fixed 0.93 were written after it,
the sample raised from 200 worlds to 500 on the second run's spread, and the pilot run again.

**The question** (the library's ADR 0040). A go-dark test on a set of geos reads the tested
channel's curve over the spend those geos run at. Where geos differ in spend per head, a set of
heavy spenders reads the curve's bend and a set of light ones little more than its slope, and a
plan past the tested spend needs the bend. :func:`chc.allocation.decision_weight` prices a
parameter's error in the plan's own regret, so a set can be chosen by the regret its test is
expected to leave. The literature chooses otherwise: Abadie and Zhao's (2021) synthetic design
picks the set that represents the market, and practice picks the set its synthetic control fits
best.

**The world** is a geo panel over Track M v2's market (a drawn world,
:func:`causaldyn_bench.mmm_decision.drawn`): the market's weekly spend and sales before media are
his, and 40 geos share them. Each geo has a population share (Dirichlet, concentration 4) and, on
each channel, a spend per head relative to the market's (log-normal, log-sd 0.5, its
population-weighted mean 1). A geo's sales per head are the market's before media, moved by two
common AR(1) factors (persistence 0.9, shocks of 2 percent) through loadings of its own, plus each
channel's effect at its own spend per head, ``beta tanh(s_g a_t / K)``, ``a`` the channel's
adstock, plus noise that falls with the geo's size. The market's channel is the geos' sum,
``beta sum_g p_g tanh(s_g a / K)``, a curve :class:`GeoMixture` holds: a new generator, not
Heusch's.

**The test** is his go-dark design on paid shopping, four tests of four weeks from weeks 80, 100,
120 and 140, each read over four weeks before it and eight after, in a set of geos holding 9 to
11 percent of the population: the matched cost. The set's sales per head are read against a
synthetic control of the other geos (:func:`chc.scm.synthetic_control`), fitted on the 75 weeks
before the first readout. The gaps are fitted by :func:`chc.lift.fit_lift` with the set's own
mixture, the analyst knowing each geo's spend and population, and the market's paid shopping is
the same mixture over every geo at the fitted parameters. The other channels are known.

**The candidates**: the heaviest geos by paid shopping's spend per head, the lightest, and 60
random sets, each filled in its order while the share stays at or under 11 percent until it
reaches 9. Each is read before any test runs, on two windows of the pre-period: its synthetic
control's error on weeks 36 to 55, fitted on the weeks before them, the *screen*, and on weeks 56
to 75, fitted on the weeks before them, the *placebo*; and by how far its sales per head stray from
the market's over the pre-period.

**The prediction**: a set's *exposure*, ``tr(W (J'J)^-1) / 2`` per euro of the budget, the regret
its test is expected to leave per unit of its reading's noise variance, ``J`` the set's gaps'
Jacobian in paid shopping's retention, scale and coefficient and ``W`` paid shopping's block of the
decision weight, both read at 16 draws of paid shopping's parameters from the ranges the world
draws them from, and averaged. The covariance is the least-squares fit's own, with no prior: the
arms fit by least squares. The choice reads the exposure times the screen squared; the regret a
set's test is expected to leave is the exposure times the placebo squared, a window the choice did
not read.

**The arms**, each a set from the pool, its test read and fitted, the quarter planned by
:func:`chc.allocation.allocate` on the fitted channel and the known ones:

* *regret*, the least exposure times the screen squared, a tie, where the plan pins paid shopping
  at every draw, going to the least screen;
* *Abadie-Zhao*, the set whose sales per head track the market's closest over the pre-period;
* *least RMSPE*, the least screen;
* *random*, a set drawn from the pool;
* *heaviest*, the heaviest geos, reported: whether reading the bend needs ``W`` at all.

A fit that raises plays the status quo, and is counted; a fitted coefficient below nought is
planned as nought, as :mod:`causaldyn_bench.family_regret` plans it.

**The score**: the realised regret per euro of each arm's plan on the world's own channels, the
best plan's by :func:`causaldyn_bench.mmm_decision.oracle` on the market's mixtures, written apart
from :mod:`chc.allocation`; paired by world. And, for each arm, each world's share of readout weeks
whose reading errs by at most 1.96 placebo errors: whether the placebo holds for the set chosen.

**The gate**: over 500 worlds the regret arm's mean regret lies below Abadie-Zhao's and the random
arm's, each paired difference's 95 % interval under nought. **The kill**: the regret arm picks
Abadie-Zhao's set or the least-RMSPE set in half the worlds or more, or its coverage lies more than
0.02 under the random arm's, paired by world, the choice having selected noise its placebo does
not see. A random set's coverage is not held to 0.95: twenty weeks of autocorrelated error make a
noisy placebo.

**The sample**: 500 worlds from seed 50 000. The pilot's paired differences against the random
arm and Abadie-Zhao's have sds near 0.087 and 0.031 per euro, so 500 worlds read the two means to
about 0.008 and 0.003 either side.

**Predictions**, from the second pilot's 40 worlds, before the run:

* **against the random arm the gate is met**: the pilot's difference was -0.0356 [-0.0635,
  -0.0077] per euro, the regret arm lower in 24 worlds and higher in 9;
* **against Abadie-Zhao's it is a lean, not a forecast**: -0.0074 [-0.0173, +0.0025], lower in
  17 worlds, tied in 12 and higher in 11, and the medians alike, 0.0034 and 0.0033. The difference
  lives in a few worlds where a set's test reads paid shopping's curve as a line, its scale and
  coefficient running off together, and the plan runs paid shopping to the top of its box. The
  regret arm's sets read a line in 6 worlds of 40, Abadie-Zhao's in 8, the least-RMSPE and random
  arms' in 14. Met if the pilot's mean holds; not if the truth is under about half of it;
* **the kill does not fire**: the regret arm took Abadie-Zhao's set in 1 world of 40 and the
  least-RMSPE set in 5, and its coverage stood 0.028 [0.008, 0.048] above the random arm's;
* the regret arm beats the least-RMSPE arm, -0.0222 [-0.0380, -0.0064]: the weight moves the
  choice, not the precision alone; its sets run at 1.53 times the market's spend per head
  (median), between Abadie-Zhao's 0.97 and the heaviest's 2.16;
* the heaviest arm is the worst, 0.0535 per euro: its sets read a line least (3 of 40), but their
  synthetic controls fit worst (a median placebo of 77 per head, against 33 to 39), so the bend
  read alone does not pay;
* the expected regret under-states the realised: 0.0047 [0.0026, 0.0067] per euro expected of
  the regret arm's tests at the placebo, 0.0079 left. Reported, not gated: it ranks sets, and it is
  a local quadratic at independent weekly errors, where the synthetic control's errors are
  autocorrelated and a line reading is not local.

By construction (R19): the panel is a generator written here, not Heusch's: the geos, their spend
per head, factors and noise are this track's choices, and the market's channel is the geos'
mixture. The regret arm's draws come from the ranges the world draws paid shopping's parameters
from, so it averages over the distribution the truth came from, as no analyst can. Every arm plans
with the other channels known, and fits with each geo's spend per head and population known. The
coverage reads the generator's own gaps. Of the prior's draws 0.54 hold paid shopping at an end
of its box in the pilot, where the weight is nought.

Not in this run: choosing the channel, the spend change or the length of the test, MM6's other
questions; an augmented synthetic control, tried in a scratch prototype on three worlds and set
aside for the classic one; a prior read from the history, whose fit of paid shopping the prototype
found at a coefficient of nought or pinned at an end of the box.

Precision: ``JAX_ENABLE_X64=1``. The worlds are dealt to ``--workers`` processes; each is scored
apart from the others, so the split moves no number.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import time
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from chc.allocation import decision_weight
from chc.lift import GeoArm, LiftFit, LiftTest, fit_lift
from chc.response import Channel, GeometricAdstock, Saturation
from chc.scm import synthetic_control
from numpy.typing import ArrayLike
from scipy.stats import norm

from causaldyn_bench.budget_regret import OUT_OF_PLAN, Comparison, Mean, planned
from causaldyn_bench.endogenous_mmm import Curve, EndogenousMediaMix, MediaMixWorld, Series
from causaldyn_bench.mmm_decision import PLANNED, Plan, Quarter, drawn, oracle, worth

TRACK = "M2-geo-selection"
LEVEL = 0.95
TOLERANCE = 0.02
GEOS = 40
SHARE = (0.09, 0.11)  # a treated set's population share: the matched cost
POOL = 60  # random candidate sets, beside the heaviest and the lightest
STARTS = (80, 100, 120, 140)  # the tests' first weeks, all after the synthetic control's fit
PRE, TEST, COOLDOWN = 4, 4, 8  # his 2026a readout
HOLD = 20  # the pre-period's last weeks, held out for the in-time placebo
DRAWS = 16  # draws of the tested channel's parameters the expected regret averages over
TESTED = 0  # paid shopping's column
POPULATION = 4.0  # the Dirichlet's concentration for the geos' population shares
INTENSITY = 0.5  # the log-sd of a geo's spend per head over the market's
FACTORS, PERSISTENCE, SHOCK = 2, 0.9, 0.02  # the baseline's common factors, AR(1)
NOISE = 0.02  # a geo's noise per head at a share of 1/GEOS, over the market's mean weekly sales
SINGULAR = 1e12  # a scaled J'J past this condition leaves the set's regret unbounded
LINE = 1e6  # a fitted scale past this, in thousands of euros, reads the curve as a line
PANEL_STREAM, POOL_STREAM, PRIOR_STREAM = 11, 12, 13  # apart from the world's own draws
WORLDS = 500
FIRST = 50_000  # the scored seeds, apart from every other Track M v2 track's
PILOTS = range(900, 940)  # the seeds the design was set on
ARMS = ("regret", "Abadie-Zhao", "least RMSPE", "random", "heaviest")
GATED = ("Abadie-Zhao", "random")
SAME = 0.5  # the kill: the regret arm's set is another arm's in this share of worlds or more
FIRST_READ = STARTS[0] - 1 - PRE  # the first readout week, from 0: the pre-period's end


class GeoMixture(Saturation):
    """``sum_g w_g tanh(s_g z)``: a set of geos' curve per head, ``w`` their population shares
    within the set and ``s`` their spend per head over the set's. Concave from zero."""

    scale: jax.Array = eqx.field(converter=lambda value: jnp.asarray(value, dtype=float))
    weights: tuple[float, ...] = eqx.field(static=True)
    intensities: tuple[float, ...] = eqx.field(static=True)

    def standard(self, z: jax.Array) -> jax.Array:
        spread = jnp.asarray(z)[..., None] * jnp.asarray(self.intensities)
        # chc.response.Tanh's form, which rises where XLA's tanh falls by an ulp
        rising = -jnp.expm1(-2.0 * spread) / (1.0 + jnp.exp(-2.0 * spread))
        return jnp.sum(jnp.asarray(self.weights) * rising, axis=-1)

    def _standard_inflection(self) -> float:
        return 0.0


def mixture_curve(population: Series, intensity: Series) -> Curve:
    """The market's mixture as :mod:`causaldyn_bench.mmm_decision` reads a curve: of the adstock
    ``a`` and his ``lambda``, ``tanh(lambda a / 2)`` summed over the geos."""

    def value(adstock: Series, lam: float) -> Series:
        return np.tanh(np.asarray(adstock)[..., None] * intensity * lam / 2.0) @ population

    def slope(adstock: Series, lam: float) -> Series:
        spread = np.asarray(adstock)[..., None] * intensity * lam / 2.0
        return np.cosh(spread) ** -2.0 @ (population * intensity * lam / 2.0)

    return Curve(value, slope, concave=True)


def _kernel(retention: float, length: int) -> Series:
    weights = retention ** np.arange(length)
    return weights / weights.sum()


def dark(spend: Series) -> Series:
    """The spend with the tests' weeks dark."""
    out = spend.copy()
    for start in STARTS:
        out[start - 1 : start - 1 + TEST] = 0.0
    return out


def readouts() -> list[slice]:
    return [slice(s - 1 - PRE, s - 1 + TEST + COOLDOWN) for s in STARTS]


@dataclass(frozen=True)
class GeoPanel:
    """One world's geos: the market's history, and what each geo's sales per head are made of."""

    world: MediaMixWorld
    population: Series  # (geos,) shares, summing to 1
    intensity: Series  # (geos, channels) spend per head over the market's; weighted mean 1
    baseline: Series  # (geos, weeks) sales per head before media
    noise: Series  # (geos, weeks) per head

    @classmethod
    def draw(cls, seed: int) -> GeoPanel:
        world = drawn(seed).simulate(seed)
        rng = np.random.default_rng((PANEL_STREAM, seed))
        population = rng.dirichlet(np.full(GEOS, POPULATION))
        intensity = np.exp(rng.normal(0.0, INTENSITY, (GEOS, len(world.channels))))
        intensity = intensity / (population @ intensity)
        weeks = world.week.size
        shocks = rng.normal(0.0, SHOCK, (FACTORS, weeks))
        factors = np.zeros((FACTORS, weeks))
        for t in range(weeks):
            factors[:, t] = (PERSISTENCE * factors[:, t - 1] if t else 0.0) + shocks[:, t]
        loadings = rng.normal(0.0, 1.0, (GEOS, FACTORS))
        baseline = world.premedia[None, :] * (1.0 + loadings @ factors)
        spread = NOISE * float(np.mean(world.sales)) / np.sqrt(population * GEOS)
        noise = rng.normal(0.0, 1.0, (GEOS, weeks)) * spread[:, None]
        return cls(world, population, intensity, baseline, noise)

    def media(self, column: int, spend: Series, geos: ArrayLike | None = None) -> Series:
        """``(geos, weeks)``: each geo's effect per head of ``spend`` at the market's scale."""
        w = self.world
        chosen = np.arange(GEOS) if geos is None else np.asarray(geos)
        adstock = np.convolve(spend, _kernel(w.retention[column], w.kernel_length))[: spend.size]
        spread = np.outer(self.intensity[chosen, column], adstock) * w.saturation[column] / 2.0
        return w.effect[column] * np.tanh(spread)

    def sales(self, dark_geos: ArrayLike = ()) -> Series:
        """``(geos, weeks)`` per head, the geos in ``dark_geos`` dark on paid shopping in the
        tests' weeks."""
        sales = self.baseline + self.noise
        for column in range(len(self.world.channels)):
            sales = sales + self.media(column, self.world.spend[:, column])
        geos = np.asarray(dark_geos, dtype=int)
        if geos.size:
            spend = self.world.spend[:, TESTED]
            sales[geos] += self.media(TESTED, dark(spend), geos) - self.media(TESTED, spend, geos)
        return sales

    def curves(self) -> list[Curve]:
        return [mixture_curve(self.population, column) for column in self.intensity.T]

    def channel(self, column: int, retention: float, scale: float, coefficient: float) -> Channel:
        """The market's channel at the given parameters: the mixture over every geo."""
        return Channel(
            GeometricAdstock(retention, length=self.world.kernel_length, normalized=True),
            GeoMixture(
                scale,
                tuple(self.population.tolist()),
                tuple(self.intensity[:, column].tolist()),
            ),
            coefficient,
        )

    def truth(self) -> tuple[Channel, ...]:
        w = self.world
        return tuple(
            self.channel(c, w.retention[c], 2.0 / w.saturation[c], w.effect[c]) for c in range(3)
        )

    def theta(self) -> tuple[float, float, float]:
        """Paid shopping's retention, scale and coefficient."""
        w = self.world
        return w.retention[TESTED], 2.0 / w.saturation[TESTED], w.effect[TESTED]


@dataclass(frozen=True)
class Candidate:
    name: str  # "heaviest", "lightest", "random 7"
    geos: tuple[int, ...]


def _fill(population: Series, order: ArrayLike) -> tuple[int, ...] | None:
    """The geos of ``order`` taken while the share stays within :data:`SHARE`'s ceiling, until it
    reaches the floor; ``None`` if the order runs out first."""
    chosen, share = [], 0.0
    for geo in np.asarray(order):
        if share + population[geo] <= SHARE[1]:
            chosen.append(int(geo))
            share += float(population[geo])
            if share >= SHARE[0]:
                return tuple(sorted(chosen))
    return None


def pool(panel: GeoPanel, seed: int) -> tuple[Candidate, ...]:
    order = np.argsort(-panel.intensity[:, TESTED], kind="stable")
    candidates = []
    for name, ranked in (("heaviest", order), ("lightest", order[::-1])):
        geos = _fill(panel.population, ranked)
        if geos is None:
            raise ValueError(f"no {name} set reaches {SHARE[0]} of the population")
        candidates.append(Candidate(name, geos))
    rng = np.random.default_rng((POOL_STREAM, seed))
    while len(candidates) < POOL + 2:
        geos = _fill(panel.population, rng.permutation(GEOS))
        if geos is not None:
            candidates.append(Candidate(f"random {len(candidates) - 2}", geos))
    return tuple(candidates)


@dataclass(frozen=True)
class Design:
    """A candidate as the pre-period reads it, before any test."""

    candidate: Candidate
    donors: tuple[int, ...]
    weights: Series  # the synthetic control's, fitted on the whole pre-period
    screen: float  # its error on the HOLD weeks before the last HOLD, fitted on the weeks before
    placebo: float  # its error on the pre-period's last HOLD weeks, fitted on the weeks before
    representative: float  # the set's sales per head against the market's, RMS over the pre-period

    def share(self, panel: GeoPanel) -> Series:
        """Each of the set's geos' share of the set's population."""
        within = panel.population[list(self.candidate.geos)]
        return within / within.sum()

    def spend_per_head(self, panel: GeoPanel) -> float:
        """The set's paid-shopping spend per head over the market's."""
        return float(self.share(panel) @ panel.intensity[list(self.candidate.geos), TESTED])


def design(panel: GeoPanel, sales: Series, candidate: Candidate) -> Design:
    geos = list(candidate.geos)
    donors = tuple(g for g in range(GEOS) if g not in candidate.geos)
    within = panel.population[geos] / panel.population[geos].sum()
    treated = within @ sales[geos]
    outcomes = np.vstack([treated[None, :], sales[list(donors)]])
    screened = synthetic_control(outcomes[:, : FIRST_READ - HOLD], 0, FIRST_READ - 2 * HOLD)
    held = synthetic_control(outcomes[:, :FIRST_READ], 0, FIRST_READ - HOLD)
    fitted = synthetic_control(outcomes, 0, FIRST_READ)
    market = panel.population @ sales
    representative = float(np.sqrt(np.mean((treated - market)[:FIRST_READ] ** 2)))
    return Design(
        candidate,
        donors,
        np.asarray(fitted.weights, dtype=float),
        float(np.sqrt(np.mean(np.asarray(screened.att) ** 2))),
        float(np.sqrt(np.mean(np.asarray(held.att) ** 2))),
        representative,
    )


def gaps(panel: GeoPanel, chosen: Design, theta: ArrayLike) -> Series:
    """The set's gaps per head over the readouts at paid shopping's ``theta``: its effect with the
    tests' weeks dark less without, the model :func:`chc.lift.fit_lift` fits."""
    retention, scale, coefficient = (float(v) for v in np.asarray(theta, dtype=float))
    spend = panel.world.spend[:, TESTED]
    kernel = _kernel(retention, panel.world.kernel_length)
    intensity = panel.intensity[list(chosen.candidate.geos), TESTED]

    def effect(series: Series) -> Series:
        adstock = np.convolve(series, kernel)[: series.size]
        return coefficient * (chosen.share(panel) @ np.tanh(np.outer(intensity, adstock) / scale))

    gap = effect(dark(spend)) - effect(spend)
    return np.concatenate([gap[window] for window in readouts()])


def jacobian(panel: GeoPanel, chosen: Design, theta: ArrayLike) -> Series:
    """``(readout weeks, 3)``: the gaps' slopes in retention, scale and coefficient, by central
    differences a millionth of each parameter wide."""
    at = np.asarray(theta, dtype=float)
    columns = []
    for k in range(3):
        step = 1e-6 * abs(at[k])
        up, down = at.copy(), at.copy()
        up[k] += step
        down[k] -= step
        columns.append((gaps(panel, chosen, up) - gaps(panel, chosen, down)) / (2.0 * step))
    return np.column_stack(columns)


def prior(seed: int) -> Series:
    """``(DRAWS, 3)``: paid shopping's retention, scale and coefficient drawn from the ranges
    :func:`causaldyn_bench.mmm_decision.drawn` draws every channel's from."""
    reference = EndogenousMediaMix()
    rng = np.random.default_rng((PRIOR_STREAM, seed))
    retention = rng.uniform(min(reference.retention), max(reference.retention), DRAWS)
    logged = [
        np.exp(rng.uniform(math.log(min(values)), math.log(max(values)), DRAWS))
        for values in (reference.saturation, reference.effect)
    ]
    lam, effect = logged
    return np.column_stack([retention, 2.0 / lam, effect])


def weights(panel: GeoPanel, quarter: Quarter, draws: Series) -> Series:
    """``(DRAWS, 3, 3)``: paid shopping's block of the decision weight at each draw, the other
    channels at their own parameters; nought where the plan holds paid shopping at an end."""
    names = [f"{TESTED}.kernel.retention", f"{TESTED}.curve.scale", f"{TESTED}.coefficient"]
    known = panel.truth()
    blocks = []
    for retention, scale, coefficient in draws:
        channels = list(known)
        channels[TESTED] = panel.channel(TESTED, retention, scale, coefficient)
        weight = decision_weight(
            channels,
            quarter.budget,
            PLANNED,
            lower=quarter.lower,
            upper=quarter.upper,
            history=quarter.history,
        )
        index = [weight.parameters.index(name) for name in names]
        blocks.append(weight.matrix[np.ix_(index, index)])
    return np.array(blocks)


def exposure(
    panel: GeoPanel, chosen: Design, draws: Series, blocks: Series, budget: float
) -> float:
    """``tr(W (J'J)^-1) / 2`` per euro, averaged over the draws: the regret the set's test is
    expected to leave per unit of its reading's noise variance, the least-squares covariance read
    in coordinates scaled by the draw so that the condition reads identification, not units."""
    total = 0.0
    for theta, block in zip(draws, blocks, strict=True):
        if not np.any(block):
            continue
        scaled = jacobian(panel, chosen, theta) * theta
        information = scaled.T @ scaled
        if not np.all(np.isfinite(information)) or np.linalg.cond(information) > SINGULAR:
            return math.inf
        covariance = np.linalg.inv(information) * np.outer(theta, theta)
        total += 0.5 * float(np.sum(block * covariance))
    return total / len(draws) / budget


def lift_tests(panel: GeoPanel, chosen: Design, sales: Series) -> tuple[LiftTest, ...]:
    """The set's four tests, per head: the treated arm its spend dark in the tests' weeks and its
    sales; the control its spend without the tests and the synthetic control's sales."""
    geos = list(chosen.candidate.geos)
    treated = chosen.share(panel) @ sales[geos]
    counterfactual = sales[list(chosen.donors)].T @ chosen.weights
    spend = chosen.spend_per_head(panel) * panel.world.spend[:, TESTED]
    before = panel.world.kernel_length - 1
    return tuple(
        LiftTest(
            GeoArm(dark(spend)[window.start - before : window.stop], treated[window]),
            GeoArm(spend[window.start - before : window.stop], counterfactual[window]),
        )
        for window in readouts()
    )


def template(panel: GeoPanel, chosen: Design, tests: Sequence[LiftTest]) -> Channel:
    """Where the fit starts, as Track M v2's: a retention of 0.5, the scale at the control
    readouts' mean spend; the curve the set's own mixture."""
    geos = list(chosen.candidate.geos)
    spend = np.concatenate([test.control.spend[test.control.history :] for test in tests])
    relative = panel.intensity[geos, TESTED] / chosen.spend_per_head(panel)
    return Channel(
        GeometricAdstock(0.5, length=panel.world.kernel_length, normalized=True),
        GeoMixture(
            float(np.mean(spend)), tuple(chosen.share(panel).tolist()), tuple(relative.tolist())
        ),
        1.0,
    )


def reading_error(panel: GeoPanel, chosen: Design, sales: Series) -> Series:
    """The set's reading over the readouts, treated less its synthetic control, less the gaps
    the test made: the synthetic control's own error, which the placebo predicts."""
    geos = list(chosen.candidate.geos)
    treated = chosen.share(panel) @ sales[geos]
    counterfactual = sales[list(chosen.donors)].T @ chosen.weights
    reading = np.concatenate([(treated - counterfactual)[window] for window in readouts()])
    return reading - gaps(panel, chosen, panel.theta())


@dataclass(frozen=True)
class ArmScore:
    candidate: str
    geos: tuple[int, ...]
    spend_per_head: float
    screen: float
    placebo: float
    exposure: float  # the expected regret per euro per unit of the reading's noise variance
    estimate: tuple[float, ...] | None  # retention, scale, coefficient; None when the fit raised
    failure: str | None
    weekly: tuple[float, ...]
    moved: float  # how far the plan was moved into the box, over the budget
    regret: float  # per euro
    error: float  # the reading's error, RMS over the readouts
    covered: int  # readout weeks whose reading errs by at most 1.96 placebo errors
    weeks: int

    @property
    def predicted(self) -> float:
        """The regret per euro the set's test is expected to leave, at the placebo the choice
        did not read."""
        return self.placebo**2 * self.exposure


@dataclass(frozen=True)
class WorldScore:
    seed: int
    budget: float
    best: float
    arms: dict[str, ArmScore]
    expected: dict[str, float]  # the pool's regrets as the choice reads them: quartiles, least,
    # and the share unbounded
    pinned: float  # the share of draws whose plan holds paid shopping at an end of its box


def _plan(panel: GeoPanel, quarter: Quarter, fit: LiftFit | None) -> tuple[Series, float]:
    if fit is None:
        return quarter.status_quo, 0.0
    read = dict(zip(fit.parameters, np.asarray(fit.estimate, dtype=float), strict=True))
    channels = list(panel.truth())
    channels[TESTED] = panel.channel(
        TESTED,
        float(read["kernel.retention"]),
        float(read["curve.scale"]),
        max(float(read["coefficient"]), 0.0),
    )
    weekly = planned(channels, quarter)
    if quarter.feasible(weekly, tolerance=OUT_OF_PLAN):
        return weekly, 0.0
    moved = quarter.project(weekly)
    return moved, float(np.abs(moved - weekly).sum()) * PLANNED / quarter.budget


def score_arm(
    panel: GeoPanel, quarter: Quarter, best: Plan, chosen: Design, exposed: float
) -> ArmScore:
    sales = panel.sales(chosen.candidate.geos)
    tests = lift_tests(panel, chosen, sales)
    fit: LiftFit | None
    try:
        fit = fit_lift(tests, template(panel, chosen, tests))
        failure = None
    except (RuntimeError, ValueError) as error:
        fit, failure = None, f"{type(error).__name__}: {error}"
    weekly, moved = _plan(panel, quarter, fit)
    value = worth(panel.world, quarter, weekly, panel.curves())
    error = reading_error(panel, chosen, sales)
    band = float(norm.ppf(0.5 + LEVEL / 2.0)) * chosen.placebo
    return ArmScore(
        candidate=chosen.candidate.name,
        geos=chosen.candidate.geos,
        spend_per_head=chosen.spend_per_head(panel),
        screen=chosen.screen,
        placebo=chosen.placebo,
        exposure=exposed,
        estimate=None if fit is None else tuple(float(v) for v in fit.estimate),
        failure=failure,
        weekly=tuple(float(v) for v in weekly),
        moved=moved,
        regret=(best.worth - value) / quarter.budget,
        error=float(np.sqrt(np.mean(error**2))),
        covered=int(np.sum(np.abs(error) <= band)),
        weeks=int(error.size),
    )


def choose(designs: Sequence[Design], exposures: Series, seed: int) -> dict[str, int]:
    """Each arm's set, as its index in ``designs``. The regret arm and the least-RMSPE arm read
    the screen, not the placebo, and the regret arm's ties, where the plan pins paid shopping at
    every draw, go to the least screen."""
    rng = np.random.default_rng((POOL_STREAM, seed, 1))
    names = [d.candidate.name for d in designs]
    screens = np.array([d.screen for d in designs])
    return {
        "regret": int(np.lexsort((screens, screens**2 * exposures))[0]),
        "Abadie-Zhao": int(np.argmin([d.representative for d in designs])),
        "least RMSPE": int(np.argmin(screens)),
        "random": int(rng.integers(len(designs))),
        "heaviest": names.index("heaviest"),
    }


def score_world(seed: int) -> WorldScore:
    panel = GeoPanel.draw(seed)
    quarter = Quarter.after(panel.world)
    best = oracle(panel.world, quarter, panel.curves())
    untested = panel.sales()
    designs = [design(panel, untested, c) for c in pool(panel, seed)]
    draws = prior(seed)
    blocks = weights(panel, quarter, draws)
    exposures = np.array([exposure(panel, d, draws, blocks, quarter.budget) for d in designs])
    picks = choose(designs, exposures, seed)
    scored: dict[int, ArmScore] = {}
    for index in dict.fromkeys(picks.values()):
        scored[index] = score_arm(panel, quarter, best, designs[index], float(exposures[index]))
    expected = np.array([d.screen**2 for d in designs]) * exposures
    finite = expected[np.isfinite(expected)]
    quartiles = np.percentile(finite, [25, 50, 75]) if finite.size else np.full(3, math.nan)
    return WorldScore(
        seed=seed,
        budget=quarter.budget,
        best=best.worth,
        arms={arm: scored[index] for arm, index in picks.items()},
        expected={
            "q25": float(quartiles[0]),
            "median": float(quartiles[1]),
            "q75": float(quartiles[2]),
            "least": float(np.min(expected)),
            "unbounded": float(np.mean(~np.isfinite(expected))),
        },
        pinned=float(np.mean([not np.any(b) for b in blocks])),
    )


@dataclass(frozen=True)
class Run:
    seeds: tuple[int, ...]
    scores: tuple[WorldScore, ...]
    seconds: float

    def regrets(self, arm: str) -> Series:
        return np.array([s.arms[arm].regret for s in self.scores])

    @property
    def means(self) -> dict[str, Mean]:
        return {arm: Mean.of(self.regrets(arm)) for arm in ARMS}

    @property
    def comparisons(self) -> tuple[Comparison, ...]:
        chosen = self.regrets("regret")
        return tuple(Comparison.of(a, chosen, self.regrets(a)) for a in ARMS if a != "regret")

    def same(self, arm: str) -> float:
        """The share of worlds where the regret arm's set is ``arm``'s."""
        return float(np.mean([s.arms["regret"].geos == s.arms[arm].geos for s in self.scores]))

    def coverage(self, arm: str) -> Series:
        """Each world's share of readout weeks within the placebo's band."""
        return np.array([s.arms[arm].covered / s.arms[arm].weeks for s in self.scores])

    @property
    def coverage_gap(self) -> Mean:
        """The regret arm's coverage less the random arm's, paired by world: what choosing on a
        screen costs the placebo the choice did not read."""
        return Mean.of(self.coverage("regret") - self.coverage("random"))

    def failures(self, arm: str) -> int:
        return sum(s.arms[arm].failure is not None for s in self.scores)

    def lines(self, arm: str) -> int:
        """The worlds where the arm's test read paid shopping's curve as a line."""
        estimates = [s.arms[arm].estimate for s in self.scores]
        return sum(e is not None and e[1] > LINE for e in estimates)

    @property
    def passes(self) -> bool:
        compared = {c.arm: c for c in self.comparisons}
        return (
            all(compared[arm].difference.interval[1] < 0.0 for arm in GATED)
            and self.same("Abadie-Zhao") < SAME
            and self.same("least RMSPE") < SAME
            and self.coverage_gap.mean >= -TOLERANCE
        )


def run(seeds: Sequence[int], pool_: Executor) -> Run:
    began = time.perf_counter()
    scores = tuple(pool_.map(score_world, seeds, chunksize=1))
    return Run(tuple(seeds), scores, time.perf_counter() - began)


def _markdown(result: Run, *, pilot: bool) -> str:
    n = len(result.scores)
    title = "Track M v2, geo selection" + (", pilot" if pilot else "")
    lines = [
        f"# {title} ({n} worlds from seed {result.seeds[0]}, float64)",
        "",
        "Regret per euro of the quarter's budget of the plan each arm's test leads to, paid "
        "shopping fitted from the test and the other channels known. Mean with its 95 % Student-t "
        "interval, and the median.",
        "",
    ]
    if pilot:
        lines += [
            "The pilot: the seeds the design was set on, apart from every scored one. No "
            "gate reads it.",
            "",
        ]
    lines += [
        "| arm | mean [95 %] | median | spend per head | screen | placebo | line | failed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, mean in result.means.items():
        chosen = [s.arms[arm] for s in result.scores]
        lines.append(
            f"| {arm} | {mean.mean:.4f} [{mean.interval[0]:.4f}, {mean.interval[1]:.4f}] | "
            f"{mean.median:.4f} | {np.median([a.spend_per_head for a in chosen]):.2f} | "
            f"{np.median([a.screen for a in chosen]):.1f} | "
            f"{np.median([a.placebo for a in chosen]):.1f} | {result.lines(arm)} | "
            f"{result.failures(arm)} |"
        )
    lines += [
        "",
        "Spend per head, screen and placebo are the chosen sets' medians: paid shopping's spend "
        "per head over the market's, and the synthetic control's in-time errors per head on the "
        "two windows, the screen the one the choice reads. Line counts the worlds whose fit ran "
        f"the scale past {LINE:.0e}, reading the curve as a line.",
        "",
        "The regret arm against each: its regret less the arm's over the same worlds, and the "
        "worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).",
        "",
        "| against | difference [95 %] | lower | tied | higher | lower share [95 %] | same set |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in result.comparisons:
        d, (low, high) = c.difference, c.lower_share
        lines.append(
            f"| {c.arm} | {d.mean:+.4f} [{d.interval[0]:+.4f}, {d.interval[1]:+.4f}] | "
            f"{c.lower} | {c.tied} | {c.higher} | {c.lower / n:.2f} [{low:.2f}, {high:.2f}] | "
            f"{result.same(c.arm):.2f} |"
        )
    lines += [
        "",
        "The reading's error over the readouts: each world's share of weeks within 1.96 of the "
        "chosen set's placebo errors, its mean over the worlds with a Student-t interval, and the "
        "median of the error's RMS over the placebo and over the screen.",
        "",
        "| arm | covered [95 %] | error / placebo | error / screen |",
        "|---|---|---|---|",
    ]
    for arm in ARMS:
        covered = Mean.of(result.coverage(arm))
        chosen = [s.arms[arm] for s in result.scores]
        over_placebo = np.median([a.error / a.placebo for a in chosen])
        over_screen = np.median([a.error / a.screen for a in chosen])
        lines.append(
            f"| {arm} | {covered.mean:.3f} [{covered.interval[0]:.3f}, "
            f"{covered.interval[1]:.3f}] | {over_placebo:.2f} | {over_screen:.2f} |"
        )
    gap = result.coverage_gap
    predicted = Mean.of(np.array([s.arms["regret"].predicted for s in result.scores]))
    pinned = float(np.mean([s.pinned for s in result.scores]))
    unbounded = float(np.mean([s.expected["unbounded"] for s in result.scores]))
    lines += [
        "",
        f"The regret arm's coverage less the random arm's, paired by world: {gap.mean:+.3f} "
        f"[{gap.interval[0]:+.3f}, {gap.interval[1]:+.3f}]. The regret its sets' tests were "
        f"expected to leave, at the placebo: {predicted.mean:.4f} [{predicted.interval[0]:.4f}, "
        f"{predicted.interval[1]:.4f}] per euro, against the {result.means['regret'].mean:.4f} "
        "they left.",
        "",
        f"Of the prior's draws, {pinned:.2f} hold paid shopping at an end of its box, where it "
        f"weighs nothing; of the pool's sets, {unbounded:.2f} leave the regret unbounded, their "
        "test unable to part the three parameters.",
    ]
    if not pilot:
        verdict = "met" if result.passes else "NOT met"
        lines += [
            "",
            f"**Gate** (the regret arm's mean below {' and '.join(GATED)}'s, each interval under "
            f"nought; its set another's in under {SAME:.0%} of worlds; its coverage at most "
            f"{TOLERANCE:.2f} under the random arm's): {verdict}.",
        ]
    return "\n".join(lines) + "\n"


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _record(result: Run, *, pilot: bool) -> dict[str, Any]:
    return _jsonable(
        {
            "track": TRACK,
            "pilot": pilot,
            "x64": True,
            "design": {
                "geos": GEOS,
                "share": list(SHARE),
                "pool": POOL,
                "starts": list(STARTS),
                "readout": [PRE, TEST, COOLDOWN],
                "hold": HOLD,
                "draws": DRAWS,
                "level": LEVEL,
                "tolerance": TOLERANCE,
                "same": SAME,
                "gated": list(GATED),
            },
            "gate": None if pilot else result.passes,
            "wall_clock": "seconds; recorded, not quoted: not measured on a clean machine",
            "seconds": result.seconds,
            "seeds": list(result.seeds),
            "worlds": [asdict(s) for s in result.scores],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="the design's seeds, not the scored")
    parser.add_argument("--worlds", type=int, default=WORLDS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--name", help="default track_m2_geo, with _pilot for the pilot")
    args = parser.parse_args()
    name = args.name or ("track_m2_geo_pilot" if args.pilot else "track_m2_geo")
    if not bool(jnp.zeros(()).dtype == jnp.float64):
        raise SystemExit("JAX_ENABLE_X64=1 is required: the arms were piloted at float64")
    # the pool is the parallelism: the workers inherit one BLAS thread each, whatever the shell set
    os.environ["OMP_NUM_THREADS"] = "1"
    seeds = list(PILOTS) if args.pilot else list(range(FIRST, FIRST + args.worlds))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context) as executor:
        result = run(seeds, executor)
    text = _markdown(result, pilot=args.pilot)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.md").write_text(text)
    (args.out / f"{name}.json").write_text(
        json.dumps(_record(result, pilot=args.pilot), separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(text)
    print(f"written to {args.out}/{name}.md and {args.out}/{name}.json")


if __name__ == "__main__":
    main()

# Track D-causal — does de-confounding the control channel pay on a real emulator?

Live BOPTEST-Service (`bestest_hydronic_heat_pump`) via `causaldyn_bench.boptest_causal`, Fedora /
Podman. 20-day identification episodes (960 steps), 7-day control episodes (336 steps), 30-minute
control step, 5 seeds. Every arm sees the same model class, the same MPC and the same iteration
count, and both logging policies get the same number of samples, so no arm is bought extra data or
extra compute.

Reproduce against a running service:

```bash
JAX_ENABLE_X64=1 BOPTEST_URL=http://127.0.0.1:8000 uv run python -c "
from causaldyn_bench.boptest_causal import run_case, summarise
print(summarise(run_case(seeds=(0,1,2,3,4))))"
```

## The question the previous harness could not ask

`causaldyn_bench.boptest_chc` identifies its thermal model from a **randomised** slow-PRBS episode.
That is a clean experiment, and it is the one thing production HVAC data never is: real logs come
from a controller, and every sensible controller is weather-compensated. Outdoor-reset curves are the
textbook design. So the logged action is a function of the outdoor temperature, which is also what
drives the zone:

```
u  <- outdoor reset      (colder outside => command more heat)
T' <- outdoor loss       (colder outside => temperature falls faster)
```

`Cov(u, eps) < 0`, and a regression of the rate on `(1, T, u)` credits the heating to a milder
outdoors rather than to the actuator. The experiment is a 2×2 over {outdoor-reset, randomised PRBS} ×
{adjust for weather, don't}, so **both** directions are falsifiable: adjustment must repair the
confounded arm *and* leave the randomised one alone.

There is **no ground-truth channel** on an emulator. The reference is identification *by design* —
the randomised log, fitted without adjustment, because a randomised action needs none.

## 1. Identification — the headline

8-hour open-loop step response of the zone to a unit step in heat-pump modulation, mean ± s.e. over
5 seeds. Reported instead of the steady-state gain `−b/a` because a 20-day window at half-hour
resolution does **not** identify a building's infinite-horizon gain: an early pass produced a pole of
`−3e-4` and hence a DC gain of 1493 K, which is a near-unit-root artefact of fitting one thermal
time constant to a plant that has at least two.

| arm | 8h step response (K) | range over seeds | fitted pole | \|deviation from reference\| |
|---|---:|---|---:|---:|
| **prbs-naive** (reference) | 4.429 ± 0.162 | [4.052, 4.852] | −0.0195 | — |
| prbs-adjusted | 4.427 ± 0.097 | [4.184, 4.682] | −0.0246 | **0.002** |
| reset-adjusted | 4.285 ± 0.385 | [3.301, 5.168] | −0.0421 | **0.144** |
| reset-naive | 2.187 ± 0.207 | [1.514, 2.738] | −0.0439 | **2.242** |

Three claims, all falsifiable, all confirmed:

1. **The confounded arm is badly wrong.** Logged by a weather-compensated controller, the naive fit
   understates the heat pump's 8-hour authority by **50.6%** (2.187 K against 4.429 K).
2. **Adjustment repairs it.** The orthogonal fit lands within 0.144 K of the randomised reference — a
   **15.6× reduction** in deviation — using the same 20 days of *observational* data.
3. **Adjustment does not distort a design that was already clean.** On the randomised log it moves
   the answer by 0.002 K, and *reduces* the spread (s.e. 0.097 against 0.162). An estimator that
   "helped" the randomised arm too would be distorting rather than de-confounding.

All four arms are stable (negative pole). That took a model fix, not a constraint — see §7.

This table is identification only, so it is untouched by the solver defect in §7 — the fits do not
involve the controller, and re-running them after the fix reproduced every entry. It did reproduce
them in a stronger sense than intended: the re-run was accidentally launched **without**
`JAX_ENABLE_X64`, and float32 returned 4.4292 / 4.4270 / 4.2848 / 2.1867 against the x64 table's
4.429 / 4.427 / 4.285 / 2.187, with poles matching to five digits. On the same rows before the
library's conditioning fix, float32 returned `nan`. An accident is not a designed experiment, but it
is the check that fix needed on real data rather than on a fixture.

## 2. Overlap — the assumption everything rests on

Identification rides entirely on the part of the action that survives partialling out the covariates.
On the reset log, per seed:

| seed | residual share of action variance | nuisance R²(action) | share of steps at an actuator bound |
|---|---:|---:|---:|
| 0 | 0.151 | 0.849 | 0.210 |
| 1 | 0.139 | 0.861 | 0.209 |
| 2 | 0.156 | 0.844 | 0.221 |
| 3 | 0.157 | 0.843 | 0.197 |
| 4 | 0.132 | 0.868 | 0.222 |

About **15%** of the action's variance is exogenous. That is the whole identifying budget. The ~21%
of steps sitting on a bound is the night setback driving the command to zero — what a real controller
does — and those steps carry no action information at all.

## 3. Overlap ablation — driving the budget to zero

Shrink the exploration noise and watch identification die. `run_overlap_ablation`, seed 0, 480 steps:

| exploration sd | residual share | nuisance R²(action) | 8h step response (K) | fitted `b₀` |
|---:|---:|---:|---:|---:|
| 0.25 | 0.441 | 0.559 | 4.131 | 0.982 |
| 0.12 | 0.143 | 0.857 | 4.447 | 1.720 |
| 0.04 | 0.020 | 0.980 | 5.613 | 1.458 |
| 0.01 | 0.004 | 0.996 | 5.249 | **−4.165** |
| 0.00 | 0.003 | 0.997 | 3.313 | **−5.327** |

Adjustment is not magic. Above a residual share of ~0.14 the fit sits near the reference (4.13, 4.45
against 4.43). Below 0.02 the moment has almost no regressor left, and by sd 0.01 the fitted channel
goes **negative** — the model says more heating cools the room. A perfectly deterministic reset
policy makes the action an exact function of the covariates and nothing is identified at any sample
size. This is the row to check before believing anything above it.

## 4. Closed loop — de-confounding pays, and the earlier "no" was the optimiser

Same MPC, same comfort margin (1.5 K), 5 seeds, 7-day episodes. `sat` is the share of control steps
whose command sat on an actuator limit.

| arm | `tdis_tot` (K·h) mean ± s.e. | range over seeds | `ener_tot` | `sat` | authority at 21 °C |
|---|---:|---|---:|---:|---:|
| **reset-adjusted** | **7.295 ± 0.011** | [7.276, 7.321] | 2.1868 | 0.343 | +0.629 |
| reset-naive | 7.866 ± **0.514** | [7.321, **9.918**] | 2.1825 | **0.610** | +0.322 |
| prbs-adjusted | 7.321 ± 0.000 | [7.321, 7.321] | 2.1598 | 0.304 | +0.606 |
| prbs-naive | 7.321 ± 0.000 | [7.321, 7.321] | 2.1605 | 0.305 | +0.597 |
| baseline (tuned BOPTEST built-in) | 11.299 | — | 2.1422 | — | — |

**An earlier version of this section reported the opposite ordering and it was wrong.** It had
`reset-adjusted` at 16.838 K·h — the worst arm, with 2.4× the reference's spread — and concluded that
de-confounding buys the mean back by giving up variance. That number was an artefact of the MPC's
constant step size, which sat 75–284× past the solver's stability limit and, because the limit scales
with the authority a model *believes* it has, punished the arms that identified the channel best. It
is defect 5 in §7. Everything below was re-measured after the fix.

Read the table with `7.321` in mind: it is a **floor**, not a score. It appears to four decimals in
both PRBS arms at every seed and in several reset-arm seeds, because the setpoint schedule builds in
discomfort during setback recovery that no command in this actuator's range can avoid. So `tdis_tot`
separates arms only by how often they fail to reach the floor, and the informative columns are the
spread and `sat`.

Three things follow:

1. **De-confounding wins on the axis it was supposed to and on the one it was accused of losing.**
   `reset-adjusted` has the best mean (7.295, slightly *below* the floor — it pre-heats through the
   setback) and a seed-to-seed s.e. of 0.011 against `reset-naive`'s 0.514, a **48× tighter** spread.
   The variance story in the earlier version was backwards.
2. **The confounded arm's failure is a sign error, not an attenuation.** Its fitted channel is
   `+2.989 − 0.1321·T`, which crosses zero at **22.62 °C** — inside the occupied comfort band. Above
   that temperature the model believes the heat pump *cools* the room (−0.182 K/h per unit at 24 °C)
   and pins the command at zero. The de-confounded channel `+2.525 − 0.0978·T` crosses at 25.82 °C,
   outside the band, and stays positive wherever the controller operates. The one seed at 9.918 K·h is
   the one that spent 98.5% of its steps on a bound.
3. **Energy does not separate them.** 2.1868 against 2.1825 is 0.2%, and the total heat a building
   needs over a week is set by the weather, not by the controller. A gain error moves *timing*, which
   comfort sees, far more than it moves total energy. The tuned baseline spends 2% less energy than
   any arm and pays 55% worse discomfort for it.

The randomised arms are the control on the whole experiment: adjusted and naive land on the identical
floor with identical spread and authorities within 1.5% of each other (+0.606, +0.597), which is what
"a randomised action needs no adjustment" should look like.

One caveat this section keeps from the earlier version, because it survived the fix: which way a
channel error moves the controller is a property of the **cost**, not of the error. Under a quadratic
tracking objective an attenuated channel under-actuates; under BOPTEST's comfort-dominated objective
it over-actuates. Comparing two controllers at one operating point is a category error when the model
error itself moves the operating point — hence the sweep below.

### Margin sweep — the frontier separates them, once the planner converges

`run_pareto`, seed 0, requested margin 0.0 → 2.5 K:

| margin | naive `tdis` | naive `ener` | naive `sat` | adjusted `tdis` | adjusted `ener` | adjusted `sat` |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 7.408 | 2.0244 | 0.286 | 7.647 | **2.0097** | 0.235 |
| 0.5 | 7.321 | 2.0860 | 0.315 | 7.321 | 2.0633 | 0.265 |
| 1.0 | 7.321 | 2.1501 | 0.366 | 7.321 | 2.1167 | 0.283 |
| 1.5 | 7.321 | 2.2100 | 0.420 | 7.276 | 2.1742 | 0.295 |
| 2.0 | 8.258 | 2.2910 | 0.473 | **7.166** | 2.2393 | 0.289 |
| 2.5 | **61.749** | 2.3701 | 0.518 | **7.088** | 2.3057 | 0.280 |

The earlier version of this sweep had both arms collapsing past 1.5 K (39.0 and 39.7 at margin 2.0,
55.3 and 90.4 at 2.5) and concluded that the frontier does not separate them. Half of that was the
solver. With it fixed:

- **The margin is a conservatism dial for the de-confounded model and a mis-specification for the
  confounded one.** `adjusted` improves monotonically across the whole sweep, 7.647 → 7.088, and its
  best comfort is at the *largest* margin. `naive` bottoms out at 1.5 K and then collapses to 61.7.
- **The mechanism is visible in `sat`.** The de-confounded arm's saturation is flat across the sweep
  (0.235 → 0.295, no trend), so extra margin buys pre-heating. The confounded arm's rises
  monotonically, 0.286 → 0.518: asking it for more margin pushes it into the actuator bound, where the
  command is set by the box rather than the model, and its sign flip at 22.62 °C means the extra
  authority it thinks it needs is not there to spend.
- **The de-confounded model still uses less energy at every margin** (0.7% to 2.7%), which is the one
  claim that survived the fix unchanged. At the exactly matched-discomfort points — margin 0.5 and
  1.0, both arms at 7.3212 K·h to all digits — it uses **1.09%** and **1.55%** less.

So the two models are not interchangeable at a re-tuned operating point, which is what the earlier
"the frontier does not separate them either" asserted. Removing the confounding bias buys a
controller that can actually be made conservative.

## 5. Spending the radius — pessimism has nothing left to buy here

This section was written to answer a §4 diagnosis that no longer holds. The old §4 said the
de-confounded arm fails on *variance*, so planning against a weaker plant was the obvious remedy.
With the solver fixed that arm has the *tightest* spread of any, and the question changes from "can
pessimism rescue it" to "is there anything left for pessimism to buy". The answer is no, and the way
it fails is more informative than the old one.

`ThermalFit.pessimistic(shrink)` reduces the believed gain `b₀`; `run_pessimism_sweep` sweeps it.

### The first radius was dimensionally wrong, and the emulator said so in one run

The natural radius is the estimator's own `channel_error`, so that is what the first version used:
shrink by `k · channel_error`. Refuted immediately.

| `k` | believed `b₀` (seed 0) | 8h rise (K) | `tdis_tot` (K·h) | `ener_tot` | `sat` |
|---:|---:|---:|---:|---:|---:|
| 0 | +1.258 | 4.946 | 7.276 | 2.1742 | 0.295 |
| 1 | +0.750 | 1.362 | 74.529 | 2.3274 | 0.619 |
| 2 | +0.242 | −2.222 | 1291.225 | 0.2261 | 1.000 |
| 3 | −0.266 | −5.806 | 1475.040 | 0.0001 | 1.000 |

Re-measured under the fixed solver of §7, which moved the top two rows (9.742 → 7.276, 76.533 →
74.529) and left the bottom two **byte-identical**. That is not luck: at `k ≥ 2` the believed 8-hour
rise is negative, the model asserts that heating cools the room, and the command pins at zero every
step. A controller that has stopped deciding cannot be affected by how well its optimiser converges,
so the two solvers agree exactly — which is the cleanest available confirmation that the collapse in
these rows is the model's, not the planner's.

The controller did not become conservative — it gave up, pinned at a bound every step with energy at
zero. The cause is dimensional. `channel_error` is the root-mean diagonal of the estimator's sandwich
over the **whole** channel matrix, so it mixes `b₀` with `b₁`, whose natural scale is smaller by a
factor of the operating temperature. It is a scalar summary, not `b₀`'s standard error, and using it
as one over-states the radius by **9.3×** here: `channel_error ≈ 0.51`, while one *measured* standard
error of the quantity the controller uses — the 8-hour step response, 0.385 K across five seeds, at
7.06 K of rise per unit `b₀` — is a shrink of only **0.055**.

`pessimistic` now takes an absolute shrink in `b₀`'s units and refuses to pretend it has a sigma. The
ratio `channel_error / |b₀|` is data-dependent — 0.08 on the synthetic fixture where shrinking by it
is harmless, 0.40 on the emulator where it is fatal — so it is carried on the fit and documented as
the thing to check before spending any radius.

### With a correctly scaled radius, pessimism buys nothing and then costs a tail

0, 1, 2 and 3 measured standard errors, 5 seeds:

| shrink (`b₀` units) | `tdis_tot` mean ± s.e. | range | `ener_tot` | `sat` | mean 8h rise |
|---:|---:|---|---:|---:|---:|
| 0.000 | 7.2954 ± 0.011 | [7.276, 7.321] | 2.1868 | 0.343 | 4.285 |
| 0.055 | **7.2894 ± 0.014** | [7.247, 7.321] | 2.2051 | 0.368 | 3.907 |
| 0.109 | 8.4389 ± 0.888 | [7.245, 11.866] | 2.2202 | 0.399 | 3.537 |
| 0.164 | 11.4303 ± 3.498 | [7.244, **25.255**] | 2.2238 | 0.449 | 3.159 |

The best shrink is one standard error and it improves the mean by **0.08%** — 7.2894 against
7.2954, comfortably inside a standard error of either. Past that, the mean degrades and the *spread*
explodes by two orders of magnitude, from ±0.011 to ±3.498. Four of twenty episodes exceed 8 K·h and
all four are at shrink ≥ 0.109. So the honest reading is not "pessimism is mildly bad" but
**pessimism is indistinguishable from nothing until it is catastrophic**, which is a worse property
than a smooth trade-off: there is no setting of this dial that a practitioner could be told to use.

### Why: the channel is already good enough, and shrinking it is a real intervention

The sweep is the more informative measurement here, because it *varies the channel deliberately*
rather than watching it vary. Across its twenty episodes:

| relationship | correlation |
|---|---:|
| `tdis_tot` vs believed 8-hour rise | **−0.538** |
| `tdis_tot` vs `at_bound` | +0.491 |
| `tdis_tot` vs shrink | +0.380 |

Less believed authority means more discomfort, and it arrives through saturation: the controller
that thinks the plant is weak commands more, hits the actuator bound, and stops deciding. That is a
causal statement about the channel, not a correlational one, since the shrink was imposed.

The purely observational correlations tell the older story, but weakly enough that they should not
carry an argument. Across the five `reset-adjusted` seeds:

| fitted quantity | corr with `tdis_tot` | range over seeds | coeff. of variation |
|---|---:|---|---:|
| drift bias `d` | +0.672 | [+0.260, +0.990] | **0.418** |
| pole `a` | −0.636 | [−0.0629, −0.0239] | 0.314 |
| 8h authority | +0.111 | [+0.471, +0.812] | 0.212 |
| saturation | +0.151 | [+0.295, +0.417] | 0.162 |

The signs and the ordering match the earlier version (which had +0.790 / −0.788 / +0.428): the drift
still tracks discomfort better than the channel does, and it is still the least stable part of the
fit, varying about twice as much across seeds. **But `tdis_tot` for this arm now ranges over 0.045
K·h — six hundredths of a percent — so these are correlations computed against noise at the comfort
floor, and the earlier version's confident reading of them was borrowing significance from the
solver's variance.** The drift-versus-channel claim is retained as a hypothesis worth testing on a
plant whose closed loop actually moves, not as a measured conclusion here.

The scope limit it points at is real regardless: only the **channel** is interventional in
`chc.dynamics_id`. `a_θ` is fitted by least squares on whatever the channel leaves behind, so it
stays observational-conditional — and an MPC uses the drift as well as the channel.

So the actionable conclusion is not "tune the radius" and no longer "the radius has to cover the
drift" either — that was inferred from correlations this document can no longer support. It is
narrower and better founded: **on a plant where de-confounding already puts the closed loop on its
comfort floor, channel pessimism has no headroom to buy and a tail to lose.** Whether a pessimism
radius pays at all is a question this case cannot answer, because this case has no gap left for it to
close. That needs a plant whose closed loop is not floor-limited — which is a reason to run the other
two cases and a non-building environment, not a reason to tune a knob.

## 6. Physics-off — the black box is unbiased on average and unusable per seed

`run_structure_ablation`, 5 seeds, one confounded (`reset`) 20-day log per seed, three arms fitted on
the same rows. The black box is an MLP for `dT/dt` given `(T, z, u)`: the confounder is inside its
conditioning set, it may fit nonlinearities the affine model cannot, and it gets far more fitting
compute — 3000 Adam steps against one linear solve. It differs from the structured arms in the
model and in nothing else — `_mpc_solver` reads only `PlantModel.rate`, so all three plan through the
same objective, horizon and iteration budget. The two affine rows below are the same computation as
§1's `reset-*` rows on the same seeds, not independent evidence; they are repeated for comparison.

| arm | 8h step response (K) | range over seeds | pole (1/h) | held-out MSE | \|dev of mean\| | mean \|dev\| |
|---|---:|---|---:|---:|---:|---:|
| **prbs-naive** (reference, §1) | 4.429 ± 0.162 | [4.052, 4.852] | −0.0195 | — | — | — |
| black box | 4.456 ± 0.829 | [2.407, **7.267**] | **−0.1722** | 0.0464 ± 0.0040 | **0.027** | 1.406 |
| affine-adjusted | 4.285 ± 0.385 | [3.300, 5.168] | −0.0421 | 0.0218 ± 0.0007 | 0.144 | **0.699** |
| affine-naive | 2.187 ± 0.207 | [1.514, 2.738] | −0.0439 | 0.0247 ± 0.0011 | 2.243 | 2.243 |

Read the last two columns together, because separately they say opposite things. **The black box's
mean 8-hour rise is the closest to the reference of any arm** — 0.027 K, five times closer than the
de-confounded affine fit. Per seed it is twice as far: 1.406 K against 0.699 K, over a range of
2.41 to 7.27 K on the one quantity a controller consumes. Its errors change sign across seeds and
cancel in the mean; the affine arms' do not. The naive arm's two columns agree to three decimals,
which is what a *bias* looks like, and is why §1 can call it wrong by 50.6% and mean it.

So the three arms fail in three distinct ways — bias, small consistent error, and unbiased noise —
and **only the third is invisible to a leaderboard that averages over seeds before reporting.** A
five-seed mean is exactly the summary that would have ranked the black box first here.

Prediction does penalise the black box here — 0.0464 against 0.0218, **2.13×** — which is the
opposite of the synthetic fixture, where the two were indistinguishable. That makes the penalty a
property of this plant rather than a law: 768 nonstationary rows at 15% overlap is not where an MLP
is strong. But being penalised is not the same as being ranked, and the two orderings are not the
same ordering:

| ranked by | 1st | 2nd | 3rd |
|---|---|---|---|
| held-out one-step MSE | adjusted 0.0218 | **naive 0.0247** | **black box 0.0464** |
| per-seed identification error | adjusted 0.699 | **black box 1.406** | **naive 2.243** |

They agree on the winner and **swap the other two**. Prediction says the confounded affine fit is the
second-best model on this plant; the quantity a controller consumes says it is the worst of the
three, by a factor of 1.6 over the black box. So even on the plant where held-out error does notice
the black box, it still does not reproduce the ranking that matters.

Nor does it see the *magnitude* of what went wrong. The black box's fitted pole comes back at −0.172
against −0.042 — a thermal time constant **4.1× too fast**, a 5.8-hour building instead of a 24-hour
one — from a model whose one-step error is only twice as large. One-step error integrates over a
rollout; a derivative does not.

Closed loop, same MPC for all three:

| arm | tdis ± s.e. (K·h) | range over seeds | ener | sat | wall/episode |
|---|---:|---|---:|---:|---:|
| **affine-adjusted** | **7.295 ± 0.011** | [7.276, 7.321] | 2.1868 | **0.343** | 39 s |
| affine-naive | 7.866 ± 0.514 | [7.321, 9.918] | 2.1825 | 0.610 | 40 s |
| black box | 12.413 ± 3.565 | [7.321, **25.657**] | 2.2053 | 0.548 | 40 s |

Bimodal, not merely worse: the comfort floor on three seeds, 14.2 and 25.7 K·h on the other two, and
a seed-to-seed spread **337×** the de-confounded arm's. The two failures are not the same failure.
Seed 0 fails through the drift — pole −0.297, seven times too fast, 98% of steps on an actuator bound
and the lowest energy the arm spends. Seed 4 fails through the channel — a fitted authority of
**+2.023**, three times the structured fit's mean and 2.5× the largest value any structured seed
produced — at the highest energy of all fifteen runs. Neither is the naive arm's failure, which is
one-sided: understate the channel, over-actuate, buy comfort with energy.

The sharpest row in the ablation is a pair. Seeds 0 and 2 have held-out one-step errors of **0.04724
and 0.04730** — equal to three significant figures, the closest any two runs come — and closed-loop
discomfort of **25.657 and 7.321 K·h**, a factor of 3.5. Within one model class, on one plant, at one
sample size, held-out prediction is not a sufficient statistic for the closed loop.

Two things this does not claim. The configuration is the *kindest* one found, not a handicap: width
and depth and step count were swept, the smallest box won on every axis, and held-out error rises
monotonically with compute — at 12000 Adam steps the implied pole came back at −10.1 and +3.6 across
two inits, with one 8-hour step response at **−7.7 K**, a model asserting that heating cools the room.
And with five seeds and a spread this wide, "the black box is worse in the mean" is not a claim the
sample size supports; the claim is about the spread, which is 337× and does not need a t-test.

## 7. Five defects the runs exposed, each fixed at its cause

1. **The steady-state gain is not identified** over this window; `b₀` and the pole are collinear.
   Fixed by reporting the finite-horizon step response the MPC actually acts on.
2. **The PRBS band moved the operating point.** Uniform excitation on [0, 1] parks the heat pump at
   50% duty — a different linearisation, poles two orders of magnitude apart between arms. Fixed by
   exciting a band *around* the reset policy's operating point, which keeps the estimand fixed while
   staying exogenous.
3. **A positive (unstable) thermal pole in three of four arms.** Not a bug in `fit_causal_residual`
   but its documented scope on real data: only the *channel* is interventional, `a_θ` mops up the
   rest of the rate, so the drift stays observational-conditional — and a trending outdoor
   temperature then gets charged to positive feedback in `T`. An MPC uses the drift too, so this is
   fatal regardless of how good the channel is. Fixed with the correct model: weather is observed and
   exogenous, so it belongs in the drift as a regressor, and BOPTEST serves the forecast needed to
   use it at control time. The Robinson moment still earns its keep — it is what keeps the channel
   robust to weather entering *nonlinearly*, which a linear `c'z` would miss.
4. **The 2×2 was unresolvable at `weather_gain = 0.02`** — the confounding channel sat below the
   estimator's noise floor. Raised to 0.05 on fidelity grounds (a real reset curve spans the actuator
   over roughly 20 K). The logging policy was recalibrated on **design** metrics only — saturation
   and overlap — never on the channel estimate.
5. **The MPC's constant step size decided the comparison.** "Every arm sees the same MPC and the same
   iteration count" was true of the code and false of the experiment. The comfort term's curvature is
   `w_comfort · (dt · authority)²`, so the largest stable gradient step `2/L` is a function of the
   authority each arm *believes* it has. At `lr = 0.05` that was 75× past the limit for the confounded
   fit, 252× for the de-confounded one and 284× for the black box. The iterate never converged: it
   clipped to the actuator bound, decayed by `1 − 2·lr` per step until the predicted temperature
   recrossed the comfort hinge, and clipped again — a period-8 limit cycle whose objective after 400
   iterations was *worse* than after 60 (267.7 against 8.5 for the black box). The commanded action
   was therefore wherever the cycle happened to be at iteration 60, and reading 0.729 / 0.900 / 0.590
   / 0.674 at budgets of 60 / 240 / 1000 / 4000 on identical inputs.

   The bias had a direction: **a model that identifies a larger channel earns a larger Hessian and is
   punished harder by a shared constant step**, so the better estimate was handed the worse
   controller. That is precisely the §4 result the earlier version of this document reported as a
   finding about de-confounding.

   Fixed with a step taken from the model rather than from a constant: `L` is the largest eigenvalue
   of the Hessian of the same objective with the hinge forced active, evaluated per solve, and the
   step is `1/L`. Conditioning then bites — the action penalty carries curvature 2 against the comfort
   term's ~1e4, so plain gradient descent needs O(κ) with κ ≈ 5000 — so the descent is Nesterov
   accelerated, needing O(√κ), and the budget went 60 → 600 iterations. Both cost exactly one gradient
   evaluation per iteration for every arm, so equal compute stays literally true rather than
   approximately. Barzilai-Borwein was tried and rejected (it stalls the black-box arm at its
   initialisation, where the curvature pair `s'y` collapses); a box-QP solver was rejected because it
   would be exact for the affine arms and unavailable for the black box, i.e. a different optimiser
   per arm.

   The general lesson is the one worth keeping: **equal compute is not an equal iteration count, it is
   an optimiser whose behaviour does not depend on the scale of the model being compared.**

## 8. What this track does not claim

- No ground-truth channel; the reference is a randomised design, not a known number.
- No steady-state gain (§1).
- **The closed-loop win is small in absolute terms and floor-limited.** De-confounding takes
  discomfort from 7.866 to 7.295 K·h with a 48× tighter spread, and it is the only arm whose margin
  behaves like a conservatism dial (§4) — but `7.321` is a floor this case's setpoint schedule builds
  in, so the headroom being competed for is under 8% of the metric. A plant with more room to lose
  would test the claim harder than this one does.
- **No win from channel pessimism** (§5), and no supported explanation for why. The earlier version
  attributed it to the drift being the dominant error, on correlations that turn out to be computed
  against 0.045 K·h of variation at the floor. That hypothesis is now untested rather than confirmed.
- The pessimism radius is **not calibrated**. `run_pessimism_sweep`'s default grid comes from the
  seed-to-seed spread of a five-seed run on this one case, which is an empirical yardstick for this
  plant and not a coverage guarantee for any other.
- **The physics-off arm is not shown to be worse in the mean** (§6). Five seeds against a spread of
  ±3.6 K·h cannot support that, and its mean 8-hour step response is in fact the closest of any arm
  to the randomised reference. The claim is about the *spread* — 337× the structured arm's, bimodal,
  with two seeds an order of magnitude apart on the closed loop at held-out one-step errors equal to
  three significant figures. It is also a claim about *this* box on *this* log: 768 nonstationary
  rows with 15% overlap. A longer log justifies a bigger box and might change the answer.
- Only `bestest_hydronic_heat_pump` has been run. It is also the only one of the three wired cases
  that is physically control-affine in its action — the other two actuate a temperature setpoint
  behind a local PI loop, where control-affinity is a local approximation.

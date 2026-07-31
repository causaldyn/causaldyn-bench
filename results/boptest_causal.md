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

Every derivative below is read at the action the log actually sat at — `ū ≈ 0.32` modulation — and
not at `u = 0`. On this case `u = 0` is genuinely in support: the night setback drives the command to
zero on 21% of steps, so the previous version of this table was answering a defined question rather
than a wrong one. It was answering a *different* one, and the one it answered is not comparable
across actuators. §7 is why.

| arm | 8h step response (K) | range over seeds | fitted decay `λ(ū)` | \|deviation from reference\| |
|---|---:|---|---:|---:|
| **prbs-naive** (reference) | 4.405 ± 0.227 | [4.017, 5.133] | −0.0215 | — |
| prbs-adjusted | 4.474 ± 0.099 | [4.271, 4.730] | −0.0217 | **0.068** |
| reset-adjusted | 4.195 ± 0.487 | [3.051, 5.511] | −0.0511 | **0.211** |
| reset-naive | 1.992 ± 0.213 | [1.307, 2.578] | −0.0719 | **2.414** |

Three claims, all falsifiable, all confirmed:

1. **The confounded arm is badly wrong.** Logged by a weather-compensated controller, the naive fit
   understates the heat pump's 8-hour authority by **54.8%** (1.992 K against 4.405 K).
2. **Adjustment repairs it.** The orthogonal fit lands within 0.211 K of the randomised reference — an
   **11.5× reduction** in deviation — using the same 20 days of *observational* data.
3. **Adjustment does not distort a design that was already clean.** On the randomised log it moves
   the answer by 0.068 K — 1.5% of the level — and cuts the spread by **2.3×** (s.e. 0.099 against
   0.227). An estimator that "helped" the randomised arm too would be distorting rather than
   de-confounding.

All four arms decay *at their operating action*. Not all of them decay everywhere the actuator can
reach: `prbs-naive` on seed 1 has `λ(1.0) = +0.0074`, and the box scan of §7 refuses to plan on it.
That is exactly the distinction a single reported "pole" could not make. Getting the drift to decay
at all took a model fix, not a constraint — see §8, defect 3.

This table is identification only, so it is untouched by the solver defect in §8 — the fits do not
involve the controller. It has now been re-measured twice for two unrelated reasons, and both
re-measurements land on the same rows:

- **float32 against x64.** A re-run after the solver fix was accidentally launched **without**
  `JAX_ENABLE_X64`, and float32 returned 4.4292 / 4.4270 / 4.2848 / 2.1867 against x64's 4.429 /
  4.427 / 4.285 / 2.187, with poles matching to five digits. On the same rows before the library's
  conditioning fix, float32 returned `nan`. An accident is not a designed experiment, but it is the
  check that fix needed on real data rather than on a fixture. Measured under the previous reporting
  convention — the agreement is a property of the linear solve, not of where a derivative is read.
- **The convention change, isolated.** All ten episodes were re-logged on a restarted stack in a
  later session, and rebuilding the *old* convention from the new fits returns 4.4292 / 4.4270 /
  4.2849 / 2.1872 with poles −0.0195 / −0.0246 / −0.0421 / −0.0439: the previous table to four
  decimals. The emulator, the policies and the estimator are bit-reproducible here, so **every
  difference between this table and the previous one is the change of reporting point**, not drift.

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
| 0.25 | 0.441 | 0.559 | 4.014 | 0.982 |
| 0.12 | 0.143 | 0.857 | 4.104 | 1.720 |
| 0.04 | 0.020 | 0.980 | 5.337 | 1.459 |
| 0.01 | 0.004 | 0.996 | 7.356 | **−4.162** |
| 0.00 | 0.003 | 0.997 | 4.803 | **−5.323** |

Adjustment is not magic. Above a residual share of ~0.14 the fit lands within 9% of the reference
(4.01, 4.10 against 4.41). Below 0.02 the moment has almost no regressor left, and by sd 0.01 the
fitted channel goes **negative** — the model says more heating cools the room. A perfectly
deterministic reset policy makes the action an exact function of the covariates and nothing is
identified at any sample size. This is the row to check before believing anything above it.

The bottom two rows say so a second way, without being asked to. Re-running this table on a
restarted stack reproduces `residual share` and `nuisance R²` exactly at every scale, and `b₀`
exactly at sd 0.25 and 0.12 — but only to three decimals at sd 0.01 and 0.00 (−4.162 against −4.165,
−5.323 against −5.327). Same logs, same estimator: where the identifying budget is 0.4% of the
action's variance, the Gram matrix is near-singular and the *last digits of the answer are not
reproducible either*.

## 4. Closed loop — de-confounding pays, and the earlier "no" was the optimiser

Same MPC, same comfort margin (1.5 K), 5 seeds, 7-day episodes. `sat` is the share of control steps
whose command sat on an actuator limit.

| arm | `tdis_tot` (K·h) mean ± s.e. | range over seeds | `ener_tot` | `sat` | authority at 21 °C |
|---|---:|---|---:|---:|---:|
| **reset-adjusted** | **7.295 ± 0.011** | [7.276, 7.321] | 2.1868 | 0.343 | +0.629 |
| reset-naive | 7.892 ± **0.539** | [7.321, **10.044**] | 2.1829 | **0.608** | +0.322 |
| prbs-adjusted | 7.321 ± 0.000 | [7.321, 7.321] | 2.1598 | 0.304 | +0.606 |
| prbs-naive (4 of 5 seeds) | 7.321 ± 0.000 | [7.321, 7.321] | 2.1617 | 0.304 | +0.597 |
| baseline (tuned BOPTEST built-in) | 11.299 | — | 2.1422 | — | — |

`prbs-naive` is short a seed because seed 1's fit decays at the operating action but grows at the top
of the actuator box (`λ(1.0) = +0.0074`), and the box scan of §7 refuses to plan on it. That is a
refusal the previous version of this harness could not produce: it tested one number, and that number
was negative.

**An earlier version of this section reported the opposite ordering and it was wrong.** It had
`reset-adjusted` at 16.838 K·h — the worst arm, with 2.4× the reference's spread — and concluded that
de-confounding buys the mean back by giving up variance. That number was an artefact of the MPC's
constant step size, which sat 75–284× past the solver's stability limit and, because the limit scales
with the authority a model *believes* it has, punished the arms that identified the channel best. It
is defect 5 in §8. Everything below was re-measured after the fix.

Read the table with `7.321` in mind: it is a **floor**, not a score. It appears to four decimals in
both PRBS arms at every seed and in several reset-arm seeds, because the setpoint schedule builds in
discomfort during setback recovery that no command in this actuator's range can avoid. So `tdis_tot`
separates arms only by how often they fail to reach the floor, and the informative columns are the
spread and `sat`.

Three things follow:

1. **De-confounding wins on the axis it was supposed to and on the one it was accused of losing.**
   `reset-adjusted` has the best mean (7.295, slightly *below* the floor — it pre-heats through the
   setback) and a seed-to-seed s.e. of 0.011 against `reset-naive`'s 0.539, a **51× tighter** spread.
   The variance story in the earlier version was backwards.
2. **The confounded arm's failure is a sign error, not an attenuation.** Its fitted channel is
   `+2.989 − 0.1321·T`, which crosses zero at **22.62 °C** — inside the occupied comfort band. Above
   that temperature the model believes the heat pump *cools* the room (−0.182 K/h per unit at 24 °C)
   and pins the command at zero. The de-confounded channel `+2.525 − 0.0978·T` crosses at 25.82 °C,
   outside the band, and stays positive wherever the controller operates. The one seed at 10.044 K·h is
   the one that spent 98.8% of its steps on a bound.
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
| 0 | +1.258 | 4.799 | 7.276 | 2.1742 | 0.295 |
| 1 | +0.750 | 1.322 | 74.529 | 2.3274 | 0.619 |
| 2 | +0.242 | −2.156 | 1291.225 | 0.2261 | 1.000 |
| 3 | −0.266 | −5.633 | 1475.040 | 0.0001 | 1.000 |

Re-measured under the fixed solver of §8, which moved the top two rows (9.742 → 7.276, 76.533 →
74.529) and left the bottom two **byte-identical**. That is not luck: at `k ≥ 2` the believed 8-hour
rise is negative, the model asserts that heating cools the room, and the command pins at zero every
step. A controller that has stopped deciding cannot be affected by how well its optimiser converges,
so the two solvers agree exactly — which is the cleanest available confirmation that the collapse in
these rows is the model's, not the planner's. Re-measured a third time on a restarted stack, all
four rows now reproduce to every digit printed, `ener_tot` and `sat` included.

The controller did not become conservative — it gave up, pinned at a bound every step with energy at
zero. The cause is dimensional. `channel_error` is the root-mean diagonal of the estimator's sandwich
over the **whole** channel matrix, so it mixes `b₀` with `b₁`, whose natural scale is smaller by a
factor of the operating temperature. It is a scalar summary, not `b₀`'s standard error, and using it
as one over-states the radius by **6.9×** here: `channel_error ≈ 0.51`, while one *measured* standard
error of the quantity the controller uses — the 8-hour step response, 0.487 K across five seeds, at
6.64 K of rise per unit `b₀` — is a shrink of only **0.073**.

`pessimistic` now takes an absolute shrink in `b₀`'s units and refuses to pretend it has a sigma. The
ratio `channel_error / |b₀|` is data-dependent — 0.08 on the synthetic fixture where shrinking by it
is harmless, 0.40 on the emulator where it is fatal — so it is carried on the fit and documented as
the thing to check before spending any radius.

### With a correctly scaled radius, pessimism buys nothing and then costs a tail

0, 1, 2 and 3 measured standard errors, 5 seeds:

| shrink (`b₀` units) | `tdis_tot` mean ± s.e. | range | `ener_tot` | `sat` | mean 8h rise |
|---:|---:|---|---:|---:|---:|
| **0.0000** | **7.2954 ± 0.011** | [7.276, 7.321] | 2.1868 | 0.343 | 4.195 |
| 0.0733 | 7.3582 ± 0.068 | [7.246, 7.623] | 2.2097 | 0.376 | 3.708 |
| 0.1466 | 11.0403 ± 2.437 | [7.244, 19.214] | 2.2282 | 0.427 | 3.222 |
| 0.2199 | 12.3169 ± 5.036 | [7.244, **32.462**] | 2.2029 | 0.533 | 2.735 |

**No shrink is the best shrink**, on the mean and on the spread at once. One standard error costs
0.9% of the mean (7.3582 against 7.2954) and already multiplies the spread by six; past that the mean
degrades by half and the spread explodes by **475×**, from ±0.011 to ±5.036. Three of twenty episodes
exceed 8 K·h and all three are at shrink ≥ 0.147. So the honest reading is not "pessimism is mildly
bad" but **pessimism is indistinguishable from nothing until it is catastrophic**, which is a worse
property than a smooth trade-off: there is no setting of this dial that a practitioner could be told
to use.

An earlier version of this table read the radius as 0.055 and reported the one-standard-error row as
a 0.08% *improvement*. The radius was wrong for the reason §7 gives — the 8-hour rise it is a
standard error *of* was integrating the decay at `u = 0` — and one standard error is 0.073, a third
larger. The direction of the conclusion did not survive the correction, and the corrected version is
the cleaner negative: at the radius its own data supports, pessimism is already a loss.

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
| decay `λ(ū)` | −0.326 | [−0.0630, −0.0424] | 0.143 |
| 8h authority | +0.112 | [+0.471, +0.812] | 0.212 |
| saturation | +0.151 | [+0.295, +0.417] | 0.162 |

The signs and the ordering match the earlier version (which had +0.790 / −0.788 / +0.428): the drift
still tracks discomfort better than the channel does, and it is still the least stable part of the
fit, varying about twice as much across seeds. The decay row is the one §7 moved: read at `u = 0` it
varied by a factor of 2.6 across seeds (CV 0.314) and correlated −0.636 with discomfort; read at the
operating action it varies by 1.5 (CV 0.143) and correlates −0.326. Half of what looked like
seed-to-seed instability in the drift was the seeds disagreeing about `b₁`, which the reporting point
then multiplied in. **But `tdis_tot` for this arm now ranges over 0.045
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

## 6. Physics-off — the black box does not return a plannable model on most seeds

`run_structure_ablation`, 5 seeds, one confounded (`reset`) 20-day log per seed, three arms fitted on
the same rows. The black box is an MLP for `dT/dt` given `(T, z, u)`: the confounder is inside its
conditioning set, it may fit nonlinearities the affine model cannot, and it gets far more fitting
compute — 3000 Adam steps against one linear solve. It differs from the structured arms in the
model and in nothing else — `_mpc_solver` reads only `PlantModel.rate`, so all three plan through the
same objective, horizon and iteration budget. The two affine rows below are the same computation as
§1's `reset-*` rows on the same seeds, not independent evidence; they are repeated for comparison.

| arm | 8h step response (K) | range over seeds | decay `λ(ū)` (1/h) | held-out MSE | \|dev of mean\| | mean \|dev\| |
|---|---:|---|---:|---:|---:|---:|
| **prbs-naive** (reference, §1) | 4.405 ± 0.227 | [4.017, 5.133] | −0.0215 | — | — | — |
| black box | 2.837 ± 0.589 | [1.467, 4.747] | −0.0179 | 0.0482 ± 0.0123 | 1.568 | 1.705 |
| affine-adjusted | 4.195 ± 0.487 | [3.051, 5.511] | −0.0511 | 0.0218 ± 0.0007 | **0.211** | **0.857** |
| affine-naive | 1.992 ± 0.213 | [1.307, 2.578] | −0.0719 | 0.0247 ± 0.0011 | 2.414 | 2.414 |

**An earlier version of this table said the opposite about the black box and it was wrong.** It
reported a mean 8-hour rise of 4.456 K — *closer to the randomised reference than any other arm* —
and built the section on the reading that the physics-off arm is unbiased in the mean and unusable
per seed. That row was produced under float32 and with the derivative read at `u = 0`. Both are now
fixed, and the retraction is exact rather than approximate: refitting in float32 and reading at
`u = 0` returns 4.456 ± 0.829 over [2.407, 7.267], the published row to three decimals including
both range endpoints, and the seed whose pole the old text quoted as −0.297 returns −0.297153.

| what is being reported | mean 8h rise | s.e. | range |
|---|---:|---:|---|
| float32, derivative at `u = 0` (previous record) | 4.456 | 0.829 | [2.407, 7.267] |
| float32, derivative at `ū` | 3.005 | 0.617 | [1.430, 5.169] |
| float64, derivative at `ū` (this table) | 2.837 | 0.589 | [1.467, 4.747] |

The reporting point costs 33% and the precision another 6%. The same change of reporting point moves
the **affine** fit by 2.1% (4.285 to 4.195), and float32 leaves it alone entirely — §1's accidental
x32 run agreed with x64 to five digits. So this is not a general statement about numerics: it is that
a model with no structure has no stable answer to "what is this plant's decay", and which number gets
reported decides whether it looks unbiased.

With that fixed, the black box understates the 8-hour authority by **36%** (2.837 against 4.405) and
its two error columns nearly agree — 1.568 against 1.705 — which is what a *bias* looks like, the
same signature the naive affine arm shows to four decimals (2.4136 twice). The arm whose errors now
cancel in the mean is the **de-confounded affine** one, 0.211 against 0.857, a factor of 4.

Prediction penalises the black box — 0.0482 against 0.0218, **2.2×** — which is the opposite of the
synthetic fixture, where the two were indistinguishable. That makes the penalty a property of this
plant rather than a law: 768 nonstationary rows at 15% overlap is not where an MLP is strong. But
being penalised is not the same as being ranked, and the two orderings are not the same ordering:

| ranked by | 1st | 2nd | 3rd |
|---|---|---|---|
| held-out one-step MSE | adjusted 0.0218 | **naive 0.0247** | **black box 0.0482** |
| per-seed identification error | adjusted 0.857 | **black box 1.705** | **naive 2.414** |

They agree on the winner and **swap the other two**. Prediction says the confounded affine fit is the
second-best model on this plant; the quantity a controller consumes says it is the worst of the
three, by a factor of 1.4 over the black box.

Nor does prediction see what the stability check sees. Read at the action it was logged at, the black
box's decay is **positive on three of five seeds** — +0.023, +0.216, +0.094 — so those fits assert a
zone that warms away from its own equilibrium with the actuator held where the data sat. The §7 box
scan refuses all three offline, before a single control step is spent.

Closed loop, same MPC for all three:

| arm | tdis ± s.e. (K·h) | range over seeds | ener | sat | wall/episode | refused |
|---|---:|---|---:|---:|---:|---:|
| **affine-adjusted** | **7.295 ± 0.011** | [7.276, 7.321] | 2.1868 | **0.343** | 25 s | 0/5 |
| affine-naive | 7.892 ± 0.539 | [7.321, 10.044] | 2.1829 | 0.608 | 25 s | 0/5 |
| black box | — | 7.280 and 15.776 | 2.1439 | 0.430 | 28 s | **3/5** |

The physics-off arm no longer has a mean worth printing: three of its five fits are unplannable, and
of the two that survive, one reaches the comfort floor and the other spends 15.776 K·h — 2.2× the
de-confounded arm on the same plant, the same log and the same optimiser.

The sharpest row in the ablation is again a pair, and it is sharper than the one it replaces. Ranked
by held-out one-step error the five black-box fits go 0.0301, 0.0364, 0.0377, 0.0400, 0.0971. **The
best predictor of the five is refused** — its decay is +0.094 inside the actuator box — and the
worst, at 3.2× the held-out error of the best, is one of the two that plans. Within one model class,
on one plant, at one sample size, held-out prediction is not just an insufficient statistic for the
closed loop; here it is ordered against it.

Three things this does not claim. The configuration is the *kindest* one found, not a handicap: a
12-fit sweep on one 768-row head — width 8 and 64, depth 1 and 3, 1000/3000/12000 Adam steps, two
inits each, all in float64 — gives the small box a lower held-out error than the big one at every
step count (0.026 / 0.039 / 0.072 against 0.069 / 0.099 / 0.083), and for the small box the error
rises monotonically with compute. The *derivatives* rot faster than the fit does: at width 64 and
depth 3 the two inits of one configuration return decays of **−7.99 and +1.12** — a factor of nine
apart and of opposite sign, so they disagree about whether the building is stable at all — while
their held-out errors differ by 4%. Second, with five seeds and a spread this wide, "the black box is
worse in the mean" is not a claim this sample size supports; the claims here are about the *sign* of
a fitted decay and about refusals, which are counts, not means. Third, three refusals out of five is
a property of this box, this log length and this actuator range, not a rate anyone should quote.

## 7. The other two cases — the reported pole was a choice of units

The two remaining wired cases actuate a **temperature setpoint** behind a local PI loop, and both are
flagged `control_affine_in_action = False`. Running them produced, at first, a clean negative result:
every arm on both cases under both policies fitted a **positive** thermal pole, and the 8-hour step
response came back between `1e4` and `1e10` K.

| case | policy | adjusted | pole | 8h rise (K) | authority | overlap share |
|---|---|---|---:|---:|---:|---:|
| heat_pump | reset | no | −0.0485 | 2.738e+00 | +0.409 | 0.978 |
| heat_pump | reset | yes | −0.0341 | 4.946e+00 | +0.701 | 0.151 |
| heat_pump | prbs | no | −0.0186 | 4.052e+00 | +0.543 | 1.001 |
| heat_pump | prbs | yes | −0.0268 | 4.184e+00 | +0.578 | 0.960 |
| hydronic | reset | no | **+8.3633** | 3.995e+10 | +1.237 | 0.618 |
| hydronic | reset | yes | **+6.4200** | 2.065e+09 | +1.361 | 0.313 |
| hydronic | prbs | no | **+2.6266** | 2.838e+05 | +1.108 | 0.692 |
| hydronic | prbs | yes | **+1.8807** | 2.506e+04 | +1.167 | 0.655 |
| air | reset | no | **+5.9480** | 5.759e+08 | +0.885 | 0.881 |
| air | reset | yes | **+5.8787** | 5.487e+08 | +0.959 | 0.175 |
| air | prbs | no | **+5.7908** | 5.251e+08 | +1.082 | 0.815 |
| air | prbs | yes | **+5.5567** | 3.592e+08 | +1.157 | 0.747 |

The separation on `control_affine_in_action` is perfect, 4/4 against 8/8, and the obvious reading —
setpoint actuation breaks the model class — is **wrong**. Three things ruled it out.

**The channel is fine.** Authority is stable at +0.89 to +1.36 across all eight setpoint rows. What
explodes is the horizon, which integrates the *drift*.

**Overlap does not gate it.** The unit-free identifying budget `1 − R²(action | nuisance)` is
*better* on the setpoint cases (0.313, 0.175) than on the case that works (0.151).

**Collinearity does not explain it either, and this was a refuted hypothesis, not a confirmed one.**
The mechanism guessed first was that a tracked setpoint makes `u ≈ T`, leaving the drift and the
channel jointly unidentified. Measured: `corr(u, T)` after partialling the covariates out is +0.030
on heat_pump, +0.484 on hydronic — and **−0.045 on air**, which explodes just as hard. Second-stage
conditioning is worse on the setpoint cases (1.1e5 and 8.0e4 against 1.5e3), but 1e5 in float64 costs
five significant digits and cannot force a sign error of this size. Collinearity amplifies; it does
not cause.

### What it actually is

The fitted class is `dT/dt = a·T + d + c'z + (b₀ + b₁·T)·u`. Substitute an affine change of actuator
coordinates `u = α·v + β` and the class is closed, with (Maxima, closure residual exactly 0):

```
a → a + β·b₁        b₀ → α·b₀        b₁ → α·b₁
```

So **`a` is not a property of the building.** It is the decay at `u = 0`, and where zero sits is a
reporting convention. `HEAT_PUMP` modulates on `[0, 1]`, so `β = 0` and the convention is harmless.
Both setpoint cases report their action as a setpoint in `[15, 25] °C`, so `β = 15` and the reported
pole is the decay at a setpoint of **0 °C**, 15 K below anything the actuator can command.

The coordinate-free quantity is the decay at a stated actuator position, `λ(u) = a + b₁·u`, which the
same derivation shows is invariant. Refitting each log with the action re-expressed as a fraction of
travel (`α = 10`, `β = 15`) — the same rows, the same estimator, only the units changed. One seed,
`reset`, the whole 960-row log, which is what §1 fits too, so the `heat_pump` row below **is** §1's
own seed-0 `reset-adjusted` fit read two ways:

| case | β | `drift[1]` raw | `drift[1]` frac | `b₁` raw | ū | **λ(ū) raw** | **λ(ū) frac** |
|---|---:|---:|---:|---:|---:|---:|---:|
| heat_pump | 0 | −0.0341 | −0.0341 | −0.0265 | 0.315 | **−0.0424** | **−0.0424** |
| hydronic | 15 | +6.4197 | +0.7519 | −0.3779 | 20.685 | **−1.3969** | **−1.3970** |
| air | 15 | +5.8776 | +1.1595 | −0.3146 | 20.311 | **−0.5119** | **−0.5120** |

`heat_pump` is the null control: at `α = 1, β = 0` the map is the identity and the two fits are equal.
On the other two, `drift[1]` moves by `β·b₁` to three decimals of the prediction while `λ(ū)` agrees
to four, and `authority` is invariant up to the span (+1.3610 against +1.3611). **All three buildings
decay.** The step response inherits the same defect and the same fix: it integrated `drift[1]`, which
is why it reported `2e9` K.

Cross-checks, because a symbolic identity is a hypothesis until it is measured. Octave, on a
synthetic bilinear plant fitted in both coordinate systems: the law holds to 3.3e-16, `λ(ū)` to
5.5e-16, and the Frisch–Waugh–Lovell decomposition of the fitted coefficient to 6.4e-16. On this
module's own synthetic fixture, shifting the action onto a `[15, 25]` scale moves `drift[1]` by
0.0113 while `decay()` moves by 2.6e-7 — four orders of separation, now pinned by an offline test.

### What was wrong in this harness, and what remains true

`ThermalFit.pole`, `.stable`, `.step_response`, `NeuralFit.pole`, `.authority` and
`finite_difference_step_response` all read their derivative at `action = 0`. Each now reads it at the
log's own operating action, and `decay(action)` is on the `PlantModel` protocol so both arms are
scored by one definition. `RunawayDriftError` was added earlier in this same investigation and fired
on every setpoint arm; **every one of those was a false positive**, and a refusal keyed on a quantity
a change of units can flip is not a safety check. It now scans the actuator box.

That scan is not decoration. The hydronic fit crosses zero at a setpoint of **17.0 °C** and the air
fit at **18.7 °C**, both inside `[15, 25]`: at the operating point the model decays, and a plan that
reaches for the low end of the box is extrapolating against a growing one. That is a real, coordinate-
free pathology of the fitted model, it is a plausible reason the first closed-loop attempts on these
cases stalled the emulator's own solver, and it is what the refusal should have been testing all
along.

The same defect is in the library. `ControlAffineResidual.drift_jacobian` documented its spectrum as
telling you "whether the plant runs away on its own", which is true only at `degree = 0`; at any
state-dependent channel the horizon follows `∂(a_θ + B_θ u)/∂x`. The claim is corrected and
`closed_loop_jacobian` added.

### The closed loop is still refused, and now for a reason that survives a change of units

One control episode was attempted per case and arm, with the corrected check in place:

| case | arm | `drift[1]` | `b₁` | λ(ū) | λ(15 °C) | λ(25 °C) | sign change | outcome |
|---|---|---:|---:|---:|---:|---:|---:|---|
| hydronic | naive | +8.3629 | −0.4747 | −1.4559 | **+1.2429** | −3.5038 | 17.62 °C | refused |
| hydronic | adjusted | +6.4197 | −0.3779 | −1.3969 | **+0.7515** | −3.0273 | 16.99 °C | refused |
| air | naive | +5.9454 | −0.3196 | −0.5466 | **+1.1509** | −2.0454 | 18.60 °C | refused |
| air | adjusted | +5.8776 | −0.3146 | −0.5119 | **+1.1588** | −1.9871 | 18.68 °C | refused |

So the setpoint cases remain unplannable, but the sentence has changed: *not* "the fit says the
building runs away", which was units, but "the fit decays at the setpoint it was logged at and grows
at the bottom of the actuator's range, and an MPC's horizon is free to go there".

**What was not done, and why.** The obvious repair is to plan on the certified sub-box
`{u : λ(u) < 0}` — `[17.0, 25]` for hydronic-adjusted, `[17.6, 25]` for hydronic-naive — which is
tempting here because hydronic's baseline discomfort is **33.03 K·h** against the heat pump's
`7.32` floor, i.e. 4.5× the headroom the §4 result had to compete for. It is not done because each
arm certifies a *different* box, so the arms would be optimising over different feasible sets and
any win would confound model quality with how much actuator the model was allowed to use. A common
box would have to be justified from the plant rather than from the fits, and nothing here does that.

The boundary ordering is also not a result. De-confounding buys hydronic a wider certified box
(16.99 against 17.62) and air a narrower one (18.68 against 18.60) — one case each way, which is
noise at two cases.

Not established: *why* `b₁` is large and negative on a setpoint-tracked zone. The sign is what a
tracking loop should give — authority falls as the zone approaches its setpoint — but the magnitude
is not derived from anything here.

## 8. Seven defects the runs exposed, each fixed at its cause

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

6. **The reported pole was the decay at `action = 0`** (§7), which an affine change of actuator
   coordinates shifts by `β·b₁`. Fixed by reading every derivative at the log's operating action and
   scanning the actuator box for the stability check.

   Five of these seven are the same defect: **a constant whose meaning depends on the scale of its
   input.** The MPC step size (defect 5), the nuisance ridge on an unstandardised basis, the effort
   term in raw actuator units, the client timeout, and now the operating point at which a derivative
   is read. The module docstring already argued this correctly for the *state* — "`b₀` is the
   intercept at `T = 0`, some 21 K outside anything a heated building ever visits" — and missed the
   identical argument for the action.

7. **The floating-point precision was never pinned, and one arm depends on it.** §6's black-box row
   was measured in float32. The affine arms are insensitive — §1's accidental x32 run agreed with x64
   to five digits, because a 5×5 least-squares solve at this conditioning does not care — so the
   flag looked optional, and `README.md` said so in as many words. It is not optional for the MLP:
   float32 moves its fitted decay from −0.034 to −0.297 on one seed and changes the sign of the
   stability verdict on others, because 3000 Adam steps compound rounding into the *derivative* of
   the fitted surface rather than into the surface. Diagnosed by exact reproduction, not by
   inspection: refitting under float32 with the derivative read at `u = 0` returns the retracted row
   to three decimals.

   This one is not a scale defect, and that is why it is worth listing separately. It is the
   *reproducibility* failure the other six could hide behind: five of the seven are caught by asking
   "in what units", and this one is only caught by re-running.

## 9. Certificate off against certificate on — the audit separates the arms, the filter almost never acts

The closed half of the CHC spine, run on the emulator: every MPC solve is wrapped in a
`chc.plan.CausalPlan` and priced by `chc.plan.certify_safety`, and the certificate-on arms additionally
pass the applied command through `chc.barrier.robust_safety_filter`. `run_certificate_ablation`, one
confounded 20-day log per seed with both fits taken off it, 7-day control episodes, two seeds:

```bash
JAX_ENABLE_X64=1 uv run python -c "
from causaldyn_bench.boptest_causal import run_certificate_ablation
for row in run_certificate_ablation(seed=1): print(row)"
```

**Wiring, and the three choices in it.** The barrier is `h(T) = T − c` with `c` the comfort bound in
force *now*, held fixed across the horizon; `‖∇h‖ = 1`, so the identification radius enters as Δ
exactly rather than inflated by an augmented state. The bound BOPTEST forecasts is a two-level step
function — 21 °C occupied, 15 °C setback, ten ±6 K jumps a week — and a barrier that tracked it would
demand 12 K/h of the zone at every transition, twenty times this heat pump's full authority; freezing
it is the standard CBF treatment of a moving safe set, and anticipating the next bound is the MPC's
job. The MPC's 1.5 K margin is deliberately *not* in the barrier: the certificate is about the bound
`tdis_tot` is billed against, not about the planner's conservatism dial. And Δ is §5's calibrated
0.073 K/h per unit of modulation — one measured standard error of `b₀`, not the estimator's
`channel_error`, which over-states it by 6.9×. At a unit gap that is Γ = 1.1575.

`causal_plan` is not the planner. It minimises a `QuadraticCost` by projected gradient, and this
harness plans against a hinge on a forecast comfort bound; swapping the objective would move every
number in §4 and turn the ablation into "is a different controller better". `certify_safety` prices a
*finished* plan by design, which is exactly what the MPC hands it.

**The certificate-off arm is the un-audited loop, measured rather than asserted.** The audit is
read-only by construction, and a live check confirms it: two six-step episodes, one with `audit=None`
and one with `SafetyAudit(enforce=False)`, agree on every KPI to every digit — only `wall_s` and
BOPTEST's `time_rat` move, because a `certify_safety` trace per step is not free.

### What the audit reads, without executing anything

Certificate-off rows; the on rows differ only through the single step of the next table.

| seed | arm | channel | crosses zero at | authority at 21 °C | steps with no Γ\* | certified horizon | uncertified applied step |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | adjusted | `+1.2582 − 0.0265·T` | 47.4 °C | **+0.701** | **7.44%** | 90.7% | 4.76% |
| 1 | adjusted | `+2.5257 − 0.0978·T` | 25.8 °C | +0.471 | 7.74% | 90.8% | 5.36% |
| 0 | naive | `+1.5391 − 0.0538·T` | 28.6 °C | +0.409 | 15.18% | 89.5% | 5.36% |
| 1 | naive | `+2.9880 − 0.1321·T` | **22.6 °C** | **+0.214** | **21.73%** | 86.9% | 5.95% |

"No Γ\*" is §40's genuinely empty case: the deficit exceeds `u_max · |channel|`, so nothing certifies
*even at exact identification*. The share is **monotone in the authority the model believes** —
0.701 / 0.471 / 0.409 / 0.214 K/h per unit against 7.4 / 7.7 / 15.2 / 21.7% — and within each seed the
confounded arm believes less, by 1.7× on seed 0 and 2.2× on seed 1. So the audit separates the arms
**offline, from the plan alone**, at 2.0× and 2.8× the uncertifiable share. It does *not* order the
four fits by where their channel crosses zero, which was the mechanism this ablation was written to
look for; the crossing matters through the authority it destroys, not on its own. Seed 1's confounded
fit is the case where the two coincide — it crosses at 22.62 °C, within 0.2 K of where this MPC targets
(`bound + margin` = 22.5), so its believed authority collapses exactly where the controller lives, and
that arm is both the least certifiable and the one pinned at an actuator bound 98.8% of the time.

### What enforcement changes: one command in 336, in all eight episodes

| seed | arm | `tdis_tot` off → on | `ener_tot` off → on | `sat` off → on | commands moved |
|---:|---|---|---|---|---:|
| 0 | adjusted | 7.2764 → **7.1415** (−1.85%) | 2.1742 → 2.1764 (+0.10%) | 0.295 → 0.295 | 1 / 336 |
| 0 | naive | 7.3212 → **7.1415** (−2.45%) | 2.2099 → 2.2156 (+0.26%) | 0.420 → 0.414 | 1 / 336 |
| 1 | adjusted | 7.2774 → **7.1415** (−1.87%) | 2.2206 → 2.2192 (−0.06%) | 0.417 → 0.420 | 1 / 336 |
| 1 | naive | 7.3212 → **7.1415** (−2.45%) | 1.9374 → 1.9371 (−0.01%) | 0.988 → 0.982 | 1 / 336 |

**The filter is nearly inert, and the reason is not that the certificate is slack.** On seed 0's
de-confounded arm the certificate refuses the applied step 16 times, and **15 of those 16 steps were
already at full modulation**. With `w_comfort = 800` against an effort weight of 1, a comfort MPC
saturates wherever the zone is below its bound, so the barrier and the objective want the same thing
and there is nothing left to clip. On this plant the certificate can only change a command where the
*objective*, not the actuator, left the margin unspent. That is a property of a comfort-dominated cost,
not of certificates, and it is the honest reason the enforcement column is this thin.

**The one step it does move is a discretisation the planner cannot see.** Traced on one episode
(seed 0, de-confounded, audit only) and inferred for the other seven from the identical `1 / 336` and
a mean `|Δu|` of exactly 1.0: it is **step 14**, the last occupied half-hour before the night setback.
Every episode starts from the same `initialize(0, 0)`, so the schedule is shared. The MPC's forecast
window is `slice(1, 17)`, so
its first *scored* time is `t + Δt` and every bound it sees is already 15 °C: it commands **0**. The
barrier's floor is the bound in force at `t`, 21 °C, and the zone sits 0.397 K below it, so the
guaranteed derivative is −0.520 against a required +0.397, Γ\* is undefined, and the filter raises the
command to 1. A receding-horizon planner cannot score the constraint active at `t` — no action taken at
`t` changes the temperature at `t` — while a control-barrier condition constrains the *rate leaving*
`t`, which the action does control. The two are asking different questions, and this is the step where
the difference is load-bearing.

**`7.321` was not the floor.** §4 reads that number as one the setpoint schedule builds in and no
command can avoid. All four certificate-on arms land on **7.1415** K·h — to seven digits, from four
different trajectories with saturation shares from 0.295 to 0.982 — so the floor of *this* schedule is
7.1415, and 7.321 was the floor of controllers that miss the pre-setback step. The 2.5% that separates
them is one half-hour of heating per week.

**Do not over-read the gain.** BOPTEST bills `tdis_tot` against the same comfort bound the barrier is
built on, so here the safety constraint and the scored objective coincide; a certificate that helps a
KPI it is aligned with is not evidence that certificates pay in general. What generalises is the
diagnostic — the uncertifiable share is computed from the plan and the fit, before any command is
issued, and it separates a confounded model from a de-confounded one by 2–3× on a real plant.

## 10. What this track does not claim

- No ground-truth channel; the reference is a randomised design, not a known number.
- No steady-state gain (§1).
- **The closed-loop win is small in absolute terms and floor-limited.** De-confounding takes
  discomfort from 7.892 to 7.295 K·h with a 51× tighter spread, and it is the only arm whose margin
  behaves like a conservatism dial (§4) — but the setpoint schedule builds a floor in at `7.1415`
  (§9 corrects the `7.321` §4 reads as one), so the headroom being competed for is under 8% of the
  metric. A plant with more room to lose would test the claim harder than this one does.
- **No win from channel pessimism** (§5), and no supported explanation for why. The earlier version
  attributed it to the drift being the dominant error, on correlations that turn out to be computed
  against 0.045 K·h of variation at the floor. That hypothesis is now untested rather than confirmed.
- The pessimism radius is **not calibrated**. `run_pessimism_sweep`'s default grid comes from the
  seed-to-seed spread of a five-seed run on this one case, which is an empirical yardstick for this
  plant and not a coverage guarantee for any other.
- **The physics-off arm is not shown to be worse in the mean** (§6). With three of five fits refused
  there is no mean to compare, and five seeds would not support the comparison anyway. The claims
  there are counts and signs — a fitted decay that is positive at the logged action on three seeds,
  three refusals, and one ordering reversal between held-out error and plannability. It is also a
  claim about *this* box on *this* log: 768 nonstationary rows with 15% overlap. A longer log
  justifies a bigger box and might change the answer.
- **No number in this document should be read as reproducible below the digits it is printed to.**
  Measured, not assumed: the ten identification logs and the affine fits on them are bit-reproducible
  across sessions (§1); the closed loop is exact wherever an arm sits on the comfort floor and
  reproduces to ~1% on the one episode that saturates 99% of its steps (§4); `b₀` in the
  zero-overlap rows of §3 reproduces only to three decimals; and the black-box arm is reproducible
  only at a *pinned precision* (§8, defect 7).
- **Only `bestest_hydronic_heat_pump` has been run closed-loop.** The other two cases have been
  logged and fitted (§7) and their fits are stable at the operating point, but no MPC episode has
  completed on either, so every closed-loop number in §4–§6 and §9 is from the one case.
- **§7 identifies the artefact, not the plant.** Knowing that `λ(ū)` is −1.40 on hydronic says the
  fit decays where it was logged; it does not say the affine class is adequate there. Both setpoint
  cases actuate behind a local PI loop, where control-affinity is a local approximation, and the fit
  changing sign inside the actuator box is evidence against the class rather than for it.

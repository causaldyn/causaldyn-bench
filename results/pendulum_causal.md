# Track J — the same question on a plant that is not a building

`Pendulum-v1` from Gymnasium 1.3.0 via `causaldyn_bench.pendulum_causal`, Fedora, CPU, JAX float64.
4000-step identification logs (3200 after a 20% chronological holdout), 200-step control episodes,
5 seeds, `dt = 0.05 s`. Every arm sees the same model class, the same MPC, the same 400 optimiser
iterations and the same number of samples.

```bash
JAX_ENABLE_X64=1 uv run python -c "
from causaldyn_bench.pendulum_causal import run_case, summarise
case = run_case(seeds=(0,1,2,3,4))
for name, rows in sorted(case['arms'].items()):
    print(name, summarise(rows, 'gain'))"
```

## Why a second plant, and why this one

Every number in `results/boptest_causal.md` comes from one plant, and a thermal zone is a forgiving
one: slow, over-damped, open-loop stable, and — the part that matters most — **without a ground
truth**. On an emulator the best available yardstick is identification *by design*: fit the
randomised log and call it the reference. That is a good yardstick and it is not the answer.

`Pendulum-v1` gives the answer. Its `step` integrates

```
thdot' = thdot + (3g/(2l) sin(th) + 3/(m l^2) u) dt
th'    = th + thdot' dt                                  (semi-implicit: the *updated* velocity)
```

with `g = 10, m = 1, l = 1, dt = 0.05`, so the control channel is exactly `3.0` and the gravity
coefficient exactly `15.0`. Both are read off the live object rather than transcribed. Three further
differences from the emulator, each deliberate:

* **Open-loop unstable** where the controller works. `th = 0` is upright; gravity pushes away from
  it with a time constant of `1/sqrt(15) = 0.26 s`. A thermal zone forgives a wrong model for hours.
* **A physics prior worth having.** `sin(th)` is known from first principles; the actuator gain —
  the inverse inertia — is not. That is exactly the split the library is named for.
* **Excitation and control live in different places.** The pendulum can be swung to any amplitude
  about the *downward* equilibrium and cannot be held anywhere past `|th| = asin(u_max * 3 / 15) =
  0.41 rad` of upright. So the identification log covers `±1.6 rad` about hanging and the control
  task sits 3 radians away from all of it. Extrapolating that far is what structure buys.

**The confounder is a wind torque.** An exogenous AR(1) disturbance `w` (`sd 0.175`, `rho 0.6`,
capped at `±0.5`) is added to the commanded torque, so the applied torque is `u + w` and `w` enters
the state rate directly. The logging operator partially compensates it:

```
u  <- -k w + e     (see the wind, push back, explore with e)
th'' <- +3 w       (the wind moves the plant)
```

`Cov(u, w) < 0`, and a regression of the rate on `(1, th, thdot, u)` credits the actuator with the
wind's work. `w` is *logged*; withholding it is the causality-off ablation, exactly as withholding
the weather channel is on BOPTEST. Neither clip of the environment is ever reached — the harness
raises `ActuatorClipError` rather than fitting a plant that saturated.

The 2×2 is the emulator's, with the logging policy in place of the outdoor reset. The `random` arm
commands the **same action variance** as the `reactive` one (`var(e') = k² var(w) + var(e)`), so no
arm is bought extra excitation.

## 1. Identification — scored against the answer, not against a reference

Fitted control channel `b`, truth `3.0`, mean ± s.e. over 5 seeds.

| arm | fitted `b` | range over seeds | \|error\| | held-out rate RMSE | action-residual variance |
|---|---:|---|---:|---:|---:|
| **reactive-adjusted** | **+3.000000 ± 0.000000** | [+3.000000, +3.000000] | **2.6e−07** | 0.5141 | 0.00359 |
| random-adjusted | +3.000000 ± 0.000000 | [+3.000000, +3.000000] | 8.4e−08 | 0.5141 | 0.01123 |
| random-naive | +2.984575 ± 0.037009 | [+2.865210, +3.093884] | 0.058672 | 0.5143 | 0.01122 |
| **reactive-naive** | **−1.052820 ± 0.025123** | [−1.106743, **−0.968651**] | **4.052820** | **0.2970** | 0.01112 |

Four things, in the order they matter:

1. **The confounded arm gets the sign wrong on every seed.** Not attenuated — reversed. The range
   `[−1.107, −0.969]` does not come near zero, so this is not a noisy estimate that happened to
   cross over; the estimator is confident that pushing the pendulum clockwise accelerates it
   anticlockwise. This is `chc.benchmark.CausalDynamicsTask`'s headline sign flip reproduced on
   third-party physics rather than on a self-authored DGP.
2. **Adjustment recovers the constant to 7 digits.** The plant is deterministic and the fitted class
   contains it, so the Robinson residual is exactly zero and the recovery is exact up to float64.
   The reported sandwich standard error is `0.0` — the one situation where that is not an
   understatement.
3. **Adjustment does not distort the randomised arm.** `random-naive` at `+2.985 ± 0.037` and
   `random-adjusted` at `+3.000000`: the estimator improves an already-unbiased design by removing
   its finite-sample noise, and does not move it anywhere else. That direction is the falsifier, and
   it holds.
4. **Held-out prediction ranks the arms backwards.** `reactive-naive` predicts the next angular
   acceleration **1.73× better** than the arm that gets the channel right (0.297 against 0.514), and
   it does so *by construction*: the unadjusted fit is the best linear predictor of the rate given
   the state and the commanded action, and the holdout is drawn from the same confounded policy. A
   dynamics benchmark scored in rollout error would rank the sign-flipped model first. This is the
   same claim §6 of `results/boptest_causal.md` makes, without the emulator's ambiguity about what
   the right answer was.

The adjusted arm identifies from a **3.1× smaller slice** of the action's variance (0.0036 against
0.0111), because adjusting for the wind removes the `−k w` component that the naive arm is using.
On a noisy plant that is a variance premium and it is why the BOPTEST reset-adjusted arm carries
2.4× the reference's seed spread. Here the plant is deterministic, so the premium costs nothing and
**this track cannot measure it**. That limitation is the emulator's to cover, not this one's.

## 2. Structure and causality are different axes, and neither is prediction

Same five logs, `reactive` policy. `extrapolation` is the RMS error of the predicted angular
acceleration over the *upright* box (`|th| < 0.5`, `|thdot| < 2`, `|u| < 1`) — where the controller
works and where the log never goes.

| arm | fitted `b` | range over seeds | held-out RMSE | extrapolation RMSE |
|---|---:|---|---:|---:|
| **physics-adjusted** | **+3.000000 ± 0.000000** | [+3.0000, +3.0000] | 0.5141 | **0.0271** |
| flexible-adjusted | +2.872090 ± 0.107510 | [+2.6166, +3.1734] | **0.5109** | **22.4717** |
| physics-naive | −1.052820 ± 0.025123 | [−1.1067, −0.9687] | 0.2970 | 2.3335 |
| flexible-naive | −0.959499 ± 0.085655 | [−1.2675, −0.7400] | 0.2991 | 9.8911 |

`flexible` replaces the known `sin` with a degree-5 polynomial drift in `(th, thdot)` — *more*
parameters than the structured arm, fitted on the same rows, so this is not a straw man.

* **Dropping the physics prior does not bias the channel.** `+2.872 ± 0.108` against a truth of
  `3.0`; the logged action is exogenous to the angle, so the polynomial's misfit of gravity lands in
  the error term rather than in the estimate. What it costs is *precision*: a seed spread of 0.24
  where the structured arm has none.
* **Dropping the physics prior costs 829× on extrapolation** (22.47 against 0.0271 rad/s²), and the
  gap does not close with data — the same measurement on un-split logs of 4000 and 12000 rows reads
  16.13 and 8.87, against a structured arm that stays under 0.04.
* **Dropping the adjustment costs the sign and *improves* held-out error** (0.297 against 0.514).
* **Among the two arms that get the sign right, the flexible one predicts marginally better**
  (0.5109 against 0.5141). It is also the one that cannot be planned with at all — see §5.

So: structure buys extrapolation, adjustment buys the channel, and held-out prediction buys neither
and mis-ranks both. Three metrics, three different winners, on one set of fits.

## 3. Overlap — the assumption everything rests on

Seed 0, `reactive`, sweeping the operator's exploration noise `sd(e)` to zero.

| `sd(e)` | action-residual variance | naive `b` | adjusted `b` | adjusted \|error\| |
|---:|---:|---:|---:|---:|
| 0.250 | 6.343e−02 | +2.3240 | +3.0000 | 0.0000 |
| 0.120 | 1.460e−02 | +0.8892 | +3.0000 | 0.0000 |
| 0.060 | 3.648e−03 | −1.0914 | +3.0000 | 0.0000 |
| 0.025 | 6.332e−04 | −2.5411 | +3.0000 | 0.0000 |
| **0.000** | **1.984e−21** | **−3.0000** | **−0.0000** | **3.0000** |

The last row is the whole argument in one line. With no exploration the operator's action is exactly
`−0.5 w`, so the applied torque `u + w` is identically `−u`, and the unadjusted fit returns
**minus the truth**: `−3.000` for a `+3.0` actuator. It is not a large error, it is the negation of
the answer, and no sample size repairs it. The adjusted arm returns `0.000` instead — with the action
residual at `2e−21` there is nothing left to regress on — which is the honest failure of the two: a
zero says *not identified*, a confident `−3` does not.

Everything above the last row is monotone and unsurprising, which is the point of including it: the
sign flip is not a knife-edge, it is the middle of a continuum.

## 4. The bias is a formula, and the formula was falsifiable

With applied torque `b(u + w)`, `u = -k w + e`, and `w` independent of `e` and of the state, the
unadjusted least-squares channel is

```
b_hat = b * (k(k-1) s_w² + s_e²) / (k² s_w² + s_e²)
```

which is negative exactly for `k` between the roots of `k² - k + s_e²/s_w²` — an interval strictly
inside `(0, 1)`, and `(0.132, 0.868)` at the shipped scales. Two predictions made before the sweep:
the bias is **not monotone** in the operator's diligence, and an operator who compensates the wind
*perfectly* is better off than one who compensates a quarter of it.

Seed 0, sweeping `k`:

| compensation `k` | naive `b` | closed form | gap | adjusted `b` |
|---:|---:|---:|---:|---:|
| 0.00 | +2.9173 | +3.0000 | **0.0827** | +3.000000 |
| 0.25 | **−1.2201** | −1.2292 | 0.0092 | +3.000000 |
| 0.50 | −1.0914 | −1.1114 | 0.0200 | +3.000000 |
| 0.75 | −0.3089 | −0.3218 | 0.0130 | +3.000000 |
| 1.00 | **+0.3172** | +0.3090 | 0.0081 | +3.000000 |
| 1.50 | +1.1011 | +1.0971 | 0.0039 | +3.000000 |
| 2.00 | +1.5444 | +1.5419 | 0.0025 | +3.000000 |

Both predictions hold, and the sign changes where the algebra says it does. The largest gap is at
`k = 0`, where the closed form *is* the truth and the 0.083 is the estimator's own sampling error
rather than the formula's; wherever the sign flips the gap is at most 0.020. The wind is
autocorrelated (`rho = 0.6`) while the derivation assumes it white, so agreement this close is
worth naming: the state's dependence on past wind is worth about 2% of the gain here.

The sign flip is not a seed accident either — at `k = 0.5` the five seeds give `−1.091, −0.986,
−1.118, −1.061, −1.040`, against closed-form predictions of `−1.111, −0.942, −1.102, −1.028,
−1.051`.

## 5. Closed loop — regret against the answer

Stabilise the pendulum upright from `th0 = 0.15 rad` against the same wind, 200 steps (10 s), scored
by **Gymnasium's own reward function** (`angle_normalize(th)² + 0.1 thdot² + 0.001 u²`, accumulated).
The commanded torque is bounded by `max_torque − wind_cap = 1.5` so `u + w` never reaches the
environment's clip. No arm sees the wind ahead of time.

| arm | refused | cumulative cost | range over seeds | fell | worst \|th\| | Σu² | regret vs oracle |
|---|---:|---:|---|---:|---:|---:|---:|
| **oracle** (the environment's own dynamics) | 0/5 | 0.3189 ± 0.0222 | [0.2827, 0.4038] | 0/5 | 0.1457 | 12.79 | — |
| **adjusted** | 0/5 | **0.3289 ± 0.0292** | [0.2828, 0.4425] | 0/5 | 0.1457 | 13.12 | **0.0100** |
| **naive** | 0/5 | **1537.04 ± 3.61** | [1527.86, 1548.53] | **5/5** | 3.1260 | **450.00** | **1536.72** |
| flexible-adjusted | **5/5** | — | — | — | — | — | — |

* The de-confounded controller is **within 3.1% of the oracle's total cost**, and its residual gap
  is not the channel — that is exact to 2.6e−07 — but its fitted degree-1 drift, whose coefficients
  come back at the `1e−3` level instead of the zero they should be. Its worst deviation is 0.1446 to
  0.1466 rad across seeds, against a 0.15 rad start: it never overshoots, on any seed.
* The confounded controller **falls on all five seeds** and spends `Σu² = 450.000` doing it, which is
  exactly `200 × 1.5²`: the actuator is saturated at the bound on every step of every episode,
  pushing the wrong way, because that is what its model says to do. Its regret is **1.5e+03 against
  the adjusted arm's 1.0e−02**, a factor of 1.5e+05.
* The flexible-drift arm **refuses**: its degree-5 polynomial, fitted across the swing and evaluated
  3 radians away, diverges over the horizon and the plan comes back non-finite on all five seeds.
  Recorded as a refusal and not as a large cost, because a model with no plan has not produced a bad
  one, and averaging the two together would report neither. The arm with the best held-out one-step
  error in §2 is the arm with no controller.

This is the comparison an emulator cannot offer: not "which arm is closest to the randomised
reference" but "how far is each arm from the answer", in the units of the environment's own
objective.

## 6. The solver, and a defect that would have decided the benchmark

The BOPTEST harness plans with Nesterov at a step of `1/L`, `L` the largest eigenvalue of the
objective's Hessian at the initialisation. Both halves fail here and both were caught by measuring
rather than by reading:

* The horizon rollout is **nonconvex**. At the zero-action initialisation from `th = 0.15` the
  Hessian's spectrum runs `[−15.14, +0.45]`, so the largest *algebraic* eigenvalue is not a bound
  on the gradient's Lipschitz constant and `1/L` overshoots by a factor of 33. With that step the
  oracle model's controller saturated at `+1.5` — the wrong sign — and fell.
* With the correct `max|lambda|` the accelerated method still does not converge, because momentum
  tuned for a convex landscape oscillates on this one. On the oracle model the objective read
  **290.7 at 60 iterations, 7.85 at 600, and 206.1 at 2000**. A method whose answer is non-monotone
  in its budget cannot be handed a fixed budget and called equal compute.

Projected **Adam** ships instead: the same optimiser, the same 400 iterations and the same
projection for every arm, and its per-coordinate normalisation is what makes that fair — Adam's step
is invariant to rescaling the objective, so the arm that identifies a larger channel is not punished
by a shared constant. That was the BOPTEST defect exactly; the fix there was to derive the step from
the model, and the fix here is to use a method that does not need to. Against `scipy.optimize`
L-BFGS-B on the same box-constrained problem Adam lands within 14% of the optimum on the oracle model
and hits it exactly on the confounded one. L-BFGS-B was rejected as the shipped solver because it
terminates adaptively, so the gradient count being equalised would differ per arm. A test pins the
gap.

This is the eighth instance in this benchmark's history of the same family of defect — a constant
whose meaning depends on something it does not name. Here the something is convexity.

## 7. Precision

Identification was re-run at float32 (no `JAX_ENABLE_X64`) on seed 0:

| arm | float64 | float32 |
|---|---:|---:|
| reactive-naive | −1.091441 | −1.091441 |
| reactive-adjusted | +3.000000 | +2.999998 |
| random-naive | +2.954266 | +2.954266 |
| random-adjusted | +3.000000 | +3.000001 |

float32 reproduces float64 to **six significant figures** on every arm, and the upright-region
extrapolation errors agree to four decimals. Unlike the BOPTEST black-box arm — where float32 moved
the fitted decay by 10× and produced numbers that were later retracted — the fits here are linear
solves and the precision costs only the last digits of the exactness claim. The recorded numbers are
float64 and the closed loop should stay there, but nothing in §1–§4 hinges on it.

The test suite is run at both. Because CI's ordinary job uses JAX's default float32, the assertions
that state exactness carry two bounds: `1e-06` on the channel and `1e-12` on the oracle's step at
float64, relaxed to a measured float32 noise floor (`1.7e-06`, `1.3e-05` relative between the two
channel rows, and `2.2e-07` on the step) otherwise. The seven-digit figure this section reports is
therefore not checked by the ordinary job — a second CI job, `test-x64`, re-runs the whole suite
under `JAX_ENABLE_X64=1` and gates it.

Running the suite at both precisions turned up something worth stating separately, because it is not
a precision effect and it is easy to mistake for one: **a `seed` identifies a synthetic dataset only
at a fixed dtype.** `jax.random` spends a different number of threefry bits per element for a float64
output than for a float32 one, so `JAX_ENABLE_X64=1` draws a *different* Monte-Carlo instance rather
than a higher-precision copy of the same one. On Track H every coordinate of the market's `demand`
moves by O(1) and the oracle's headroom falls from 2.42 to 1.06 — which is why that track's
assertions are now shares of the oracle lift rather than absolute lifts. Track J is immune by
construction: `Pendulum-v1`'s constants are the environment's, not a draw's, and the wind is drawn by
`numpy` at float64 regardless.

## 8. What this track does not claim

* **It does not measure the variance premium of adjustment.** The plant is deterministic, so the
  adjusted arm's 3.1× smaller identifying slice costs it nothing. On a noisy plant it would; the
  emulator track measures that and this one cannot.
* **The confounding mechanism is ours.** The plant, its constants, its integrator, its clips and its
  objective are Gymnasium's; the wind and the operator who compensates it are added by this harness
  through the action port. What is external here is the *physics being identified*, not the
  experimental design.
* **The excitation design is not a production log.** Segments of 200 steps from random amplitudes
  about the downward equilibrium is a laboratory experiment, chosen because the plant is unstable
  upright and the logging operator is a disturbance-rejection controller rather than a stabiliser.
  A log from a real stabilising controller would sit near upright and identify from far less state
  variation.
* **One horizon, one cost, one initial displacement.** §5 is a single operating point, not a
  frontier. The BOPTEST track needed `run_pareto` to show that a channel error's *direction* of harm
  depends on the cost; nothing here contradicts that, and nothing here tests it either.
* **`b = 3.0` is the whole estimand.** The pendulum's channel is a scalar constant. The matrix and
  state-dependent cases the library supports are exercised on the synthetic fixture, not here.

## 9. Summary

* On a plant where the answer is known, causal adjustment recovers the control channel to **7
  digits**, from a log where the unadjusted fit gets the **sign backwards on every seed**.
* The bias follows a closed form that predicts, correctly, that **compensating a quarter of the
  disturbance is worse than compensating all of it** — the sign flips only on an interval strictly
  inside `(0, 1)`.
* With no exploration the unadjusted fit returns **exactly minus the truth**, and the adjusted fit
  returns zero. Overlap is not a technicality.
* Closed loop, in the environment's own objective: the de-confounded controller is **3.1% off the
  oracle**; the confounded one **saturates the actuator on every step and falls on every seed**.
* **Held-out prediction ranks all of this backwards.** The best one-step predictor in §1 has the
  wrong sign; the best one-step predictor in §2 has no plan at all.

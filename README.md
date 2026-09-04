# causaldyn-bench

[![ci](https://github.com/causaldyn/causaldyn-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/causaldyn/causaldyn-bench/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![doi](https://zenodo.org/badge/DOI/10.5281/zenodo.22139814.svg)](https://doi.org/10.5281/zenodo.22139814)

A benchmark for **causal, constrained, dynamical decision-making** — it scores *predictions and
decisions*, not just one-step error. The point is to measure methods on the axis each deserves, so a
gradient-boosted tree can win one-step prediction while a causal, constrained controller wins the
closed-loop decision. Built on
[`causal-hybrid-control`](https://github.com/causaldyn/causal-hybrid-control).

## The tracks

| track | question | metric | expected winner |
|---|---|---|---|
| **A** one-step | `x_t, u_t → x_{t+1}` | RMSE | tree surrogate |
| **B** rollout | `x_t, u_{t:t+H} → x_{t+1:t+H}` | rollout RMSE | hybrid (physics anchors the horizon) |
| **C** counterfactual | `x_{t+1}(do(u))` under confounding | \|effect − truth\| | causal / Double ML |
| **D** control | `u_t = π(x_t)` under constraints | regret vs oracle | causal hybrid controller |
| **D-planner** planner vs model | the same objective planned by a gradient and by a sampler, on three models | regret on the true plant | *neither* — the model axis decides it |
| **E** systems | control-solve latency | ms (min of 7, after 2 warm-ups) | the known-only model — fewer terms to differentiate, same compiled solver |
| **F** structure | which lagged parents drive the target, under confounding + autocorrelation | F1 / control payoff | discovery-informed residual |
| **G** dynamic effect | the impulse response `∂x_{t+h}/∂u_t`, not just `h = 1` | IRF error / control payoff | structured (Levinson) IRF |
| **H** marketplace | offline incentive allocation when SUTVA fails through a shared equilibrium | regret vs equilibrium-aware oracle | de-confounded + equilibrium-aware |
| **I** sensitivity | control when **no adjustment set exists** — the assumed `Γ` is the only lever | worst-case closed-loop cost | a *calibrated* `Γ`, not the largest one |
| **D-causal** identification | the control channel of a *real* emulator, logged by a weather-compensated controller | \|8h step response − randomised reference\| | orthogonal (de-confounded) fit |
| **J** identification, non-building | the control channel of a *third-party* plant whose answer is known exactly (`Pendulum-v1`) | \|fitted gain − 3.0\| | orthogonal (de-confounded) fit |
| **K** delay identification | *when* the incentive acts, from a log whose confounder acts at a **different** lag | \|τ̂ − τ\| / closed-loop regret | adjusted local projection |
| **N** fold design | which cross-fitting **split** to use under network interference — the method is held fixed and only the folds vary | MSE vs a graph-blind unit split | *the design law, negatively* — it convicts the two splits a practitioner reaches for |

**Track N** (`causaldyn_bench.fold_design`) is the only track that varies nothing but the
cross-fitting split, and its result is an ordering whose useful half is negative. On `C_12` with two
clusters, over 120 draws: the Result 52 design **ties** the graph-blind split that keeps units
intact, while the two splits a practitioner reaches for first cost **+37%** MSE (contiguous graph
blocks) and **+66%** (Emmenegger-style neighbour exclusion). One number explains all three — the
fraction of edges left inside a fold: `0.50` designed, `0.46` random units, `0.83` contiguous.

The law also forecasts *where* it matters. Its mass ratio designed/contiguous is `0.720` on the
cycle, `0.974` on a `3×4` torus and `0.966` on a random cubic graph; measured, the arms separate on
the cycle and scatter inside `0.85–1.16` on the other two. And the whole effect is `O(1/g)` in the
number of independent clusters (`1.370, 1.139, 1.083, 1.047` across `g = 2, 4, 8, 20`), so fold
design is a **small-cluster-count** instrument — a handful of cities, not twenty replicas. Neighbour
exclusion is worse than every alternative where it runs and cannot run at all at `K = 2` on either
denser graph: its hop-1 neighbourhood covers the training fold. Buying validity by discarding data
needs a split that is already graph-aware — the design it was meant to replace.

**Track K** (`causaldyn_bench.delay_identification`) is the only track whose payoff is
*discontinuous*. Every other board scores a cost gap; here the closed loop is `x' = -K·x(t − τ)`,
whose exact boundary is `K·τ = π/2`, so getting the delay wrong enough is a Hopf bifurcation rather
than a worse number. It also scores something no other track does — the **argmax** of an effect. The
confounder acts at 0.6 s while the incentive acts at 1.0 s, so an unadjusted impulse response peaks
on the *confounder's* lag: the estimate is wrong about **when**, not about how much.

The three tiers are the result. Ignoring the delay diverges (regret `18953`, gain `3.10` = 1.97× past
`π/2`); estimating it *badly* still stabilises and pays `0.68`, because the stabilising set in delay
space is a **half-line** — under-estimating survives down to `2/(πe) = 0.234`, and `0.6/1.0` is well
inside it; adjusting reaches `0.016`, and sub-grid refinement `0.0067`. Note the two ratios: a 4×
smaller delay error buys a **42×** smaller regret. Cross-correlation and an *unadjusted* local
projection are carried as separate rows and score identically — they are the same argmax, so the
board's gap is adjustment, not the estimator family. This track is not allowed to flatter CHC by
comparing its adjusted estimator against a weaker unadjusted one.

The track also states the limit of its own claim. Aggregated over one observation stride the
incentive splits 2:1 across lags 3 and 4 while the confounder lands wholly in lag 2, so the peak
relocates iff `σ_η² < 1.5·|c·κ|·σ_z²/|b| − κ²·σ_z²` — `σ_η = 1.4697` here, bracketed by measurement
in `[1.45, 1.50]`, and the 2:1 split is visible in the fitted response (`0.168` at lag 3, `0.085` at
lag 4). So **enough exploration in the logging policy recovers the delay with no adjustment at all**.
The failure is a property of a thin log, not of cross-correlation; the shipped `σ_η = 0.5` sits
deliberately below the threshold.

Track I is the odd one out on purpose: it scores a **modelling assumption**, not a method. The
confounder is absent from the log, so nothing can be estimated better; the board carries a
deliberately under-assumed and a deliberately over-assumed `Γ` beside the calibrated one, and the
score is non-monotone in `Γ`. "More pessimism is better" is a claim this track exists to refute.

Track D bundles the CHC oracle-regret tasks (pricing / inventory / support-shift) **and** an
**adaptive-CV-compute** task (`adaptive_cv`): split a shared GPU budget across video streams under
known, bursty arrivals and heterogeneous priorities. A priority-blind, load-proportional myopic split
crowds out critical streams; the constrained CHC-MPC plans over the known dynamics and matches the
oracle. First numbers — CHC-MPC regret `0.0`, myopic `166`, uniform `330`.

**Track D-planner** (`causaldyn_bench.shooting`) exists to let CHC lose. Cross-entropy-method
planning needs no adjoint, no Jacobian and no differentiable model — only rollout evaluations —
so crossing it with the gradient planner over three models (true plant / learned hybrid /
physics-only) separates what the library's adjoint machinery is worth from what *learning the
residual* is worth. It is not close, and not in the flattering direction for the adjoint: regret
on the true plant is `plant/gradient` 0, `hybrid/gradient` 0.0016, `plant/cem` 0.0019,
`hybrid/cem` 0.0030, against `known_only/*` at **1.72**. The model axis is ~1000× the planner
axis, and the sampling planner with full knowledge of the plant is *worse* than the gradient
planner on a learned model. The defensible claim is therefore about identification and the
learned residual, not about the solver.

Design rule: to be honest about the win, **never** claim "best model" — claim the decision under a stated
budget. Track A is expected to go to the trees; the value is Tracks B–D.

## Run

```bash
uv sync --extra trees --extra gym        # tree baselines for A/B, Gymnasium for Track J (optional)
uv run python -m causaldyn_bench         # print the leaderboard
uv run python -m causaldyn_bench --save  # also write results/leaderboard.{md,json}
uv run pytest                            # smoke tests
```

This repo depends on `causal-hybrid-control` through a sibling **path**, so it expects the two checked
out next to each other. CI reproduces that layout with two checkouts and runs the full suite.

Without the `trees` extra the tree baselines are simply omitted; the hybrid/causal methods still run.
On Tracks A/B the dynamics competitors are **known-only** (true physics), **dlm** (data-driven linear /
state-space), **tree-surrogate** (LightGBM), and **hybrid-CHC** (physics + learned residual).

A committed snapshot lives in [`results/leaderboard.md`](results/leaderboard.md): hybrid wins B-rollout
~18× over the tree and ~14× over the DLM, causal wins C ~800× over naive, causal-CHC is near-oracle on
D while predictive blows up, and CHC-MPC matches the oracle on the adaptive-CV task (myopic loses).

For the visual version — leaderboard bar charts + the "prediction ≠ decision" figure — see the executed
notebook [`notebooks/leaderboard.ipynb`](notebooks/leaderboard.ipynb) (renders on GitHub), or run it:

```bash
uv sync --extra trees --group notebooks
uv run --group notebooks jupyter lab   # notebooks/leaderboard.ipynb
```

## Running BOPTEST — the real Track-D HVAC target (Fedora / Podman)

Track D can run against a live **BOPTEST-Service** via `causaldyn_bench.boptest`. Current BOPTEST deploys
as a web-service; on Fedora it runs under **Podman** (no Docker needed):

```bash
# one-time tooling (podman ships with Fedora; add the compose front-end, no sudo)
uv tool install podman-compose               # or: sudo dnf install -y podman-compose

# clone + bring up the service (first build is ~4 GB, ~15-30 min)
git clone https://github.com/ibpsa/project1-boptest.git
cd project1-boptest

# SELinux: relabel the bind mount, or the containers cannot read the tree they are handed.
# Fedora mounts with SELinux enforcing and the upstream compose file predates rootless Podman,
# so without the :z suffix the worker fails on permission errors that name no cause.
sed -i 's|- ./:/usr/src/boptest$|- ./:/usr/src/boptest:z|' docker-compose.yml

podman-compose up web worker provision       # REST API at http://127.0.0.1:8000
curl http://127.0.0.1:8000/version           # sanity-check once it is up
```

Then point the client at it (from this repo):

```bash
BOPTEST_URL=http://127.0.0.1:8000 uv run pytest tests/test_boptest.py   # the live episode test runs
BOPTEST_URL=http://127.0.0.1:8000 uv run python -c \
  "from causaldyn_bench.boptest import boptest_track; print(boptest_track())"   # baseline KPIs
```

Shut down with `podman-compose down`. Test cases include `bestest_hydronic_heat_pump` (default),
`bestest_air`, `singlezone_commercial_hydronic`, and others. The CHC hybrid-MPC controller
(RC-thermal + learned residual, MPC under comfort constraints) is wired in via
`causaldyn_bench.boptest_chc` — see the results below.

Two operational notes, both learned the hard way. The `mc` bucket-init container exits non-zero on a
second bring-up (the bucket already exists), which blocks `podman start` of anything that depends on
it — `podman-compose down && podman-compose up -d` is the reliable cycle. And the worker runs **one
test at a time**: a client killed with `SIGTERM` skips its `finally: client.stop(testid)`, leaks the
registration in redis, and every later `select` then blocks until the leak is cleared
(`redis-cli KEYS 'tests:*'`). Run long sweeps detached, not under a timeout that kills them.

### Track D-causal — does de-confounding the control channel pay on a real emulator?

`causaldyn_bench.boptest_causal` asks the question the existing harness cannot. `boptest_chc`
identifies its thermal model from a **randomised** exploration episode: a clean experiment, and the
one thing production HVAC data never is. Real logs come from a controller, and every sensible
controller is weather-compensated, so the logged action is a function of the outdoor temperature —
which is also what drives the zone. Regress the temperature rate on `(1, T, u)` and the outdoor term
lands in the error term.

The experiment is a 2×2 over {outdoor-reset, randomised PRBS} × {adjust for weather, don't}, so both
directions are falsifiable rather than only the flattering one: adjustment must repair the
confounded arm **and** must leave the randomised arm alone. If it "helped" the randomised arm too,
the estimator would be distorting rather than de-confounding.

```bash
# JAX_ENABLE_X64=1 is required, not cosmetic. float32 and float64 agree on the *affine* channel to
# 0.6% -- two orders of magnitude inside the effect -- but 3000 Adam steps compound rounding into
# the derivative of the fitted surface, and the physics-off arm's decay moves by 10x and changes
# sign. An earlier float32 run is what produced the retracted numbers in results/ SS6.
JAX_ENABLE_X64=1 BOPTEST_URL=http://127.0.0.1:8000 uv run python -c \
  "from causaldyn_bench.boptest_causal import track_boptest_causal as t; print(t())"
```

Three things this track deliberately does not claim:

- **No ground-truth channel exists** on an emulator. The reference is identification *by design* —
  the randomised log, fitted without adjustment — not a known number.
- **The steady-state gain `−b/a` is not identified** from a 5–20 day window at 30-minute resolution;
  `b₀` and the pole are collinear over that span. The reported quantity is the finite-horizon
  8-hour step response, which the data does pin down.
- **Closed-loop KPIs cannot rank models at a single operating point.** Which way a channel error
  moves the controller is a property of the cost, not of the error: under BOPTEST's
  comfort-dominated objective an *attenuated* channel makes the controller over-actuate, so the bias
  acts as an unintended safety margin that buys comfort and pays energy. `run_pareto` sweeps the
  requested margin and compares frontiers instead.

Identification also needs **overlap**. A perfectly deterministic reset policy makes the action an
exact function of the covariates, the orthogonal moment has no regressor left, and nothing is
identified at any sample size. `overlap_report` measures the surviving share and
`run_overlap_ablation` drives the exploration noise to zero to show the collapse — the assumption
gets a falsifiable curve, not a sentence.

**Physics-off ablation** (`run_structure_ablation`). Against the structured arms sits a black box: an
MLP for `dT/dt` given `(T, z, u)`, trained on the same log, planning through the *same* MPC — the two
arms differ in the model and in nothing else, because the solver takes any `PlantModel` and reads
only its `rate`. It is not a straw man: the confounder is inside its conditioning set, it can fit
nonlinearities the affine model cannot, and it gets far more fitting compute than a closed-form fit.

The point of it is what identification benchmarks usually miss. On the synthetic fixture the black
box's held-out one-step error is **indistinguishable** from the structured causal fit — 4.20e-4
against 4.13e-4, under half a seed standard deviation, both on the 4.0e-4 noise floor — while its
control authority carries **2.6× the RMSE** (0.047 against 0.018 about a truth of 1.200) and a 3.4%
bias against 0.07%. Two models that agree on every forecast they will ever be scored on disagree
measurably on the one number a controller consumes. *Held-out predictive accuracy does not rank
causal models*, so a dynamics leaderboard reported in rollout error cannot see this failure at all.
The other side is tested too: omitting the confounder entirely — the `naive` affine arm — attenuates
the authority to 14% of truth **and** costs 19× the held-out error, and prediction *does* catch that.

The reported estimand is the **authority** `∂(dT/dt)/∂u` at the operating point, not the affine
`b₀`. `b₀` is that channel extrapolated to 0 °C, some 21 K outside anything a heated building
visits; reading it off a nonlinear model measures the extrapolation, and on the emulator it came back
between −0.045 and +0.254 for a plant whose structured fit says +1.25 — a factor of 31, while the two
arms' *authorities* differ by 1.5×.

On the emulator the black box's failure is blunter than on the fixture. Read at the action the log
sat at, its fitted decay is **positive on three of five seeds**: those models assert a zone that
warms away from its own equilibrium, the stability check refuses them offline, and the arm has no
closed-loop mean left to report. Of the two seeds that do plan, one reaches the comfort floor and the
other spends 2.2× the de-confounded arm's discomfort. Sharper still, *within* the black-box arm
prediction is ordered against plannability: of its five fits the **best** one-step predictor
(held-out 0.0301, against 0.0971 for the worst) is one of the three that gets refused.

**Measured result** (5 seeds, 20-day identification episodes, `results/boptest_causal.md`): logged by
a weather-compensated controller the naive fit understates the heat pump's 8-hour authority by
**54.8%**; de-confounding recovers it to within **0.211 K** of the randomised reference, an 11.5×
reduction, and moves the randomised arm by **0.068 K** — 1.5% of the level — while cutting that arm's
spread by 2.3×. Closed loop, the de-confounded arm has both the best mean discomfort (7.295 K·h
against 7.892) and a **51× tighter** seed-to-seed spread (s.e. 0.011 against 0.539), at 0.2% more
energy and roughly half the actuator saturation.

The confounded arm's failure is sharper than attenuation: on seed 1 its fitted channel
`+2.989 − 0.1321·T` changes **sign at 22.62 °C**, inside the occupied comfort band, so above that
temperature the model believes the heat pump cools the room and pins the command at zero. The
de-confounded channel crosses at 25.82 °C, outside the band.

**The certificate closes the loop, and separates the arms before it acts.** Wrapping every MPC horizon
in a `chc.plan.CausalPlan` and pricing it with `certify_safety` (§9) says the confounded fit is
uncertifiable at **any** sensitivity level on 15.2% and 21.7% of control steps, against 7.4% and 7.7%
for the de-confounded one — a 2–3× separation read off the plan, with nothing executed. Enforcing it
through `robust_safety_filter` then moves **one command in 336**: where the barrier is in deficit a
comfort-dominated MPC is already saturated, so the two agree everywhere except the last occupied
half-hour before the night setback, which the planner's horizon cannot score and the barrier can.

An earlier version of this paragraph reported the opposite closed-loop ordering. That was an artefact
of the planner's constant step size, which sat 75–284× past its stability limit — and because the
limit scales with the authority a model *believes* it has, the shared constant punished exactly the
arms that identified the channel best. Equal compute is not an equal iteration count; it is an
optimiser that does not depend on the scale of the model being compared. See §7 of
`results/boptest_causal.md`.

## Status

v0.0.1 scaffold: all five tracks run on the synthetic CHC systems (a damped oscillator with hidden cubic
physics for A/B/E, a confounded linear system for C, the CHC oracle-regret tasks plus the
**adaptive-CV-compute** task for D). A **BOPTEST** (HVAC control) client + control episode ship in
`causaldyn_bench.boptest`, gated on a running BOPTEST service (`BOPTEST_URL`). The CHC identification +
forecast-MPC (`causaldyn_bench.boptest_chc`) is **validated live** on `bestest_hydronic_heat_pump`: it
beats the tuned built-in baseline on *every* KPI at once — a clean Pareto win (see `results/boptest.md`).
**Track D-causal** (`causaldyn_bench.boptest_causal`) adds the falsifiable 2×2 that the randomised-only
harness could not ask, plus a physics-off black-box arm planning through the identical MPC —
`results/boptest_causal.md`.

**Track J** (`causaldyn_bench.pendulum_causal`, needs the `gym` extra) re-asks that 2×2 on a plant
that is **not a building** and whose answer is known exactly: Gymnasium's `Pendulum-v1`, whose
`step` fixes the control channel at `3/(m l²) = 3.0`. Adjustment recovers it to seven digits from a
log where the unadjusted fit returns `−1.053 ± 0.025` — the **sign backwards on all five seeds** —
and closes the loop within 3.1% of an oracle that the confounded controller misses by a factor of
4800 while saturating the actuator on every step. Held-out one-step error ranks all of it backwards.
Reproducing the seven digits needs `JAX_ENABLE_X64=1`; at JAX's default float32 the recovery is exact
to six. See `results/pendulum_causal.md`.

```python
from causaldyn_bench.boptest import BOPTestClient, baseline_controller, run_episode

kpis = run_episode(
    BOPTestClient("http://127.0.0.1:8000"), baseline_controller()
)  # needs a live service
```

## License

MIT © Ilia Gradina

# causaldyn-bench

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
| **E** systems | control-solve latency | ms | (a compiled runtime, later) |
| **F** structure | which lagged parents drive the target, under confounding + autocorrelation | F1 / control payoff | discovery-informed residual |
| **G** dynamic effect | the impulse response `∂x_{t+h}/∂u_t`, not just `h = 1` | IRF error / control payoff | structured (Levinson) IRF |
| **H** marketplace | offline incentive allocation when SUTVA fails through a shared equilibrium | regret vs equilibrium-aware oracle | de-confounded + equilibrium-aware |
| **I** sensitivity | control when **no adjustment set exists** — the assumed `Γ` is the only lever | worst-case closed-loop cost | a *calibrated* `Γ`, not the largest one |
| **D-causal** identification | the control channel of a *real* emulator, logged by a weather-compensated controller | \|8h step response − randomised reference\| | orthogonal (de-confounded) fit |

Track I is the odd one out on purpose: it scores a **modelling assumption**, not a method. The
confounder is absent from the log, so nothing can be estimated better; the board carries a
deliberately under-assumed and a deliberately over-assumed `Γ` beside the calibrated one, and the
score is non-monotone in `Γ`. "More pessimism is better" is a claim this track exists to refute.

Track D bundles the CHC oracle-regret tasks (pricing / inventory / support-shift) **and** an
**adaptive-CV-compute** task (`adaptive_cv`): split a shared GPU budget across video streams under
known, bursty arrivals and heterogeneous priorities. A priority-blind, load-proportional myopic split
crowds out critical streams; the constrained CHC-MPC plans over the known dynamics and matches the
oracle. First numbers — CHC-MPC regret `0.0`, myopic `166`, uniform `330`.

Design rule: to be honest about the win, **never** claim "best model" — claim the decision under a stated
budget. Track A is expected to go to the trees; the value is Tracks B–D.

## Run

```bash
uv sync --extra trees                   # tree baselines for Tracks A/B (optional)
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

The confounded arm's failure is sharper than attenuation: its fitted channel `+2.989 − 0.1321·T`
changes **sign at 22.62 °C**, inside the occupied comfort band, so above that temperature the model
believes the heat pump cools the room and pins the command at zero. The de-confounded channel crosses
at 25.82 °C, outside the band.

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

```python
from causaldyn_bench.boptest import BOPTestClient, baseline_controller, run_episode
kpis = run_episode(BOPTestClient("http://127.0.0.1:8000"), baseline_controller())  # needs a live service
```

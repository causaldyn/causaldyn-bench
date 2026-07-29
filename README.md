# causaldyn-bench

A benchmark for **causal, constrained, dynamical decision-making** — it scores *predictions and
decisions*, not just one-step error. The point is to measure methods on the axis each deserves, so a
gradient-boosted tree can win one-step prediction while a causal, constrained controller wins the
closed-loop decision. Built on [`causal-hybrid-control`](../causal-hybrid-control).

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
out next to each other. CI therefore lints and format-checks only — a runner has no sibling to resolve
against while that repo is private, and a test job would be red for a reason that has nothing to do
with the benchmark.

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

## Status

v0.0.1 scaffold: all five tracks run on the synthetic CHC systems (a damped oscillator with hidden cubic
physics for A/B/E, a confounded linear system for C, the CHC oracle-regret tasks plus the
**adaptive-CV-compute** task for D). A **BOPTEST** (HVAC control) client + control episode ship in
`causaldyn_bench.boptest`, gated on a running BOPTEST service (`BOPTEST_URL`). The CHC identification +
forecast-MPC (`causaldyn_bench.boptest_chc`) is **validated live** on `bestest_hydronic_heat_pump`: it
beats the tuned built-in baseline on *every* KPI at once — a clean Pareto win (see `results/boptest.md`).

```python
from causaldyn_bench.boptest import BOPTestClient, baseline_controller, run_episode
kpis = run_episode(BOPTestClient("http://127.0.0.1:8000"), baseline_controller())  # needs a live service
```

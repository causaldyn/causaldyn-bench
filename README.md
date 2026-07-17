# causaldyn-bench

A benchmark for **causal, constrained, dynamical decision-making** — it scores *predictions and
decisions*, not just one-step error. The point is to measure methods on the axis each deserves, so a
gradient-boosted tree can win one-step prediction while a causal, constrained controller wins the
closed-loop decision. Built on [`causal-hybrid-control`](../causal-hybrid-control).

## The five tracks

| track | question | metric | expected winner |
|---|---|---|---|
| **A** one-step | `x_t, u_t → x_{t+1}` | RMSE | tree surrogate |
| **B** rollout | `x_t, u_{t:t+H} → x_{t+1:t+H}` | rollout RMSE | hybrid (physics anchors the horizon) |
| **C** counterfactual | `x_{t+1}(do(u))` under confounding | \|effect − truth\| | causal / Double ML |
| **D** control | `u_t = π(x_t)` under constraints | regret vs oracle | causal hybrid controller |
| **E** systems | control-solve latency | ms | (a compiled runtime, later) |

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

## Status

v0.0.1 scaffold: all five tracks run on the synthetic CHC systems (a damped oscillator with hidden cubic
physics for A/B/E, a confounded linear system for C, the CHC oracle-regret tasks plus the
**adaptive-CV-compute** task for D). A **BOPTEST** (HVAC control) client + control episode ship in
`causaldyn_bench.boptest`, gated on a running BOPTEST service (`BOPTEST_URL`); wiring a CHC hybrid-MPC
controller into BOPTEST is next.

```python
from causaldyn_bench.boptest import BOPTestClient, baseline_controller, run_episode
kpis = run_episode(BOPTestClient("http://127.0.0.1:5000"), baseline_controller())  # needs a live service
```

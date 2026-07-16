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

Design rule: to be honest about the win, **never** claim "best model" — claim the decision under a stated
budget. Track A is expected to go to the trees; the value is Tracks B–D.

## Run

```bash
uv sync --extra trees          # tree baselines for Tracks A/B (optional)
uv run python -m causaldyn_bench
uv run pytest                  # smoke tests
```

Without the `trees` extra the tree baselines are simply omitted; the hybrid/causal methods still run.

## Status

v0.0.1 scaffold: all five tracks run on the synthetic CHC systems (a damped oscillator with hidden cubic
physics for A/B/E, a confounded linear system for C, the CHC oracle-regret tasks for D). Real-data tasks
are next — the top targets are **BOPTEST** (HVAC control) and an **adaptive-CV-compute** benchmark.

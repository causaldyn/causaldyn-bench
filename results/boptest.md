# BOPTEST — real Track-D HVAC results (`bestest_hydronic_heat_pump`)

Live BOPTEST-Service via `causaldyn_bench.boptest` (Fedora / Podman), 1-day episodes, 30-min control
step. Reproduce with `causaldyn_bench.boptest_chc.compare_controllers` against a running service
(`results/boptest_compare.json` is the saved run below).

Identified thermal model (`identify_thermal_model`, slow-PRBS exploration, stability-constrained):

```
T_next = 0.980·T + 0.243·u + 5.76      (u = heat-pump modulation ∈ [0, 1])
```

| KPI | baseline (BOPTEST built-in) | CHC naive (setpoint-tracking) | **CHC forecast-MPC** |
|---|---:|---:|---:|
| `tdis_tot` (thermal discomfort, K·h) | 8.01 | 21.98 | **7.32** |
| `ener_tot` (energy) | 0.393 | 0.201 | **0.354** |
| `cost_tot` | 0.100 | 0.051 | **0.090** |
| `emis_tot` (emissions) | 0.066 | 0.034 | **0.059** |

**The forecast-driven comfort MPC beats the tuned baseline on every KPI at once** — less discomfort,
less energy, lower cost, fewer emissions. A clean Pareto win on a real emulator, not a synthetic task.

## How the honest win was earned (each step was a real failure fixed at its root)

The first controller (1-step certainty-equivalent) under-heated badly (discomfort 8 → 22). Chasing
that to its cause, not papering over it, is the whole story:

1. **Comfort-MPC ≡ naive, byte-for-byte, across three different models.** Identical KPIs under
   wildly different `(a, b, d)` meant the *controller* was never the variable. The models were the
   problem.
2. **I.i.d. per-step exploration collapses the DC gain.** A building low-passes i.i.d. modulation to
   its mean, so least-squares fits a steady-state gain far below the truth and every controller then
   believes the heat pump can't reach setpoint. Fix: a **slow PRBS** (hold each level for several
   steps) that excites the slow thermal mode — persistent excitation, not a controller tweak.
3. **PRBS then drives `a` to a unit-root (`a = 1.03`, unstable).** OLS on a short, highly
   autocorrelated slow series over-fits an unstable pole, and the MPC's multi-step rollout diverges.
   Fix: impose the **physics prior that a hydronic building is BIBO-stable** (`a ≤ 0.98`) and refit
   the gain/offset — exactly the "known structure + data" stance CHC advocates, in miniature.
4. **The overwrite works, but authority is genuinely limited.** Forcing `oveHeaPumY_u = 0` cools the
   zone (proving the overwrite path), but full power raises it only ~0.4 K per 30-min step. So the
   task *requires* anticipation: you cannot recover comfort reactively.
5. **The real failure was tracking the night setback, not the model.** The setpoint-tracking
   controllers followed `reaTSetHea_y` down at night and were then too weak to catch the morning
   occupancy ramp. Fix: an MPC that plans against the **`LowerSetp[1]` forecast** (the comfort bound
   `tdis` is actually scored on) and **pre-heats** ahead of the ramp. Horizon 16 (8 h look-ahead) and
   a comfort-dominated weight were what turned the corner.

## What this validates

The moat is a **real, method-agnostic control task with an external ground truth**, and CHC's stance
— physics-structured, stability-constrained identification plus forecast-aware constrained control —
produces a controller that dominates the reference on a standard emulator. The naive column is kept
deliberately: it is the cautionary tale that motivates every one of the fixes above.

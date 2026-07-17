# BOPTEST — real Track-D HVAC results (`bestest_hydronic_heat_pump`)

Live BOPTEST-Service via `causaldyn_bench.boptest` (Fedora / Podman), 1-day episodes, 30-min control step.

Identified thermal model from a short exploration episode (`causaldyn_bench.boptest_chc`):

```
T_next = 0.737·T + 0.070·u + 76.8      (u = heat-pump modulation ∈ [0, 1])
```

| KPI | baseline (BOPTEST built-in) | CHC (certainty-equivalent) |
|---|---:|---:|
| `tdis_tot` (thermal discomfort, K·h) | 8.01 | 21.98 |
| `ener_tot` (energy) | 0.393 | 0.201 |
| `cost_tot` | 0.100 | 0.051 |
| `emis_tot` (emissions) | 0.066 | 0.034 |
| `pele_tot` (peak electrical) | 0.019 | 0.019 |

## Honest read

The integration works end to end — a CHC controller runs in the loop on the real FMU and returns real
KPIs. This **first** controller (1-step certainty-equivalent on a crude 1st-order model, no weather
forecast) **cuts energy / cost / emissions ~50 % but roughly triples thermal discomfort** — it
under-heats. On BOPTEST's balanced objective the well-tuned baseline wins.

Beating the baseline needs a **comfort-weighted MPC** (not a myopic one-step policy), a better model
(outdoor-temperature forecast, higher order), and tuning of the comfort target — that is the next step.
What is settled today: the real, method-agnostic Track-D task is wired, reproducible, and reported
honestly. This is the step that turns the benchmark from a self-made demonstration into a real one.

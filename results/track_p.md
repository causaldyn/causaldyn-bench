# Track P — the channel-drift monitor on plants CHC did not write

300 runs per arm, each environment stepped through its own `step` (`causaldyn_bench.drift_calibration`, x64 True), with `A = 1000`. A null run is censored at 10000 decisions, a moved channel's at 4000. Alarm-rate intervals are Clopper-Pearson's.

## The gate

The type-I error under continuous monitoring: on each null, read after every decision, the share of runs alarmed by `H = 100` must not exceed `alpha = H / A = 0.1` beyond Monte Carlo error, the interval's lower end at most `alpha`.

| environment | plan | channel | actions clipped | alarmed by `H` | 95% interval | pass |
|---|---|---|---|---|---|---|
| Pendulum-v1 | inside | modelled | 0.1% | 0.003 | [0.000, 0.018] | yes |
| Pendulum-v1 | inside | on the edge | 0.1% | 0.010 | [0.002, 0.029] | yes |
| Pendulum-v1 | on the bound | modelled | 26.0% | 0.003 | [0.000, 0.018] | yes |
| Pendulum-v1 | on the bound | on the edge | 25.1% | 0.003 | [0.000, 0.018] | yes |
| MountainCarContinuous-v0 | inside | modelled | 0.1% | 0.003 | [0.000, 0.018] | yes |
| MountainCarContinuous-v0 | inside | on the edge | 0.1% | 0.010 | [0.002, 0.029] | yes |
| MountainCarContinuous-v0 | on the bound | modelled | 24.1% | 0.013 | [0.004, 0.034] | yes |
| MountainCarContinuous-v0 | on the bound | on the edge | 22.9% | 0.003 | [0.000, 0.018] | yes |

## Pendulum-v1

Actuator bound 2; the model's channel [0.0075000000000000015, 0.15000000000000002]. The environment clipped its state after 0 decisions in all.

| plan | channel | reading | actions clipped | run length / `A` | censored | caught | mean delay (median) |
|---|---|---|---|---|---|---|---|
| inside | modelled | draw | 0.1% | ≥ 2.52 | 0% | | |
| inside | on the edge | draw | 0.1% | ≥ 1.90 | 0% | | |
| inside | moved | draw | 0.1% | | | 1.000 | 101 (96) |
| on the bound | modelled | draw | 26.0% | ≥ 2.02 | 0% | | |
| on the bound | modelled | clipped zeroed | 26.0% | ≥ 9.96 | 99% | | |
| on the bound | on the edge | draw | 25.1% | ≥ 1.77 | 0% | | |
| on the bound | on the edge | clipped zeroed | 25.1% | ≥ 9.98 | 99% | | |
| on the bound | moved | draw | 18.1% | | | 1.000 | 223 (202) |
| on the bound | moved | clipped zeroed | 18.1% | | | 0.007 | 1912 (1912) |

## MountainCarContinuous-v0

Actuator bound 1; the model's channel [0.0015, 0.0015]. The environment clipped its state after 0 decisions in all.

| plan | channel | reading | actions clipped | run length / `A` | censored | caught | mean delay (median) |
|---|---|---|---|---|---|---|---|
| inside | modelled | draw | 0.1% | ≥ 2.57 | 0% | | |
| inside | on the edge | draw | 0.1% | ≥ 1.92 | 0% | | |
| inside | moved | draw | 0.1% | | | 1.000 | 96 (91) |
| on the bound | modelled | draw | 24.1% | ≥ 2.03 | 0% | | |
| on the bound | modelled | clipped zeroed | 24.1% | ≥ 9.96 | 99% | | |
| on the bound | on the edge | draw | 22.9% | ≥ 1.84 | 0% | | |
| on the bound | on the edge | clipped zeroed | 22.9% | ≥ 9.97 | 99% | | |
| on the bound | moved | draw | 13.0% | | | 1.000 | 204 (180) |
| on the bound | moved | clipped zeroed | 13.0% | | | 0.020 | 769 (34) |

Wall-clock per environment is recorded in `track_p.json` and quoted nowhere: it was not measured on a clean machine.

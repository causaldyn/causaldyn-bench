# Track O — off-policy interval coverage on plants CHC did not write

500 replicates of 4000 logged transitions after a 500-step burn-in, nominal 0.95, each environment stepped through its own `step` (`causaldyn_bench.ope_calibration`, x64 True). The truth is each plan's online average cost over 400000 steps. Coverage intervals are Clopper-Pearson's.

## The gate

Nominal design, fitted model. `model_error = 0` must be within two points of 0.95; the default `model_error = 1` must not fall below 0.93.

| environment | plan | method | `model_error` | coverage | 95% interval | pass |
|---|---|---|---|---|---|---|
| Pendulum-v1 | moderate | `mis` | 0 | 0.964 | [0.944, 0.979] | yes |
| Pendulum-v1 | moderate | `mis` | 1 | 1.000 | [0.993, 1.000] | yes |
| Pendulum-v1 | moderate | `dr` | 0 | 0.958 | [0.937, 0.974] | yes |
| Pendulum-v1 | moderate | `dr` | 1 | 1.000 | [0.993, 1.000] | yes |
| Pendulum-v1 | moderate | `fqe` | 0 | 0.954 | [0.932, 0.971] | yes |
| Pendulum-v1 | aggressive | `mis` | 0 | 0.930 | [0.904, 0.951] | yes |
| Pendulum-v1 | aggressive | `mis` | 1 | 1.000 | [0.993, 1.000] | yes |
| Pendulum-v1 | aggressive | `dr` | 0 | 0.944 | [0.920, 0.962] | yes |
| Pendulum-v1 | aggressive | `dr` | 1 | 1.000 | [0.993, 1.000] | yes |
| Pendulum-v1 | aggressive | `fqe` | 0 | 0.946 | [0.922, 0.964] | yes |
| MountainCarContinuous-v0 | moderate | `mis` | 0 | 0.930 | [0.904, 0.951] | yes |
| MountainCarContinuous-v0 | moderate | `mis` | 1 | 1.000 | [0.993, 1.000] | yes |
| MountainCarContinuous-v0 | moderate | `dr` | 0 | 0.950 | [0.927, 0.967] | yes |
| MountainCarContinuous-v0 | moderate | `dr` | 1 | 1.000 | [0.993, 1.000] | yes |
| MountainCarContinuous-v0 | moderate | `fqe` | 0 | 0.944 | [0.920, 0.962] | yes |
| MountainCarContinuous-v0 | aggressive | `mis` | 0 | 0.949 | [0.922, 0.969] | yes |
| MountainCarContinuous-v0 | aggressive | `mis` | 1 | 1.000 | [0.991, 1.000] | yes |
| MountainCarContinuous-v0 | aggressive | `dr` | 0 | 0.962 | [0.938, 0.978] | yes |
| MountainCarContinuous-v0 | aggressive | `dr` | 1 | 1.000 | [0.991, 1.000] | yes |
| MountainCarContinuous-v0 | aggressive | `fqe` | 0 | 0.946 | [0.922, 0.964] | yes |

## Pendulum-v1

Actuator bound 2, disturbance sd 0.3, logger velocity gain 0.3873. Steps clipped in the logs, mean / max over replicates: nominal 0.08% / 0.20%, stress 2.10% / 2.73%.

| plan | online truth | its SE | linearised | steps clipped |
|---|---|---|---|---|
| moderate | 0.146864 | 0.00035 | 0.146666 | 0.000% |
| aggressive | 0.0455943 | 0.0001 | 0.0456827 | 0.000% |

| design | plan | model | method | `model_error` | scored | coverage | above / below | mean error ± SE | half-width | model share | dof |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | moderate | fitted | `mis` | 0 | 500 | 0.964 | 0.016 / 0.020 | 0.00121 ± 0.00051 | 0.0243 | 0.20 | 12.3 |
| nominal | moderate | fitted | `mis` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.00119 ± 0.00051 | 0.0578 | 0.18 | 11.9 |
| nominal | moderate | fitted | `dr` | 0 | 500 | 0.958 | 0.004 / 0.038 | 0.000398 ± 0.00085 | 0.0462 | 0.20 | 7.9 |
| nominal | moderate | fitted | `dr` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.000461 ± 0.00088 | 0.0815 | 0.18 | 7.6 |
| nominal | moderate | fitted | `fqe` | 0 | 500 | 0.954 | 0.038 / 0.008 | 0.00133 ± 0.0009 | 0.0387 | 0.00 | — |
| nominal | moderate | linearisation | `mis` | 0 | 500 | 0.974 | 0.008 / 0.018 | 0.00123 ± 0.00048 | 0.0244 | 0.20 | 12.3 |
| nominal | moderate | linearisation | `mis` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.00122 ± 0.00048 | 0.0579 | 0.18 | 12.0 |
| nominal | moderate | linearisation | `dr` | 0 | 500 | 0.958 | 0.006 / 0.036 | 0.000295 ± 0.00085 | 0.0461 | 0.20 | 7.8 |
| nominal | moderate | linearisation | `dr` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.000346 ± 0.00088 | 0.0814 | 0.18 | 7.6 |
| nominal | moderate | linearisation | `fqe` | 0 | 500 | 0.954 | 0.038 / 0.008 | 0.00133 ± 0.0009 | 0.0387 | 0.00 | — |
| nominal | aggressive | fitted | `mis` | 0 | 500 | 0.930 | 0.042 / 0.028 | 0.00079 ± 0.00026 | 0.012 | 0.34 | 11.2 |
| nominal | aggressive | fitted | `mis` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.00079 ± 0.00026 | 0.0363 | 0.34 | 11.2 |
| nominal | aggressive | fitted | `dr` | 0 | 500 | 0.944 | 0.012 / 0.044 | 0.000198 ± 0.00034 | 0.0172 | 0.35 | 7.1 |
| nominal | aggressive | fitted | `dr` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.000198 ± 0.00034 | 0.0415 | 0.35 | 7.1 |
| nominal | aggressive | fitted | `fqe` | 0 | 500 | 0.946 | 0.046 / 0.008 | 0.000666 ± 0.00041 | 0.0177 | 0.00 | — |
| nominal | aggressive | linearisation | `mis` | 0 | 500 | 0.944 | 0.036 / 0.020 | 0.000807 ± 0.00026 | 0.012 | 0.34 | 11.1 |
| nominal | aggressive | linearisation | `mis` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.000807 ± 0.00026 | 0.0363 | 0.34 | 11.1 |
| nominal | aggressive | linearisation | `dr` | 0 | 500 | 0.950 | 0.012 / 0.038 | 0.000156 ± 0.00035 | 0.0172 | 0.35 | 7.0 |
| nominal | aggressive | linearisation | `dr` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.000156 ± 0.00035 | 0.0415 | 0.35 | 7.0 |
| nominal | aggressive | linearisation | `fqe` | 0 | 500 | 0.946 | 0.046 / 0.008 | 0.000666 ± 0.00041 | 0.0177 | 0.00 | — |
| stress | moderate | fitted | `mis` | 0 | 500 | 0.960 | 0.022 / 0.018 | 0.00138 ± 0.00092 | 0.0444 | 0.50 | 10.6 |
| stress | moderate | fitted | `mis` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.00138 ± 0.00092 | 0.191 | 0.50 | 10.6 |
| stress | moderate | fitted | `dr` | 0 | 500 | 0.966 | 0.004 / 0.030 | -0.00054 ± 0.001 | 0.055 | 0.50 | 7.4 |
| stress | moderate | fitted | `dr` | 1 | 500 | 1.000 | 0.000 / 0.000 | -0.00054 ± 0.001 | 0.202 | 0.50 | 7.4 |
| stress | moderate | fitted | `fqe` | 0 | 500 | 0.900 | 0.092 / 0.008 | 0.018 ± 0.0014 | 0.06 | 0.00 | — |
| stress | moderate | linearisation | `mis` | 0 | 500 | 0.960 | 0.026 / 0.014 | 0.00169 ± 0.00091 | 0.0447 | 0.50 | 10.7 |
| stress | moderate | linearisation | `mis` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.00169 ± 0.00091 | 0.196 | 0.50 | 10.7 |
| stress | moderate | linearisation | `dr` | 0 | 500 | 0.966 | 0.004 / 0.030 | -0.00177 ± 0.001 | 0.0542 | 0.51 | 7.5 |
| stress | moderate | linearisation | `dr` | 1 | 500 | 1.000 | 0.000 / 0.000 | -0.00177 ± 0.001 | 0.205 | 0.51 | 7.5 |
| stress | moderate | linearisation | `fqe` | 0 | 500 | 0.900 | 0.092 / 0.008 | 0.018 ± 0.0014 | 0.06 | 0.00 | — |

## MountainCarContinuous-v0

Actuator bound 1, disturbance sd 0.15, logger velocity gain 17.32. Steps clipped in the logs, mean / max over replicates: nominal 0.07% / 0.24%, stress 2.00% / 2.67%.

| plan | online truth | its SE | linearised | steps clipped |
|---|---|---|---|---|
| moderate | 0.107338 | 0.0003 | 0.106994 | 0.000% |
| aggressive | 0.0262085 | 6.2e-05 | 0.026136 | 0.000% |

| design | plan | model | method | `model_error` | scored | coverage | above / below | mean error ± SE | half-width | model share | dof |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | moderate | fitted | `mis` | 0 | 500 | 0.930 | 0.038 / 0.032 | 0.00153 ± 0.00061 | 0.0286 | 0.40 | 7.3 |
| nominal | moderate | fitted | `mis` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.00153 ± 0.00061 | 0.102 | 0.40 | 7.3 |
| nominal | moderate | fitted | `dr` | 0 | 500 | 0.950 | 0.006 / 0.044 | -0.000159 ± 0.00085 | 0.0432 | 0.41 | 5.7 |
| nominal | moderate | fitted | `dr` | 1 | 500 | 1.000 | 0.000 / 0.000 | -0.000159 ± 0.00085 | 0.117 | 0.41 | 5.7 |
| nominal | moderate | fitted | `fqe` | 0 | 500 | 0.944 | 0.044 / 0.012 | 0.00333 ± 0.00087 | 0.0359 | 0.00 | — |
| nominal | moderate | linearisation | `mis` | 0 | 500 | 0.938 | 0.038 / 0.024 | 0.00145 ± 0.0006 | 0.0286 | 0.40 | 7.3 |
| nominal | moderate | linearisation | `mis` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.00145 ± 0.0006 | 0.101 | 0.40 | 7.3 |
| nominal | moderate | linearisation | `dr` | 0 | 500 | 0.948 | 0.006 / 0.046 | -0.000138 ± 0.00085 | 0.0434 | 0.41 | 5.7 |
| nominal | moderate | linearisation | `dr` | 1 | 500 | 1.000 | 0.000 / 0.000 | -0.000138 ± 0.00085 | 0.116 | 0.41 | 5.7 |
| nominal | moderate | linearisation | `fqe` | 0 | 500 | 0.944 | 0.044 / 0.012 | 0.00333 ± 0.00087 | 0.0359 | 0.00 | — |
| nominal | aggressive | fitted | `mis` | 0 | 393 (107 refused) | 0.949 | 0.018 / 0.033 | 0.000602 ± 0.00045 | 0.0244 | 0.63 | 4.9 |
| nominal | aggressive | fitted | `mis` | 1 | 393 (107 refused) | 1.000 | 0.000 / 0.000 | 0.000602 ± 0.00045 | 0.0719 | 0.63 | 4.9 |
| nominal | aggressive | fitted | `dr` | 0 | 393 (107 refused) | 0.962 | 0.013 / 0.025 | 0.0004 ± 0.00028 | 0.0136 | 0.64 | 5.5 |
| nominal | aggressive | fitted | `dr` | 1 | 393 (107 refused) | 1.000 | 0.000 / 0.000 | 0.0004 ± 0.00028 | 0.0611 | 0.64 | 5.5 |
| nominal | aggressive | fitted | `fqe` | 0 | 500 | 0.946 | 0.040 / 0.014 | 0.0012 ± 0.00035 | 0.0146 | 0.00 | — |
| nominal | aggressive | linearisation | `mis` | 0 | 412 (88 refused) | 0.930 | 0.017 / 0.053 | 0.000545 ± 0.00049 | 0.0268 | 0.64 | 4.8 |
| nominal | aggressive | linearisation | `mis` | 1 | 412 (88 refused) | 1.000 | 0.000 / 0.000 | 0.000545 ± 0.00049 | 0.0752 | 0.64 | 4.8 |
| nominal | aggressive | linearisation | `dr` | 0 | 412 (88 refused) | 0.959 | 0.015 / 0.027 | 0.000405 ± 0.00028 | 0.0135 | 0.64 | 5.4 |
| nominal | aggressive | linearisation | `dr` | 1 | 412 (88 refused) | 1.000 | 0.000 / 0.000 | 0.000405 ± 0.00028 | 0.0618 | 0.64 | 5.4 |
| nominal | aggressive | linearisation | `fqe` | 0 | 500 | 0.946 | 0.040 / 0.014 | 0.0012 ± 0.00035 | 0.0146 | 0.00 | — |
| stress | moderate | fitted | `mis` | 0 | 500 | 0.946 | 0.020 / 0.034 | 0.00141 ± 0.0013 | 0.0609 | 0.69 | 6.9 |
| stress | moderate | fitted | `mis` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.00141 ± 0.0013 | 0.307 | 0.69 | 6.9 |
| stress | moderate | fitted | `dr` | 0 | 500 | 0.968 | 0.010 / 0.022 | 0.00135 ± 0.00094 | 0.0493 | 0.70 | 5.8 |
| stress | moderate | fitted | `dr` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.00135 ± 0.00094 | 0.296 | 0.70 | 5.8 |
| stress | moderate | fitted | `fqe` | 0 | 500 | 0.954 | 0.042 / 0.004 | 0.0089 ± 0.0013 | 0.0554 | 0.00 | — |
| stress | moderate | linearisation | `mis` | 0 | 500 | 0.948 | 0.018 / 0.034 | 0.00148 ± 0.0013 | 0.0613 | 0.69 | 6.9 |
| stress | moderate | linearisation | `mis` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.00148 ± 0.0013 | 0.312 | 0.69 | 6.9 |
| stress | moderate | linearisation | `dr` | 0 | 500 | 0.960 | 0.010 / 0.030 | 0.000146 ± 0.00093 | 0.0486 | 0.70 | 5.8 |
| stress | moderate | linearisation | `dr` | 1 | 500 | 1.000 | 0.000 / 0.000 | 0.000146 ± 0.00093 | 0.299 | 0.70 | 5.8 |
| stress | moderate | linearisation | `fqe` | 0 | 500 | 0.954 | 0.042 / 0.004 | 0.0089 ± 0.0013 | 0.0554 | 0.00 | — |

Wall-clock per environment is recorded in `track_o.json` and quoted nowhere: it was not measured on a clean machine.

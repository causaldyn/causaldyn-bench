# causaldyn-bench leaderboard

## A-onestep  (metric: rmse, lower is better)

| rank | method | value |
|---|---|---|
| 1 | hybrid-CHC **(best)** | 0.0017 |
| 2 | tree-surrogate | 0.0335 |
| 3 | dlm | 0.0447 |
| 4 | known-only | 0.0701 |

## B-rollout  (metric: rmse, lower is better)

| rank | method | value |
|---|---|---|
| 1 | hybrid-CHC **(best)** | 0.0260 |
| 2 | dlm | 0.3563 |
| 3 | tree-surrogate | 0.4652 |
| 4 | known-only | 0.4676 |

## C-effect  (metric: ate_error, lower is better)

| rank | method | value |
|---|---|---|
| 1 | double-ml-CHC **(best)** | 0.0015 |
| 2 | backdoor-CHC | 0.0015 |
| 3 | naive | 1.1998 |

## D-control  (metric: regret, lower is better)

| rank | method | value |
|---|---|---|
| 1 | pricing/predictive-all **(best)** | -0.0025 |
| 2 | pricing/causal-CHC | -0.0025 |
| 3 | pricing/oracle | 0.0000 |
| 4 | inventory/oracle | 0.0000 |
| 5 | support-shift/oracle | 0.0000 |
| 6 | inventory/causal-CHC | 0.0000 |
| 7 | inventory/predictive-all | 0.0000 |
| 8 | inventory/predictive-omitted | 1.0977 |
| 9 | support-shift/pessimistic | 2.4284 |
| 10 | support-shift/greedy | 7.4206 |
| 11 | pricing/predictive-omitted | 13735.4927 |

## D-planner  (metric: regret, lower is better)

| rank | method | value |
|---|---|---|
| 1 | plant/gradient **(best)** | 0.0000 |
| 2 | hybrid/gradient | 0.0016 |
| 3 | plant/cem | 0.0019 |
| 4 | hybrid/cem | 0.0030 |
| 5 | known_only/cem | 1.7177 |
| 6 | known_only/gradient | 1.7181 |

## D-adaptive-cv  (metric: regret, lower is better)

| rank | method | value |
|---|---|---|
| 1 | CHC-MPC **(best)** | 0.0000 |
| 2 | myopic | 166.0176 |
| 3 | uniform | 330.3924 |

## D-interference  (metric: regret, lower is better)

| rank | method | value |
|---|---|---|
| 1 | equilibrium-CHC **(best)** | 0.0000 |
| 2 | no-incentive | 1.1451 |
| 3 | naive-uplift | 1.9274 |

## H-marketplace  (metric: regret, lower is better)

| rank | method | value |
|---|---|---|
| 1 | equilibrium-CHC **(best)** | 0.1015 |
| 2 | equilibrium-CHC expected | 0.1031 |
| 3 | predictive-MOPO | 3.2436 |
| 4 | naive-causal | 3.4957 |

The market is `SharedStateMarket(seed=0)`, one of the 30 that `results/track_h_markets_v3.md` scores in float32 (`just track-h`), and the one of them on which both SUTVA planners leave their largest regret. Over the 30, equilibrium-CHC expected has less regret than both SUTVA planners on every one, and equilibrium-CHC on 29 of them; the regrets run equilibrium-CHC expected from -0.000 to 0.175 (median 0.008), equilibrium-CHC from 0.001 to 0.916 (median 0.147), naive-causal from 0.303 to 3.496 (median 1.273), predictive-MOPO from 1.093 to 3.244 (median 2.168). On 30 markets drawn under `JAX_ENABLE_X64=1` (`results/track_h_markets_v3_x64.md`) equilibrium-CHC expected has less regret than both SUTVA planners on every one, and equilibrium-CHC on every one.

## I-sensitivity  (metric: worst-case closed-loop cost, lower is better)

| rank | method | value |
|---|---|---|
| 1 | robust-CHC (Gamma=2.5) **(best)** | 4.9241 |
| 2 | robust-CHC (Gamma=6.0, over) | 6.7738 |
| 3 | robust-CHC (Gamma=1.3, under) | 14.7709 |
| 4 | certainty-equivalence | 18.9691 |

## F-structure  (metric: edge_f1, higher is better)

| rank | method | value |
|---|---|---|
| 1 | chc-discovery **(best)** | 0.9333 |
| 2 | naive-correlation | 0.2745 |

## F-payoff  (metric: ate_error, lower is better)

| rank | method | value |
|---|---|---|
| 1 | chc-discovery-adjusted **(best)** | 0.0010 |
| 2 | naive-unadjusted | 1.2276 |

## G-effect  (metric: irf_error, lower is better)

| rank | method | value |
|---|---|---|
| 1 | structured-toeplitz **(best)** | 0.0145 |
| 2 | local-projections | 0.0315 |
| 3 | naive-static | 0.6000 |

## G-payoff  (metric: track_error, lower is better)

| rank | method | value |
|---|---|---|
| 1 | chc-irf **(best)** | 0.0326 |
| 2 | one-step | 1.0049 |

## K-delay  (metric: |tau_hat - tau|, lower is better)

| rank | method | value |
|---|---|---|
| 1 | adjusted-LP-refined **(best)** | 0.0672 |
| 2 | adjusted-LP | 0.1000 |
| 3 | cross-correlation | 0.4000 |
| 4 | local-projection | 0.4000 |

## K-payoff  (metric: closed-loop regret, lower is better)

| rank | method | value |
|---|---|---|
| 1 | oracle **(best)** | 0.0000 |
| 2 | adjusted-LP-refined | 0.0067 |
| 3 | adjusted-LP | 0.0162 |
| 4 | cross-correlation | 0.6768 |
| 5 | local-projection | 0.6768 |
| 6 | delay-blind | 18953.5046 |

## N-fold-direct  (metric: MSE / random unit folds (2 clusters), lower is better)

| rank | method | value |
|---|---|---|
| 1 | random rows **(best)** | 0.8610 |
| 2 | designed folds (chc) | 0.9059 |
| 3 | random units | 1.0000 |
| 4 | contiguous folds | 1.3702 |
| 5 | neighbour exclusion | 1.6661 |

## N-fold-spillover  (metric: MSE / random unit folds (2 clusters), lower is better)

| rank | method | value |
|---|---|---|
| 1 | random rows **(best)** | 0.9755 |
| 2 | designed folds (chc) | 0.9949 |
| 3 | random units | 1.0000 |
| 4 | contiguous folds | 1.3575 |
| 5 | neighbour exclusion | 1.6124 |

## M-allocation  (metric: lift, higher is better)

| rank | method | value |
|---|---|---|
| 1 | myopic-greedy **(best)** | 44.1599 |
| 2 | CHC-adjusted | 42.9841 |
| 3 | equal-split | 41.1925 |
| 4 | naive-MMM | 37.6745 |

## E-systems  (metric: solve_ms, lower is better)

| rank | method | value |
|---|---|---|
| 1 | known-only **(best)** | 0.6366 |
| 2 | hybrid-CHC | 5.9378 |


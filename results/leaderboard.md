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
| 1 | pricing/causal-CHC **(best)** | -0.0025 |
| 2 | pricing/oracle | 0.0000 |
| 3 | inventory/oracle | 0.0000 |
| 4 | support-shift/oracle | 0.0000 |
| 5 | inventory/causal-CHC | 0.0000 |
| 6 | inventory/predictive | 1.0977 |
| 7 | support-shift/pessimistic | 2.4169 |
| 8 | support-shift/greedy | 6.7992 |
| 9 | pricing/predictive | 13735.4927 |

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
| 1 | equilibrium-CHC **(best)** | 0.0483 |
| 2 | predictive-MOPO | 3.2322 |
| 3 | naive-causal | 3.4844 |

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

## E-systems  (metric: solve_ms, lower is better)

| rank | method | value |
|---|---|---|
| 1 | known-only **(best)** | 0.6684 |
| 2 | hybrid-CHC | 5.1748 |


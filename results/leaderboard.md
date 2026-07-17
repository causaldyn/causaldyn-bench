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
| 7 | support-shift/pessimistic | 2.4178 |
| 8 | support-shift/greedy | 5.5574 |
| 9 | pricing/predictive | 13735.4927 |

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

## E-systems  (metric: solve_ms, lower is better)

| rank | method | value |
|---|---|---|
| 1 | known-only **(best)** | 10.9196 |
| 2 | hybrid-CHC | 19.3016 |


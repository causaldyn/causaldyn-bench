# Track H over 30 markets

`SharedStateMarket(seed=s)` for 30 seeds, each scored on 400 logged blocks drawn from `jax.random.key(s + 1)`, in float32, JAX's default (`causaldyn_bench.marketplace`, `just track-h`). The other precision is `track_h_markets_x64.md`: a seed draws another market there, so the two runs are 60 markets. The leaderboard scores seed 0, one of these: equilibrium-CHC's regret on it reads 0.0869, naive-causal's 3.4870 and predictive-MOPO's 3.2349.

A planner's share is its lift over doing nothing as a share of the oracle's (`oracle_allocation(steps=2500)`), its regret the oracle's completions less its own; the oracle's ascent is local, so a plan can pass it. The naive-causal planner's interference over-count is the lift its per-zone response predicts for its allocation less the lift the market realises.

The confounding is read against two reruns of the logged blocks, which no log gives: each block keeps its demand, the attractiveness that moves with it and its completions noise, and its drivers settle anew under other incentives. In the *randomised twin* each block takes another block's incentives (`numpy.random.default_rng(s).permutation`), which no longer follow its demand; the naive-causal regression on it is the reference, and a response's bias is its mean over the zones less the reference's. In the *complete adjustment* a zone's incentive chases its own logged demand and, in every other zone, the permuted block's, so its own demand is its only confounder and adjusting for it is complete. The logging policy splits one budget by a softmax of every zone's demand, so the per-zone adjustment cannot block the other zones' demand, which moves a zone's incentive and draws its drivers: the naive-causal response's excess over the complete adjustment is that confounding. Equilibrium-CHC's response, adjusted for each zone's own demand and the block's total incentive, is read against its own regression on the twin: at a fixed total and own demand a zone's incentive moves only with the other zones' demand, so holding the total takes out the randomised budget and keeps the confounding. The planner reads that response only through its pessimism tilt. A response's correlation with the zones' demand is no test of de-confounding: the randomised twin's response follows demand too.

| reading | min | 25% | median | 75% | max | mean | SD |
|---|---|---|---|---|---|---|---|
| equilibrium-CHC's share of the oracle's lift | 0.741 | 0.854 | 0.925 | 0.958 | 0.999 | 0.904 | 0.069 |
| naive-causal's share of the oracle's lift | -0.437 | -0.002 | 0.298 | 0.569 | 0.843 | 0.267 | 0.368 |
| predictive-MOPO's share of the oracle's lift | -0.944 | -0.131 | 0.036 | 0.172 | 0.363 | -0.038 | 0.295 |
| equilibrium-CHC's regret | 0.001 | 0.089 | 0.148 | 0.235 | 0.727 | 0.204 | 0.175 |
| naive-causal's regret | 0.303 | 0.790 | 1.256 | 2.087 | 3.487 | 1.498 | 0.845 |
| predictive-MOPO's regret | 1.093 | 1.627 | 2.126 | 2.385 | 3.235 | 2.046 | 0.503 |
| predictive response less the randomised twin's | 0.238 | 0.397 | 0.446 | 0.526 | 0.643 | 0.454 | 0.102 |
| naive-causal response less the randomised twin's | -0.028 | 0.107 | 0.139 | 0.180 | 0.318 | 0.146 | 0.072 |
| naive-causal bias as a share of the predictive one | -0.055 | 0.216 | 0.352 | 0.419 | 0.634 | 0.333 | 0.155 |
| naive-causal response less the completely adjusted one | 0.129 | 0.173 | 0.223 | 0.263 | 0.384 | 0.223 | 0.059 |
| equilibrium-CHC's response less its own on the randomised twin | 0.200 | 0.308 | 0.367 | 0.421 | 0.568 | 0.370 | 0.088 |
| randomised twin's response, correlation with demand | 0.493 | 0.694 | 0.766 | 0.835 | 0.888 | 0.746 | 0.109 |
| naive-causal response, correlation with demand | -0.226 | 0.053 | 0.201 | 0.351 | 0.499 | 0.187 | 0.187 |
| naive-causal interference over-count | 1.487 | 2.286 | 2.861 | 3.222 | 4.326 | 2.762 | 0.714 |
| oracle's lift over doing nothing | 0.639 | 1.751 | 2.157 | 2.650 | 3.303 | 2.103 | 0.663 |

- equilibrium-CHC has the least regret on 30 of the 30 markets.
- The predictive response exceeds the randomised twin's on 30. The naive-causal response exceeds it on 29, by less than the predictive one on 30, and exceeds the complete adjustment on 30. Equilibrium-CHC's response exceeds its own on the twin by more than the naive-causal one does on 30.
- The naive-causal planner predicts more lift than it realises on 30, and realises less than doing nothing on 8; the predictive planner on 13.

`tests/test_marketplace.py` asserts on seeds 0-2 only what every market of the two runs supports: the orderings and signs above that hold on all of them, and bounds at the markets' mean less or plus three SDs, at whichever precision reads wider, rounded outward to two decimals.

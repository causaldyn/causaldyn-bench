# Track H over 30 markets, version 2

`SharedStateMarket(seed=s)` for 30 seeds, each scored on 400 logged blocks drawn from `jax.random.key(s + 1)`, in float32, JAX's default, on causal-hybrid-control 0.22.0 (`causaldyn_bench.marketplace`, `just track-h`). The other precision is `track_h_markets_v2_x64.md`: a seed draws another market there, so the two runs are 60 markets. The leaderboard scores seed 0, one of these: equilibrium-CHC's regret on it reads 0.1015, naive-causal's 3.4957 and predictive-MOPO's 3.2436.

A planner's share is its lift over doing nothing as a share of the oracle's, its regret the oracle's completions less its own. The oracle is the exact optimum (`SharedStateMarket.oracle_allocation()`, `chc.games.exact_congestion_allocation`), so no plan passes it by more than the equilibrium solve's tolerance. The naive-causal planner's interference over-count is the lift its per-zone response predicts for its allocation less the lift the market realises.

Version 1, `track_h_markets.md`, scored the same markets on causal-hybrid-control 0.21.0 against a 2500-step ascent of the oracle's and through an equilibrium fit that stopped short of convergence. The ascent fell short of the exact optimum on 13 of the 30 float32 markets, by up to 0.303 rides, and on 17 of the 30 float64 ones, by up to 0.215. Version 1 is reproducible from this repository at `cd43c6d` beside the library at `v0.21.0`.

The confounding is read against two reruns of the logged blocks, which no log gives: each block keeps its demand, the attractiveness that moves with it and its completions noise, and its drivers settle anew under other incentives. In the *randomised twin* each block takes another block's incentives (`numpy.random.default_rng(s).permutation`), which no longer follow its demand; the naive-causal regression on it is the reference, and a response's bias is its mean over the zones less the reference's. In the *complete adjustment* a zone's incentive chases its own logged demand and, in every other zone, the permuted block's, so its own demand is its only confounder and adjusting for it is complete. The logging policy splits one budget by a softmax of every zone's demand, so the per-zone adjustment cannot block the other zones' demand, which moves a zone's incentive and draws its drivers: the naive-causal response's excess over the complete adjustment is that confounding. Equilibrium-CHC's response, adjusted for each zone's own demand and the block's total incentive, is read against its own regression on the twin: at a fixed total and own demand a zone's incentive moves only with the other zones' demand, so holding the total takes out the randomised budget and keeps the confounding. The planner reads that response only through its pessimism tilt. A response's correlation with the zones' demand is no test of de-confounding: the randomised twin's response follows demand too.

| reading | min | 25% | median | 75% | max | mean | SD |
|---|---|---|---|---|---|---|---|
| equilibrium-CHC's share of the oracle's lift | 0.694 | 0.880 | 0.923 | 0.951 | 0.999 | 0.907 | 0.076 |
| naive-causal's share of the oracle's lift | -0.435 | -0.002 | 0.283 | 0.556 | 0.805 | 0.261 | 0.363 |
| predictive-MOPO's share of the oracle's lift | -0.944 | -0.131 | 0.035 | 0.162 | 0.347 | -0.040 | 0.293 |
| equilibrium-CHC's regret | 0.001 | 0.102 | 0.147 | 0.244 | 0.916 | 0.209 | 0.215 |
| naive-causal's regret | 0.303 | 0.818 | 1.273 | 2.203 | 3.496 | 1.535 | 0.848 |
| predictive-MOPO's regret | 1.093 | 1.664 | 2.168 | 2.394 | 3.244 | 2.083 | 0.525 |
| predictive response less the randomised twin's | 0.238 | 0.397 | 0.446 | 0.526 | 0.643 | 0.454 | 0.102 |
| naive-causal response less the randomised twin's | -0.028 | 0.107 | 0.139 | 0.180 | 0.318 | 0.146 | 0.072 |
| naive-causal bias as a share of the predictive one | -0.055 | 0.216 | 0.352 | 0.419 | 0.634 | 0.333 | 0.155 |
| naive-causal response less the completely adjusted one | 0.129 | 0.173 | 0.223 | 0.263 | 0.384 | 0.223 | 0.059 |
| equilibrium-CHC's response less its own on the randomised twin | 0.200 | 0.308 | 0.367 | 0.421 | 0.568 | 0.370 | 0.088 |
| randomised twin's response, correlation with demand | 0.493 | 0.694 | 0.766 | 0.835 | 0.888 | 0.746 | 0.109 |
| naive-causal response, correlation with demand | -0.226 | 0.053 | 0.201 | 0.351 | 0.499 | 0.187 | 0.187 |
| naive-causal interference over-count | 1.487 | 2.286 | 2.861 | 3.222 | 4.326 | 2.762 | 0.714 |
| oracle's lift over doing nothing | 0.639 | 1.751 | 2.163 | 2.710 | 3.458 | 2.140 | 0.701 |

- equilibrium-CHC has the least regret on 29 of the 30 markets.
- The predictive response exceeds the randomised twin's on 30. The naive-causal response exceeds it on 29, by less than the predictive one on 30, and exceeds the complete adjustment on 30. Equilibrium-CHC's response exceeds its own on the twin by more than the naive-causal one does on 30.
- The naive-causal planner predicts more lift than it realises on 30, and realises less than doing nothing on 8; the predictive planner on 13.

`tests/test_marketplace.py` asserts on seeds 0-2 the orderings and signs above that hold on every market of the two runs, and equilibrium-CHC's least regret, which holds on the markets counted above, and bounds at the markets' mean less or plus three SDs, at whichever precision reads wider, rounded outward to two decimals.

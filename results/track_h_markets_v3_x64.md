# Track H over 30 markets, version 3

`SharedStateMarket(seed=s)` for 30 seeds, each scored on 400 logged blocks drawn from `jax.random.key(s + 1)`, in float64, under `JAX_ENABLE_X64=1`, on causal-hybrid-control 0.23.0 (`causaldyn_bench.marketplace`, `just track-h`). The other precision is `track_h_markets_v3.md`: a seed draws another market there, so the two runs are 60 markets. The leaderboard, in float32, scores the other run's seed 0.

A planner's share is its lift over doing nothing as a share of the oracle's, its regret the oracle's completions less its own. The oracle is the exact optimum (`SharedStateMarket.oracle_allocation()`, `chc.games.exact_congestion_allocation`), so no plan passes it by more than the equilibrium solve's tolerance. The naive-causal planner's interference over-count is the lift its per-zone response predicts for its allocation less the lift the market realises.

The two equilibrium-CHC planners plan through the same equilibrium, fitted to the logs. Equilibrium-CHC maximises the fit's completions plus the shared-state response's pessimism tilt (`pessimistic_equilibrium_allocation`'s default, `objective="tilted"`), in a 600-step ascent from the even split. Equilibrium-CHC expected (`objective="expected"`) maximises the fit's expected completions under its own uncertainty, the fit's sandwich covariance and its planning demand's, ascended from the even split and from the fitted game's exact optimum; it reads no response.

Version 2, `track_h_markets_v2_x64.md`, scored the same markets on causal-hybrid-control 0.22.0 against the exact oracle, without the expected planner, which that version did not have; every reading it shares with this version is computed the same way. Version 2 is reproducible from this repository at `c2dce11` beside the library at `v0.22.0`.

Version 1, `track_h_markets_x64.md`, scored the same markets on causal-hybrid-control 0.21.0 against a 2500-step ascent of the oracle's and through an equilibrium fit that stopped short of convergence. The ascent fell short of the exact optimum on 13 of the 30 float32 markets, by up to 0.303 rides, and on 17 of the 30 float64 ones, by up to 0.215. Version 1 is reproducible from this repository at `cd43c6d` beside the library at `v0.21.0`.

The confounding is read against two reruns of the logged blocks, which no log gives: each block keeps its demand, the attractiveness that moves with it and its completions noise, and its drivers settle anew under other incentives. In the *randomised twin* each block takes another block's incentives (`numpy.random.default_rng(s).permutation`), which no longer follow its demand; the naive-causal regression on it is the reference, and a response's bias is its mean over the zones less the reference's. In the *complete adjustment* a zone's incentive chases its own logged demand and, in every other zone, the permuted block's, so its own demand is its only confounder and adjusting for it is complete. The logging policy splits one budget by a softmax of every zone's demand, so the per-zone adjustment cannot block the other zones' demand, which moves a zone's incentive and draws its drivers: the naive-causal response's excess over the complete adjustment is that confounding. Equilibrium-CHC's response, adjusted for each zone's own demand and the block's total incentive, is read against its own regression on the twin: at a fixed total and own demand a zone's incentive moves only with the other zones' demand, so holding the total takes out the randomised budget and keeps the confounding. The planner reads that response only through its pessimism tilt. A response's correlation with the zones' demand is no test of de-confounding: the randomised twin's response follows demand too.

| reading | min | 25% | median | 75% | max | mean | SD |
|---|---|---|---|---|---|---|---|
| equilibrium-CHC expected's share of the oracle's lift | 0.920 | 0.951 | 0.971 | 1.000 | 1.000 | 0.971 | 0.025 |
| equilibrium-CHC's share of the oracle's lift | 0.754 | 0.835 | 0.918 | 0.953 | 1.000 | 0.900 | 0.074 |
| naive-causal's share of the oracle's lift | -0.983 | 0.071 | 0.354 | 0.503 | 0.792 | 0.228 | 0.392 |
| predictive-MOPO's share of the oracle's lift | -0.378 | -0.093 | 0.003 | 0.148 | 0.300 | 0.019 | 0.180 |
| equilibrium-CHC expected's regret | -0.000 | 0.000 | 0.071 | 0.124 | 0.231 | 0.079 | 0.072 |
| equilibrium-CHC's regret | 0.000 | 0.091 | 0.177 | 0.420 | 0.701 | 0.255 | 0.206 |
| naive-causal's regret | 0.492 | 1.246 | 1.607 | 2.354 | 2.786 | 1.717 | 0.682 |
| predictive-MOPO's regret | 0.883 | 1.722 | 2.393 | 2.768 | 4.000 | 2.326 | 0.767 |
| predictive response less the randomised twin's | 0.301 | 0.399 | 0.445 | 0.482 | 0.542 | 0.438 | 0.062 |
| naive-causal response less the randomised twin's | 0.043 | 0.094 | 0.150 | 0.195 | 0.327 | 0.156 | 0.081 |
| naive-causal bias as a share of the predictive one | 0.090 | 0.242 | 0.367 | 0.458 | 0.724 | 0.359 | 0.173 |
| naive-causal response less the completely adjusted one | 0.129 | 0.173 | 0.194 | 0.235 | 0.301 | 0.204 | 0.042 |
| equilibrium-CHC's response less its own on the randomised twin | 0.231 | 0.294 | 0.368 | 0.435 | 0.586 | 0.376 | 0.097 |
| randomised twin's response, correlation with demand | 0.391 | 0.632 | 0.723 | 0.809 | 0.871 | 0.697 | 0.138 |
| naive-causal response, correlation with demand | -0.411 | 0.092 | 0.188 | 0.325 | 0.415 | 0.170 | 0.184 |
| naive-causal interference over-count | 1.169 | 2.139 | 2.618 | 3.107 | 4.525 | 2.692 | 0.766 |
| oracle's lift over doing nothing | 0.971 | 2.065 | 2.310 | 2.874 | 3.417 | 2.383 | 0.687 |

- Against the two SUTVA planners, equilibrium-CHC expected has the least regret on 30 of the 30 markets and equilibrium-CHC on 30. Equilibrium-CHC expected leaves less regret than equilibrium-CHC on 27, more on 1 and the same, to within 0.0005, on the rest.
- The predictive response exceeds the randomised twin's on 30. The naive-causal response exceeds it on 30, by less than the predictive one on 30, and exceeds the complete adjustment on 30. Equilibrium-CHC's response exceeds its own on the twin by more than the naive-causal one does on 30.
- The naive-causal planner predicts more lift than it realises on 30, and realises less than doing nothing on 7; the predictive planner on 15.

`tests/test_marketplace.py` asserts on seeds 0-2 the orderings and signs above that hold on every market of the two runs, and the equilibrium-CHC planners' least regret, which holds on the markets counted above, and bounds at the markets' mean less or plus three SDs, at whichever precision reads wider, rounded outward to two decimals.

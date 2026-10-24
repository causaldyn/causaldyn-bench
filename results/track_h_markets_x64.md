# Track H over 30 markets

`SharedStateMarket(seed=s)` for 30 seeds, each scored on 400 logged blocks drawn from `jax.random.key(s + 1)`, in float64, under `JAX_ENABLE_X64=1` (`causaldyn_bench.marketplace`, `just track-h`). The other precision is `track_h_markets.md`: a seed draws another market there, so the two runs are 60 markets. The leaderboard, in float32, scores the other run's seed 0.

A planner's share is its lift over doing nothing as a share of the oracle's (`oracle_allocation(steps=2500)`), its regret the oracle's completions less its own; the oracle's ascent is local, so a plan can pass it. The naive-causal planner's interference over-count is the lift its per-zone response predicts for its allocation less the lift the market realises.

The confounding is read against two reruns of the logged blocks, which no log gives: each block keeps its demand, the attractiveness that moves with it and its completions noise, and its drivers settle anew under other incentives. In the *randomised twin* each block takes another block's incentives (`numpy.random.default_rng(s).permutation`), which no longer follow its demand; the naive-causal regression on it is the reference, and a response's bias is its mean over the zones less the reference's. In the *complete adjustment* a zone's incentive chases its own logged demand and, in every other zone, the permuted block's, so its own demand is its only confounder and adjusting for it is complete. The logging policy splits one budget by a softmax of every zone's demand, so the per-zone adjustment cannot block the other zones' demand, which moves a zone's incentive and draws its drivers: the naive-causal response's excess over the complete adjustment is that confounding. A response's correlation with the zones' demand is no test of de-confounding: the randomised twin's response follows demand too.

| reading | min | 25% | median | 75% | max | mean | SD |
|---|---|---|---|---|---|---|---|
| equilibrium-CHC's share of the oracle's lift | 0.770 | 0.877 | 0.917 | 0.962 | 1.005 | 0.917 | 0.063 |
| naive-causal's share of the oracle's lift | -0.983 | 0.073 | 0.356 | 0.512 | 0.792 | 0.235 | 0.395 |
| predictive-MOPO's share of the oracle's lift | -0.378 | -0.097 | 0.003 | 0.155 | 0.300 | 0.020 | 0.182 |
| equilibrium-CHC's regret | -0.015 | 0.096 | 0.146 | 0.292 | 0.568 | 0.195 | 0.157 |
| naive-causal's regret | 0.492 | 1.183 | 1.569 | 2.266 | 2.749 | 1.661 | 0.671 |
| predictive-MOPO's regret | 0.883 | 1.722 | 2.342 | 2.717 | 3.962 | 2.271 | 0.737 |
| predictive response less the randomised twin's | 0.301 | 0.399 | 0.445 | 0.482 | 0.542 | 0.438 | 0.062 |
| naive-causal response less the randomised twin's | 0.043 | 0.094 | 0.150 | 0.195 | 0.327 | 0.156 | 0.081 |
| naive-causal bias as a share of the predictive one | 0.090 | 0.242 | 0.367 | 0.458 | 0.724 | 0.359 | 0.173 |
| naive-causal response less the completely adjusted one | 0.129 | 0.173 | 0.194 | 0.235 | 0.301 | 0.204 | 0.042 |
| randomised twin's response, correlation with demand | 0.391 | 0.632 | 0.723 | 0.809 | 0.871 | 0.697 | 0.138 |
| naive-causal response, correlation with demand | -0.411 | 0.092 | 0.188 | 0.325 | 0.415 | 0.170 | 0.184 |
| naive-causal interference over-count | 1.169 | 2.139 | 2.618 | 3.107 | 4.525 | 2.692 | 0.766 |
| oracle's lift over doing nothing | 0.971 | 1.997 | 2.274 | 2.784 | 3.380 | 2.328 | 0.649 |

- equilibrium-CHC has the least regret on 30 of the 30 markets.
- The predictive response exceeds the randomised twin's on 30. The naive-causal response exceeds it on 30, by less than the predictive one on 30, and exceeds the complete adjustment on 30.
- The naive-causal planner predicts more lift than it realises on 30, and realises less than doing nothing on 7; the predictive planner on 15.

`tests/test_marketplace.py` asserts on seeds 0-2 only what every market of the two runs supports: the orderings and signs above that hold on all of them, and bounds at the markets' mean less or plus three SDs, at whichever precision reads wider, rounded outward to two decimals.

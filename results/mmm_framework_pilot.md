# Family 21: mmm-framework's answer keys, pilot (float64)

The two data sets mmm-framework 1.5.0 ships with its wheel, read as data: a rival's worlds, written to test its own model. Each arm's reading of each channel's contribution over the file's weeks and its ROAS, against what the tool's truth file states, with the relative error, the reading over the stated less 1, in brackets.

## national: realistic

What it breaks, by the truth file: (realistic mix: confounding + many controls + weak/noisy media). The tool's model can represent it: yes, by the same file. The bench's arm reads the controls category_demand, distribution, price, competitor_promo, weather, holiday.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 6487.8 | 4707.0 [-0.274] | 5148.3 [-0.206] | 3.949 | 2.415 [-0.388] | 2.642 [-0.331] (1.151, 4.760) | 0.8430 |
| Search | 1728.0 | 2871.8 [+0.662] | 2286.1 [+0.323] | 0.337 | 0.507 [+0.504] | 0.403 [+0.197] (0.134, 0.771) | 0.9048 |
| Social | 2756.1 | 2834.1 [+0.028] | 2564.4 [-0.070] | 3.193 | 2.729 [-0.145] | 2.470 [-0.227] (0.710, 5.585) | 0.8312 |
| Display | 2151.0 | 2525.6 [+0.174] | 2555.9 [+0.188] | 0.765 | 0.794 [+0.039] | 0.804 [+0.051] (0.222, 1.915) | 0.8846 |
| Video | 5113.0 | 5302.3 [+0.037] | 4589.9 [-0.102] | 3.581 | 2.978 [-0.169] | 2.578 [-0.280] (1.175, 4.942) | 0.8018 |
| Radio | 1829.4 | 1633.5 [-0.107] | 2256.4 [+0.233] | 0.958 | 0.734 [-0.234] | 1.014 [+0.059] (0.022, 2.859) | 0.8582 |
| Print | 1151.0 | 2545.4 [+1.212] | 2392.9 [+1.079] | 0.946 | 1.767 [+0.867] | 1.661 [+0.755] (0.081, 4.544) | 0.8441 |

observational: the relative error's size averaged over the channels, 0.356 on the contribution and 0.335 on the ROAS.
mmm-framework: the relative error's size averaged over the channels, 0.315 on the contribution and 0.271 on the ROAS; its interval, in parentheses, holds the stated ROAS of 7 of 7 channels.

## geo: geo_heterogeneous

What it breaks, by the truth file: global media coefficients (per-geo effectiveness homogeneity). The tool's model can represent it: no, by the same file. The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 11470.6 | 11323.9 [-0.013] | 7046.6 [-0.386] | 0.218 | 0.215 [-0.013] | 0.134 [-0.386] (0.020, 0.332) | 1.0000 |
| Search | 7517.7 | 4628.0 [-0.384] | 9029.6 [+0.201] | 2.725 | 1.677 [-0.384] | 3.273 [+0.201] (0.849, 7.303) | 1.0000 |
| Social | 7249.3 | 19791.2 [+1.730] | 9029.9 [+0.246] | 2.705 | 7.386 [+1.730] | 3.370 [+0.246] (1.142, 7.300) | 1.0000 |
| Display | 2355.1 | 3129.5 [+0.329] | 4642.3 [+0.971] | 0.132 | 0.175 [+0.329] | 0.260 [+0.971] (0.008, 0.754) | 1.0000 |

observational: the relative error's size averaged over the channels, 0.614 on the contribution and 0.614 on the ROAS.
mmm-framework: the relative error's size averaged over the channels, 0.451 on the contribution and 0.451 on the ROAS; its interval, in parentheses, holds the stated ROAS of 4 of 4 channels.

The last column is each channel's stated contribution over its stated ROAS, as a share of its spend in the file: 1 where the stated ROAS is read on the file's spend. On the national set it is not, and the spend it is read on is not in the files, so a ROAS read on the file's spend, as both arms read it, is scored against a number on another denominator; the contribution needs none.

## The tool's own fit

mmm-framework fitted as its README's Python example fits a model (`scripts/mmm_framework_arm.py`, in the environment `scripts/mmm_framework_arm.txt` pins): `BayesianMMM` on the file through `create_simple_mff_config` and `load_mff`, a linear trend, NumPyro's NUTS at the example's chains, draws and tuning, read off `MMMAnalyzer.compute_channel_roi`. Its interval is that table's HDI of the contribution over the channel's spend, at the level the analyser uses, which the table does not name. It reads the same controls as the bench's arm. The seed does not fix its numbers: four runs of the script on one machine moved each channel's contribution by at most 0.5 % of its mean.

national: 4 chains of 1000 tuning and 2000 kept draws, seed 42.
- UserWarning: `init='adapt_diag'` is ignored by `nuts_sampler='numpyro'`; the external sampler uses its own initialization. (once)
- RuntimeWarning: invalid value encountered in scalar divide (626 times)
- No ConvergenceWarning, which the tool raises where its own check of a fit fails.

geo: 4 chains of 1000 tuning and 2000 kept draws, seed 42.
- UserWarning: `init='adapt_diag'` is ignored by `nuts_sampler='numpyro'`; the external sampler uses its own initialization. (once)
- RuntimeWarning: invalid value encountered in scalar divide (2928 times)
- No ConvergenceWarning, which the tool raises where its own check of a fit fails.

## Arms without a cell

- status quo: a plan, and the truth states no curve to score a plan on.
- equal split: a plan, and the truth states no curve to score a plan on.
- CHC: reads each channel from its lift tests, and the files hold none.
- myopic: reads each channel from its lift tests, and the files hold none.
- The worlds the tool's generator draws are read apart: `results/mmm_framework_markets.md`.

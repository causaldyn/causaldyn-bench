# Track L -- DCBO's three dynamic SCMs

Regret of the intervention sequence against the best intervention per step, each step scored against the best response to the arm's own history, on the noise-free SCM (`causaldyn_bench.dcbo_scm`). Every arm reads one observational log per seed: N = 10 series of T = 3. Seeds 0..9 (10); 95% percentile intervals from 10000 paired bootstrap resamples of seeds. DCBO reference at commit `85a9bdf`, seed s being its replicate s on log s; CHC fitted at float64.

## Total regret over the T = 3 steps, mean [95%]

| arm | stat | ind | nonstat |
|---|---|---|---|
| oracle | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| oracle (reference grid) | 0.0000 [0.0000, 0.0000] | 0.0004 [0.0004, 0.0004] | 0.0304 [0.0304, 0.0304] |
| DCBO | 0.4975 [0.3202, 0.6740] | 2.5448 [2.1229, 2.9033] | 1.3945 [0.8323, 2.0640] |
| CBO | 0.3946 [0.2726, 0.5425] | 2.4026 [2.0546, 2.8101] | 1.5363 [1.2704, 1.8372] |
| ABO | 0.0635 [0.0285, 0.1034] | 1.1738 [0.8297, 1.5019] | 4.1100 [3.3372, 4.9551] |
| BO | 0.2678 [0.1698, 0.3694] | 1.9343 [1.5176, 2.3597] | 4.1599 [3.3472, 5.0685] |
| CHC-prescribe | 0.4216 [0.1807, 0.6625] | 5.8759 [5.7074, 5.9729] | 6.4764 [5.2965, 8.2159] |

## Regret per step t = 0, 1, 2, mean over seeds

| arm | stat | ind | nonstat |
|---|---|---|---|
| oracle | 0.0000, 0.0000, 0.0000 | 0.0000, 0.0000, 0.0000 | 0.0000, 0.0000, 0.0000 |
| oracle (reference grid) | 0.0000, 0.0000, 0.0000 | 0.0001, 0.0001, 0.0001 | 0.0000, 0.0001, 0.0303 |
| DCBO | 0.1548, 0.1679, 0.1747 | 1.0156, 0.6112, 0.9180 | 0.2658, 0.5062, 0.6224 |
| CBO | 0.1547, 0.1388, 0.1010 | 1.0169, 0.7283, 0.6573 | 0.1513, 1.1507, 0.2343 |
| ABO | 0.0165, 0.0211, 0.0259 | 0.6778, 0.3112, 0.1848 | 1.1929, 2.7513, 0.1658 |
| BO | 0.0165, 0.0764, 0.1750 | 0.6778, 0.6957, 0.5607 | 1.1929, 2.8283, 0.1386 |
| CHC-prescribe | 0.1205, 0.1506, 0.1506 | 1.8988, 1.9849, 1.9922 | 1.8663, 1.6100, 3.0000 |

## CHC-prescribe against DCBO on the same logs

|  | stat | ind | nonstat |
|---|---|---|---|
| total regret, CHC minus DCBO [95%] | -0.0759 [-0.3030, 0.1154] | 3.3311 [3.0380, 3.6155] | 5.0819 [3.8155, 6.5488] |
| seeds CHC lower / higher | 7 / 3 | 0 / 10 | 0 / 10 |

## The set point `prescribe` needs in place of *minimise*

| target, below the lowest logged Y | stat | ind | nonstat |
|---|---|---|---|
| 1000 log-ranges (the arm above) | 0.4216 | 5.8759 | 6.4764 |
| 1 log-range | 0.9352 | 5.5738 | 6.8948 |

## The reference's recorded outcomes, replayed on the port

Runs whose every recorded outcome the port reproduces from the same decisions.

| method | stat | ind | nonstat |
|---|---|---|---|
| DCBO | 10 / 10 | 10 / 10 | 10 / 10 |
| CBO | 10 / 10 | 10 / 10 | 7 / 10 |
| ABO | 10 / 10 | 10 / 10 | 10 / 10 |
| BO | 10 / 10 | 10 / 10 | 10 / 10 |

Wall-clock per arm and seed is recorded in `track_l.json` and quoted nowhere: it was not measured on a clean machine.

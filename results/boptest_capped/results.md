# P3.3 -- the capped block against the taper on the live BOPTEST heat pump

Produced by `causaldyn_bench.boptest_capped` at its pre-registered defaults; the
pre-registration is that module's docstring, committed before this run.

- test case `bestest_hydronic_heat_pump`, rounds of 2 h, target 21.5 C, 288 rounds per episode
- precision: **float64** (`JAX_ENABLE_X64`)
- windows start on days 1, 9, 17, 25, 33, 41, 49; 24 seeds each; 168 pairs

## Stages 0 and 1

- drift `(0.5212, -0.03002)`, weather `(0.0354, 0.1325, 0.0016)`, PRBS authority at the target `0.5175`, PRBS residual sd `0.1594` K/h
- `theta_ref = 0.6387` K/h per unit modulation from probe energy `57.456` over 900 rounds; residual sd `s = 0.1723` K/h
- halves `0.6434` / `0.6327`; response to the previous round's probe `lag1 = 0.0198`

## Plan (printed before any arm ran)

- `A = 1.6320`, `K = 0.5460`, `c = 33.701`, `I0 = 9.804`, mean oracle cap `0.0933`; floor-identity residual `0.0e+00`
- budget `M* = 1.3613`, delivered by the capped block in 13 rounds of the mean schedule; taper scale `kappa = 0.0705`
- predicted regret: block `5.1715`, taper `5.7395` K^2 per episode
- surrogate simulation (280 pairs): gap `1.1617` +- `20.4544` (sd); pairs for 80% power `2431`; power at the design's 168 pairs `0.111`

## The gate

`D = R_taper - R_block`: mean **-0.3084** K^2, 95% interval [-1.4956, 0.8974] (stratified bootstrap, 20000 resamples); the taper won 86 of 168 pairs.

- decision: **INCONCLUSIVE**
- V1 (exploring matters): lazy mean regret `16.3757` against the block's upper bound `12.0575` -> pass
- V2 (same budget): mean shortfall `0.0000` -> pass
- the verdict STANDS

## Arms (means over pairs; rounds are medians)

| arm | regret K^2 | predicted | spent | rounds probed | half spent by | last probe | error rms | coverage | var z | floored | tdis K h | ener kWh/m2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block | 9.7191 | 5.1715 | 1.3413 | 14.8 | 10 | 22 | 0.1824 | 0.893 | 1.683 | 1.8 | 3.457 | 5.758 |
| taper | 9.4107 | 5.7395 | 1.3413 | 150.4 | 58 | 185 | 0.1390 | 0.958 | 0.939 | 2.7 | 2.351 | 5.719 |
| lazy | 16.3757 | -- | 0 | 0 | -- | -- | -- | -- | -- | -- | 5.300 | 5.773 |

## Per window

| window | day | oracle loss | mean cap | saturated | mean D | taper wins |
|---|---|---|---|---|---|---|
| 0 | 1 | 31.1059 | 0.1128 | 0.108 | -3.7084 | 19/24 |
| 1 | 9 | 39.9434 | 0.1031 | 0.125 | 0.4209 | 11/24 |
| 2 | 17 | 45.0239 | 0.1042 | 0.128 | -1.7584 | 14/24 |
| 3 | 25 | 97.9514 | 0.1035 | 0.156 | -1.5441 | 12/24 |
| 4 | 33 | 91.5838 | 0.0872 | 0.174 | 2.8915 | 5/24 |
| 5 | 41 | 131.6057 | 0.0760 | 0.236 | -1.0717 | 13/24 |
| 6 | 49 | 126.7433 | 0.0664 | 0.264 | 2.6117 | 12/24 |

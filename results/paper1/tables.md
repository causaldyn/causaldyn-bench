# P1 tables -- debias every channel

Scalar transfer plant `b = 1`, `rr = 0.5`, `xt = 1`; its window expansion `lambda = 1.33333`, `c2 = -1.77778` is reconstructed here in closed form and gated against the certificate (order-doubling residual `0.0e+00`, prediction excess over the next order `0.0e+00`). Every exponent below is a fit to a log-log sweep, so every table names the window it was fitted on.

## Table 1 -- a fitted slope is the exponent plus a window term, and the term is exact

| window in `delta` | p | 2p | measured | `lambda` term | `c2` term | residual | float floor | limited by |
|---|---|---|---|---|---|---|---|---|
| [0.01, 0.2] | 1 | 2 | 2.048478 | +7.68e-02 | -3.59e-02 | +7.6e-03 | 2.8e-13 | window |
| [0.01, 0.2] | 2 | 4 | 4.012425 | +1.35e-02 | -1.10e-03 | +5.6e-05 | 2.8e-11 | window |
| [0.01, 0.2] | 3 | 6 | 6.002273 | +2.31e-03 | -3.82e-05 | +4.3e-07 | 2.8e-09 | window |
| [0.001, 0.02] | 1 | 2 | 2.007332 | +7.68e-03 | -3.59e-04 | +9.3e-06 | 2.8e-12 | window |
| [0.001, 0.02] | 2 | 4 | 4.000135 | +1.35e-04 | -1.10e-07 | +8.4e-11 | 2.8e-09 | float |
| [0.001, 0.02] | 3 | 6 | 6.000002 | +2.31e-06 | -3.82e-11 | +2.1e-08 | 2.8e-06 | float |
| [0.0001, 0.002] | 1 | 2 | 2.000765 | +7.68e-04 | -3.59e-06 | +9.6e-09 | 2.8e-11 | window |
| [0.0001, 0.002] | 2 | 4 | 4.000001 | +1.35e-06 | -1.10e-11 | +1.3e-08 | 2.8e-07 | float |
| [0.0001, 0.002] | 3 | 6 | 6.000098 | +2.31e-09 | -3.82e-17 | +9.8e-05 | 2.8e-03 | float |
| [1e-05, 0.0002] | 1 | 2 | 2.000077 | +7.68e-05 | -3.59e-08 | +5.9e-12 | 2.8e-10 | float |
| [1e-05, 0.0002] | 2 | 4 | 4.000001 | +1.35e-08 | -1.10e-15 | +5.5e-07 | 2.8e-05 | float |
| [1e-05, 0.0002] | 3 | 6 | 6.035485 | +2.31e-12 | -3.82e-23 | +3.5e-02 | 2.8e+00 | float |

`2.048` is not `2` up to noise; it is `2` plus `+4.85e-02` of window, of which the two closed-form terms account for `+4.09e-02`, and what is left is one more power of `delta`. Read the table down a column and the two failure modes bracket the usable window from opposite ends: 5 of the 12 cells are still window-limited, the rest have had the window term shrink below what double precision invents, and at the bottom right the fit is reporting the rounding of a difference of two nearly equal numbers rather than an exponent.

## Table 2 -- the bottleneck is a MIN over channels, not an average

| window in `delta` | 2-channel, spillover plug-in | 2-channel, both debiased | 3-channel, `W` plug-in | 3-channel, all debiased | underflowed |
|---|---|---|---|---|---|
| [0.01, 0.2] | 1.9997 | 3.9855 | 2.0087 | 3.9849 | no |
| [0.001, 0.02] | 1.9984 | 3.9997 | 1.9942 | 4.0005 | no |
| [0.0001, 0.002] | 1.9998 | nan | 1.9994 | nan | yes |
| [1e-05, 0.0002] | 2.0000 | nan | 1.9999 | nan | yes |

One plug-in channel caps the exponent at `2` whether there are two channels or three, and debiasing the others does not move it -- which is the content of the claim that the rate is a minimum over channels, and the two bottleneck columns stay within `8.7e-03` of `2` on every window. The full-orthogonality columns are the ones the float floor reaches first, because `delta^4` underflows two windows before `delta^2` does; the last column marks where a regret in the sweep hit exactly zero, and a `nan` slope beside it is the fit taking `log 0`.

## Table 3 -- end to end: the `delta` regime is exact, the `G` regime is a window

(a) deterministic bias order, `5`-point sweep:

| arm | fitted slope | theory |
|---|---|---|
| half-orthogonal (spillover plug-in) | 2.0031 | 2 |
| fully orthogonal | 3.9776 | 4 |

(b) real cross-fit DML, sampling-dominated, as the cluster grid moves up:

| `G` window | slope | nested prefixes | pooled sampling scale | regret at the largest `G` |
|---|---|---|---|---|
| 10 .. 160 | -1.1394 | -1.1190, -1.1329, -1.1415 | 0.0133 | 1.167e-03 |
| 20 .. 320 | -1.1212 | -1.0547, -1.0465, -1.0461 | 0.0385 | 5.700e-04 |
| 40 .. 640 | -1.0539 | -1.0283, -0.9834, -0.9734 | 0.0237 | 3.204e-04 |
| 80 .. 1280 | -1.0242 | -1.0343, -1.0038, -1.0513 | 0.0194 | 1.528e-04 |

The slope ends at `-1.0242` on the top window against `-1.1394` on the shipped one. Whether that is the window or the seeds is what the third column decides, and it is not an independent replicate -- the certificate seeds by index, so the `n/2` run is a prefix of the `n` run. That is what makes it usable: a prefix mean minus the full mean has exactly the variance of the full mean, so it is an error bar; it is pooled over 3 nested rungs rather than read off one, because one draw of `|N(0, sigma^2)|` is too noisy to decide anything. Across the ladder the statistic moves `0.1151` against a pooled sampling scale of `0.0235` on the two endpoints -- `4.9x`, so the walk is the window and not the seeds.

## Table 4 -- the floor is two-sided, once `G` is large enough

| `G` window | plateau slope | nested prefixes | pooled sampling scale | `c0` | `G x regret` range |
|---|---|---|---|---|---|
| 10 .. 160 | -0.1767 | -0.1903, -0.1976, -0.2651 | 0.0212 | 0.1931 | 0.193 .. 0.331 |
| 20 .. 320 | -0.1147 | -0.0809, -0.1232, -0.1103 | 0.0263 | 0.1835 | 0.184 .. 0.254 |
| 40 .. 640 | -0.0544 | -0.0272, -0.0874, -0.0112 | 0.0365 | 0.2018 | 0.184 .. 0.223 |
| 80 .. 1280 | -0.0454 | -0.0018, -0.0412, +0.0279 | 0.0359 | 0.1827 | 0.183 .. 0.219 |

`G * regret` has to flatten onto a positive constant for `1/G` to bound the regret from below as well as above. The plateau slope walks from `-0.1767` to `-0.0454` while `c0` stays inside `0.1827 .. 0.2018`, so the constant was never the thing in doubt -- the shipped grid was. Across the ladder the statistic moves `0.1313` against a pooled sampling scale of `0.0417` on the two endpoints -- `3.1x`, so the walk is the window and not the seeds.

## Table 5 -- the quadratic law on random error, and the event it is conditional on

`400` perturbation draws per level, 6 seeds.

| seed | fitted exponent | largest unbounded share over the levels |
|---|---|---|
| 0 | 2.0154 | 0.0000 |
| 1 | 2.0329 | 0.0000 |
| 2 | 2.0302 | 0.0000 |
| 3 | 2.0270 | 0.0000 |
| 4 | 2.0873 | 0.0000 |
| 5 | 2.0913 | 0.0025 |

Range `2.0154 .. 2.0913`, median `2.0316`, relative spread `3.7e-02`. Every seed sits above `2` and none reaches `2.1`: the excess is the same window term Table 1 prices, not sampling noise, and the appendix's single `2.05` is one draw from this range rather than the number. The last column is the honesty the exponent needs -- a draw whose perturbed plant is unstabilisable, or whose gain fails to stabilise the true plant, has no finite regret at all, so the exponent is conditional on the complement of an event that reaches `0.0025` here.

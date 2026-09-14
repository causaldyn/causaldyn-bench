# P3 tables -- information-exploration duality

Plant `A = 1.5`, `K = 0.0740741`, `c = 1.22449`, `I0 = 1`, recovered from the certificates and checked against them (`c_causal` residual `0.0e+00`, floor residual `0.0e+00`). Tables 1-4 are exact functions of the model and carry no intervals; Table 5 is a range over seeds, and says which of its columns moved.

## Table 1 -- the sequential minimax floor, and who attains it

| T | floor | burst | taper | constant v | greedy |
|---|---|---|---|---|---|
| 1,000 | 17.83 | 1.004 | 1.353 | 1.72 | 4.155 |
| 10,000 | 59.02 | 1.001 | 1.378 | 3.036 | 12.55 |
| 100,000 | 189.3 | 1 | 1.398 | 8.152 | 39.13 |
| 1,000,000 | 601.2 | 1 | 1.407 | 25.04 | 123.2 |

Ratios to the floor, so a cell below 1 falsifies the bound; the minimum over all policies and horizons is `1.000123`. The constant is SHARP, not a rate: burst reaches `1.000123` while taper sits at `1.4072` against `sqrt(2) = 1.4142`. `c_causal = 0.602464` and its log-log slope in `eta` is `-0.5000` -- the `1/sqrt(eta)` causal scaling.

## Table 2 -- a cap costs an additive logarithm, so its ratio to the floor decreases

| cap | T = 1,000 | T = 10,000 | T = 100,000 | T = 1,000,000 |
|---|---|---|---|---|
| 0.1 | 1.0415 | 1.0236 | 1.0110 | 1.0046 |
| 0.03 | 1.1308 | 1.0767 | 1.0360 | 1.0152 |
| 0.01 | 1.3631 | 1.2258 | 1.1079 | 1.0454 |

Cost over the uncapped floor. Every row decreases: unlike the taper's `sqrt(2)` factor, a cap adds a constant to a growing floor. Block lengths, same cells:

| cap | T = 1,000 | T = 10,000 | T = 100,000 | T = 1,000,000 |
|---|---|---|---|---|
| 0.1 | 54 | 191 | 625 | 1,998 |
| 0.03 | 166 | 621 | 2,068 | 6,645 |
| 0.01 | 407 | 1,753 | 6,287 | 20,018 |

## Table 3 -- the stopping MASS is the invariant, not the block length

| schedule | block rounds | delivered mass | fixed point | gap, in caps | Result 56 / delivered |
|---|---|---|---|---|---|
| constant 0.03 | 376 | 11.2800 | 11.2727 | 0.243 | 1.054 |
| ramp up 0.01->0.05 | 767 | 10.6083 | 10.6019 | 0.363 | 1.120 |
| ramp down 0.05->0.01 | 236 | 11.5226 | 11.5040 | 0.391 | 1.031 |
| dead first third | 1,632 | 8.9700 | 8.9557 | 0.476 | 1.325 |
| uniform noise U(0, 0.06) | 352 | 11.3065 | 11.3110 | 0.500 | 1.051 |

The block lengths span nearly an order of magnitude and every delivered mass lands on the same fixed point. The last column is why the leading form was replaced: it does not know when the caps open.

## Table 4 -- what the closed form drops is a constant, `K/(2 A c cap) = 0.672154`

(a) horizon ladder at `cap = 0.03`:

| T | exact mass | Result 56 - exact | over the ceiling | residual x sqrt(T) |
|---|---|---|---|---|
| 10,000 | 18.6319 | 0.633608 | 0.942654 | -3.8546 |
| 100,000 | 62.0287 | 0.659954 | 0.981850 | -3.8579 |
| 1,000,000 | 199.3364 | 0.668295 | 0.994260 | -3.8582 |
| 10,000,000 | 633.5653 | 0.670934 | 0.998185 | -3.8583 |
| 100,000,000 | 2006.7252 | 0.671768 | 0.999426 | -3.8583 |

The gap rises to the ceiling and stops -- it never reaches it, and never exceeds it. The last column converges to `-3.85826`, the coefficient the expansion predicts.

(b) cap ladder at `T = 4,000`:

| cap | exact mass | Result 56 - exact | ceiling | as a share of the mass |
|---|---|---|---|---|
| 0.1 | 11.6973 | 0.1871 | 0.2016 | 1.60% |
| 0.03 | 11.2731 | 0.6113 | 0.6722 | 5.42% |
| 0.01 | 10.1544 | 1.7300 | 2.0165 | 17.04% |

(c) the library's fixed point against the closed-form root, in units of the cap:

| T | fixed point | root | gap, in caps |
|---|---|---|---|
| 10,000 | 18.630889 | 18.631861 | 0.0324 |
| 100,000 | 62.028548 | 62.028669 | 0.0040 |
| 1,000,000 | 199.336351 | 199.336397 | 0.0015 |

## Table 5 -- the matrix floor is a trace, so confounding is priced by ALIGNMENT

Range over 5 seeds. A column marked exact did not move at all; with a single seed the column is `?`, because one draw cannot tell an exact quantity from a lucky one.

| quantity | range | relative spread | seed-invariant |
|---|---|---|---|
| aligned_ratio | 2.833 .. 3.643 | 2.2e-01 | -- |
| orthogonal_ratio | 1 .. 1 | 3.3e-16 | yes |
| worst_single_direction | 2.833 .. 3.643 | 2.2e-01 | -- |
| information_loss | 4 | 0.0e+00 | yes |
| live_channel_weight | 0.07407 | 0.0e+00 | yes |
| knife_edge_weight | 1.233e-32 | 0.0e+00 | yes |
| floor | 0.01319 | 0.0e+00 | yes |
| plugin_ratio | 1.67 .. 1.686 | 9.1e-03 | -- |
| hodges_pointwise_ratio | 0 | 0.0e+00 | yes |
| hodges_bayes_ratio | 31.35 .. 32.06 | 2.2e-02 | -- |

The same information loss costs a factor in the direction the optimal action leans on and exactly nothing in a direction in the kernel of `Psi'` -- which is the whole content of the trace form, and is invisible to a scalar plant that has only one direction.

The spread column is the reason this table is a range. The certificate draws the 2x2 effect matrix from its seed, so the SIZE of the aligned factor is a property of that draw and moves more across seeds than either estimator column does. The BRACKET does not: `orthogonal_ratio` is 1 to floating-point zero at every seed, `worst_single_direction` reproduces `aligned_ratio` exactly, and the factor stays strictly inside `(1, k)`. Quote the bracket; the factor is an instance of it.

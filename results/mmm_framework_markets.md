# Family 21: mmm-framework's generated worlds, pilot (float64)

The 26 worlds of mmm-framework 1.5.0's generator the bench holds, each drawn once by the tool's own code at its default seed and length (`mmm_framework.synth.generate_mff(name)`) and read as data: a rival's worlds, written to test its own model. The tool's pages count sixteen stress worlds, but none of its wheel's documents names them, so every world of its registry the bench can read is here. Each arm's reading of each channel's contribution over the file's weeks and its ROAS is scored against the world's answer key: the relative error, the reading over the stated less 1, its size averaged over the channels.

| world | what it breaks, by its key | the tool's model can represent it | channels | observational: contribution, ROAS | mmm-framework: contribution, ROAS | mmm-framework's interval holds | the tool's own check |
|---|---|---|---|---|---|---|---|
| realistic | (realistic mix: confounding + many controls + weak/noisy media) | yes | 7 | 0.145, 0.145 | 0.344, 0.344 | 6 of 7 | silent |
| clean | nothing, by its key | yes | 4 | 0.045, 0.045 | 0.168, 0.168 | 4 of 4 | silent |
| unobserved_confounding | no unobserved confounding (exogeneity of spend) | yes | 4 | 0.611, 0.611 | 0.291, 0.291 | 4 of 4 | silent |
| confounding_controlled | no unobserved confounding (exogeneity of spend) | yes | 4 | 0.213, 0.213 | 0.239, 0.239 | 4 of 4 | silent |
| reverse_causality | no unobserved confounding (exogeneity of spend) | yes | 4 | 0.098, 0.098 | 0.152, 0.152 | 3 of 4 | silent |
| multicollinearity | identifiability (independent spend variation) | yes | 4 | 0.763, 0.763 | 0.164, 0.164 | 4 of 4 | silent |
| adstock_misspec | adstock functional form / carryover window | yes | 4 | 0.873, 0.873 | 0.712, 0.712 | 2 of 4 | a ConvergenceWarning |
| saturation_misspec | saturation functional form (concavity / no threshold) | yes | 4 | 0.379, 0.379 | 0.222, 0.222 | 3 of 4 | silent |
| time_varying_beta | time-invariant coefficients (stationarity) | yes | 4 | 0.103, 0.103 | 0.242, 0.242 | 3 of 4 | silent |
| heavy_tailed_noise | Gaussian, homoscedastic likelihood | yes | 4 | 0.077, 0.077 | 0.360, 0.360 | 4 of 4 | silent |
| synergy | additive separability of channels | yes | 4 | 0.129, 0.129 | 0.209, 0.209 | 3 of 4 | silent |
| spend_outliers | robustness of per-channel max-normalization | yes | 4 | 0.743, 0.814 | 0.235, 0.445 | 1 of 4 | silent |
| mixed_data_errors | accurate recording of spend & demand events | yes | 4 | 0.398, 0.411 | 0.357, 0.371 | 2 of 4 | silent |
| negative_effect | positivity of media effects (Gamma prior, beta >= 0) | no | 4 | 0.423, 0.423 | 0.504, 0.504 | 3 of 4 | silent |
| trend_break | smooth global trend (no structural breaks) | yes | 4 | 0.185, 0.185 | 0.239, 0.239 | 4 of 4 | silent |
| seasonality_misspec | static low-order Fourier seasonality | yes | 4 | 0.295, 0.295 | 0.161, 0.161 | 4 of 4 | silent |
| dense_controls | a parsimonious, correctly-chosen control set | yes | 4 | 0.099, 0.099 | 0.142, 0.142 | 4 of 4 | silent |
| economic_health | no unobserved confounding (exogeneity of spend) | yes | 4 | 2.159, 2.159 | 1.196, 1.196 | 3 of 4 | silent |
| reach_frequency | impressions are a pure volume substitute (no frequency saturation) | yes | 4 | 0.229, 0.229 | 0.247, 0.247 | 2 of 4 | silent |
| promo_and_media | nothing, by its key | yes | 4 | 0.394, 0.394 | 0.350, 0.350 | 4 of 4 | silent |
| promo_endogenous | lever exogeneity (promo/price respond to the demand shock) | yes | 4 | 0.298, 0.298 | 0.460, 0.460 | 4 of 4 | silent |
| breakout_heterogeneous | nothing, by its key | yes | 6 | 0.557, 0.557 | 0.347, 0.347 | 5 of 6 | silent |
| breakout_homogeneous | nothing, by its key | yes | 6 | 0.381, 0.381 | 0.222, 0.222 | 5 of 6 | silent |
| breakout_collinear | identifiability (independent breakout variation) | yes | 6 | 1.744, 1.744 | 1.102, 1.102 | 5 of 6 | silent |
| geo_clean | nothing, by its key | yes | 4 | 0.630, 0.630 | 0.612, 0.612 | 4 of 4 | silent |
| geo_heterogeneous | global media coefficients (per-geo effectiveness homogeneity) | no | 4 | 0.440, 0.440 | 0.537, 0.537 | 4 of 4 | silent |

observational: a cell on 26 of 26 worlds; the contribution's error, averaged over a world's channels, 0.380 at the median world, 0.045 to 2.159.
mmm-framework: a cell on 26 of 26 worlds; the contribution's error, averaged over a world's channels, 0.269 at the median world, 0.142 to 1.196.
mmm-framework's contribution error is the smaller of the two on 13 of 26 worlds; its interval holds every channel's stated ROAS on 12 of the 26; its own check raised a ConvergenceWarning on 1 of them.

"The tool's own check" is whether the tool's fit raised a ConvergenceWarning, which it raises where its own check of a fit fails; "silent" where it raised none.

## The worlds, channel by channel

The last column is each channel's stated contribution over its stated ROAS, as a share of its spend in the file: 1 where the stated ROAS is read on the file's spend. Where the tool plants recording errors in the spend the file holds, the stated ROAS is read on the true spend, which the key's notes state; the contribution needs none.

### realistic

What it breaks, by its key: (realistic mix: confounding + many controls + weak/noisy media). The bench's arm reads the controls category_demand, distribution, price, competitor_promo, weather, holiday.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 8948.4 | 6603.7 [-0.262] | 5143.2 [-0.425] | 0.794 | 0.586 [-0.262] | 0.456 [-0.425] (0.287, 0.663) | 1.0000 |
| Search | 3977.8 | 4193.4 [+0.054] | 5919.2 [+0.488] | 0.508 | 0.536 [+0.054] | 0.756 [+0.488] (0.345, 1.360) | 1.0000 |
| Social | 3947.6 | 5665.2 [+0.435] | 5554.3 [+0.407] | 0.638 | 0.916 [+0.435] | 0.898 [+0.407] (0.394, 1.759) | 1.0000 |
| Display | 3090.5 | 3034.2 [-0.018] | 3934.3 [+0.273] | 0.613 | 0.602 [-0.018] | 0.781 [+0.273] (0.206, 1.767) | 1.0000 |
| Video | 8286.2 | 8260.4 [-0.003] | 6592.4 [-0.204] | 1.932 | 1.926 [-0.003] | 1.537 [-0.204] (0.890, 2.630) | 1.0000 |
| Radio | 3301.0 | 3924.6 [+0.189] | 3377.4 [+0.023] | 1.021 | 1.214 [+0.189] | 1.045 [+0.023] (0.105, 2.753) | 1.0000 |
| Print | 2096.6 | 1989.6 [-0.051] | 3333.9 [+0.590] | 1.008 | 0.956 [-0.051] | 1.602 [+0.590] (0.110, 4.306) | 1.0000 |

### clean

What it breaks, by its key: nothing. The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 5667.6 | 5109.7 [-0.098] | 4341.8 [-0.234] | 0.616 | 0.556 [-0.098] | 0.472 [-0.234] (0.322, 0.675) | 1.0000 |
| Search | 4530.9 | 4427.4 [-0.023] | 4102.0 [-0.095] | 0.785 | 0.767 [-0.023] | 0.711 [-0.095] (0.493, 1.029) | 1.0000 |
| Social | 4017.2 | 4230.3 [+0.053] | 3325.2 [-0.172] | 0.854 | 0.899 [+0.053] | 0.707 [-0.172] (0.513, 0.932) | 1.0000 |
| Display | 2957.9 | 2975.4 [+0.006] | 2448.1 [-0.172] | 0.806 | 0.811 [+0.006] | 0.667 [-0.172] (0.380, 1.074) | 1.0000 |

### unobserved_confounding

What it breaks, by its key: no unobserved confounding (exogeneity of spend). The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 6309.2 | 5432.5 [-0.139] | 4163.3 [-0.340] | 0.682 | 0.587 [-0.139] | 0.450 [-0.340] (0.249, 0.709) | 1.0000 |
| Search | 3653.8 | 9913.0 [+1.713] | 5004.6 [+0.370] | 0.630 | 1.709 [+1.713] | 0.863 [+0.370] (0.530, 1.288) | 1.0000 |
| Social | 2344.8 | 3483.2 [+0.486] | 3355.9 [+0.431] | 0.469 | 0.696 [+0.486] | 0.671 [+0.431] (0.337, 1.136) | 1.0000 |
| Display | 2577.0 | 2846.5 [+0.105] | 2631.7 [+0.021] | 0.668 | 0.738 [+0.105] | 0.682 [+0.021] (0.159, 1.523) | 1.0000 |

### confounding_controlled

What it breaks, by its key: no unobserved confounding (exogeneity of spend). The bench's arm reads the controls Price, CategoryDemand.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 6309.2 | 6457.0 [+0.023] | 4712.1 [-0.253] | 0.682 | 0.698 [+0.023] | 0.509 [-0.253] (0.316, 0.770) | 1.0000 |
| Search | 3653.8 | 5667.3 [+0.551] | 4499.6 [+0.231] | 0.630 | 0.977 [+0.551] | 0.776 [+0.231] (0.512, 1.104) | 1.0000 |
| Social | 2344.8 | 2935.3 [+0.252] | 3072.7 [+0.310] | 0.469 | 0.587 [+0.252] | 0.614 [+0.310] (0.329, 1.021) | 1.0000 |
| Display | 2577.0 | 2511.6 [-0.025] | 2996.7 [+0.163] | 0.668 | 0.651 [-0.025] | 0.777 [+0.163] (0.293, 1.598) | 1.0000 |

### reverse_causality

What it breaks, by its key: no unobserved confounding (exogeneity of spend). The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 7210.9 | 6609.5 [-0.083] | 4863.1 [-0.326] | 0.586 | 0.537 [-0.083] | 0.395 [-0.326] (0.281, 0.539) | 1.0000 |
| Search | 4716.3 | 5699.7 [+0.209] | 5058.6 [+0.073] | 0.548 | 0.662 [+0.209] | 0.588 [+0.073] (0.426, 0.781) | 1.0000 |
| Social | 3611.1 | 3423.6 [-0.052] | 2896.1 [-0.198] | 0.511 | 0.484 [-0.052] | 0.410 [-0.198] (0.261, 0.609) | 1.0000 |
| Display | 2495.2 | 2610.9 [+0.046] | 2465.3 [-0.012] | 0.492 | 0.515 [+0.046] | 0.486 [-0.012] (0.271, 0.783) | 1.0000 |

### multicollinearity

What it breaks, by its key: identifiability (independent spend variation). The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 8800.3 | 5771.7 [-0.344] | 8070.5 [-0.083] | 0.980 | 0.643 [-0.344] | 0.899 [-0.083] (0.173, 1.763) | 1.0000 |
| Search | 5677.6 | 1896.6 [-0.666] | 4903.5 [-0.136] | 1.032 | 0.345 [-0.666] | 0.891 [-0.136] (0.124, 2.232) | 1.0000 |
| Social | 5263.3 | 11712.5 [+1.225] | 6235.3 [+0.185] | 1.147 | 2.552 [+1.225] | 1.359 [+0.185] (0.224, 3.151) | 1.0000 |
| Display | 3531.3 | 6418.1 [+0.817] | 4420.9 [+0.252] | 0.963 | 1.750 [+0.817] | 1.205 [+0.252] (0.171, 3.171) | 1.0000 |

### adstock_misspec

What it breaks, by its key: adstock functional form / carryover window. The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 7008.2 | 0.0 [-1.000] | 492.8 [-0.930] | 0.713 | 0.000 [-1.000] | 0.050 [-0.930] (-0.058, 0.196) | 1.0000 |
| Search | 5066.0 | 9273.7 [+0.831] | 3383.3 [-0.332] | 0.917 | 1.679 [+0.831] | 0.613 [-0.332] (-0.019, 1.608) | 1.0000 |
| Social | 4151.6 | 0.0 [-1.000] | 485.4 [-0.883] | 0.846 | 0.000 [-1.000] | 0.099 [-0.883] (-0.107, 0.374) | 1.0000 |
| Display | 2529.6 | 861.1 [-0.660] | 752.3 [-0.703] | 0.649 | 0.221 [-0.660] | 0.193 [-0.703] (-0.125, 0.750) | 1.0000 |

### saturation_misspec

What it breaks, by its key: saturation functional form (concavity / no threshold). The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 3774.5 | 7315.5 [+0.938] | 6094.3 [+0.615] | 0.390 | 0.756 [+0.938] | 0.630 [+0.615] (0.435, 0.877) | 1.0000 |
| Search | 3805.7 | 4530.6 [+0.190] | 4037.5 [+0.061] | 0.639 | 0.761 [+0.190] | 0.678 [+0.061] (0.475, 0.931) | 1.0000 |
| Social | 3374.0 | 4433.4 [+0.314] | 3271.6 [-0.030] | 0.718 | 0.943 [+0.314] | 0.696 [-0.030] (0.487, 0.920) | 1.0000 |
| Display | 2284.6 | 2450.3 [+0.073] | 1872.3 [-0.180] | 0.593 | 0.637 [+0.073] | 0.486 [-0.180] (0.234, 0.748) | 1.0000 |

### time_varying_beta

What it breaks, by its key: time-invariant coefficients (stationarity). The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 6505.1 | 5306.9 [-0.184] | 3523.2 [-0.458] | 0.658 | 0.537 [-0.184] | 0.357 [-0.458] (0.171, 0.596) | 1.0000 |
| Search | 5563.7 | 5259.2 [-0.055] | 4037.1 [-0.274] | 1.012 | 0.956 [-0.055] | 0.734 [-0.274] (0.491, 1.012) | 1.0000 |
| Social | 4166.1 | 4832.4 [+0.160] | 4224.1 [+0.014] | 0.919 | 1.066 [+0.160] | 0.932 [+0.014] (0.551, 1.449) | 1.0000 |
| Display | 3055.7 | 3016.5 [-0.013] | 2382.1 [-0.220] | 0.782 | 0.772 [-0.013] | 0.609 [-0.220] (0.209, 1.148) | 1.0000 |

### heavy_tailed_noise

What it breaks, by its key: Gaussian, homoscedastic likelihood. The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 7002.4 | 7409.5 [+0.058] | 4148.5 [-0.408] | 0.771 | 0.816 [+0.058] | 0.457 [-0.408] (0.176, 0.815) | 1.0000 |
| Search | 4162.2 | 3809.2 [-0.085] | 2527.7 [-0.393] | 0.786 | 0.719 [-0.085] | 0.477 [-0.393] (0.057, 1.047) | 1.0000 |
| Social | 3996.9 | 3980.4 [-0.004] | 2676.9 [-0.330] | 0.937 | 0.933 [-0.004] | 0.627 [-0.330] (0.154, 1.178) | 1.0000 |
| Display | 2560.1 | 2151.3 [-0.160] | 1769.4 [-0.309] | 0.663 | 0.557 [-0.160] | 0.458 [-0.309] (-0.050, 1.061) | 1.0000 |

### synergy

What it breaks, by its key: additive separability of channels. The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 8958.5 | 8048.7 [-0.102] | 7630.2 [-0.148] | 0.936 | 0.841 [-0.102] | 0.797 [-0.148] (0.559, 1.107) | 1.0000 |
| Search | 6690.0 | 6480.7 [-0.031] | 5146.5 [-0.231] | 1.170 | 1.133 [-0.031] | 0.900 [-0.231] (0.709, 1.117) | 1.0000 |
| Social | 3848.0 | 2871.4 [-0.254] | 2534.8 [-0.341] | 0.829 | 0.619 [-0.254] | 0.546 [-0.341] (0.314, 0.830) | 1.0000 |
| Display | 2382.1 | 2685.8 [+0.127] | 2107.9 [-0.115] | 0.614 | 0.692 [+0.127] | 0.543 [-0.115] (0.236, 0.967) | 1.0000 |

### spend_outliers

What it breaks, by its key: robustness of per-channel max-normalization. The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 7222.2 | 607.1 [-0.916] | 3905.3 [-0.459] | 0.768 | 0.047 [-0.939] | 0.300 [-0.609] (0.167, 0.458) | 0.7232 |
| Search | 4625.9 | 1496.7 [-0.676] | 3832.2 [-0.172] | 0.801 | 0.185 [-0.769] | 0.473 [-0.410] (0.285, 0.698) | 0.7124 |
| Social | 4331.0 | 1543.0 [-0.644] | 3575.3 [-0.174] | 0.891 | 0.232 [-0.740] | 0.537 [-0.397] (0.321, 0.792) | 0.7300 |
| Display | 2920.8 | 766.3 [-0.738] | 2525.4 [-0.135] | 0.751 | 0.145 [-0.806] | 0.479 [-0.362] (0.152, 0.916) | 0.7379 |

### mixed_data_errors

What it breaks, by its key: accurate recording of spend & demand events. The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 7368.4 | 4014.6 [-0.455] | 4875.4 [-0.338] | 0.807 | 0.396 [-0.509] | 0.481 [-0.404] (0.286, 0.728) | 0.9004 |
| Search | 11004.1 | 14213.3 [+0.292] | 2774.5 [-0.748] | 1.163 | 1.514 [+0.302] | 0.296 [-0.746] (0.022, 0.652) | 1.0078 |
| Social | 3302.7 | 5581.8 [+0.690] | 3901.1 [+0.181] | 0.700 | 1.177 [+0.681] | 0.823 [+0.175] (0.451, 1.318) | 0.9946 |
| Display | 2816.0 | 3247.6 [+0.153] | 2363.4 [-0.161] | 0.711 | 0.819 [+0.153] | 0.596 [-0.161] (0.122, 1.227) | 1.0000 |

### negative_effect

What it breaks, by its key: positivity of media effects (Gamma prior, beta >= 0). The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 7480.0 | 6531.9 [-0.127] | 4323.1 [-0.422] | 0.833 | 0.727 [-0.127] | 0.481 [-0.422] (0.198, 0.850) | 1.0000 |
| Search | 4723.9 | 6284.0 [+0.330] | 4185.3 [-0.114] | 0.793 | 1.055 [+0.330] | 0.703 [-0.114] (0.320, 1.172) | 1.0000 |
| Social | 3818.6 | 2917.6 [-0.236] | 2226.5 [-0.417] | 0.825 | 0.630 [-0.236] | 0.481 [-0.417] (0.087, 0.922) | 1.0000 |
| Display | -7788.2 | 0.0 [-1.000] | 495.0 [-1.064] | -2.116 | 0.000 [-1.000] | 0.135 [-1.064] (-0.306, 0.615) | 1.0000 |

### trend_break

What it breaks, by its key: smooth global trend (no structural breaks). The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 5467.7 | 3096.1 [-0.434] | 2401.6 [-0.561] | 0.437 | 0.247 [-0.434] | 0.192 [-0.561] (0.015, 0.438) | 1.0000 |
| Search | 4930.9 | 5306.8 [+0.076] | 4691.4 [-0.049] | 0.891 | 0.959 [+0.076] | 0.848 [-0.049] (0.405, 1.466) | 1.0000 |
| Social | 4153.8 | 4392.5 [+0.057] | 3478.0 [-0.163] | 0.882 | 0.933 [+0.057] | 0.739 [-0.163] (0.259, 1.394) | 1.0000 |
| Display | 2127.6 | 1761.3 [-0.172] | 1733.3 [-0.185] | 0.431 | 0.357 [-0.172] | 0.351 [-0.185] (-0.061, 0.946) | 1.0000 |

### seasonality_misspec

What it breaks, by its key: static low-order Fourier seasonality. The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 7751.1 | 11539.8 [+0.489] | 6907.1 [-0.109] | 0.808 | 1.203 [+0.489] | 0.720 [-0.109] (0.298, 1.272) | 1.0000 |
| Search | 4990.3 | 6923.2 [+0.387] | 4759.9 [-0.046] | 0.900 | 1.248 [+0.387] | 0.858 [-0.046] (0.324, 1.563) | 1.0000 |
| Social | 2818.5 | 2107.0 [-0.252] | 2525.5 [-0.104] | 0.409 | 0.306 [-0.252] | 0.366 [-0.104] (0.046, 0.799) | 1.0000 |
| Display | 2363.9 | 2237.9 [-0.053] | 1456.6 [-0.384] | 0.616 | 0.583 [-0.053] | 0.380 [-0.384] (-0.194, 1.268) | 1.0000 |

### dense_controls

What it breaks, by its key: a parsimonious, correctly-chosen control set. The bench's arm reads the controls demand_proxy, price, weather, noise_1, noise_2, noise_3, noise_4, noise_5, noise_6, noise_7, noise_8, noise_9, noise_10, noise_11, noise_12, noise_13, noise_14, noise_15, noise_16, noise_17, noise_18, decoy_tv, decoy_search, decoy_social, decoy_display.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 4932.2 | 5573.3 [+0.130] | 5780.9 [+0.172] | 0.477 | 0.540 [+0.130] | 0.560 [+0.172] (0.315, 0.926) | 1.0000 |
| Search | 3307.7 | 3188.5 [-0.036] | 4574.2 [+0.383] | 0.552 | 0.532 [-0.036] | 0.764 [+0.383] (0.289, 1.555) | 1.0000 |
| Social | 4368.6 | 4596.9 [+0.052] | 4366.9 [-0.000] | 0.924 | 0.973 [+0.052] | 0.924 [-0.000] (0.424, 1.741) | 1.0000 |
| Display | 2611.3 | 2145.6 [-0.178] | 2641.7 [+0.012] | 0.681 | 0.560 [-0.178] | 0.689 [+0.012] (0.077, 1.847) | 1.0000 |

### economic_health

What it breaks, by its key: no unobserved confounding (exogeneity of spend). The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 5720.4 | 9486.4 [+0.658] | 6496.9 [+0.136] | 0.629 | 1.044 [+0.658] | 0.715 [+0.136] (0.251, 1.399) | 1.0000 |
| Search | 2701.6 | 18975.5 [+6.024] | 13207.1 [+3.889] | 0.385 | 2.701 [+6.024] | 1.880 [+3.889] (0.935, 3.011) | 1.0000 |
| Social | 1901.0 | 4055.7 [+1.133] | 2873.2 [+0.511] | 0.370 | 0.790 [+1.133] | 0.560 [+0.511] (0.036, 1.315) | 1.0000 |
| Display | 2199.3 | 398.0 [-0.819] | 1657.4 [-0.246] | 0.615 | 0.111 [-0.819] | 0.463 [-0.246] (-0.312, 1.732) | 1.0000 |

### reach_frequency

What it breaks, by its key: impressions are a pure volume substitute (no frequency saturation). The bench's arm reads the controls Price, Frequency.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 13755.7 | 21949.2 [+0.596] | 8354.3 [-0.393] | 196.810 | 314.038 [+0.596] | 119.530 [-0.393] (74.306, 170.386) | 1.0000 |
| Search | 4519.4 | 3848.6 [-0.148] | 3102.9 [-0.313] | 0.782 | 0.666 [-0.148] | 0.537 [-0.313] (0.355, 0.739) | 1.0000 |
| Social | 3897.8 | 4274.6 [+0.097] | 3269.0 [-0.161] | 0.836 | 0.917 [+0.097] | 0.701 [-0.161] (0.439, 1.035) | 1.0000 |
| Display | 2864.7 | 3078.4 [+0.075] | 2520.1 [-0.120] | 0.770 | 0.827 [+0.075] | 0.677 [-0.120] (0.354, 1.134) | 1.0000 |

### promo_and_media

What it breaks, by its key: nothing. The bench's arm reads the controls Price, Promo, Demand_Proxy.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 16451.1 | 8802.0 [-0.465] | 10463.9 [-0.364] | 1.807 | 0.967 [-0.465] | 1.149 [-0.364] (0.665, 1.886) | 1.0000 |
| Search | 9404.2 | 6722.4 [-0.285] | 6606.2 [-0.298] | 1.526 | 1.091 [-0.285] | 1.072 [-0.298] (0.634, 1.920) | 1.0000 |
| Social | 9624.8 | 7238.8 [-0.248] | 7064.3 [-0.266] | 2.092 | 1.574 [-0.248] | 1.536 [-0.266] (0.792, 2.854) | 1.0000 |
| Display | 6532.6 | 2752.3 [-0.579] | 3439.9 [-0.473] | 1.709 | 0.720 [-0.579] | 0.900 [-0.473] (0.241, 2.172) | 1.0000 |

### promo_endogenous

What it breaks, by its key: lever exogeneity (promo/price respond to the demand shock). The bench's arm reads the controls Price, Promo.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 16893.3 | 13209.3 [-0.218] | 8531.5 [-0.495] | 1.880 | 1.470 [-0.218] | 0.950 [-0.495] (0.245, 1.937) | 1.0000 |
| Search | 9828.1 | 5741.1 [-0.416] | 5098.1 [-0.481] | 1.640 | 0.958 [-0.416] | 0.851 [-0.481] (0.010, 2.124) | 1.0000 |
| Social | 9035.9 | 7192.3 [-0.204] | 5326.0 [-0.411] | 1.837 | 1.462 [-0.204] | 1.083 [-0.411] (0.051, 2.635) | 1.0000 |
| Display | 6457.6 | 4166.7 [-0.355] | 3526.4 [-0.454] | 1.674 | 1.080 [-0.355] | 0.914 [-0.454] (-0.281, 2.666) | 1.0000 |

### breakout_heterogeneous

What it breaks, by its key: nothing. The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV_Premium | 5784.3 | 18915.8 [+2.270] | 3879.5 [-0.329] | 1.303 | 4.262 [+2.270] | 0.874 [-0.329] (0.514, 1.352) | 1.0000 |
| TV_Standard | 2202.5 | 1164.3 [-0.471] | 1206.3 [-0.452] | 0.664 | 0.351 [-0.471] | 0.364 [-0.452] (0.072, 0.830) | 1.0000 |
| TV_Remnant | 464.4 | 679.9 [+0.464] | 775.8 [+0.671] | 0.258 | 0.377 [+0.464] | 0.430 [+0.671] (-0.011, 1.113) | 1.0000 |
| Search | 4490.5 | 4399.7 [-0.020] | 3458.2 [-0.230] | 0.782 | 0.766 [-0.020] | 0.602 [-0.230] (0.459, 0.759) | 1.0000 |
| Social | 4223.4 | 4044.1 [-0.042] | 3548.8 [-0.160] | 0.892 | 0.854 [-0.042] | 0.750 [-0.160] (0.498, 1.082) | 1.0000 |
| Display | 2678.5 | 2477.0 [-0.075] | 2035.8 [-0.240] | 0.726 | 0.671 [-0.075] | 0.552 [-0.240] (0.297, 0.872) | 1.0000 |

### breakout_homogeneous

What it breaks, by its key: nothing. The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV_Premium | 3709.0 | 8755.2 [+1.360] | 1841.1 [-0.504] | 0.841 | 1.985 [+1.360] | 0.417 [-0.504] (0.082, 0.971) | 1.0000 |
| TV_Standard | 2695.2 | 3088.0 [+0.146] | 2683.6 [-0.004] | 0.797 | 0.913 [+0.146] | 0.793 [-0.004] (0.357, 1.426) | 1.0000 |
| TV_Remnant | 1554.4 | 2134.7 [+0.373] | 1293.4 [-0.168] | 0.789 | 1.084 [+0.373] | 0.657 [-0.168] (0.182, 1.228) | 1.0000 |
| Search | 4447.6 | 3990.4 [-0.103] | 3657.0 [-0.178] | 0.771 | 0.691 [-0.103] | 0.634 [-0.178] (0.451, 0.853) | 1.0000 |
| Social | 4116.4 | 3429.6 [-0.167] | 2579.8 [-0.373] | 0.807 | 0.673 [-0.167] | 0.506 [-0.373] (0.307, 0.742) | 1.0000 |
| Display | 2839.6 | 3227.8 [+0.137] | 2537.1 [-0.107] | 0.779 | 0.886 [+0.137] | 0.696 [-0.107] (0.337, 1.229) | 1.0000 |

### breakout_collinear

What it breaks, by its key: identifiability (independent breakout variation). The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV_Premium | 5249.7 | 0.0 [-1.000] | 2265.9 [-0.568] | 1.247 | 0.000 [-1.000] | 0.538 [-0.568] (0.016, 1.402) | 1.0000 |
| TV_Standard | 1972.2 | 5825.6 [+1.954] | 2724.6 [+0.381] | 0.605 | 1.786 [+1.954] | 0.835 [+0.381] (0.047, 1.931) | 1.0000 |
| TV_Remnant | 421.7 | 3490.7 [+7.277] | 2496.2 [+4.919] | 0.225 | 1.859 [+7.277] | 1.329 [+4.919] (0.060, 3.286) | 1.0000 |
| Search | 4427.9 | 4473.7 [+0.010] | 3680.0 [-0.169] | 0.709 | 0.716 [+0.010] | 0.589 [-0.169] (0.412, 0.825) | 1.0000 |
| Social | 3388.9 | 3100.4 [-0.085] | 2675.1 [-0.211] | 0.703 | 0.643 [-0.085] | 0.555 [-0.211] (0.362, 0.791) | 1.0000 |
| Display | 2849.2 | 2460.6 [-0.136] | 1819.6 [-0.361] | 0.760 | 0.656 [-0.136] | 0.485 [-0.361] (0.262, 0.740) | 1.0000 |

### geo_clean

What it breaks, by its key: nothing. The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 6229.1 | 6165.5 [-0.010] | 5537.1 [-0.111] | 0.221 | 0.218 [-0.010] | 0.196 [-0.111] (0.089, 0.453) | 1.0000 |
| Search | 4771.8 | 15002.5 [+2.144] | 8123.9 [+0.702] | 0.294 | 0.925 [+2.144] | 0.501 [+0.702] (0.123, 1.266) | 1.0000 |
| Social | 4459.9 | 5154.1 [+0.156] | 6164.7 [+0.382] | 0.334 | 0.386 [+0.156] | 0.462 [+0.382] (0.133, 1.314) | 1.0000 |
| Display | 3017.4 | 3646.3 [+0.208] | 6798.6 [+1.253] | 0.277 | 0.335 [+0.208] | 0.625 [+1.253] (0.116, 1.608) | 1.0000 |

### geo_heterogeneous

What it breaks, by its key: global media coefficients (per-geo effectiveness homogeneity). The bench's arm reads the controls Price.

| channel | stated contribution | observational | mmm-framework | stated ROAS | observational | mmm-framework | stated over the file's spend |
|---|---|---|---|---|---|---|---|
| TV | 7046.3 | 6952.4 [-0.013] | 6375.3 [-0.095] | 0.194 | 0.191 [-0.013] | 0.175 [-0.095] (0.075, 0.390) | 1.0000 |
| Search | 8687.5 | 6511.0 [-0.251] | 10032.9 [+0.155] | 0.405 | 0.303 [-0.251] | 0.467 [+0.155] (0.097, 1.109) | 1.0000 |
| Social | 3985.7 | 6024.7 [+0.512] | 7470.3 [+0.874] | 0.226 | 0.342 [+0.512] | 0.424 [+0.874] (0.083, 1.102) | 1.0000 |
| Display | 2778.5 | 5517.4 [+0.986] | 5627.8 [+1.025] | 0.211 | 0.418 [+0.986] | 0.427 [+1.025] (0.063, 1.261) | 1.0000 |

## The tool's own fit

mmm-framework fitted as its README's Python example fits a model (`scripts/mmm_framework_arm.py`, in the environment `scripts/mmm_framework_arm.txt` pins), as on the two shipped sets, and given the same controls as the bench's arm. Its interval is its ROI table's HDI of the contribution over the channel's spend. These are the example's settings, not those of the tool's own stress harness, which no document of its wheel states in full. Each world was fitted once; the seed does not fix the tool's numbers, which four runs on each shipped set moved by at most 0.5 % of a channel's contribution.

- 4 chains of 1000 tuning and 2000 kept draws, seed 42: 26 of 26 worlds.
- UserWarning: `init='adapt_diag'` is ignored by `nuts_sampler='numpyro'`; the external sampler uses its own initialization. (on 26 of 26 worlds)
- RuntimeWarning: invalid value encountered in scalar divide (on 26 of 26 worlds)
- adstock_misspec: ConvergenceWarning: BayesianMMM fit has NOT converged: 171 divergent transition(s). The posterior is unreliable — do not trust the intervals/ROI. Increase tune/draws/chains or reparameterize and re-fit before acting.

## Arms and worlds without a cell

- status quo: a plan, and the truth states no curve to score a plan on.
- equal split: a plan, and the truth states no curve to score a plan on.
- CHC: reads each channel from its lift tests, and the files hold none.
- myopic: reads each channel from its lift tests, and the files hold none.
- The world aurora_kitchen_sink: the generator refuses to draw it outside the tool's repository.
- The world geo_product: its file fills the product dimension, which the bench does not read.

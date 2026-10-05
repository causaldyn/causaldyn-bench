# Family 15's worlds: Meta's siMMMulator 1.1.0 running its own documented demo over four years, one
# world a seed, each written as JSON for causaldyn_bench.scorecard.simmmulator.
#
# siMMMulator is never a dependency of the bench. It runs from a scratch R library of its own, which
# holds it and what it adds (data.table, msm, expm, mvtnorm) and reads the rest from the Robyn arm's
# library, never writing there; scripts/simmmulator_worlds.lock.txt records the commit, the
# tarball's digest and every version. Each world is the demo of siMMMulator-website/docs/
# demo_code.md at that commit, at years = 4, with set.seed(seed) before its first step. The file
# holds what the bench's map reads, each day's spend, cost per impression or click and conversion
# rate for each channel and the baseline, with the simulator's constants and the inflexions its
# diminishing-returns step computes; and what the simulator itself returned, each day's impressions
# or clicks, conversions and revenue, which the bench holds its map to.
#
#   R_LIBS=LIB:ROBYN_LIB R_LIBS_USER=LIB timeout SECONDS Rscript --vanilla \
#       scripts/simmmulator_worlds.R --seeds FIRST:STOP --out DIR
#
# --seeds keeps the worlds from seed FIRST up to STOP, STOP left out, as Python's range.

suppressPackageStartupMessages({
  library(siMMMulator)
  library(dplyr)
})

COMMIT <- "da44d9afdbf3012078deba7a98f4d2999dcccf93"
YEARS <- 4
CHANNELS <- c("Facebook", "TV", "Search")
KINDS <- c("impressions", "impressions", "clicks")
DEMO <- list(
  true_cvr = c(0.001, 0.002, 0.003),
  revenue_per_conv = 1,
  base_p = 10000, trend_p = 0.5, temp_var = 2, temp_coef_mean = 100, temp_coef_sd = 500,
  error_std = 100,
  campaign_spend_mean = 329000, campaign_spend_std = 100000,
  max_min_proportion_on_each_channel = c(0.45, 0.55, 0.1, 0.15),
  true_cpm = c(2, 20, NA), true_cpc = c(NA, NA, 0.25),
  mean_noisy_cpm_cpc = c(1, 0.05, 0.01), std_noisy_cpm_cpc = c(0.01, 0.15, 0.01),
  mean_noisy_cvr = c(0, 0.0001, 0.0002), std_noisy_cvr = c(0.001, 0.002, 0.003),
  true_lambda_decay = c(0.1, 0.2, 0.3),
  alpha_saturation = c(2, 2, 2), gamma_saturation = c(0.1, 0.2, 0.3)
)

# a step's own messages and printouts, which the demo fills the console with, are dropped
quiet <- function(expr) {
  value <- NULL
  invisible(capture.output(value <- suppressMessages(expr)))
  value
}

simulate <- function(seed) {
  set.seed(seed)
  d <- DEMO
  v <- quiet(step_0_define_basic_parameters(
    years = YEARS, channels_impressions = CHANNELS[KINDS == "impressions"],
    channels_clicks = CHANNELS[KINDS == "clicks"], frequency_of_campaigns = 1,
    true_cvr = d$true_cvr, revenue_per_conv = d$revenue_per_conv, start_date = "2017/1/1"
  ))
  baseline <- quiet(step_1_create_baseline(
    my_variables = v, base_p = d$base_p, trend_p = d$trend_p, temp_var = d$temp_var,
    temp_coef_mean = d$temp_coef_mean, temp_coef_sd = d$temp_coef_sd, error_std = d$error_std
  ))
  s2 <- quiet(step_2_ads_spend(
    my_variables = v, campaign_spend_mean = d$campaign_spend_mean,
    campaign_spend_std = d$campaign_spend_std,
    max_min_proportion_on_each_channel = d$max_min_proportion_on_each_channel
  ))
  s3 <- quiet(step_3_generate_media(
    my_variables = v, df_ads_step2 = s2, true_cpm = d$true_cpm, true_cpc = d$true_cpc,
    mean_noisy_cpm_cpc = d$mean_noisy_cpm_cpc, std_noisy_cpm_cpc = d$std_noisy_cpm_cpc
  ))
  s4 <- quiet(step_4_generate_cvr(
    my_variables = v, df_ads_step3 = s3, mean_noisy_cvr = d$mean_noisy_cvr,
    std_noisy_cvr = d$std_noisy_cvr
  ))
  s5a <- quiet(step_5a_pivot_to_mmm_format(my_variables = v, df_ads_step4 = s4))
  s5b <- quiet(step_5b_decay(
    my_variables = v, df_ads_step5a_before_mmm = s5a, true_lambda_decay = d$true_lambda_decay
  ))
  s5c <- quiet(step_5c_diminishing_returns(
    my_variables = v, df_ads_step5b = s5b, alpha_saturation = d$alpha_saturation,
    gamma_saturation = d$gamma_saturation
  ))
  s6 <- quiet(step_6_calculating_conversions(my_variables = v, df_ads_step5c = s5c))
  s7 <- as.data.frame(quiet(step_7_expanded_df(
    my_variables = v, df_ads_step6 = s6, df_baseline = baseline
  )))
  s5a <- as.data.frame(s5a)
  s5b <- as.data.frame(s5b)
  # with one campaign a day, a campaign's row of a channel is that day's
  s3 <- as.data.frame(s3)
  cost <- sapply(seq_along(CHANNELS), function(c) {
    rows <- s3[s3$channel == CHANNELS[c], ]
    if (KINDS[c] == "impressions") rows$noisy_cpm else rows$noisy_cpc
  })
  kind <- ifelse(KINDS == "impressions", "imps", "clicks")
  media <- paste0("sum_n_", CHANNELS, "_", kind, "_this_day")
  # the inflexions as step_5c_diminishing_returns computes them, on each channel's adstock
  inflexions <- lapply(media, function(column) {
    x <- s5b[[paste0(column, "_adstocked")]]
    unname(round(quantile(seq(range(x)[1], range(x)[2], length.out = 100), d$gamma_saturation), 4))
  })
  list(
    seed = seed,
    simmmulator = as.character(packageVersion("siMMMulator")),
    commit = COMMIT,
    r = R.version.string,
    years = YEARS,
    channels = CHANNELS,
    kinds = KINDS,
    constants = list(
      true_cvr = d$true_cvr, revenue_per_conv = d$revenue_per_conv,
      decay = d$true_lambda_decay, alpha = d$alpha_saturation, gamma = d$gamma_saturation,
      cost_mean = c(d$true_cpm[1:2], d$true_cpc[3]) + d$mean_noisy_cpm_cpc,
      cost_sd = d$std_noisy_cpm_cpc, cvr_noise_mean = d$mean_noisy_cvr,
      cvr_noise_sd = d$std_noisy_cvr, inflexions = inflexions
    ),
    daily = list(
      spend = lapply(CHANNELS, function(name) s5a[[paste0("sum_spend_", name, "_this_day")]]),
      cost = lapply(seq_along(CHANNELS), function(c) cost[, c]),
      cvr = lapply(CHANNELS, function(name) s5a[[paste0("cvr_", name, "_this_day")]]),
      baseline = s7$baseline_revenue,
      media = lapply(media, function(column) s5a[[column]]),
      conversions = lapply(CHANNELS, function(name) s7[[paste0("conv_", name)]]),
      revenue = s7$total_revenue
    )
  )
}

args <- commandArgs(trailingOnly = TRUE)
option <- function(name) {
  at <- match(name, args)
  if (is.na(at) || at == length(args)) stop("give ", name)
  args[at + 1]
}
bounds <- as.integer(strsplit(option("--seeds"), ":", fixed = TRUE)[[1]])
out <- option("--out")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
for (seed in seq(bounds[1], bounds[2] - 1)) {
  path <- file.path(out, paste0(seed, ".json"))
  if (file.exists(path)) next
  started <- Sys.time()
  world <- simulate(seed)
  partial <- paste0(path, ".partial")
  jsonlite::write_json(world, partial, digits = I(17), auto_unbox = TRUE)
  file.rename(partial, path)
  cat(seed, format(Sys.time() - started), "\n")
}

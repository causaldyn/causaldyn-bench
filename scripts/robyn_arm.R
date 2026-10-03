# Track M v2's Robyn arm, outside the bench's environment.
#
# Meta's Robyn is an R package that brings prophet, rstan and, through reticulate, Python's
# nevergrad, so it is never a dependency of the bench: this script runs on a scratch R library and
# a throwaway Python environment, reads the worlds causaldyn_bench.budget_regret's `export` wrote,
# and writes each world's plan as JSON, which the bench scores on the world's own channels.
#
# Each world is fitted as a practitioner fits Robyn 3.12.1 by its demo (demo/demo.R):
#
# * weeks dated Mondays from 2023-01-02, as the other arms'; the dependent variable the world's
#   weekly sales as revenue; each channel's spend as its paid media (no exposure metric);
#   prophet's trend and season with no holidays; the observed promotion indicator and price as
#   context variables; the modelling window the whole history;
# * geometric adstock, each channel searched over one of two settings of the demo's ranges, which
#   --ranges names. `genre`: the demo's ranges for the demo channel each world channel stands for,
#   pla a search channel (search_clicks_P's: alphas 0.5-3, gammas 0.3-1, thetas 0-0.3), meta a
#   social one (facebook_I's, the same) and tv tv_S's (alphas 0.5-1, gammas 0.3-1, thetas 0.3-0.8),
#   the thetas its rule of thumb for digital and TV decay. `union`: every channel over the union of
#   the demo's ranges, thetas 0-0.8 (the rule of thumb joined over digital, OOH, print, radio and
#   TV), alphas 0.5-3 and gammas 0.3-1. CHOICE: the drawn worlds give every channel, whatever its
#   genre, a retention drawn uniformly from 0.2 to 0.7, four fifths of which the digital rule of
#   thumb rules out for pla and meta and the union holds; the scored run takes the setting a pilot
#   of both found better for Robyn, pre-registered in causaldyn_bench.external_arms. And the demo's
#   train_size 0.5-0.8, inert with ts_validation off, as in the demo;
# * every geo test the bench kept as one calibration_input row: the test's four dark weeks,
#   liftStartDate the Monday of the first and liftEndDate the Monday of the last. CHOICE: Robyn
#   scales its prediction by the lift's days over the days its decomposition covers, Monday to
#   Monday on weekly data, so ending the lift on the last dark week's Monday keeps that scale at
#   one where its Sunday would inflate the prediction by 27/21. liftAbs is the sales the test lost
#   over its dark and cooldown weeks (-delta_y times the dark weeks), spend the spend withheld
#   (the channel's spend over the dark weeks, as Robyn checks it), confidence one minus the
#   two-sided p-value of delta_y / sigma (the demo's "1 - pvalue"; Robyn only warns on it), metric
#   the dependent variable. CHOICE: calibration_scope "total", against the demo's advice of
#   "immediate" for experiments: Robyn's "immediate" counts what the window's spend returns inside
#   the window, while the lift also counts what it returns in the cooldown weeks after; "total"
#   counts everything the channel returns inside the window, which, at the steady spend around a
#   test, equals the whole return of the window's spend, since the carryover flowing in from
#   before the window stands in for the carryover flowing out after it;
# * robyn_run at the demo's settings: 2000 iterations, TwoPointsDE, ts_validation and
#   add_penalty_factor off, seeded by the world's seed; cores 4. CHOICE: the demo takes every core
#   but one, and nevergrad asks for as many candidates at a time as there are cores, so the count
#   is part of the result and is fixed here. 10 trials, not the demo's 5: the demo runs 5 on its
#   dummy data uncalibrated, and Robyn's own check asks a calibrated model for at least 10;
# * model selection fixed before any score was read, Robyn's own: robyn_outputs with its
#   defaults (Pareto fronts "auto", calibration_constraint 0.1, clusters on), then among the
#   clusters' top models the one with the lowest error score, Robyn's normalised distance of
#   NRMSE, DECOMP.RSSD and the calibration MAPE from zero; if the clustering fails, the Pareto
#   model with the lowest such score, recorded as the route taken;
# * the quarter planned by robyn_allocator's "max_response" scenario at the world's budget over
#   date_range "last_13", so Robyn spreads the budget over 13 weeks. CHOICE: Robyn bounds a
#   channel by multiples of its mean spend over date_range, and the bench's box is 0.5 to 2 times
#   the last 52 weeks' mean, so each channel's multiples are the box over its last 13 weeks' mean;
#   the record carries how far Robyn's absolute bounds fell from the box. keep_zero_coefs on
#   (CHOICE): Robyn otherwise drops a channel whose coefficient is zero and plans it nothing,
#   outside the box. As documented, Robyn plans one steady week: each channel's response is its
#   Hill curve at the planned spend plus the window's mean carryover, and `weekly` is that week's
#   spend (optmSpendUnit).
#
# Version 2. A scorecard export (causaldyn_bench.scorecard.observe, `version` 2) names its controls
# and holds the quarter's status quo. It is fitted and planned as above, the controls read by name
# as context variables, with 5 trials where no lift test calibrates the model, as the demo runs, and
# 10 where one does, as Robyn's own check asks. Robyn's geometric adstock carries each week's spend
# over every later week of the series, so it takes no kernel length, and the record says so. Each
# record carries, beside the plan:
#
# * `gain`: the plan's response less the status quo's over the quarter, as the allocator counts a
#   response: 13 steady weeks, each channel's Hill curve at its weekly spend plus the window's mean
#   carryover, times its coefficient. The arm checks that this reproduces the allocator's own
#   response at the plan. As documented, it counts nothing the quarter's spend returns after it;
# * `flags`: the selected model's NRMSE, DECOMP.RSSD, R-squared and, where lift tests calibrate it,
#   its calibration MAPE, and robyn_run's convergence messages, as Robyn reports them;
# * `response`: the selected model, one draw, mapped onto chc.response
#   (causaldyn_bench.scorecard.mapping): the kernel GeometricAdstock over every week of the history,
#   unnormalised, as Robyn's recursion; the curve Hill at the model's inflexion with its alpha as
#   the slope; the coefficient the ridge's. Beside them, the selected model's own decomposition of
#   the history (xDecompVecCollect), which the bench checks them against;
# * `settings`, `versions` and `cost`, the wall and CPU seconds of the whole arm;
# * `forecast` and `gain_interval` are n/a: Robyn forecasts no week after its modelling window, and
#   plans on one selected model, a point estimate, and the record says so.
#
# --smoke runs 400 iterations and 4 trials to check the plumbing; its records say they are a smoke
# run, not a result. A version-1 export is fitted as before, by the same code, in full.
#
#   R_LIBS=LIB R_LIBS_USER=LIB RETICULATE_PYTHON=VENV/bin/python \
#       timeout SECONDS Rscript --vanilla scripts/robyn_arm.R --worlds DIR --out DIR \
#       --ranges genre|union [--seeds FIRST:STOP] [--shard K/N] [--smoke]
#
# --seeds keeps the worlds from seed FIRST up to STOP, STOP left out, as Python's range, so a run
# can fit a part of the worlds exported.
#
# scripts/robyn_arm.lock.txt records R, every package in the scratch library, the Python
# environment's pins and the commands that rebuild them; each record names the versions it ran on
# and echoes the world's digest, so the bench scores a plan only on the data it was fitted to.

suppressPackageStartupMessages(library(Robyn))

FIRST_MONDAY <- as.Date("2023-01-02")
ITERATIONS <- 2000L
TRIALS <- 10L
CORES <- 4L
RANGES <- list(
  genre = list(
    pla = list(alphas = c(0.5, 3), gammas = c(0.3, 1), thetas = c(0, 0.3)), # search_clicks_P
    meta = list(alphas = c(0.5, 3), gammas = c(0.3, 1), thetas = c(0, 0.3)), # facebook_I
    tv = list(alphas = c(0.5, 1), gammas = c(0.3, 1), thetas = c(0.3, 0.8)) # tv_S
  ),
  union = list(alphas = c(0.5, 3), gammas = c(0.3, 1), thetas = c(0, 0.8)) # every channel's
)
PACKAGES <- c(
  "Robyn", "prophet", "rstan", "StanHeaders", "glmnet", "nloptr", "reticulate", "doRNG",
  "doParallel", "foreach", "dplyr", "jsonlite"
)
SECOND <- 2L # the scorecard's export version
UNCALIBRATED_TRIALS <- 5L
# a calibrated run keeps the tenth of its models nearest the lifts, and Robyn's automatic Pareto
# fronts need 100 of them
SMOKE <- list(iterations = 400L, trials = 4L)
AGREE <- 1e-9 # how near the arm's response at the plan lies to the allocator's own, relative to it
NOT_GIVEN <- list(
  forecast = "Robyn forecasts no week after its modelling window",
  gain_interval =
    "Robyn plans on one selected model, a point estimate, with no interval of its response"
)
NO_KERNEL_LENGTH <- paste(
  "Robyn's geometric adstock carries each week's spend over every later week of the series,",
  "so it takes no kernel length"
)
GAIN_WINDOW <- paste(
  "13 steady weeks of the allocator's response, the window's mean carryover held:",
  "nothing after the quarter"
)

read_world <- function(path) {
  numpy <- reticulate::import("numpy", convert = FALSE)
  saved <- numpy$load(path)
  value <- function(key) reticulate::py_to_r(saved$`__getitem__`(key))
  vector <- function(key) as.vector(value(key))
  world <- lapply(
    c(
      seed = "seed", digest = "digest", channels = "channels", sales = "sales",
      promotion = "promotion", price = "price", planned = "planned", budget = "budget",
      lower = "lower", upper = "upper", lift_channel = "lift_channel", lift_x = "lift_x",
      lift_delta_y = "lift_delta_y", lift_sigma = "lift_sigma", lift_start = "lift_start",
      lift_weeks = "lift_weeks", lift_dropped = "lift_dropped"
    ),
    vector
  )
  world$spend <- unclass(value("spend"))
  saved$close()
  world
}

calibration <- function(world, dates) {
  if (length(world$lift_channel) == 0) {
    return(NULL)
  }
  last <- world$lift_start + world$lift_weeks - 1
  data.frame(
    channel = world$lift_channel,
    liftStartDate = dates[world$lift_start],
    liftEndDate = dates[last],
    liftAbs = -world$lift_delta_y * world$lift_weeks,
    spend = world$lift_x * world$lift_weeks,
    confidence = 1 - 2 * pnorm(-abs(world$lift_delta_y) / world$lift_sigma),
    metric = "sales",
    calibration_scope = "total"
  )
}

hyperparameters <- function(channels, setting) {
  ranges <- list()
  for (name in channels) {
    own <- if (setting == "union") RANGES$union else RANGES$genre[[name]]
    if (is.null(own)) stop("no ", setting, " ranges for channel ", name)
    for (kind in names(own)) ranges[[paste0(name, "_", kind)]] <- own[[kind]]
  }
  ranges$train_size <- c(0.5, 0.8)
  ranges
}

# Robyn's own ranking: the clusters' top models by error score, or the Pareto models if the
# clustering failed
select_model <- function(OutputCollect) {
  models <- OutputCollect$clusters$models
  if (is.data.frame(models) && nrow(models) > 0) {
    best <- models[which.min(models$error_score), ]
    return(list(solID = best$solID, route = "clusters", error_score = best$error_score))
  }
  pareto <- OutputCollect$resultHypParam
  pareto <- pareto[pareto$solID %in% OutputCollect$allSolutions, ]
  scores <- Robyn:::errors_scores(pareto, ts_validation = FALSE)
  at <- which.min(scores)
  list(solID = pareto$solID[at], route = "pareto", error_score = scores[at])
}

timed <- function(expr) {
  began <- proc.time()[["elapsed"]]
  value <- expr
  list(value = value, seconds = proc.time()[["elapsed"]] - began)
}

plan_world <- function(path, setting) {
  world <- read_world(path)
  seed <- as.integer(world$seed)
  set.seed(seed)
  channels <- world$channels
  weeks <- length(world$sales)
  dates <- FIRST_MONDAY + 7L * (seq_len(weeks) - 1L)
  frame <- data.frame(
    DATE = dates, sales = world$sales, promotion = world$promotion, price = world$price
  )
  for (k in seq_along(channels)) frame[[channels[k]]] <- world$spend[, k]
  lifts <- calibration(world, dates)

  InputCollect <- robyn_inputs(
    dt_input = frame,
    dt_holidays = Robyn::dt_prophet_holidays,
    date_var = "DATE",
    dep_var = "sales",
    dep_var_type = "revenue",
    prophet_vars = c("trend", "season"),
    context_vars = c("promotion", "price"),
    paid_media_spends = channels,
    window_start = as.character(dates[1]),
    window_end = as.character(dates[weeks]),
    adstock = "geometric",
    hyperparameters = hyperparameters(channels, setting),
    calibration_input = lifts
  )
  fit <- timed({
    # quiet stays at its default, as in the demo: 3.12.1's robyn_run(quiet = TRUE) stops at the end
    # of every trial, closing a progress bar it opens only when not quiet
    OutputModels <- robyn_run(
      InputCollect = InputCollect,
      cores = CORES,
      iterations = ITERATIONS,
      trials = TRIALS,
      ts_validation = FALSE,
      add_penalty_factor = FALSE,
      seed = seed
    )
    robyn_outputs(
      InputCollect, OutputModels,
      pareto_fronts = "auto",
      csv_out = NULL,
      clusters = TRUE,
      plot_pareto = FALSE,
      plot_folder = tempdir(),
      export = FALSE,
      quiet = TRUE
    )
  })
  OutputCollect <- fit$value
  chosen <- select_model(OutputCollect)

  recent <- colMeans(world$spend[(weeks - 12):weeks, , drop = FALSE])
  allocation <- timed(robyn_allocator(
    InputCollect = InputCollect,
    OutputCollect = OutputCollect,
    select_model = chosen$solID,
    scenario = "max_response",
    total_budget = world$budget,
    date_range = "last_13",
    channel_constr_low = world$lower / recent,
    channel_constr_up = world$upper / recent,
    keep_zero_coefs = TRUE,
    plots = FALSE,
    export = FALSE,
    quiet = TRUE
  ))
  planned <- allocation$value$dt_optimOut
  at <- match(channels, planned$channels)
  weekly <- planned$optmSpendUnit[at]
  box <- c(planned$constr_low_abs[at] - world$lower, planned$constr_up_abs[at] - world$upper)

  model <- OutputCollect$resultHypParam[OutputCollect$resultHypParam$solID == chosen$solID, ]
  decomposition <- OutputCollect$xDecompAgg[OutputCollect$xDecompAgg$solID == chosen$solID, ]
  per_channel <- lapply(setNames(channels, channels), function(name) {
    row <- decomposition[decomposition$rn == name, ]
    list(
      theta = model[[paste0(name, "_thetas")]],
      alpha = model[[paste0(name, "_alphas")]],
      gamma = model[[paste0(name, "_gammas")]],
      coef = row$coef,
      roi_total = row$roi_total
    )
  })
  lift_fit <- NULL
  if (!is.null(OutputCollect$resultCalibration)) {
    calibrated <- OutputCollect$resultCalibration
    calibrated <- calibrated[calibrated$solID == chosen$solID, ]
    lift_fit <- lapply(seq_len(nrow(calibrated)), function(i) {
      list(
        channel = calibrated$rn[i],
        start = as.character(calibrated$liftStart[i]),
        lift = calibrated$liftAbs[i],
        predicted = calibrated$decompAbsScaled[i],
        mape = calibrated$mape_lift[i]
      )
    })
  }
  solver <- allocation$value$nlsMod
  list(
    seed = seed,
    digest = world$digest,
    weekly = weekly,
    quarter_spend = weekly * world$planned,
    fit_seconds = fit$seconds,
    plan_seconds = allocation$seconds,
    selected = chosen$solID,
    selection = chosen$route,
    error_score = chosen$error_score,
    clusters = if (is.null(OutputCollect$clusters$n_clusters)) NULL else OutputCollect$clusters$n_clusters,
    pareto_models = length(OutputCollect$allSolutions),
    pareto_fronts = OutputCollect$pareto_fronts,
    nrmse = model$nrmse,
    decomp_rssd = model$decomp.rssd,
    calibration_mape = model$mape,
    rsq_train = model$rsq_train,
    lambda = model$lambda,
    channels = per_channel,
    lift_fit = lift_fit,
    convergence = I(OutputModels$convergence$conv_msg), # a list in the JSON, one message or many
    allocator = list(
      status = solver$status,
      message = solver$message,
      iterations = solver$iterations,
      box_error = max(abs(box)),
      skipped_coef0 = allocation$value$skipped_coef0,
      initial_response = planned$initResponseUnitTotal[1],
      planned_response = planned$optmResponseUnitTotal[1]
    ),
    iterations = ITERATIONS,
    trials = TRIALS,
    cores = CORES,
    ranges = RANGES[[setting]],
    lift_rows = length(world$lift_channel),
    lift_dropped = world$lift_dropped,
    error = NULL
  )
}

exported_version <- function(path) {
  numpy <- reticulate::import("numpy", convert = FALSE)
  saved <- numpy$load(path)
  files <- reticulate::py_to_r(saved$files)
  version <- if ("version" %in% files) {
    as.integer(reticulate::py_to_r(saved$`__getitem__`("version")))
  } else {
    1L
  }
  saved$close()
  if (!version %in% c(1L, SECOND)) stop(path, " is a version-", version, " export")
  version
}

read_second <- function(path) {
  numpy <- reticulate::import("numpy", convert = FALSE)
  saved <- numpy$load(path)
  value <- function(key) reticulate::py_to_r(saved$`__getitem__`(key))
  vector <- function(key) as.vector(value(key))
  world <- lapply(
    c(
      family = "family", environment = "environment", seed = "seed", k = "k", digest = "digest",
      channels = "channels", sales = "sales", control_names = "control_names",
      kernel_length = "kernel_length", planned = "planned", budget = "budget", lower = "lower",
      upper = "upper", status_quo = "status_quo", lift_channel = "lift_channel",
      lift_x = "lift_x", lift_delta_y = "lift_delta_y", lift_sigma = "lift_sigma",
      lift_start = "lift_start", lift_weeks = "lift_weeks", lift_dropped = "lift_dropped"
    ),
    vector
  )
  world$spend <- unclass(value("spend"))
  world$controls <- unclass(value("controls"))
  saved$close()
  world
}

# Each channel's response a week, as the allocator computes it: the Hill curve at the week's spend
# plus the window's mean carryover, times the coefficient
weekly_response <- function(spend, channels, allocation, model, coefficients) {
  carried <- as.data.frame(allocation$mainPoints)
  carried <- carried[carried$type == "Carryover", ]
  vapply(seq_along(channels), function(c) {
    name <- channels[c]
    Robyn:::fx_objective(
      x = spend[c],
      coeff = coefficients$coef[coefficients$rn == name],
      alpha = model[[paste0(name, "_alphas")]],
      inflexion = model[[paste0(name, "_inflexion")]],
      x_hist_carryover = carried$spend_point[carried$channel == name],
      get_sum = FALSE
    )
  }, 0)
}

# The selected model as chc.response reads it, one draw, beside its own decomposition of the history
mapped_response <- function(OutputCollect, solID, channels, weeks) {
  model <- OutputCollect$resultHypParam[OutputCollect$resultHypParam$solID == solID, ]
  coefficients <- OutputCollect$xDecompAgg[OutputCollect$xDecompAgg$solID == solID, ]
  decomposed <- OutputCollect$xDecompVecCollect
  decomposed <- decomposed[decomposed$solID == solID, ]
  decomposed <- decomposed[order(decomposed$ds), ]
  if (nrow(decomposed) != weeks) {
    stop("the decomposition holds ", nrow(decomposed), " weeks, not ", weeks)
  }
  draw <- function(value) I(as.numeric(value)) # a list over the one draw
  named <- setNames(channels, channels)
  list(
    kernel = "GeometricAdstock",
    length = weeks,
    normalized = FALSE,
    curve = "Hill",
    draws = 1L,
    parameters = lapply(named, function(name) {
      list(
        retention = draw(model[[paste0(name, "_thetas")]]),
        scale = draw(model[[paste0(name, "_inflexion")]]),
        slope = draw(model[[paste0(name, "_alphas")]]),
        coefficient = draw(coefficients$coef[coefficients$rn == name])
      )
    }),
    decomposition = list(
      total = lapply(named, function(name) draw(sum(decomposed[[name]]))),
      held = I(0L),
      weekly = lapply(named, function(name) list(as.numeric(decomposed[[name]])))
    )
  )
}

plan_second <- function(path, setting, smoke) {
  world <- read_second(path)
  seed <- as.integer(world$seed)
  set.seed(seed)
  channels <- world$channels
  weeks <- length(world$sales)
  dates <- FIRST_MONDAY + 7L * (seq_len(weeks) - 1L)
  controls <- as.character(world$control_names)
  frame <- data.frame(DATE = dates, sales = world$sales)
  for (j in seq_along(controls)) frame[[controls[j]]] <- world$controls[, j]
  for (k in seq_along(channels)) frame[[channels[k]]] <- world$spend[, k]
  lifts <- calibration(world, dates)
  iterations <- if (smoke) SMOKE$iterations else ITERATIONS
  trials <- if (smoke) SMOKE$trials else if (is.null(lifts)) UNCALIBRATED_TRIALS else TRIALS

  InputCollect <- robyn_inputs(
    dt_input = frame,
    dt_holidays = Robyn::dt_prophet_holidays,
    date_var = "DATE",
    dep_var = "sales",
    dep_var_type = "revenue",
    prophet_vars = c("trend", "season"),
    context_vars = if (length(controls) > 0) controls else NULL,
    paid_media_spends = channels,
    window_start = as.character(dates[1]),
    window_end = as.character(dates[weeks]),
    adstock = "geometric",
    hyperparameters = hyperparameters(channels, setting),
    calibration_input = lifts
  )
  # quiet stays at its default, as in version 1
  OutputModels <- robyn_run(
    InputCollect = InputCollect,
    cores = CORES,
    iterations = iterations,
    trials = trials,
    ts_validation = FALSE,
    add_penalty_factor = FALSE,
    seed = seed
  )
  OutputCollect <- robyn_outputs(
    InputCollect, OutputModels,
    pareto_fronts = "auto",
    csv_out = NULL,
    clusters = TRUE,
    plot_pareto = FALSE,
    plot_folder = tempdir(),
    export = FALSE,
    quiet = TRUE
  )
  chosen <- select_model(OutputCollect)

  recent <- colMeans(world$spend[(weeks - 12):weeks, , drop = FALSE])
  allocation <- robyn_allocator(
    InputCollect = InputCollect,
    OutputCollect = OutputCollect,
    select_model = chosen$solID,
    scenario = "max_response",
    total_budget = world$budget,
    date_range = "last_13",
    channel_constr_low = world$lower / recent,
    channel_constr_up = world$upper / recent,
    keep_zero_coefs = TRUE,
    plots = FALSE,
    export = FALSE,
    quiet = TRUE
  )
  planned <- allocation$dt_optimOut
  at <- match(channels, planned$channels)
  weekly <- planned$optmSpendUnit[at]
  box <- c(planned$constr_low_abs[at] - world$lower, planned$constr_up_abs[at] - world$upper)

  model <- OutputCollect$resultHypParam[OutputCollect$resultHypParam$solID == chosen$solID, ]
  decomposition <- OutputCollect$xDecompAgg[OutputCollect$xDecompAgg$solID == chosen$solID, ]
  at_plan <- weekly_response(weekly, channels, allocation, model, decomposition)
  own <- planned$optmResponseUnit[at]
  if (max(abs(at_plan - own)) > AGREE * max(abs(own))) {
    stop("the response at the plan, ", sum(at_plan), ", is not the allocator's own, ", sum(own))
  }
  at_status_quo <- weekly_response(world$status_quo, channels, allocation, model, decomposition)
  per_channel <- lapply(setNames(channels, channels), function(name) {
    row <- decomposition[decomposition$rn == name, ]
    list(
      theta = model[[paste0(name, "_thetas")]],
      alpha = model[[paste0(name, "_alphas")]],
      gamma = model[[paste0(name, "_gammas")]],
      inflexion = model[[paste0(name, "_inflexion")]],
      coef = row$coef,
      roi_total = row$roi_total
    )
  })
  lift_fit <- NULL
  if (!is.null(OutputCollect$resultCalibration)) {
    calibrated <- OutputCollect$resultCalibration
    calibrated <- calibrated[calibrated$solID == chosen$solID, ]
    lift_fit <- lapply(seq_len(nrow(calibrated)), function(i) {
      list(
        channel = calibrated$rn[i],
        start = as.character(calibrated$liftStart[i]),
        lift = calibrated$liftAbs[i],
        predicted = calibrated$decompAbsScaled[i],
        mape = calibrated$mape_lift[i]
      )
    })
  }
  solver <- allocation$nlsMod
  list(
    version = SECOND,
    seed = seed,
    digest = world$digest,
    family = as.integer(world$family),
    environment = world$environment,
    k = as.integer(world$k),
    weekly = weekly,
    quarter_spend = weekly * world$planned,
    forecast = NULL,
    gain = world$planned * (sum(at_plan) - sum(at_status_quo)),
    gain_interval = NULL,
    flags = list(
      nrmse = model$nrmse,
      decomp_rssd = model$decomp.rssd,
      rsq_train = model$rsq_train,
      calibration_mape = if (is.null(lifts)) NULL else model$mape,
      convergence = I(OutputModels$convergence$conv_msg)
    ),
    response = mapped_response(OutputCollect, chosen$solID, channels, weeks),
    selected = chosen$solID,
    selection = chosen$route,
    error_score = chosen$error_score,
    clusters = if (is.null(OutputCollect$clusters$n_clusters)) NULL else OutputCollect$clusters$n_clusters,
    pareto_models = length(OutputCollect$allSolutions),
    pareto_fronts = OutputCollect$pareto_fronts,
    lambda = model$lambda,
    channels = per_channel,
    lift_fit = lift_fit,
    allocator = list(
      status = solver$status,
      message = solver$message,
      iterations = solver$iterations,
      box_error = max(abs(box)),
      skipped_coef0 = allocation$skipped_coef0,
      initial_response = planned$initResponseUnitTotal[1],
      planned_response = planned$optmResponseUnitTotal[1],
      status_quo_response = sum(at_status_quo)
    ),
    settings = list(
      ranges = setting,
      hyperparameters = RANGES[[setting]],
      iterations = iterations,
      trials = trials,
      cores = CORES,
      context_vars = I(controls),
      kernel_length = NULL,
      kernel = NO_KERNEL_LENGTH,
      calibration_scope = "total",
      date_range = "last_13",
      gain_window = GAIN_WINDOW,
      smoke = smoke
    ),
    `n/a` = NOT_GIVEN,
    lift_rows = length(world$lift_channel),
    lift_dropped = world$lift_dropped,
    error = NULL
  )
}

versions <- function() {
  python <- reticulate::py_config()
  list(
    r = R.version.string,
    packages = lapply(setNames(PACKAGES, PACKAGES), function(p) as.character(packageVersion(p))),
    python = python$version_string,
    nevergrad = reticulate::import("nevergrad")$`__version__`,
    numpy = reticulate::import("numpy")$`__version__`
  )
}

option <- function(args, name, default = NULL) {
  at <- match(name, args)
  if (is.na(at)) {
    if (is.null(default)) stop("missing ", name)
    return(default)
  }
  args[at + 1L]
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  worlds <- option(args, "--worlds")
  out <- option(args, "--out")
  shard <- as.integer(strsplit(option(args, "--shard", "0/1"), "/", fixed = TRUE)[[1]])
  setting <- option(args, "--ranges")
  if (!setting %in% names(RANGES)) stop("--ranges takes genre or union, not ", setting)
  smoke <- "--smoke" %in% args
  library_path <- Sys.getenv("R_LIBS_USER")
  if (!nzchar(library_path) || Sys.getenv("R_LIBS") != library_path ||
    normalizePath(.libPaths()[1]) != normalizePath(library_path) ||
    dirname(find.package("Robyn")) != normalizePath(library_path)) {
    stop("R_LIBS and R_LIBS_USER must both name the scratch library Robyn is loaded from")
  }
  if (!nzchar(Sys.getenv("RETICULATE_PYTHON"))) stop("RETICULATE_PYTHON must name the venv's python")
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  paths <- sort(Sys.glob(file.path(worlds, "world_*.npz")), method = "radix")
  seeds <- option(args, "--seeds", "")
  if (nzchar(seeds)) {
    span <- as.integer(strsplit(seeds, ":", fixed = TRUE)[[1]])
    if (length(span) != 2L || anyNA(span)) stop("--seeds takes FIRST:STOP, not ", seeds)
    seed <- as.integer(sub("^world_([0-9]+)\\.npz$", "\\1", basename(paths)))
    paths <- paths[seed >= span[1] & seed < span[2]]
  }
  paths <- paths[(seq_along(paths) - 1L) %% shard[2] == shard[1]]
  tools <- versions()
  for (path in paths) {
    stem <- sub("\\.npz$", "", basename(path))
    target <- file.path(out, paste0(stem, ".json"))
    if (file.exists(target)) next
    second <- exported_version(path) == SECOND
    if (smoke && !second) stop(path, " is a version-1 export, fitted as version 1 fitted it")
    began <- proc.time()
    calls <- NULL
    seen <- character()
    record <- tryCatch(
      withCallingHandlers(
        if (second) plan_second(path, setting, smoke) else plan_world(path, setting),
        error = function(e) calls <<- sys.calls(),
        warning = function(w) {
          seen <<- union(seen, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) { # a world that breaks the arm is recorded, not fatal
        world <- if (second) read_second(path) else read_world(path)
        failed <- list(
          seed = as.integer(world$seed),
          digest = world$digest,
          error = paste0(class(e)[1], ": ", conditionMessage(e)),
          traceback = paste(
            vapply(calls, function(call) paste(deparse(call, nlines = 1L), collapse = ""), ""),
            collapse = "\n"
          )
        )
        if (!second) {
          return(failed)
        }
        c(list(version = SECOND), failed, list(settings = list(ranges = setting, smoke = smoke)))
      }
    )
    record$warnings <- seen
    spent <- proc.time() - began
    if (second) {
      record$cost <- list(
        wall_seconds = spent[["elapsed"]],
        cpu_seconds = sum(spent[c("user.self", "sys.self", "user.child", "sys.child")])
      )
      if (smoke) record$smoke <- "smoke, not a result"
    } else {
      record$seconds <- spent[["elapsed"]]
    }
    record$versions <- tools
    jsonlite::write_json(
      record, target,
      auto_unbox = TRUE, digits = NA, null = "null", na = "null"
    )
    cat(stem, spent[["elapsed"]], if (is.null(record$error)) "NA" else record$error, "\n")
  }
}

if (sys.nframe() == 0L) main()

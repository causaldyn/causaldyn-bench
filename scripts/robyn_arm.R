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
# * geometric adstock, and the demo's example hyperparameter ranges for the demo channel each
#   world channel stands for. CHOICE: pla is a search channel, so it takes search_clicks_P's
#   ranges (alphas 0.5-3, gammas 0.3-1, thetas 0-0.3); meta is social, so facebook_I's (the same);
#   tv takes tv_S's (alphas 0.5-1, gammas 0.3-1, thetas 0.3-0.8), which are the demo's rule of
#   thumb for digital and TV decay; and the demo's train_size 0.5-0.8, inert with ts_validation
#   off, as in the demo;
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
# * robyn_run at the demo's settings: 2000 iterations, 5 trials, TwoPointsDE, ts_validation and
#   add_penalty_factor off, seeded by the world's seed; cores 4. CHOICE: the demo takes every core
#   but one, and nevergrad asks for as many candidates at a time as there are cores, so the count
#   is part of the result and is fixed here;
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
#   R_LIBS=LIB R_LIBS_USER=LIB RETICULATE_PYTHON=VENV/bin/python \
#       timeout SECONDS Rscript --vanilla scripts/robyn_arm.R --worlds DIR --out DIR [--shard K/N]
#
# scripts/robyn_arm.lock.txt records R, every package in the scratch library, the Python
# environment's pins and the commands that rebuild them; each record names the versions it ran on
# and echoes the world's digest, so the bench scores a plan only on the data it was fitted to.

suppressPackageStartupMessages(library(Robyn))

FIRST_MONDAY <- as.Date("2023-01-02")
ITERATIONS <- 2000L
TRIALS <- 5L
CORES <- 4L
DEMO_RANGES <- list(
  pla = list(alphas = c(0.5, 3), gammas = c(0.3, 1), thetas = c(0, 0.3)), # search_clicks_P
  meta = list(alphas = c(0.5, 3), gammas = c(0.3, 1), thetas = c(0, 0.3)), # facebook_I
  tv = list(alphas = c(0.5, 1), gammas = c(0.3, 1), thetas = c(0.3, 0.8)) # tv_S
)
PACKAGES <- c(
  "Robyn", "prophet", "rstan", "StanHeaders", "glmnet", "nloptr", "reticulate", "doRNG",
  "doParallel", "foreach", "dplyr", "jsonlite"
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

hyperparameters <- function(channels) {
  unknown <- setdiff(channels, names(DEMO_RANGES))
  if (length(unknown)) stop("no demo ranges for channel(s): ", paste(unknown, collapse = ", "))
  ranges <- list()
  for (name in channels) {
    for (kind in c("alphas", "gammas", "thetas")) {
      ranges[[paste0(name, "_", kind)]] <- DEMO_RANGES[[name]][[kind]]
    }
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

plan_world <- function(path) {
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
    hyperparameters = hyperparameters(channels),
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
    convergence = OutputModels$convergence$conv_msg,
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
  library_path <- Sys.getenv("R_LIBS_USER")
  if (!nzchar(library_path) || Sys.getenv("R_LIBS") != library_path ||
    normalizePath(.libPaths()[1]) != normalizePath(library_path) ||
    dirname(find.package("Robyn")) != normalizePath(library_path)) {
    stop("R_LIBS and R_LIBS_USER must both name the scratch library Robyn is loaded from")
  }
  if (!nzchar(Sys.getenv("RETICULATE_PYTHON"))) stop("RETICULATE_PYTHON must name the venv's python")
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  paths <- sort(Sys.glob(file.path(worlds, "world_*.npz")), method = "radix")
  paths <- paths[(seq_along(paths) - 1L) %% shard[2] == shard[1]]
  tools <- versions()
  for (path in paths) {
    stem <- sub("\\.npz$", "", basename(path))
    target <- file.path(out, paste0(stem, ".json"))
    if (file.exists(target)) next
    began <- proc.time()[["elapsed"]]
    calls <- NULL
    seen <- character()
    record <- tryCatch(
      withCallingHandlers(
        plan_world(path),
        error = function(e) calls <<- sys.calls(),
        warning = function(w) {
          seen <<- union(seen, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) { # a world that breaks the arm is recorded, not fatal
        world <- read_world(path)
        list(
          seed = as.integer(world$seed),
          digest = world$digest,
          error = paste0(class(e)[1], ": ", conditionMessage(e)),
          traceback = paste(
            vapply(calls, function(call) paste(deparse(call, nlines = 1L), collapse = ""), ""),
            collapse = "\n"
          )
        )
      }
    )
    record$warnings <- seen
    record$seconds <- proc.time()[["elapsed"]] - began
    record$versions <- tools
    jsonlite::write_json(
      record, target,
      auto_unbox = TRUE, digits = NA, null = "null", na = "null"
    )
    cat(stem, record$seconds, if (is.null(record$error)) "NA" else record$error, "\n")
  }
}

if (sys.nframe() == 0L) main()

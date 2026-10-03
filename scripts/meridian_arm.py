"""Track M v2's Meridian arm, outside the bench's environment.

Google's Meridian brings JAX, TensorFlow Probability and TensorFlow, so it is never a dependency of
the bench: this script runs in a throwaway environment, reads the worlds
:func:`causaldyn_bench.budget_regret.export` wrote, and writes each world's plan as JSON, which the
bench scores on the world's own channels.

Each world is fitted as a practitioner fits Meridian 2.1.0, on its default JAX backend:

* a national model (the frame has no geo column), revenue KPI the world's weekly sales, each
  channel's spend as both its media units and its spend, the observed promotion indicator and price
  as controls;
* ``ModelSpec`` at its defaults, geometric adstock before Hill saturation, with ``max_lag`` the
  world's kernel length (6) and automatic knot selection on, as Meridian's Getting Started demo
  sets it; CHOICE: without it a national model has one knot, a baseline with no time effect, and
  the world's sales carry a seasonality of about a quarter of their level;
* every prior at Meridian's default but the channels' ROI, which is how Meridian takes experiments
  (``roi_m``). CHOICE: a test's ROI is ``delta_y / delta_x``, sales lost per euro withheld, with
  standard error ``sigma / |delta_x|``; a channel's rows are pooled by inverse variance and its
  prior is the LogNormal of that mean and standard error (Meridian's
  ``lognormal_dist_from_mean_std``). The row's lift counts the cooldown's carryover, so it is the
  whole return of the spend withheld, the ROI Meridian's prior is on. A channel whose every test
  was dropped (its sales did not fall) keeps Meridian's default ``LogNormal(0.2, 0.9)``. The ROI
  prior covers the whole history (``roi_calibration`` unset), as Meridian's documentation
  recommends. Not used: Meridian 2.1's ``CalibrationBuilder``, whose duration adjustment scales an
  experiment's ROI up by the share of the adstock its window leaves out, which these lifts already
  count, and whose recency and spend adjustments widen a test's error by the weeks since it ran and
  by how far its spend sat from the channel's mean;
* sampling as the Getting Started demo does: 500 prior draws, then NUTS with 10 chains of 1000
  adaptation, 500 burn-in and 1000 kept draws, seeded by the world's seed. Recorded: the model
  reviewer's health checks (Meridian's R-hat among them), and ArviZ's largest R-hat and smallest
  bulk ESS over every posterior variable;
* the quarter planned by ``BudgetOptimizer.optimize`` at the world's budget, each channel held to
  the bench's box: the spend constraints are ``1 - lower / mean`` and ``upper / mean - 1`` around
  the status quo, last year's mean. CHOICE: Meridian plans future weeks through ``new_data``, so
  the data given to the optimiser is the history and then the quarter's 13 weeks at the status quo,
  and the plan's window is those 13 weeks; the history's adstock runs into the quarter. Two
  documented properties of its optimiser differ from the bench's accounting: it scales a channel's
  media before the window by the same ratio as the window's, and it does not count what the
  quarter's spend returns after the quarter ends. Its grid search moves spend in whole units of the
  budget's rounding (``gtol`` 1e-4 of the budget, its default), so a plan can miss the budget by
  about a unit, which the bench's scoring then moves into the box and records.

Weeks are dated Mondays from 2023-01-02, as the PyMC-Marketing arm's.

**Version 2.** A scorecard export (:mod:`causaldyn_bench.scorecard.observe`, ``version`` 2) names
its controls and holds the quarter's status quo. It is fitted and planned as above, the controls
read by name and the status quo the export's, with one change: ``max_lag`` is the kernel's length
less 1. Meridian's window is ``max_lag + 1`` weeks, so version 1's ran a week past the world's
kernel, a family that does not contain it, where PyMC-Marketing's ``l_max`` gives its arm the
kernel's own; version 1 keeps it, as its committed results were fitted. Each record carries,
beside the plan:

* ``gain`` and ``gain_interval``: the posterior mean, and the 5 % and 95 % quantiles, of the plan's
  incremental outcome less the status quo's, each as the optimiser computes it: every channel's
  media over the history and the quarter scaled by the plan's spend over the status quo's, the
  outcome counted over the quarter's weeks. The optimiser's own non-optimised spend is the status
  quo rounded to its grid, so the arm checks that the same computation at that spend gives the
  optimiser's optimised total less its non-optimised one, and records the gain over the status quo
  itself. As documented, it counts nothing the quarter's spend returns after the quarter;
* ``flags``: the divergences, the largest R-hat and smallest bulk ESS over every posterior
  variable, and the model reviewer's overall status, health score and each check's status, the
  whole review beside them under ``review``;
* ``response``: up to 400 posterior draws of each channel mapped onto ``chc.response``
  (:mod:`causaldyn_bench.scorecard.mapping`): the kernel ``GeometricAdstock`` over ``max_lag + 1``
  weeks, normalised, as Meridian's adstock weighs its window; the curve ``Hill`` at ``ec`` times
  the channel's media scale (the median of its positive weeks) with Meridian's slope; the
  coefficient ``beta`` times the KPI's population-scaled standard deviation and the population.
  Beside them, ``Analyzer.incremental_outcome`` week by week over the history, the decomposition
  the bench checks them against;
* ``settings``, ``versions`` and ``cost``, the wall and CPU seconds of the whole arm;
* ``forecast`` is n/a: Meridian's ``expected_outcome`` refuses periods outside the training window,
  since its weekly effects would need a model of their own there, and the record says so.

``--smoke`` samples 100 prior draws and 2 chains of 100 adaptation, 50 burn-in and 100 kept draws
to check the plumbing; its records say they are a smoke run, not a result. A version-1 export is
fitted as before, by the same code, in full.

    uv run --no-project --python 3.12 --with-requirements scripts/meridian_arm.txt \\
        python scripts/meridian_arm.py --worlds DIR --out DIR [--smoke]

``scripts/meridian_arm.txt`` pins the whole environment, and each record names the versions it ran
on and echoes the world's digest, so the bench scores a plan only on the data it was fitted to.
"""

from __future__ import annotations

import argparse
import enum
import importlib.metadata
import json
import math
import platform
import resource
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import arviz as az  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd
from meridian import backend, constants  # ty: ignore[unresolved-import]
from meridian.analysis import analyzer, optimizer, tensors  # ty: ignore[unresolved-import]
from meridian.analysis.review import reviewer  # ty: ignore[unresolved-import]
from meridian.data import data_frame_input_data_builder  # ty: ignore[unresolved-import]
from meridian.model import model, prior_distribution, spec  # ty: ignore[unresolved-import]

FIRST_MONDAY = "2023-01-02"
YEAR = 52  # the status quo is last year's mean weekly spend, as the bench's box is centred
PRIOR_DRAWS = 500
SAMPLING = {"n_chains": 10, "n_adapt": 1000, "n_burnin": 500, "n_keep": 1000}
DEFAULT_ROI = (0.2, 0.9)  # Meridian's default roi_m, LogNormal(loc, scale)
SECOND = 2  # the scorecard's export version
SMOKE_PRIOR_DRAWS = 100
SMOKE = {"n_chains": 2, "n_adapt": 100, "n_burnin": 50, "n_keep": 100}
DRAWS = 400  # the most draws of the response a record keeps
HELD = 4  # draws whose weekly decomposition a record keeps
LEVEL = 0.9  # the gain interval's
AGREE = 1e-9  # how near the arm's mean gain lies to the optimiser's own, relative to its totals
NO_FORECAST = (
    "Meridian's expected_outcome refuses periods outside the training window: its weekly effects "
    "would need a model of their own there"
)
PACKAGES = (
    "google-meridian",
    "jax",
    "jaxlib",
    "tfp-nightly",
    "tensorflow",
    "arviz",
    "numpy",
    "pandas",
    "xarray",
)


def roi_prior(world: Any, channels: list[str]) -> tuple[Any, dict[str, Any]]:
    """Each channel's ``roi_m`` prior from its lift rows, pooled by inverse variance."""
    loc, scale, pooled = [], [], {}
    for name in channels:
        rows = np.array([str(c) == name for c in world["lift_channel"]], dtype=bool)
        if not rows.any():
            loc.append(DEFAULT_ROI[0])
            scale.append(DEFAULT_ROI[1])
            pooled[name] = None
            continue
        delta_x = np.abs(world["lift_delta_x"][rows])
        roi = world["lift_delta_y"][rows] / world["lift_delta_x"][rows]
        weight = (delta_x / world["lift_sigma"][rows]) ** 2
        mean = float(np.sum(weight * roi) / np.sum(weight))
        error = float(1.0 / math.sqrt(np.sum(weight)))
        fitted = prior_distribution.lognormal_dist_from_mean_std(mean, error)
        loc.append(float(np.asarray(fitted.loc)))
        scale.append(float(np.asarray(fitted.scale)))
        pooled[name] = {"rows": int(rows.sum()), "roi": mean, "error": error}
    prior = backend.tfd.LogNormal(
        np.array(loc, dtype=backend.np_float_dtype),
        np.array(scale, dtype=backend.np_float_dtype),
        name=constants.ROI_M,
    )
    return prior, {"pooled": pooled, "loc": loc, "scale": scale}


def _plain(value: Any) -> Any:
    """A health check's details as JSON: enums by name, arrays as lists, non-finite as None."""
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int | str | bool) or value is None:
        return value
    return str(value)


def review(mmm: Any) -> dict[str, Any]:
    """Meridian's post-modelling health checks, as its demo runs them."""
    summary = reviewer.ModelReviewer(mmm).run()
    checks = {}
    for result in summary.results:
        name = type(result).__name__.removesuffix("Result")
        try:
            details = _plain(dict(result.details))
        except Exception as error:  # a check whose details do not serialise is kept by its case
            details = f"{type(error).__name__}: {error}"
        checks[name] = {
            "case": result.case.name,
            "status": result.case.status.name,
            "details": details,
        }
    return {
        "overall": summary.overall_status.name,
        "message": summary.summary_message,
        "health_score": _plain(float(summary.health_score)),
        "checks": checks,
    }


def plan(path: Path) -> dict[str, Any]:
    world = np.load(path)
    channels = [str(c) for c in world["channels"]]
    spend = world["spend"]
    weeks, planned = world["sales"].size, int(world["planned"])
    dates = pd.date_range(FIRST_MONDAY, periods=weeks + planned, freq="W-MON")
    history = [d.strftime("%Y-%m-%d") for d in dates[:weeks]]
    frame = pd.DataFrame(
        {
            "time": history,
            "sales": world["sales"],
            **{name: spend[:, c] for c, name in enumerate(channels)},
            "promotion": world["promotion"],
            "price": world["price"],
        }
    )
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE, default_kpi_column="sales", default_time_column="time"
    )
    data = (
        builder.with_kpi(frame)
        .with_controls(frame, control_cols=["promotion", "price"])
        .with_media(frame, media_cols=channels, media_spend_cols=channels, media_channels=channels)
        .build()
    )
    prior, calibration = roi_prior(world, channels)
    model_spec = spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(roi_m=prior),
        max_lag=int(world["kernel_length"]),
        enable_aks=True,
    )
    mmm = model.Meridian(input_data=data, model_spec=model_spec)
    seed = int(world["seed"])
    started = time.perf_counter()
    mmm.sample_prior(PRIOR_DRAWS, seed=seed)
    mmm.sample_posterior(**SAMPLING, seed=seed)
    fitted = time.perf_counter() - started

    posterior = mmm.inference_data.posterior
    rhat = az.rhat(posterior)
    ess = az.ess(posterior, method="bulk")
    stats = mmm.inference_data.sample_stats
    divergences = int(stats["diverging"].sum()) if "diverging" in stats else None
    health = review(mmm)

    status_quo = spend[-YEAR:].mean(axis=0)
    budget = float(world["budget"])
    quarter = np.tile(status_quo, (planned, 1))
    budget_optimizer = optimizer.BudgetOptimizer(mmm)
    new_data = budget_optimizer.create_optimization_tensors(
        time=[d.strftime("%Y-%m-%d") for d in dates],
        cpmu=np.ones(len(channels)),
        # with its one geo already: 2.1.0 allocates a geo-less tensor by dividing by the
        # population as an xarray, which its JAX backend rejects
        media_spend=np.vstack([spend, quarter])[np.newaxis],
        # what Meridian sets for a revenue KPI, and new data over new weeks must carry it
        revenue_per_kpi=np.ones((1, len(dates))),
    )
    started = time.perf_counter()
    result = budget_optimizer.optimize(
        new_data=new_data,
        start_date=dates[weeks].strftime("%Y-%m-%d"),
        end_date=dates[-1].strftime("%Y-%m-%d"),
        budget=budget,
        pct_of_spend=list(status_quo / status_quo.sum()),
        spend_constraint_lower=list(1.0 - world["lower"] / status_quo),
        spend_constraint_upper=list(world["upper"] / status_quo - 1.0),
    )
    planning = time.perf_counter() - started
    optimized = result.optimized_data[constants.SPEND]
    total = [float(optimized.sel(channel=name)) for name in channels]
    return {
        "seed": seed,
        "digest": str(world["digest"]),
        "weekly": [t / planned for t in total],
        "quarter_spend": total,
        "fit_seconds": fitted,
        "plan_seconds": planning,
        "max_rhat": float(max(float(rhat[v].max()) for v in rhat.data_vars)),
        "min_ess_bulk": float(min(float(ess[v].min()) for v in ess.data_vars)),
        "divergences": divergences,
        "review": health,
        "roi_prior": calibration,
        "posterior_mean": {
            name: {
                "roi": float(posterior["roi_m"].sel(media_channel=name).mean()),
                "retention": float(posterior["alpha_m"].sel(media_channel=name).mean()),
                "ec": float(posterior["ec_m"].sel(media_channel=name).mean()),
                "slope": float(posterior["slope_m"].sel(media_channel=name).mean()),
            }
            for name in channels
        },
        "knots": [int(k) for k in mmm.knot_info.knot_locations],
        "float_dtype": str(backend.np_float_dtype.__name__),
        "lift_rows": int(world["lift_channel"].size),
        "lift_dropped": int(world["lift_dropped"]),
        "error": None,
    }


def _cpu() -> float:
    """CPU seconds this process, and any finished child, has spent."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def _kept(samples: int, count: int) -> np.ndarray:
    """``count`` of ``samples`` draws, evenly spaced, or all of them where there are fewer."""
    return np.unique(np.linspace(0, samples - 1, min(samples, count)).round().astype(int))


def _flags(mmm: Any, health: dict[str, Any]) -> dict[str, Any]:
    """The sampler's and the reviewer's diagnostics, one number or status each."""
    posterior = mmm.inference_data.posterior
    rhat = az.rhat(posterior)
    ess = az.ess(posterior, method="bulk")
    stats = mmm.inference_data.sample_stats
    return {
        "divergences": int(stats["diverging"].sum()) if "diverging" in stats else None,
        "max_rhat": float(max(float(rhat[v].max()) for v in rhat.data_vars)),
        "min_ess_bulk": float(min(float(ess[v].min()) for v in ess.data_vars)),
        "review": health["overall"],
        "health_score": health["health_score"],
        **{name: check["status"] for name, check in health["checks"].items()},
    }


def _response(mmm: Any, analysis: Any, channels: list[str]) -> dict[str, Any]:
    """Up to :data:`DRAWS` posterior draws of each channel as ``chc.response`` reads it, beside
    Meridian's own decomposition of the history, each channel's incremental outcome each week."""
    posterior = mmm.inference_data.posterior
    if [str(c) for c in posterior[constants.MEDIA_CHANNEL].values] != channels:
        raise RuntimeError("the posterior's channels are not the export's, in its order")
    stacked = posterior.stack(sample=(constants.CHAIN, constants.DRAW))
    samples = stacked.sizes["sample"]
    kept = _kept(samples, DRAWS)
    held = _kept(kept.size, HELD)
    weekly_outcome = np.asarray(
        analysis.incremental_outcome(aggregate_times=False, include_non_paid_channels=False)
    )  # (chains, draws, weeks, channels), in the order the stack reads them
    weekly_outcome = weekly_outcome.reshape(samples, *weekly_outcome.shape[2:])[kept]
    media_scale = np.asarray(mmm.media_tensors.media_transformer.scale_factors_gm)[0]
    unit = float(np.asarray(mmm.kpi_transformer.population_scaled_stdev)) * float(
        np.asarray(mmm.population)[0]
    )
    parameters, total, weekly = {}, {}, {}
    for c, name in enumerate(channels):

        def drawn(variable: str, at: str = name) -> np.ndarray:
            values = stacked[variable].sel({constants.MEDIA_CHANNEL: at})
            if constants.GEO in values.dims:
                values = values.isel({constants.GEO: 0})
            return np.asarray(values.transpose("sample").values)[kept]

        series = weekly_outcome[:, :, c]
        parameters[name] = {
            "retention": drawn(constants.ALPHA_M).tolist(),
            "scale": (drawn(constants.EC_M) * float(media_scale[c])).tolist(),
            "slope": drawn(constants.SLOPE_M).tolist(),
            "coefficient": (drawn(constants.BETA_GM) * unit).tolist(),
        }
        total[name] = series.sum(axis=1).tolist()
        weekly[name] = series[held].tolist()
    return {
        "kernel": "GeometricAdstock",
        "length": int(mmm.model_spec.max_lag) + 1,
        "normalized": True,
        "curve": "Hill",
        "draws": int(kept.size),
        "parameters": parameters,
        "decomposition": {"total": total, "held": held.tolist(), "weekly": weekly},
    }


def _outcome(
    analysis: Any, new_data: Any, quarter_spend: np.ndarray, status_quo: np.ndarray, window: list
) -> np.ndarray:
    """Each draw's incremental outcome of ``quarter_spend`` over ``window``, as the optimiser
    computes it: every channel's media over all weeks scaled by its spend over the status quo's
    (``BudgetOptimizer._get_incremental_outcome_tensors``)."""
    scaled = tensors.DataTensors(
        media=backend.to_tensor(
            np.asarray(new_data.media) * (quarter_spend / status_quo), dtype=backend.float_dtype
        ),
        revenue_per_kpi=new_data.revenue_per_kpi,
        time=new_data.time,
    )
    drawn = analysis.incremental_outcome(
        new_data=scaled, selected_times=window, include_non_paid_channels=False
    )
    return np.asarray(drawn).sum(axis=-1).reshape(-1)


def plan_second(path: Path, smoke: bool) -> dict[str, Any]:
    """A version-2 export's plan, with what the scorecard reads beside it."""
    world = np.load(path)
    channels = [str(c) for c in world["channels"]]
    controls = [str(c) for c in world["control_names"]]
    spend = world["spend"]
    weeks, planned = world["sales"].size, int(world["planned"])
    dates = pd.date_range(FIRST_MONDAY, periods=weeks + planned, freq="W-MON")
    days = [d.strftime("%Y-%m-%d") for d in dates]
    frame = pd.DataFrame(
        {
            "time": days[:weeks],
            "sales": world["sales"],
            **{name: spend[:, c] for c, name in enumerate(channels)},
            **{name: world["controls"][:, j] for j, name in enumerate(controls)},
        }
    )
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE, default_kpi_column="sales", default_time_column="time"
    ).with_kpi(frame)
    if controls:
        builder = builder.with_controls(frame, control_cols=controls)
    data = builder.with_media(
        frame, media_cols=channels, media_spend_cols=channels, media_channels=channels
    ).build()
    prior, calibration = roi_prior(world, channels)
    model_spec = spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(roi_m=prior),
        # Meridian's window is max_lag + 1 weeks, so this one is the world's kernel
        max_lag=int(world["kernel_length"]) - 1,
        enable_aks=True,
    )
    mmm = model.Meridian(input_data=data, model_spec=model_spec)
    seed = int(world["seed"])
    sampling = SMOKE if smoke else SAMPLING
    mmm.sample_prior(SMOKE_PRIOR_DRAWS if smoke else PRIOR_DRAWS, seed=seed)
    mmm.sample_posterior(**sampling, seed=seed)
    health = review(mmm)

    status_quo = np.asarray(world["status_quo"], dtype=float)
    quarter = np.tile(status_quo, (planned, 1))
    budget_optimizer = optimizer.BudgetOptimizer(mmm)
    new_data = budget_optimizer.create_optimization_tensors(
        time=days,
        cpmu=np.ones(len(channels)),
        # with its one geo already: 2.1.0 allocates a geo-less tensor by dividing by the
        # population as an xarray, which its JAX backend rejects
        media_spend=np.vstack([spend, quarter])[np.newaxis],
        # what Meridian sets for a revenue KPI, and new data over new weeks must carry it
        revenue_per_kpi=np.ones((1, len(dates))),
    )
    result = budget_optimizer.optimize(
        new_data=new_data,
        start_date=days[weeks],
        end_date=days[-1],
        budget=float(world["budget"]),
        pct_of_spend=list(status_quo / status_quo.sum()),
        spend_constraint_lower=list(1.0 - world["lower"] / status_quo),
        spend_constraint_upper=list(world["upper"] / status_quo - 1.0),
    )
    optimized = result.optimized_data[constants.SPEND]
    total = np.array([float(optimized.sel(channel=name)) for name in channels])
    rounded = result.nonoptimized_data[constants.SPEND]  # the status quo on the optimiser's grid

    analysis = analyzer.Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
    held, window = status_quo * planned, days[weeks:]  # the status quo's spend over the quarter
    at_plan = _outcome(analysis, new_data, total, held, window)
    at_rounded = _outcome(
        analysis,
        new_data,
        np.array([float(rounded.sel(channel=n)) for n in channels]),
        held,
        window,
    )
    claimed = (
        result.optimized_data.attrs[constants.TOTAL_INCREMENTAL_OUTCOME]
        - result.nonoptimized_data.attrs[constants.TOTAL_INCREMENTAL_OUTCOME]
    )
    scale = abs(result.optimized_data.attrs[constants.TOTAL_INCREMENTAL_OUTCOME])
    if abs(float(np.mean(at_plan - at_rounded)) - claimed) > AGREE * scale:
        raise RuntimeError(
            f"the draws' mean gain {float(np.mean(at_plan - at_rounded))} over the optimiser's "
            f"non-optimised spend is not the optimiser's {claimed}"
        )
    gain = at_plan - _outcome(analysis, new_data, held, held, window)

    return {
        "version": SECOND,
        "seed": seed,
        "digest": str(world["digest"]),
        "family": int(world["family"]),
        "environment": str(world["environment"]),
        "k": int(world["k"]),
        "weekly": (total / planned).tolist(),
        "quarter_spend": total.tolist(),
        "forecast": None,
        "gain": float(np.mean(gain)),
        "gain_interval": [
            float(np.quantile(gain, (1.0 - LEVEL) / 2.0)),
            float(np.quantile(gain, (1.0 + LEVEL) / 2.0)),
        ],
        "flags": _flags(mmm, health),
        "review": health,
        "response": _response(mmm, analysis, channels),
        "settings": {
            "max_lag": int(mmm.model_spec.max_lag),
            "enable_aks": True,
            "prior_draws": SMOKE_PRIOR_DRAWS if smoke else PRIOR_DRAWS,
            "sampling": sampling,
            "roi_prior": calibration,
            "knots": [int(k) for k in mmm.knot_info.knot_locations],
            "float_dtype": str(backend.np_float_dtype.__name__),
            "controls": controls,
            "gain_window": "the quarter's weeks alone, as Meridian's optimiser counts them",
            "gain_level": LEVEL,
            "smoke": smoke,
        },
        "n/a": {"forecast": NO_FORECAST},
        "lift_rows": int(world["lift_channel"].size),
        "lift_dropped": int(world["lift_dropped"]),
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worlds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--shard",
        default="0/1",
        help="K/N: every Nth world from the Kth, so N processes share the worlds; each world is "
        "fitted from its own seed, so the split moves no number",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="a reduced sampler on version-2 exports, to check the plumbing: not a result",
    )
    args = parser.parse_args()
    first, step = map(int, args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)
    for path in sorted(args.worlds.glob("world_*.npz"))[first::step]:
        target = args.out / f"{path.stem}.json"
        if target.exists():
            continue
        with np.load(path) as saved:
            second = "version" in saved.files
        if not second and args.smoke:
            raise SystemExit(f"{path} is a version-1 export, fitted as version 1 fitted it")
        began, spent = time.perf_counter(), _cpu()
        record: dict[str, Any]
        try:
            record = plan_second(path, args.smoke) if second else plan(path)
        except Exception as error:  # a world that breaks the arm is recorded, not fatal
            world = np.load(path)
            record = {
                "seed": int(world["seed"]),
                "digest": str(world["digest"]),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            if second:
                record = {"version": SECOND, **record}
        if second:
            record["cost"] = {
                "wall_seconds": time.perf_counter() - began,
                "cpu_seconds": _cpu() - spent,
            }
            if args.smoke:
                record["smoke"] = "smoke, not a result"
        else:
            record["seconds"] = time.perf_counter() - began
        record["python"] = platform.python_version()
        record["versions"] = {name: importlib.metadata.version(name) for name in PACKAGES}
        target.write_text(json.dumps(record, allow_nan=False))
        print(path.stem, record.get("seconds"), record.get("error"), flush=True)


if __name__ == "__main__":
    main()

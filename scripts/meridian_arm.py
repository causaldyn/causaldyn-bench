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

    uv run --no-project --python 3.12 --with-requirements scripts/meridian_arm.txt \\
        python scripts/meridian_arm.py --worlds DIR --out DIR

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
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import arviz as az  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd
from meridian import backend, constants  # ty: ignore[unresolved-import]
from meridian.analysis import optimizer  # ty: ignore[unresolved-import]
from meridian.analysis.review import reviewer  # ty: ignore[unresolved-import]
from meridian.data import data_frame_input_data_builder  # ty: ignore[unresolved-import]
from meridian.model import model, prior_distribution, spec  # ty: ignore[unresolved-import]

FIRST_MONDAY = "2023-01-02"
YEAR = 52  # the status quo is last year's mean weekly spend, as the bench's box is centred
PRIOR_DRAWS = 500
SAMPLING = {"n_chains": 10, "n_adapt": 1000, "n_burnin": 500, "n_keep": 1000}
DEFAULT_ROI = (0.2, 0.9)  # Meridian's default roi_m, LogNormal(loc, scale)
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
    args = parser.parse_args()
    first, step = map(int, args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)
    for path in sorted(args.worlds.glob("world_*.npz"))[first::step]:
        target = args.out / f"{path.stem}.json"
        if target.exists():
            continue
        began = time.perf_counter()
        record: dict[str, Any]
        try:
            record = plan(path)
        except Exception as error:  # a world that breaks the arm is recorded, not fatal
            world = np.load(path)
            record = {
                "seed": int(world["seed"]),
                "digest": str(world["digest"]),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        record["seconds"] = time.perf_counter() - began
        record["python"] = platform.python_version()
        record["versions"] = {name: importlib.metadata.version(name) for name in PACKAGES}
        target.write_text(json.dumps(record, allow_nan=False))
        print(path.stem, record.get("seconds"), record.get("error"), flush=True)


if __name__ == "__main__":
    main()

"""Track M v2's PyMC-Marketing arm, outside the bench's environment.

PyMC-Marketing needs Python 3.12 or later and brings PyMC, PyTensor and ArviZ, so it is never a
dependency of the bench: this script runs in a throwaway environment, reads the worlds
:func:`causaldyn_bench.budget_regret.export` wrote, and writes each world's plan as JSON, which the
bench scores on the world's own channels.

Each world is fitted as a practitioner fits PyMC-Marketing 1.2.0's media-mix model:

* ``MMM`` with its defaults, ``GeometricAdstock(l_max=6)`` (normalised by default) and
  ``LogisticSaturation``, the world's forms; the observed promotion indicator and price as
  controls; three yearly harmonics, as Heusch's (2026a) realistic specification;
* every geo test entered through ``add_lift_test_measurements``, as the bench reduced it;
* NUTS at PyMC's defaults: 4 chains of 1000 tuning and 1000 kept draws, ``target_accept`` 0.8;
* the quarter planned by its own optimiser, ``budget_optimizer(start, end).allocate_budget``, at
  the bench's weekly budget and box, starting the week after the history so the history's
  carryover runs into the plan.

Weeks are dated Mondays from 2023-01-02; PyMC-Marketing's harmonics read the day of the year over
365.25 days, which drifts from the world's 52-week year by under a week over its three years.

**Version 2.** A scorecard export (:mod:`causaldyn_bench.scorecard.observe`, ``version`` 2) names
its controls and holds the quarter's controls and its status quo. It is fitted as above, the
controls read by name, in one of three settings ``--setting`` names:

* ``static``: the model above;
* ``shared``: PyMC-Marketing's time-varying media, ``time_varying_media=True``: one HSGP multiplier
  over the weeks, shared by every channel, at the default's hyperparameters;
* ``channel``: a ``SoftPlusHSGP`` at the same hyperparameters with a channel dimension, passed as
  ``time_varying_media``: one multiplier for each channel.

A time-varying model reads a lift test at a date, so in those settings each lift row carries its
test's date. CHOICE: the Monday of its first dark week. Each record carries, beside the plan:

* ``forecast``: up to 400 of the posterior predictive's draws of the quarter's weekly sales, every
  channel at the status quo and the controls at the quarter's, with the history's last ``l_max``
  weeks carried in (``include_last_observations``), in the outcome's units;
* ``gain`` and ``gain_interval``: the posterior mean, and the 5 % and 95 % quantiles, of the plan's
  media response less the status quo's over the optimiser's window
  (``evaluate_response_distribution``), the quarter and ``l_max`` weeks after it. The window's
  leading ``l_max`` weeks hold the history's spend under both plans, so their response cancels.
  The arm checks that the plan's mean is the response the optimiser maximised;
* ``flags``: the divergences, the draws that reached the largest tree depth, and the largest R-hat
  and smallest bulk ESS over the free variables, the quantities PyMC's own convergence check reads;
* ``response``: on the static setting, up to 400 draws of each channel mapped onto
  ``chc.response`` (:mod:`causaldyn_bench.scorecard.mapping`): the kernel ``GeometricAdstock``
  over ``l_max`` weeks, normalised; the curve ``Tanh`` at ``2 s / lam`` for the channel's scale
  ``s``, the largest spend of the history; the coefficient ``beta`` times the target's scale, the
  largest sales of the history. Beside them, ``channel_contribution`` in the outcome's units, the
  decomposition the bench checks them against. A time-varying setting multiplies each week's
  contribution by its latent process, which no ``chc.response`` channel holds: ``unmapped``;
* ``settings``, ``versions`` and ``cost``, the wall and CPU seconds of the whole arm.

``--smoke`` samples 2 chains of 100 tuning and 100 kept draws to check the plumbing; its records
say they are a smoke run, not a result. A version-1 export is fitted as before, by the same code,
statically and in full.

    uv run --no-project --python 3.12 --with-requirements scripts/pymc_marketing_arm.txt \\
        python scripts/pymc_marketing_arm.py --worlds DIR --out DIR \\
        [--setting static|shared|channel] [--smoke]

``scripts/pymc_marketing_arm.txt`` pins the whole environment, and each record names the versions
it ran on and echoes the world's digest, so the bench scores a plan only on the data it was fitted
to.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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
import xarray as xr  # ty: ignore[unresolved-import]
from pymc_marketing.hsgp_kwargs import HSGPKwargs  # ty: ignore[unresolved-import]
from pymc_marketing.mmm import (  # ty: ignore[unresolved-import]
    MMM,
    GeometricAdstock,
    LogisticSaturation,
)
from pymc_marketing.mmm.tvp import create_hsgp_from_config  # ty: ignore[unresolved-import]

FIRST_MONDAY = "2023-01-02"
HARMONICS = 3
PACKAGES = ("pymc-marketing", "pymc", "pytensor", "arviz", "numpy", "scipy", "pandas")
SECOND = 2  # the scorecard's export version
SETTINGS = {  # each --setting, as a record names it
    "static": "static media",
    "shared": "time-varying media, one HSGP shared by every channel",
    "channel": "time-varying media, a SoftPlusHSGP for each channel",
}
# the default's own HSGP hyperparameters for time-varying media (MMM.default_model_config)
MEDIA_HSGP = HSGPKwargs(m=200, L=None, eta_lam=1, ls_mu=5, ls_sigma=10, cov_func=None)
SMOKE = {"chains": 2, "tune": 100, "draws": 100}
DRAWS = 400  # the most draws of the forecast and the response a record keeps
HELD = 4  # draws whose weekly decomposition a record keeps
LEVEL = 0.9  # the gain interval's
AGREE = 1e-9  # how near the plan's mean response lies to the optimiser's own, relative to it


def plan(path: Path) -> dict[str, Any]:
    world = np.load(path)
    channels = [str(c) for c in world["channels"]]
    weeks = world["sales"].size
    dates = pd.date_range(FIRST_MONDAY, periods=weeks, freq="W-MON")
    frame = pd.DataFrame(
        {
            "date": dates,
            **{name: world["spend"][:, c] for c, name in enumerate(channels)},
            "promotion": world["promotion"],
            "price": world["price"],
        }
    )
    sales = pd.Series(world["sales"], name="sales")
    model = MMM(
        date_column="date",
        channel_columns=channels,
        control_columns=["promotion", "price"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=int(world["kernel_length"])),
        saturation=LogisticSaturation(),
        yearly_seasonality=HARMONICS,
    )
    model.build_model(frame, sales)
    if world["lift_channel"].size:
        model.add_lift_test_measurements(
            pd.DataFrame(
                {
                    "channel": [str(c) for c in world["lift_channel"]],
                    "x": world["lift_x"],
                    "delta_x": world["lift_delta_x"],
                    "delta_y": world["lift_delta_y"],
                    "sigma": world["lift_sigma"],
                }
            )
        )
    seed = int(world["seed"])
    started = time.perf_counter()
    trace = model.fit(frame, sales, random_seed=seed, progressbar=False)
    fitted = time.perf_counter() - started
    planned = int(world["planned"])
    start = dates[-1] + pd.Timedelta(weeks=1)
    optimiser = model.budget_optimizer(
        start_date=start, end_date=start + pd.Timedelta(weeks=planned - 1)
    )
    bounds = {
        name: (float(world["lower"][c]), float(world["upper"][c]))
        for c, name in enumerate(channels)
    }
    result = optimiser.allocate_budget(float(world["budget"]) / planned, budget_bounds=bounds)
    weekly = [float(result.budgets.sel(channel=name)) for name in channels]
    summary = az.summary(trace, var_names=["saturation_beta", "saturation_lam", "adstock_alpha"])
    return {
        "seed": seed,
        "digest": str(world["digest"]),
        "weekly": weekly,
        "optimiser_success": bool(result.scipy_result.success),
        "fit_seconds": fitted,
        "divergences": int(trace.sample_stats["diverging"].sum()),
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "posterior_mean": {
            name: {
                "retention": float(trace.posterior["adstock_alpha"].sel(channel=name).mean()),
                "lam": float(trace.posterior["saturation_lam"].sel(channel=name).mean()),
                "beta": float(trace.posterior["saturation_beta"].sel(channel=name).mean()),
            }
            for name in channels
        },
        "lift_rows": int(world["lift_channel"].size),
        "lift_dropped": int(world["lift_dropped"]),
        "error": None,
    }


def _cpu() -> float:
    """CPU seconds this process and its finished children, the samplers, have spent."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def _kept(samples: int, count: int) -> np.ndarray:
    """``count`` of ``samples`` draws, evenly spaced, or all of them where there are fewer."""
    return np.unique(np.linspace(0, samples - 1, min(samples, count)).round().astype(int))


def _media(setting: str, weeks: int) -> Any:
    """``time_varying_media`` for ``setting``."""
    if setting == "static":
        return False
    if setting == "shared":
        return True
    return create_hsgp_from_config(
        X=xr.DataArray(np.arange(weeks), dims=("date",)),
        dims=("date", "channel"),
        config=MEDIA_HSGP,
    )


def _flags(trace: Any, model: Any) -> dict[str, Any]:
    """The divergences, the draws at the largest tree depth, and the largest R-hat and smallest
    bulk ESS over the free variables, as ``pm.sample``'s convergence check reads them."""
    posterior = trace.posterior.to_dataset()
    names = [rv.name for rv in model.model.free_RVs if rv.name in posterior]
    summary = az.summary(trace, var_names=names)
    stats = trace.sample_stats
    return {
        "divergences": int(stats["diverging"].sum()),
        "max_treedepth": int(stats["reached_max_treedepth"].sum()),
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
    }


def _response(model: Any, trace: Any, channels: list[str]) -> dict[str, Any]:
    """Up to :data:`DRAWS` draws of each channel as ``chc.response`` reads it, beside the tool's
    own decomposition of the history, ``channel_contribution`` in the outcome's units."""
    posterior = trace.posterior.to_dataset().stack(sample=("chain", "draw"))
    kept = _kept(posterior.sizes["sample"], DRAWS)
    held = _kept(kept.size, HELD)
    target = float(model.scalers["_target"])
    contribution = posterior["channel_contribution"].transpose("sample", "date", "channel")
    parameters, total, weekly = {}, {}, {}
    for name in channels:
        lam = posterior["saturation_lam"].sel(channel=name).values[kept]
        beta = posterior["saturation_beta"].sel(channel=name).values[kept]
        series = contribution.sel(channel=name).values[kept] * target
        parameters[name] = {
            "retention": posterior["adstock_alpha"].sel(channel=name).values[kept].tolist(),
            "scale": (2.0 * float(model.scalers["_channel"].sel(channel=name)) / lam).tolist(),
            "coefficient": (beta * target).tolist(),
        }
        total[name] = series.sum(axis=1).tolist()
        weekly[name] = series[held].tolist()
    return {
        "kernel": "GeometricAdstock",
        "length": int(model.adstock.l_max),
        "normalized": bool(model.adstock.normalize),
        "curve": "Tanh",
        "draws": int(kept.size),
        "parameters": parameters,
        "decomposition": {"total": total, "held": held.tolist(), "weekly": weekly},
    }


def plan_second(path: Path, setting: str, smoke: bool) -> dict[str, Any]:
    """A version-2 export's plan, with what the scorecard reads beside it."""
    world = np.load(path)
    channels = [str(c) for c in world["channels"]]
    controls = [str(c) for c in world["control_names"]]
    weeks, planned = world["sales"].size, int(world["planned"])
    dates = pd.date_range(FIRST_MONDAY, periods=weeks + planned, freq="W-MON")
    frame = pd.DataFrame(
        {
            "date": dates[:weeks],
            **{name: world["spend"][:, c] for c, name in enumerate(channels)},
            **{name: world["controls"][:, j] for j, name in enumerate(controls)},
        }
    )
    sales = pd.Series(world["sales"], name="sales")
    model = MMM(
        date_column="date",
        channel_columns=channels,
        control_columns=controls or None,
        target_column="sales",
        adstock=GeometricAdstock(l_max=int(world["kernel_length"])),
        saturation=LogisticSaturation(),
        yearly_seasonality=HARMONICS,
        time_varying_media=_media(setting, weeks),
    )
    model.build_model(frame, sales)
    rows = int(world["lift_channel"].size)
    if rows:
        lift = pd.DataFrame(
            {
                "channel": [str(c) for c in world["lift_channel"]],
                "x": world["lift_x"],
                "delta_x": world["lift_delta_x"],
                "delta_y": world["lift_delta_y"],
                "sigma": world["lift_sigma"],
            }
        )
        if setting != "static":
            lift["date"] = dates[world["lift_start"] - 1]
        model.add_lift_test_measurements(lift)
    seed = int(world["seed"])
    trace = model.fit(frame, sales, random_seed=seed, progressbar=False, **(SMOKE if smoke else {}))

    optimiser = model.budget_optimizer(start_date=dates[weeks], end_date=dates[-1])
    bounds = {
        name: (float(world["lower"][c]), float(world["upper"][c]))
        for c, name in enumerate(channels)
    }
    result = optimiser.allocate_budget(float(world["budget"]) / planned, budget_bounds=bounds)
    status_quo = xr.DataArray(world["status_quo"], dims=("channel",), coords={"channel": channels})
    at_plan = optimiser.evaluate_response_distribution({"channel_data": result.budgets})
    optimised = -float(result.scipy_result.fun)  # the mean response, the default utility
    if abs(float(at_plan.mean()) - optimised) > AGREE * abs(optimised):
        raise RuntimeError(
            f"the plan's mean response {float(at_plan.mean())} is not the optimiser's {optimised}"
        )
    gain = np.asarray(
        at_plan - optimiser.evaluate_response_distribution({"channel_data": status_quo})
    )

    future = pd.DataFrame(
        {
            "date": dates[weeks:],
            **{name: np.full(planned, world["status_quo"][c]) for c, name in enumerate(channels)},
            **{name: world["future_controls"][:, j] for j, name in enumerate(controls)},
        }
    )
    predicted = model.sample_posterior_predictive(
        future,
        extend_idata=False,
        include_last_observations=True,
        var_names=["y"],  # the lift likelihood has no draw to make over the quarter
        random_seed=seed,
        progressbar=False,
    )
    forecast = predicted["y"].transpose("sample", "date").values * float(model.scalers["_target"])

    posterior = trace.posterior.to_dataset()
    if setting == "static":
        response = _response(model, trace, channels)
    else:
        response = {
            "unmapped": "a time-varying setting multiplies each week's channel contribution by its "
            "latent process, which no chc.response channel holds"
        }
    return {
        "version": SECOND,
        "seed": seed,
        "digest": str(world["digest"]),
        "family": int(world["family"]),
        "environment": str(world["environment"]),
        "k": int(world["k"]),
        "weekly": [float(result.budgets.sel(channel=name)) for name in channels],
        "optimiser_success": bool(result.scipy_result.success),
        "forecast": forecast[_kept(forecast.shape[0], DRAWS)].tolist(),
        "gain": float(np.mean(gain)),
        "gain_interval": [
            float(np.quantile(gain, (1.0 - LEVEL) / 2.0)),
            float(np.quantile(gain, (1.0 + LEVEL) / 2.0)),
        ],
        "flags": _flags(trace, model),
        "response": response,
        "settings": {
            "setting": setting,
            "media": SETTINGS[setting],
            "l_max": int(world["kernel_length"]),
            "yearly_seasonality": HARMONICS,
            "controls": controls,
            "chains": int(posterior.sizes["chain"]),
            "draws": int(posterior.sizes["draw"]),
            "lift_dated": setting != "static",
            "gain_level": LEVEL,
            "smoke": smoke,
        },
        "n/a": {},
        "lift_rows": rows,
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
        "--setting",
        choices=sorted(SETTINGS),
        default="static",
        help="how a version-2 export's media vary over the weeks; a version-1 export is static",
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
        if not second and (args.setting != "static" or args.smoke):
            raise SystemExit(f"{path} is a version-1 export, fitted as version 1 fitted it")
        began, spent = time.perf_counter(), _cpu()
        record: dict[str, Any]
        try:
            record = plan_second(path, args.setting, args.smoke) if second else plan(path)
        except Exception as error:  # a world that breaks the arm is recorded, not fatal
            world = np.load(path)
            record = {
                "seed": int(world["seed"]),
                "digest": str(world["digest"]),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            if second:
                record = {"version": SECOND, **record, "settings": {"setting": args.setting}}
        if second:
            record["cost"] = {
                "wall_seconds": time.perf_counter() - began,
                "cpu_seconds": _cpu() - spent,
            }
            if args.smoke:
                record["smoke"] = "smoke, not a result"
        record["python"] = platform.python_version()
        record["versions"] = {name: importlib.metadata.version(name) for name in PACKAGES}
        target.write_text(json.dumps(record, allow_nan=False))
        print(path.stem, record.get("fit_seconds"), record.get("error"), flush=True)


if __name__ == "__main__":
    main()

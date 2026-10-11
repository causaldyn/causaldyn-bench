"""Track M v2's PyMC-Marketing arm, outside the bench's environment.

PyMC-Marketing needs Python 3.12 or later and brings PyMC, PyTensor and ArviZ, so it is never a
dependency of the bench: this script runs in a throwaway environment, reads the worlds
:func:`causaldyn_bench.budget_regret.export` wrote, and writes each world's plan as JSON, which the
bench scores on the world's own channels.

Each world is fitted as a practitioner fits PyMC-Marketing 1.2.0's media-mix model:

* ``MMM`` with its defaults, ``GeometricAdstock(l_max=6)`` (normalised by default) and
  ``LogisticSaturation``, the world's forms; the observed promotion indicator and price as
  controls; three yearly harmonics, as Heusch's (2026a) realistic specification;
* every geo test its lift likelihood takes entered through ``add_lift_test_measurements``, as the
  bench reduced it, at the tool's default likelihood, a Gamma of the change's size. An export
  carries every test, whichever way its sales moved; the likelihood takes a test whose sales fell,
  so the arm drops the rest and counts them (``lift_dropped``, :mod:`pymc_marketing_lift`);
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
* ``response``: up to 400 draws of each channel mapped onto ``chc.response``
  (:mod:`causaldyn_bench.scorecard.mapping`): the kernel ``GeometricAdstock`` over ``l_max`` weeks,
  normalised; the curve ``Tanh`` at ``2 s / lam`` for the channel's scale ``s``, the largest spend
  of the history; the coefficient ``beta`` times the target's scale, the largest sales of the
  history. A time-varying setting multiplies each week's contribution by its latent process,
  ``media_temporal_latent_multiplier``, so there each draw's coefficient is a path over the
  history's weeks, that coefficient times the process in each week. Beside them,
  ``channel_contribution`` in the outcome's units, the decomposition the bench checks them against;
* ``multiplier``, in a time-varying setting, None in the static one: the latent process as the
  optimiser evaluates it on the history's last week and the quarter's weeks, the HSGP extended to
  those weeks, at the draws ``response`` keeps. For each channel and week, the draws' mean, their
  5 % and 95 % quantiles, and their correlation with the history's last week; beside them the
  lengthscale's mean and quantiles, in weeks. The optimiser's graph draws nothing, so no other
  number of the record moves;
* ``settings``, ``versions`` and ``cost``, the wall and CPU seconds of the whole arm.

**A geo export** (family 8, one holding ``population``) is fitted by PyMC-Marketing's
multidimensional ``MMM`` with ``dims=("geo",)``, static media alone: each geo's KPI and spend, a
national channel's spend allotted to the geos by population as the world allots it, ``l_max`` the
export's kernel length. Its lift likelihood reads each row at one coordinate of every dimension, so
one geo's change in sales, and a family-8 test takes several geos dark at once: the arm apportions
each test across its tested geos (:func:`pymc_marketing_lift.apportioned`). CHOICE, the arm's
assumption and not the tool's: one ROI across a test's geos, as the test's own design assumes. Each
geo's row withholds the geo's own mean spend over the dark weeks and reads that share of the test's
change in sales, at an error that lets the ``m`` rows, read as independent on a Gaussian
likelihood, hold the one test's information. So the arm reads these rows on a Normal likelihood,
``dist=pymc.dims.Normal``, where the identity is exact. The tool's default, a Gamma of the change's
size, has an information about its mean at a fixed error of ``(4 a psi'(a) - 3) / sigma**2`` at
shape ``a``, the squared z; a row's shape is its test's over ``m``, so on the Gamma the five rows of
a weak test would hold up to 5.49 times its information (at a squared z of 0.39), twice it where the
test's squared z is 7.8, and 1.5 times it at 15.9: a handicap the apportionment, not the tool, would
make. A national export's tests keep the Gamma, one row a test. The tool still departs from the
identity once: its geo model gives each geo its own saturation and coefficient on each channel, so
a row informs its own geo's curve alone, with a share of the test's information, and no common
return adds the rows up. Each row takes its test's sign, so a test whose sales did not fall drops
its every row: ``lift_rows`` and ``lift_dropped`` count rows, ``lift_tests`` the tests taken and
dropped.

The quarter is planned by its own optimiser over every geo's cells, from the status quo, at the
weekly budget and in each cell's box, each national channel's spend in a geo held to the geo's
share of the population of the channel's (one equality a geo but the last), and recorded one weekly
spend a cell. ``forecast``, ``gain`` and ``gain_interval`` are the market's, summed over the geos.
``response`` is not mapped: the mapping reads one response of the market's spend, where a geo
model's is each geo's own curve.

``--smoke`` samples 2 chains of 100 tuning and 100 kept draws to check the plumbing; its records
say they are a smoke run, not a result. A version-1 export is fitted as before, by the same code,
statically and in full.

    uv run --no-project --python 3.12 --with-requirements scripts/pymc_marketing_arm.txt \\
        python scripts/pymc_marketing_arm.py --worlds DIR --out DIR \\
        [--setting static|shared|channel] [--smoke]

``scripts/pymc_marketing_arm.txt`` pins the whole environment, and each record names the versions
it ran on and echoes the world's digest, so the bench scores a plan only on the data it was fitted
to.

PyTensor compiles the model without numba's file cache. With it, the compiled code depends on what
the cache held when the model was compiled, and so do the draws: one world fitted from one seed drew
three different chains from three states of the cache, whose gradients of the log density differed
by about 1e-13 at ordinary points before the sampler's chaos grew the difference. Without it, two
fits of that world were the same bit for bit, at the cost of compiling every function in every
process.
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
import pymc.dims as pmd  # ty: ignore[unresolved-import]
import pytensor  # ty: ignore[unresolved-import]
import xarray as xr  # ty: ignore[unresolved-import]
from geo_export import cells, geo, panel
from pymc_marketing.hsgp_kwargs import HSGPKwargs  # ty: ignore[unresolved-import]
from pymc_marketing.mmm import (  # ty: ignore[unresolved-import]
    MMM,
    GeometricAdstock,
    LogisticSaturation,
)
from pymc_marketing.mmm.constraints import (  # ty: ignore[unresolved-import]
    Constraint,
    build_default_sum_constraint,
)
from pymc_marketing.mmm.tvp import create_hsgp_from_config  # ty: ignore[unresolved-import]
from pymc_marketing_lift import apportioned, taken

pytensor.config.numba__cache = False

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
GEO_LIFT = (
    "each test apportioned across its tested geos at one ROI, the arm's assumption: a geo's row "
    "withholds its own mean spend over the dark weeks, reads that share s of the change in sales, "
    "and an error of s * sigma * sqrt(m) for a test of m geos, read on a Normal likelihood"
)
GEO_UNMAPPED = (
    "a geo model's response is each geo's own curve of its own spend, where the mapping reads one "
    "response of the market's spend"
)


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
    keep, dropped = taken(world)
    if keep.any():
        model.add_lift_test_measurements(
            pd.DataFrame(
                {
                    "channel": [str(c) for c in world["lift_channel"][keep]],
                    "x": world["lift_x"][keep],
                    "delta_x": world["lift_delta_x"][keep],
                    "delta_y": world["lift_delta_y"][keep],
                    "sigma": world["lift_sigma"][keep],
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
        "lift_rows": int(np.count_nonzero(keep)),
        "lift_dropped": int(world["lift_dropped"]) + dropped,
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
    own decomposition of the history, ``channel_contribution`` in the outcome's units. Where the
    media's effect moves over the weeks, each draw's coefficient is its path over them."""
    posterior = trace.posterior.to_dataset().stack(sample=("chain", "draw"))
    kept = _kept(posterior.sizes["sample"], DRAWS)
    held = _kept(kept.size, HELD)
    target = float(model.scalers["_target"])
    contribution = posterior["channel_contribution"].transpose("sample", "date", "channel")
    process = posterior.get("media_temporal_latent_multiplier")
    parameters, total, weekly = {}, {}, {}
    for name in channels:
        lam = posterior["saturation_lam"].sel(channel=name).values[kept]
        coefficient = posterior["saturation_beta"].sel(channel=name).values[kept] * target
        if process is not None:  # one process for every channel, or one a channel
            at = process.sel(channel=name) if "channel" in process.dims else process
            coefficient = coefficient[:, None] * at.transpose("sample", "date").values[kept]
        series = contribution.sel(channel=name).values[kept] * target
        parameters[name] = {
            "retention": posterior["adstock_alpha"].sel(channel=name).values[kept].tolist(),
            "scale": (2.0 * float(model.scalers["_channel"].sel(channel=name)) / lam).tolist(),
            "coefficient": coefficient.tolist(),
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


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """Pearson's correlation of two weeks' draws, None where either does not vary."""
    if not (np.std(first) > 0.0 and np.std(second) > 0.0):
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _multiplier(
    optimiser: Any, plan: Any, trace: Any, channels: list[str], dates: pd.DatetimeIndex
) -> dict[str, Any] | None:
    """The time-varying media's latent process on ``dates`` as the optimiser evaluates it,
    summarised over the draws :func:`_response` keeps, with the HSGP's lengthscale; None for
    static media."""
    posterior = trace.posterior.to_dataset().stack(sample=("chain", "draw"))
    if "media_temporal_latent_multiplier" not in posterior:
        return None
    kept = _kept(posterior.sizes["sample"], DRAWS)
    with warnings.catch_warnings():  # the process does not read the plan, and says so
        warnings.simplefilter("ignore", UserWarning)
        process = optimiser.evaluate_response_distribution(
            {"channel_data": plan}, response_variable="media_temporal_latent_multiplier"
        ).sel(date=dates)
    tail = (1.0 - LEVEL) / 2.0
    summary = {}
    for name in channels:
        at = process.sel(channel=name) if "channel" in process.dims else process
        path = at.transpose("sample", "date").values[kept]
        summary[name] = {
            "mean": path.mean(axis=0).tolist(),
            "interval": np.quantile(path, [tail, 1.0 - tail], axis=0).T.tolist(),
            "correlation": [_correlation(path[:, 0], path[:, h]) for h in range(dates.size)],
        }
    scale = posterior.get("media_temporal_latent_multiplier_raw_ls")
    lengthscale = None
    if scale is not None:
        drawn = scale.values[kept]
        lengthscale = {
            "mean": float(drawn.mean()),
            "interval": np.quantile(drawn, [tail, 1.0 - tail]).tolist(),
        }
    return {"weeks": int(dates.size), "lengthscale": lengthscale, "channels": summary}


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
    keep, dropped = taken(world)
    rows = int(np.count_nonzero(keep))
    if rows:
        lift = pd.DataFrame(
            {
                "channel": [str(c) for c in world["lift_channel"][keep]],
                "x": world["lift_x"][keep],
                "delta_x": world["lift_delta_x"][keep],
                "delta_y": world["lift_delta_y"][keep],
                "sigma": world["lift_sigma"][keep],
            }
        )
        if setting != "static":
            lift["date"] = dates[world["lift_start"][keep] - 1]
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
        "response": _response(model, trace, channels),
        "multiplier": _multiplier(optimiser, result.budgets, trace, channels, dates[weeks - 1 :]),
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
        "lift_dropped": int(world["lift_dropped"]) + dropped,
        "error": None,
    }


def _geos(world: Any) -> list[str]:
    """Each geo's name, zero-padded so that sorted coordinates keep the export's order."""
    return [f"geo_{g:02d}" for g in range(np.asarray(world["population"]).size)]


def _long(world: Any, dates: pd.DatetimeIndex, spend: np.ndarray) -> pd.DataFrame:
    """Each geo's weeks of ``dates`` as the model reads them, one row a geo and week, each
    channel's spend ``spend[g, t, c]``, a national channel's allotted by population."""
    channels = [str(c) for c in world["channels"]]
    return pd.concat(
        [
            pd.DataFrame(
                {"date": dates, "geo": name, **{n: spend[g, :, c] for c, n in enumerate(channels)}}
            )
            for g, name in enumerate(_geos(world))
        ],
        ignore_index=True,
    )


def _cell_grid(world: Any, weekly: np.ndarray) -> xr.DataArray:
    """A weekly spend a cell as the optimiser's budgets, one a geo and channel."""
    return xr.DataArray(
        panel(world, weekly),
        dims=("geo", "channel"),
        coords={"geo": _geos(world), "channel": [str(c) for c in world["channels"]]},
    )


def _national(world: Any) -> list[Constraint]:
    """Each national channel's spend in each geo held at the geo's share of the population of the
    channel's spend over the geos, as the world allots it: one equality a geo but the last, whose
    share the others' fix."""
    shares = np.asarray(world["population"], dtype=float)
    channels = [str(c) for c in world["channels"]]
    out = []
    for name in (str(c) for c in world["national"]):
        c = channels.index(name)
        for g in range(shares.size - 1):

            def held(budgets: Any, total: Any, optimizer: Any, g: int = g, c: int = c) -> Any:
                spent = budgets.isel(channel=c)
                return spent.isel(geo=g) - float(shares[g]) * spent.sum("geo")

            out.append(Constraint(key=f"{name}_{g}", constraint_type="eq", constraint_fun=held))
    return out


def plan_geo(path: Path, smoke: bool) -> dict[str, Any]:
    """A geo export's plan: the geo model, its tests as its likelihood takes them, and its own
    optimiser over every geo's cells."""
    world = np.load(path)
    channels = [str(c) for c in world["channels"]]
    weeks, planned = world["sales"].shape[0], int(world["planned"])
    dates = pd.date_range(FIRST_MONDAY, periods=weeks + planned, freq="W-MON")
    names = _geos(world)
    frame = _long(world, dates[:weeks], np.transpose(world["spend"], (1, 0, 2)))
    sales = pd.Series(np.asarray(world["sales"], dtype=float).T.ravel(), name="sales")
    model = MMM(
        date_column="date",
        channel_columns=channels,
        target_column="sales",
        adstock=GeometricAdstock(l_max=int(world["kernel_length"])),
        saturation=LogisticSaturation(),
        yearly_seasonality=HARMONICS,
        dims=("geo",),
    )
    model.build_model(frame, sales)
    split, keep = apportioned(world)
    rows = int(np.count_nonzero(keep))
    if rows:
        model.add_lift_test_measurements(
            pd.DataFrame(
                {
                    "channel": split["channel"][keep],
                    "geo": [names[int(g)] for g in split["geo"][keep]],
                    **{key: split[key][keep] for key in ("x", "delta_x", "delta_y", "sigma")},
                }
            ),
            dist=pmd.Normal,
        )
    tests = int(np.asarray(world["lift_channel"]).size)
    taken_tests = int(np.unique(split["test"][keep]).size)
    seed = int(world["seed"])
    trace = model.fit(frame, sales, random_seed=seed, progressbar=False, **(SMOKE if smoke else {}))

    status_quo = _cell_grid(world, world["status_quo"])
    bounds = xr.concat(
        [_cell_grid(world, world["lower"]), _cell_grid(world, world["upper"])], dim="bound"
    ).transpose("geo", "channel", "bound")
    optimiser = model.budget_optimizer(
        start_date=dates[weeks],
        end_date=dates[-1],
        constraints=[build_default_sum_constraint(), *_national(world)],
    )
    result = optimiser.allocate_budget(
        float(world["budget"]) / planned, budget_bounds=bounds, x0=status_quo
    )
    plan = result.budgets.transpose("geo", "channel")
    at_plan = optimiser.evaluate_response_distribution({"channel_data": result.budgets})
    optimised = -float(result.scipy_result.fun)  # the mean response, the default utility
    if abs(float(at_plan.mean()) - optimised) > AGREE * abs(optimised):
        raise RuntimeError(
            f"the plan's mean response {float(at_plan.mean())} is not the optimiser's {optimised}"
        )
    gain = np.asarray(
        at_plan - optimiser.evaluate_response_distribution({"channel_data": status_quo})
    )

    future = _long(
        world,
        dates[weeks:],
        np.repeat(status_quo.transpose("geo", "channel").values[:, np.newaxis], planned, axis=1),
    )
    predicted = model.sample_posterior_predictive(
        future,
        extend_idata=False,
        include_last_observations=True,
        var_names=["y"],
        random_seed=seed,
        progressbar=False,
    )
    market = (predicted["y"] * model.scalers["_target"]).sum("geo")
    forecast = market.transpose("sample", "date").values

    posterior = trace.posterior.to_dataset()
    weekly = cells(world, plan.values)
    return {
        "version": SECOND,
        "seed": seed,
        "digest": str(world["digest"]),
        "family": int(world["family"]),
        "environment": str(world["environment"]),
        "k": int(world["k"]),
        "weekly": weekly.tolist(),
        "optimiser_success": bool(result.scipy_result.success),
        "forecast": forecast[_kept(forecast.shape[0], DRAWS)].tolist(),
        "gain": float(np.mean(gain)),
        "gain_interval": [
            float(np.quantile(gain, (1.0 - LEVEL) / 2.0)),
            float(np.quantile(gain, (1.0 + LEVEL) / 2.0)),
        ],
        "flags": _flags(trace, model),
        "response": {"unmapped": GEO_UNMAPPED},
        "multiplier": None,
        "settings": {
            "setting": "static",
            "media": SETTINGS["static"],
            "l_max": int(world["kernel_length"]),
            "yearly_seasonality": HARMONICS,
            "controls": [],
            "dims": ["geo"],
            "geos": len(names),
            "national": [str(c) for c in world["national"]],
            "plan": "every geo's cells by its own optimiser from the status quo, each national "
            "channel held to the geos' shares of the population",
            "chains": int(posterior.sizes["chain"]),
            "draws": int(posterior.sizes["draw"]),
            "lift_dated": False,
            "lift": GEO_LIFT,
            "gain_level": LEVEL,
            "smoke": smoke,
        },
        "n/a": {},
        "lift_rows": rows,
        "lift_dropped": int(keep.size) - rows,
        "lift_tests": {
            "taken": taken_tests,
            "dropped": tests - taken_tests + int(world["lift_dropped"]),
        },
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
            on_geos = geo(saved)
        if not second and (args.setting != "static" or args.smoke):
            raise SystemExit(f"{path} is a version-1 export, fitted as version 1 fitted it")
        if on_geos and args.setting != "static":
            raise SystemExit(f"{path} is a geo export, which the arm fits with static media alone")
        began, spent = time.perf_counter(), _cpu()
        record: dict[str, Any]
        try:
            if on_geos:
                record = plan_geo(path, args.smoke)
            elif second:
                record = plan_second(path, args.setting, args.smoke)
            else:
                record = plan(path)
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

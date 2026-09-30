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

    uv run --no-project --python 3.12 --with-requirements scripts/pymc_marketing_arm.txt \\
        python scripts/pymc_marketing_arm.py --worlds DIR --out DIR

``scripts/pymc_marketing_arm.txt`` pins the whole environment, and each record names the versions
it ran on and echoes the world's digest, so the bench scores a plan only on the data it was fitted
to.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import arviz as az  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd
from pymc_marketing.mmm import (  # ty: ignore[unresolved-import]
    MMM,
    GeometricAdstock,
    LogisticSaturation,
)

FIRST_MONDAY = "2023-01-02"
HARMONICS = 3
PACKAGES = ("pymc-marketing", "pymc", "pytensor", "arviz", "numpy", "scipy", "pandas")


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
        record["python"] = platform.python_version()
        record["versions"] = {name: importlib.metadata.version(name) for name in PACKAGES}
        target.write_text(json.dumps(record, allow_nan=False))
        print(path.stem, record.get("fit_seconds"), record.get("error"), flush=True)


if __name__ == "__main__":
    main()

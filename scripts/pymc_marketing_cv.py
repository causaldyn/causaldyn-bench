"""The PyMC-Marketing arm's time-slice cross-validation, the tuning its tuned arm is chosen by.

A practitioner picks PyMC-Marketing's spec by how well each candidate forecasts weeks it was not
fitted to, with the library's own ``TimeSliceCrossValidator``. This script does that for one spec
(:class:`pymc_marketing_arm.Spec`) on each version-2 national export of ``--worlds``, the model
built as :mod:`pymc_marketing_arm` builds it (static media, its adstock, seasonality and controls,
the spec's saturation and intercept), and writes one JSON record a world. It plans nothing, and no
record of it is ever scored: the tuning reads the forecasts alone, never the world's truth.

* **The folds.** Every fold forecasts a quarter, :data:`HORIZON` weeks, after a history that grows
  by a quarter from fold to fold, the last fold's quarter the export's last weeks: ``n_init`` is
  the larger of half the history and the history less :data:`YEARS` years, so a 156-week export
  gives four folds from week 104, a 52-week one two from week 26. Each fold is a fresh model fitted
  to its own history (``TimeSliceCrossValidator.run`` with ``mmm=``), its forecast of the quarter
  the posterior predictive over the history and the quarter, the media and controls as observed.
* **The tests.** None: the cross-validation reads the sales alone, at rung 0. A spec is a choice
  of model, made before any test is read; the tuned arm reads its rung's tests as the arm does.
* **The sampler.** :data:`SAMPLER`, 4 chains of 500 tuning and 500 kept draws, seeded by the
  world's seed. CHOICE: half the arm's draws, so the tuning costs about half a fit a fold; a CRPS
  over a quarter needs far fewer draws than a plan's tails.
* **The score.** ``cv.summary.crps()``, the library's mean CRPS of each fold's test weeks and of its
  training weeks, in the outcome's units; the record also carries each in units of the history's
  mean weekly sales, as the bench reads a forecast's CRPS, so worlds of different size pool.

A fold whose fit raised fails the world's record, as the arm records a world that breaks it.

    uv run --no-project --python 3.12 --with-requirements scripts/pymc_marketing_arm.txt \\
        python scripts/pymc_marketing_cv.py --worlds DIR --out DIR \\
        [--saturation logistic|hill] [--intercept static|varying] [--smoke]

``--smoke`` samples 2 chains of 100 tuning and 100 kept draws, to check the plumbing.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import platform
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arm_machine import machine
from geo_export import geo
from pymc_marketing.mmm import MMM, GeometricAdstock  # ty: ignore[unresolved-import]
from pymc_marketing.mmm.time_slice_cross_validation import (  # ty: ignore[unresolved-import]
    TimeSliceCrossValidator,
)
from pymc_marketing_arm import (
    FIRST_MONDAY,
    HARMONICS,
    INTERCEPTS,
    PACKAGES,
    SATURATIONS,
    SECOND,
    Spec,
    _cpu,
)

HORIZON = 13  # a fold's test weeks: the quarter the arm plans
YEARS = 1  # the folds start this many years before the history's end, or at its half
SAMPLER = {"chains": 4, "tune": 500, "draws": 500}
SMOKE = {"chains": 2, "tune": 100, "draws": 100}


def initial(weeks: int) -> int:
    """The first fold's history: the larger of half the weeks and the weeks less :data:`YEARS`
    years, so that the folds' quarters end on the history's last week.

    Raises:
        ValueError: on a history too short for one fold.
    """
    first = max(weeks // 2, weeks - 52 * YEARS)
    first += (weeks - first) % HORIZON  # whole quarters after it
    if weeks - first < HORIZON:
        raise ValueError(f"a history of {weeks} weeks holds no fold of {HORIZON}")
    return first


def validate(path: Path, spec: Spec, smoke: bool) -> dict[str, Any]:
    """One world's time-slice cross-validation at ``spec``."""
    world = np.load(path)
    if int(world["version"]) != SECOND or geo(world):
        raise ValueError("the cross-validation reads a version-2 national export")
    channels = [str(c) for c in world["channels"]]
    controls = [str(c) for c in world["control_names"]]
    weeks = world["sales"].size
    frame = pd.DataFrame(
        {
            "date": pd.date_range(FIRST_MONDAY, periods=weeks, freq="W-MON"),
            **{name: world["spend"][:, c] for c, name in enumerate(channels)},
            **{name: world["controls"][:, j] for j, name in enumerate(controls)},
        }
    )
    sales = pd.Series(world["sales"], name="sales")
    template = MMM(
        date_column="date",
        channel_columns=channels,
        control_columns=controls or None,
        target_column="sales",
        adstock=GeometricAdstock(l_max=int(world["kernel_length"])),
        yearly_seasonality=HARMONICS,
        **spec.components(),
    )
    seed = int(world["seed"])
    first = initial(weeks)
    sampler = (SMOKE if smoke else SAMPLER) | {"random_seed": seed, "progressbar": False}
    cv = TimeSliceCrossValidator(
        n_init=first, forecast_horizon=HORIZON, date_column="date", step_size=HORIZON
    )
    cv.run(frame, sales, sampler_config=sampler, mmm=template)
    table = cv.summary.crps()
    unit = float(np.mean(world["sales"]))
    folds, tested = [], []
    for i, label in enumerate(table["cv"].drop_duplicates()):
        rows = table[table["cv"] == label].set_index("split")["mean_crps"]
        tested.append(float(rows["test"]) / unit)
        folds.append(
            {
                "fold": str(label),
                "history": first + i * HORIZON,
                "test_crps": float(rows["test"]),
                "train_crps": float(rows["train"]),
                "test_crps_scaled": tested[-1],
            }
        )
    return {
        "version": SECOND,
        "seed": seed,
        "digest": str(world["digest"]),
        "family": int(world["family"]),
        "environment": str(world["environment"]),
        "k": int(world["k"]),
        "spec": dataclasses.asdict(spec),
        "folds": folds,
        "mean_test_crps_scaled": float(np.mean(tested)),
        "unit": unit,
        "settings": {
            "n_init": first,
            "horizon": HORIZON,
            "step": HORIZON,
            "sampler": SMOKE if smoke else SAMPLER,
            "lift": "not read: the cross-validation reads the sales alone",
            "unit": "the history's mean weekly sales",
            "smoke": smoke,
        },
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worlds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--saturation", choices=sorted(SATURATIONS), default="logistic")
    parser.add_argument("--intercept", choices=INTERCEPTS, default="static")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    spec = Spec(args.saturation, args.intercept)
    args.out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)
    for path in sorted(args.worlds.glob("world_*.npz")):
        target = args.out / f"{path.stem}.json"
        if target.exists():
            continue
        began, spent = time.perf_counter(), _cpu()
        record: dict[str, Any]
        try:
            record = validate(path, spec, args.smoke)
        except Exception as error:  # a world that breaks the tuning is recorded, not fatal
            world = np.load(path)
            record = {
                "version": SECOND,
                "seed": int(world["seed"]),
                "digest": str(world["digest"]),
                "spec": dataclasses.asdict(spec),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        record["cost"] = {
            "wall_seconds": time.perf_counter() - began,
            "cpu_seconds": _cpu() - spent,
            **machine(),
        }
        if args.smoke:
            record["smoke"] = "smoke, not a result"
        record["python"] = platform.python_version()
        record["versions"] = {name: importlib.metadata.version(name) for name in PACKAGES}
        target.write_text(json.dumps(record, allow_nan=False))
        print(path.stem, record.get("mean_test_crps_scaled"), record.get("error"), flush=True)


if __name__ == "__main__":
    main()

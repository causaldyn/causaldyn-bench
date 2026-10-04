"""Orbit's KTR arm for the scorecard: one record a version-2 export (``causaldyn_bench.scorecard``).

Orbit's KTR (``orbit-ml``, Uber) is a Bayesian regression whose coefficients move over time: a
level and a seasonality fitted first by KTRLite, a maximum a posteriori in Stan, then the level and
the coefficients drawn as kernel-weighted knots by stochastic variational inference in Pyro (Ng,
Wang and Dai 2021, *Bayesian Time Varying Coefficient Model with Applications to Marketing Mix
Modeling*). Orbit ships no budget optimiser, so the arm plans on its fit.

**The model**, as that paper writes a media mix (its equations 2 and 3): the log of each week's
sales on each channel's media, ``log y_t = l_t + s_t + sum_p beta_{t,p} z_{t,p}``, every
``beta_{t,p} >= 0`` (``regressor_sign`` ``+``), and the controls as regressors of either sign
(``=``). The paper's ``z`` is the log of the channel's media. CHOICE: ``z = log(1 + a / s)`` for the
channel's adstock ``a``, its spend through the normalised geometric kernel over the kernel's length
at a retention ``d``, and a scale ``s``: the paper's ``log a`` up to a constant where ``a`` is large
against ``s``, finite at no spend, and near ``a / s`` where ``a`` is small, so ``s`` is where a
channel's returns start to diminish. Each channel's ``z`` is centred over the history: KTR fits its
level before its regression and holds the level near that fit, so a regressor whose mean is not 0
has its mean read as level, and its coefficient shrinks toward 0, by ``var / (var + mean^2)`` for
one regressor. Orbit's own examples regress on regressors of mean 0.

CHOICE: ``(d, s)`` on a 9-point grid, a retention of 0.2, 0.5 or 0.8 times a scale of 0.5, 1 or 2
of the channel's mean positive spend, the same for every channel, chosen by the error, in sales, of
a fit to all but the last 13 weeks on those weeks; the chosen point is fitted again on the whole
history. The seasonality is one of 52 weeks, the world's year, with 3 harmonics, as the
PyMC-Marketing arm's. The sampler runs at KTR's defaults but for the draws it keeps, 400.

**Lift tests** enter as KTR reads an experiment, a prior on the channel's coefficient over the
test's readout (``coef_prior_list``), its dark weeks and their cooldown. CHOICE: its mean is the
coefficient at which the model's sales over the readout, the history's own times
``exp(beta (z_dark - z))``, change by the row's lift, its weekly change times the dark weeks, and
its standard deviation the lift's standard error through that function's slope there. KTR puts
the prior on each week of the window, so each week's is wider by the square root of the window's
weeks, and the window reads as one test. A fit to all but the last 13 weeks reads the rows whose
readouts end before them.

**The plan**: the weekly spend in the box at the budget with the largest posterior mean of the
sales over the quarter and the kernel's tail after it, with the history carried in and nothing spent
after the quarter, each draw's sales its prediction without the observation's noise: KTR's own
forecast of the level, the seasonality and each coefficient over those weeks
(``coefficient_method="smooth"``, KTR's default), read through its internal ``_model.predict`` and
``_get_regression_coefs_matrix``, since the public ``predict`` adds the noise. A grid of 101
points a side of the box, refined by SLSQP from its best points apart. The arm checks that the
prediction it plans on is KTR's own, draw by draw.

Each record carries, beside the plan:

* ``forecast``: the 400 draws of KTR's posterior predictive of the quarter's weekly sales, every
  channel at the status quo and the controls at the quarter's, in the outcome's units;
* ``gain`` and ``gain_interval``: the posterior mean, and the 5 % and 95 % quantiles, of the plan's
  sales less the status quo's over the quarter and the kernel's tail;
* ``flags``: the variational loss over the last tenth of the steps and its relative drift from the
  tenth before, the chosen point's hold-out error, and how far the arm's prediction strays from
  KTR's own;
* ``response``: ``unmapped``. The model multiplies its channels, so no channel's contribution is a
  part of the sales on its own, which ``chc.response`` needs;
* ``settings``, ``versions`` and ``cost``, the wall and CPU seconds of the whole arm.

``--smoke`` fits the grid's middle point alone, at 101 steps and with no hold-out fit, its error
null, to check the plumbing; its records say they are a smoke run, not a result.

    uv run --no-project --python 3.12 --extra-index-url https://download.pytorch.org/whl/cpu \\
        --index-strategy unsafe-best-match --with-requirements scripts/orbit_arm.txt \\
        python scripts/orbit_arm.py --worlds DIR --out DIR [--smoke]

``scripts/orbit_arm.txt`` pins the whole environment, torch as its CPU build from PyTorch's own
index, which ``uv run`` reads from its command line only. Each record names the versions it ran on
and echoes the world's digest, so the bench scores a plan only on the data it was fitted to.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import logging
import math
import platform
import resource
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from orbit.models import KTR  # ty: ignore[unresolved-import]
from scipy.optimize import brentq, minimize

FIRST_MONDAY = "2023-01-02"
PACKAGES = ("orbit-ml", "torch", "pyro-ppl", "cmdstanpy", "numpy", "scipy", "pandas")
SECOND = 2  # the scorecard's export version
RETENTIONS = (0.2, 0.5, 0.8)
MULTIPLES = (0.5, 1.0, 2.0)  # a channel's scale, in its mean positive spends
HOLD = 13  # the weeks a grid point is judged on
YEAR = 52  # the world's year, in weeks
HARMONICS = 3
DRAWS = 400  # the posterior draws KTR keeps, every one of which a record reads
LEVEL = 0.9  # the gain interval's
SIDE = 101  # grid points a side of the box
STARTS = 3  # the grid's best points SLSQP refines
APART = 4  # a grid point this many steps or fewer from a start, along each axis, is no start
CAP = 50.0  # the largest coefficient a lift row's prior is centred on
AGREE = 1e-9  # how near the arm's prediction lies to KTR's own, relative to it
SMOKE = {"num_steps": 101}
MODEL = "log-log KTR (Ng, Wang and Dai 2021)"


def _cpu() -> float:
    """CPU seconds this process and its finished children, KTRLite's Stan runs, have spent."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def _adstock(spend: np.ndarray, retention: float, length: int) -> np.ndarray:
    """Each column's spend through the normalised geometric kernel, the weeks before the first read
    as no spend."""
    kernel = retention ** np.arange(length)
    kernel = kernel / kernel.sum()
    padded = np.vstack([np.zeros((length - 1, spend.shape[1])), spend])
    return np.stack(
        [np.convolve(padded[:, c], kernel, mode="valid") for c in range(spend.shape[1])], axis=1
    )


@dataclasses.dataclass(frozen=True)
class Point:
    """A point of the grid and its hold-out error, None where the point was not judged."""

    retention: float
    multiple: float
    rmse: float | None


class Media:
    """The channels' media at one grid point: ``z = log(1 + a / s)``, centred over the history."""

    def __init__(self, world: Any, retention: float, multiple: float) -> None:
        spend = np.asarray(world["spend"], dtype=float)
        self.retention, self.multiple = retention, multiple
        self.length = int(world["kernel_length"])
        self.scale = multiple * np.array(
            [spend[spend[:, c] > 0.0, c].mean() for c in range(spend.shape[1])]
        )
        self.centre = np.zeros(spend.shape[1])
        self.centre = self.z(spend).mean(axis=0)

    def z(self, spend: np.ndarray) -> np.ndarray:
        """The centred media of a spend path from the history's first week."""
        return np.log1p(_adstock(spend, self.retention, self.length) / self.scale) - self.centre


def _priors(world: Any, media: Media, end: int) -> list[dict[str, Any]]:
    """KTR's prior on each tested channel's coefficient over each readout that ends by week
    ``end``, numbered from 0.

    Raises:
        ValueError: on a row whose ``x`` is not the history's mean spend over its dark weeks, or
            whose change is not the whole of it, as the harness reads a row.
    """
    channels = [str(c) for c in world["channels"]]
    spend = np.asarray(world["spend"], dtype=float)
    sales = np.asarray(world["sales"], dtype=float)
    dark, cooldown = int(world["lift_weeks"]), int(world["lift_cooldown"])
    out = []
    for i, name in enumerate(str(c) for c in world["lift_channel"]):
        first = int(world["lift_start"][i]) - 1
        last = first + dark + cooldown
        if last > end:
            continue
        x, change = float(world["lift_x"][i]), float(world["lift_delta_x"][i])
        mean = float(spend[first : first + dark, channels.index(name)].mean())
        if abs(x - mean) > 1e-9 * abs(mean) or change != -x:
            raise ValueError(f"{name}'s row {i} reads x {x} and {change}; the history spent {mean}")
        tested = spend[:last].copy()
        tested[first : first + dark, channels.index(name)] = 0.0
        moved = (media.z(tested) - media.z(spend[:last]))[first:last, channels.index(name)]
        base = sales[first:last]
        lift = float(world["lift_delta_y"][i]) * dark
        error = float(world["lift_sigma"][i]) * dark

        def excess(
            beta: float, moved: np.ndarray = moved, base: np.ndarray = base, lift: float = lift
        ) -> float:
            """The model's change of the readout's sales at ``beta``, less the row's lift."""
            return float(np.sum(base * np.expm1(beta * moved))) - lift

        if excess(0.0) <= 0.0:
            beta = 0.0
        elif excess(CAP) > 0.0:
            beta = CAP
        else:
            beta = brentq(excess, 0.0, CAP, xtol=1e-12)
        slope = abs(float(np.sum(base * moved * np.exp(beta * moved))))
        out.append(
            {
                "name": f"row_{i}",
                "prior_start_tp_idx": first,
                "prior_end_tp_idx": last,
                "prior_mean": [beta],
                "prior_sd": [error / slope * math.sqrt(last - first)],
                "prior_regressor_col": [name],
            }
        )
    return out


def _frame(world: Any, media: Media, weeks: int) -> pd.DataFrame:
    """The history's first ``weeks`` weeks as KTR reads them: the log of the sales, each channel's
    centred media and the controls by name."""
    channels = [str(c) for c in world["channels"]]
    controls = [str(c) for c in world["control_names"]]
    z = media.z(np.asarray(world["spend"], dtype=float))
    dates = pd.date_range(FIRST_MONDAY, periods=weeks, freq="W-MON")
    return pd.DataFrame(
        {
            "date": dates,
            "sales": np.log(np.asarray(world["sales"], dtype=float)[:weeks]),
            **{name: z[:weeks, c] for c, name in enumerate(channels)},
            **{name: world["controls"][:weeks, j] for j, name in enumerate(controls)},
        }
    )


def _fit(world: Any, frame: pd.DataFrame, priors: list[dict[str, Any]], smoke: bool) -> Any:
    channels = [str(c) for c in world["channels"]]
    controls = [str(c) for c in world["control_names"]]
    model = KTR(
        response_col="sales",
        date_col="date",
        regressor_col=channels + controls,
        regressor_sign=["+"] * len(channels) + ["="] * len(controls),
        seasonality=[YEAR],
        seasonality_fs_order=[HARMONICS],
        coef_prior_list=priors or None,
        seed=int(world["seed"]),
        num_sample=DRAWS,
        estimator="pyro-svi",
        verbose=False,
        **(SMOKE if smoke else {}),
    )
    model.fit(frame)
    return model


def _held(world: Any, media: Media, smoke: bool) -> float:
    """The root mean square error, in sales, of KTR's median forecast of the history's last
    :data:`HOLD` weeks from a fit to the weeks before them."""
    weeks = np.asarray(world["sales"]).size
    whole = _frame(world, media, weeks)
    model = _fit(world, whole.iloc[: weeks - HOLD], _priors(world, media, weeks - HOLD), smoke)
    predicted = model.predict(whole.iloc[weeks - HOLD :], seed=int(world["seed"]))
    sales = np.asarray(world["sales"], dtype=float)[weeks - HOLD :]
    return float(np.sqrt(np.mean((np.exp(predicted["prediction"].to_numpy()) - sales) ** 2)))


def _future(world: Any, media: Media, weekly: np.ndarray, horizon: int) -> pd.DataFrame:
    """The ``horizon`` weeks after the history as KTR reads them, ``weekly`` a week over the
    quarter and nothing after it, the history carried in, and the controls at the quarter's, held
    at its last week's after it."""
    channels = [str(c) for c in world["channels"]]
    controls = [str(c) for c in world["control_names"]]
    spend = np.asarray(world["spend"], dtype=float)
    weeks, planned = spend.shape[0], int(world["planned"])
    path = np.vstack(
        [spend, np.tile(weekly, (planned, 1)), np.zeros((horizon - planned, spend.shape[1]))]
    )
    z = media.z(path)[weeks:]
    quarter = np.asarray(world["future_controls"], dtype=float)
    held = np.vstack([quarter, np.tile(quarter[-1], (horizon - planned, 1))])
    dates = pd.date_range(FIRST_MONDAY, periods=weeks + horizon, freq="W-MON")[weeks:]
    return pd.DataFrame(
        {
            "date": dates,
            **{name: z[:, c] for c, name in enumerate(channels)},
            **{name: held[:, j] for j, name in enumerate(controls)},
        }
    )


def _predicted(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """KTR's prediction of the log sales in each draw and week of ``frame``, without the
    observation's noise, `(draws, weeks)`."""
    model._set_prediction_meta(frame)
    return model._model.predict(
        posterior_estimates=model._posterior_samples,
        df=frame,
        training_meta=model.get_training_meta(),
        prediction_meta=model.get_prediction_meta(),
        include_error=False,
    )["prediction"]


class Worth:
    """Each draw's sales over the quarter and the kernel's tail as a function of the weekly plan:
    ``exp(base + sum_p beta_p z_p)``, its ``base`` and ``beta`` KTR's, its ``z`` affine in the plan
    through the adstock."""

    def __init__(self, world: Any, media: Media, model: Any) -> None:
        channels = [str(c) for c in world["channels"]]
        spend = np.asarray(world["spend"], dtype=float)
        self.planned = int(world["planned"])
        self.horizon = self.planned + media.length - 1
        none = np.zeros(spend.shape[1])
        frame = _future(world, media, none, self.horizon)
        unit = _adstock(
            np.vstack(
                [
                    np.zeros_like(spend),
                    np.ones((self.planned, spend.shape[1])),
                    np.zeros((self.horizon - self.planned, spend.shape[1])),
                ]
            ),
            media.retention,
            media.length,
        )[spend.shape[0] :]
        self.media = media
        self.carry = _adstock(
            np.vstack([spend, np.zeros((self.horizon, spend.shape[1]))]),
            media.retention,
            media.length,
        )[spend.shape[0] :]
        self.reach = unit
        order = list(model._model._regressor_col)
        columns = [order.index(name) for name in channels]
        betas = model._model._get_regression_coefs_matrix(
            model.get_training_meta(),
            model._posterior_samples,
            "smooth",
            date_array=frame["date"],
        )
        self.beta = betas[:, :, columns]
        silent = frame.copy()
        for name in channels:
            silent[name] = 0.0  # the centred media's 0, so the prediction is the base alone
        self.base = _predicted(model, silent)

    def logs(self, weekly: np.ndarray) -> np.ndarray:
        """The log sales of each draw in each week, `(plans, draws, weeks)`, for plans
        `(plans, channels)`."""
        adstock = self.carry[None] + weekly[:, None, :] * self.reach[None]
        z = np.log1p(adstock / self.media.scale) - self.media.centre
        return self.base[None] + np.einsum("dtp,ktp->kdt", self.beta, z)

    def draws(self, weekly: np.ndarray) -> np.ndarray:
        """Each draw's sales over the quarter and the tail, `(plans, draws)`."""
        return np.exp(self.logs(weekly)).sum(axis=-1)

    def mean(self, weekly: np.ndarray, chunk: int = 256) -> np.ndarray:
        """The posterior mean of the sales over the quarter and the tail, `(plans,)`."""
        return np.concatenate(
            [self.draws(weekly[i : i + chunk]).mean(axis=1) for i in range(0, len(weekly), chunk)]
        )


def _plan(worth: Worth, world: Any) -> tuple[np.ndarray, bool]:
    """The weekly plan with the largest posterior mean in the box at the budget: the grid's best
    points, refined by SLSQP. Three channels, as the bench's searched plan takes."""
    lower = np.asarray(world["lower"], dtype=float)
    upper = np.asarray(world["upper"], dtype=float)
    weekly = float(world["budget"]) / int(world["planned"])
    if lower.size != 3:
        raise ValueError(f"the arm's grid plans three channels, not {lower.size}")
    first = np.linspace(lower[0], upper[0], SIDE)
    second = np.linspace(lower[1], upper[1], SIDE)
    i, j = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    third = weekly - first[i] - second[j]
    inside = (third >= lower[2]) & (third <= upper[2])
    if not inside.any():
        raise ValueError("no point of the grid spends the budget inside the box")
    points = np.column_stack([first[i[inside]], second[j[inside]], third[inside]])
    steps = np.column_stack([i[inside], j[inside]])
    values = worth.mean(points)
    reference = float(worth.mean(np.asarray(world["status_quo"], dtype=float)[None])[0])
    starts: list[int] = []
    for k in np.argsort(-values):
        if all(np.abs(steps[k] - steps[s]).max() > APART for s in starts):
            starts.append(int(k))
        if len(starts) == STARTS:
            break
    best, success = points[starts[0]], True
    best_value = float(values[starts[0]])
    for k in starts:
        result = minimize(
            lambda w: -float(worth.mean(w[None])[0]) / reference,
            points[k],
            method="SLSQP",
            bounds=list(zip(lower, upper, strict=True)),
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - weekly) / weekly}],
            options={"ftol": 1e-12, "maxiter": 500},
        )
        candidate = np.clip(result.x, lower, upper)
        value = float(worth.mean(candidate[None])[0])
        if value > best_value and abs(float(np.sum(candidate)) - weekly) <= 1e-9 * weekly:
            best, best_value, success = candidate, value, bool(result.success)
    return best, success


def _flags(model: Any, held: float | None, agree: float) -> dict[str, Any]:
    """The variational loss over the last tenth of the steps, its drift from the tenth before,
    the chosen point's hold-out error and the arm's distance from KTR's own prediction."""
    loss = np.asarray(model.get_training_metrics()["loss_elbo"], dtype=float)
    tenth = max(1, loss.size // 10)
    last, before = float(loss[-tenth:].mean()), float(loss[-2 * tenth : -tenth].mean())
    return {
        "elbo": last,
        "elbo_drift": (last - before) / abs(last),
        "holdout_rmse": held,
        "prediction_gap": agree,
    }


def plan_second(path: Path, smoke: bool) -> dict[str, Any]:
    """A version-2 export's plan, with what the scorecard reads beside it."""
    world = np.load(path)
    seed = int(world["seed"])
    weeks = np.asarray(world["sales"]).size
    points = [(0.5, 1.0)] if smoke else [(d, m) for d in RETENTIONS for m in MULTIPLES]
    grid = [
        Point(d, m, None if smoke else _held(world, Media(world, d, m), smoke)) for d, m in points
    ]
    chosen = min(grid, key=lambda point: -math.inf if point.rmse is None else point.rmse)
    media = Media(world, chosen.retention, chosen.multiple)
    priors = _priors(world, media, weeks)
    model = _fit(world, _frame(world, media, weeks), priors, smoke)

    worth = Worth(world, media, model)
    status_quo = np.asarray(world["status_quo"], dtype=float)
    own = _predicted(model, _future(world, media, status_quo, worth.horizon))
    agree = float(np.max(np.abs(worth.logs(status_quo[None])[0] - own) / np.abs(own)))
    if agree > AGREE:
        raise RuntimeError(f"the arm's prediction strays {agree:.2e} from KTR's own")
    weekly, success = _plan(worth, world)
    gain = worth.draws(weekly[None])[0] - worth.draws(status_quo[None])[0]

    planned = int(world["planned"])
    model.predict(
        _future(world, media, status_quo, worth.horizon).iloc[:planned],
        store_prediction_array=True,
        seed=seed,
    )
    forecast = np.exp(np.asarray(model.prediction_array, dtype=float))
    controls = [str(c) for c in world["control_names"]]
    return {
        "version": SECOND,
        "seed": seed,
        "digest": str(world["digest"]),
        "family": int(world["family"]),
        "environment": str(world["environment"]),
        "k": int(world["k"]),
        "weekly": [float(w) for w in weekly],
        "optimiser_success": success,
        "forecast": forecast.tolist(),
        "gain": float(np.mean(gain)),
        "gain_interval": [
            float(np.quantile(gain, (1.0 - LEVEL) / 2.0)),
            float(np.quantile(gain, (1.0 + LEVEL) / 2.0)),
        ],
        "flags": _flags(model, chosen.rmse, agree),
        "response": {
            "unmapped": "a log-log model multiplies its channels, so no channel's contribution is "
            "a part of the sales on its own"
        },
        "settings": {
            "model": MODEL,
            "retention": chosen.retention,
            "scale_multiple": chosen.multiple,
            "grid": [dataclasses.asdict(point) for point in grid],
            "holdout_weeks": HOLD,
            "kernel_length": media.length,
            "seasonality": YEAR,
            "harmonics": HARMONICS,
            "controls": controls,
            "num_steps": SMOKE["num_steps"] if smoke else "KTR's default",
            "num_sample": DRAWS,
            "lift_priors": len(priors),
            "gain_level": LEVEL,
            "smoke": smoke,
        },
        "n/a": {},
        "lift_rows": int(np.asarray(world["lift_channel"]).size),
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
        help="the grid's middle point alone at 101 steps, to check the plumbing: not a result",
    )
    args = parser.parse_args()
    first, step = map(int, args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    for name in ("orbit", "cmdstanpy"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for path in sorted(args.worlds.glob("world_*.npz"))[first::step]:
        target = args.out / f"{path.stem}.json"
        if target.exists():
            continue
        with np.load(path) as saved:
            if "version" not in saved.files:
                raise SystemExit(f"{path} is a version-1 export; the Orbit arm reads version 2")
        began, spent = time.perf_counter(), _cpu()
        record: dict[str, Any]
        try:
            record = plan_second(path, args.smoke)
        except Exception as error:  # a world that breaks the arm is recorded, not fatal
            world = np.load(path)
            record = {
                "version": SECOND,
                "seed": int(world["seed"]),
                "digest": str(world["digest"]),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "settings": {"model": MODEL},
            }
        record["cost"] = {
            "wall_seconds": time.perf_counter() - began,
            "cpu_seconds": _cpu() - spent,
        }
        if args.smoke:
            record["smoke"] = "smoke, not a result"
        record["python"] = platform.python_version()
        record["versions"] = {name: importlib.metadata.version(name) for name in PACKAGES}
        target.write_text(json.dumps(record, allow_nan=False))
        print(path.stem, record["cost"]["wall_seconds"], record.get("error"), flush=True)


if __name__ == "__main__":
    main()

"""Track P: does the channel-drift monitor keep its false-alarm bound, and catch a moved channel, on
plants CHC did not write?

:func:`chc.gate.channel_drift_evalues` reads per-decision e-values off a deployed plan's logged
Gaussian dither, and :class:`chc.gate.DriftAlarm` runs e-Shiryaev-Roberts on them. While every entry
of the plan's one-step channel lies within its radius of the model's, the alarm sounds by decision
``H`` on at most ``H / A`` of the runs, whatever the model gets wrong in the drift and whatever the
noise law; a clipped decision is read with its draw. The library measures that on plants it wrote.
This track measures it on Track O's two Gymnasium environments, stepped through their own ``step``:
``Pendulum-v1`` about its hanging equilibrium and ``MountainCarContinuous-v0`` about its valley
floor. Both are affine in the action over one step while their state stays inside its own clips,
and both clip the action.

The environments are deterministic, so process noise ``B w``, with ``w`` Gaussian of sd 0.15 of the
actuator bound, is added to the state after every step: it enters after the actuator, so the clip
never sees it. The design is fixed by rule, the same for both environments:

* **plans**: Track O's moderate LQR plan, its offset set so that on the linearisation at rest the
  command averages 0.1 of the bound, *inside*, or 0.8, *on the bound*. Every decision adds a
  Gaussian dither of a quarter of the bound, logs it as drawn, and logs the action as the
  environment applied it;
* **model**: the environment's linearisation. Its drift is wrong away from the equilibrium, which
  the monitor does not need; its channel is the environment's own;
* **radius**: 2% of each entry of the channel, and the residual scale is the process noise's sd per
  state;
* **channels**: as modelled; grown by exactly the radius, the least favourable null, on whose
  growth side the e-values average exactly 1, clipped or not; and moved to 1.25 times the model's
  from the first decision, through the pendulum's mass and the car's power.

**The gate** (plans/26, the 0.11.0 release: the drift monitor's type-I error at most ``alpha``
under continuous monitoring): on both environments, both plans and both nulls, read after every
decision at ``A = 10^3``, the share of runs that alarm by ``H = 100`` decisions does not exceed
``alpha = H / A = 0.1`` beyond Monte Carlo error: its Clopper-Pearson lower bound is at most
``alpha``. Reported beside it: the run length over ``A``, censored at ``10 A``; on the moved
channel, the share caught within 4000 decisions and the delay; on the plan on the bound, the same
alarm with the clipped decisions' e-values set to 0, which is also valid, for contrast; and every
run's first alarm, so that another horizon can be read without a rerun.

Precision: run with ``JAX_ENABLE_X64=1`` (``just track-p``). The plans' gains come from Track O's
design, whose costs pass through JAX.

Requires the ``gym`` extra (``pip install causaldyn-bench[gym]``).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from chc.gate import DecisionLog, DriftAlarm, channel_drift_evalues

from causaldyn_bench.ope_calibration import (
    DISTURBANCE,
    DITHER,
    ENVIRONMENTS,
    Operating,
    clopper_pearson,
    design,
)

TRACK = "P-drift-calibration"
METRIC = "false_alarm_share"
SEED = 20260929
PATHS = 300
ARL = 1_000.0
HORIZON = 100  # the gate's H: on a null, an alarm by H has probability at most H / A
CAP = 10_000  # a null run is censored here
MOVE_STEPS = 4000
CHUNK = 500
RADIUS = 0.02  # of each entry of the channel
PLANS = {"inside": 0.1, "on the bound": 0.8}  # the command's mean at rest, as a share of the bound
CHANNELS = {"modelled": 1.0, "on the edge": 1.0 + RADIUS, "moved": 1.25}
NULLS = ("modelled", "on the edge")
ZEROED = "clipped zeroed"


@dataclass(frozen=True)
class Plan:
    gain: np.ndarray  # (2,)
    offset: float


def plan_at(operating: Operating, share: float) -> Plan:
    """Track O's moderate plan, its offset set so that on the linearisation at rest the command
    averages ``share`` of the bound."""
    gain = design(operating).plans["moderate"].gain
    held = np.linalg.solve(np.eye(2) - operating.a, operating.b)  # the state a unit command holds
    return Plan(gain[0], share * operating.bound * (1.0 - float((gain @ held)[0, 0])))


def set_channel(env: Any, key: str, factor: float) -> None:
    """Scale the environment's one-step channel by ``factor`` and nothing else: the pendulum's
    through its mass, since the torque enters as ``3 / (m l^2)``, the car's through its power."""
    if key == "pendulum":
        env.m = env.m / factor
    else:
        env.power = env.power * factor


def run_path(
    operating: Operating, env: Any, plan: Plan, steps: int, rng: np.random.Generator
) -> Iterator[tuple[DecisionLog, np.ndarray, int]]:
    """A path's decisions from the equilibrium, ``CHUNK`` at a time: the log, with the action as
    the environment applied it and the dither as drawn; the residual against the linearisation,
    one row per decision; and on how many of the steps the environment clipped its state, where
    it is not affine in the action."""
    bound, b = operating.bound, operating.b[:, 0]
    sigma, noise = DITHER * bound, DISTURBANCE * bound
    operating.place(env, np.zeros(2))
    x = operating.read(env)
    for start in range(0, steps, CHUNK):
        rows = min(CHUNK, steps - start)
        drawn = sigma * rng.standard_normal(rows)
        shock = noise * rng.standard_normal(rows)
        applied, saturated = np.empty(rows), np.empty(rows, dtype=bool)
        residual = np.empty((rows, 2))
        state_clips = 0
        for t in range(rows):
            command = float(plan.gain @ x) + plan.offset + drawn[t]
            applied[t] = min(max(command, -bound), bound)
            saturated[t] = applied[t] != command
            env.step(np.array([command]))
            state_clips += operating.clipped(env, 0.0)
            operating.place(env, operating.read(env) + b * shock[t])
            after = operating.read(env)
            residual[t] = after - operating.a @ x - b * applied[t]
            x = after
        density = np.exp(-0.5 * (drawn / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
        yield DecisionLog(applied, density, saturated, drawn), residual, state_clips


def first_alarm(alarm: DriftAlarm, evalues: np.ndarray) -> int | None:
    """The row of ``evalues`` at which ``alarm`` sounds, if it does, found by replaying the rows
    one at a time from the statistic as it stood before them."""
    before = copy.deepcopy(alarm)
    if not alarm.update(evalues):
        return None
    for t in range(evalues.shape[0]):
        if before.update(evalues[t : t + 1]):
            return t
    raise AssertionError("the alarm sounded on the rows together and on none of them alone")


@dataclass(frozen=True)
class NullScore:
    alarmed_by_horizon: float
    interval: tuple[float, float]  # Clopper-Pearson, 95%
    bound: float  # H / A
    run_length_over_arl: float  # a lower bound: runs are censored at CAP
    censored: float


@dataclass(frozen=True)
class MoveScore:
    caught: float  # within MOVE_STEPS
    mean_delay: float | None
    median_delay: float | None


@dataclass(frozen=True)
class Arm:
    plan: str
    channel: str
    reading: str  # "draw", the library's, or ZEROED
    clipped: float  # the share of decisions whose action the environment clipped
    state_clipped: int  # decisions after which the environment clipped its state
    null: NullScore | None
    move: MoveScore | None
    first_alarms: list[int | None]  # per run, None where censored


def gate(arm: Arm) -> bool | None:
    """The 0.11.0 reading: on a null, the library's reading alarms by ``H`` on no more than
    ``alpha = H / A`` of the runs beyond Monte Carlo error. ``None`` for the arms it does not
    read."""
    if arm.null is None or arm.reading != "draw":
        return None
    return arm.null.interval[0] <= arm.null.bound


def _null(first: np.ndarray, cap: int) -> NullScore:
    alarmed = int(np.sum(first <= HORIZON))
    return NullScore(
        alarmed_by_horizon=alarmed / first.size,
        interval=clopper_pearson(alarmed, first.size),
        bound=HORIZON / ARL,
        run_length_over_arl=float(np.mean(np.minimum(first, cap)) / ARL),
        censored=float(np.mean(~np.isfinite(first))),
    )


def _move(first: np.ndarray) -> MoveScore:
    caught = np.isfinite(first)
    return MoveScore(
        caught=float(caught.mean()),
        mean_delay=float(np.mean(first[caught])) if caught.any() else None,
        median_delay=float(np.median(first[caught])) if caught.any() else None,
    )


def run_arm(
    key: str, operating: Operating, plan_name: str, plan: Plan, channel: str, paths: int, steps: int
) -> tuple[dict[str, np.ndarray], float, int]:
    """Each reading's first alarm per path, ``inf`` where none came within ``steps``; the share of
    decisions clipped; and the decisions after which the state was. Path ``p`` draws from
    ``(SEED, env, plan, channel, p)``, and every reading of a path reads the same decisions."""
    b = np.abs(operating.b)
    sigma = DITHER * operating.bound
    scale = b[:, 0] * DISTURBANCE * operating.bound
    readings = ["draw", ZEROED] if plan_name == "on the bound" else ["draw"]
    first = {r: np.full(paths, np.inf) for r in readings}
    clipped = decided = state_clipped = 0
    index = [list(ENVIRONMENTS).index(key), list(PLANS).index(plan_name)]
    for p in range(paths):
        rng = np.random.default_rng([SEED, *index, list(CHANNELS).index(channel), p])
        alarms = {r: DriftAlarm(ARL) for r in readings}
        with closing(operating.make()) as env:
            set_channel(env, key, CHANNELS[channel])
            for chunk, (log, residual, state_clips) in enumerate(
                run_path(operating, env, plan, steps, rng)
            ):
                clipped += int(log.saturated.sum())
                decided += log.saturated.size
                state_clipped += state_clips
                evalues = channel_drift_evalues(
                    log, residual, dither_scale=sigma, radius=RADIUS * b, residual_scale=scale
                )
                for r in readings:
                    if np.isfinite(first[r][p]):
                        continue
                    read = (
                        np.where(log.saturated[:, None], 0.0, evalues) if r == ZEROED else evalues
                    )
                    t = first_alarm(alarms[r], read)
                    if t is not None:
                        first[r][p] = chunk * CHUNK + t + 1
                if all(np.isfinite(first[r][p]) for r in readings):
                    break
    return first, clipped / decided, state_clipped


@dataclass(frozen=True)
class EnvironmentRun:
    name: str
    bound: float
    channel: list[float]
    plans: dict[str, dict[str, Any]]  # gain and offset, per plan
    arms: list[Arm]
    seconds: float


def run_environment(
    key: str,
    paths: int = PATHS,
    cap: int = CAP,
    move_steps: int = MOVE_STEPS,
    progress: Callable[[str], None] | None = None,
) -> EnvironmentRun:
    """Track P on one environment: every plan on every channel, ``paths`` runs each."""
    start = time.perf_counter()
    operating = ENVIRONMENTS[key]()
    plans = {name: plan_at(operating, share) for name, share in PLANS.items()}
    arms = []
    for plan_name, plan in plans.items():
        for channel in CHANNELS:
            null = channel in NULLS
            first, clipped, state_clipped = run_arm(
                key, operating, plan_name, plan, channel, paths, cap if null else move_steps
            )
            for reading, times in first.items():
                arms.append(
                    Arm(
                        plan_name,
                        channel,
                        reading,
                        clipped,
                        state_clipped,
                        _null(times, cap) if null else None,
                        None if null else _move(times),
                        [int(t) if math.isfinite(t) else None for t in times],
                    )
                )
            if progress is not None:
                progress(f"{key} {plan_name} {channel}")
    return EnvironmentRun(
        name=operating.name,
        bound=operating.bound,
        channel=operating.b[:, 0].tolist(),
        plans={
            name: {"gain": plan.gain.tolist(), "offset": plan.offset}
            for name, plan in plans.items()
        },
        arms=arms,
        seconds=time.perf_counter() - start,
    )


def _markdown(runs: Sequence[EnvironmentRun], paths: int, cap: int, x64: bool) -> str:
    lines = [
        "# Track P — the channel-drift monitor on plants CHC did not write",
        "",
        f"{paths} runs per arm, each environment stepped through its own `step` "
        f"(`causaldyn_bench.drift_calibration`, x64 {x64}), with `A = {ARL:g}`. A null run is "
        f"censored at {cap} decisions, a moved channel's at {MOVE_STEPS}. Alarm-rate intervals "
        "are Clopper-Pearson's.",
        "",
        "## The gate",
        "",
        "The type-I error under continuous monitoring: on each null, read after every decision, "
        f"the share of runs alarmed by `H = {HORIZON}` must not exceed `alpha = H / A = "
        f"{HORIZON / ARL:g}` beyond Monte Carlo error, the interval's lower end at most `alpha`.",
        "",
        "| environment | plan | channel | actions clipped | alarmed by `H` | 95% interval | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        for arm in run.arms:
            verdict = gate(arm)
            if verdict is None or arm.null is None:
                continue
            low, high = arm.null.interval
            lines.append(
                f"| {run.name} | {arm.plan} | {arm.channel} | {100 * arm.clipped:.1f}% | "
                f"{arm.null.alarmed_by_horizon:.3f} | [{low:.3f}, {high:.3f}] | "
                f"{'yes' if verdict else '**no**'} |"
            )
    for run in runs:
        state_clipped = sum(arm.state_clipped for arm in run.arms if arm.reading == "draw")
        lines += [
            "",
            f"## {run.name}",
            "",
            f"Actuator bound {run.bound:g}; the model's channel {run.channel}. The environment "
            f"clipped its state after {state_clipped} decisions in all.",
            "",
            "| plan | channel | reading | actions clipped | run length / `A` | censored | caught | "
            "mean delay (median) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for arm in run.arms:
            head = f"| {arm.plan} | {arm.channel} | {arm.reading} | {100 * arm.clipped:.1f}% |"
            if arm.null is not None:
                lines.append(
                    f"{head} ≥ {arm.null.run_length_over_arl:.2f} | "
                    f"{100 * arm.null.censored:.0f}% | | |"
                )
                continue
            assert arm.move is not None
            delay = (
                "—"
                if arm.move.mean_delay is None
                else f"{arm.move.mean_delay:.0f} ({arm.move.median_delay:.0f})"
            )
            lines.append(f"{head} | | {arm.move.caught:.3f} | {delay} |")
    lines += [
        "",
        "Wall-clock per environment is recorded in `track_p.json` and quoted nowhere: it was not "
        "measured on a clean machine.",
    ]
    return "\n".join(lines) + "\n"


def _record(runs: Sequence[EnvironmentRun], paths: int, cap: int, x64: bool) -> dict[str, Any]:
    return {
        "track": TRACK,
        "metric": METRIC,
        "paths": paths,
        "arl": ARL,
        "horizon": HORIZON,
        "cap": cap,
        "move_steps": MOVE_STEPS,
        "seed": SEED,
        "x64": x64,
        "gymnasium": gym.__version__,
        "rule": {
            "disturbance": DISTURBANCE,
            "dither": DITHER,
            "radius": RADIUS,
            "plans": PLANS,
            "channels": CHANNELS,
        },
        "wall_clock": "environments[*].seconds; recorded, not quoted: not measured on a clean "
        "machine",
        "environments": [
            {**asdict(run), "arms": [{**asdict(arm), "gate": gate(arm)} for arm in run.arms]}
            for run in runs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=PATHS)
    parser.add_argument(
        "--environments", nargs="+", choices=list(ENVIRONMENTS), default=list(ENVIRONMENTS)
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    logging.getLogger("chc.gate").setLevel(logging.ERROR)  # an alarm per run is the data here

    x64 = bool(jnp.zeros(()).dtype == jnp.float64)
    runs = [
        run_environment(key, args.paths, progress=lambda line: print(line, flush=True))
        for key in args.environments
    ]
    text = _markdown(runs, args.paths, CAP, x64)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "track_p.md").write_text(text)
    (args.out / "track_p.json").write_text(
        json.dumps(_record(runs, args.paths, CAP, x64), indent=1) + "\n"
    )
    print(text)
    print(f"written to {args.out}/track_p.md and {args.out}/track_p.json")


if __name__ == "__main__":
    main()

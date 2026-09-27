"""Track L: DCBO's three dynamic SCMs -- the regret of an intervention sequence, both ways round.

DCBO (Aglietti, Dhir, Gonzalez, Damoulas, NeurIPS 2021) chooses, at every time step, which
variables to intervene on and at what level, from an observational log plus explorative
interventions: the nearest academic statement of "which lever, how much, when". The SCMs, the oracle
and the regret are :mod:`causaldyn_bench.dcbo_scm`. Here every arm reads the same log per seed and
is scored on the same noise-free SCM:

* ``oracle`` -- the best response per step, refined past the reference's grid; zero by construction.
* ``oracle (reference grid)`` -- the reference's own oracle, a 100-point grid per variable. What it
  loses is what the grid alone leaves, and is why the scored oracle is refined.
* ``DCBO``, ``CBO``, ``ABO``, ``BO`` -- the reference implementation at commit ``85a9bdf``, run by
  ``scripts/dcbo_reference.py`` in an isolated environment and read back from
  ``results/track_l_dcbo.json``. :func:`load_reference` refuses a file drawn from other logs.
* ``CHC-prescribe`` -- :func:`chc.prescribe` on the same log, stated as closely as it allows.

**The reference conditions CBO on a noisier history than DCBO.** After a single-variable
intervention, ``assign_blanket`` (``dcbo/utils/utilities.py``, line 327) fills the intervened
variable's child in the slice -- ``Z_t`` after ``do(X_t)`` -- with a draw from the true SEM under
fresh ``N(0, 1)`` noise, and CBO plays its next steps against that blanket; DCBO plays against
``assigned_blanket_hat``, which leaves the child to the noise-free SEM. Every arm here is scored on
the noise-free history of its own decisions, and the report counts, per method, the runs whose
recorded outcomes the port reproduces. On ``nonstat``, where ``Y_2`` reads ``Z_1``, a CBO run whose
``t = 1`` decision is ``do(X)`` alone was optimised against a history no arm is scored on.

**What** ``prescribe`` **cannot state about these problems.** Each is a property of the facade, not
of the SCMs, and none is patched over here:

1. *Minimise.* The objective is quadratic tracking of a set point. The arm tracks one
   ``SETPOINT_SPAN`` log-ranges below the lowest logged ``Y``, where tracking is minimising
   ``sum_t Y_t`` -- ``(Y - y*)^2 = -2 y* Y + ...`` as ``y* -> -inf`` -- which is DCGO's per-step
   argmin wherever ``Y_t`` is additive in its own step's intervention (``stat``, ``ind``). The span
   is a free parameter the facade forces, so the report also scores a set point ``NEAR_SPAN``
   log-ranges down.
2. *Which variables.* Every lever is set at every step. The arm always plays ``do(X, Z)``, which is
   one of the exploration sets but is not a choice among them.
3. *Experiments.* The fit reads the log and nothing else. The reference methods also spend
   ``number_of_trials`` explorative interventions per step on the true SCM.
4. *The model class.* One control-affine, time-invariant transition, pooled over units and periods
   and Markov in the target. ``ind``'s optimum is an interior bump no affine channel points at;
   ``nonstat`` changes its equations at the change point, and its ``Z_{t-1} -> Y_t`` edge
   (Fig. 3(c)) has no place in a transition whose only state is the target.

**And the other way round.** The CHC-shaped problem is Track M's
(:mod:`causaldyn_bench.allocation`): a weekly media log whose planner chased the season, to a
schedule at matched budget, from the log alone. DCBO cannot run on it. ``Root.__init__``
instantiates the true SEM from its ``sem`` argument and every explorative evaluation reads it
(``dcbo/bases/root.py`` at ``85a9bdf``, lines 63-72, 199-203 and 698): the true SEM is the
interventional oracle, and a log-only problem has none to hand over. The paper says as much --
Appendix G assumes the agent can "repeatedly intervene in the system", which reverts after each
experiment -- and its one real-data study (E.7) fits a GP SCM to the log and queries the fit
instead, which makes DCBO a planner on a fitted model. Even then Eq. (1) is a per-step argmin over a
box per variable: the objective of Track M's myopic arm rather than its horizon arm, with no
constraint across steps in which a matched total budget could be written.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from chc import CausalGraph, Lever, Panel, Target, prescribe

from causaldyn_bench.dcbo_scm import (
    DOMAIN,
    EXPLORATION_SETS,
    HORIZON,
    N_OBSERVATIONS,
    SCMS,
    Array,
    Decision,
    DynamicSCM,
    ExplorationSet,
    oracle,
    sample_log,
    score,
)
from causaldyn_bench.tracks import TrackResult

REFERENCE = Path(__file__).resolve().parents[2] / "results" / "track_l_dcbo.json"
METHODS = ("DCBO", "CBO", "ABO", "BO")
CHC = "CHC-prescribe"
ARMS = ("oracle", "oracle (reference grid)", *METHODS, CHC)
SEEDS = tuple(range(10))
SETPOINT_SPAN = 1000.0
NEAR_SPAN = 1.0


@dataclass(frozen=True)
class ReferenceRun:
    """One reference method on one log: what it implemented, and what it recorded for it."""

    decisions: tuple[Decision, ...]
    reported: tuple[float, ...]
    seconds: float


@dataclass(frozen=True)
class Reference:
    """``scripts/dcbo_reference.py``'s output, checked against this port on the way in."""

    commit: str
    environment: dict[str, str]
    oracle_sets: dict[str, tuple[ExplorationSet, ...]]
    oracle_values: dict[str, tuple[float, ...]]
    runs: dict[str, dict[int, dict[str, ReferenceRun]]]  # scm -> seed -> method


def _exploration_set(names: Sequence[str]) -> ExplorationSet:
    for candidate in EXPLORATION_SETS:
        if tuple(names) == candidate:
            return candidate
    raise ValueError(f"{tuple(names)} is not one of {EXPLORATION_SETS}")


def load_reference(path: Path = REFERENCE) -> Reference:
    """Read the reference's decisions, refusing a file run on other settings or other logs.

    The logs are compared rather than trusted: the pairing between the DCBO arms and the CHC arm is
    that they read the same data, and a file drawn by an older port would pair nothing. Compared to
    a relative ``1e-12``, not bitwise, because the reference ran under another NumPy.
    """
    raw = json.loads(path.read_text())
    settings = raw["settings"]
    expected: dict[str, Any] = {
        "methods": list(METHODS),
        "n_observations": N_OBSERVATIONS,
        "horizon": HORIZON,
        "domain": {name: list(bounds) for name, bounds in DOMAIN.items()},
        "exploration_sets": [list(s) for s in EXPLORATION_SETS],
    }
    for key, value in expected.items():
        if settings[key] != value:
            raise ValueError(
                f"{path} was run with {key} = {settings[key]!r}; the port has {value!r}"
            )
    runs: dict[str, dict[int, dict[str, ReferenceRun]]] = {}
    for scm in SCMS:
        runs[scm.name] = {}
        for key, run in raw["runs"][scm.name].items():
            seed = int(key)
            log = sample_log(scm, seed)
            for name, values in log.items():
                if not np.allclose(run["log"][name], values, rtol=1e-12, atol=1e-12):
                    raise ValueError(f"{path}: {name} in the {scm.name} log of seed {seed} differs")
            runs[scm.name][seed] = {
                method: ReferenceRun(
                    decisions=tuple(
                        Decision(_exploration_set(d["variables"]), tuple(d["levels"]))
                        for d in record["decisions"]
                    ),
                    reported=tuple(d["reported"] for d in record["decisions"]),
                    seconds=record["seconds"],
                )
                for method, record in run["methods"].items()
            }
    return Reference(
        commit=raw["dcbo_commit"],
        environment=raw["environment"],
        oracle_sets={
            name: tuple(_exploration_set(s) for s in record["sets"])
            for name, record in raw["reference_oracle"].items()
        },
        oracle_values={
            name: tuple(record["values"]) for name, record in raw["reference_oracle"].items()
        },
        runs=runs,
    )


def _panel(log: dict[str, Array], seed: int) -> Panel:
    """The log as ``prescribe`` reads it: row ``k`` holds ``Y_{k-1}`` beside step ``k``'s levers.

    ``prescribe`` fits ``x_next = x + f(x, u)`` on consecutive rows of a unit, and ``Y_t`` answers
    the same step's ``X_t, Z_t``, so the target runs one row behind the levers. Row 0 holds
    ``Y_{-1} = 0``, the zero history of Appendix E. The last row repeats the last levers and is
    never read: it has no successor.
    """
    n, horizon = log["Y"].shape
    unit, period = np.divmod(np.arange(n * (horizon + 1)), horizon + 1)
    step = np.minimum(period, horizon - 1)
    lagged = np.concatenate([np.zeros((n, 1)), log["Y"]], axis=1)
    return Panel.from_frame(
        {
            "unit": unit,
            "period": period,
            "Y": lagged[unit, period],
            "X": log["X"][unit, step],
            "Z": log["Z"][unit, step],
        },
        unit="unit",
        time="period",
        seed=seed,
    )


def chc_decisions(
    scm: DynamicSCM, seed: int, *, span: float = SETPOINT_SPAN
) -> tuple[Decision, ...]:
    """``prescribe``'s schedule on the seed's log, planned from the zero history.

    The graph is the SCM's within-slice DAG, from which ``prescribe`` derives the adjustment set.
    Euler, because the SCM is a discrete recursion and the one-step map *is* the model.
    """
    log = sample_log(scm, seed)
    lo, hi = float(np.min(log["Y"])), float(np.max(log["Y"]))
    prescription = prescribe(
        _panel(log, seed),
        levers=[Lever(name, *DOMAIN[name]) for name in ("X", "Z")],
        target=Target("Y", value=lo - span * (hi - lo)),
        horizon=HORIZON,
        adjustment=CausalGraph.from_edges(scm.edges),
        x0=jnp.zeros(1),
        integrator="euler",
        seed=seed,
    )
    magnitudes = np.asarray(prescription.schedule.magnitudes, dtype=float)
    return tuple(Decision(("X", "Z"), (float(x), float(z))) for x, z in magnitudes)


@dataclass(frozen=True)
class ArmRun:
    """One arm on one log, scored."""

    decisions: tuple[Decision, ...]
    outcomes: tuple[float, ...]  # Y_t along the arm's own noise-free path
    regret: tuple[float, ...]
    seconds: float


def _timed_oracle(scm: DynamicSCM, *, refine: bool) -> tuple[tuple[Decision, ...], float]:
    started = time.perf_counter()
    path = oracle(scm, refine=refine)
    return tuple(r.decision for r in path), time.perf_counter() - started


def _timed_chc(scm: DynamicSCM, seed: int) -> tuple[tuple[Decision, ...], float]:
    started = time.perf_counter()
    decisions = chc_decisions(scm, seed)
    return decisions, time.perf_counter() - started


def run_arms(
    scm: DynamicSCM, seeds: Sequence[int], reference: Reference
) -> dict[str, tuple[ArmRun, ...]]:
    """Every arm on every seed's log, each scored against the best response to its own history.

    The oracles do not depend on the log and are recomputed per seed anyway, so every arm's
    wall-clock is a per-seed series of the same shape.
    """
    missing = sorted(set(seeds) - set(reference.runs[scm.name]))
    if missing:
        raise ValueError(
            f"the reference has no {scm.name} run at seeds {missing}; "
            "rerun scripts/dcbo_reference.py with them"
        )
    runs: dict[str, list[ArmRun]] = {arm: [] for arm in ARMS}
    for seed in seeds:
        played = {
            "oracle": _timed_oracle(scm, refine=True),
            "oracle (reference grid)": _timed_oracle(scm, refine=False),
            **{
                method: (run.decisions, run.seconds)
                for method, run in reference.runs[scm.name][seed].items()
            },
            CHC: _timed_chc(scm, seed),
        }
        for arm in ARMS:
            decisions, seconds = played[arm]
            scored = score(scm, decisions)
            runs[arm].append(ArmRun(decisions, scored.outcomes, scored.regret, seconds))
    return {arm: tuple(series) for arm, series in runs.items()}


def track_dcbo(seeds: Sequence[int] = SEEDS, reference: Path = REFERENCE) -> list[TrackResult]:
    """Mean total regret over ``seeds`` for each method, one board per SCM.

    At whatever precision the process runs. The committed report is 64-bit (``just track-l``), and
    the CHC fit moves with the precision, so a float32 row need not match it.
    """
    loaded = load_reference(reference)
    out: list[TrackResult] = []
    for scm in SCMS:
        runs = run_arms(scm, seeds, loaded)
        for arm in (*METHODS, CHC):
            total = float(np.mean([sum(r.regret) for r in runs[arm]]))
            out.append(TrackResult(f"L-dcbo-{scm.name}", arm, "regret", total))
    return out


@dataclass(frozen=True)
class Interval:
    """A mean over seeds with a percentile bootstrap interval."""

    mean: float
    lo: float
    hi: float

    def cell(self) -> str:
        return f"{self.mean:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def paired_intervals(
    samples: dict[str, Array], *, n_boot: int = 10_000, seed: int = 0
) -> dict[str, Interval]:
    """Percentile intervals for each series' mean under ONE resample of seeds shared by all.

    Every arm reads the same logs, so a difference between two of them is only honest under the same
    resample -- the convention of :func:`causaldyn_bench.paper_two.paired_ratio_ci`, for a mean. A
    difference of two arms is passed in as a series of its own.
    """
    shapes = {values.shape for values in samples.values()}
    if len(shapes) != 1:
        raise ValueError(f"paired series need one shape, got {sorted(shapes)}")
    (size,) = shapes.pop()
    idx = np.random.default_rng(seed).integers(0, size, size=(n_boot, size))
    out: dict[str, Interval] = {}
    for name, values in samples.items():
        lo, hi = np.quantile(values[idx].mean(axis=1), [0.025, 0.975])
        out[name] = Interval(float(values.mean()), float(lo), float(hi))
    return out


@dataclass(frozen=True)
class Summary:
    """One SCM's arms over the seeds: the numbers ``track_l.md`` prints."""

    total: dict[str, Interval]  # arm -> total regret
    per_step: dict[str, tuple[float, ...]]  # arm -> mean regret at each step
    difference: Interval  # CHC's total regret minus DCBO's, on the same logs
    lower: int  # seeds at which CHC's total regret is below DCBO's
    higher: int  # ... and above it
    near: float  # CHC's mean total regret with the set point NEAR_SPAN log-ranges down
    replayed: dict[str, int]  # method -> runs whose recorded outcomes the port reproduces


def replays(run: ArmRun, recorded: ReferenceRun) -> bool:
    """Whether the port, playing a reference run's decisions, gets the outcomes it recorded."""
    return bool(np.allclose(run.outcomes, recorded.reported, rtol=1e-12, atol=1e-12))


def summarise(
    scm: DynamicSCM,
    runs: dict[str, tuple[ArmRun, ...]],
    seeds: Sequence[int],
    reference: Reference,
    n_boot: int,
) -> Summary:
    regret = {arm: np.array([r.regret for r in runs[arm]]) for arm in ARMS}
    total = {arm: values.sum(axis=1) for arm, values in regret.items()}
    difference = total[CHC] - total["DCBO"]
    intervals = paired_intervals({**total, "difference": difference}, n_boot=n_boot)
    near = [score(scm, chc_decisions(scm, seed, span=NEAR_SPAN)).total_regret for seed in seeds]
    return Summary(
        total={arm: intervals[arm] for arm in ARMS},
        per_step={arm: tuple(regret[arm].mean(axis=0).tolist()) for arm in ARMS},
        difference=intervals["difference"],
        lower=int((difference < 0.0).sum()),
        higher=int((difference > 0.0).sum()),
        near=float(np.mean(near)),
        replayed={
            method: sum(
                replays(run, reference.runs[scm.name][seed][method])
                for seed, run in zip(seeds, runs[method], strict=True)
            )
            for method in METHODS
        },
    )


def _markdown(
    summaries: dict[str, Summary], *, seeds: Sequence[int], n_boot: int, commit: str, x64: bool
) -> str:
    names = list(summaries)

    def row(label: str, cells: Iterable[str]) -> str:
        return f"| {label} | " + " | ".join(cells) + " |"

    header = [row("arm", names), "|---|" + "---|" * len(names)]
    lines = [
        "# Track L -- DCBO's three dynamic SCMs",
        "",
        "Regret of the intervention sequence against the best intervention per step, each step "
        "scored against the best response to the arm's own history, on the noise-free SCM "
        "(`causaldyn_bench.dcbo_scm`). Every arm reads one observational log per seed: "
        f"N = {N_OBSERVATIONS} series of T = {HORIZON}. Seeds {min(seeds)}..{max(seeds)} "
        f"({len(seeds)}); 95% percentile intervals from {n_boot} paired bootstrap resamples of "
        f"seeds. DCBO reference at commit `{commit[:7]}`, seed s being its replicate s on log s; "
        f"CHC fitted at {'float64' if x64 else 'float32'}.",
        "",
        f"## Total regret over the T = {HORIZON} steps, mean [95%]",
        "",
        *header,
        *(row(arm, (summaries[n].total[arm].cell() for n in names)) for arm in ARMS),
        "",
        f"## Regret per step t = {', '.join(str(t) for t in range(HORIZON))}, mean over seeds",
        "",
        *header,
        *(
            row(arm, (", ".join(f"{v:.4f}" for v in summaries[n].per_step[arm]) for n in names))
            for arm in ARMS
        ),
        "",
        f"## {CHC} against DCBO on the same logs",
        "",
        row("", names),
        "|---|" + "---|" * len(names),
        row("total regret, CHC minus DCBO [95%]", (summaries[n].difference.cell() for n in names)),
        row(
            "seeds CHC lower / higher",
            (f"{summaries[n].lower} / {summaries[n].higher}" for n in names),
        ),
        "",
        "## The set point `prescribe` needs in place of *minimise*",
        "",
        row("target, below the lowest logged Y", names),
        "|---|" + "---|" * len(names),
        row(
            f"{SETPOINT_SPAN:g} log-ranges (the arm above)",
            (f"{summaries[n].total[CHC].mean:.4f}" for n in names),
        ),
        row(f"{NEAR_SPAN:g} log-range", (f"{summaries[n].near:.4f}" for n in names)),
        "",
        "## The reference's recorded outcomes, replayed on the port",
        "",
        "Runs whose every recorded outcome the port reproduces from the same decisions.",
        "",
        row("method", names),
        "|---|" + "---|" * len(names),
        *(
            row(method, (f"{summaries[n].replayed[method]} / {len(seeds)}" for n in names))
            for method in METHODS
        ),
        "",
        "Wall-clock per arm and seed is recorded in `track_l.json` and quoted nowhere: it was not "
        "measured on a clean machine.",
    ]
    return "\n".join(lines) + "\n"


def _run_record(seed: int, run: ArmRun) -> dict[str, Any]:
    return {
        "seed": seed,
        "decisions": [
            {"variables": list(d.variables), "levels": list(d.levels)} for d in run.decisions
        ],
        "outcomes": list(run.outcomes),
        "regret": list(run.regret),
        "seconds": run.seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--boot", type=int, default=10_000, help="paired bootstrap resamples")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    reference = load_reference(args.reference)
    x64 = bool(jnp.zeros(()).dtype == jnp.float64)
    summaries: dict[str, Summary] = {}
    record: dict[str, Any] = {
        "seeds": args.seeds,
        "boot": args.boot,
        "x64": x64,
        "setpoint_span": SETPOINT_SPAN,
        "near_span": NEAR_SPAN,
        "dcbo_commit": reference.commit,
        "dcbo_environment": reference.environment,
        "wall_clock": "runs[*].seconds; recorded, not quoted: not measured on a clean machine",
        "scms": {},
    }
    for scm in SCMS:
        runs = run_arms(scm, args.seeds, reference)
        summary = summarise(scm, runs, args.seeds, reference, args.boot)
        summaries[scm.name] = summary
        record["scms"][scm.name] = {
            **asdict(summary),
            "runs": {
                arm: [
                    _run_record(seed, run) for seed, run in zip(args.seeds, runs[arm], strict=True)
                ]
                for arm in ARMS
            },
        }
        print(scm.name, {arm: round(summary.total[arm].mean, 4) for arm in ARMS}, flush=True)

    text = _markdown(
        summaries, seeds=args.seeds, n_boot=args.boot, commit=reference.commit, x64=x64
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "track_l.md").write_text(text)
    (args.out / "track_l.json").write_text(json.dumps(record, indent=1) + "\n")
    print(text)
    print(f"written to {args.out}/track_l.md and {args.out}/track_l.json")


if __name__ == "__main__":
    main()

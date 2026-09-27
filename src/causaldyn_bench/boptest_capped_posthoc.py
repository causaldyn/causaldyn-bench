"""P3.3 after its gate: three premises of Theorem 5, asked of the live plant. POST HOC.

Nothing here was pre-registered. It was written while the scored run was under way, after the
interim pairs in its journal had been read; it reads the finished ``results.json`` and cannot change
the verdict printed beside it. The verdict says how the block and the taper ordered on the live
plant, not why, and the greedy fill's optimality (Result 72(d)) rests on premises the plant can be
asked about directly. ``just paper-3-live-posthoc`` runs::

    JAX_ENABLE_X64=1 timeout -s INT 7200 uv run python -u \\
        -m causaldyn_bench.boptest_capped_posthoc --url http://127.0.0.1:8000

**Same energy, same estimate?** The estimation term sees a schedule only through its prefix
energy, so two arms that spend the same energy on the same signs end, on average, on the same
estimate. The paired difference of the final estimates, taper minus block, under the gate's
window-stratified bootstrap, answers that from the scored episodes without touching the emulator;
so does the regret split by the side the prior erred on.

**Is exploring priced at one constant ``A``?** The theorem charges every unit of probe energy the
same ``A`` whenever it is spent, and says nothing once that fails. Measured with learning switched
off: the controller holds a fixed estimate -- its prior information so large that no probe moves
it -- while the block's and the taper's schedules each spend the scored budget on the scored run's
own signs. The loss above the same controller's no-probe episode, which the scored run already
holds (its oracle, or a lazy arm), is what the probes cost; per unit energy it is to be read against
the plan's ``A``.

**Where is the regret lowest?** The model's estimation cost is lowest at the true effect. On the
plant the scored run's no-probe controller acts on a FIXED estimate ``m theta_ref``, in every
window, for ``m`` on a grid. ``m = 1`` is the scored oracle and ``m = 1 +- prior_spread`` its two
lazy arms, so the rerun must reproduce those episodes -- the emulator is deterministic -- and
refuses to continue if it does not: the curve is anchored to the scored episodes rather than
measured beside them.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from causaldyn_bench.boptest import DEFAULT_URL, BOPTestClient, is_available
from causaldyn_bench.boptest_capped import (
    DESIGN,
    Arm,
    Design,
    Plant,
    ZoneModel,
    _precision,
    _selected,
    _stratified_interval,
    loss,
    probe_signs,
    run_episode,
)

MULTIPLIERS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)
COST_MULTIPLIERS = (0.5, 1.0, 1.5)  # the two priors and the truth: the no-probe episodes on record
COST_SEEDS = (0, 1)
FROZEN = 1e12  # prior information no probe can move: the estimate stays where it was put
RESULTS = Path("results/boptest_capped/results.json")


def _stored_model(results: Mapping[str, Any]) -> ZoneModel:
    stored = results["model"]
    return ZoneModel(
        bias=float(stored["bias"]),
        pole=float(stored["pole"]),
        weather=tuple(float(c) for c in stored["weather"]),
        weather_mean=tuple(float(c) for c in stored["weather_mean"]),
        weather_scale=tuple(float(c) for c in stored["weather_scale"]),
        authority=float(stored["authority"]),
        residual_sd=float(stored["residual_sd"]),
    )


def _no_probe_losses(results: Mapping[str, Any]) -> dict[tuple[int, int], float]:
    """The scored no-probe episodes by (window, side): 0 is the oracle, +-1 the lazy arms."""
    held = {(row["window"], 0): float(row["loss"]) for row in results["oracle"]}
    held.update({(e["window"], e["sign"]): float(e["loss"]) for e in results["lazy"]})
    return held


def paired_estimates(results: Mapping[str, Any], design: Design = DESIGN) -> dict[str, Any]:
    """Final estimate, taper minus block, per pair: zero on average if energy alone set it."""
    pairs = results["pairs"]
    windows = np.array([p["window"] for p in pairs])
    diff = np.array([p["taper"]["estimate"] - p["block"]["estimate"] for p in pairs])
    rng = np.random.default_rng(design.bootstrap_seed)
    mean, low, high = _stratified_interval(diff, windows, rng, design.bootstrap)
    return {
        "pairs": int(diff.size),
        "mean": mean,
        "low": low,
        "high": high,
        "taper_lower": int(np.sum(diff < 0.0)),
        "taper_estimate": float(np.mean([p["taper"]["estimate"] for p in pairs])),
        "block_estimate": float(np.mean([p["block"]["estimate"] for p in pairs])),
        "effect": float(results["reference"]["effect"]),
    }


def by_prior_sign(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each arm's regret and final estimate, by the side of ``theta_ref`` its prior was on."""
    rows = []
    for sign in (1, -1):
        mine = [p for p in results["pairs"] if p["sign"] == sign]
        lazy = [entry["regret"] for entry in results["lazy"] if entry["sign"] == sign]
        gaps = [p["taper"]["regret"] - p["block"]["regret"] for p in mine]
        rows.append(
            {
                "sign": sign,
                "pairs": len(mine),
                "taper_regret": float(np.mean([p["taper"]["regret"] for p in mine])),
                "block_regret": float(np.mean([p["block"]["regret"] for p in mine])),
                "gap": float(np.mean(gaps)),
                "taper_wins": int(sum(g < 0.0 for g in gaps)),
                "taper_estimate": float(np.mean([p["taper"]["estimate"] for p in mine])),
                "block_estimate": float(np.mean([p["block"]["estimate"] for p in mine])),
                "lazy_regret": float(np.mean(lazy)),
            }
        )
    return rows


def by_window(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each window's gap and the two arms' mean final estimates, beside the curve's lowest point."""
    rows = []
    for oracle in results["oracle"]:
        mine = [p for p in results["pairs"] if p["window"] == oracle["window"]]
        if not mine:
            continue
        gaps = [p["taper"]["regret"] - p["block"]["regret"] for p in mine]
        rows.append(
            {
                "window": oracle["window"],
                "day": oracle["day"],
                "pairs": len(mine),
                "gap": float(np.mean(gaps)),
                "taper_wins": int(sum(g < 0.0 for g in gaps)),
                "taper_estimate": float(np.mean([p["taper"]["estimate"] for p in mine])),
                "block_estimate": float(np.mean([p["block"]["estimate"] for p in mine])),
            }
        )
    return rows


def _fixed_estimate(multiplier: float, effect: float, design: Design) -> tuple[float, int | None]:
    """``multiplier * theta_ref``, computed exactly as the scored run computed any episode it
    already holds -- the lazy arms' ``theta_ref + sign * (spread * theta_ref)`` -- so that a
    reproduction check compares one input, not two roundings of it."""
    for sign in (1, -1):
        if math.isclose(multiplier, 1.0 + sign * design.prior_spread, rel_tol=0.0, abs_tol=1e-12):
            return effect + sign * (design.prior_spread * effect), sign
    return effect + (multiplier - 1.0) * effect, None


def fixed_estimate_curve(
    plant: Plant,
    results: Mapping[str, Any],
    design: Design = DESIGN,
    multipliers: Sequence[float] = MULTIPLIERS,
    *,
    say: Any = print,
) -> list[dict[str, Any]]:
    """The no-probe controller on a fixed estimate, per window, anchored to the scored episodes."""
    if not any(math.isclose(m, 1.0) for m in multipliers):
        raise ValueError("the grid must contain m = 1, the scored oracle every regret is read from")
    model = _stored_model(results)
    effect = float(results["reference"]["effect"])
    noise_sd = float(results["reference"]["noise_sd"])
    held = _no_probe_losses(results)
    ones = np.ones(design.rounds)
    rows: list[dict[str, Any]] = []
    for window, day in enumerate(design.window_days):
        testid = _selected(plant, design)
        try:
            for multiplier in multipliers:
                fixed, sign = _fixed_estimate(multiplier, effect, design)
                arm = Arm("none", 0.0, 0.0, fixed, 1.0, noise_sd, 0.0, fixed=fixed)
                trace = run_episode(plant, testid, model, design, arm, start_day=day, signs=ones)
                value = loss(trace, design)
                side = 0 if math.isclose(multiplier, 1.0) else sign
                anchor = held.get((window, side)) if side is not None else None
                if anchor is not None and not math.isclose(value, anchor, rel_tol=1e-9):
                    raise RuntimeError(
                        f"window {window}, m = {multiplier:g}: loss {value!r} does not reproduce "
                        f"the scored episode's {anchor!r}; the emulator is not the one that ran it"
                    )
                rows.append(
                    {
                        "window": window,
                        "day": day,
                        "multiplier": multiplier,
                        "estimate": fixed,
                        "loss": value,
                        "regret": value - held[(window, 0)],
                        "anchored": anchor is not None,
                    }
                )
                say(f"window {window} m {multiplier:g}: regret {value - held[(window, 0)]:.4f}")
        finally:
            plant.stop(testid)
    return rows


def probe_cost(
    plant: Plant,
    results: Mapping[str, Any],
    design: Design = DESIGN,
    multipliers: Sequence[float] = COST_MULTIPLIERS,
    seeds: Sequence[int] = COST_SEEDS,
    *,
    say: Any = print,
) -> list[dict[str, Any]]:
    """What the scored budget costs as a block and as a taper when nothing is learned from it.

    The estimate is held at ``m theta_ref`` for the whole episode, so the probe changes the loss
    only by what it does to the zone; subtracting the same controller's scored no-probe episode
    leaves the probe's cost. Only multipliers with such an episode on record are accepted.
    """
    model = _stored_model(results)
    effect = float(results["reference"]["effect"])
    noise_sd = float(results["reference"]["noise_sd"])
    mass, scale = float(results["plan"]["mass"]), float(results["plan"]["scale"])
    held = _no_probe_losses(results)
    rows: list[dict[str, Any]] = []
    for window, day in enumerate(design.window_days):
        testid = _selected(plant, design)
        try:
            for multiplier in multipliers:
                fixed, sign = _fixed_estimate(multiplier, effect, design)
                side = 0 if math.isclose(multiplier, 1.0) else sign
                if side is None or (window, side) not in held:
                    raise ValueError(f"no scored no-probe episode at m = {multiplier:g}")
                for seed in seeds:
                    signs = probe_signs(window, seed, design)
                    for schedule in ("block", "taper"):
                        arm = Arm(schedule, mass, scale, fixed, FROZEN, noise_sd, 0.0)
                        trace = run_episode(
                            plant, testid, model, design, arm, start_day=day, signs=signs
                        )
                        spent = float(np.sum(trace.probe**2))
                        cost = loss(trace, design) - held[(window, side)]
                        rows.append(
                            {
                                "window": window,
                                "multiplier": multiplier,
                                "seed": seed,
                                "schedule": schedule,
                                "spent": spent,
                                "cost": cost,
                                "per_energy": cost / spent if spent > 0.0 else float("nan"),
                                "drift": float(np.max(np.abs(trace.estimate - fixed))),
                            }
                        )
                        say(
                            f"window {window} m {multiplier:g} seed {seed} {schedule}: cost "
                            f"{cost:.4f} for energy {spent:.4f}"
                        )
        finally:
            plant.stop(testid)
    return rows


def cost_summary(rows: Sequence[Mapping[str, Any]], curvature: float) -> list[dict[str, Any]]:
    """Per fixed estimate: each shape's cost per unit energy, and block minus taper, paired."""
    table = []
    for multiplier in sorted({row["multiplier"] for row in rows}):
        mine = [row for row in rows if row["multiplier"] == multiplier]
        shape = {s: [row for row in mine if row["schedule"] == s] for s in ("block", "taper")}
        paired = [
            b["per_energy"] - t["per_energy"]
            for b, t in zip(shape["block"], shape["taper"], strict=True)
        ]
        table.append(
            {
                "multiplier": multiplier,
                "episodes": len(paired),
                "block_spent": float(np.mean([r["spent"] for r in shape["block"]])),
                "taper_spent": float(np.mean([r["spent"] for r in shape["taper"]])),
                "block_per_energy": float(np.mean([r["per_energy"] for r in shape["block"]])),
                "taper_per_energy": float(np.mean([r["per_energy"] for r in shape["taper"]])),
                "block_minus_taper": float(np.mean(paired)),
                "block_dearer": int(sum(d > 0.0 for d in paired)),
                "model": curvature,
            }
        )
    return table


def curve_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean regret over windows per multiplier, and where each window's curve is lowest."""
    multipliers = sorted({row["multiplier"] for row in rows})
    windows = sorted({row["window"] for row in rows})
    regret = {(row["window"], row["multiplier"]): float(row["regret"]) for row in rows}
    mean = {m: float(np.mean([regret[(w, m)] for w in windows])) for m in multipliers}
    lowest = {w: min(multipliers, key=lambda m, w=w: regret[(w, m)]) for w in windows}
    return {
        "multipliers": multipliers,
        "windows": windows,
        "mean_regret": [mean[m] for m in multipliers],
        "lowest_mean": min(multipliers, key=lambda m: mean[m]),
        "lowest_by_window": [lowest[w] for w in windows],
    }


def analyse(
    plant: Plant | None, results: Mapping[str, Any], design: Design = DESIGN, *, say: Any = print
) -> dict[str, Any]:
    """All three questions; the two that need the emulator only if a plant is given."""
    rows = fixed_estimate_curve(plant, results, design, say=say) if plant is not None else []
    costs = probe_cost(plant, results, design, say=say) if plant is not None else []
    return {
        "precision": _precision(),
        "scored_precision": results["precision"],
        "decision": results["verdict"]["decision"],
        "effect": float(results["reference"]["effect"]),
        "prbs_authority": float(results["model"]["authority"]),
        "paired": paired_estimates(results, design),
        "by_sign": by_prior_sign(results),
        "by_window": by_window(results),
        "curve": rows,
        "summary": curve_summary(rows) if rows else None,
        "costs": costs,
        "cost_summary": (
            cost_summary(costs, float(results["plan"]["curvature"])) if costs else None
        ),
    }


def markdown(post: Mapping[str, Any]) -> str:
    """The post-hoc file, every number in it read off ``post``."""
    paired, summary = post["paired"], post["summary"]
    lines = [
        "# P3.3 after the gate -- POST HOC, not pre-registered",
        "",
        "Produced by `causaldyn_bench.boptest_capped_posthoc` from the scored run's",
        f"`results.json` (its verdict: **{post['decision']}**, precision "
        f"{post['scored_precision']}); nothing here changes that verdict.",
        "",
        f"- precision of this run: **{post['precision']}** (`JAX_ENABLE_X64`)",
        f"- `theta_ref = {post['effect']:.4f}` K/h; the stage-0 PRBS fit's authority at the target "
        f"`{post['prbs_authority']:.4f}`, i.e. `{post['prbs_authority'] / post['effect']:.3f}` "
        "theta_ref",
        "",
        "## Same energy, same estimate?",
        "",
        f"Final estimate, taper minus block, over {paired['pairs']} pairs: mean "
        f"**{paired['mean']:.4f}** K/h, 95% interval [{paired['low']:.4f}, {paired['high']:.4f}] "
        f"(stratified bootstrap); the taper ended lower in {paired['taper_lower']} of "
        f"{paired['pairs']}. Mean final estimates: taper `{paired['taper_estimate']:.4f}`, block "
        f"`{paired['block_estimate']:.4f}`, against `theta_ref = {paired['effect']:.4f}`.",
        "",
        "| prior | pairs | taper regret | block regret | mean D | taper wins | taper estimate | "
        "block estimate | lazy regret |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in post["by_sign"]:
        side = "above theta_ref" if row["sign"] > 0 else "below theta_ref"
        lines.append(
            f"| {side} | {row['pairs']} | {row['taper_regret']:.4f} | {row['block_regret']:.4f} | "
            f"{row['gap']:.4f} | {row['taper_wins']}/{row['pairs']} | "
            f"{row['taper_estimate']:.4f} | {row['block_estimate']:.4f} | "
            f"{row['lazy_regret']:.4f} |"
        )
    lowest = (
        dict(zip(summary["windows"], summary["lowest_by_window"], strict=True)) if summary else {}
    )
    lines += [
        "",
        "## Per window",
        "",
        "| window | day | mean D | taper wins | taper estimate | block estimate | lowest m |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in post["by_window"]:
        best = f"{lowest[row['window']]:g}" if row["window"] in lowest else "--"
        lines.append(
            f"| {row['window']} | {row['day']} | {row['gap']:.4f} | "
            f"{row['taper_wins']}/{row['pairs']} | {row['taper_estimate']:.4f} | "
            f"{row['block_estimate']:.4f} | {best} |"
        )
    if post["cost_summary"] is not None:
        moved = max(row["drift"] for row in post["costs"])
        lines += [
            "",
            "## Is exploring priced at one constant A?",
            "",
            "Learning switched off: the estimate held at `m theta_ref` (it moved at most "
            f"`{moved:.1e}` K/h), the scored budget spent by each schedule on the scored signs;",
            "cost = loss above the same controller's scored no-probe episode, per unit of probe",
            "energy, against the plan's `A`.",
            "",
            "| m | episodes | block energy | taper energy | block cost / energy | "
            "taper cost / energy | block minus taper | block dearer | model A |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in post["cost_summary"]:
            lines.append(
                f"| {row['multiplier']:g} | {row['episodes']} | {row['block_spent']:.4f} | "
                f"{row['taper_spent']:.4f} | {row['block_per_energy']:.4f} | "
                f"{row['taper_per_energy']:.4f} | {row['block_minus_taper']:.4f} | "
                f"{row['block_dearer']}/{row['episodes']} | {row['model']:.4f} |"
            )
    if summary is not None:
        windows = summary["windows"]
        regret = {(row["window"], row["multiplier"]): row["regret"] for row in post["curve"]}
        lines += [
            "",
            "## Where is the regret lowest?",
            "",
            "Regret of the no-probe controller acting on a fixed `m theta_ref`, against the scored",
            "oracle (`m = 1`), per window [K^2 per episode]; the `m = 1` and lazy-arm episodes",
            "reproduced the scored ones exactly.",
            "",
            "| m | " + " | ".join(f"window {w}" for w in windows) + " | mean |",
            "|---" * (len(windows) + 2) + "|",
        ]
        for m, mean in zip(summary["multipliers"], summary["mean_regret"], strict=True):
            cells = " | ".join(f"{regret[(w, m)]:.4f}" for w in windows)
            lines.append(f"| {m:g} | {cells} | {mean:.4f} |")
        lines += [
            "",
            f"- lowest mean regret at `m = {summary['lowest_mean']:g}`; lowest per window at "
            f"`m = {', '.join(f'{m:g}' for m in summary['lowest_by_window'])}`",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if _precision() != "float64":
        raise SystemExit("JAX_ENABLE_X64=1 is required, as for the scored run it reads")
    if not is_available(args.url):
        raise RuntimeError(f"no BOPTEST-Service at {args.url}; bring it up and pass --url")
    results = json.loads(args.results.read_text())
    client = BOPTestClient(args.url, timeout=600.0)
    post = analyse(client, results, Design())
    out = args.results.parent
    (out / "posthoc.json").write_text(json.dumps(post, indent=2))
    text = markdown(post)
    (out / "posthoc.md").write_text(text)
    print(text)
    print(f"written to {out}/posthoc.md and {out}/posthoc.json")


if __name__ == "__main__":
    main()

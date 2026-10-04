"""Track R: one simulated day of FleetPy's Manhattan case study under a pricing policy.

Runs in the environment ``scripts/fleetpy.txt`` pins, one day per process, with ``FLEETPY`` naming
the clone ``fetch.py`` made:

    FLEETPY=~/.cache/causaldyn-bench/fleetpy/FleetPy uv run --no-project --python 3.12 \\
        --with-requirements scripts/fleetpy.txt python scripts/fleetpy/day.py \\
        --policy '{"kind": "surge"}' --day-file 2018-11-12_sample_5_1.csv \\
        --world 0 --day 0 --vehicles 200 --out runs/surge_w0_d0

FleetPy's ``src`` and this folder's ``dev`` go on the path before FleetPy is imported, since FleetPy
looks for ``dev`` when it is first imported.

**The plant.** FleetPy's immediate-offer pooling operator, ``PoolingIRSOnly``, under
``ImmediateDecisionsSimulation``: each request is offered a pooled ride, if any vehicle can take it
within 480 s and a 40 % detour, the moment it arrives, and decides on it at once. The day's demand
is one of the case study's samples, on its network with the day's travel-time factor, which the
case study's files hold as one network-wide number a day (2.38 to 3.70 over its eight days). The
case study's own operator, Alonso-Mora batching with repositioning, needs a commercial solver, so
this track prices and does not rebalance.

**The fare.** The list fare is the 2018 New York yellow-cab meter's distance tariff, $2.50 plus
$2.50 a mile, in FleetPy's cents and metres. The lever is a factor per origin zone of the
12-minute zone system and per 15-minute period, from ``policies.py``, on ``[0.6, 2]``.

**The riders.** A request takes an offer up to its ratio times the trip's list fare. A world draws
each origin zone's median ratio once, from U[1.1, 1.6], and each request's ratio as that median
times exp(sd N(0, 1)): SeedSequence((world, 0)) for the medians and SeedSequence((world, 1, day))
for a day's ratios, so a day re-run reads the same riders whatever else ran. The law is the
bench's: the case study's demand files hold no price response. Riders carry the operator's wait
limit and no detour limit of their own. FleetPy prices a rider's direct route when it loads the
demand, before the day's travel-time factor applies, so a detour limit set then is too short for
the day's roads, and the operator, which adopts it, would offer almost no one a ride.

**Out:** ``decisions.csv``, one row per request; ``periods.npz``, per period and zone, the counts,
fares and lost list fares, the vehicles idle at the period's start and the factors set, with the
logger's status quo and clip flags when it ran; ``meta.json``, the run's settings and totals.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if "FLEETPY" not in os.environ:
    sys.exit("FLEETPY must name the FleetPy clone fetch.py made")
FLEETPY = Path(os.environ["FLEETPY"]).expanduser().resolve()
sys.path[:0] = [str(FLEETPY), str(HERE)]
os.chdir(FLEETPY)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import policies  # noqa: E402
from dev import pricing  # noqa: E402
from src.misc import config  # noqa: E402  # ty: ignore[unresolved-import]
from src.misc.init_modules import (  # noqa: E402  # ty: ignore[unresolved-import]
    load_simulation_environment,
)

NETWORK = "Manhattan_2019_corrected"
DEMAND_DIR = FLEETPY / "data" / "demand" / "Manhattan_2018" / "matched" / NETWORK
ZONES_CSV = (
    FLEETPY / "data" / "zones" / "Manhattan_corrected_12min_max" / NETWORK / "node_zone_info.csv"
)
STUDY = FLEETPY / "studies" / "causaldyn_track_r"
MEDIANS = (1.1, 1.6)
METRES_A_MILE = 1609.344

CONSTANT = f"""\
sim_env: ImmediateDecisionsSimulation
log_level: warning
network_type: NetworkTTMatrix
network_name: {NETWORK}
demand_name: Manhattan_2018
random_seed: 0
start_time: 0
end_time: 86400
time_step: 30
user_max_decision_time: 0
user_max_wait_time: 480
route_output_flag: False
replay_flag: False
nr_mod_operators: 1
rq_type: ListFareRatioRequest
op_module: PoolingIRSOnly
op_vr_control_func_dict: {{func_key: distance_and_user_times_man, vot: 0.45833, dc: 0.0694}}
op_min_wait_time: 0
op_max_wait_time: 480
op_max_detour_time_factor: 40
op_const_boarding_time: 30
op_add_boarding_time: 0
op_base_fare: 250
op_distance_fare: {250 / METRES_A_MILE!r}
op_time_fare: 0
op_min_standard_fare: 0
op_reoptimisation_timestep: 30
op_dyn_pricing_method: ZonalFactorPricing
"""


def _atomic(path: Path, text: str) -> None:
    """Days run side by side and write the same shared files; a reader sees a whole one."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def zone_of_node() -> np.ndarray:
    table = pd.read_csv(ZONES_CSV)
    out = np.full(int(table["node_index"].max()) + 1, -1, dtype=np.int64)
    out[table["node_index"].to_numpy()] = table["zone_id"].to_numpy()
    return out


def medians(world: int, zones: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence((world, 0))).uniform(*MEDIANS, zones)


def write_demand(day_file: str, world: int, day: int, sd: float) -> str:
    """The day's requests with a ``wtp_ratio`` column, written beside FleetPy's own files."""
    requests = pd.read_csv(DEMAND_DIR / day_file)
    zones = zone_of_node()
    median = medians(world, int(zones.max()) + 1)[zones[requests["start"].to_numpy()]]
    draws = np.random.default_rng(np.random.SeedSequence((world, 1, day))).standard_normal(
        len(requests)
    )
    requests["wtp_ratio"] = median * np.exp(sd * draws)
    name = f"track_r_w{world}_d{day}_sd{sd:g}_{day_file}"
    _atomic(DEMAND_DIR / name, requests.to_csv(index=False))
    return name


def scenario(rq_file: str, tt_file: str, vehicles: int, name: str) -> dict:
    (STUDY / "scenarios").mkdir(parents=True, exist_ok=True)
    constant_file = STUDY / "scenarios" / "constant.yaml"
    _atomic(constant_file, CONSTANT)
    scenario_file = STUDY / "scenarios" / f"{name}.csv"
    _atomic(
        scenario_file,
        "scenario_name,rq_file,nw_dynamic_f,op_fleet_composition\n"
        f"{name},{rq_file},{tt_file},default_vehtype:{vehicles}\n",
    )
    constant = config.ConstantConfig(str(constant_file))
    constant["study_name"] = STUDY.name
    constant["n_cpu_per_sim"] = 1
    constant["evaluate"] = 0
    constant["log_level"] = "warning"
    constant["keep_old"] = False
    (single,) = config.ScenarioConfig(str(scenario_file))
    return constant + single


def run_day(policy, *, name: str, day_file: str, world: int, day: int, vehicles: int, sd: float):
    """One day under ``policy(period, ledger) -> factor per zone``, as FleetPy scenario ``name``,
    which must be unique among runs at once. Returns the ledger and FleetPy's results folder."""
    tt_file = day_file.split("_sample")[0].removesuffix(".csv") + "_mean_tt_factor_hourly.csv"
    rq_file = write_demand(day_file, world, day, sd)
    parameters = scenario(rq_file, tt_file, vehicles, name)
    book = pricing.open_ledger(str(ZONES_CSV))
    sim = load_simulation_environment(parameters)
    sim.operators[0].dyn_pricing.policy = policy
    sim.run()
    return book, STUDY / "results" / name


def per_period(book, periods: int = policies.PERIODS) -> dict[str, np.ndarray]:
    """Each closed period's counts and fare sums as ``(periods, zones)`` arrays, with the vehicles
    idle at its start and the factors set."""
    closed = [book.closed(k) for k in range(periods)]
    out = {name: np.stack([c[name] for c in closed]) for name in pricing.COLUMNS}
    out["idle"] = np.stack([book.idle[k] for k in range(periods)])
    out["factors"] = np.stack([book.factors[k] for k in range(periods)])
    return out


def policy_from(spec: dict, *, world: int, day: int, zones: int):
    kind = spec["kind"]
    if kind == "constant":
        return policies.Constant(float(spec["factor"]))
    if kind == "surge":
        return policies.Surge()
    if kind == "dithered":
        draws = np.random.default_rng(np.random.SeedSequence((world, 2, day))).standard_normal(
            (policies.PERIODS, zones)
        )
        return policies.Dithered(float(spec["scale"]), draws)
    if kind == "schedule":
        return policies.Schedule(np.load(spec["path"]))
    if kind == "switchback":
        arms = tuple(policy_from(arm, world=world, day=day, zones=zones) for arm in spec["arms"])
        return policies.Switchback(arms, np.asarray(spec["assignment"], dtype=np.int64))
    raise ValueError(f"unknown policy kind {kind!r}")


def _logged(policy, name: str, zones: int) -> np.ndarray | None:
    """The logger's record ``name``, per period, from whichever arm ran the logger."""
    for arm in getattr(policy, "arms", (policy,)):
        records = getattr(arm, name, None)
        if records:
            out = np.full((policies.PERIODS, zones), np.nan)
            for period, row in records.items():
                out[period] = row
            return out
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True, help='JSON, e.g. {"kind": "surge"}')
    parser.add_argument("--day-file", required=True, help="a demand sample of the case study")
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--day", type=int, required=True, help="the day's index in its world")
    parser.add_argument("--vehicles", type=int, required=True)
    parser.add_argument("--sd", type=float, default=0.4, help="the log-sd of the ratios")
    parser.add_argument("--out", type=Path, required=True, help="a new folder, unique to this run")
    args = parser.parse_args()
    spec = json.loads(args.policy)
    zones = int(zone_of_node().max()) + 1
    policy = policy_from(spec, world=args.world, day=args.day, zones=zones)
    out = args.out.expanduser().resolve()
    book, _ = run_day(
        policy,
        name=out.name,
        day_file=args.day_file,
        world=args.world,
        day=args.day,
        vehicles=args.vehicles,
        sd=args.sd,
    )
    out.mkdir(parents=True, exist_ok=True)
    book.frame().to_csv(out / "decisions.csv")
    arrays = per_period(book)
    for name in ("base", "clipped"):
        logged = _logged(policy, name, zones)
        if logged is not None:
            arrays[name] = logged
    np.savez(out / "periods.npz", allow_pickle=False, **arrays)
    commit = subprocess.run(
        ["git", "-C", str(FLEETPY), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    meta = {
        "policy": spec,
        "day_file": args.day_file,
        "world": args.world,
        "day": args.day,
        "vehicles": args.vehicles,
        "sd": args.sd,
        "medians": medians(args.world, zones).tolist(),
        "fleetpy_commit": commit,
        "requests": len(book.decisions),
        "offered": int(arrays["offered"].sum()),
        "accepted": int(arrays["accepted"].sum()),
        "fares_cents": float(arrays["fares"].sum()),
        "lost_list_fares_cents": float(arrays["lost_list_fares"].sum()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

"""Track R's plug-in checks, on one day at a constant factor: the plug-ins do what ``day.py`` says.

1. every request of the day is decided once;
2. an offer's fare is the factor times the list fare, rounded down to a cent, as FleetPy rounds;
3. a rider accepts an offer exactly when it is within the rider's limit, so nothing else declines;
4. per zone, the share within the limit is the law's, ``P(ratio >= floor(f L) / L)``, to sampling
   error;
5. the ledger's riders and fares are FleetPy's own, from its user statistics.

    FLEETPY=... uv run --no-project --python 3.12 --with-requirements scripts/fleetpy.txt \\
        python scripts/fleetpy/check_day.py --factor 1 --vehicles 80 --out runs/check

Measured 2026-10-04 at factor 1 and 80 vehicles on 2018-11-12's ``sample_5_1``, world 0: all five
hold, the largest ``|z|`` over the eight zones 1.75, and 4600 riders paying 3 411 893 cents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import day
import numpy as np
import pandas as pd
from scipy.stats import norm

DAY_FILE = "2018-11-12_sample_5_1.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--vehicles", type=int, default=80)
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument("--sd", type=float, default=0.4)
    parser.add_argument("--out", type=Path, required=True, help="a new folder for the report")
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    book, results = day.run_day(
        lambda period, ledger: np.full(ledger.zones, args.factor),
        name=out.name,
        day_file=DAY_FILE,
        world=args.world,
        day=0,
        vehicles=args.vehicles,
        sd=args.sd,
    )
    d = book.frame()
    requests = pd.read_csv(day.DEMAND_DIR / DAY_FILE)
    offered = d[d["offered"]]
    expected = np.floor(args.factor * offered["list_fare"].to_numpy())

    m = day.medians(args.world, book.zones)
    zones = []
    for zone, group in offered.groupby("zone"):
        bound = np.floor(args.factor * group["list_fare"]) / group["list_fare"]
        law = float(np.mean(norm.sf((np.log(bound) - np.log(m[zone])) / args.sd)))
        n, within = len(group), int(group["within_limit"].sum())
        z = (within / n - law) / np.sqrt(law * (1.0 - law) / n)
        zones.append({"zone": int(zone), "n": n, "share": within / n, "law": law, "z": z})

    users = pd.read_csv(results / "1_user-stats.csv")
    served = users[users["operator_id"] == 0]
    report = {
        "1_decided_once": bool(len(d) == len(requests) and d.index.is_unique),
        "2_fare_cent_error": float(np.max(np.abs(offered["offered_fare"].to_numpy() - expected))),
        "3_accept_iff_within_limit": bool((offered["accepted"] == offered["within_limit"]).all()),
        "4_largest_abs_z": max(abs(z["z"]) for z in zones),
        "4_zones": zones,
        "5_riders": [len(served), int(d["accepted"].sum())],
        "5_fares_cents": [
            float(served["fare"].sum()),
            float(d.loc[d["accepted"], "offered_fare"].sum()),
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(report, indent=1))
    passed = (
        report["1_decided_once"]
        and report["2_fare_cent_error"] == 0.0
        and report["3_accept_iff_within_limit"]
        and report["4_largest_abs_z"] < 4.0
        and report["5_riders"][0] == report["5_riders"][1]
        and report["5_fares_cents"][0] == report["5_fares_cents"][1]
    )
    print(json.dumps({key: value for key, value in report.items() if key != "4_zones"}))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

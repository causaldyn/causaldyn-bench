"""Meridian's own transforms on worlds of family 12, recorded for the test that holds them.

Runs Meridian 2.1.0's ``MediaTransformer`` (``meridian/model/transformers.py``),
``AdstockTransformer`` and ``HillTransformer`` (``meridian/model/adstock_hill.py``) in the Meridian
arm's pinned environment (``scripts/meridian_arm.txt``), on its default JAX backend at float64, on
each channel of the family's first pilot worlds at the world's decay and ec, and on the first
world's channels at two corners of the priors' support: decay 0 and ec 0.1, and decay 1 and ec 10.
Each channel is a national model's media, its spend as its media units: Meridian scales it by the
median of its positive weeks, adstocks it over ``max_lag + 1`` weeks and puts it through its Hill at
slope 1. Beside the cases the record holds the defaults Meridian itself reads, ``ModelSpec``'s
``max_lag`` and ``PriorDistribution``'s ``alpha_m``, ``ec_m``, ``slope_m`` and ``roi_m``, and writes
it all to results/meridian_form.json.

The bench's side runs in the bench's environment and runs this script again in the arm's, which
reads the spend from a file the first writes; Meridian is never installed in the bench's.

    uv run python scripts/meridian_form.py [--worlds 3]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "results" / "meridian_form.json"
ARM = ROOT / "scripts" / "meridian_arm.txt"
TIMEOUT = 1800  # seconds, the arm's environment's first install among them
CORNERS = {"low": (0.0, 0.1), "high": (1.0, 10.0)}  # decay and ec at the priors' support's ends


def cases(worlds: int) -> list[dict[str, Any]]:
    """Each channel of the first ``worlds`` pilot worlds at its own parameters, then the first
    world's channels at the low and the high corner of the priors' support, each with its spend."""
    from causaldyn_bench.scorecard.meridian import MERIDIAN, MeridianWorld

    def history(seed: int) -> MeridianWorld:
        drawn = MERIDIAN.world("priors", seed).history
        assert isinstance(drawn, MeridianWorld)
        return drawn

    seeds = MERIDIAN.pilots["priors"][:worlds]
    out = []
    for seed in seeds:
        world = history(seed)
        for c, channel in enumerate(world.channels):
            out.append(
                {
                    "seed": seed,
                    "channel": channel,
                    "corner": None,
                    "decay": world.retention[c],
                    "ec": world.ec[c],
                    "spend": world.spend[:, c].tolist(),
                }
            )
    first = history(seeds[0])
    for corner, (decay, ec) in CORNERS.items():
        for c, channel in enumerate(first.channels):
            out.append(
                {
                    "seed": seeds[0],
                    "channel": channel,
                    "corner": corner,
                    "decay": decay,
                    "ec": ec,
                    "spend": first.spend[:, c].tolist(),
                }
            )
    return out


def _number(tensor: Any) -> float:
    import numpy as np

    return float(np.asarray(tensor))


def transformed(inputs: Path, outputs: Path) -> None:
    """Meridian's side: each case's media scale, adstock and Hill, as Meridian computes them, and
    Meridian's defaults, from ``inputs`` to ``outputs``."""
    import importlib.metadata
    import platform

    import numpy as np
    from meridian import backend  # ty: ignore[unresolved-import]
    from meridian.model import (  # ty: ignore[unresolved-import]
        adstock_hill,
        prior_distribution,
        spec,
        transformers,
    )

    max_lag = spec.ModelSpec().max_lag
    prior = prior_distribution.PriorDistribution()
    slope = _number(prior.slope_m.loc)
    read = []
    for case in json.loads(inputs.read_text()):
        spend = np.asarray(case["spend"], dtype=np.float64)
        media = backend.to_tensor(spend[None, :, None], dtype=backend.float_dtype)
        scaler = transformers.MediaTransformer(
            media, backend.to_tensor(np.ones(1), dtype=backend.float_dtype)
        )
        adstock = adstock_hill.AdstockTransformer(
            alpha=backend.to_tensor([case["decay"]], dtype=backend.float_dtype),
            max_lag=max_lag,
            n_times_output=spend.size,
        ).forward(scaler.forward(media))
        curve = adstock_hill.HillTransformer(
            ec=backend.to_tensor([case["ec"]], dtype=backend.float_dtype),
            slope=backend.to_tensor([slope], dtype=backend.float_dtype),
        ).forward(adstock)
        read.append(
            {
                "scale": _number(scaler.population_scaled_median_m[0]),
                "adstock": np.asarray(adstock)[0, :, 0].tolist(),
                "saturated": np.asarray(curve)[0, :, 0].tolist(),
            }
        )
    defaults = {
        "max_lag": max_lag,
        "alpha_m": [_number(prior.alpha_m.low), _number(prior.alpha_m.high)],
        "ec_m": [
            _number(prior.ec_m.loc),
            _number(prior.ec_m.scale),
            _number(prior.ec_m.low),
            _number(prior.ec_m.high),
        ],
        "slope_m": slope,
        "roi_m": [_number(prior.roi_m.loc), _number(prior.roi_m.scale)],
        "distributions": {
            name: type(getattr(prior, name)).__name__
            for name in ("alpha_m", "ec_m", "slope_m", "roi_m")
        },
    }
    outputs.write_text(
        json.dumps(
            {
                "meridian": importlib.metadata.version("google-meridian"),
                "python": platform.python_version(),
                "jax": importlib.metadata.version("jax"),
                "dtype": str(np.dtype(backend.np_float_dtype)),
                "defaults": defaults,
                "cases": read,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--worlds", type=int, default=3)
    parser.add_argument("--meridian", nargs=2, type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.meridian:
        transformed(*args.meridian)
        return
    asked = cases(args.worlds)
    with tempfile.TemporaryDirectory() as scratch:
        inputs, outputs = Path(scratch) / "in.json", Path(scratch) / "out.json"
        inputs.write_text(json.dumps(asked))
        subprocess.run(
            [
                "timeout",
                str(TIMEOUT),
                "uv",
                "run",
                "--no-project",
                "--python",
                "3.12",
                "--with-requirements",
                str(ARM),
                "python",
                str(Path(__file__).resolve()),
                "--meridian",
                str(inputs),
                str(outputs),
            ],
            check=True,
            env=os.environ | {"JAX_PLATFORMS": "cpu", "MERIDIAN_ENABLE_JAX_X64": "true"},
        )
        read = json.loads(outputs.read_text())
    if read["dtype"] != "float64":
        sys.exit(f"Meridian ran at {read['dtype']}, not float64")
    record = {key: read[key] for key in ("meridian", "python", "jax", "dtype", "defaults")}
    record["cases"] = [
        {key: value for key, value in c.items() if key != "spend"} | r
        for c, r in zip(asked, read["cases"], strict=True)
    ]
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{len(asked)} cases, Meridian {read['meridian']}: {RECORD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

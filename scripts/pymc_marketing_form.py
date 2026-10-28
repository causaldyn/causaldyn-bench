"""PyMC-Marketing's own transforms on worlds of family 13, recorded for the test that holds them.

Runs PyMC-Marketing 1.2.0's ``GeometricAdstock`` (``pymc_marketing/mmm/components/adstock.py``),
whose ``function`` is ``transformers.geometric_adstock`` at the component's own ``normalize``, and
``transformers.logistic_saturation`` (``pymc_marketing/mmm/transformers.py``), in the PyMC-Marketing
arm's pinned environment (``scripts/pymc_marketing_arm.txt``) at float64, on each channel of the
family's first pilot worlds at the world's ``alpha`` and ``lam``, and on the first world's channels
at two corners of the priors' central 99 %: both parameters at their 0.5 % quantile, and both at
their 99.5 %. Each channel's spend is scaled as the MMM's default
``DataDerivedScaling(method="max")`` scales it, by its largest week, adstocked over Track M v2's
kernel length and saturated. Beside the cases the record holds what the components read by
default, ``GeometricAdstock``'s ``normalize`` and the default priors of both components, and
writes it all to results/pymc_marketing_form.json.

The bench's side runs in the bench's environment and runs this script again in the arm's, which
reads the spend from a file the first writes; PyMC-Marketing is never installed in the bench's.

    uv run python scripts/pymc_marketing_form.py [--worlds 3]
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
RECORD = ROOT / "results" / "pymc_marketing_form.json"
ARM = ROOT / "scripts" / "pymc_marketing_arm.txt"
TIMEOUT = 1800  # seconds, the arm's environment's first install among them
CORNERS = {"low": 0.005, "high": 0.995}  # each prior's quantile at the central 99 %'s ends


def cases(worlds: int) -> tuple[int, list[dict[str, Any]]]:
    """The kernel's length, and each channel of the first ``worlds`` pilot worlds at its own
    parameters, then the first world's channels at the low and the high corner of the priors'
    central 99 %, each with its spend."""
    from scipy import stats

    from causaldyn_bench.scorecard.pymc_marketing import ALPHA, LAM, PYMC_MARKETING, parameters

    alpha_prior, lam_prior = stats.beta(*ALPHA), stats.gamma(LAM[0], scale=1.0 / LAM[1])
    seeds = PYMC_MARKETING.pilots["priors"][:worlds]
    out = []
    for seed in seeds:
        history = PYMC_MARKETING.world("priors", seed).history
        alpha, lam = parameters(history.channels, seed)
        for c, channel in enumerate(history.channels):
            out.append(
                {
                    "seed": seed,
                    "channel": channel,
                    "corner": None,
                    "alpha": float(alpha[c]),
                    "lam": float(lam[c]),
                    "spend": history.spend[:, c].tolist(),
                }
            )
    first = PYMC_MARKETING.world("priors", seeds[0]).history
    for corner, q in CORNERS.items():
        for c, channel in enumerate(first.channels):
            out.append(
                {
                    "seed": seeds[0],
                    "channel": channel,
                    "corner": corner,
                    "alpha": float(alpha_prior.ppf(q)),
                    "lam": float(lam_prior.ppf(q)),
                    "spend": first.spend[:, c].tolist(),
                }
            )
    return first.kernel_length, out


def _prior(prior: Any) -> list[Any]:
    """A PyMC-Marketing prior as its distribution's name and its parameters."""
    return [prior.distribution, {name: float(value) for name, value in prior.parameters.items()}]


def transformed(inputs: Path, outputs: Path) -> None:
    """PyMC-Marketing's side: each case's scale, adstock and saturation, as PyMC-Marketing
    computes them, and its components' defaults, from ``inputs`` to ``outputs``."""
    import importlib.metadata
    import platform

    import numpy as np
    import pytensor  # ty: ignore[unresolved-import]
    import pytensor.tensor as pt  # ty: ignore[unresolved-import]
    import pytensor.xtensor as ptx  # ty: ignore[unresolved-import]
    from pymc_marketing.mmm import (  # ty: ignore[unresolved-import]
        GeometricAdstock,
        LogisticSaturation,
    )
    from pymc_marketing.mmm.transformers import (  # ty: ignore[unresolved-import]
        logistic_saturation,
    )

    asked = json.loads(inputs.read_text())
    adstock = GeometricAdstock(l_max=asked["length"])
    saturation = LogisticSaturation()
    read = []
    for case in asked["cases"]:
        spend = np.asarray(case["spend"], dtype=np.float64)
        scale = float(spend.max())
        x = ptx.as_xtensor(pt.as_tensor_variable(spend / scale), dims=("date",))
        carried = adstock.function(x, alpha=case["alpha"], dim="date")
        curve = logistic_saturation(carried, case["lam"])
        read.append(
            {
                "scale": scale,
                "adstock": np.asarray(carried.eval(), dtype=np.float64).tolist(),
                "saturated": np.asarray(curve.eval(), dtype=np.float64).tolist(),
            }
        )
    defaults = {
        "l_max": adstock.l_max,
        "normalize": adstock.normalize,
        "alpha": _prior(adstock.default_priors["alpha"]),
        "lam": _prior(saturation.default_priors["lam"]),
        "beta": _prior(saturation.default_priors["beta"]),
    }
    outputs.write_text(
        json.dumps(
            {
                "pymc_marketing": importlib.metadata.version("pymc-marketing"),
                "python": platform.python_version(),
                "pytensor": importlib.metadata.version("pytensor"),
                "dtype": str(pytensor.config.floatX),
                "defaults": defaults,
                "cases": read,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--worlds", type=int, default=3)
    parser.add_argument("--pymc-marketing", nargs=2, type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.pymc_marketing:
        transformed(*args.pymc_marketing)
        return
    length, asked = cases(args.worlds)
    with tempfile.TemporaryDirectory() as scratch:
        inputs, outputs = Path(scratch) / "in.json", Path(scratch) / "out.json"
        inputs.write_text(json.dumps({"length": length, "cases": asked}))
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
                "--pymc-marketing",
                str(inputs),
                str(outputs),
            ],
            check=True,
            env=os.environ | {"PYTENSOR_FLAGS": "floatX=float64"},
        )
        read = json.loads(outputs.read_text())
    if read["dtype"] != "float64":
        sys.exit(f"PyMC-Marketing ran at {read['dtype']}, not float64")
    record = {key: read[key] for key in ("pymc_marketing", "python", "pytensor", "dtype")}
    record["defaults"] = read["defaults"]
    record["cases"] = [
        {key: value for key, value in c.items() if key != "spend"} | r
        for c, r in zip(asked, read["cases"], strict=True)
    ]
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"{len(asked)} cases, PyMC-Marketing {read['pymc_marketing']}: {RECORD.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()

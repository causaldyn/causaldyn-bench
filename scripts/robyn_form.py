"""Robyn's own transforms on worlds of family 14, recorded for the test that holds them.

Runs Robyn's ``adstock_geometric`` and ``saturation_hill`` (scripts/robyn_form.R) in the Robyn
arm's pinned scratch library on each channel of the family's first pilot worlds, at the world's
decay, shape and gamma, and on the first world's channels at both corners of their genre ranges,
and writes the cases and what Robyn returns to results/robyn_form.json. R reads the spend from a
file this script writes; nothing is installed.

    uv run python scripts/robyn_form.py --library LIB [--worlds 3]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from causaldyn_bench.scorecard.robyn import GENRE, ROBYN, RobynWorld

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "results" / "robyn_form.json"
TIMEOUT = 300  # seconds


def history(seed: int) -> RobynWorld:
    """The history of the family's world ``seed``."""
    drawn = ROBYN.world("genre", seed).history
    assert isinstance(drawn, RobynWorld)
    return drawn


def cases(worlds: int) -> list[dict]:
    """Each channel of the first ``worlds`` pilot worlds at its own parameters, then the first
    world's channels at the low and the high corner of their ranges."""
    seeds = ROBYN.pilots["genre"][:worlds]
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
                    "shape": world.shape[c],
                    "gamma": world.gamma[c],
                }
            )
    for corner, end in (("low", 0), ("high", 1)):
        for channel in history(seeds[0]).channels:
            decay, shape, gamma = (bounds[end] for bounds in GENRE[channel])
            out.append(
                {
                    "seed": seeds[0],
                    "channel": channel,
                    "corner": corner,
                    "decay": decay,
                    "shape": shape,
                    "gamma": gamma,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--library", required=True, help="the Robyn arm's scratch R library")
    parser.add_argument("--worlds", type=int, default=3)
    args = parser.parse_args()
    asked = cases(args.worlds)
    histories = {seed: history(seed) for seed in {c["seed"] for c in asked}}
    with tempfile.TemporaryDirectory() as scratch:
        inputs, outputs = Path(scratch) / "in.json", Path(scratch) / "out.json"
        spends = []
        for case in asked:
            world = histories[case["seed"]]
            spends.append(world.spend[:, world.channels.index(case["channel"])].tolist())
        inputs.write_text(
            json.dumps([c | {"spend": s} for c, s in zip(asked, spends, strict=True)])
        )
        library = str(Path(args.library).resolve())
        subprocess.run(
            [
                "timeout",
                str(TIMEOUT),
                "Rscript",
                "--vanilla",
                str(ROOT / "scripts" / "robyn_form.R"),
                str(inputs),
                str(outputs),
            ],
            check=True,
            env=os.environ | {"R_LIBS": library, "R_LIBS_USER": library},
        )
        read = json.loads(outputs.read_text())
    record = {
        "robyn": read["robyn"],
        "r": read["r"],
        "cases": [c | r for c, r in zip(asked, read["cases"], strict=True)],
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{len(asked)} cases, Robyn {read['robyn']}: {RECORD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

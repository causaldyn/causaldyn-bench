"""Record family 15's worlds: each record ``scripts/simmmulator_worlds.R`` wrote to a directory,
once it parses as the family's demo, by its SHA-256 in the table the family reads them by,
``src/causaldyn_bench/scorecard/simmmulator_worlds.json``.

A world is recorded once: a record whose seed the table holds with another digest is refused, and
the table is left as it was. Seeds outside the family's pilot and scored blocks are refused too.

    uv run python scripts/simmmulator_register.py DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from causaldyn_bench.scorecard import simmmulator
from causaldyn_bench.scorecard.simmmulator import COMMIT, SIMMMULATOR, Record, Simmmulator

TABLE = Path(simmmulator.__file__).with_name(simmmulator.TABLE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    table = json.loads(TABLE.read_text())
    worlds = {int(seed): digest for seed, digest in table["worlds"].items()}
    blocks = [*Simmmulator.pilots.values(), *Simmmulator.scored.values()]
    added = 0
    for path in sorted(args.directory.glob("*.json"), key=lambda p: int(p.stem)):
        seed = int(path.stem)
        if not any(seed in block for block in blocks):
            raise SystemExit(f"{path}: {seed} is no seed of the family's blocks")
        data = path.read_bytes()
        record = Record.parse(json.loads(data))
        if record.seed != seed:
            raise SystemExit(f"{path} holds world {record.seed}")
        digest = hashlib.sha256(data).hexdigest()
        if worlds.setdefault(seed, digest) != digest:
            raise SystemExit(f"{path}: world {seed} is recorded with another digest")
        added += 1
    table = {
        "simmmulator": SIMMMULATOR,
        "commit": COMMIT,
        "worlds": {str(seed): worlds[seed] for seed in sorted(worlds)},
    }
    TABLE.write_text(json.dumps(table, indent=1) + "\n")
    print(f"{added} records read, {len(worlds)} worlds recorded: {TABLE.name}")


if __name__ == "__main__":
    main()

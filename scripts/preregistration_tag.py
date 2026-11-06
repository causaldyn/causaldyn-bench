"""Print the command that time-stamps a pre-registration: a signed tag of this repository whose
message is the file's SHA-256 line, as ``sha256sum`` prints it.

    uv run python scripts/preregistration_tag.py PATH [--name NAME]

The maintainer runs the printed ``git tag -s`` and pushes the tag before the first scored seed is
drawn. This script only prints: it creates no tag and pushes nothing. The tag's name is
``preregistration-<the file's stem>`` unless ``--name`` gives another.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from causaldyn_bench.scorecard.preregistration import tag_command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the command that signs a pre-registration's tag."
    )
    parser.add_argument("path", type=Path, help="the pre-registration file")
    parser.add_argument("--name", help="the tag's name")
    args = parser.parse_args()
    print(tag_command(args.path, args.name or f"preregistration-{args.path.stem}"))


if __name__ == "__main__":
    main()

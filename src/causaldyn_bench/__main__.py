"""CLI: ``python -m causaldyn_bench [--save]`` runs every track and prints the leaderboard."""

from __future__ import annotations

import sys

from causaldyn_bench.leaderboard import format_leaderboard, run_all, save_results


def main() -> None:
    print("causaldyn-bench leaderboard")
    print("=" * 44)
    results = run_all()
    print(format_leaderboard(results))
    if "--save" in sys.argv:
        out = save_results(results)
        print(f"\nsaved snapshot to {out}/leaderboard.md and {out}/leaderboard.json")


if __name__ == "__main__":
    main()

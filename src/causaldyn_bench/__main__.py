"""CLI: ``python -m causaldyn_bench`` runs all five tracks and prints the leaderboard."""

from __future__ import annotations

from causaldyn_bench.leaderboard import format_leaderboard, run_all


def main() -> None:
    print("causaldyn-bench - 5-track leaderboard")
    print("=" * 44)
    print(format_leaderboard(run_all()))


if __name__ == "__main__":
    main()

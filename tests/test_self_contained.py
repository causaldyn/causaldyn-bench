"""Nothing this repository publishes points into the research repository it was developed in.

The research repository is private: its plans, the `plans/<n>` paths, and its numbered results, the
`Result <n>` labels and the `discoveries` notes they live in, are not there for a reader of this one
to follow. A docstring, a test, a script or a committed results file names a result by what it is,
and points at a file a reader can open. Every tracked text file is read, the generated results
included, since a label a producer prints reaches them; the lock files are read by no one, and a
binary holds no text.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# This file is tracked too. `[/]` keeps the pattern's own text from matching it; the other three
# need no such help, since a bracket is neither a digit nor a space.
PRIVATE = re.compile(rb"plans/[0-9]|discoveries[/]|\bResults? [0-9]|\bresult_[0-9]")
LOCKS = (".lock", ".lock.txt")


def _readable(path: Path) -> bool:
    """Not a lock file, and not a binary (a NUL byte, git's test)."""
    return not path.name.endswith(LOCKS) and b"\0" not in path.read_bytes()


def _tracked_text() -> list[Path]:
    """Every file git tracks here that :func:`_readable` lets through."""
    listing = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    paths = [ROOT / name.decode() for name in listing.split(b"\0") if name]
    return [p for p in paths if _readable(p)]


def _pointers(paths: list[Path], root: Path = ROOT) -> list[str]:
    """``file:line`` of every line naming a private path or result."""
    return [
        f"{path.relative_to(root).as_posix()}:{number}"
        for path in sorted(paths)
        for number, line in enumerate(path.read_bytes().splitlines(), start=1)
        if PRIVATE.search(line)
    ]


def test_no_tracked_file_points_into_the_research_repository() -> None:
    assert _pointers(_tracked_text()) == []


def test_the_scan_reads_the_files_it_must() -> None:
    # A scan that read nothing would pass on any tree.
    scanned = {path.relative_to(ROOT).as_posix() for path in _tracked_text()}
    assert {
        "README.md",
        "justfile",
        "src/causaldyn_bench/paper_two.py",
        "results/paper3/tables.json",
        "results/paper2/tables.md",
        "tests/test_self_contained.py",
    } <= scanned
    assert "uv.lock" not in scanned


def test_the_pattern_catches_every_form_of_a_private_label() -> None:
    # Spelled in two pieces each, or this file's own lines would be what the scan finds.
    caught = [
        b"See plans/" + b"24.",
        b"discoveries" + b"/theorems.md",
        b"(Result " + b"52)",
        b"Results " + b"56 and 66",
        b"result_" + b"56_mass",
    ]
    missed = [b"plans/<n>", b"Result <n>", b"results/paper2", b"Resultant 5", b"result_over"]
    assert [bool(PRIVATE.search(text)) for text in caught] == [True] * len(caught)
    assert [bool(PRIVATE.search(text)) for text in missed] == [False] * len(missed)


def test_a_label_anywhere_in_a_line_is_reported_by_file_and_line(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "clean.md").write_text("nothing private\n")
    (tmp_path / "docs" / "notes.py").write_bytes(
        b"x = 1\n# as the scored run found (Result " + b"52)\ny = 2\n"
    )
    found = _pointers(sorted(tmp_path.rglob("*.*")), tmp_path)
    assert found == ["docs/notes.py:2"]


def test_lock_files_and_binaries_are_not_read(tmp_path: Path) -> None:
    files = {
        "uv.lock": b"see plans/" + b"1\n",
        "robyn_arm.lock.txt": b"text\n",
        "figure.png": b"\x89PNG\x00\x01",
        "notes.md": b"text\n",
        "locked.md": b"text\n",
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    assert {name for name in files if _readable(tmp_path / name)} == {"notes.md", "locked.md"}

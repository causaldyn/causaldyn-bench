"""The leaderboard states, under each track's heading, the pre-registration its numbers were read
under, with the file's SHA-256, or that the track was not pre-registered, and its JSON holds the
same beside every number. A page written before its pre-registration changed, or a pre-registered
track whose page or records do not name the file, is caught here."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from causaldyn_bench.leaderboard import (
    EXPLORATORY,
    PREREGISTRATIONS,
    Preregistration,
    save_results,
)
from causaldyn_bench.scorecard.preregistration import digest
from causaldyn_bench.tracks import TrackResult

ROOT = Path(__file__).resolve().parents[1]
STATED = re.compile(
    r"^pre-registration (?P<path>.+), SHA-256 (?P<sha256>[0-9a-f]{64}), "
    r"frozen in (?P<commit>\S+), (?:signed tag (?P<tag>\S+)|no signed tag)$"
)


def _declared(track: str, root: Path) -> dict[str, Any] | None:
    """What a track's records must hold: its declared file, at that file's SHA-256 now."""
    entry = PREREGISTRATIONS.get(track)
    if entry is None:
        return None
    sha256 = digest(root / entry.path)
    return {"path": entry.path, "sha256": sha256, "commit": entry.commit, "tag": entry.tag}


def _misstated(board: str, rows: list[dict[str, Any]], root: Path) -> list[str]:
    """Each track whose page or records do not state its declared pre-registration at the file's
    SHA-256 now, or do not state that it has none."""
    found = []
    sections = re.split(r"^## ", board, flags=re.MULTILINE)[1:]
    tracks = [section.split("  (metric: ", 1)[0] for section in sections]
    for track, section in zip(tracks, sections, strict=True):
        stated = [s for s in section.splitlines() if s == EXPLORATORY or STATED.match(s)]
        if len(stated) != 1:
            found.append(f"{track}: {len(stated)} pre-registration lines")
            continue
        match = STATED.match(stated[0])
        if (None if match is None else match.groupdict()) != _declared(track, root):
            found.append(f"{track}: the page states another")
    for row in rows:
        if "preregistration" not in row or row["preregistration"] != _declared(row["track"], root):
            found.append(f"{row['track']}: {row['method']}'s record states another")
    found += [
        f"{track}: declared, not on the page" for track in PREREGISTRATIONS if track not in tracks
    ]
    return found


def test_every_committed_section_states_the_preregistration_its_numbers_were_read_under() -> None:
    board = (ROOT / "results" / "leaderboard.md").read_text()
    rows = json.loads((ROOT / "results" / "leaderboard.json").read_text())
    assert _misstated(board, rows, ROOT) == []
    # A page with no section would pass the reading above.
    assert board.count("\n## ") == len({row["track"] for row in rows}) > 1


def test_a_preregistered_track_shows_its_files_sha256_as_the_file_stands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    design = Path("design.md")
    design.write_text("# The design, fixed before the first scored seed\n")
    tagged = Preregistration("design.md", "abc1234", "preregistration-design")
    monkeypatch.setitem(PREREGISTRATIONS, "A-onestep", tagged)
    monkeypatch.setitem(PREREGISTRATIONS, "C-effect", Preregistration("design.md", "abc1234"))
    results = [
        TrackResult("A-onestep", "m1", "rmse", 0.1),
        TrackResult("A-onestep", "m2", "rmse", 0.2),
        TrackResult("C-effect", "m3", "ate_error", 0.3),
        TrackResult("B-rollout", "m4", "rmse", 0.4),
    ]

    def written() -> tuple[str, list[dict[str, Any]]]:
        out = save_results(results)
        board = (out / "leaderboard.md").read_text()
        return board, json.loads((out / "leaderboard.json").read_text())

    board, rows = written()
    fixed = digest(design)
    assert _misstated(board, rows, tmp_path) == []
    stamp = f"pre-registration design.md, SHA-256 {fixed}, frozen in abc1234"
    assert f"{stamp}, signed tag preregistration-design" in board.splitlines()
    assert f"{stamp}, no signed tag" in board.splitlines()
    assert EXPLORATORY in board.split("## B-rollout", 1)[1].splitlines()
    assert [row["preregistration"] is None for row in rows] == [False, False, False, True]

    design.write_text("# The design, changed after its tag\n")
    stale = [
        "A-onestep: the page states another",
        "C-effect: the page states another",
        "A-onestep: m1's record states another",
        "A-onestep: m2's record states another",
        "C-effect: m3's record states another",
    ]
    assert _misstated(board, rows, tmp_path) == stale
    board, rows = written()
    assert _misstated(board, rows, tmp_path) == [] and fixed not in board

    unstated = "\n".join(s for s in board.splitlines() if "signed tag preregistration" not in s)
    bare = [{k: v for k, v in row.items() if k != "preregistration"} for row in rows]
    assert _misstated(unstated, bare, tmp_path) == [
        "A-onestep: 0 pre-registration lines",
        *(f"{row['track']}: {row['method']}'s record states another" for row in rows),
    ]
    monkeypatch.delitem(PREREGISTRATIONS, "C-effect")
    assert _misstated(board, rows, tmp_path) == [
        "C-effect: the page states another",
        "C-effect: m3's record states another",
    ]
    monkeypatch.setitem(PREREGISTRATIONS, "Z-absent", tagged)
    assert _misstated(board, rows, tmp_path)[-1] == "Z-absent: declared, not on the page"

    design.unlink()
    with pytest.raises(FileNotFoundError):
        save_results(results)

"""A run's pre-registration stamp and the command that time-stamps the file: the stamp is the
file's SHA-256 on one line, a reader holds it to the file, and the tag's message is the file's line
as ``sha256sum`` prints it."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from causaldyn_bench.scorecard.preregistration import Stamp, digest, tag_command, verify

ROOT = Path(__file__).resolve().parents[1]


def _preregistration(tmp_path):
    path = tmp_path / "m2.md"
    path.write_text("# The design, fixed before the first scored world\n")
    return path


def test_a_stamp_is_the_files_sha256_and_a_reader_holds_it_to_the_file(tmp_path):
    path = _preregistration(tmp_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    stamp = Stamp.of(path)
    assert stamp == Stamp(str(path), actual) and digest(path) == actual
    assert Stamp.from_line(f"a report\n{stamp.line}\nits cells") == stamp
    assert Stamp.from_json(stamp.as_json()) == stamp
    assert verify(stamp) == stamp
    assert verify(Stamp(path.name, actual), tmp_path) == Stamp(path.name, actual)
    with pytest.raises(ValueError, match="the file there is"):
        verify(Stamp(str(path), "0" * 64))
    with pytest.raises(FileNotFoundError):
        verify(Stamp(path.name, actual))
    with pytest.raises(ValueError, match="one pre-registration line, not 2"):
        Stamp.from_line(f"{stamp.line}\n{stamp.line}")


def test_the_tags_command_carries_the_files_line_as_sha256sum_prints_it(tmp_path):
    path = _preregistration(tmp_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    command = tag_command(path, "preregistration-m2")
    assert command == f"git tag -s preregistration-m2 -m '{actual}  {path}'"
    printed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preregistration_tag.py"), str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert printed == f"git tag -s preregistration-m2 -m '{actual}  {path}'\n"
    for name in ("-d", "a b", "x..y", "x.lock", "x/"):
        with pytest.raises(ValueError, match="not a plain tag name"):
            tag_command(path, name)
    with pytest.raises(FileNotFoundError):
        tag_command(tmp_path / "absent.md", "preregistration-absent")


def test_a_stamp_is_a_sha256_on_one_line():
    with pytest.raises(ValueError, match="64 lowercase hex digits"):
        Stamp("m2.md", "ABC")
    with pytest.raises(ValueError, match="one line"):
        Stamp("m2\n.md", "0" * 64)
    assert Stamp("m2.md", "0" * 64).line == "pre-registration m2.md, SHA-256 " + "0" * 64

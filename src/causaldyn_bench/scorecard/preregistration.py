"""The pre-registration a scored run ran under, as its report prints it: the file a run declares,
and the SHA-256 of its bytes.

A scored run is read against the design it fixed before its first scored world was drawn, so its
report says which design that was: the path the run declares and the file's SHA-256, in the
report's header and in its JSON (:class:`Stamp`). A reader holds the report to the file by
:func:`verify`, which refuses a missing file and a hash that is not the file's.

The file's hash is made public before the first scored seed is drawn: the maintainer signs a tag of
this repository whose message is the file's line as ``sha256sum`` prints it, ``<sha256>  <file>``
(:func:`tag_command`; ``scripts/preregistration_tag.py`` prints the command and never runs it).
The tag's signature dates the hash; the report's stamp names it; a stamp that is not the tag's
hash was read under another file.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HEADER = "pre-registration"  # what a report's header line starts with
LINE = re.compile(rf"^{HEADER} (?P<path>.+), SHA-256 (?P<sha256>[0-9a-f]{{64}})$")
TAG_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")  # a name git takes as a tag's, unquoted


def digest(path: Path | str) -> str:
    """The SHA-256 of the file's bytes, as ``sha256sum`` prints it.

    Raises:
        FileNotFoundError: no file is at ``path``.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Stamp:
    """The pre-registration a report was read under: the path its run declared and the SHA-256 of
    the file there when the report was written."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError(f"a SHA-256 is 64 lowercase hex digits, not {self.sha256!r}")
        if not self.path or "\n" in self.path:
            raise ValueError(f"a pre-registration's path is one line, not {self.path!r}")

    @classmethod
    def of(cls, path: Path | str) -> Stamp:
        """The stamp of the file a run declares.

        Raises:
            FileNotFoundError: no file is at ``path``.
        """
        return cls(str(path), digest(path))

    @property
    def line(self) -> str:
        """The report's header line."""
        return f"{HEADER} {self.path}, SHA-256 {self.sha256}"

    def as_json(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Stamp:
        return cls(str(data["path"]), str(data["sha256"]))

    @classmethod
    def from_line(cls, text: str) -> Stamp:
        """The stamp a report's header printed, read off the report's text.

        Raises:
            ValueError: the text holds no pre-registration line, or more than one.
        """
        found = [m for m in map(LINE.match, text.splitlines()) if m is not None]
        if len(found) != 1:
            raise ValueError(f"a report prints one pre-registration line, not {len(found)}")
        return cls(found[0]["path"], found[0]["sha256"])


def verify(stamp: Stamp, root: Path | None = None) -> Stamp:
    """``stamp``, once the file at its path, read under ``root`` where the path is relative,
    hashes to it.

    Raises:
        FileNotFoundError: no file is at the stamp's path.
        ValueError: the file's SHA-256 is not the stamp's.
    """
    path = Path(stamp.path)
    if root is not None and not path.is_absolute():
        path = root / path
    actual = digest(path)
    if actual != stamp.sha256:
        raise ValueError(
            f"the report was read under {stamp.path} at SHA-256 {stamp.sha256}; the file there is "
            f"{actual}"
        )
    return stamp


def tag_command(path: Path | str, name: str) -> str:
    """The command that signs the tag ``name``, whose message is ``path``'s line as ``sha256sum``
    prints it, so ``git tag -v`` checks the signature and ``sha256sum -c`` the file.

    Raises:
        FileNotFoundError: no file is at ``path``.
        ValueError: ``name`` is not a plain tag name.
    """
    if not TAG_NAME.fullmatch(name) or ".." in name or name.endswith((".", "/", ".lock")):
        raise ValueError(f"{name!r} is not a plain tag name")
    message = f"{digest(path)}  {path}"
    return f"git tag -s {name} -m {shlex.quote(message)}"

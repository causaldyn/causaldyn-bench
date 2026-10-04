"""Track R's plant: FleetPy at a pinned commit, with its Manhattan case study's input data.

FleetPy (Engelhardt et al., TUM-VT, MIT licence, arXiv:2207.14246) is cloned at ``COMMIT``. The
Manhattan case study's input data (Engelhardt and Dandl 2025, Zenodo 10.5281/zenodo.15187906,
CC BY 4.0) is downloaded, checked against the MD5 Zenodo publishes for it, and unpacked into the
clone's ``data`` folder, as FleetPy's README asks. Neither is copied into this repository.

    uv run --no-project --python 3.12 --with-requirements scripts/fleetpy.txt \\
        python scripts/fleetpy/fetch.py --into ~/.cache/causaldyn-bench/fleetpy

The archive is 409 MB and unpacks to 1.0 GB. A second run finds both in place and does nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

REPOSITORY = "https://github.com/TUM-VT/FleetPy.git"
COMMIT = "c813728b6d188df8dd7fd5d3d36dcb0a55c898a2"
DATA_URL = "https://zenodo.org/api/records/15187906/files/FleetPy_Manhattan.zip/content"
DATA_MD5 = "8b11882ae9c6d87f666bf6e006806744"
ARCHIVE_ROOT = "FleetPy_Manhattan"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def clone(into: Path) -> Path:
    fleetpy = into / "FleetPy"
    if not (fleetpy / ".git").is_dir():
        _git("clone", "--quiet", REPOSITORY, str(fleetpy))
    _git("-C", str(fleetpy), "checkout", "--quiet", "--detach", COMMIT)
    head = _git("-C", str(fleetpy), "rev-parse", "HEAD")
    if head != COMMIT:
        raise RuntimeError(f"{fleetpy} is at {head}, not {COMMIT}")
    return fleetpy


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(into: Path) -> Path:
    archive = into / "FleetPy_Manhattan.zip"
    if archive.is_file() and md5(archive) == DATA_MD5:
        return archive
    partial = archive.with_name(archive.name + ".part")
    with urllib.request.urlopen(DATA_URL, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    found = md5(partial)
    if found != DATA_MD5:
        partial.unlink()
        raise RuntimeError(f"{DATA_URL}: MD5 {found}, Zenodo publishes {DATA_MD5}")
    partial.replace(archive)
    return archive


def unpack(archive: Path, fleetpy: Path) -> None:
    """Every member of the archive's root folder into ``data``, none outside it."""
    data = fleetpy / "data"
    marker = data / ".manhattan.md5"
    if marker.is_file() and marker.read_text() == DATA_MD5:
        return
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = Path(member.filename).relative_to(ARCHIVE_ROOT)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{archive}: member {member.filename!r} leaves the data folder")
            target = data / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out, 1 << 20)
    marker.write_text(DATA_MD5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--into", type=Path, required=True, help="a folder for FleetPy and the data"
    )
    args = parser.parse_args()
    into = args.into.expanduser().resolve()
    into.mkdir(parents=True, exist_ok=True)
    fleetpy = clone(into)
    unpack(download(into), fleetpy)
    print(fleetpy)


if __name__ == "__main__":
    main()

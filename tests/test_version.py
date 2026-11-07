"""The package names the version its metadata and its citation file name."""

from __future__ import annotations

import tomllib
from pathlib import Path

import causaldyn_bench

ROOT = Path(__file__).resolve().parent.parent


def test_the_package_version_is_the_project_s_and_the_citation_s() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert causaldyn_bench.__version__ == project["version"]
    assert f"\nversion: {project['version']}\n" in citation

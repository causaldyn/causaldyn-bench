"""Where a tool's arm ran, as the scorecard's compute table reads a record's ``cost``
(:mod:`causaldyn_bench.scorecard.report`): the CPU's model, the threads its process was given,
``OMP_NUM_THREADS`` as the harness sets it, and the peak resident memory so far, the largest of the
process's own and its finished children's, a sampler's chains among them.

Kept apart from the arms, which import their tools, so the bench's tests read it in the bench's
environment. The standard library alone; Linux reports a peak in KiB, the record in bytes.
"""

from __future__ import annotations

import os
import platform
import resource
from pathlib import Path
from typing import Any


def cpu_model() -> str | None:
    """The CPU's model as the kernel names it, or the platform's processor where it does not."""
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        return platform.processor() or None
    named = (line.split(":", 1)[1].strip() for line in lines if line.startswith("model name"))
    return next(named, None)


def machine() -> dict[str, Any]:
    """The fields a record's ``cost`` holds beside its seconds."""
    peak = max(
        resource.getrusage(who).ru_maxrss
        for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    )
    return {
        "cpu_model": cpu_model(),
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "peak_rss_bytes": 1024 * peak,
    }

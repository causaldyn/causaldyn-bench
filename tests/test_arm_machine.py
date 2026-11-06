"""Where a tool's arm ran, as its record's ``cost`` holds it, from ``scripts/arm_machine.py``
without any tool: the CPU's model, the threads its process was given, and its peak memory in
bytes."""

import importlib.util
import resource
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "arm_machine.py"
_SPEC = importlib.util.spec_from_file_location("arm_machine", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
arm_machine = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(arm_machine)


def test_a_record_says_where_its_arm_ran(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    held = arm_machine.machine()
    assert set(held) == {"cpu_model", "threads", "peak_rss_bytes"}
    assert held["threads"] == "1"
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert held["peak_rss_bytes"] >= 1024 * own > 0
    model = held["cpu_model"]
    assert model is None or (isinstance(model, str) and model.strip() == model and model)
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert arm_machine.machine()["threads"] is None

"""BOPTEST integration: a thin REST client + a control episode, gated on a running BOPTEST service.

BOPTEST (https://ibpsa.github.io/project1-boptest/) is *the* standard framework for building/HVAC
control benchmarking -- realistic emulators, weather scenarios, and standard KPIs (energy, thermal
discomfort, cost, emissions, peak power, computational time); the top real-data target for Track D.

It runs as an external service (Docker; default ``http://127.0.0.1:5000``), so there is no live test
here -- point ``BOPTEST_URL`` at a running instance to exercise it. The client uses only the stdlib
(``urllib``), so it adds no dependency. Control inputs follow BOPTEST's overwrite convention: for a
control point ``p`` send ``{"p_u": value, "p_activate": 1}``; an empty dict uses the baseline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_URL = os.environ.get("BOPTEST_URL", "http://127.0.0.1:5000")

# A controller maps measurements to BOPTEST control inputs (empty dict = use the baseline).
Controller = Callable[[Mapping[str, float]], Mapping[str, float]]


def _unwrap(parsed: Any) -> Any:
    """Recent BOPTEST wraps responses as ``{status, message, payload}``; older returns the value."""
    if isinstance(parsed, dict) and "payload" in parsed and {"status", "message"} & parsed.keys():
        return parsed["payload"]
    return parsed


def _request(
    url: str, method: str, payload: Mapping[str, Any] | None = None, timeout: float = 60.0
) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # JSON REST API at a fixed base URL
        return _unwrap(json.loads(resp.read().decode()))


@dataclass
class BOPTestClient:
    """Minimal client for the BOPTEST REST API (one test case per instance)."""

    base_url: str = DEFAULT_URL

    def name(self) -> Any:
        return _request(f"{self.base_url}/name", "GET")

    def measurements(self) -> Any:
        return _request(f"{self.base_url}/measurements", "GET")

    def inputs(self) -> Any:
        return _request(f"{self.base_url}/inputs", "GET")

    def set_step(self, step_s: float) -> Any:
        return _request(f"{self.base_url}/step", "PUT", {"step": step_s})

    def initialize(self, start_time: float, warmup_period: float) -> Any:
        payload = {"start_time": start_time, "warmup_period": warmup_period}
        return _request(f"{self.base_url}/initialize", "PUT", payload)

    def advance(self, u: Mapping[str, float]) -> Any:
        return _request(f"{self.base_url}/advance", "POST", dict(u))

    def scenario(self, **kwargs: Any) -> Any:
        return _request(f"{self.base_url}/scenario", "PUT", kwargs)

    def kpi(self) -> Any:
        return _request(f"{self.base_url}/kpi", "GET")


def is_available(base_url: str = DEFAULT_URL, timeout: float = 3.0) -> bool:
    """True iff a BOPTEST service answers ``GET /name`` at ``base_url`` within ``timeout``."""
    try:
        _request(f"{base_url}/name", "GET", timeout=timeout)
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
        return False
    return True


def baseline_controller() -> Controller:
    """Hand control to BOPTEST's built-in baseline (empty overwrite each step)."""

    def control(_measurements: Mapping[str, float]) -> Mapping[str, float]:
        return {}

    return control


def run_episode(
    client: BOPTestClient,
    controller: Controller,
    *,
    start_time: float = 0.0,
    warmup_period: float = 0.0,
    step_s: float = 3600.0,
    horizon_steps: int = 24,
) -> dict[str, float]:
    """Step ``controller`` through one episode and return the BOPTEST KPI dict."""
    client.set_step(step_s)
    measurements = client.initialize(start_time, warmup_period)
    for _ in range(horizon_steps):
        measurements = client.advance(controller(measurements))
    return client.kpi()


def boptest_track(
    base_url: str = DEFAULT_URL, controllers: Mapping[str, Controller] | None = None, **episode: Any
) -> list[Any]:
    """Run controllers on a live BOPTEST case and return KPIs as ``TrackResult`` rows (Track D).

    Requires a BOPTEST service; raises ``RuntimeError`` otherwise. ``controllers`` defaults to the
    built-in baseline; add a CHC hybrid-MPC controller once a residual is identified on-line.
    """
    from causaldyn_bench.tracks import TrackResult

    if not is_available(base_url):
        raise RuntimeError(
            f"no BOPTEST service at {base_url}; set BOPTEST_URL to a running instance"
        )
    controllers = controllers or {"baseline": baseline_controller()}
    results: list[Any] = []
    for method, controller in controllers.items():
        kpis = run_episode(BOPTestClient(base_url), controller, **episode)
        for kpi_name, value in kpis.items():
            if isinstance(value, (int, float)):
                results.append(TrackResult("D-boptest", method, kpi_name, float(value)))
    return results

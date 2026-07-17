"""BOPTEST-Service client + control episode, gated on a running BOPTEST deployment.

BOPTEST (https://ibpsa.github.io/project1-boptest/) is *the* standard framework for building/HVAC
control benchmarking -- realistic emulators, weather scenarios, and standard KPIs (energy, thermal
discomfort, cost, emissions, peak power, computational time); the top real-data target for Track D.

The current BOPTEST deploys as **BOPTEST-Service** (a local web-service, default
``http://127.0.0.1:8000``), so there is no live test here -- bring it up (repo README; on Fedora:
``podman-compose up web worker provision``) and point ``BOPTEST_URL`` at it. Its REST API is
**testid-based**: select a test case to get a ``testid``, then drive that test. Responses wrap as
``{status, message, payload}``. The client uses only the stdlib. Control inputs follow BOPTEST's
overwrite convention: for a point ``p`` send ``{"p_u": value, "p_activate": 1}``; ``{}`` = baseline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_URL = os.environ.get("BOPTEST_URL", "http://127.0.0.1:8000")
DEFAULT_TESTCASE = "bestest_hydronic_heat_pump"

# A controller maps measurements to BOPTEST control inputs (empty dict = use the baseline).
Controller = Callable[[Mapping[str, float]], Mapping[str, float]]


def _unwrap(parsed: Any) -> Any:
    """BOPTEST-Service wraps responses as ``{status, message, payload}``; return the payload."""
    if isinstance(parsed, dict) and "payload" in parsed and {"status", "message"} & parsed.keys():
        return parsed["payload"]
    return parsed


def _request(
    url: str, method: str, payload: Mapping[str, Any] | None = None, timeout: float = 60.0
) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    # only advertise a JSON body when we send one: BOPTEST's Node server does JSON.parse on any
    # application/json request and 500s on an empty body (e.g. the no-payload POST /select).
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # JSON REST API at a fixed base URL
        body = resp.read().decode()
    if not body.strip():
        return None
    try:
        return _unwrap(json.loads(body))
    except json.JSONDecodeError:
        return body  # e.g. /stop returns a plain-text confirmation, not JSON


@dataclass
class BOPTestClient:
    """Minimal client for the BOPTEST-Service REST API (testid-based; one test per ``testid``)."""

    base_url: str = DEFAULT_URL

    def version(self) -> Any:
        return _request(f"{self.base_url}/version", "GET")

    def testcases(self) -> Any:
        return _request(f"{self.base_url}/testcases", "GET")

    def select(self, testcase: str) -> str:
        """Select a test case and return its ``testid`` (required by every test-scoped call)."""
        payload = _request(f"{self.base_url}/testcases/{testcase}/select", "POST")
        return payload["testid"]

    def set_step(self, testid: str, step_s: float) -> Any:
        return _request(f"{self.base_url}/step/{testid}", "PUT", {"step": step_s})

    def initialize(self, testid: str, start_time: float, warmup_period: float) -> Any:
        payload = {"start_time": start_time, "warmup_period": warmup_period}
        return _request(f"{self.base_url}/initialize/{testid}", "PUT", payload)

    def advance(self, testid: str, u: Mapping[str, float]) -> Any:
        return _request(f"{self.base_url}/advance/{testid}", "POST", dict(u))

    def inputs(self, testid: str) -> Any:
        return _request(f"{self.base_url}/inputs/{testid}", "GET")

    def kpi(self, testid: str) -> Any:
        return _request(f"{self.base_url}/kpi/{testid}", "GET")

    def stop(self, testid: str) -> Any:
        """Stop the test and free its worker (best practice after each episode)."""
        return _request(f"{self.base_url}/stop/{testid}", "PUT")


def is_available(base_url: str = DEFAULT_URL, timeout: float = 3.0) -> bool:
    """True iff a BOPTEST-Service answers ``GET /version`` at ``base_url`` within ``timeout``."""
    try:
        _request(f"{base_url}/version", "GET", timeout=timeout)
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
    testcase: str,
    controller: Controller,
    *,
    start_time: float = 0.0,
    warmup_period: float = 0.0,
    step_s: float = 3600.0,
    horizon_steps: int = 24,
) -> dict[str, float]:
    """Select ``testcase``, step ``controller`` through one episode, and return the BOPTEST KPIs."""
    testid = client.select(testcase)
    try:
        client.set_step(testid, step_s)
        measurements = client.initialize(testid, start_time, warmup_period)
        for _ in range(horizon_steps):
            measurements = client.advance(testid, controller(measurements))
        return client.kpi(testid)
    finally:
        client.stop(testid)  # free the worker even if the episode raises


def boptest_track(
    base_url: str = DEFAULT_URL,
    testcase: str = DEFAULT_TESTCASE,
    controllers: Mapping[str, Controller] | None = None,
    **episode: Any,
) -> list[Any]:
    """Run controllers on a live BOPTEST case and return KPIs as ``TrackResult`` rows (Track D).

    Requires a reachable service; raises ``RuntimeError`` otherwise. ``controllers`` defaults to the
    built-in baseline; add a CHC hybrid-MPC controller once a residual is identified on-line.
    """
    from causaldyn_bench.tracks import TrackResult

    if not is_available(base_url):
        raise RuntimeError(
            f"no BOPTEST-Service at {base_url}; set BOPTEST_URL to a running instance"
        )
    controllers = controllers or {"baseline": baseline_controller()}
    results: list[Any] = []
    for method, controller in controllers.items():
        kpis = run_episode(BOPTestClient(base_url), testcase, controller, **episode)
        for kpi_name, value in kpis.items():
            if isinstance(value, (int, float)):
                results.append(
                    TrackResult("D-boptest", f"{testcase}/{method}", kpi_name, float(value))
                )
    return results

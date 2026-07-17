"""BOPTEST guards run offline; a live episode runs only when BOPTEST_URL is set."""

import os

import jax.numpy as jnp
import numpy as np
import pytest

from causaldyn_bench.boptest import (
    DEFAULT_TESTCASE,
    BOPTestClient,
    baseline_controller,
    boptest_track,
    is_available,
    run_episode,
)
from causaldyn_bench.boptest_chc import (
    U_ACT,
    U_HP,
    _make_mpc_solver,
    mpc_controller,
)

_URL = os.environ.get("BOPTEST_URL")
# The identified stable model (a, b, d) for T_next = a*T + b*u + d; steady state spans ~14.6-26.8 C,
# so a 21 C comfort bound is reachable and the solver's response is not clamped to a boundary.
_MODEL = np.array([0.98, 0.2435, 5.756])


def test_mpc_solver_heats_harder_when_colder() -> None:
    solve = _make_mpc_solver(_MODEL)
    setpoints = jnp.full(16, 294.15)  # comfort lower bound 21 C over the horizon
    cold, warm = solve(289.0, setpoints), solve(298.0, setpoints)
    assert 0.0 <= warm < cold <= 1.0  # far below setpoint -> full heat; already warm -> back off


def test_mpc_controller_emits_a_bounded_heatpump_overwrite() -> None:
    action = mpc_controller(_MODEL)({"reaTZon_y": 289.0, "reaTSetHea_y": 294.15})
    assert action[U_ACT] == 1 and 0.0 <= action[U_HP] <= 1.0


def test_mpc_controller_no_overwrite_without_measurements() -> None:
    assert mpc_controller(_MODEL)({}) == {}


def test_is_available_false_for_unreachable_service() -> None:
    assert is_available("http://127.0.0.1:1", timeout=1.0) is False


def test_baseline_controller_hands_back_to_emulator() -> None:
    assert baseline_controller()({"reaTZon_y": 293.15}) == {}


def test_boptest_track_raises_without_a_service() -> None:
    with pytest.raises(RuntimeError, match="no BOPTEST-Service"):
        boptest_track("http://127.0.0.1:1")


@pytest.mark.skipif(not _URL, reason="set BOPTEST_URL to a running BOPTEST-Service")
def test_live_baseline_episode_returns_standard_kpis() -> None:
    if not is_available(_URL):
        pytest.skip("BOPTEST_URL is set but the service is unreachable")
    kpis = run_episode(
        BOPTestClient(_URL), DEFAULT_TESTCASE, baseline_controller(), horizon_steps=6
    )
    assert any(key in kpis for key in ("tdis_tot", "ener_tot", "cost_tot"))

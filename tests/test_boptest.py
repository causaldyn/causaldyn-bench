"""BOPTEST guards run offline; a live episode runs only when BOPTEST_URL is set."""

import os

import pytest

from causaldyn_bench.boptest import (
    BOPTestClient,
    baseline_controller,
    boptest_track,
    is_available,
    run_episode,
)

_URL = os.environ.get("BOPTEST_URL")


def test_is_available_false_for_unreachable_service() -> None:
    assert is_available("http://127.0.0.1:1", timeout=1.0) is False


def test_baseline_controller_hands_back_to_emulator() -> None:
    assert baseline_controller()({"reaTZon_y": 293.15}) == {}


def test_boptest_track_raises_without_a_service() -> None:
    with pytest.raises(RuntimeError, match="no BOPTEST service"):
        boptest_track("http://127.0.0.1:1")


@pytest.mark.skipif(not _URL, reason="set BOPTEST_URL to a running BOPTEST service")
def test_live_baseline_episode_returns_standard_kpis() -> None:
    if not is_available(_URL):
        pytest.skip("BOPTEST_URL is set but the service is unreachable")
    kpis = run_episode(BOPTestClient(_URL), baseline_controller(), horizon_steps=6)
    assert any(key in kpis for key in ("tdis_tot", "ener_tot", "cost_tot"))

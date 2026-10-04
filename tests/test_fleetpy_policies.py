"""Track R's pricing policies, read from ``scripts/fleetpy/policies.py`` without FleetPy: the status
quo's surge rule, the logger's dither and its log, and the switchback's dispatch.
"""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fleetpy" / "policies.py"
_SPEC = importlib.util.spec_from_file_location("track_r_policies", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
policies = importlib.util.module_from_spec(_SPEC)
# dataclasses resolve a module's annotations through sys.modules
sys.modules[_SPEC.name] = policies
_SPEC.loader.exec_module(policies)


@dataclass
class _Book:
    requests: dict[int, np.ndarray]
    idle: dict[int, np.ndarray]
    zones: int = 3
    asked: list[int] = field(default_factory=list)

    def closed(self, period: int) -> dict[str, np.ndarray]:
        self.asked.append(period)
        return {"requests": self.requests[period]}


def _book(requests, idle) -> _Book:
    return _Book({0: np.asarray(requests, dtype=float)}, {1: np.asarray(idle, dtype=float)})


def test_surge_rises_by_half_a_factor_per_request_per_idle_vehicle_above_one():
    # u = 2, 3 and 0.5: 1 + (u - 1) / 2 is 1.5, 2 and 0.75, the last held up to the list fare
    book = _book([4.0, 6.0, 1.0], [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(policies.Surge()(1, book), [1.5, 2.0, 1.0])
    assert book.asked == [0]


def test_surge_caps_at_the_box_top_and_counts_no_idle_vehicle_as_one():
    book = _book([9.0, 3.0, 0.0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(policies.Surge()(1, book), [2.0, 2.0, 1.0])


def test_surge_prices_at_the_list_fare_in_the_first_period_when_nothing_has_closed():
    book = _Book({}, {0: np.zeros(3)})
    np.testing.assert_array_equal(policies.Surge()(0, book), np.ones(3))
    assert book.asked == []


def test_dither_adds_its_draws_to_the_rule_held_three_scales_under_the_top():
    draws = np.zeros((policies.PERIODS, 3))
    draws[1] = [0.5, -20.0, 5.0]
    logger = policies.Dithered(0.1, draws)
    book = _book([4.0, 2.0, 6.0], [2.0, 2.0, 2.0])  # u = 2, 1, 3: the rule 1.5, 1 and 1.7, not 2
    np.testing.assert_allclose(
        logger(1, book), [1.55, policies.BOX[0], policies.BOX[1]], atol=1e-15
    )
    np.testing.assert_allclose(logger.base[1], [1.5, 1.0, 1.7], atol=1e-15)
    np.testing.assert_array_equal(logger.clipped[1], [False, True, True])


def test_dither_draws_at_the_rules_top_leave_the_box_only_past_three_scales():
    draws = np.zeros((policies.PERIODS, 3))
    draws[1] = [2.999, 3.001, 0.0]
    logger = policies.Dithered(0.12, draws)
    book = _book([9.0, 9.0, 9.0], [1.0, 1.0, 1.0])  # the rule at its top, 2 - 0.36
    logger(1, book)
    np.testing.assert_array_equal(logger.clipped[1], [False, True, False])


@pytest.mark.parametrize("scale", [0.0, 0.134])
def test_dither_refuses_a_scale_without_three_of_them_above_the_floor(scale):
    with pytest.raises(ValueError, match="three of the box's floor"):
        policies.Dithered(scale, np.zeros((policies.PERIODS, 3)))


def test_switchback_lets_the_assigned_arm_decide_each_period():
    book = _book([4.0, 6.0, 1.0], [2.0, 2.0, 2.0])
    assignment = np.zeros(policies.PERIODS, dtype=np.int64)
    assignment[1] = 1
    switchback = policies.Switchback((policies.Constant(0.8), policies.Surge()), assignment)
    np.testing.assert_array_equal(switchback(0, book), [0.8, 0.8, 0.8])
    np.testing.assert_array_equal(switchback(1, book), [1.5, 2.0, 1.0])


@pytest.mark.parametrize("period", [0, 95])
def test_schedule_reads_its_period_row(period):
    table = np.arange(policies.PERIODS * 3, dtype=float).reshape(policies.PERIODS, 3)
    np.testing.assert_array_equal(
        policies.Schedule(table)(period, _book([0, 0, 0], [0, 0, 0])), table[period]
    )

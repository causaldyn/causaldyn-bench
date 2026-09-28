"""Track M: the two axes of the media-planning problem, and which one the plant decides."""

import pytest
from chc.mmm import MmmReport, run_marketing_mix

from causaldyn_bench.allocation import CARRYOVER_DOMINANT, track_allocation


@pytest.fixture(scope="module")
def shipped() -> MmmReport:
    return run_marketing_mix(seed=0)


@pytest.fixture(scope="module")
def carryover_dominant() -> MmmReport:
    return run_marketing_mix(CARRYOVER_DOMINANT, seed=0)


def test_identification_is_what_pays_on_the_shipped_plant(shipped: MmmReport) -> None:
    """Both rules that adjust for the season beat the equal split, and the one that does not loses
    to it. Over eight seeds in ``allocation_report`` the adjusted arm holds at 8 of 8 and the other
    two at 7 of 8, both missing at seed 4; seed 0 is the regression pin."""
    assert shipped.lift("adjusted") > shipped.lift("flat")
    assert shipped.lift("myopic") > shipped.lift("flat")
    assert shipped.lift("confounded") < shipped.lift("flat")


def test_the_horizon_buys_nothing_when_the_two_orderings_agree_on_what_to_drop(
    shipped: MmmReport,
) -> None:
    """The finding the track was not built to show. ``myopic`` reads the same identified fit and
    spends it on this week alone, and on the shipped plant the two are a tie -- 3/5 on sign over
    eight seeds. Asserted as a bound on the gap rather than an order, because an order would be
    asserting a sign the measurement says is not there."""
    gap = abs(shipped.lift("adjusted") - shipped.lift("myopic"))
    assert gap < 0.1 * shipped.lift("adjusted")


def test_the_horizon_pays_once_the_carryover_ordering_contradicts_the_immediate_one(
    carryover_dominant: MmmReport,
) -> None:
    """The falsification arm, and what makes the null above a measurement rather than an absence.
    Same ``gamma``, so the myopic rule's ordering is untouched; only ``beta/theta`` moves, and moves
    until the best immediate channel is the worst carryover channel. 8 of 8 seeds."""
    assert carryover_dominant.lift("adjusted") > carryover_dominant.lift("myopic")
    gap = carryover_dominant.lift("adjusted") - carryover_dominant.lift("myopic")
    assert gap > 0.05 * carryover_dominant.lift("myopic")


def test_the_track_scores_four_rules_on_one_axis_read_upward() -> None:
    results = track_allocation(seed=0)
    assert {r.method for r in results} == {
        "CHC-adjusted",
        "myopic-greedy",
        "equal-split",
        "naive-MMM",
    }
    assert all(r.track == "M-allocation" and not r.lower_is_better for r in results)
    scores = {r.method: r.value for r in results}
    assert scores["naive-MMM"] < scores["equal-split"] < scores["CHC-adjusted"]

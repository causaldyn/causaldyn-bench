"""P1's table generator: the gates that make its tables checkable rather than transcribed."""

import json

import numpy as np
import pytest
from chc.regret import CompositionTransferCurve

from causaldyn_bench import paper_one
from causaldyn_bench.paper_one import (
    _jsonable,
    _monte_carlo_scale,
    _prefix_seed_counts,
    _walk_verdict,
    table_five,
    table_four,
    table_one,
    table_three,
    table_two,
    transfer_constants,
)

_SHIPPED = (0.01, 0.2)
_TIGHT = (1e-4, 2e-3)


def test_the_window_expansion_is_checked_against_the_library_not_copied_from_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`lambda` and `c2` are Maxima's, not the certificate's, so a reconstruction nobody checks is
    a second copy that drifts. The prediction must reproduce the certificate's own fitted slope to
    within the next order, and the gate must actually fire when the plant moves."""
    constants = transfer_constants()
    assert constants.doubling_residual == 0.0
    assert constants.prediction_residual == 0.0
    assert constants.first == pytest.approx(4.0 / 3.0)
    assert constants.second == pytest.approx(-16.0 / 9.0)

    # the drift the gate exists to catch is the library's plant moving while this module's
    # closed form does not, so that is what the mutation does -- moving `_B` here would move both
    original = paper_one.composition_transfer_certificate

    def moved_plant(
        *, b: float, rr: float, xt: float, delta_lo: float, delta_hi: float, n_delta: int
    ) -> CompositionTransferCurve:
        return original(b=1.4, rr=rr, xt=xt, delta_lo=delta_lo, delta_hi=delta_hi, n_delta=n_delta)

    monkeypatch.setattr(paper_one, "composition_transfer_certificate", moved_plant)
    with pytest.raises(RuntimeError, match="no longer reproduces the window expansion"):
        transfer_constants()


def test_the_second_window_term_improves_every_cell_the_window_still_limits() -> None:
    """The whole claim of Table 1 is that the miss is deterministic and expandable. Where the
    window still dominates, two terms must beat one -- if they only matched, the `c2` column would
    be decoration."""
    constants = transfer_constants()
    rows = table_one((_SHIPPED, (1e-3, 2e-2)), 12, constants)
    limited = [row for row in rows if row["limited_by"] == "window"]
    assert limited, "at these windows some cell must still be window-limited"
    for row in limited:
        assert abs(row["residual_second"]) < abs(row["residual_first"])


def test_every_residual_is_bounded_by_the_next_order_or_by_double_precision() -> None:
    """Table 1's two-sided claim, asserted per cell: what two closed-form terms leave behind is
    either one more power of `delta` or the rounding of a difference of two nearly equal numbers,
    and the `limited_by` column has to name the right one."""
    constants = transfer_constants()
    rows = table_one((_SHIPPED, (1e-3, 2e-2), _TIGHT, (1e-5, 2e-4)), 12, constants)
    for row in rows:
        next_order = abs(row["first_order"]) * row["delta_hi"] ** row["order"]
        assert abs(row["residual_second"]) <= next_order + row["cancellation_floor"]
        if row["limited_by"] == "float":
            assert abs(row["residual_second"]) < row["cancellation_floor"]

    # the floor has to GROW as the window drops -- a smaller `e` cancels more of `u*(b + e)`
    # against `u*(b)` -- or the column is decoration rather than the reason the fit stops working
    order_three = [row["cancellation_floor"] for row in rows if row["order"] == 3]
    assert order_three == sorted(order_three)


def test_one_plug_in_channel_caps_the_exponent_whether_there_are_two_or_three() -> None:
    """The bottleneck is a MIN over channels. Two channels with the spillover plugged in and three
    with the exposure map plugged in must land on the same `2`, and debiasing everything must
    reach `4` -- the pair of readings is the claim, neither alone is."""
    rows = table_two((_SHIPPED, (1e-3, 2e-2)), 10)
    for row in rows:
        assert row["two_channel_bottleneck"] == pytest.approx(2.0, abs=0.02)
        assert row["three_channel_bottleneck"] == pytest.approx(2.0, abs=0.02)
        assert row["two_channel_full"] == pytest.approx(4.0, abs=0.02)
        assert row["three_channel_full"] == pytest.approx(4.0, abs=0.02)
        assert not row["underflowed"]


def test_the_tight_window_underflows_and_the_table_reports_it_rather_than_the_slope() -> None:
    """The lower end of the usable window is double precision and it fails catastrophically:
    `delta^4` regret reaches exactly zero, the fit takes `log 0`, and the slope is `nan`. A table
    that printed the `nan` without the flag beside it would be reporting a measurement."""
    with pytest.warns(RuntimeWarning, match="divide by zero encountered in log"):
        (row,) = table_two((_TIGHT,), 10)
    assert row["underflowed"]
    assert np.isnan(row["two_channel_full"])
    assert np.isnan(row["three_channel_full"])
    assert row["two_channel_bottleneck"] == pytest.approx(2.0, abs=0.02)  # delta^2 still survives


def test_the_sampling_scale_is_pooled_over_nested_prefixes_not_read_off_one_draw() -> None:
    """Neither `G` certificate takes a seed offset, so the error bar has to come from the nested
    prefixes. Each rung estimates a KNOWN multiple of the wanted variance -- rung `j` is `2^(j-1)`
    times it -- so the rescaling is what makes pooling legitimate, and getting it wrong would
    silently inflate or deflate every verdict."""
    assert _prefix_seed_counts(240) == [240, 120, 60, 30]
    assert _prefix_seed_counts(4) == [4, 2]  # the chain stops rather than reaching one seed
    assert _prefix_seed_counts(3) == [3]

    # one rung reproduces the plain difference; a second rung enters divided by sqrt(2), so a
    # rung that is sqrt(2) larger than the first contributes exactly as much as it
    assert _monte_carlo_scale([-1.0, -1.2]) == pytest.approx(0.2)
    assert _monte_carlo_scale([-1.0, -1.2, -1.2 - 0.2 * np.sqrt(2.0)]) == pytest.approx(0.2)
    assert _monte_carlo_scale([-1.0]) == 0.0


def test_the_ladder_states_whether_its_walk_clears_the_sampling_scale() -> None:
    """The claim is that the statistic walks end to end, so that is the statistic tested -- and a
    run that cannot separate the walk from the seeds has to say so rather than quote the ladder."""
    wide = [
        {"g_slope": -1.3, "monte_carlo_scale": 0.01},
        {"g_slope": -1.0, "monte_carlo_scale": 0.01},
    ]
    narrow = [
        {"g_slope": -1.3, "monte_carlo_scale": 0.5},
        {"g_slope": -1.0, "monte_carlo_scale": 0.5},
    ]
    assert "so the walk is the window" in _walk_verdict(wide, "g_slope")
    assert "does not separate" in _walk_verdict(narrow, "g_slope")
    assert "not a ladder" in _walk_verdict(wide[:1], "g_slope")

    three = table_three(((10, 20, 40), (20, 40, 80)), 4, 5)
    four = table_four(((20, 40, 80), (40, 80, 160)), 4)
    for row in three["g_regime"] + four:
        assert row["monte_carlo_scale"] >= 0.0
        assert row["prefix_seed_counts"] == [4, 2]
    assert three["delta_regime"]["half_slope"] == pytest.approx(2.0, abs=0.02)
    assert three["delta_regime"]["full_slope"] == pytest.approx(4.0, abs=0.05)
    for row in four:
        assert row["c0"] > 0.0  # the floor is two-sided only if the constant is positive


def test_the_exponent_is_reported_with_the_event_it_is_conditional_on() -> None:
    """A draw whose perturbed plant is unstabilisable has no finite regret, so the fitted exponent
    is conditional. Table 5 reports the share alongside every seed; dropping the column would make
    the exponent unreadable rather than merely incomplete."""
    five = table_five((0, 1, 2), 100)
    assert len(five["exponents"]) == len(five["unbounded_share"]) == 3
    assert five["lo"] <= five["median"] <= five["hi"]
    assert all(exponent > 2.0 for exponent in five["exponents"])  # the window term is one-sided
    assert five["worst_unbounded_share"] == max(five["unbounded_share"])


def test_the_json_artefact_is_readable_by_a_strict_parser() -> None:
    """Table 2's `nan` is a finding and stays in the markdown, but `json.dumps` writes it as the
    literal `NaN`, which only Python's own loader accepts. The artefact has to survive a parser
    that does not."""
    payload = _jsonable({"slope": float("nan"), "rows": [{"x": 1.0}, {"y": float("inf")}]})
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text) == {"slope": None, "rows": [{"x": 1.0}, {"y": None}]}

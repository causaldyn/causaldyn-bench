"""P3's finite check of the greedy-fill theorem: the numbers the paper quotes, and its power."""

from causaldyn_bench.greedy_fill_grid import DRAWS, check


def test_no_schedule_beats_the_greedy_fill_and_the_back_loaded_fill_loses() -> None:
    """The paper quotes 395 954 feasible schedules over 300 draws, every phi_2 or phi_3 non-convex,
    none below the greedy fill of its own total (the least gap is exactly 0, a tie on identical
    prefix sums), and the back-loaded fill beaten by 11.89 somewhere. The counts are integers of an
    exact enumeration; only the control's depth is a float."""
    found = check()
    assert found.draws == DRAWS
    assert found.nonconvex_draws == DRAWS
    assert found.schedules == 395_954
    assert found.ties == 198_645
    assert found.worst == 0.0
    assert -11.9 < found.control_worst < -11.88

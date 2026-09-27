"""Track L: the DCBO reference read back and replayed on the port, and the CHC arm via prescribe."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from causaldyn_bench.dcbo_scm import (
    DOMAIN,
    HORIZON,
    SCMS,
    STAT,
    Decision,
    oracle,
    sample_log,
    score,
)
from causaldyn_bench.dcbo_track import (
    CHC,
    METHODS,
    REFERENCE,
    ArmRun,
    Reference,
    _panel,
    chc_decisions,
    load_reference,
    paired_intervals,
    replays,
    track_dcbo,
)


@pytest.fixture(scope="module")
def reference() -> Reference:
    return load_reference()


def test_the_reference_oracle_is_the_ports_grid_oracle(reference: Reference) -> None:
    """``optimal_sequence_of_interventions`` on the reference's own SEM classes, against the port's
    grid oracle: the same exploration set at every step, and the same optimum."""
    for scm in SCMS:
        path = oracle(scm, refine=False)
        assert reference.oracle_sets[scm.name] == tuple(r.decision.variables for r in path)
        np.testing.assert_allclose(
            reference.oracle_values[scm.name], [r.value for r in path], rtol=1e-12
        )


def _reads_a_sampled_child(name: str, decisions: tuple[Decision, ...]) -> bool:
    """Whether a later step reads a ``Z_t`` that ``assign_blanket`` drew with noise after
    ``do(X_t)`` alone: ``stat``'s ``Z_{t+1}`` under ``do(X_{t+1})`` alone, and ``nonstat``'s
    ``Z_1`` in ``Y_2`` (and, under ``do(X_1)`` alone, ``Z_0`` in ``Z_1``). In ``ind``, ``X_t`` has
    no child in its slice.
    """
    alone = [d.variables == ("X",) for d in decisions]
    if name == "stat":
        return any(a and b for a, b in pairwise(alone))
    if name == "nonstat":
        return alone[1]
    return False


def test_the_port_replays_every_recorded_outcome_but_cbos_noisy_history(
    reference: Reference,
) -> None:
    """Each run's decisions, played on the port from the zero history, give the outcomes the
    reference recorded for them -- except in exactly the CBO runs whose history carries a child
    ``assign_blanket`` sampled with noise, where they must differ."""
    for scm in SCMS:
        for runs in reference.runs[scm.name].values():
            for method, recorded in runs.items():
                scored = score(scm, recorded.decisions)
                played = ArmRun(recorded.decisions, scored.outcomes, scored.regret, 0.0)
                noisy = method == "CBO" and _reads_a_sampled_child(scm.name, recorded.decisions)
                assert replays(played, recorded) is not noisy, (scm.name, method)


def test_a_file_drawn_from_other_logs_is_refused(tmp_path: Path) -> None:
    raw = json.loads(REFERENCE.read_text())
    raw["runs"]["ind"]["3"]["log"]["Z"][4][1] += 1e-6
    path = tmp_path / "track_l_dcbo.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Z in the ind log of seed 3 differs"):
        load_reference(path)


def test_the_panel_runs_the_target_one_row_behind_the_levers() -> None:
    """``prescribe`` fits ``x_next = x + f(x, u)`` on consecutive rows of a unit, and ``Y_t``
    answers its own step's ``X_t, Z_t``: row ``k`` holds ``Y_{k-1}`` -- zero at ``k = 0`` -- beside
    ``X_k, Z_k``."""
    log = sample_log(STAT, seed=0)
    panel = _panel(log, seed=0)
    np.testing.assert_array_equal(panel.wide("Y")[:, 0], 0.0)
    np.testing.assert_array_equal(panel.wide("Y")[:, 1:], log["Y"])
    np.testing.assert_array_equal(panel.wide("X")[:, :HORIZON], log["X"])
    np.testing.assert_array_equal(panel.wide("Z")[:, :HORIZON], log["Z"])


def test_the_chc_arm_sets_both_levers_inside_their_boxes_at_every_step() -> None:
    decisions = chc_decisions(STAT, seed=0)
    assert len(decisions) == HORIZON
    for decision in decisions:
        assert decision.variables == ("X", "Z")
        for name, level in zip(decision.variables, decision.levels, strict=True):
            assert DOMAIN[name][0] <= level <= DOMAIN[name][1]
    assert score(STAT, decisions).total_regret >= 0.0


def test_paired_intervals_resample_every_series_with_one_index_set() -> None:
    """A series shifted by a constant gets the same interval shifted, which independent resamples
    would not give; and series of different lengths cannot be paired at all."""
    values = np.random.default_rng(1).normal(size=10)
    out = paired_intervals({"a": values, "a + 1": values + 1.0}, n_boot=500)
    assert out["a + 1"].mean == pytest.approx(out["a"].mean + 1.0)
    assert out["a + 1"].lo == pytest.approx(out["a"].lo + 1.0)
    assert out["a + 1"].hi == pytest.approx(out["a"].hi + 1.0)
    assert out["a"].lo < out["a"].mean < out["a"].hi
    with pytest.raises(ValueError, match="one shape"):
        paired_intervals({"a": values, "b": values[:5]})


def test_the_track_scores_every_method_on_one_board_per_scm() -> None:
    results = track_dcbo(seeds=(0,))
    assert {r.track for r in results} == {f"L-dcbo-{scm.name}" for scm in SCMS}
    assert {r.method for r in results} == {*METHODS, CHC}
    assert all(r.metric == "regret" and r.lower_is_better for r in results)
    assert min(r.value for r in results) > -1e-9

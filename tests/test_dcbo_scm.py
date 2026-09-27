"""Track L's port of DCBO's three SCMs, against closed forms and the reference notebooks."""

import math

import numpy as np
import pytest

from causaldyn_bench.dcbo_scm import (
    DOMAIN,
    EXPLORATION_SETS,
    HORIZON,
    IND,
    N_OBSERVATIONS,
    NONSTAT,
    SCMS,
    STAT,
    Decision,
    DynamicSCM,
    oracle,
    sample_log,
    score,
)


def test_stat_is_best_driven_to_the_bottom_of_z() -> None:
    """``Y_t = cos Z_t - exp(-Z_t / 20) + Y_{t-1}``, and the step term's slope at ``z = -3`` is
    ``sin 3 + e^0.15 / 20 > 0``: the minimum over ``D(Z)`` is its lower end, so
    ``y*_t = (t + 1)(cos 3 - e^0.15)``. ``{X, Z}`` reaches it too and loses the tie to the earlier
    set, refined or not."""
    for refine in (False, True):
        for t, response in enumerate(oracle(STAT, refine=refine)):
            assert response.decision == Decision(("Z",), (-3.0,))
            assert response.value == pytest.approx(
                (t + 1) * (math.cos(3.0) - math.exp(0.15)), abs=1e-12
            )


def test_ind_is_best_at_the_large_bump_pulled_toward_the_small_one() -> None:
    """``-2A - B`` with ``A = exp(-(x - 1)^2 - (z - 1)^2)`` and ``B = exp(-(x + 1)^2 - z^2)``. The
    grid holds the large bump's centre, worth ``-2 - e^-5`` a step; the refined optimum is where the
    gradient vanishes, ``x - 1 = -(x + 1) B / 2A`` and ``z - 1 = -z B / 2A``, and is lower."""
    grid = oracle(IND, refine=False)
    for t, response in enumerate(grid):
        assert response.decision.variables == ("X", "Z")
        assert response.decision.levels == pytest.approx((1.0, 1.0), abs=1e-12)
        assert response.value == pytest.approx((t + 1) * (-2.0 - math.exp(-5.0)), abs=1e-12)
    refined = oracle(IND)[0]
    x, z = refined.decision.levels
    big = math.exp(-((x - 1.0) ** 2) - (z - 1.0) ** 2)
    small = math.exp(-((x + 1.0) ** 2) - z**2)
    assert x - 1.0 == pytest.approx(-(x + 1.0) * small / (2.0 * big), abs=1e-6)
    assert z - 1.0 == pytest.approx(-z * small / (2.0 * big), abs=1e-6)
    assert refined.value < grid[0].value


def test_nonstat_is_best_at_the_closed_form_of_each_regime() -> None:
    """Eq. (20), each regime by hand, along the oracle's own path.

    ``f`` (``t = 0``): ``sqrt|36 - (z - 1)^2| + 1`` falls as ``(z - 1)^2`` nears 36, and
    ``do(X = -4)`` reaches ``Z_0 = -4``: ``y*_0 = 1 + sqrt 11``. ``g`` (``t = 1``): ``do(X_1)``
    makes ``Z_1 = -X_1 / X_0 + Z_0 = X_1 / 4 - 4``, and ``z cos(pi z)`` is stationary at
    ``z = -4 - e`` with ``tan(pi e) = 1 / (pi (4 + e))``. ``h`` (``t = 2``):
    ``|Z_2| - Y_1 - Z_1`` is least at ``Z_2 = 0``, which only the refinement reaches -- the grid's
    nearest points are ``+-3/99`` away.
    """
    path = oracle(NONSTAT)
    assert path[0].decision == Decision(("X",), (-4.0,))
    assert path[0].value == pytest.approx(1.0 + math.sqrt(11.0), abs=1e-12)

    e = 0.0
    for _ in range(100):
        e = math.atan(1.0 / (math.pi * (4.0 + e))) / math.pi
    z1 = -4.0 - e
    assert path[1].decision.variables == ("X",)
    assert path[1].decision.levels[0] == pytest.approx(4.0 * (z1 + 4.0), abs=1e-6)
    assert path[1].value == pytest.approx(z1 * math.cos(math.pi * z1) - path[0].value, abs=1e-10)

    assert path[2].decision.variables == ("Z",)
    assert path[2].decision.levels[0] == pytest.approx(0.0, abs=1e-6)
    assert path[2].value == pytest.approx(-path[1].value - z1, abs=1e-6)
    grid = oracle(NONSTAT, refine=False)
    assert abs(grid[2].decision.levels[0]) == pytest.approx(3.0 / 99.0, abs=1e-12)


@pytest.mark.parametrize(
    ("scm", "printed"),
    [
        (STAT, (-2.152, -4.304, -6.455)),
        (IND, (-2.007, -4.013, -6.02)),
        (NONSTAT, (4.317, -8.329, 12.387)),
    ],
    ids=["stat", "ind", "nonstat"],
)
def test_the_grid_oracle_is_what_the_reference_notebooks_print(
    scm: DynamicSCM, printed: tuple[float, ...]
) -> None:
    """``notebooks/{stat,ind,nonstat}_scm.ipynb`` at ``85a9bdf``: "True optimal outcome values",
    rounded to three places as they print them."""
    values = [response.value for response in oracle(scm, refine=False)]
    assert np.round(values, 3).tolist() == list(printed)


def test_regret_is_zero_on_the_oracle_and_never_below_it() -> None:
    """Conditional regret prices each step against the best response to the arm's own past, so no
    sequence goes below zero -- by more than the refinement's resolution, since a continuous level
    can beat a finite grid. ``nonstat``'s ``g`` divides by ``X_0``, so its sequences set ``X_0``."""
    rng = np.random.default_rng(0)
    for scm in SCMS:
        assert score(scm, [r.decision for r in oracle(scm)]).regret == (0.0,) * HORIZON
        for _ in range(10):
            decisions = []
            for t in range(HORIZON):
                options = [s for s in EXPLORATION_SETS if scm is not NONSTAT or t > 0 or "X" in s]
                variables = options[rng.integers(len(options))]
                levels = tuple(float(rng.uniform(*DOMAIN[name])) for name in variables)
                decisions.append(Decision(variables, levels))
            assert min(score(scm, decisions).regret) > -1e-9


def test_the_log_is_the_noise_block_pushed_through_the_equations() -> None:
    """``N`` series of ``T`` slices; ``X_0`` of ``stat`` is its noise draw alone, because every lag
    of the first slice is zero, and the draws are the seed's first ``(T, 3, N)`` block."""
    log = sample_log(STAT, seed=3)
    assert {name: values.shape for name, values in log.items()} == {
        name: (N_OBSERVATIONS, HORIZON) for name in ("X", "Z", "Y")
    }
    noise = np.random.default_rng(3).standard_normal((HORIZON, 3, N_OBSERVATIONS))
    np.testing.assert_array_equal(log["X"][:, 0], noise[0, 0])
    np.testing.assert_allclose(log["Z"][:, 0], np.exp(-noise[0, 0]) + noise[0, 1], rtol=1e-15)
    np.testing.assert_array_equal(sample_log(STAT, seed=3)["Y"], log["Y"])


def test_a_decision_outside_its_domain_or_set_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        Decision(("X",), (1.5,))
    with pytest.raises(ValueError, match="needs 2 levels"):
        Decision(("X", "Z"), (0.0,))

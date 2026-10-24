"""Track H over the markets, not one: equilibrium-aware CHC has the least regret where the SUTVA
planners leave large regret, and the logs' confounding is read against a rerun of the logs that the
logs themselves cannot give.

``seed`` identifies a market only at a fixed dtype. ``jax.random`` spends a different number of
threefry bits per element for a float64 output than for a float32 one, so under
``JAX_ENABLE_X64=1`` a seed draws a *different* market rather than a higher-precision copy of the
same one: at seed 0 the headroom between doing nothing and the oracle reads 2.43 against 1.06. The
committed runs, ``results/track_h_markets.json`` (float32) and ``results/track_h_markets_x64.json``
(float64), are 60 markets, and every claim tested here on seeds 0-2 holds on all 60. An ordering
or a sign is asserted as it stands. A bound is the 30 markets' mean less three standard deviations,
or plus three, at the precision that reads it wider, rounded outward to two decimals: a reading of
the distribution, not of the markets tested here. The bounds are checked against the committed
runs, so a rerun that moves the distribution fails until they are read again; the committed runs
are checked to be their own markets, and to be what the code computes for seeds 0-2: on any
machine in every reading no ascent ends, and in the ascents' ends only on the machine class that
recorded them.
"""

import importlib.metadata
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chc.marketplace import SharedStateMarket

from causaldyn_bench.marketplace import (
    NAMES,
    PLANNERS,
    TRACK,
    chase,
    leaderboard_note,
    marketplace_report,
    readings,
    record,
    rerun,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
LOWER = {
    "chc_share": 0.69,
    "predictive_confounding_bias": 0.14,
    "naive_cross_zone_confounding": 0.04,
    "chc_confounding_bias": 0.08,
    "randomised_demand_corr": 0.28,
    "naive_interference_overcount": 0.39,
}
UPPER = {"chc_regret": 0.73, "naive_kept_share": 0.88}
# The committed readings carry three decimals in results/track_h_markets*.md, so a run reproduces
# them to half of the last.
TO_THE_QUOTED_DIGIT = 5e-4
# The readings that end a nonconvex ascent: the oracle's on the market's equilibrium, and
# equilibrium-CHC's on the one it fits. On a CPU with AVX-512, where numpy and XLA run other
# vector kernels, the same programs round otherwise and an ascent can end at another local
# optimum: there market 1's float64 oracle read 2.961 of lift against the committed 3.093, and
# market 0's float32 CHC plan 2.328 against 2.340.
ASCENT_ENDS = frozenset(
    {"oracle_lift", "chc_realised_lift"} | {f"{key}_regret" for key in PLANNERS.values()}
)
# The versions the runs were recorded under, on x86 without AVX-512. Others compile and solve the
# ascents otherwise too: the Python 3.11 leg, on jax 0.10.2, numpy 2.4.6 and scipy 1.17.1, ended
# market 0's float32 oracle at 2.4241 of lift against the committed 2.4268. A rerun under other
# versions reads them again.
RECORDED_UNDER = {"jax": "0.11.2", "jaxlib": "0.11.2", "numpy": "2.5.4", "scipy": "1.18.1"}


def _x64() -> bool:
    return bool(jnp.zeros(()).dtype == jnp.float64)


def _other_versions() -> str | None:
    """Why the versions here are not the ones the committed runs were recorded under, or None."""
    running = {name: importlib.metadata.version(name) for name in RECORDED_UNDER}
    if running == RECORDED_UNDER:
        return None
    return f"{running} are not the versions the committed runs were recorded under"


def _committed(x64: bool) -> dict:
    return json.loads((ROOT / "results" / f"{NAMES[x64]}.json").read_text())


@pytest.fixture(scope="module")
def reports() -> dict[int, dict[str, float]]:
    return {seed: marketplace_report(seed=seed) for seed in SEEDS}


@pytest.mark.parametrize("seed", SEEDS)
def test_chc_has_the_least_regret_and_most_of_the_oracle_s_lift(reports, seed: int) -> None:
    r = readings(reports[seed])
    assert r["chc_regret"] < r["naive_regret"]  # SUTVA planners leave large regret on the table
    assert r["chc_regret"] < r["predictive_regret"]
    assert r["chc_regret"] < UPPER["chc_regret"]
    assert r["chc_share"] > LOWER["chc_share"]  # equilibrium-aware


@pytest.mark.parametrize("seed", SEEDS)
def test_the_report_shows_confounding_and_interference(reports, seed: int) -> None:
    """The logging policy chased demand, so the predictive response reads above the randomised
    twin's; adjusting each zone for its own demand removes part of that, and the part that runs
    through the other zones' demand stays. Equilibrium-CHC's response, which holds the block's total
    incentive as well, keeps more of it than the naive-causal one. The twin's response follows
    demand, which is why a correlation with demand is no test of de-confounding. The naive planner
    over-counts the lift it predicts by the drivers its zones take from each other."""
    r = readings(reports[seed])
    assert r["chc_confounding_bias"] > r["naive_confounding_bias"]
    assert r["chc_confounding_bias"] > LOWER["chc_confounding_bias"]
    assert r["predictive_confounding_bias"] > LOWER["predictive_confounding_bias"]
    assert r["naive_kept_share"] < UPPER["naive_kept_share"]
    assert r["naive_cross_zone_confounding"] > LOWER["naive_cross_zone_confounding"]
    assert r["randomised_demand_corr"] > LOWER["randomised_demand_corr"]
    assert r["naive_interference_overcount"] > LOWER["naive_interference_overcount"]


def test_the_reruns_replay_the_logged_blocks() -> None:
    """The twins rerun the logged blocks under other incentives, so with the logged incentives they
    must give back the logs: :func:`chase` is the logging policy, to the rounding of a twelve-term
    sum and two products, at most about 8 epsilons (twice that is allowed), and :func:`rerun`
    returns the logged completions, to one rounding of the noise it takes off and adds back."""
    market = SharedStateMarket(seed=0)
    logs = market.generate_logs(400, jax.random.key(1))
    eps = float(jnp.finfo(logs["u"].dtype).eps)
    policy = chase(market, logs["demand"], logs["aggregate"][:, :1])
    np.testing.assert_allclose(policy, logs["u"], rtol=16 * eps, atol=0.0)
    replayed = rerun(market, logs, logs["u"])
    atol = 2 * eps * float(jnp.max(jnp.abs(logs["y"])))
    np.testing.assert_allclose(replayed["y"], logs["y"], rtol=0.0, atol=atol)


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_the_committed_runs_hold_the_claims_on_every_market(x64: bool) -> None:
    counts = _committed(x64)["counts"]
    n = counts["markets"]
    assert n == 30
    assert counts["chc_least_regret"] == n
    assert counts["predictive_confounding_bias_positive"] == n
    assert counts["naive_bias_below_predictive"] == n
    assert counts["naive_cross_zone_confounding_positive"] == n
    assert counts["chc_bias_above_naive"] == n
    assert counts["naive_interference_overcount_positive"] == n


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_a_committed_run_is_its_markets(x64: bool) -> None:
    """Its summary and counts are what its markets read, and its precision is its name's."""
    committed = _committed(x64)
    assert committed["x64"] is x64
    assert record(committed["markets"], committed["blocks"], x64) == committed


def test_the_bounds_are_read_off_the_committed_runs() -> None:
    summaries = [_committed(x64)["summary"] for x64 in NAMES]
    for key, bound in LOWER.items():
        wider = min(s[key]["mean"] - 3.0 * s[key]["sd"] for s in summaries)
        assert bound == math.floor(100.0 * wider) / 100.0, key
    for key, bound in UPPER.items():
        wider = max(s[key]["mean"] + 3.0 * s[key]["sd"] for s in summaries)
        assert bound == math.ceil(100.0 * wider) / 100.0, key


def test_the_committed_run_is_what_the_code_computes(reports) -> None:
    """At this precision, the committed run's markets 0-2 read what is computed here in every
    reading no ascent ends: the responses' confounding, the SUTVA planners' lifts and the naive
    planner's over-count."""
    markets = {m["seed"]: m["report"] for m in _committed(_x64())["markets"]}
    for seed in SEEDS:
        committed = {k: v for k, v in markets[seed].items() if k not in ASCENT_ENDS}
        computed = {k: reports[seed][k] for k in committed}
        assert computed == pytest.approx(committed, rel=0.0, abs=TO_THE_QUOTED_DIGIT)


def test_the_ascents_end_where_the_committed_run_s_machine_class_ends_them(
    reports, other_kernels
) -> None:
    """On x86 without AVX-512 under the versions the runs were recorded under, the oracle's lift,
    equilibrium-CHC's and the regrets read as committed too, and the leaderboard's Track H rows,
    measured in float32, are market 0's regrets. Elsewhere the regrets' order and bounds hold
    (above), not their digits."""
    for reason in (other_kernels, _other_versions()):
        if reason is not None:
            pytest.skip(reason)
    x64 = _x64()
    markets = {m["seed"]: m["report"] for m in _committed(x64)["markets"]}
    for seed in SEEDS:
        assert reports[seed] == pytest.approx(markets[seed], rel=0.0, abs=TO_THE_QUOTED_DIGIT)
    if not x64:
        rows = json.loads((ROOT / "results" / "leaderboard.json").read_text())
        board = {r["method"]: r["value"] for r in rows if r["track"] == "H-marketplace"}
        regrets = {name: reports[0][f"{key}_regret"] for name, key in PLANNERS.items()}
        assert board == pytest.approx(regrets, rel=0.0, abs=TO_THE_QUOTED_DIGIT)


def test_the_leaderboard_prints_the_note_its_runs_read() -> None:
    """``--save`` writes the leaderboard's paragraph under Track H from the committed runs, so the
    committed leaderboard carries the note they read, under the track's table."""
    note = leaderboard_note(ROOT / "results")
    assert note is not None
    board = (ROOT / "results" / "leaderboard.md").read_text()
    section = board.split(f"## {TRACK}", 1)[1].split("\n## ", 1)[0]
    assert note in section

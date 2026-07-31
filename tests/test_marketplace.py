"""Track H: equilibrium-aware CHC recovers the oracle where SUTVA planners leave large regret."""

from causaldyn_bench.marketplace import marketplace_report, track_marketplace


def _by_method(results: list, track: str) -> dict[str, float]:
    return {r.method: r.value for r in results if r.track == track}


def test_marketplace_chc_recovers_oracle_baselines_do_not() -> None:
    regret = _by_method(track_marketplace(seed=0), "H-marketplace")
    assert regret["equilibrium-CHC"] < 0.3  # de-confounded + equilibrium-aware ~ recovers oracle
    assert regret["naive-causal"] > 0.8  # SUTVA planners leave large regret on the table
    assert regret["predictive-MOPO"] > 0.8


def test_marketplace_report_shows_confounding_and_interference() -> None:
    """Each lift is asserted as a share of the oracle's, because the absolute one is a draw.

    ``seed`` identifies this market only at a fixed dtype. ``jax.random`` spends a different number
    of threefry bits per element for a float64 output than for a float32 one, so under
    ``JAX_ENABLE_X64=1`` the market is a *different* Monte-Carlo instance rather than a
    higher-precision copy of the same one -- every coordinate of ``demand`` moves by O(1), and the
    headroom between doing nothing and the oracle moves with it (2.42 against 1.06). What is a
    property of the planners, and does not move, is the share of that headroom each one captures.
    """
    rep = marketplace_report(seed=0)
    oracle = rep["oracle_lift"]
    assert rep["predictive_confounding_corr"] > 0.25  # logging chased demand -> confounded response
    assert abs(rep["naive_deconfounded_corr"]) < 0.2  # backdoor removes it
    assert rep["chc_realised_lift"] > 0.9 * oracle  # CHC delivers most of the oracle lift...
    assert rep["naive_realised_lift"] < 0.25 * oracle  # ...while the SUTVA planners barely help
    assert rep["predictive_realised_lift"] < 0.25 * oracle  # (or hurt)
    assert rep["naive_interference_overcount"] > 1.0  # SUTVA predicts a lift it never realises

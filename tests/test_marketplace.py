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
    rep = marketplace_report(seed=0)
    assert rep["predictive_confounding_corr"] > 0.25  # logging chased demand -> confounded response
    assert abs(rep["naive_deconfounded_corr"]) < 0.2  # backdoor removes it
    assert rep["chc_realised_lift"] > 2.0  # CHC delivers most of the oracle lift...
    assert rep["naive_realised_lift"] < 0.5  # ...while naive barely helps (or hurts)
    assert rep["naive_interference_overcount"] > 1.0  # SUTVA predicts a lift it never realises

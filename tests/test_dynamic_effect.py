"""Track G: the IRF methods recover the dynamic effect, and the recovered effect controls better."""

from causaldyn_bench.dynamic_effect import track_dynamic_effect


def _by_method(results: list, track: str) -> dict[str, float]:
    return {r.method: r.value for r in results if r.track == track}


def test_dynamic_effect_recovery_and_control_payoff() -> None:
    results = track_dynamic_effect(seed=0)

    effect = _by_method(results, "G-effect")
    assert effect["local-projections"] < effect["naive-static"]  # lower IRF error than static
    assert effect["structured-toeplitz"] < effect["naive-static"]

    payoff = _by_method(results, "G-payoff")
    assert payoff["chc-irf"] < 0.3 * payoff["one-step"]  # deconvolving tracks; one-step overshoots

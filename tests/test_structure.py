"""Track F: discovery beats naive screening at structure recovery, and the structure pays off."""

from causaldyn_bench.structure import track_structure


def _by_method(results: list, track: str) -> dict[str, float]:
    return {r.method: r.value for r in results if r.track == track}


def test_discovery_wins_structure_recovery_and_the_downstream_decision() -> None:
    results = track_structure(seed=0)

    structure = _by_method(results, "F-structure")
    assert structure["chc-discovery"] > structure["naive-correlation"]  # higher edge-F1

    payoff = _by_method(results, "F-payoff")
    assert payoff["chc-discovery-adjusted"] < payoff["naive-unadjusted"]  # lower ATE error
    assert payoff["chc-discovery-adjusted"] < 0.1  # the discovered adjustment recovers the effect

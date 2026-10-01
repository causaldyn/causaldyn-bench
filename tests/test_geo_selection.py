"""Track M v2, geo selection: the market is its geos summed, the oracle's curves are the planner's
channels, a set's gaps are the ones its test reads, and the gate reads what it says."""

import dataclasses
import json
import math
import sys

import jax
import numpy as np
import pytest
from chc.allocation import decision_weight
from chc.lift import LiftFit
from chc.response import Channel, GeometricAdstock

from causaldyn_bench import geo_selection
from causaldyn_bench.budget_regret import planned
from causaldyn_bench.geo_selection import (
    ARMS,
    FIRST,
    GEOS,
    PILOTS,
    POOL,
    SHARE,
    STARTS,
    TESTED,
    WORLDS,
    ArmScore,
    Candidate,
    Design,
    GeoMixture,
    GeoPanel,
    Run,
    WorldScore,
    _markdown,
    _plan,
    _record,
    choose,
    dark,
    design,
    exposure,
    gaps,
    jacobian,
    lift_tests,
    main,
    mixture_curve,
    pool,
    reading_error,
    readouts,
    score_world,
    template,
    weights,
)
from causaldyn_bench.mmm_decision import PLANNED, Quarter, oracle, worth
from causaldyn_bench.observational_check import gaps as predicted_gaps


def _kernel(retention: float, length: int) -> np.ndarray:
    weights = retention ** np.arange(length)
    return weights / weights.sum()


def test_the_mixture_is_the_geos_tanh_summed_alike_in_the_planner_and_the_oracle():
    population = np.array([0.5, 0.3, 0.2])
    intensity = np.array([0.6, 1.2, 1.9])
    intensity = intensity / (population @ intensity)
    lam = 0.008
    adstock = np.linspace(0.0, 400.0, 9)
    by_hand = np.tanh(np.outer(adstock, intensity) * lam / 2.0) @ population
    with jax.enable_x64(True):
        curve = GeoMixture(2.0 / lam, tuple(population.tolist()), tuple(intensity.tolist()))
        np.testing.assert_allclose(np.asarray(curve(adstock)), by_hand, rtol=1e-12, atol=1e-15)
        assert curve.inflection() == 0.0
    oracle_curve = mixture_curve(population, intensity)
    np.testing.assert_allclose(oracle_curve.value(adstock, lam), by_hand, rtol=1e-12)
    step = 1e-3
    central = (
        oracle_curve.value(adstock + step, lam) - oracle_curve.value(adstock - step, lam)
    ) / (2.0 * step)
    np.testing.assert_allclose(oracle_curve.slope(adstock, lam), central, rtol=1e-6)


def test_the_market_s_media_is_its_geos_summed_by_population_and_its_channel_s():
    panel = GeoPanel.draw(900)
    world = panel.world
    assert panel.population.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(panel.population @ panel.intensity, 1.0, rtol=1e-12)
    with jax.enable_x64(True):  # the channels' parameters take the precision they are made at
        channels = panel.truth()
        for column, (curve, channel) in enumerate(zip(panel.curves(), channels, strict=True)):
            spend = world.spend[:, column]
            kernel = _kernel(world.retention[column], world.kernel_length)
            adstock = np.convolve(spend, kernel)[: spend.size]
            market = panel.population @ panel.media(column, spend)
            expected = world.effect[column] * curve.value(adstock, world.saturation[column])
            np.testing.assert_allclose(market, expected, rtol=1e-12)
            np.testing.assert_allclose(np.asarray(channel(spend)), expected, rtol=1e-10)


def test_a_test_moves_only_its_geos_and_only_from_its_first_dark_week():
    panel = GeoPanel.draw(901)
    geos = [3, 17, 28]
    moved = panel.sales(geos) - panel.sales()
    others = np.setdiff1d(np.arange(GEOS), geos)
    first = STARTS[0] - 1
    assert not np.any(moved[others])
    assert not np.any(moved[:, :first])
    assert np.all(moved[geos] <= 0.0)
    assert np.all(moved[geos, first] < 0.0)
    assert not np.any(dark(panel.world.spend[:, TESTED])[first : first + 4])


def test_every_candidate_holds_the_matched_share_and_the_pool_is_its_seed_s():
    panel = GeoPanel.draw(902)
    candidates = pool(panel, 902)
    assert len(candidates) == POOL + 2
    assert candidates == pool(panel, 902)
    assert candidates != pool(panel, 903)
    for candidate in candidates:
        assert SHARE[0] <= panel.population[list(candidate.geos)].sum() <= SHARE[1]
    per_head = {c.name: design(panel, panel.sales(), c).spend_per_head(panel) for c in candidates}
    drawn = [v for name, v in per_head.items() if name.startswith("random")]
    assert per_head["heaviest"] > max(drawn) and per_head["lightest"] < min(drawn)


def test_a_design_reads_the_pre_period_alone_with_its_weights_on_the_simplex():
    panel = GeoPanel.draw(903)
    candidate = pool(panel, 903)[2]
    before = design(panel, panel.sales(), candidate)
    during = design(panel, panel.sales(candidate.geos), candidate)
    assert set(before.donors).isdisjoint(candidate.geos) and len(before.donors) == GEOS - len(
        candidate.geos
    )
    assert np.all(before.weights >= 0.0) and before.weights.sum() == pytest.approx(1.0)
    assert min(before.screen, before.placebo, before.representative) > 0.0
    assert before.screen != before.placebo
    np.testing.assert_array_equal(before.weights, during.weights)
    read = (before.screen, before.placebo, before.representative)
    assert read == (during.screen, during.placebo, during.representative)


def test_a_set_s_gaps_are_what_its_test_reads_and_what_its_fitted_channel_predicts():
    """The regret arm's prediction and the fit read one model: the set's gaps at the truth are the
    panel's own, and the template's channel at the truth predicts them from the tests' spend."""
    panel = GeoPanel.draw(904)
    candidate = pool(panel, 904)[0]
    untested, tested = panel.sales(), panel.sales(candidate.geos)
    chosen = design(panel, untested, candidate)
    geos = list(candidate.geos)
    own = chosen.share(panel) @ (tested[geos] - untested[geos])
    truth = np.concatenate([own[window] for window in readouts()])
    np.testing.assert_allclose(gaps(panel, chosen, panel.theta()), truth, rtol=1e-10, atol=1e-9)
    tests = lift_tests(panel, chosen, tested)
    retention, scale, coefficient = panel.theta()
    with jax.enable_x64(True):
        start = template(panel, chosen, tests)
        read = Channel(
            GeometricAdstock(retention, length=panel.world.kernel_length, normalized=True),
            dataclasses.replace(start.curve, scale=scale),
            coefficient,
        )
        np.testing.assert_allclose(predicted_gaps(read, tests), truth, rtol=1e-9, atol=1e-9)


def test_the_oracle_on_the_mixtures_is_the_library_s_plan_on_the_true_channels():
    panel = GeoPanel.draw(905)
    quarter = Quarter.after(panel.world)
    best = oracle(panel.world, quarter, panel.curves())
    with jax.enable_x64(True):
        weekly = planned(panel.truth(), quarter)
    np.testing.assert_allclose(weekly, best.weekly, rtol=1e-6)
    assert worth(panel.world, quarter, weekly, panel.curves()) == pytest.approx(
        best.worth, rel=1e-9
    )
    assert PLANNED * best.weekly.sum() == pytest.approx(quarter.budget, rel=1e-9)


def test_the_weight_is_the_tested_channel_s_block_and_nought_where_the_plan_pins_it():
    panel = GeoPanel.draw(906)
    quarter = Quarter.after(panel.world)
    retention, scale, _ = panel.theta()
    draws = np.array([panel.theta(), (retention, scale, 1.0)])
    with jax.enable_x64(True):
        blocks = weights(panel, quarter, draws)
        whole = decision_weight(
            panel.truth(),
            quarter.budget,
            PLANNED,
            lower=quarter.lower,
            upper=quarter.upper,
            history=quarter.history,
        )
    assert TESTED not in whole.pinned
    np.testing.assert_allclose(blocks[0], whole.matrix[:3, :3], rtol=1e-12)
    assert np.any(blocks[0]) and not np.any(blocks[1])


def test_the_exposure_reads_the_weight_against_the_test_s_covariance_at_unit_noise():
    panel = GeoPanel.draw(907)
    chosen = design(panel, panel.sales(), pool(panel, 907)[0])
    theta = np.array(panel.theta())
    draws = np.array([theta, theta])
    blocks = np.array([np.diag([4.0, 1e-4, 1e-6]), np.zeros((3, 3))])
    exposed = exposure(panel, chosen, draws, blocks, 100.0)
    slopes = jacobian(panel, chosen, theta)
    covariance = np.linalg.inv(slopes.T @ slopes)
    assert exposed == pytest.approx(0.5 * np.sum(blocks[0] * covariance) / 2 / 100.0, rel=1e-6)
    louder = dataclasses.replace(chosen, screen=3.0 * chosen.screen, placebo=2.0 * chosen.placebo)
    assert exposure(panel, louder, draws, blocks, 100.0) == exposed
    assert exposure(panel, chosen, draws, np.zeros((2, 3, 3)), 100.0) == 0.0
    # where the curve is a line the test reads its slope alone, and the scale and the coefficient
    # do not part: unbounded where they matter, nothing where the plan pins the channel
    line = np.array([[theta[0], 1e12, 1e12 * theta[2] / theta[1]]])
    assert exposure(panel, chosen, line, blocks[:1], 100.0) == math.inf
    assert exposure(panel, chosen, line, np.zeros((1, 3, 3)), 100.0) == 0.0


def test_a_fit_that_reads_a_loss_is_planned_as_nought_and_no_fit_as_the_status_quo():
    panel = GeoPanel.draw(908)
    quarter = Quarter.after(panel.world)
    retention, scale, coefficient = panel.theta()
    names = ("kernel.retention", "curve.scale", "coefficient")

    def fit(value: float) -> LiftFit:
        estimate = np.array([retention, scale, value])
        channel = panel.truth()[TESTED]
        return LiftFit(channel, names, estimate, estimate, estimate, 0.95, 1.0, 61, (0.0, 1.0), ())

    with jax.enable_x64(True):
        loss, _ = _plan(panel, quarter, fit(-coefficient))
        nought, _ = _plan(panel, quarter, fit(0.0))
        read, moved = _plan(panel, quarter, fit(coefficient))
        best = planned(panel.truth(), quarter)
    np.testing.assert_array_equal(loss, nought)
    assert loss[TESTED] == pytest.approx(quarter.lower[TESTED])
    np.testing.assert_allclose(read, best, rtol=1e-12)
    assert moved == 0.0
    status_quo, _ = _plan(panel, quarter, None)
    np.testing.assert_array_equal(status_quo, quarter.status_quo)


def _designs(screens: list[float], placebos: list[float]) -> list[Design]:
    names = ["heaviest", "lightest", "random 0", "random 1", "random 2"]
    representative = [4.0, 2.0, 6.0, 0.5, 3.0]
    return [
        Design(Candidate(name, (k,)), (), np.ones(1), screen, placebo, representative[k])
        for k, (name, screen, placebo) in enumerate(zip(names, screens, placebos, strict=True))
    ]


def test_each_arm_takes_its_own_least_and_the_random_one_its_seed_s():
    """The regret arm reads the screen squared times the exposure, and the least-RMSPE arm the
    screen, neither the placebo; the regret arm's ties go to the least screen."""
    designs = _designs([3.0, 2.0, 1.0, 1.5, 4.0], [0.1, 9.0, 9.0, 9.0, 9.0])
    exposures = np.array([0.5, 1.0, 3.0, 1.0, 0.01])
    picks = choose(designs, exposures, 11)  # screen**2 * exposure: 4.5, 4, 3, 2.25, 0.16
    assert picks["regret"] == 4 and picks["Abadie-Zhao"] == 3 and picks["least RMSPE"] == 2
    assert picks["heaviest"] == 0 and 0 <= picks["random"] < len(designs)
    assert picks == choose(designs, exposures, 11)
    assert set(picks) == set(ARMS)
    squared = choose(designs, np.array([9.0, 9.0, 3.0, 1.5, 9.0]), 11)
    assert squared["regret"] == 2  # 3 against 3.375; by the screen unsquared, 3 against 2.25
    assert choose(designs, np.zeros(5), 11)["regret"] == 2


def test_an_arm_s_prediction_reads_the_placebo_the_choice_did_not():
    arm = dataclasses.replace(_arm((1,), 0.01), screen=2.0, placebo=5.0, exposure=0.003)
    assert arm.predicted == pytest.approx(25.0 * 0.003)


def _arm(geos: tuple[int, ...], regret: float, covered: int = 62) -> ArmScore:
    return ArmScore(
        candidate="random 0",
        geos=geos,
        spend_per_head=1.0,
        screen=9.0,
        placebo=10.0,
        exposure=1e-4,
        estimate=(0.3, 250.0, 800.0),
        failure=None,
        weekly=(1.0, 1.0, 1.0),
        moved=0.0,
        regret=regret,
        error=11.0,
        covered=covered,
        weeks=64,
    )


QUARTILES = {"q25": 0.002, "median": 0.004, "q75": 0.008, "least": 0.001, "unbounded": 0.0}


def _run(
    rng: np.random.Generator, *, same: str | None = None, covered: int = 62, random: int = 62
) -> Run:
    """Worlds where the regret arm is the least by about 0.02 and each arm takes a set of its
    own, or the regret arm takes ``same``'s; the random arm's readings cover ``random`` weeks."""
    scores = []
    for seed in range(40):
        regret = abs(rng.normal(0.01, 0.002))
        arms = {
            arm: _arm((seed % 7, 10 + k), regret + 0.02 + rng.normal(0.0, 0.002))
            for k, arm in enumerate(ARMS)
        }
        arms["random"] = dataclasses.replace(arms["random"], covered=random)
        geos = (8,) if same is None else arms[same].geos
        arms["regret"] = _arm(geos, regret, covered)
        scores.append(WorldScore(seed, 1000.0, 5000.0, arms, QUARTILES, 0.2))
    return Run(tuple(range(40)), tuple(scores), 1.0)


def test_the_gate_wants_the_least_regret_another_set_and_a_placebo_that_holds():
    assert _run(np.random.default_rng(1)).passes
    assert not _run(np.random.default_rng(1), same="Abadie-Zhao").passes
    assert not _run(np.random.default_rng(1), same="least RMSPE").passes
    assert _run(np.random.default_rng(1), same="heaviest").passes
    assert not _run(np.random.default_rng(1), covered=60).passes  # 2 of 64 weeks under random's
    assert _run(np.random.default_rng(1), covered=61).passes
    assert not _run(np.random.default_rng(1), random=64).passes  # read against the random arm
    worse = _run(np.random.default_rng(1))
    flipped = tuple(
        dataclasses.replace(
            s,
            arms={
                **s.arms,
                "Abadie-Zhao": dataclasses.replace(s.arms["Abadie-Zhao"], regret=0.0),
            },
        )
        for s in worse.scores
    )
    assert not dataclasses.replace(worse, scores=flipped).passes


def test_a_fit_whose_scale_runs_past_the_line_s_counts_as_one_and_a_failed_fit_does_not():
    run = _run(np.random.default_rng(3))
    assert run.lines("regret") == 0
    read = {0: (0.3, 2e9, 6e9), 1: (0.3, 2e9, 6e9), 2: None}
    scores = tuple(
        dataclasses.replace(
            s,
            arms={
                **s.arms,
                "regret": dataclasses.replace(s.arms["regret"], estimate=read[s.seed]),
            },
        )
        if s.seed in read
        else s
        for s in run.scores
    )
    assert dataclasses.replace(run, scores=scores).lines("regret") == 2


def test_the_report_parses_and_a_pilot_has_no_gate():
    run = _run(np.random.default_rng(2))
    assert "Gate" not in _markdown(run, pilot=True) and "Gate" in _markdown(run, pilot=False)
    record = json.loads(json.dumps(_record(run, pilot=True), allow_nan=False))
    assert record["gate"] is None and len(record["worlds"]) == 40


def test_the_scored_seeds_are_fresh():
    scored = set(range(FIRST, FIRST + WORLDS))
    assert not scored & set(PILOTS)
    used = {
        *range(500),
        *range(10_000, 10_200),
        *range(20_000, 20_100),
        *range(30_000, 30_100),
        *range(40_000, 40_500),
    }
    assert not scored & used  # lift calibration's, budgets', families' and the check's


def test_one_world_scores_every_arm_against_the_oracle(monkeypatch):
    monkeypatch.setattr(geo_selection, "POOL", 4)
    monkeypatch.setattr(geo_selection, "DRAWS", 2)
    with jax.enable_x64(True):
        score = score_world(900)
    assert set(score.arms) == set(ARMS)
    assert score.arms["heaviest"].candidate == "heaviest"
    panel = GeoPanel.draw(900)
    for arm in score.arms.values():
        assert arm.regret >= -1e-9 * score.best / score.budget
        assert arm.failure is None or arm.estimate is None
        chosen = design(panel, panel.sales(), Candidate(arm.candidate, arm.geos))
        assert (arm.screen, arm.placebo) == (chosen.screen, chosen.placebo)
        error = reading_error(panel, chosen, panel.sales(arm.geos))
        assert arm.weeks == error.size == 4 * 16
        assert arm.covered == np.sum(np.abs(error) <= 1.959963984540054 * chosen.placebo)
        assert arm.error == pytest.approx(np.sqrt(np.mean(error**2)))


def test_main_refuses_to_run_at_float32(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["geo_selection"])
    with jax.enable_x64(False), pytest.raises(SystemExit, match="JAX_ENABLE_X64"):
        main()

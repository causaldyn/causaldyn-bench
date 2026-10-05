"""Family 14: the world's transforms are Robyn's own, as Robyn's library returned them; the world is
Track M v2's beneath its media; and the oracle, the quarter and the returns read Robyn's carryover
and curve, which Robyn's recursion run on past the history confirms."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chc.response import Channel, GeometricAdstock, Hill, marginal_roi, roi

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.mmm_decision import PLANNED, drawn
from causaldyn_bench.scorecard import ladder
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.robyn import (
    GENRE,
    ROBYN,
    adstocked,
    hill,
    parameters,
    saturated,
)
from causaldyn_bench.scorecard.truth import worth

ROOT = Path(__file__).resolve().parents[1]
WEEKS = 156
AGREE = 1e-9  # the record's agreement with Robyn's own functions, relative


@pytest.fixture(scope="module")
def worlds():
    return [ROBYN.world("genre", seed) for seed in ROBYN.scored["genre"][:4]]


def test_the_worlds_transforms_are_robyns_own():
    record = json.loads((ROOT / "results" / "robyn_form.json").read_text())
    assert record["robyn"] == "3.12.1"
    histories = {}
    for case in record["cases"]:
        seed = case["seed"]
        history = histories.setdefault(seed, ROBYN.world("genre", seed).history)
        c = history.channels.index(case["channel"])
        adstock = adstocked(history.spend[:, c], case["decay"])
        np.testing.assert_allclose(adstock, case["adstock"], rtol=AGREE, atol=0.0)
        inflexion = case["gamma"] * adstock.max()
        assert inflexion == pytest.approx(case["inflexion"], rel=AGREE, abs=0.0)
        curve = saturated(adstock, case["shape"], inflexion)
        np.testing.assert_allclose(curve, case["saturated"], rtol=AGREE, atol=0.0)
        if case["corner"] is None:
            ours = (history.retention[c], history.shape[c], history.gamma[c])
            assert ours == (case["decay"], case["shape"], case["gamma"])
            robyns = history.effect[c] * np.asarray(case["saturated"])
            np.testing.assert_allclose(history.media[:, c], robyns, rtol=AGREE, atol=0.0)
    corners = {(case["channel"], case["corner"]) for case in record["cases"]}
    assert corners >= {(name, end) for name in GENRE for end in ("low", "high")}


def test_a_world_is_track_m2s_beneath_its_media(worlds):
    for world in worlds:
        history = world.history
        base = drawn(world.seed).simulate(world.seed)
        assert np.array_equal(history.spend, base.spend)
        assert np.array_equal(history.premedia, base.premedia)
        assert np.array_equal(history.sales, base.premedia + history.media.sum(axis=1))
        np.testing.assert_allclose(history.media.sum(axis=0), base.media.sum(axis=0), rtol=1e-13)
        assert not np.allclose(history.media, base.media, rtol=0.01)


def test_each_channels_parameters_lie_on_its_genres_ranges():
    seeds = ROBYN.scored["genre"][:300]
    drawn_ = np.stack([np.stack(parameters(("pla", "meta", "tv"), seed)) for seed in seeds])
    for c, name in enumerate(("pla", "meta", "tv")):
        for p, (low, high) in enumerate(GENRE[name]):
            values = drawn_[:, p, c]
            width = high - low
            assert low <= values.min() < low + 0.02 * width
            assert high - 0.02 * width < values.max() <= high
            assert np.mean(values < low + width / 2) == pytest.approx(0.5, abs=0.08)


def test_each_inflexion_is_gamma_times_its_channels_peak_adstock(worlds):
    for world in worlds:
        history = world.history
        for c in range(len(history.channels)):
            adstock = adstocked(history.spend[:, c], history.retention[c])
            assert history.saturation[c] == history.gamma[c] * adstock.max()


def test_hills_slope_is_its_values_derivative_and_vanishes_with_the_adstock():
    adstock = np.linspace(1.0, 400.0, 50)
    for shape in (0.5, 1.0, 1.7, 3.0):
        curve = hill(shape)
        step = 1e-5 * adstock
        numeric = (curve.value(adstock + step, 120.0) - curve.value(adstock - step, 120.0)) / (
            2 * step
        )
        np.testing.assert_allclose(curve.slope(adstock, 120.0), numeric, rtol=1e-7)
        assert curve.slope(np.zeros(1), 120.0)[0] == 0.0
        assert curve.concave == (shape <= 1.0)


def test_a_geo_test_reads_robyns_carryover_from_its_first_week_on(worlds):
    history = worlds[0].history
    tv = history.channels.index("tv")
    start = 100
    gap = history.geo_test("tv", (start,), test=4).true_gap
    assert np.all(gap[: start - 1] == 0.0)
    assert np.all(gap[start - 1 :] < 0.0)  # Robyn's carryover never dies out
    assert history.retention[tv] > 0.3


FAR = 5_000  # weeks after which Robyn's carryover returns nothing a float64 sum can hold


def _returned(history, c: int, spend: np.ndarray, after: int) -> np.ndarray:
    """What channel ``c`` returns in each week of Robyn's recursion run on ``spend``, then ``after``
    weeks with nothing spent."""
    adstock = adstocked(np.concatenate([spend, np.zeros(after)]), history.retention[c])
    return history.effect[c] * saturated(adstock, history.shape[c], history.saturation[c])


def test_the_oracles_cells_and_the_quarter_read_robyns_recursion(worlds):
    """Over the quarter each cell returns what the recursion run on past the history does; over its
    horizon it counts all the recursion returns but what the carryover returns more than the
    kernel's length on, which the module's docstring bounds."""
    for world in worlds:
        history, truth = world.history, ROBYN.truth(world)
        quarter = truth.quarter
        for c, cell in enumerate(truth.cells):
            weekly = float(quarter.status_quo[c])
            spend = np.concatenate([history.spend[:, c], np.full(PLANNED, weekly)])
            ours = _returned(history, c, spend, FAR)[WEEKS:]
            assert cell.carry.size == PLANNED + WEEKS - 1
            np.testing.assert_allclose(cell.returns(weekly)[:PLANNED], ours[:PLANNED], rtol=1e-12)
            np.testing.assert_allclose(world.shadow.media[:, c], ours[:PLANNED], rtol=1e-12)
            assert cell.worth(weekly) == pytest.approx(ours.sum(), rel=4e-8, abs=0.0)


def test_the_true_returns_are_robyns_recursion_with_the_windows_spend_removed(worlds):
    for world in worlds:
        history, returns = world.history, ROBYN.truth(world).returns
        first, last = returns.window
        inside = np.zeros(WEEKS, dtype=bool)
        inside[first - 1 : last] = True
        for c in range(len(history.channels)):
            spend = history.spend[:, c]
            spent = spend[inside].sum()

            def total(series, c=c, history=history):
                return _returned(history, c, series, FAR).sum()

            roi = (total(spend) - total(np.where(inside, 0.0, spend))) / spent
            assert returns.roi[c] == pytest.approx(roi, rel=2e-9, abs=0.0)
            step = 1e-6
            up = total(np.where(inside, spend * (1 + step), spend))
            down = total(np.where(inside, spend * (1 - step), spend))
            marginal = (up - down) / (2 * step * spent)
            assert returns.marginal[c] == pytest.approx(marginal, rel=1e-6, abs=0.0)


def test_the_world_and_its_returns_are_chcs_channels_in_robyns_form(worlds):
    """The arms' vocabulary (:mod:`causaldyn_bench.scorecard.mapping`) reads Robyn's model as chc's
    channel of an unnormalised geometric kernel over the history and a Hill at the inflexion; the
    world's media and returns are that channel's."""
    history = worlds[0].history
    returns = ROBYN.truth(worlds[0]).returns
    first, last = returns.window
    with jax.enable_x64(True):
        for c in range(len(history.channels)):
            channel = Channel(
                GeometricAdstock(history.retention[c], length=WEEKS, normalized=False),
                Hill(history.saturation[c], history.shape[c]),
                history.effect[c],
            )
            media = np.asarray(channel(jnp.asarray(history.spend[:, c])))
            np.testing.assert_allclose(media, history.media[:, c], rtol=1e-13)
            spend = jnp.asarray(np.concatenate([history.spend[:, c], np.zeros(WEEKS - 1)]))
            window = slice(first - 1, last)
            ours = (returns.roi[c], returns.marginal[c])
            chcs = (float(roi(channel, spend, window)), float(marginal_roi(channel, spend, window)))
            assert ours == pytest.approx(chcs, rel=1e-12, abs=0.0)


def test_no_plan_in_the_box_beats_the_oracle(worlds):
    rng = np.random.default_rng(14)
    for world in worlds:
        truth = ROBYN.truth(world)
        quarter = truth.quarter
        assert quarter.feasible(truth.best.weekly)
        for _ in range(200):
            weekly = quarter.project(rng.uniform(quarter.lower, quarter.upper))
            assert worth(truth.cells, weekly) <= truth.best.worth + TIE * quarter.budget


def test_the_rungs_nest_up_to_what_an_earlier_test_carries_into_a_later_readout(worlds):
    """At rung 4 the tests from weeks 20, 55 and 100 go dark in the latest test's universe too, and
    Robyn's carryover brings a little of each into its readout, at least 28 weeks on."""
    for world in worlds:
        history = world.history
        tv = history.channels.index("tv")
        latest = [ladder.rung(history, world.seed, k)["tv"][-1].difference for k in (1, 4)]
        moved = np.abs(latest[0] - latest[1]).max() / np.abs(latest[1]).max()
        assert 0.0 < moved <= 2.0 * history.retention[tv] ** 28


def test_every_arm_is_told_the_kernel_spans_the_history(worlds):
    observation = ROBYN.observe(worlds[0], 4)
    assert number(ROBYN) == 14
    assert observation.family == 14 and observation.environment == "genre"
    assert observation.kernel_length == WEEKS == worlds[0].history.week.size
    assert set(ROBYN.pilots) == set(ROBYN.scored) == {"genre"}

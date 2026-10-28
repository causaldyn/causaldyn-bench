"""Family 9, family 8's geo worlds with ring spillover: each world its seed alone, drawn from the
family's blocks; each environment family 8's mix at policy 0 with its spillover, its spreads from
the family's own stream, and at no spillover family 8's world once family 8's spreads are given;
every cell's worth is what chc's world lifts over the quarter and its tail, the spill counted, and
every cell is concave, so the oracle is exact, checked against a search written apart from it; a
channel's returns are the market's, a geo's own channels' ``1 + s`` times their returns at home; a
geo test's controls receive what spills from its tested geos, so on a world still but for its
season a reading is the tested geos' change less the slope times the spill; every world exports,
reads back and scores, the oracle's own plan at nought; and family 8's and family 10's exports
keep their bits now that a world may spill."""

import dataclasses
import json
import math

import jax
import numpy as np
import pytest
from chc.geo_world import GeoMediaWorld
from scipy.optimize import minimize

from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.lift_calibration import COOLDOWN, TEST
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard import truth
from causaldyn_bench.scorecard.continuation import lifted
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.geo import GEO, drawn, mix, window
from causaldyn_bench.scorecard.ladder import GEO_STARTS, dark_geos, geo_rung
from causaldyn_bench.scorecard.observe import VERSION, export, read
from causaldyn_bench.scorecard.returns import true_geo_returns
from causaldyn_bench.scorecard.scoring import WorldRecord, score_world
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored
from causaldyn_bench.scorecard.spillover import FAMILY, SPILLOVER, SPILLOVERS
from causaldyn_bench.scorecard.truth import allotted, geo_cells, geo_quarter

LENGTH = 8  # the kernel's
HORIZON = PLANNED + LENGTH - 1

# Each environment's first pilot world's export at rungs 0 and 4, read before ``mix`` took a
# spillover.
DIGESTS = {
    (GEO, "policy-0"): ("b35cccc30d161f74", "6960cc8dcd601dd1"),
    (GEO, "policy-0.5"): ("a846b14a3a7adf90", "7876835bb73ac1b3"),
    (FEW_GEOS, "geos-3"): ("ef143986dfd2274a", "1859cea3dad99599"),
    (FEW_GEOS, "geos-50"): ("93e080fc8fbe9bdc", "06667d86d2629408"),
}


def _redrawn(world: GeoMediaWorld, seed: int, **change) -> GeoMediaWorld:
    """``world``'s mix with ``change``, drawn from ``seed`` at double precision."""
    with jax.enable_x64(True):
        return dataclasses.replace(world.mix, **change).draw(seed)


def test_the_family_draws_its_two_environments_from_its_own_blocks():
    assert number(SPILLOVER) == FAMILY == 9 and tuple(SPILLOVER.pilots) == tuple(SPILLOVER.scored)
    assert dict(SPILLOVERS) == {"spillover-0.2": 0.2, "spillover-0.5": 0.5}
    for e, name in enumerate(SPILLOVERS):
        assert SPILLOVER.pilots[name] == pilot(9, e) and SPILLOVER.scored[name] == scored(9, e)
    assert SPILLOVER.labels[: len(GEO.labels)] == GEO.labels
    assert len(SPILLOVER.labels) == len(GEO.labels) + 2
    assert "s/2" in SPILLOVER.labels[-2] and "national media do not spill" in SPILLOVER.labels[-2]
    assert "controls" in SPILLOVER.labels[-1] and "spill" in SPILLOVER.labels[-1]
    with pytest.raises(ValueError, match="no environment 'policy-0'"):
        SPILLOVER.world("policy-0", scored(9, 0)[0])


@pytest.mark.parametrize("name", list(SPILLOVERS))
def test_a_world_is_its_seed_alone_at_double_precision_whatever_the_setting(name):
    seed = SPILLOVER.scored[name][1]
    first, again, other = (SPILLOVER.world(name, s) for s in (seed, seed, seed + 1))
    with jax.enable_x64(True):
        wide = SPILLOVER.world(name, seed)
    assert first.history.media.dtype == np.float64
    for k in (0, 4):
        stamp = SPILLOVER.observe(first, k).digest()
        assert stamp == SPILLOVER.observe(again, k).digest()
        assert stamp == SPILLOVER.observe(wide, k).digest()
        assert stamp != SPILLOVER.observe(other, k).digest()
    ours, theirs = SPILLOVER.truth(first), SPILLOVER.truth(again)
    assert ours.best.worth == theirs.best.worth
    np.testing.assert_array_equal(ours.target, theirs.target)


@pytest.mark.parametrize("name", list(SPILLOVERS))
def test_an_environment_is_family_8s_mix_at_policy_0_with_its_spillover(name):
    """Every parameter but the spillover is family 8's at policy 0; each channel's spread is drawn
    from the family's own stream."""
    seed = SPILLOVER.pilots[name][3]
    world = SPILLOVER.world(name, seed).history
    assert world.mix.spillover == SPILLOVERS[name] and world.mix.policy == 0.0
    expected = rng(9, Role.PARAMETERS, seed).uniform(0.2, 0.6, 3)
    np.testing.assert_array_equal([c.spread for c in world.mix.channels], expected)
    assert not np.array_equal(expected, rng(8, Role.PARAMETERS, seed).uniform(0.2, 0.6, 3))
    eight = mix(0.0, seed, family=9)
    for field in dataclasses.fields(world.mix):
        if field.name not in {"channels", "spillover"}:
            assert getattr(world.mix, field.name) == getattr(eight, field.name), field.name

    def stated(channel) -> tuple:
        curve = channel.curve
        return (
            channel.name,
            type(curve).__name__,
            float(curve.scale),
            channel.effect,
            channel.spread,
            channel.retention,
            channel.spend,
            channel.national,
        )

    assert [stated(c) for c in world.mix.channels] == [stated(c) for c in eight.channels]


@pytest.mark.parametrize("name", list(SPILLOVERS))
def test_without_spill_and_at_family_8s_spreads_a_world_is_family_8s_bit_for_bit(name):
    """The premedia, the spend, the populations and the noise are family 8's at the seed, whatever
    the spreads; given family 8's spreads and no spillover, the effects and the KPI are too. At the
    spillover the KPI is the same world's with what spills in added, a geo's own media alone."""
    seed = SPILLOVER.pilots[name][0]
    world = SPILLOVER.world(name, seed).history
    eight = GEO.world("policy-0", seed).history
    for field in ("base", "spend", "population", "noise"):
        np.testing.assert_array_equal(getattr(world, field), getattr(eight, field), err_msg=field)
    given = dataclasses.replace(world.mix, channels=mix(0.0, seed).channels, spillover=0.0)
    flat = drawn("policy-0", seed, given).history
    for field in ("base", "spend", "effect", "media", "noise", "kpi"):
        np.testing.assert_array_equal(getattr(flat, field), getattr(eight, field), err_msg=field)
    np.testing.assert_array_equal(flat.spilled, 0.0)
    unspilled = _redrawn(world, seed, spillover=0.0)
    np.testing.assert_array_equal(unspilled.media, world.media)
    home = world.media[:, :, :2].sum(axis=2)
    spill = SPILLOVERS[name] / 2.0 * (np.roll(home, 1, axis=0) + np.roll(home, -1, axis=0))
    np.testing.assert_allclose(world.spilled, spill, rtol=1e-15)
    np.testing.assert_allclose(world.kpi, unspilled.kpi + world.spilled, rtol=1e-14)


@pytest.mark.parametrize("name", list(SPILLOVERS))
def test_a_cells_worth_is_what_chcs_world_lifts_over_the_quarter_and_its_tail(name):
    """Every geo's KPI over the quarter and the kernel's tail, spent the plan and then nothing,
    from chc's own world run on, what spills into each geo in: summed, the cells' worth to
    rounding, the history's carryover in."""
    world = SPILLOVER.world(name, SPILLOVER.pilots[name][2]).history
    quarter = geo_quarter(world)
    on = geo_cells(world, quarter)
    assert len(on) == 21 and all(cell.concave for cell in on)
    draw = np.random.default_rng(1)
    plans = [quarter.status_quo, quarter.lower, quarter.upper] + [
        quarter.project(draw.uniform(quarter.lower, quarter.upper)) for _ in range(4)
    ]
    for weekly in plans:
        kpi = lifted(world, allotted(world, weekly), HORIZON)
        assert truth.worth(on, weekly) == pytest.approx(float(kpi.sum()), rel=1e-12, abs=0.0)


def _apart(on, quarter: Quarter, starts: list[np.ndarray]) -> float:
    """The best worth SLSQP reaches from ``starts`` with finite-difference gradients, each cell's
    spend in units of its status quo: a search that shares nothing with the oracle's. The worth is
    a sum over the cells, so each cell's difference reads that cell alone."""
    unit = quarter.status_quo
    rate = quarter.budget / PLANNED
    step = 1e-7

    def gradient(u: np.ndarray) -> np.ndarray:
        return (
            np.array(
                [
                    (cell.worth((x + step) * w) - cell.worth((x - step) * w)) / (2 * step)
                    for cell, x, w in zip(on, u, unit, strict=True)
                ]
            )
            / -quarter.budget
        )

    best = -math.inf
    for start in starts:
        solution = minimize(
            lambda u: -truth.worth(on, u * unit) / quarter.budget,
            start / unit,
            jac=gradient,
            method="SLSQP",
            bounds=list(zip(quarter.lower / unit, quarter.upper / unit, strict=True)),
            constraints={"type": "eq", "fun": lambda u: float(u @ unit) / rate - 1.0},
            options={"ftol": 1e-15, "maxiter": 1000},
        )
        weekly = solution.x * unit
        if quarter.feasible(weekly, tolerance=1e-9):
            best = max(best, truth.worth(on, weekly))
    return best


@pytest.mark.parametrize("name", list(SPILLOVERS))
def test_the_exact_plan_meets_its_conditions_and_no_search_beats_it(name):
    world = SPILLOVER.world(name, SPILLOVER.pilots[name][4])
    found = SPILLOVER.truth(world)
    quarter, plan, on = found.quarter, found.best, found.cells
    assert all(cell.concave for cell in on)
    assert quarter.feasible(plan.weekly) and math.isfinite(plan.price)
    free = 0
    for cell, weekly, low, high in zip(on, plan.weekly, quarter.lower, quarter.upper, strict=True):
        slope = cell.slope(float(weekly)) / PLANNED
        if np.isclose(weekly, low, rtol=1e-9):
            assert slope <= plan.price * (1 + 1e-9)
        elif np.isclose(weekly, high, rtol=1e-9):
            assert slope >= plan.price * (1 - 1e-9)
        else:
            free += 1
            assert slope == pytest.approx(plan.price, rel=1e-8, abs=0.0)
    assert free >= 1
    draw = np.random.default_rng(7)
    starts = [quarter.status_quo, quarter.project(draw.uniform(quarter.lower, quarter.upper))]
    apart = _apart(on, quarter, starts)
    assert plan.worth * (1 - 1e-8) <= apart <= plan.worth + TIE * quarter.budget


@pytest.mark.parametrize("name", list(SPILLOVERS))
def test_a_geo_channels_returns_are_its_home_returns_times_one_and_the_spillover(name):
    """A geo's own media lift ``s / 2`` of their home lift in each of two neighbours, so over the
    ring the market's lift of a per-geo channel is ``1 + s`` times its lift at home; TV does not
    spill."""
    seed = SPILLOVER.pilots[name][9]
    world = SPILLOVER.world(name, seed).history
    found = true_geo_returns(world, window(world))
    home = true_geo_returns(_redrawn(world, seed, spillover=0.0), window(world))
    factor = np.array([1.0 + SPILLOVERS[name]] * 2 + [1.0])
    np.testing.assert_allclose(found.roi, factor * home.roi, rtol=1e-12)
    np.testing.assert_allclose(found.marginal, factor * home.marginal, rtol=1e-9)


def _changes(world: GeoMediaWorld, c: int, j: int) -> tuple[float, float]:
    """What the ``j``-th test of channel ``c`` costs its tested geos, and the market, over its
    readout, a dark week, from chc's world: its lift with the geos dark less its lift as spent."""
    start, geos = GEO_STARTS[j], list(dark_geos(world.mix.geos, j))
    dark = world.spend.copy()
    dark[geos, start - 1 : start - 1 + TEST, c] = 0.0
    with jax.enable_x64(True):
        gap = np.asarray(world.lift(dark)) - np.asarray(world.lift(world.spend))
    readout = gap[:, start - 1 : start - 1 + TEST + COOLDOWN]
    return float(readout[geos].sum()) / TEST, float(readout.sum()) / TEST


@pytest.mark.parametrize("name", list(SPILLOVERS))
def test_on_a_world_still_but_for_its_season_a_reading_is_the_change_less_the_controls_spill(name):
    """Without the spend's jitter and the noise, the tested geos' KPI is on a line with the
    controls' at the slope ``b`` of their bases, and a tested geo's neighbours are controls: the
    tested geos lose their home lift and nothing that spills, the controls ``s`` times it, and the
    reading is ``(1 - b s)`` times the tested geos' change, where the market's is ``1 + s`` times
    it. TV does not spill, and reads its change."""
    seed = SPILLOVER.pilots[name][7]
    s = SPILLOVERS[name]
    world = _redrawn(SPILLOVER.world(name, seed).history, seed, spend_noise=0.0, noise=0.0)
    rows = geo_rung(world, 9, seed, 4)
    names = [c.name for c in world.mix.channels]
    for row, (channel, start) in enumerate(zip(rows.channel, rows.start, strict=True)):
        c, j = names.index(channel), GEO_STARTS.index(start)
        treated = list(dark_geos(10, j))
        control = [g for g in range(10) if g not in treated]
        b = world.base[treated, 0].sum() / world.base[control, 0].sum()
        tested, market = _changes(world, c, j)
        spill = 0.0 if channel == "tv" else s
        assert market == pytest.approx((1.0 + spill) * tested, rel=1e-12, abs=0.0)
        expected = (1.0 - b * spill) * tested
        assert rows.delta_y[row] == pytest.approx(expected, rel=1e-9, abs=0.0)


def test_every_world_exports_reads_back_and_scores_the_oracles_plan_at_nought(tmp_path):
    for e, name in enumerate(SPILLOVERS):
        seed = SPILLOVER.scored[name][e]
        world = SPILLOVER.world(name, seed)
        best = SPILLOVER.truth(world).best
        for k in (0, 4):
            observation = SPILLOVER.observe(world, k)
            assert (observation.family, observation.environment) == (9, name)
            assert observation.sales.shape == (104, 10)
            assert observation.spend.shape == (104, 10, 3)
            assert observation.national == ("tv",) and observation.controls == {}
            assert observation.lower.shape == (21,) and observation.kernel_length == LENGTH
            assert len(observation.lift.channel) == 3 * k
            back = read(export(observation, tmp_path / name / str(k)))
            assert back.digest() == observation.digest()
            assert (back.family, back.environment) == (9, name)
            np.testing.assert_array_equal(back.sales, world.history.kpi.T)
            np.testing.assert_array_equal(back.lift.delta_y, observation.lift.delta_y)
            record = {
                "digest": observation.digest(),
                "version": VERSION,
                "error": None,
                "weekly": best.weekly.tolist(),
            }
            scores = score_world(SPILLOVER, name, seed, k, {"oracle": record})
            assert (scores.family, scores.environment, scores.index, scores.k) == (9, name, e, k)
            assert scores.channels == 21 and set(scores.returns) == {"search", "social", "tv"}
            assert abs(scores.arms["oracle"].regret) <= TIE
            for arm in scores.arms.values():
                assert arm.regret >= -TIE and arm.realised is None and arm.failure is None
            assert scores.arms["status quo"].regret > TIE
            again = WorldRecord.from_json(json.loads(json.dumps(scores.as_json())))
            assert again.arms["oracle"].regret == scores.arms["oracle"].regret


@pytest.mark.parametrize(("family", "name"), list(DIGESTS), ids=[n for _, n in DIGESTS])
def test_family_8s_and_10s_exports_keep_their_bits(family, name):
    world = family.world(name, family.pilots[name][0])
    stamps = tuple(family.observe(world, k).digest() for k in (0, 4))
    assert stamps == DIGESTS[family, name]

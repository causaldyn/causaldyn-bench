"""The shadow quarter against the generator it continues: the same calendar, each process run on
from where the history left it, the measurement layer's rates, and media read through the world's
own channels at the status quo."""

import dataclasses

import numpy as np
import pytest

from causaldyn_bench.endogenous_mmm import EndogenousMediaMix, _media
from causaldyn_bench.mmm_decision import PLANNED, Quarter, drawn
from causaldyn_bench.scorecard.continuation import shadow

SEEDS = range(300)


@pytest.fixture(scope="module")
def runs():
    generator = EndogenousMediaMix()
    out = []
    for seed in SEEDS:
        world = generator.simulate(seed)
        out.append((world, shadow(generator, world, seed)))
    return out


def test_the_quarter_runs_on_the_historys_calendar_and_seasonality():
    longer = EndogenousMediaMix(weeks=156 + PLANNED).simulate(3)
    world = EndogenousMediaMix().simulate(3)
    quarter = shadow(EndogenousMediaMix(), world, 3)
    np.testing.assert_array_equal(quarter.week, longer.week[156:])
    np.testing.assert_array_equal(quarter.seasonality, longer.seasonality[156:])
    np.testing.assert_array_equal(quarter.discount, longer.discount[156:])
    assert quarter.discount[1:4].tolist() == [0.4] * 3  # the winter clearance, weeks 2 to 4


@pytest.mark.parametrize(
    ("name", "persistence", "drift", "sd"),
    [
        ("quality", 0.95, 0.0, 0.02),
        ("price_position", 0.9, 0.1, 0.03),
        ("sentiment", 0.7, 0.0, 0.03),
    ],
)
def test_each_process_runs_on_from_its_value_at_the_historys_end(
    runs, name, persistence, drift, sd
):
    """The quarter's first step from the history's last value, and every later one, is one of the
    process's own shocks: started anywhere else, the first would carry the process's spread."""
    first = np.array(
        [getattr(q, name)[0] - persistence * getattr(w, name)[-1] - drift for w, q in runs]
    )
    later = np.concatenate(
        [getattr(q, name)[1:] - persistence * getattr(q, name)[:-1] - drift for _, q in runs]
    )
    for shocks in (first, later):
        assert abs(np.mean(shocks)) < 4 * sd / np.sqrt(shocks.size)
        assert np.std(shocks, ddof=1) == pytest.approx(sd, rel=0.12)


def test_the_baseline_and_the_promotions_follow_the_generators_equations(runs):
    base = EndogenousMediaMix().base
    for _, quarter in runs[:20]:
        np.testing.assert_allclose(quarter.premedia, quarter.baseline + quarter.promotion)
        assert np.all(quarter.baseline >= 0.5 * base)
        promoted = quarter.discount > 0.0
        assert np.all(quarter.promotion[~promoted] == 0.0)
        assert np.all(quarter.promotion[promoted] > 0.0)
        np.testing.assert_allclose(quarter.sales, quarter.premedia + quarter.media.sum(axis=1))


def test_the_measurement_layer_reads_the_quarter_at_the_papers_rates(runs):
    promoted = np.concatenate([q.discount > 0.0 for _, q in runs])
    read = np.concatenate([q.observed_promotion for _, q in runs])
    assert np.mean(read[promoted]) == pytest.approx(0.7, abs=0.05)
    assert np.mean(read[~promoted]) == pytest.approx(0.15, abs=0.03)
    error = np.concatenate(
        [q.observed_price - q.price_position * (1 - q.discount) for _, q in runs]
    )
    assert np.std(error) == pytest.approx(0.08 / np.sqrt(1 - 0.2**2), rel=0.1)


@pytest.mark.parametrize("curve", ["tanh", "hill-2"])
def test_media_at_the_status_quo_is_each_channel_run_on_from_its_history(curve):
    generator = dataclasses.replace(drawn(11), curve=curve)
    world = generator.simulate(11)
    quarter = shadow(generator, world, 11)
    status_quo = Quarter.after(world).status_quo
    for c, (alpha, lam, beta) in enumerate(
        zip(world.retention, world.saturation, world.effect, strict=True)
    ):
        spend = np.concatenate([world.spend[:, c], np.full(PLANNED, status_quo[c])])
        run_on = _media(spend, alpha, lam, beta, world.kernel_length, curve)[156:]
        np.testing.assert_allclose(quarter.media[:, c], run_on, rtol=1e-12)


def test_an_effect_path_scales_each_weeks_media_and_moves_nothing_else():
    generator = drawn(12)
    world = generator.simulate(12)
    horizon = PLANNED + world.kernel_length - 1
    path = np.exp(np.random.default_rng(0).normal(0.0, 0.2, (len(world.channels), horizon)))
    still = shadow(generator, world, 12)
    moved = shadow(generator, world, 12, np.asarray(world.effect)[:, None] * path)
    np.testing.assert_allclose(moved.media, still.media * path[:, :PLANNED].T, rtol=1e-13)
    np.testing.assert_array_equal(moved.premedia, still.premedia)
    np.testing.assert_array_equal(moved.observed_price, still.observed_price)


def test_the_quarter_is_drawn_from_track_m2s_stream_for_the_worlds_seed():
    generator = drawn(13)
    world = generator.simulate(13)
    again = shadow(generator, world, 13)
    np.testing.assert_array_equal(again.sales, shadow(generator, world, 13).sales)
    assert not np.array_equal(again.sales, shadow(generator, world, 14).sales)
    first = np.random.default_rng((100, 3, 13)).normal(0.0, 0.02)
    assert again.quality[0] - 0.95 * world.quality[-1] == pytest.approx(first, abs=1e-15)

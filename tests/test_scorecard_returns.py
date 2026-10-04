"""The returns on a window's spend, its ROI and mROI with all the carryover: the world's channels
are its own media, every draw is read as chc reads one channel at a time, an arm whose draws are the
world's channels recovers the world's returns, and the score reads a response only where its tool's
decomposition holds it."""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chc.response import Channel, GeometricAdstock, Hill, Tanh, marginal_roi, roi

from causaldyn_bench.endogenous_mmm import CURVES, LN3
from causaldyn_bench.lift_calibration import STARTS
from causaldyn_bench.scorecard.drift import DRIFT
from causaldyn_bench.scorecard.mapping import TOLERANCE, channels, residual
from causaldyn_bench.scorecard.observe import VERSION
from causaldyn_bench.scorecard.returns import drawn_returns, true_returns
from causaldyn_bench.scorecard.scoring import WorldRecord, score_arm, score_world
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, window

STILL = (TRACK_M2, "drawn", 905)
MOVING = (DRIFT, "trend", 220100)
SPREAD = (0.9, 1.0, 1.1)  # an arm's three draws, the world's coefficient times each
TOP = len(STARTS)
NEW = dict.fromkeys(("roi", "mroi", "roi_covered", "mroi_covered", "returns", "unread"))


@pytest.fixture(scope="module", params=[STILL, MOVING], ids=["still", "moving"])
def drawn_world(request):
    family, environment, seed = request.param
    world = family.world(environment, seed)
    return family, world, family.observe(world, 0)


def _slice(observed):
    first, last = observed.roi_window
    return slice(first - 1, last)


def _chc(channel, observed, c, length):
    """chc's ROI and mROI of one channel on the observation's window, the series run on past the
    history for the ``length - 1`` weeks a kernel of ``length`` carries it into, with nothing
    spent."""
    spend = np.concatenate([observed.spend[:, c], np.zeros(length - 1)])
    return roi(channel, spend, _slice(observed)), marginal_roi(channel, spend, _slice(observed))


def test_a_worlds_channels_are_its_own_media_week_by_week(drawn_world):
    _, world, _ = drawn_world
    history = world.history
    path = history.effect_path()
    for c, (alpha, lam) in enumerate(zip(history.retention, history.saturation, strict=True)):
        weights = alpha ** np.arange(history.kernel_length)
        adstock = np.convolve(history.spend[:, c], weights / weights.sum())[: path.shape[0]]
        ours = path[:, c] * CURVES[history.curve].value(adstock, lam)
        media = history.media[:, c]
        assert np.max(np.abs(ours - media)) <= 1e-14 * np.max(np.abs(media))


@pytest.mark.parametrize(
    ("curve", "chcs"),
    [("tanh", lambda lam: Tanh(2.0 / lam)), ("hill-2", lambda lam: Hill(LN3 / lam, 2.0))],
)
def test_a_worlds_returns_are_chcs_its_effect_held_past_the_history(drawn_world, curve, chcs):
    _, world, observed = drawn_world
    history = dataclasses.replace(world.history, curve=curve)
    path = history.effect_path()
    truth = true_returns(history, observed.roi_window)
    weeks, length = path.shape[0], history.kernel_length
    pairs = zip(history.retention, history.saturation, strict=True)
    with jax.enable_x64(True):
        for c, (alpha, lam) in enumerate(pairs):
            base = Channel(GeometricAdstock(alpha, length=length, normalized=True), chcs(lam), 1.0)
            held = np.concatenate([path[:, c], np.full(weeks, path[-1, c])])

            def channel(spend, base=base, held=held):
                return jnp.asarray(held[: spend.shape[0]]) * base(spend)

            expected = _chc(channel, observed, c, length)
            assert truth.roi[c] == pytest.approx(expected[0], rel=1e-13, abs=0.0)
            assert truth.marginal[c] == pytest.approx(expected[1], rel=1e-13, abs=0.0)


def _response(observed, draws, *, curves, paths=False, normalized=True):
    """A response of ``draws`` draws whose parameters move from draw to draw, its decomposition
    chc's own: every curve in ``curves``, in turn, a coefficient a path where ``paths``, the kernel
    over every week of the history where not ``normalized``, as Robyn's runs."""
    names, spend = observed.channels, observed.spend
    weeks = spend.shape[0]
    rng = np.random.default_rng(7)
    listed = [curves[i % len(curves)] for i in range(draws)]
    parameters = {}
    for c, name in enumerate(names):
        level = float(np.mean(spend[:, c]))
        columns = {
            "retention": rng.uniform(0.0, 0.9, draws).tolist(),
            "scale": (level * rng.uniform(0.3, 3.0, draws)).tolist(),
            "coefficient": rng.uniform(100.0, 900.0, draws).tolist(),
        }
        if "Hill" in curves:
            slopes = rng.uniform(0.6, 3.5, draws)
            columns["slope"] = [
                float(s) if k == "Hill" else None for s, k in zip(slopes, listed, strict=True)
            ]
        if paths:
            walk = np.cumsum(rng.normal(0.0, 0.02, (draws, weeks)), axis=1)
            columns["coefficient"] = (
                np.array(columns["coefficient"])[:, None] * np.exp(walk)
            ).tolist()
        parameters[name] = columns
    response = {
        "kernel": "GeometricAdstock",
        "length": observed.kernel_length if normalized else spend.shape[0],
        "normalized": normalized,
        "curve": listed[0] if len(curves) == 1 else listed,
        "draws": draws,
        "parameters": parameters,
    }
    with jax.enable_x64(True):
        built = channels({**response, "decomposition": {}}, names)
        returns = [[np.asarray(draw[c](spend[:, c])) for draw in built] for c in range(len(names))]
    response["decomposition"] = {
        "total": {name: [float(r.sum()) for r in returns[c]] for c, name in enumerate(names)},
        "held": [0, draws - 1],
        "weekly": {
            name: [returns[c][0].tolist(), returns[c][-1].tolist()] for c, name in enumerate(names)
        },
    }
    return response


@pytest.mark.parametrize(
    ("curves", "paths", "normalized"),
    [
        (("Tanh",), False, True),
        (("Hill",), False, True),
        (("Tanh", "Hill"), False, True),
        (("Hill", "Tanh"), True, True),
        (("Hill",), False, False),
    ],
)
def test_every_draws_returns_are_chcs_one_channel_at_a_time(curves, paths, normalized):
    observed = TRACK_M2.observe(TRACK_M2.world(*STILL[1:]), 0)
    response = _response(observed, 6, curves=curves, paths=paths, normalized=normalized)
    with jax.enable_x64(True):
        assert residual(response, observed) <= TOLERANCE
        drawn = drawn_returns(response, observed)
        built = channels(response, observed.channels)
        assert drawn.roi.shape == drawn.marginal.shape == (6, len(observed.channels))
        for i, draw in enumerate(built):
            for c, channel in enumerate(draw):
                expected = _chc(channel, observed, c, response["length"])
                assert drawn.roi[i, c] == pytest.approx(expected[0], rel=1e-12, abs=0.0)
                assert drawn.marginal[i, c] == pytest.approx(expected[1], rel=1e-12, abs=0.0)


def _truths(world, observed, spread=SPREAD):
    """The response of an arm whose draws are the world's channels, each draw's coefficient the
    world's times one of ``spread``, its decomposition the world's own media."""
    history = world.history
    path = history.effect_path()
    tanh = history.curve == "tanh"
    moving = not np.allclose(path, path[0])
    parameters, total, weekly = {}, {}, {}
    for c, name in enumerate(observed.channels):
        lam = history.saturation[c]
        coefficient = [(f * path[:, c]).tolist() if moving else f * path[0, c] for f in spread]
        parameters[name] = {
            "retention": [history.retention[c]] * len(spread),
            "scale": [2.0 / lam if tanh else LN3 / lam] * len(spread),
            "coefficient": coefficient,
        } | ({} if tanh else {"slope": [2.0] * len(spread)})
        total[name] = [float(f * history.media[:, c].sum()) for f in spread]
        weekly[name] = [(spread[0] * history.media[:, c]).tolist()]
    return {
        "kernel": "GeometricAdstock",
        "length": history.kernel_length,
        "normalized": True,
        "curve": "Tanh" if tanh else "Hill",
        "draws": len(spread),
        "parameters": parameters,
        "decomposition": {"total": total, "held": [0], "weekly": weekly},
    }


def _plan(observed, **extra):
    return {
        "digest": observed.digest(),
        "version": VERSION,
        "error": None,
        "weekly": list(map(float, observed.status_quo)),
        **extra,
    }


def test_an_arm_whose_draws_are_the_worlds_channels_recovers_its_returns(drawn_world):
    family, world, observed = drawn_world
    truth = family.truth(world)
    response = _truths(world, observed)
    with jax.enable_x64(True):
        assert residual(response, observed) <= TOLERANCE
        arm = score_arm(truth, observed, _plan(observed, response=response))
    assert arm.unread is None
    assert arm.roi is not None and arm.mroi is not None and arm.returns is not None
    assert arm.roi <= 1e-12 and arm.mroi <= 1e-12
    assert arm.roi_covered == arm.mroi_covered == 1.0
    for c, name in enumerate(observed.channels):
        mean, low, high = arm.returns[name]["roi"]
        assert low < truth.returns.roi[c] < high
        assert mean == pytest.approx(truth.returns.roi[c], rel=1e-12, abs=0.0)


def test_an_arm_off_the_worlds_returns_scores_its_error_and_its_coverage(drawn_world):
    family, world, observed = drawn_world
    truth = family.truth(world)
    # each draw's coefficient the world's times one of these: their means 1.2 and 0.8 times its,
    # and the world's own outside the central 90 % of 21 draws, though inside all of them
    spreads = ((1.15, 1.2, 1.25), (0.75, 0.8, 0.85), (0.99, *np.linspace(1.01, 1.2, 20)))
    with jax.enable_x64(True):
        high, low, outer = (
            score_arm(truth, observed, _plan(observed, response=_truths(world, observed, spread=s)))
            for s in spreads
        )
    for arm in (high, low):
        assert arm.roi == pytest.approx(0.2, rel=1e-12, abs=0.0)
        assert arm.mroi == pytest.approx(0.2, rel=1e-12, abs=0.0)
    for arm in (high, low, outer):
        assert arm.roi_covered == arm.mroi_covered == 0.0


def _nudged(response):
    """Every coefficient a hundred-thousandth off: ten times the least slip a mapping's check
    catches."""
    parameters = {
        name: {**columns, "coefficient": [c * (1 + 1e-5) for c in columns["coefficient"]]}
        for name, columns in response["parameters"].items()
    }
    return {**response, "parameters": parameters}


def test_a_response_is_read_only_where_its_tools_decomposition_holds_it():
    world = TRACK_M2.world(*STILL[1:])
    observed = TRACK_M2.observe(world, 0)
    truth = TRACK_M2.truth(world)
    response = _truths(world, observed)
    reasons = {
        "no response": None,
        "unmapped: a time-varying multiplier": {"unmapped": "a time-varying multiplier"},
        "malformed: the response has no ['decomposition']": {
            k: v for k, v in response.items() if k != "decomposition"
        },
    }
    with jax.enable_x64(True):
        for reason, given in reasons.items():
            extra = {} if given is None else {"response": given}
            arm = score_arm(truth, observed, _plan(observed, **extra))
            assert arm.unread == reason
            assert (arm.roi, arm.mroi, arm.roi_covered, arm.returns) == (None,) * 4
        off = score_arm(truth, observed, _plan(observed, response=_nudged(response)))
        failed = score_arm(truth, observed, _plan(observed, response=response) | {"error": "x"})
    assert off.unread is not None and off.unread.endswith(
        f"off its tool's decomposition, beyond {TOLERANCE:g}"
    )
    assert off.roi is None
    assert (failed.unread, failed.roi) == (None, None)


def test_a_world_record_keeps_the_worlds_returns_and_reads_a_record_without_them():
    world = TRACK_M2.world(*STILL[1:])
    observed = TRACK_M2.observe(world, TOP)
    truth = TRACK_M2.truth(world)
    with jax.enable_x64(True):
        records = {"x": _plan(observed, response=_truths(world, observed))}
        scored = score_world(TRACK_M2, "drawn", 905, TOP, records, pilot=True)
    assert scored.returns == {
        name: {"roi": truth.returns.roi[c], "mroi": truth.returns.marginal[c]}
        for c, name in enumerate(observed.channels)
    }
    assert scored.arms["status quo"].unread is None and scored.arms["x"].unread is None
    text = json.dumps(scored.as_json(), allow_nan=False)
    assert WorldRecord.from_json(json.loads(text)) == scored
    older = json.loads(text)
    del older["returns"]
    for arm in older["arms"].values():
        for key in NEW:
            del arm[key]
    back = WorldRecord.from_json(older)
    assert back.returns == {}
    assert (back.arms["x"].roi, back.arms["x"].returns, back.arms["x"].unread) == (None,) * 3
    assert dataclasses.replace(back.arms["x"], **NEW) == dataclasses.replace(
        scored.arms["x"], **NEW
    )


def test_a_truth_and_an_export_read_returns_over_one_window():
    world = TRACK_M2.world(*STILL[1:])
    observed = TRACK_M2.observe(world, 0)
    truth = TRACK_M2.truth(world)
    assert truth.returns.window == observed.roi_window == window(world.history)
    shifted = dataclasses.replace(observed, roi_window=(100, 156))
    with pytest.raises(RuntimeError, match="the export reads returns over weeks"):
        score_arm(truth, shifted, _plan(shifted))

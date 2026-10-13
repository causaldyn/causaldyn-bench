"""The Meridian arm's ROI calibration, read from ``scripts/meridian_roi.py`` without Meridian: a
channel's tests pooled by inverse variance, and a channel whose pooled ROI is nought or below, which
a LogNormal of that mean cannot take, kept on Meridian's default prior and named, as a channel with
no test is."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.observe import export

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "meridian_roi.py"
_SPEC = importlib.util.spec_from_file_location("meridian_roi", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
roi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(roi)


def _world(channel, delta_x, delta_y, sigma):
    return {
        "lift_channel": np.array(channel),
        "lift_delta_x": np.array(delta_x, dtype=float),
        "lift_delta_y": np.array(delta_y, dtype=float),
        "lift_sigma": np.array(sigma, dtype=float),
    }


def test_a_channels_tests_are_pooled_by_inverse_variance():
    """Two tests of ROI 2 and 0.5, standard errors 0.5 and 1: weights 4 and 1, so 1.7 with a
    standard error of 1 / sqrt(5); the other channel's one test is its own."""
    world = _world(["a", "b", "a"], [-4.0, -1.0, -2.0], [-8.0, -3.0, -1.0], [2.0, 0.5, 2.0])
    pools = roi.pooled(world, ["a", "b", "c"])
    assert pools["a"]["rows"] == 2
    assert pools["a"]["roi"] == pytest.approx(1.7, rel=1e-15, abs=0.0)
    assert pools["a"]["error"] == pytest.approx(1.0 / math.sqrt(5.0), rel=1e-15, abs=0.0)
    assert pools["b"]["rows"] == 1
    assert pools["b"]["roi"] == pytest.approx(3.0, rel=1e-15, abs=0.0)
    assert pools["b"]["error"] == pytest.approx(0.5, rel=1e-15, abs=0.0)
    assert pools["c"] is None


def test_a_channel_whose_pooled_roi_is_not_positive_keeps_the_default_and_is_named():
    """A LogNormal of mean m has no parameters at m <= 0, so a channel whose tests pool to a ROI of
    nought or below keeps Meridian's default, as one with no test does; one above nought, however
    small, is calibrated."""
    world = _world(
        ["up", "flat", "flat", "down", "tiny"],
        [-2.0, -1.0, -1.0, -1.0, -1.0],
        [-3.0, 1.0, -1.0, 0.5, -1e-12],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    )
    pools = roi.pooled(world, ["up", "flat", "down", "tiny", "none"])
    assert pools["flat"]["roi"] == 0.0 and pools["down"]["roi"] == -0.5
    assert pools["tiny"]["roi"] > 0.0
    why = roi.defaulted(pools)
    assert list(why) == ["flat", "down", "none"]
    assert why["none"] == "no lift row"
    assert why["flat"] == "its pooled ROI, 0, is not positive"
    assert why["down"] == "its pooled ROI, -0.5, is not positive"


def test_on_a_geo_export_whose_tests_rose_a_channel_keeps_the_default(tmp_path):
    """Pilot world 4 of policy 0 at the top rung: four of its twelve tests rose, and social's pool
    falls below nought, which the arm once built into a NaN prior; at rung 0 no channel has a
    test."""
    world = GEO.world("policy-0", GEO.pilots["policy-0"][4])
    for k, expected in ((4, ["social"]), (0, ["search", "social", "tv"])):
        with np.load(export(GEO.observe(world, k), tmp_path / str(k))) as saved:
            channels = [str(c) for c in saved["channels"]]
            pools = roi.pooled(saved, channels)
        why = roi.defaulted(pools)
        assert list(why) == expected
        if k == 4:
            assert pools["search"]["roi"] > 0.0 and pools["search"]["rows"] == 4
            assert pools["tv"]["roi"] > 0.0 and pools["social"]["roi"] < 0.0

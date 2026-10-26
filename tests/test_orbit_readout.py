"""The Orbit arm's lift priors read off a geo export's market without Orbit:
``scripts/orbit_arm.py`` loaded beside a stand-in for ``orbit.models``, whose KTR no prior touches.
A geo test's readout is the market's spend less what its geos withheld, so its dark weeks keep the
untested geos' spend; a national test's readout is the channel dark, as before. The yearly
seasonality is fitted only where the hold-out fit's weeks hold a whole year."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.observe import export
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(patch, name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # the arm imports geo_export by name, and dataclasses resolve annotations through sys.modules
    patch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def arm():
    models = types.ModuleType("orbit.models")
    models.KTR = object  # ty: ignore[unresolved-attribute]
    package = types.ModuleType("orbit")
    package.models = models  # ty: ignore[unresolved-attribute]
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "orbit", package)
        patch.setitem(sys.modules, "orbit.models", models)
        _load(patch, "geo_export")
        yield _load(patch, "orbit_arm")


def _readouts(arm, path):
    """The national reading of the export at ``path``, its raw arrays, its priors, and the spend
    path each prior's readout reads, in the priors' order."""
    with np.load(path) as saved:
        raw = {key: saved[key] for key in saved.files}
    world = arm.national(raw)

    class Recorded(arm.Media):
        def z(self, spend):
            self.seen.append(np.array(spend))
            return super().z(spend)

    Recorded.seen = []
    media = Recorded(world, 0.5, 1.0)
    Recorded.seen = []
    priors = arm._priors(world, media, np.asarray(world["sales"]).size)
    assert len(Recorded.seen) == 2 * len(priors)
    return world, raw, priors, Recorded.seen[0::2]


def test_a_geo_tests_readout_keeps_the_untested_geos_spend(arm, tmp_path):
    world = GEO.world("policy-0.5", GEO.pilots["policy-0.5"][2])
    path = export(GEO.observe(world, 4), tmp_path)
    market, raw, priors, tested = _readouts(arm, path)
    assert len(priors) == 12
    names = [str(c) for c in raw["channels"]]
    dark = int(raw["lift_weeks"])
    for prior, readout in zip(priors, tested, strict=True):
        row = int(prior["name"].removeprefix("row_"))
        c = names.index(str(raw["lift_channel"][row]))
        first = int(raw["lift_start"][row]) - 1
        weeks = slice(first, first + dark)
        untested = ~raw["lift_geos"][row]
        assert 0 < untested.sum() < untested.size
        kept = raw["spend"][weeks][:, untested, c].sum(axis=1)
        np.testing.assert_allclose(readout[weeks, c], kept, rtol=1e-12, atol=0.0)
        assert np.all(kept > 0.0)
        untouched = np.array(readout)
        untouched[weeks, c] = market["spend"][weeks, c]
        np.testing.assert_array_equal(untouched, market["spend"][: readout.shape[0]])


def test_a_national_tests_readout_is_the_channel_dark(arm, tmp_path):
    path = export(TRACK_M2.observe(TRACK_M2.world("drawn", 905), 4), tmp_path)
    _, raw, priors, tested = _readouts(arm, path)
    assert priors
    names = [str(c) for c in raw["channels"]]
    dark = int(raw["lift_weeks"])
    for prior, readout in zip(priors, tested, strict=True):
        row = int(prior["name"].removeprefix("row_"))
        c = names.index(str(raw["lift_channel"][row]))
        first = int(raw["lift_start"][row]) - 1
        np.testing.assert_array_equal(readout[first : first + dark, c], 0.0)
        untouched = np.array(readout)
        untouched[first : first + dark, c] = raw["spend"][first : first + dark, c]
        np.testing.assert_array_equal(untouched, raw["spend"][: readout.shape[0]])


def test_a_row_whose_x_is_not_what_its_test_withheld_is_refused(arm, tmp_path):
    world = GEO.world("policy-0.5", GEO.pilots["policy-0.5"][2])
    path = export(GEO.observe(world, 4), tmp_path)
    with np.load(path) as saved:
        raw = {key: saved[key] for key in saved.files}
    market = arm.national(raw)
    media = arm.Media(market, 0.5, 1.0)
    market["lift_x"] = market["lift_x"] * (1.0 + 1e-6)
    with pytest.raises(ValueError, match="its test withheld"):
        arm._priors(market, media, np.asarray(market["sales"]).size)


@pytest.mark.parametrize(("weeks", "yearly"), [(52, False), (64, False), (65, True), (156, True)])
def test_the_yearly_seasonality_is_fitted_only_where_the_hold_out_fit_holds_a_year(
    arm, monkeypatch, weeks, yearly
):
    assert arm.seasonal(weeks) is yearly
    made = []

    class Recorded:
        def __init__(self, **kwargs):
            made.append(kwargs)

        def fit(self, frame):
            pass

    monkeypatch.setattr(arm, "KTR", Recorded)
    world = {"channels": np.array(["tv"]), "control_names": np.array(["price"]), "seed": 1}
    arm._fit(world, None, [], False, arm.seasonal(weeks))
    (kwargs,) = made
    assert kwargs["seasonality"] == ([arm.YEAR] if yearly else None)
    assert kwargs["seasonality_fs_order"] == ([arm.HARMONICS] if yearly else None)

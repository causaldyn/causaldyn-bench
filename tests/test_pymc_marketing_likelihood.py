"""The likelihood the PyMC-Marketing arm asks for, read without PyMC-Marketing:
``scripts/pymc_marketing_arm.py`` loaded beside stand-ins for the tools it imports, and its model
replaced by one that records the lift rows it is handed and stops the fit there. A geo export's
rows, apportioned across the tested geos, are read on a Normal likelihood, where their information
is the one test's; a national export's keep the tool's default."""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.observe import export
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_TOOLS = (
    "arviz",
    "pymc",
    "pymc.dims",
    "pytensor",
    "xarray",
    "pymc_marketing",
    "pymc_marketing.hsgp_kwargs",
    "pymc_marketing.mmm",
    "pymc_marketing.mmm.constraints",
    "pymc_marketing.mmm.tvp",
)


_CALLS: list = []  # each call for lift rows: the rows and the keywords


class _Stop(Exception):
    """Raised once the lift rows are in: the fit after them needs the tool."""


class _Model:
    """A stand-in MMM that records each call for lift rows and stops there."""

    def __init__(self, **kwargs):
        pass

    def build_model(self, frame, sales):
        pass

    def add_lift_test_measurements(self, rows, **kwargs):
        _CALLS.append((rows, kwargs))
        raise _Stop


def _load(patch, name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    patch.setitem(sys.modules, name, module)  # the arm imports its helpers by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def arm():
    with pytest.MonkeyPatch.context() as patch:
        stand_ins = {name: mock.MagicMock(name=name) for name in _TOOLS}
        dims = stand_ins["pymc"].dims = stand_ins["pymc.dims"]
        for name, stand_in in stand_ins.items():
            patch.setitem(sys.modules, name, stand_in)
        _load(patch, "geo_export")
        lift = _load(patch, "pymc_marketing_lift")
        module = _load(patch, "pymc_marketing_arm")
        patch.setattr(module, "MMM", _Model)
        yield module, lift, dims


def _handed(call):
    _CALLS.clear()
    with pytest.raises(_Stop):
        call()
    ((rows, kwargs),) = _CALLS
    return rows, kwargs


def test_a_geo_exports_apportioned_rows_are_read_on_a_normal_likelihood(arm, tmp_path):
    module, lift, dims = arm
    seed = GEO.pilots["policy-0"][3]
    path = export(GEO.observe(GEO.world("policy-0", seed), 4), tmp_path)
    rows, kwargs = _handed(lambda: module.plan_geo(path, smoke=False))
    assert kwargs == {"dist": dims.Normal}
    assert kwargs["dist"] is not dims.Gamma
    with np.load(path) as world:
        split, keep = lift.apportioned(world)
    assert len(rows) == int(keep.sum()) == 40
    expected = pd.DataFrame(
        {
            "channel": split["channel"][keep],
            "geo": [f"geo_{int(g):02d}" for g in split["geo"][keep]],
            **{key: split[key][keep] for key in ("x", "delta_x", "delta_y", "sigma")},
        }
    )
    pd.testing.assert_frame_equal(rows.reset_index(drop=True), expected, check_exact=True)


def test_a_national_exports_rows_keep_the_tools_default_likelihood(arm, tmp_path):
    module, lift, _ = arm
    path = export(TRACK_M2.observe(TRACK_M2.world("drawn", 900), 4), tmp_path)
    rows, kwargs = _handed(lambda: module.plan_second(path, "static", smoke=False))
    assert kwargs == {}
    with np.load(path) as world:
        keep, _ = lift.taken(world)
        assert len(rows) == int(keep.sum()) == 11
        np.testing.assert_array_equal(rows["delta_y"], world["lift_delta_y"][keep])

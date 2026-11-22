"""The sampler the PyMC-Marketing arm asks for, read without PyMC-Marketing:
``scripts/pymc_marketing_arm.py`` loaded beside stand-ins for the tools it imports, and its model
replaced by one that records the keywords its fit is handed and stops there. A full fit names its
4 chains, since PyMC's own default follows the CPUs the machine counts; a smoke names its own."""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

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

_FITS: list = []  # each fit's keywords


class _Stop(Exception):
    """Raised once the fit is asked for: what follows it needs the tool."""


class _Model:
    """A stand-in MMM that records each fit's keywords and stops there."""

    def __init__(self, **kwargs):
        pass

    def build_model(self, frame, sales):
        pass

    def add_lift_test_measurements(self, rows, **kwargs):
        pass

    def fit(self, frame, sales, **kwargs):
        _FITS.append(kwargs)
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
        stand_ins["pymc"].dims = stand_ins["pymc.dims"]
        for name, stand_in in stand_ins.items():
            patch.setitem(sys.modules, name, stand_in)
        _load(patch, "arm_machine")
        _load(patch, "geo_export")
        _load(patch, "pymc_marketing_lift")
        module = _load(patch, "pymc_marketing_arm")
        patch.setattr(module, "MMM", _Model)
        yield module


def _fitted(call):
    _FITS.clear()
    with pytest.raises(_Stop):
        call()
    (kwargs,) = _FITS
    return kwargs


@pytest.mark.parametrize("tested", [0, 4])
def test_a_national_fit_names_four_chains_at_pymcs_default_draws(arm, tmp_path, tested):
    path = export(TRACK_M2.observe(TRACK_M2.world("drawn", 900), tested), tmp_path)
    kwargs = _fitted(lambda: arm.plan_second(path, "static", smoke=False))
    assert kwargs == {"random_seed": 900, "progressbar": False, "chains": 4}


def test_a_geo_fit_names_four_chains(arm, tmp_path):
    seed = GEO.pilots["policy-0"][3]
    path = export(GEO.observe(GEO.world("policy-0", seed), 0), tmp_path)
    kwargs = _fitted(lambda: arm.plan_geo(path, smoke=False))
    assert kwargs == {"random_seed": seed, "progressbar": False, "chains": 4}


def test_a_smoke_keeps_its_own_reduced_sampler(arm, tmp_path):
    path = export(TRACK_M2.observe(TRACK_M2.world("drawn", 900), 0), tmp_path)
    kwargs = _fitted(lambda: arm.plan_second(path, "static", smoke=True))
    assert kwargs == {"random_seed": 900, "progressbar": False, **arm.SMOKE}
    assert arm.SMOKE["chains"] == 2

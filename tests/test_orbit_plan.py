"""The Orbit arm's plan search without Orbit: ``scripts/orbit_arm.py`` loaded beside a stand-in for
``orbit.models``, and its ``_plan`` handed a posterior mean whose best plan is known in closed
form. A world of two channels (the twins' family) is planned on a line of the box, one of three on
its plane; more channels are refused by name."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(patch, name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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
        _load(patch, "arm_machine")
        _load(patch, "geo_export")
        yield _load(patch, "orbit_arm")


class _LogWorth:
    """Sales Σ a_c log(1 + w_c / s_c): at weekly budget W the best plan inside a box that does not
    bind is w_c = a_c (W + Σ s) / Σ a - s_c, where every channel's margin a_c / (s_c + w_c) is
    equal."""

    def __init__(self, a, s):
        self.a, self.s = np.asarray(a, dtype=float), np.asarray(s, dtype=float)

    def mean(self, weekly):
        return (self.a * np.log1p(np.asarray(weekly, dtype=float) / self.s)).sum(axis=-1)

    def best(self, weekly):
        return self.a * (weekly + self.s.sum()) / self.a.sum() - self.s


def _world(lower, upper, weekly, planned=13):
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    return {
        "lower": lower,
        "upper": upper,
        "budget": weekly * planned,
        "planned": planned,
        "status_quo": np.full(lower.size, weekly / lower.size),
    }


@pytest.mark.parametrize(
    ("a", "s"),
    [((1.0, 2.0), (30.0, 50.0)), ((3.0, 1.0, 2.0), (20.0, 10.0, 30.0))],
    ids=["two channels", "three channels"],
)
def test_plan_finds_the_closed_form_best(arm, a, s):
    worth = _LogWorth(a, s)
    weekly = 100.0
    world = _world(np.zeros(len(a)), np.full(len(a), weekly), weekly)
    plan, success = arm._plan(worth, world)
    assert success
    assert plan.shape == (len(a),)
    assert float(plan.sum()) == pytest.approx(weekly, rel=1e-9, abs=0.0)
    assert plan == pytest.approx(worth.best(weekly), rel=0.0, abs=1e-6 * weekly)


def test_plan_refuses_four_channels(arm):
    world = _world(np.zeros(4), np.full(4, 100.0), 100.0)
    with pytest.raises(ValueError, match="two or three channels, not 4"):
        arm._plan(_LogWorth((1.0,) * 4, (10.0,) * 4), world)

"""Track M v2, lift tests: the harness, checked against the world it reads.

The track scores intervals against the generator's parameters, so the first check is that the
tests it cuts from a geo experiment carry exactly the gap the world made, and that the fit reads
his parameters back off it when the noise is gone.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from chc.lift import fit_lift

from causaldyn_bench.endogenous_mmm import EndogenousMediaMix
from causaldyn_bench.lift_calibration import (
    PARAMETERS,
    SETTINGS,
    Reading,
    Setting,
    _markdown,
    _record,
    lift_tests,
    run_setting,
    score,
    template,
    truth,
)


def test_the_tests_carry_the_worlds_gap_and_the_fit_reads_his_parameters_off_it():
    world = EndogenousMediaMix().simulate(7)
    quiet = Setting("pla", noise_share=1e-12)
    tests = lift_tests(world, quiet, seed=1)
    experiment = world.geo_test("pla", quiet.starts, noise_share=1e-12, seed=1)
    for start, test in zip(quiet.starts, tests, strict=True):
        window = experiment.readout(start)
        np.testing.assert_allclose(test.difference, experiment.true_gap[window], atol=1e-7)
    fit = fit_lift(tests, template(tests, world.kernel_length))
    expected = truth(EndogenousMediaMix(), "pla")
    assert expected == {"kernel.retention": 0.2, "curve.scale": 250.0, "coefficient": 1100.0}
    np.testing.assert_allclose(fit.estimate, [expected[name] for name in PARAMETERS], rtol=1e-4)


def _reading(lower: float, upper: float) -> Reading:
    ends = {name: 0.0 for name in PARAMETERS}
    return Reading(
        seed=0,
        estimate=ends,
        lower={**ends, "kernel.retention": lower},
        upper={**ends, "kernel.retention": upper},
        noise_sd=1.0,
        tested_adstock=(0.0, 1.0),
    )


def test_an_interval_covers_on_its_ends_and_closes_only_inside_the_box():
    readings = [_reading(0.1, 0.3), _reading(0.0, 0.5), _reading(0.25, 0.4), _reading(0.2, 1.0)]
    scored = score(readings, "kernel.retention", 0.2)
    assert scored.coverage == 0.75
    assert scored.closed == 0.5
    assert scored.median_width == pytest.approx(0.175)


def test_a_fit_that_raised_covers_nothing_and_leaves_the_median_estimate_alone():
    readings = [_reading(0.1, 0.3), Reading.failed(1, "did not converge")]
    scored = score(readings, "kernel.retention", 0.2)
    assert scored.coverage == 0.5
    assert scored.closed == 0.5
    assert scored.median_estimate == 0.0


def test_a_small_run_scores_every_parameter_and_writes_a_strict_report():
    with ThreadPoolExecutor(1) as pool:
        run = run_setting(0, 2, pool)
    assert run.gated
    assert run.truth == truth(EndogenousMediaMix(), SETTINGS[0].channel)
    assert set(run.scores) == set(PARAMETERS)
    assert [r.seed for r in run.readings] == [0, 1]
    text = _markdown([run], 2, x64=True)
    assert "**Gate** (pla, 4 tests, noise 1%" in text
    record = json.loads(json.dumps(_record([run], 2, x64=True), allow_nan=False))
    (setting,) = record["settings"]
    upper = setting["readings"]["upper"]["curve.scale"]
    fitted = [r.upper["curve.scale"] for r in run.readings]
    assert [value is None for value in upper] == [math.isinf(value) for value in fitted]
    assert setting["readings"]["failure"] == [None, None]

"""Track M v2's observational fit, checked where its answer is known: sales the model writes
exactly, from a history's own spend and controls."""

import dataclasses

import jax
import numpy as np
import pytest
from chc.response import GeometricAdstock

from causaldyn_bench.endogenous_mmm import EndogenousMediaMix
from causaldyn_bench.observational_mmm import (
    HARMONICS,
    Observed,
    _adstock,
    channel,
    fit_observational,
)

CONTROLS = np.array([5000.0, 300.0, -800.0, 120.0, -60.0, 30.0, 20.0, -10.0, 5.0])


@pytest.fixture(scope="module")
def observed():
    return Observed.of(EndogenousMediaMix().simulate(1))


def _channel(parameters):
    retention, scale, coefficient = parameters
    return channel(retention, scale, coefficient, 6)


def _written(observed, truth):
    """Sales the model writes exactly: the controls' part and every channel's return."""
    media = sum(
        np.asarray(_channel(parameters)(observed.spend[:, c])) for c, parameters in enumerate(truth)
    )
    return dataclasses.replace(observed, sales=observed.controls() @ CONTROLS + media)


def _truth(observed):
    largest = observed.spend.max(axis=0)
    return [
        (0.3, 0.8 * largest[0], 900.0),
        (0.5, 0.5 * largest[1], 400.0),
        (0.6, 1.2 * largest[2], 700.0),
    ]


def _read(fitted):
    return [(float(c.kernel.retention), float(c.curve.scale), float(c.coefficient)) for c in fitted]


@pytest.fixture
def x64():
    """The channels run in JAX; the recovery below is a float64 claim."""
    with jax.enable_x64(True):
        yield


def test_the_fit_reads_back_the_channels_that_wrote_the_sales(observed, x64):
    truth = _truth(observed)
    fitted = fit_observational(_written(observed, truth))
    np.testing.assert_allclose(_read(fitted), truth, rtol=1e-5)


def test_a_known_channel_is_returned_as_given_and_the_rest_read_given_it(observed, x64):
    truth = _truth(observed)
    known = _channel(truth[1])
    fitted = fit_observational(_written(observed, truth), known={observed.channels[1]: known})
    assert fitted[1] is known
    np.testing.assert_allclose(_read(fitted[::2]), truth[::2], rtol=1e-5)


def test_the_adstock_is_the_librarys_normalised_geometric_kernel(observed):
    spend = observed.spend[:, 0]
    library = np.asarray(GeometricAdstock(0.4, length=6, normalized=True)(spend))
    np.testing.assert_allclose(_adstock(spend, 0.4, 6), library, rtol=1e-6)


def test_the_controls_are_an_intercept_the_promotion_the_price_and_the_harmonics(observed):
    controls = observed.controls()
    assert controls.shape == (observed.week.size, 3 + 2 * HARMONICS)
    np.testing.assert_array_equal(controls[:, 0], 1.0)
    np.testing.assert_array_equal(controls[:, 1], observed.promotion.astype(float))
    np.testing.assert_array_equal(controls[:, 2], observed.price)

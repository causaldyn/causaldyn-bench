"""An arm's response mapped onto chc's: a mapping that is the tool's own reproduces the tool's
decomposition, written here as each tool writes it, every slip a mapping can make fails the check,
and an unmapped response is never read as a mapped one."""

import dataclasses

import jax
import numpy as np
import pytest

from causaldyn_bench.scorecard.mapping import DRAWS, TOLERANCE, channels, residual
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

RETENTION = (0.3, 0.55, 0.8)
LAM = (1.2, 2.5, 4.0)  # PyMC-Marketing's logistic saturation, on spend over the channel's max
SLOPE = (0.8, 1.6, 2.9)  # Robyn's Hill alpha
GAMMA = (0.3, 0.5, 0.9)  # Robyn's Hill gamma: the inflexion over the adstock's largest week
BETA = (0.4, 0.7, 1.1)
HELD = (0, 2)


@pytest.fixture(scope="module")
def observed():
    return TRACK_M2.observe(TRACK_M2.world("drawn", 905), 0)


def _pymc(spend, retention, lam, beta, length, target):
    """PyMC-Marketing's channel contribution: geometric adstock normalised over ``l_max`` lags on
    spend over its max, logistic saturation, ``beta`` times the target's max."""
    x = spend / np.max(spend)
    weights = retention ** np.arange(length)
    weights /= weights.sum()
    adstock = np.array(
        [sum(weights[lag] * x[t - lag] for lag in range(min(length, t + 1))) for t in range(x.size)]
    )
    return beta * target * (1 - np.exp(-lam * adstock)) / (1 + np.exp(-lam * adstock))


def _decayed(spend, theta):
    """Robyn's geometric adstock: a recursion over every week of the series."""
    decayed = spend.copy()
    for t in range(1, spend.size):
        decayed[t] = spend[t] + theta * decayed[t - 1]
    return decayed


def _inflexion(spend, theta, gamma):
    return gamma * _decayed(spend, theta).max()


def _robyn(spend, theta, alpha, gamma, coefficient):
    """Robyn's decomposition: Hill on the adstock with its inflexion at ``gamma`` times the
    adstock's largest week, times the ridge coefficient."""
    decayed, inflexion = _decayed(spend, theta), _inflexion(spend, theta, gamma)
    return coefficient * decayed**alpha / (decayed**alpha + inflexion**alpha)


def _record(observed, tool):
    """The ``response`` an arm of ``tool`` writes: three draws, each channel's shifted along the
    parameter lists, and the tool's own decomposition beside them."""
    names, spend = observed.channels, observed.spend
    length, target = observed.kernel_length, float(np.max(np.abs(observed.sales)))
    parameters = {name: {} for name in names}
    total = {name: [] for name in names}
    weekly = {name: [] for name in names}
    for c, name in enumerate(names):
        for i in range(3):
            j = (i + c) % 3
            if tool == "pymc":
                values = {
                    "retention": RETENTION[j],
                    "scale": 2 * float(np.max(spend[:, c])) / LAM[j],
                    "coefficient": BETA[j] * target,
                }
                decomposed = _pymc(spend[:, c], RETENTION[j], LAM[j], BETA[j], length, target)
            else:
                values = {
                    "retention": RETENTION[j],
                    "scale": _inflexion(spend[:, c], RETENTION[j], GAMMA[j]),
                    "slope": SLOPE[j],
                    "coefficient": BETA[j] * target,
                }
                decomposed = _robyn(spend[:, c], RETENTION[j], SLOPE[j], GAMMA[j], BETA[j] * target)
            for key, value in values.items():
                parameters[name].setdefault(key, []).append(value)
            total[name].append(float(decomposed.sum()))
            if i in HELD:
                weekly[name].append(decomposed.tolist())
    return {
        "kernel": "GeometricAdstock",
        "length": length if tool == "pymc" else spend.shape[0],
        "normalized": tool == "pymc",
        "curve": "Tanh" if tool == "pymc" else "Hill",
        "draws": 3,
        "parameters": parameters,
        "decomposition": {"total": total, "held": list(HELD), "weekly": weekly},
    }


@pytest.mark.parametrize("tool", ["pymc", "robyn"])
def test_a_mapping_that_is_the_tools_own_reproduces_its_decomposition(observed, tool):
    response = _record(observed, tool)
    with jax.enable_x64(True):
        drawn = channels(response, observed.channels)
        assert len(drawn) == 3 and all(len(draw) == len(observed.channels) for draw in drawn)
        assert residual(response, observed) <= TOLERANCE


def _longer(response):
    return {**response, "length": response["length"] + 1}


def _shorter(response):
    return {**response, "length": response["length"] - 1}


def _unnormalized(response):
    return {**response, "normalized": not response["normalized"]}


def _parameter(response, key, move):
    parameters = {
        name: {**columns, key: [move(value) for value in columns[key]]}
        for name, columns in response["parameters"].items()
    }
    return {**response, "parameters": parameters}


def _halved(response):
    return _parameter(response, "scale", lambda scale: scale / 2)


def _nudged(response):
    """Every coefficient a hundred-thousandth off: ten times the least slip the check catches."""
    return _parameter(response, "coefficient", lambda coefficient: coefficient * (1 + 1e-5))


def _swapped(response):
    first, second = list(response["parameters"])[:2]
    parameters = {
        **response["parameters"],
        first: response["parameters"][second],
        second: response["parameters"][first],
    }
    return {**response, "parameters": parameters}


@pytest.mark.parametrize(
    ("tool", "slip"),
    [
        ("pymc", _longer),
        ("pymc", _shorter),
        ("pymc", _unnormalized),
        ("pymc", _halved),
        ("pymc", _nudged),
        ("pymc", _swapped),
        # Robyn's kernel runs over every week, so a lag more or less never reaches the history
        ("robyn", _unnormalized),
        ("robyn", _halved),
        ("robyn", _nudged),
        ("robyn", _swapped),
    ],
)
def test_every_slip_a_mapping_can_make_fails_the_check(observed, tool, slip):
    response = slip(_record(observed, tool))
    with jax.enable_x64(True):
        assert residual(response, observed) > TOLERANCE


def test_a_slip_in_one_held_week_of_one_draw_fails_the_check(observed):
    response = _record(observed, "pymc")
    series = response["decomposition"]["weekly"][observed.channels[-1]][-1]
    series[int(np.argmax(np.abs(series)))] *= 1 + 1e-5
    with jax.enable_x64(True):
        assert residual(response, observed) > TOLERANCE


def test_a_slip_in_one_draws_total_fails_the_check(observed):
    response = _record(observed, "robyn")
    response["decomposition"]["total"][observed.channels[0]][1] *= 1 + 1e-5  # weeks not held
    with jax.enable_x64(True):
        assert residual(response, observed) > TOLERANCE


def test_a_curve_is_read_with_its_own_parameters_and_no_others(observed):
    tanh, hill = _record(observed, "pymc"), _record(observed, "robyn")
    with jax.enable_x64(True):
        with pytest.raises(ValueError, match="a Hill channel takes"):
            channels({**tanh, "curve": "Hill"}, observed.channels)
        with pytest.raises(ValueError, match="a Tanh channel takes"):
            channels({**hill, "curve": "Tanh"}, observed.channels)
        with pytest.raises(ValueError, match="no curve 'Logistic'"):
            channels({**tanh, "curve": "Logistic"}, observed.channels)
        with pytest.raises(ValueError, match="no kernel 'WeibullAdstock'"):
            channels({**tanh, "kernel": "WeibullAdstock"}, observed.channels)


def test_an_unmapped_response_is_never_read(observed):
    with jax.enable_x64(True), pytest.raises(ValueError, match="unmapped: a time-varying"):
        residual({"unmapped": "a time-varying multiplier"}, observed)


def test_the_check_runs_at_float64_alone(observed):
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="float64"):
        residual(_record(observed, "pymc"), observed)


def test_a_decomposition_short_of_a_draw_or_a_week_is_refused(observed):
    response = _record(observed, "pymc")
    decomposition, name = response["decomposition"], observed.channels[0]
    total = {**decomposition["total"], name: decomposition["total"][name][:-1]}
    weekly = {**decomposition["weekly"], name: [w[:-1] for w in decomposition["weekly"][name]]}
    with jax.enable_x64(True):
        for short in ({**decomposition, "total": total}, {**decomposition, "weekly": weekly}):
            with pytest.raises(ValueError, match="decomposition is"):
                residual({**response, "decomposition": short}, observed)


def test_draws_are_one_count_for_every_parameter_and_at_most_the_cap(observed):
    response = _record(observed, "pymc")
    name = observed.channels[0]
    columns = response["parameters"][name]
    ragged = {**response["parameters"], name: {**columns, "retention": columns["retention"][:-1]}}
    many = {
        channel: {key: column * (DRAWS // 3 + 1) for key, column in columns.items()}
        for channel, columns in response["parameters"].items()
    }
    with jax.enable_x64(True):
        for parameters in (ragged, many):
            with pytest.raises(ValueError, match="one value a draw"):
                channels({**response, "parameters": parameters}, observed.channels)


def test_a_response_maps_every_channel_and_no_other(observed):
    response = _record(observed, "pymc")
    first = observed.channels[0]
    fewer = {name: columns for name, columns in response["parameters"].items() if name != first}
    with jax.enable_x64(True), pytest.raises(ValueError, match="the response maps"):
        channels({**response, "parameters": fewer}, observed.channels)


def test_a_geo_observation_is_not_read(observed):
    geo = dataclasses.replace(
        observed,
        sales=observed.sales[:, None],
        spend=observed.spend[:, None, :],
        population=np.ones(1),
    )
    with jax.enable_x64(True), pytest.raises(ValueError, match="geo"):
        residual(_record(observed, "pymc"), geo)

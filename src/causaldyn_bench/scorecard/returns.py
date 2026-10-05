"""What each channel's spend in a window returns, for the world and for each draw of an arm's
mapped response (:mod:`.mapping`), as :func:`chc.response.roi` and
:func:`chc.response.marginal_roi` define it.

* The **ROI**: the channel's return over the history, less its return with the window's spend
  removed, over what the window spent.
* The **mROI**: the derivative of that return along the window's spend, over what the window spent,
  the return on one more euro spread over the window as it spent.

Both count all of the carryover: the series runs on past the history's last week for the weeks the
kernel carries its spend into, with nothing spent. A coefficient that moves over the history's
weeks, a drifted world's or a dynamic linear model's, is held past the last week at that week's, for
the world and every arm alike: the one convention that needs no forecast of the coefficient, and
changes nothing for one that holds still.

Written over every draw at once, where chc reads one channel at a time; its tests hold the two to
each other draw by draw.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray

from causaldyn_bench.endogenous_mmm import MediaMixWorld, Series
from causaldyn_bench.scorecard.mapping import drawn
from causaldyn_bench.scorecard.observe import Observation


@dataclass(frozen=True, eq=False)
class Returns:
    """Each channel's ROI and mROI on ``window``: ``(channels,)`` for a world, ``(draws,
    channels)`` for an arm's draws."""

    window: tuple[int, int]  # the first and the last week, numbered from 1, as an export names it
    roi: Series
    marginal: Series


def _lagged(series: Series, length: int, horizon: int) -> Series:
    """``(horizon, length)``: the series in each week at each lag, nothing outside it."""
    source = np.arange(horizon)[:, None] - np.arange(length)[None, :]
    inside = (source >= 0) & (source < series.size)
    return np.where(inside, series[np.clip(source, 0, series.size - 1)], 0.0)


def _window_returns(
    spend: Series,
    window: tuple[int, int],
    weights: Series,
    effect: Series,
    value: Callable[[Series], Series],
    marginal: Callable[[Series, Series], Series],
) -> tuple[Series, Series]:
    """Each draw's ROI and mROI of one channel: ``spend`` over the history's weeks, ``weights``
    each draw's kernel, ``(draws, length)`` from lag 0, and ``effect`` each draw's coefficient in
    each of the history's weeks. ``value`` is each draw's curve at its adstock, ``(draws, weeks +
    length - 1)``, and ``marginal`` the curve's slope at its first argument times its second.

    Raises:
        ValueError: the window spends nothing.
    """
    weeks, length = spend.size, weights.shape[1]
    horizon = weeks + length - 1
    first, last = window
    inside = np.zeros(weeks, dtype=bool)
    inside[first - 1 : last] = True
    spent = float(spend[inside].sum())
    if not spent > 0.0:
        raise ValueError(f"the window {window} spends {spent}; its return per euro is undefined")
    every = weights @ _lagged(spend, length, horizon).T
    without = weights @ _lagged(np.where(inside, 0.0, spend), length, horizon).T
    own = weights @ _lagged(np.where(inside, spend, 0.0), length, horizon).T
    held = np.concatenate([effect, np.repeat(effect[:, -1:], length - 1, axis=1)], axis=1)
    roi = np.sum(held * (value(every) - value(without)), axis=1) / spent
    return roi, np.sum(held * marginal(every, own), axis=1) / spent


def true_returns(history: MediaMixWorld, window: tuple[int, int]) -> Returns:
    """Each channel's ROI and mROI on ``window`` in ``history``: through its own kernel and curve,
    at its effect in each week."""
    effects = history.effect_path()
    roi, marginal = [], []
    for c, (curve, lam) in enumerate(zip(history.curves(), history.saturation, strict=True)):
        one, slope = _window_returns(
            history.spend[:, c],
            window,
            history.kernel(c)[None, :],
            effects[None, :, c],
            lambda a, curve=curve, lam=lam: curve.value(a, lam),
            lambda a, along, curve=curve, lam=lam: curve.slope(a, lam) * along,
        )
        roi.append(float(one[0]))
        marginal.append(float(slope[0]))
    return Returns(window, np.array(roi), np.array(marginal))


def _value(scale: Series, slope: Series, tanh: NDArray[np.bool_], adstock: Series) -> Series:
    """Each draw's curve, ``tanh(z)`` or ``z^n / (1 + z^n)`` at ``z`` its adstock over its scale."""
    z = adstock / scale[:, None]
    out = np.tanh(z)
    hill = ~tanh
    rising = z[hill] ** slope[hill, None]
    out[hill] = rising / (1.0 + rising)
    return out


def _marginal(
    scale: Series, slope: Series, tanh: NDArray[np.bool_], adstock: Series, along: Series
) -> Series:
    """Each draw's curve's slope at ``adstock`` times ``along``: ``sech^2(z) / scale``, or
    ``n z^n / (1 + z^n)^2 / adstock``, the second read where the adstock is positive, since
    ``along``, an adstock of part of the spend, vanishes wherever the whole one does."""
    z = adstock / scale[:, None]
    decay = np.exp(-2.0 * z)
    out = 4.0 * decay / (1.0 + decay) ** 2 / scale[:, None] * along
    hill = ~tanh
    rising = z[hill] ** slope[hill, None]
    share = np.divide(
        along[hill], adstock[hill], out=np.zeros_like(rising), where=adstock[hill] > 0
    )
    out[hill] = slope[hill, None] * rising / (1.0 + rising) ** 2 * share
    return out


def drawn_returns(response: Mapping[str, Any], observation: Observation) -> Returns:
    """Each draw's ROI and mROI of each channel on the observation's window, ``(draws,
    channels)``, from a response :func:`~.mapping.residual` has held to its tool.

    Raises:
        ValueError: as :func:`~.mapping.drawn` does.
    """
    parsed = drawn(response, observation.channels)
    length, normalized = int(response["length"]), bool(response["normalized"])
    weeks = observation.sales.shape[0]
    lags = np.arange(length)
    roi, marginal = [], []
    for c, name in enumerate(observation.channels):
        d = parsed[name]
        weights = d.retention[:, None] ** lags
        if normalized:
            weights = weights / weights.sum(axis=1, keepdims=True)
        effect = d.coefficient
        if effect.ndim == 1:
            effect = np.repeat(effect[:, None], weeks, axis=1)
        tanh = np.array([curve == "Tanh" for curve in d.curves])
        one, slope = _window_returns(
            observation.spend[:, c],
            observation.roi_window,
            weights,
            effect,
            partial(_value, d.scale, d.slope, tanh),
            partial(_marginal, d.scale, d.slope, tanh),
        )
        roi.append(one)
        marginal.append(slope)
    return Returns(observation.roi_window, np.stack(roi, axis=1), np.stack(marginal, axis=1))

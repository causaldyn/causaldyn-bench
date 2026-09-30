"""The media-mix model a practitioner fits to Track M v2's history: Heusch's "realistic"
specification, by least squares.

Weekly sales on an intercept, the observed promotion indicator, the observed price and three
annual Fourier harmonics, "the standard seasonal controls in Robyn, Meridian, and pymc-marketing"
(Heusch 2026a, Table II), plus each channel's return: the world's own normalised six-week
geometric kernel through ``tanh``, so every arm that reads this fit is handed the world's forms
and has only their parameters to learn. The coefficients enter linearly and are solved for at each
retention and scale (variable projection), the channels' kept at or above zero; the retentions,
boxed, and the scales, in logs, are searched from four starts.

A channel whose return is known, from an experiment, enters as an offset: its return over the
history is taken off the sales before the rest is fitted, so the others are read given it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from chc.response import Channel, GeometricAdstock, Tanh
from numpy.typing import NDArray
from scipy.optimize import least_squares, lsq_linear
from scipy.stats import gamma

from causaldyn_bench.endogenous_mmm import YEAR, MediaMixWorld, Series, Vector

HARMONICS = 3
RETENTION_BOX = (0.0, 0.95)
STARTS = ((0.3, 1.0), (0.6, 1.0), (0.3, 3.0), (0.6, 3.0))  # retention, scale over mean spend
# a scale's box, over its channel's largest weekly spend: where 98 % of the scale lies under
# PyMC-Marketing's default prior, lam ~ Gamma(3, 1) on spend over its largest value, K = 2 / lam
SCALE_BOX = tuple(2.0 / gamma(3.0).ppf(q) for q in (0.99, 0.01))
EVALUATIONS = 3000


@dataclass(frozen=True)
class Observed:
    """What an analyst sees of a history: sales, spend and the measurement layer's readings."""

    channels: tuple[str, ...]
    week: NDArray[np.int64]
    sales: Series
    spend: Series  # (weeks, channels)
    promotion: NDArray[np.bool_]
    price: Series

    @classmethod
    def of(cls, world: MediaMixWorld) -> Observed:
        return cls(
            channels=world.channels,
            week=world.week.copy(),
            sales=world.sales.copy(),
            spend=world.spend.copy(),
            promotion=world.observed_promotion.copy(),
            price=world.observed_price.copy(),
        )

    def controls(self) -> Series:
        """``(weeks, 3 + 2 * HARMONICS)``: intercept, promotion, price, then sine and cosine."""
        angle = 2.0 * math.pi * self.week / YEAR
        columns = [np.ones(self.week.size), self.promotion.astype(float), self.price]
        for k in range(1, HARMONICS + 1):
            columns += [np.sin(k * angle), np.cos(k * angle)]
        return np.column_stack(columns)


def _adstock(spend: Series, retention: float, length: int) -> Series:
    weights = retention ** np.arange(length)
    return np.convolve(spend, weights / weights.sum())[: spend.size]


def channel(retention: float, scale: float, coefficient: float, length: int) -> Channel:
    return Channel(
        GeometricAdstock(retention, length=length, normalized=True), Tanh(scale), coefficient
    )


def fit_observational(
    observed: Observed,
    *,
    length: int = 6,
    known: Mapping[str, Channel] | None = None,
) -> tuple[Channel, ...]:
    """Each channel's fitted return, in the observed channels' order; a ``known`` one as given.

    Raises:
        RuntimeError: no start converged within ``EVALUATIONS`` evaluations.
    """
    known = dict(known or {})
    unknown = [c for c, name in enumerate(observed.channels) if name not in known]
    sales = observed.sales.copy()
    for name, given in known.items():
        column = observed.channels.index(name)
        sales -= np.asarray(given(observed.spend[:, column]))
    controls = observed.controls()
    mean = observed.spend[:, unknown].mean(axis=0)
    count = len(unknown)
    floor = np.concatenate([np.full(controls.shape[1], -np.inf), np.zeros(count)])

    def design(z: Series) -> Series:
        retention: Vector = z[:count]
        scale: Vector = np.exp(z[count:])
        media = [
            np.tanh(_adstock(observed.spend[:, c], r, length) / k)
            for c, r, k in zip(unknown, retention, scale, strict=True)
        ]
        return np.column_stack([controls, *media])

    def coefficients(z: Series) -> Series:
        return lsq_linear(design(z), sales, bounds=(floor, np.inf), method="bvls").x

    def residual(z: Series) -> Series:
        return sales - design(z) @ coefficients(z)

    largest = observed.spend[:, unknown].max(axis=0)
    low = np.concatenate([np.full(count, RETENTION_BOX[0]), np.log(SCALE_BOX[0] * largest)])
    high = np.concatenate([np.full(count, RETENTION_BOX[1]), np.log(SCALE_BOX[1] * largest)])
    best = None
    for retention, over in STARTS:
        start = np.concatenate([np.full(count, retention), np.log(over * mean)])
        start = np.clip(start, low, high)
        solve = least_squares(
            residual, start, bounds=(low, high), x_scale="jac", max_nfev=EVALUATIONS
        )
        if solve.status > 0 and (best is None or solve.cost < best.cost):
            best = solve
    if best is None:
        raise RuntimeError(f"no start converged within {EVALUATIONS} evaluations")
    beta = coefficients(best.x)[controls.shape[1] :]
    fitted = {
        c: channel(float(r), float(math.exp(v)), float(b), length)
        for c, r, v, b in zip(unknown, best.x[:count], best.x[count:], beta, strict=True)
    }
    return tuple(
        known[name] if name in known else fitted[c] for c, name in enumerate(observed.channels)
    )

"""The shadow quarter: what a world goes on to do over the quarter after its history, its media held
at the status quo.

An arm forecasts the quarter's sales at the status quo, and is scored against these. The world's
processes run on from the history's last week as :mod:`causaldyn_bench.endogenous_mmm` runs them:
the seasonality and the promotions' calendar by the week; product quality, the price position and
the unobserved sentiment as the same AR(1) processes, each from its value at the history's end; the
baseline's noise; and the measurement layer, the promotion indicator read with its recall and false
positives and the price read with its autocorrelated error. Media's effect is each channel's at the
status quo's weekly spend, its carryover from the history included, on the effect path the caller
gives: the channel's constant coefficient on Track M v2, a drift law's realised path where the
effect drifts. Spend is the status quo's, so the budget and bidding rules, which set spend, do not
run.

The variates come from Track M v2's own stream, ``(100, 3, seed)`` (:mod:`.seeds`), whichever
family drew the world, so two families' worlds that share a history share its quarter.

CHOICE: the price reading's error runs on from the history's last reading less the true price
there, its error exactly unless that reading was clipped.

A geo world's quarter (:func:`geo_shadow`, a world :mod:`chc.geo_world` drew) runs on the same way
geo by geo: its base on the season every geo shares, every channel's lift at the status quo, each
national channel's allotted to the geos by population, its carryover from the history and its
spillover included, and fresh noise of the world's size a geo. Its variates come from the family's
own stream, ``(100 + f, 3, seed)``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import jax
import numpy as np
from chc.geo_world import GeoMediaWorld
from numpy.typing import ArrayLike, NDArray

from causaldyn_bench.endogenous_mmm import (
    ELASTICITY,
    YEAR,
    EndogenousMediaMix,
    MediaMixWorld,
    Series,
    _ar1,
    _calendar,
)
from causaldyn_bench.mmm_decision import PLANNED, Quarter
from causaldyn_bench.scorecard import truth
from causaldyn_bench.scorecard.seeds import Role, rng


@dataclass(frozen=True)
class Shadow:
    """The quarter's weeks, each series over them; ``media`` is ``(weeks, channels)``."""

    week: NDArray[np.int64]  # T + 1, ..., T + 13, numbered as the history's are
    seasonality: Series
    quality: Series
    price_position: Series
    sentiment: Series  # unobserved, as in the history
    baseline: Series
    discount: Series
    promotion: Series  # the promotional uplift
    premedia: Series
    media: Series  # (weeks, channels) each channel's effect at the status quo
    sales: Series  # the prediction target
    observed_promotion: NDArray[np.bool_]
    observed_price: Series


def shadow(
    generator: EndogenousMediaMix,
    world: MediaMixWorld,
    seed: int,
    effect: ArrayLike | None = None,
) -> Shadow:
    """The quarter after ``world``'s history, which ``generator`` drew from ``seed``, at the status
    quo: ``effect``, ``(channels, 13 + L - 1)``, is each channel's effect path over the quarter and
    the kernel's tail, the channel's coefficient where it is None."""
    draw = rng(0, Role.CONTINUATION, seed)
    weeks = world.week.size
    week = np.arange(weeks + 1, weeks + PLANNED + 1)
    angle = 2.0 * math.pi * week / YEAR
    seasonality = 0.15 * np.sin(angle) + 0.08 * np.sin(2 * angle) + 0.05 * np.sin(3 * angle)
    base = generator.base
    quality = _ar1(draw, PLANNED, 0.95, 0.0, 0.02, start=float(world.quality[-1]))
    price_position = _ar1(draw, PLANNED, 0.9, 0.1, 0.03, start=float(world.price_position[-1]))
    sentiment = _ar1(draw, PLANNED, 0.7, 0.0, 0.03, start=float(world.sentiment[-1]))
    noise = draw.normal(0.0, 0.02 * base, PLANNED)
    baseline = (
        base * (1 + seasonality) * (1 + quality) * price_position**ELASTICITY
        + base * sentiment
        + noise
    )
    baseline = np.maximum(baseline, 0.5 * base)
    discount, multiplier = _calendar(week)
    uplift = np.where(
        discount > 0.0,
        np.clip(np.maximum(1.0 - discount, 0.5) ** (ELASTICITY * multiplier), 0.8, 3.0),
        1.0,
    )
    promotion = baseline * (uplift - 1.0)
    premedia = baseline + promotion

    quarter = Quarter.after(world)
    cells = truth.cells(world, quarter, effect)
    media = np.column_stack(
        [
            cell.returns(float(w))[:PLANNED]
            for cell, w in zip(cells, quarter.status_quo, strict=True)
        ]
    )

    caught = draw.random(PLANNED)
    observed_promotion = np.where(discount > 0.0, caught < 0.7, caught < 0.15)
    last = float(world.observed_price[-1] - world.price_position[-1] * (1.0 - world.discount[-1]))
    error = _ar1(draw, PLANNED, 0.2, 0.0, 0.08, start=last)
    observed_price = np.clip(price_position * (1.0 - discount) + error, 0.5, 1.8)
    return Shadow(
        week=week,
        seasonality=seasonality,
        quality=quality,
        price_position=price_position,
        sentiment=sentiment,
        baseline=baseline,
        discount=discount,
        promotion=promotion,
        premedia=premedia,
        media=media,
        sales=premedia + media.sum(axis=1),
        observed_promotion=observed_promotion,
        observed_price=observed_price,
    )


@dataclass(frozen=True)
class GeoShadow:
    """A geo world's quarter at the status quo: each series ``(weeks, geos)``."""

    week: NDArray[np.int64]  # T + 1, ..., T + 13, numbered as the history's are
    base: Series
    media: Series  # every channel's lift at the status quo, at home and spilled in
    noise: Series
    sales: Series  # each geo's KPI, the sum of the three


def lifted(world: GeoMediaWorld, weekly: Series, weeks: int) -> Series:
    """``(geos, weeks)``: what the world's media lift in each geo over the ``weeks`` after its
    history, every geo spending ``weekly``, ``(geos, channels)``, in each of the quarter's weeks
    and nothing after them, the history's carryover included: :meth:`GeoMediaWorld.lift` run on,
    at double precision."""
    history = world.spend.shape[1]
    spent = np.zeros((world.mix.geos, weeks, len(world.mix.channels)))
    spent[:, :PLANNED] = np.asarray(weekly, dtype=float)[:, None, :]
    spend = np.concatenate([world.spend, spent], axis=1)
    drift = np.ones((spend.shape[1], spend.shape[2]))  # geo cells are planned without drift
    longer = dataclasses.replace(world, spend=spend, drift=drift)
    with jax.enable_x64(True):
        return np.asarray(longer.lift(spend), dtype=float)[:, history:]


def geo_shadow(world: GeoMediaWorld, family: int, seed: int) -> GeoShadow:
    """The quarter after ``world``'s history, which family ``family`` drew from ``seed``, at the
    status quo of its cells (:func:`causaldyn_bench.scorecard.truth.geo_quarter`).

    Raises:
        ValueError: the world's lifts drift.
    """
    draw = rng(family, Role.CONTINUATION, seed)
    history = world.spend.shape[1]
    week = np.arange(history + 1, history + PLANNED + 1)
    season = np.sin(2.0 * math.pi * (week - 1) / YEAR)  # chc counts its weeks from 0
    # each geo's base per head times its population is its base in week 0, where the season is 0
    base = world.base[:, :1] * (1.0 + world.mix.season * season)
    quarter = truth.geo_quarter(world)
    media = lifted(world, truth.allotted(world, quarter.status_quo), PLANNED)
    noise = world.mix.noise * np.sqrt(world.population)[:, None]
    noise = noise * draw.standard_normal((world.mix.geos, PLANNED))
    return GeoShadow(week, base.T, media.T, noise.T, (base + media + noise).T)

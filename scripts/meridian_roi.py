"""The Meridian arm's ROI calibration, read off an export's lift rows without Meridian.

A test's ROI is ``delta_y / delta_x``, sales lost per euro withheld, with standard error
``sigma / |delta_x|``; a channel's rows are pooled by inverse variance, and Meridian takes the
pooled mean and error as the channel's LogNormal ``roi_m`` prior. That LogNormal exists for a
positive mean alone, and an export carries every test whichever way its sales moved, so a channel's
tests can pool to a ROI of nought or below: that channel keeps Meridian's default prior, as one
with no test does, and the record names both and why. Kept apart from
:mod:`meridian_arm`, which imports Meridian, so the bench's tests read the rule in the bench's
environment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def pooled(world: Mapping[str, Any], channels: Sequence[str]) -> dict[str, dict[str, Any] | None]:
    """Each channel's lift rows pooled by inverse variance: their count, the pooled ROI and its
    standard error; None for a channel with no row."""
    out: dict[str, dict[str, Any] | None] = {}
    named = [str(c) for c in world["lift_channel"]]
    for name in channels:
        rows = np.array([c == name for c in named], dtype=bool)
        if not rows.any():
            out[name] = None
            continue
        delta_x = np.asarray(world["lift_delta_x"], dtype=float)[rows]
        roi = np.asarray(world["lift_delta_y"], dtype=float)[rows] / delta_x
        weight = (np.abs(delta_x) / np.asarray(world["lift_sigma"], dtype=float)[rows]) ** 2
        out[name] = {
            "rows": int(rows.sum()),
            "roi": float(np.sum(weight * roi) / np.sum(weight)),
            "error": float(1.0 / math.sqrt(np.sum(weight))),
        }
    return out


def defaulted(pools: Mapping[str, dict[str, Any] | None]) -> dict[str, str]:
    """The channels that keep Meridian's default ROI prior, each with why: one with no lift row,
    and one whose pooled ROI is nought or below, where a LogNormal of that mean has no parameters
    (``lognormal_dist_from_mean_std`` takes the mean's log)."""
    out = {}
    for name, pool in pools.items():
        if pool is None:
            out[name] = "no lift row"
        elif not pool["roi"] > 0.0:
            out[name] = f"its pooled ROI, {pool['roi']:.3g}, is not positive"
    return out

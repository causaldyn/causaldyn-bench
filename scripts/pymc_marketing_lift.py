"""The lift rows of an export that PyMC-Marketing 1.2.0's lift likelihood takes.

``add_lift_test_measurements`` refuses a row whose change in sales does not follow its change in
spend (``assert_monotonic``: ``delta_x * delta_y >= 0``), and its likelihood is a Gamma of the
change's size, with no density at nought; so it takes a row where ``delta_x * delta_y > 0``. An
export carries every test whichever way its sales moved, so the arm drops the rest itself and
counts them. Kept apart from :mod:`pymc_marketing_arm`, which imports PyMC-Marketing, so the
bench's tests read the rule in the bench's environment.

A geo model's likelihood reads each row at one coordinate of its geo axis (``exact_row_indices``):
a row is one geo's, its change in spend and in sales that geo's. A geo export's test takes ``m``
geos dark and reads their change in sales together, so the arm apportions it across them
(:func:`apportioned`). CHOICE, the arm's and not the tool's: one ROI across a test's geos, as the
test's own design assumes. Geo ``g``'s row withholds its own spend, ``delta_x_g``, the geo's mean
weekly spend over the dark weeks; it reads the share ``s_g = delta_x_g / sum delta_x`` of the
test's change in sales, and an error of ``s_g * sigma * sqrt(m)``. Read as independent, the ``m``
rows then hold the one test's information about a common return, ``delta_x**2 / sigma**2`` on a
Gaussian likelihood, and give its estimate, ``delta_y / delta_x``: the rows sum back to the test.
Each row takes its test's sign, so a test whose sales did not fall drops its every row, counted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def taken(world: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    """Which of ``world``'s lift rows the likelihood takes, a mask over them, and how many of the
    rows the export holds it does not take."""
    keep = (
        np.asarray(world["lift_delta_x"], dtype=float)
        * np.asarray(world["lift_delta_y"], dtype=float)
        > 0.0
    )
    return keep, int(keep.size - np.count_nonzero(keep))


def apportioned(world: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """A geo export's tests, each apportioned across its tested geos at one ROI: one row a test and
    tested geo, ``test`` its test's index, ``geo`` the geo's, ``channel``, ``x`` the geo's mean
    weekly spend over the dark weeks, ``delta_x`` its negative, and ``delta_y`` and ``sigma`` the
    test's apportioned; and which of the rows the likelihood takes, a mask over them."""
    names = [str(c) for c in world["channels"]]
    spend = np.asarray(world["spend"], dtype=float)  # (weeks, geos, channels)
    weeks = int(world["lift_weeks"])
    tested = np.asarray(world["lift_geos"], dtype=bool)
    parts: dict[str, list[Any]] = {
        key: [] for key in ("test", "geo", "channel", "x", "delta_x", "delta_y", "sigma")
    }
    for i, name in enumerate(str(c) for c in world["lift_channel"]):
        first = int(world["lift_start"][i]) - 1
        geos = np.flatnonzero(tested[i])
        x = spend[first : first + weeks, geos, names.index(name)].mean(axis=0)
        delta_x = -x
        share = delta_x / delta_x.sum()
        parts["test"].append(np.full(geos.size, i))
        parts["geo"].append(geos)
        parts["channel"].append(np.full(geos.size, name))
        parts["x"].append(x)
        parts["delta_x"].append(delta_x)
        parts["delta_y"].append(share * float(world["lift_delta_y"][i]))
        parts["sigma"].append(share * float(world["lift_sigma"][i]) * math.sqrt(geos.size))
    rows = {key: np.concatenate(value) if value else np.empty(0) for key, value in parts.items()}
    return rows, rows["delta_x"] * rows["delta_y"] > 0.0

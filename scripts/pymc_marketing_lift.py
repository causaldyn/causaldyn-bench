"""The lift rows of an export that PyMC-Marketing 1.3.0's lift likelihood takes.

``add_lift_test_measurements`` reads each row signed on its default likelihood, ``Prior("Normal")``
(``pymc_marketing/mmm/lift_test.py`` line 196): the model's change in the curve is the mean and the
row's ``delta_y`` the observation, at the row's ``sigma`` (lines 611-619). It no longer calls
``assert_monotonic`` (lines 359-371, deprecated): a test whose sales rose, or did not move, is
taken as it is. It refuses a row only whose ``x``, ``delta_x``, ``delta_y`` or ``sigma`` is not
finite, or whose ``sigma`` is not positive (lines 64-92, read at line 569); so the arm takes the
rest and counts what it refuses, none in an export the bench writes. Kept apart from
:mod:`pymc_marketing_arm`, which imports PyMC-Marketing, so the bench's tests read the rule in the
bench's environment.

A geo model's likelihood reads each row at one coordinate of its geo axis (``exact_row_indices``,
line 232, as in 1.2.0): a row is one geo's, its change in spend and in sales that geo's. A geo
export's test takes ``m`` geos dark and reads their change in sales together, so the arm apportions
it across them (:func:`apportioned`). CHOICE, the arm's and not the tool's: one ROI across a
test's geos, as the test's own design assumes. Geo ``g``'s row withholds its own spend,
``delta_x_g``, the geo's mean weekly spend over the dark weeks; it reads the share ``s_g = delta_x_g
/ sum delta_x`` of the test's change in sales, and an error of ``s_g * sigma * sqrt(m)``. Read as
independent, the ``m`` rows then hold the one test's information about a common return,
``delta_x**2 / sigma**2`` on a Gaussian likelihood, and give its estimate, ``delta_y / delta_x``:
the rows sum back to the test. The tool's default likelihood is that Gaussian, so the identity is
exact at the default. A geo the test took dark that spent nothing on the channel holds no share and
no error, and is refused.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def _valid(x: Any, delta_x: Any, delta_y: Any, sigma: Any) -> np.ndarray:
    """Which rows the likelihood takes: every value finite and the error positive."""
    columns = [np.asarray(column, dtype=float) for column in (x, delta_x, delta_y, sigma)]
    return np.logical_and.reduce([np.isfinite(column) for column in columns]) & (columns[3] > 0.0)


def taken(world: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    """Which of ``world``'s lift rows the likelihood takes, a mask over them, and how many of the
    rows the export holds it does not take."""
    keep = _valid(*(world[f"lift_{key}"] for key in ("x", "delta_x", "delta_y", "sigma")))
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
        taken = np.count_nonzero(x > 0.0)
        parts["sigma"].append(share * float(world["lift_sigma"][i]) * math.sqrt(taken))
    rows = {key: np.concatenate(value) if value else np.empty(0) for key, value in parts.items()}
    return rows, _valid(rows["x"], rows["delta_x"], rows["delta_y"], rows["sigma"])

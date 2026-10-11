"""The lift rows of an export that PyMC-Marketing 1.2.0's lift likelihood takes.

``add_lift_test_measurements`` refuses a row whose change in sales does not follow its change in
spend (``assert_monotonic``: ``delta_x * delta_y >= 0``), and its likelihood is a Gamma of the
change's size, with no density at nought; so it takes a row where ``delta_x * delta_y > 0``. An
export carries every test whichever way its sales moved, so the arm drops the rest itself and
counts them. Kept apart from :mod:`pymc_marketing_arm`, which imports PyMC-Marketing, so the
bench's tests read the rule in the bench's environment.

A geo model's likelihood reads each row at one coordinate of its geo axis (``exact_row_indices``):
a test is one geo's, its change in spend and in sales that geo's. A geo export's test takes a set of
geos dark and reads their change in sales together, which no row of one geo holds, so on a geo
export the arm takes a test of one geo alone, and counts the rest by why (:func:`taken_geo`).
"""

from __future__ import annotations

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


SEVERAL = "a test of several geos, whose change in sales no row of one geo holds"
AGAINST = "a change in sales that does not follow its change in spend"


def taken_geo(world: Mapping[str, Any]) -> tuple[np.ndarray, int, dict[str, int]]:
    """Which of a geo export's lift rows the likelihood takes, a mask over them: one geo's, whose
    change in sales follows its change in spend; how many it does not take; and how many of those
    for each reason, a test of several geos counted as that whatever its sales did."""
    follows, _ = taken(world)
    one = np.asarray(world["lift_geos"], dtype=bool).sum(axis=1) == 1
    keep = follows & one
    why = {SEVERAL: int(np.count_nonzero(~one)), AGAINST: int(np.count_nonzero(one & ~follows))}
    return keep, int(keep.size - np.count_nonzero(keep)), why

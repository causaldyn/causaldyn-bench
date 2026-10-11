"""The lift rows of an export that PyMC-Marketing 1.2.0's lift likelihood takes.

``add_lift_test_measurements`` refuses a row whose change in sales does not follow its change in
spend (``assert_monotonic``: ``delta_x * delta_y >= 0``), and its likelihood is a Gamma of the
change's size, with no density at nought; so it takes a row where ``delta_x * delta_y > 0``. An
export carries every test whichever way its sales moved, so the arm drops the rest itself and
counts them. Kept apart from :mod:`pymc_marketing_arm`, which imports PyMC-Marketing, so the
bench's tests read the rule in the bench's environment.
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

"""What the tools' arms read of a geo export (family 8), without any tool.

A geo export (:mod:`causaldyn_bench.scorecard.observe`) holds each geo's KPI and spend, each geo's
share of the population, the channels bought nationally, and a quarter over its cells: each geo's
own channels, geo by geo in the channels' order, then each national channel once, its spend summed
over the geos. Its tests each take a set of geos dark. This module reads it the ways the arms need:

* :func:`channel_quarter`: the quarter a channel at a time, each cell's box and status quo summed
  over the channel's cells; a plan of one weekly spend a channel is what the harness allots to the
  cells (``GeoQuarter.allot``), and a channel's box is its cells' summed, since each cell's box is
  the same multiple of its own status quo;
* :func:`panel` and :func:`cells`: a weekly spend on each cell as each geo spends it, a national
  channel's allotted to the geos by population as the world allots it, and back;
* :func:`market`: the KPI and the spend summed over the geos, the series a national tool fits;
* :func:`withheld`: each week's spend a test withheld over its dark weeks, its geos' summed;
* :func:`market_lift`: a test's lift and spend as a tool that reads a lift as the market's whole
  return over its dark weeks takes them: the lift scaled from the tested geos' spend to the
  market's at the test's ROI, and the market's spend;
* :func:`national`: the export as a national tool reads it, all of the above in one mapping.

Kept apart from the arms, which import their tools, so the bench's tests read it in the bench's
environment; the Robyn arm reads it through reticulate. numpy alone, for Python 3.11 and later.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

World = Mapping[str, Any]  # an export as np.load reads it, or a mapping of its arrays


def geo(world: World) -> bool:
    """Whether the export has a geo axis."""
    return "population" in world


def _names(world: World) -> list[str]:
    return [str(c) for c in world["channels"]]


def _national(world: World) -> list[int]:
    """The national channels' indices, in the channels' order."""
    national = {str(c) for c in world["national"]}
    return [c for c, name in enumerate(_names(world)) if name in national]


def _own(world: World) -> list[int]:
    """The channels each geo buys itself, in the channels' order."""
    national = _national(world)
    return [c for c in range(len(_names(world))) if c not in national]


def channel_of(world: World) -> np.ndarray:
    """Each cell's channel: each geo's own channels, geo by geo, then each national channel."""
    geos = np.asarray(world["population"]).size
    return np.array(_own(world) * geos + _national(world), dtype=np.int64)


def channel_quarter(world: World) -> dict[str, np.ndarray]:
    """``lower``, ``upper`` and ``status_quo``, one weekly spend a channel: on a geo export each
    cell's summed over the channel's cells, on a national export the export's own."""
    if not geo(world):
        return {
            key: np.asarray(world[key], dtype=float) for key in ("lower", "upper", "status_quo")
        }
    channel, count = channel_of(world), len(_names(world))
    return {
        key: np.bincount(channel, weights=np.asarray(world[key], dtype=float), minlength=count)
        for key in ("lower", "upper", "status_quo")
    }


def panel(world: World, weekly: Any) -> np.ndarray:
    """``(geos, channels)``: a weekly spend on each cell as each geo spends it, a national
    channel's allotted to the geos by population.

    Raises:
        ValueError: a plan that is not one weekly spend a cell.
    """
    shares = np.asarray(world["population"], dtype=float)
    own, national = _own(world), _national(world)
    weekly = np.asarray(weekly, dtype=float)
    if weekly.shape != (shares.size * len(own) + len(national),):
        raise ValueError(f"{shares.size} geos' cells are not {weekly.shape}")
    out = np.empty((shares.size, len(_names(world))))
    out[:, own] = weekly[: shares.size * len(own)].reshape(shares.size, len(own))
    out[:, national] = shares[:, None] * weekly[shares.size * len(own) :]
    return out


def cells(world: World, spend: Any) -> np.ndarray:
    """A weekly spend for each geo and channel, ``(geos, channels)``, as the cells read it: each
    geo's own channels as they are, each national channel summed over the geos."""
    spend = np.asarray(spend, dtype=float)
    own, national = _own(world), _national(world)
    if spend.shape != (np.asarray(world["population"]).size, len(_names(world))):
        raise ValueError(f"a spend for each geo and channel, not {spend.shape}")
    return np.concatenate([spend[:, own].ravel(), spend[:, national].sum(axis=0)])


def market(world: World) -> tuple[np.ndarray, np.ndarray]:
    """The KPI, ``(weeks,)``, and the spend, ``(weeks, channels)``, summed over the geos where
    there are geos."""
    sales = np.asarray(world["sales"], dtype=float)
    spend = np.asarray(world["spend"], dtype=float)
    if not geo(world):
        return sales, spend
    return sales.sum(axis=1), spend.sum(axis=1)


def withheld(world: World, row: int) -> np.ndarray:
    """Each of lift row ``row``'s dark weeks' spend its test withheld: its channel's spend in the
    geos it took dark, summed, on a geo export; the channel's spend on a national one."""
    c = _names(world).index(str(world["lift_channel"][row]))
    first = int(world["lift_start"][row]) - 1
    dark = slice(first, first + int(world["lift_weeks"]))
    spend = np.asarray(world["spend"], dtype=float)
    if not geo(world):
        return spend[dark, c]
    taken = np.asarray(world["lift_geos"], dtype=bool)[row]
    return spend[dark][:, taken, c].sum(axis=1)


def market_lift(world: World) -> dict[str, np.ndarray]:
    """Each lift row as the market's: ``lift``, the sales the test lost over its dark weeks and
    their cooldown (``-delta_y`` a dark week), times the market's spend over the dark weeks over the
    spend the test withheld, so at the test's ROI; ``spend``, the market's spend over the dark
    weeks. On a national export the test withheld the market's spend, and the ratio is 1."""
    rows = len(world["lift_channel"])
    weeks = int(world["lift_weeks"])
    names = _names(world)
    _, spend = market(world)
    lift, total = np.empty(rows), np.empty(rows)
    for row in range(rows):
        c = names.index(str(world["lift_channel"][row]))
        first = int(world["lift_start"][row]) - 1
        total[row] = spend[first : first + weeks, c].sum()
        tested = withheld(world, row).sum()
        lift[row] = -float(world["lift_delta_y"][row]) * weeks * total[row] / tested
    return {"lift": lift, "spend": total}


GEO_KEYS = ("population", "national", "lift_geos")  # what only a geo export holds


def national(world: World) -> dict[str, Any]:
    """The export as a national tool reads it: every array the export holds, but on a geo export
    the KPI and the spend summed over the geos (:func:`market`), the quarter a channel at a time
    (:func:`channel_quarter`) and none of :data:`GEO_KEYS`; and beside them, for each lift row,
    ``lift_withheld``, its dark weeks' spend withheld (:func:`withheld`), ``(rows, dark weeks)``,
    and ``lift_market_lift`` and ``lift_market_spend``, the row as the market's
    (:func:`market_lift`)."""
    out = {key: np.asarray(world[key]) for key in world if key not in GEO_KEYS}
    sales, spend = market(world)
    out |= {"sales": sales, "spend": spend, **channel_quarter(world)}
    rows, weeks = len(world["lift_channel"]), int(world["lift_weeks"])
    out["lift_withheld"] = np.array([withheld(world, row) for row in range(rows)]).reshape(
        rows, weeks
    )
    read = market_lift(world)
    out["lift_market_lift"], out["lift_market_spend"] = read["lift"], read["spend"]
    return out

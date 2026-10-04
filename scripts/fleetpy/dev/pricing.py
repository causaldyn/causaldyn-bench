"""Track R's FleetPy plug-ins: a fare factor per origin zone and period, and riders who pay up to a
drawn multiple of their trip's list fare.

FleetPy builds both objects itself and hands them nothing of the bench's, so a run's ledger is
module state: :func:`open_ledger` starts it before the simulation is built, every rider writes its
decision to it, and the pricing strategy reads the period just closed from it. One simulation per
process.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from src.demand.TravelerModels import (  # ty: ignore[unresolved-import]
    IndividualConstraintRequest,
    PriceSensitiveIndividualConstraintRequest,
)
from src.fleetctrl.pricing.DynamicPricingBase import (  # ty: ignore[unresolved-import]
    DynamicPricingBase,
)
from src.misc.globals import (  # ty: ignore[unresolved-import]
    G_OFFER_FARE,
    G_OP_FARE_B,
    G_OP_FARE_D,
    G_OP_FARE_T,
    VRL_STATES,
)

PERIOD = 900  # seconds in a decision period; a day has 96
COLUMNS = ("requests", "offered", "accepted", "fares", "list_fares", "lost_list_fares")


@dataclass(frozen=True)
class Decision:
    """One rider's answer, fares in cents. ``within_limit`` is False without an offer."""

    rq_time: float
    period: int
    zone: int
    offered: bool
    accepted: bool
    within_limit: bool
    list_fare: float
    offered_fare: float
    wtp_ratio: float


@dataclass
class Ledger:
    zone_of_node: np.ndarray
    zones: int
    decisions: dict = field(default_factory=dict)  # request id -> Decision
    by_period: dict = field(default_factory=dict)  # period -> request ids decided in it
    idle: dict = field(default_factory=dict)  # period -> vehicles idle per zone at its start
    factors: dict = field(default_factory=dict)  # period -> the policy's factor per zone

    def record(self, rid, decision: Decision) -> None:
        # A rider asked again replaces its first answer; under immediate decisions none is.
        earlier = self.decisions.get(rid)
        if earlier is not None:
            self.by_period[earlier.period].discard(rid)
        self.decisions[rid] = decision
        self.by_period.setdefault(decision.period, set()).add(rid)

    def closed(self, period: int) -> dict[str, np.ndarray]:
        """Per zone, what the riders decided in ``period``: counts, and fares in cents."""
        out = {name: np.zeros(self.zones) for name in COLUMNS}
        for rid in self.by_period.get(period, ()):
            d = self.decisions[rid]
            out["requests"][d.zone] += 1
            out["offered"][d.zone] += d.offered
            out["accepted"][d.zone] += d.accepted
            out["fares"][d.zone] += d.offered_fare if d.accepted else 0.0
            out["list_fares"][d.zone] += d.list_fare
            out["lost_list_fares"][d.zone] += 0.0 if d.accepted else d.list_fare
        return out

    def frame(self) -> pd.DataFrame:
        rows = {rid: asdict(d) for rid, d in self.decisions.items()}
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("request_id")


_LEDGER: Ledger | None = None


def open_ledger(node_zone_csv: str) -> Ledger:
    global _LEDGER
    table = pd.read_csv(node_zone_csv)
    zone_of_node = np.full(int(table["node_index"].max()) + 1, -1, dtype=np.int64)
    zone_of_node[table["node_index"].to_numpy()] = table["zone_id"].to_numpy()
    zones = int(zone_of_node.max()) + 1
    if np.any(zone_of_node < 0) or set(np.unique(zone_of_node).tolist()) != set(range(zones)):
        raise ValueError(f"{node_zone_csv}: zones are not 0..{zones - 1} over every node")
    _LEDGER = Ledger(zone_of_node, zones)
    return _LEDGER


def ledger() -> Ledger:
    if _LEDGER is None:
        raise RuntimeError("no ledger: call open_ledger before building the simulation")
    return _LEDGER


def list_fare(parameters, distance_m: float, time_s: float) -> float:
    """The operator's fare at factor 1, before FleetPy rounds it down to a cent."""
    return (
        parameters.get(G_OP_FARE_B, 0)
        + distance_m * parameters.get(G_OP_FARE_D, 0)
        + time_s * parameters.get(G_OP_FARE_T, 0)
    )


class ListFareRatioRequest(PriceSensitiveIndividualConstraintRequest):
    """A rider who takes an offer up to ``wtp_ratio`` times the list fare of the trip.

    The limit in cents is the ratio times the list fare, which the rider prices as the operator
    does, from the direct route's distance and time. The Manhattan days scale every edge by one
    factor for the whole day, so the direct route, and with it the list fare, is the same when
    FleetPy builds the rider, before the factor applies, as when the operator prices the trip.
    """

    type = "ListFareRatioRequest"

    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        # The parent reads an absolute max_fare column, which this rider does not have.
        IndividualConstraintRequest.__init__(
            self, rq_row, routing_engine, simulation_time_step, scenario_parameters
        )
        if self.nr_pax != 1:
            raise ValueError(f"request {self.rid}: {self.nr_pax} passengers; the fare is per rider")
        self.wtp_ratio = float(rq_row["wtp_ratio"])
        self.list_fare = list_fare(
            scenario_parameters, self.direct_route_travel_distance, self.direct_route_travel_time
        )
        self.max_fare = self.wtp_ratio * self.list_fare

    def choose_offer(self, sc_parameters, simulation_time):
        choice = super().choose_offer(sc_parameters, simulation_time)
        offer = self.offer.get(0)
        offered = offer is not None and not offer.service_declined()
        fare = float(offer.get(G_OFFER_FARE)) if offered else 0.0
        book = ledger()
        book.record(
            self.rid,
            Decision(
                rq_time=float(self.rq_time),
                period=int(simulation_time // PERIOD),
                zone=int(book.zone_of_node[self.o_pos[0]]),
                offered=offered,
                accepted=choice is not None and choice >= 0,
                within_limit=offered and fare <= self.max_fare,
                list_fare=self.list_fare,
                offered_fare=fare,
                wtp_ratio=self.wtp_ratio,
            ),
        )
        return choice


class ZonalFactorPricing(DynamicPricingBase):
    """A fare factor per origin zone, which the bench's policy sets at the start of every period.

    ``policy(period, ledger)`` returns the period's factor per zone. It is called once a period, at
    the period's first step, after FleetPy has moved the fleet to it and before any rider of the
    period is priced; the vehicles idle then are in ``ledger.idle[period]``.
    """

    def __init__(self, fleetctrl, operator_attributes, solver="Gurobi"):
        super().__init__(fleetctrl, operator_attributes, solver)
        self.policy: Callable[[int, Ledger], np.ndarray] | None = None  # the day runner sets it
        self.period = -1
        self.factors = np.empty(0)

    def _advance(self, sim_time) -> None:
        period = int(sim_time // PERIOD)
        if period == self.period:
            return
        if period != self.period + 1 or sim_time % PERIOD != 0:
            raise RuntimeError(f"period {self.period} -> {period} at {sim_time} s: not a boundary")
        if self.policy is None:
            raise RuntimeError("no policy: the day runner sets one before the simulation runs")
        self.period = period
        book = ledger()
        idle = np.zeros(book.zones)
        for vehicle in self.fleetctrl.sim_vehicles:
            if vehicle.status == VRL_STATES.IDLE:
                idle[book.zone_of_node[vehicle.pos[0]]] += 1
        book.idle[period] = idle
        factors = np.asarray(self.policy(period, book), dtype=np.float64)
        if factors.shape != (book.zones,) or not np.all(np.isfinite(factors)):
            raise ValueError(f"period {period}: the policy returned {factors!r}")
        self.factors = factors
        book.factors[period] = factors

    def get_elastic_price_factors(self, sim_time, expected_pu_time=None, o_pos=None, d_pos=None):
        if o_pos is None:
            raise ValueError("a fare factor is per origin zone, and FleetPy passed no origin")
        self._advance(sim_time)
        return 1.0, 1.0, float(self.factors[ledger().zone_of_node[o_pos[0]]])

    def update_current_price_factors(self, sim_time):
        self._advance(sim_time)

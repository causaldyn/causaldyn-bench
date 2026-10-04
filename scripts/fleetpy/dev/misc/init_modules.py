"""FleetPy's extension point: FleetPy imports ``dev.misc.init_modules`` if it can, and asks it for
modules by kind. Track R adds a pricing strategy and a rider; every other kind adds nothing."""


def add_dynamic_pricing_strategy_modules():
    return {"ZonalFactorPricing": ("dev.pricing", "ZonalFactorPricing")}


def add_request_models():
    return {"ListFareRatioRequest": ("dev.pricing", "ListFareRatioRequest")}


def _none():
    return {}


add_broker_modules = _none
add_charging_strategy_modules = _none
add_dev_routing_engines = _none
add_dev_simulation_environments = _none
add_dynamic_fleetsizing_strategy_modules = _none
add_fleet_control_modules = _none
add_forecast_models = _none
add_pt_control_modules = _none
add_repositioning_modules = _none
add_reservation_strategy_modules = _none
add_ride_pooling_batch_optimizer_modules = _none

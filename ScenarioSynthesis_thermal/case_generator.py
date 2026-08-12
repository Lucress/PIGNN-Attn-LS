"""Random, reproducible PyDHN thermal cases."""

from dataclasses import dataclass

import numpy as np
from pydhn import Soil

from ScenarioSynthesis_hydraulic.case_generator import generate_hydraulic_case
from ScenarioSynthesis_hydraulic.newton_raphson import solve_hydraulics_newton


@dataclass
class ThermalCase:
    """A thermal scenario with its already-solved hydraulic state."""

    net: object
    fluid: object
    soil: Soil
    seed: int
    initial_node_temperature: np.ndarray
    consumer_heat_demand: np.ndarray
    producer_supply_temperature: float
    hydraulic_result: object


def generate_thermal_case(
    seed: int | None = None,
    n_nodes: int = 16,
    heat_demand_range: tuple[float, float] = (2_000.0, 12_000.0),
    supply_temperature_range: tuple[float, float] = (75.0, 95.0),
    initial_temperature_range: tuple[float, float] = (30.0, 65.0),
    soil_temperature_range: tuple[float, float] = (5.0, 15.0),
) -> ThermalCase:
    """Build one thermal case after solving hydraulics with the custom solver.

    Heat demands are returned as positive Wh values.  They are stored on the
    consumer heat exchangers as negative ``delta_q`` setpoints because heat is
    removed from the district-heating water.
    """

    rng = np.random.default_rng(seed)
    # NetworkX can select a different valid cycle basis between interpreter
    # processes. Retry the identical physical case from deterministic cycle
    # flow initializations when Armijo stalls just above the 100 Pa threshold.
    hydraulic_case = None
    hydraulic_result = None
    for initial_cycle_mass_flow in (1e-4, 1e-3, -1e-3, 1e-2):
        hydraulic_case = generate_hydraulic_case(seed=seed, n_nodes=n_nodes)
        hydraulic_result = solve_hydraulics_newton(
            hydraulic_case.net,
            hydraulic_case.fluid,
            max_iters=100,
            error_threshold=100.0,
            compute_hydrostatic=False,
            adaptive=False,
            initial_cycle_mass_flow=initial_cycle_mass_flow,
            verbose=0,
        )
        if hydraulic_result.converged:
            break
    if not hydraulic_result.converged:
        raise RuntimeError(
            f"Hydraulic state did not converge for seed={seed}, "
            f"n_nodes={n_nodes}, final_error={hydraulic_result.final_error:.6g} Pa"
        )
    net = hydraulic_case.net

    consumer_heat_demand = rng.uniform(
        *heat_demand_range, size=len(net.consumers_mask)
    )
    net.set_edge_attribute(
        value="delta_q", name="setpoint_type_hx", mask=net.consumers_mask
    )
    net.set_edge_attribute(
        value="delta_q", name="setpoint_type_hx_rev", mask=net.consumers_mask
    )
    net.set_edge_attributes(
        -consumer_heat_demand,
        "setpoint_value_hx",
        mask=net.consumers_mask,
    )
    net.set_edge_attributes(
        consumer_heat_demand,
        "setpoint_value_hx_rev",
        mask=net.consumers_mask,
    )

    producer_supply_temperature = float(rng.uniform(*supply_temperature_range))
    net.set_edge_attribute(
        value="t_out", name="setpoint_type_hx", mask=net.producers_mask
    )
    net.set_edge_attribute(
        value="t_out", name="setpoint_type_hx_rev", mask=net.producers_mask
    )
    net.set_edge_attributes(
        np.array([producer_supply_temperature]),
        "setpoint_value_hx",
        mask=net.producers_mask,
    )
    net.set_edge_attributes(
        np.array([producer_supply_temperature]),
        "setpoint_value_hx_rev",
        mask=net.producers_mask,
    )

    initial_node_temperature = rng.uniform(
        *initial_temperature_range, size=net.n_nodes
    )
    net.set_node_attributes(initial_node_temperature, "temperature")
    soil = Soil(temp=float(rng.uniform(*soil_temperature_range)))

    return ThermalCase(
        net=net,
        fluid=hydraulic_case.fluid,
        soil=soil,
        seed=-1 if seed is None else int(seed),
        initial_node_temperature=initial_node_temperature,
        consumer_heat_demand=consumer_heat_demand,
        producer_supply_temperature=producer_supply_temperature,
        hydraulic_result=hydraulic_result,
    )

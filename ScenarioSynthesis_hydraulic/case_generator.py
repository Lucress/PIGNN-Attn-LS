"""Random, reproducible PyDHN hydraulic cases.

The topology is PyDHN's meshed ``star_network``.  Keeping a valid PyDHN
topology while randomising physical parameters and boundary conditions gives
the Newton solver non-trivial internal cycle flows without generating invalid
district-heating graphs.
"""

from dataclasses import dataclass

import numpy as np
from pydhn import ConstantWater, Network


@dataclass
class HydraulicCase:
    """A generated hydraulic scenario and the values used to create it."""

    net: object
    fluid: ConstantWater
    seed: int
    consumer_mass_flows: np.ndarray
    pressure_lift: float
    static_pressure: float
    pipe_lengths: np.ndarray
    pipe_diameters: np.ndarray
    pipe_roughness: np.ndarray


def build_two_line_network(n_nodes: int) -> Network:
    """Build a valid variable-size two-line district-heating network.

    The node budget is split between supply and return lines.  Each line has a
    connected backbone, the producer joins their roots, and one or more
    consumers join downstream nodes.  Two-hop pipe chords add internal
    hydraulic cycles whenever the size permits, so larger cases retain genuine
    nonlinear Newton unknowns instead of being only boundary-flow trees.
    """

    if not 4 <= n_nodes <= 32:
        raise ValueError("n_nodes must be between 4 and 32")
    n_supply = (n_nodes + 1) // 2
    n_return = n_nodes // 2
    net = Network()
    for i in range(n_supply):
        net.add_node(name=f"S{i}", x=float(i), y=1.0)
    for i in range(n_return):
        net.add_node(name=f"R{i}", x=float(i), y=0.0)

    for i in range(n_supply - 1):
        net.add_pipe(f"SP{i}_{i+1}", f"S{i}", f"S{i+1}", line="supply")
    for i in range(n_return - 1):
        net.add_pipe(f"RP{i+1}_{i}", f"R{i+1}", f"R{i}", line="return")

    # Add sparse internal loops without introducing parallel edges.
    for i in range(0, n_supply - 2, 3):
        net.add_pipe(f"SX{i}_{i+2}", f"S{i}", f"S{i+2}", line="supply")
    for i in range(0, n_return - 2, 3):
        net.add_pipe(f"RX{i+2}_{i}", f"R{i+2}", f"R{i}", line="return")

    n_consumers = min(n_supply - 1, n_return - 1, max(1, n_nodes // 6))
    supply_positions = np.linspace(1, n_supply - 1, n_consumers, dtype=int)
    return_positions = np.linspace(1, n_return - 1, n_consumers, dtype=int)
    # Always serve the terminal nodes; ``linspace(..., num=1)`` otherwise
    # returns only its start value and leaves the downstream pipe dead-ended.
    supply_positions[-1] = n_supply - 1
    return_positions[-1] = n_return - 1
    for i, (s_idx, r_idx) in enumerate(zip(supply_positions, return_positions)):
        net.add_consumer(f"SUB{i}", f"S{s_idx}", f"R{r_idx}")
    net.add_producer("main", "R0", "S0")
    return net


def generate_hydraulic_case(
    seed: int | None = None,
    n_nodes: int = 16,
    mass_flow_range: tuple[float, float] = (0.05, 0.20),
    length_range: tuple[float, float] = (50.0, 400.0),
    diameter_range: tuple[float, float] = (0.03, 0.07),
    roughness_range: tuple[float, float] = (0.02, 0.12),
    pressure_lift_range: tuple[float, float] = (60_000.0, 180_000.0),
    static_pressure_range: tuple[float, float] = (250_000.0, 500_000.0),
) -> HydraulicCase:
    """Build one deterministic random hydraulic case.

    Parameters use SI units: kg/s, m, m, mm, and Pa respectively.  PyDHN's
    pressure-drop convention requires the producer pressure lift to be stored
    as a negative differential pressure; ``HydraulicCase.pressure_lift`` is
    returned as a positive magnitude for easier inspection.
    """

    rng = np.random.default_rng(seed)
    net = build_two_line_network(n_nodes)
    fluid = ConstantWater()

    n_pipes = len(net.pipes_mask)
    pipe_lengths = rng.uniform(*length_range, size=n_pipes)
    pipe_diameters = rng.uniform(*diameter_range, size=n_pipes)
    pipe_roughness = rng.uniform(*roughness_range, size=n_pipes)

    net.set_edge_attributes(pipe_lengths, "length", mask=net.pipes_mask)
    net.set_edge_attributes(pipe_diameters, "diameter", mask=net.pipes_mask)
    net.set_edge_attributes(pipe_roughness, "roughness", mask=net.pipes_mask)

    consumer_mass_flows = rng.uniform(
        *mass_flow_range, size=len(net.consumers_mask)
    )
    net.set_edge_attribute(
        value="mass_flow", name="control_type", mask=net.consumers_mask
    )
    net.set_edge_attribute(
        value="mass_flow", name="setpoint_type_hyd", mask=net.consumers_mask
    )
    net.set_edge_attributes(
        consumer_mass_flows,
        "setpoint_value_hyd",
        mask=net.consumers_mask,
    )

    pressure_lift = float(rng.uniform(*pressure_lift_range))
    static_pressure = float(rng.uniform(*static_pressure_range))
    net.set_edge_attribute(
        value="pressure", name="setpoint_type_hyd", mask=net.producers_mask
    )
    net.set_edge_attributes(
        np.array([-pressure_lift]),
        "setpoint_value_hyd",
        mask=net.producers_mask,
    )
    net.set_edge_attributes(
        np.array([static_pressure]),
        "static_pressure",
        mask=net.producers_mask,
    )

    # A small non-zero initial value avoids undefined Reynolds numbers while
    # the cycle-flow initial guess is assembled by the custom solver.
    net.set_edge_attribute(value=1e-4, name="mass_flow")
    net.set_edge_attribute(value=0.0, name="delta_p")
    net.set_node_attribute(value=0.0, name="pressure")
    net.set_node_attribute(value=50.0, name="temperature")

    return HydraulicCase(
        net=net,
        fluid=fluid,
        seed=-1 if seed is None else int(seed),
        consumer_mass_flows=consumer_mass_flows,
        pressure_lift=pressure_lift,
        static_pressure=static_pressure,
        pipe_lengths=pipe_lengths,
        pipe_diameters=pipe_diameters,
        pipe_roughness=pipe_roughness,
    )

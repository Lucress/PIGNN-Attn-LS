"""Custom Newton--Raphson heat-balance solver for PyDHN networks.

PyDHN supplies the component outlet-temperature functions.  This module owns
the flow-oriented incidence matrices, nodal heat residual, analytic Jacobian,
Newton step, damping, convergence, and network state update.
"""

from dataclasses import dataclass
from typing import Any
from warnings import warn

import numpy as np
from pydhn.solving.temperature import compute_edge_temperatures


@dataclass
class ThermalNewtonResult:
    """Numerical result returned by :func:`solve_thermal_newton`."""

    converged: bool
    iterations: int
    residual_history: np.ndarray
    node_temperature: np.ndarray
    inlet_temperature: np.ndarray
    outlet_temperature: np.ndarray
    edge_temperature: np.ndarray
    outlet_temperature_derivative: np.ndarray
    delta_t: np.ndarray
    delta_q: np.ndarray

    @property
    def final_error(self) -> float:
        return float(self.residual_history[-1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "residual_history": self.residual_history,
            "node_temperature": self.node_temperature,
            "inlet_temperature": self.inlet_temperature,
            "outlet_temperature": self.outlet_temperature,
            "edge_temperature": self.edge_temperature,
            "outlet_temperature_derivative": self.outlet_temperature_derivative,
            "delta_t": self.delta_t,
            "delta_q": self.delta_q,
        }


def _oriented_balance_arrays(
    incidence_matrix: np.ndarray,
    mass_flow: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Build compact flow-oriented incidence and mixing arrays."""

    oriented = incidence_matrix * np.sign(mass_flow)[None, :]
    node_mask = np.where(np.abs(oriented).sum(axis=1) != 0.0)[0]
    edge_mask = np.where(np.abs(oriented).sum(axis=0) != 0.0)[0]
    oriented = oriented[np.ix_(node_mask, edge_mask)]
    incoming = np.maximum(oriented, 0.0)
    destination_rows = np.argmax(oriented, axis=0)
    source_columns = np.argmin(oriented, axis=0)
    inlet_mass_flow = incoming @ np.abs(mass_flow[edge_mask])
    if np.any(inlet_mass_flow <= 0.0):
        raise ValueError(
            "Every active thermal node needs incoming mass flow; check the "
            "hydraulic solution and edge directions"
        )
    return (
        oriented,
        incoming,
        edge_mask,
        node_mask,
        destination_rows,
        source_columns,
        inlet_mass_flow,
    )


def solve_thermal_newton(
    net,
    fluid,
    soil,
    *,
    max_iters: int = 100,
    error_threshold: float = 1e-6,
    damping_factor: float = 1.0,
    decreasing: bool = False,
    adaptive: bool = False,
    mass_flow_min: float = 1e-16,
    verbose: int = 1,
    ts_id: int | None = None,
) -> ThermalNewtonResult:
    """Solve the steady nodal heat balance with Newton--Raphson.

    The hydraulic mass flow must already be available on the network.  The
    input network is updated in place with node and edge temperatures.
    """

    if max_iters < 0:
        raise ValueError("max_iters must be non-negative")
    if error_threshold < 0:
        raise ValueError("error_threshold must be non-negative")
    if not 0 < damping_factor <= 1:
        raise ValueError("damping_factor must be in (0, 1]")
    if mass_flow_min <= 0:
        raise ValueError("mass_flow_min must be positive")

    edges, mass_flow = net.edges("mass_flow")
    mass_flow = np.asarray(mass_flow, dtype=float)
    small_flow = np.abs(mass_flow) < mass_flow_min
    if np.any(small_flow):
        # A tiny deterministic surrogate keeps the edge in the balance.  A
        # hydraulic solution with many zero-flow edges should be fixed before
        # relying on its thermal result.
        signs = np.sign(mass_flow)
        signs[signs == 0.0] = 1.0
        mass_flow[small_flow] = signs[small_flow] * mass_flow_min
        net.set_edge_attributes(mass_flow, "mass_flow")

    incidence_matrix = np.asarray(net.incidence_matrix, dtype=float)
    (
        _oriented,
        _incoming,
        edge_mask,
        node_mask,
        destination_rows,
        source_columns,
        inlet_mass_flow,
    ) = _oriented_balance_arrays(incidence_matrix, mass_flow)

    nodes, node_temperature = net.nodes("temperature")
    node_temperature = np.asarray(node_temperature, dtype=float)
    if np.any(~np.isfinite(node_temperature)):
        raise ValueError("All node temperatures must have finite initial values")

    dimension = len(node_mask)
    damp = float(damping_factor)
    converged = False
    residual_history: list[float] = []

    for iteration in range(max_iters + 1):
        net.set_node_attributes(node_temperature, "temperature")
        (
            inlet_temperature,
            outlet_temperature,
            edge_temperature,
            outlet_derivative,
            delta_q,
        ) = compute_edge_temperatures(
            net,
            fluid,
            soil,
            set_values=True,
            ts_id=ts_id,
        )

        active_abs_flow = np.abs(mass_flow[edge_mask])
        jacobian = np.zeros((dimension, dimension), dtype=float)
        np.add.at(
            jacobian,
            (destination_rows, source_columns),
            outlet_derivative[edge_mask] * active_abs_flow,
        )
        jacobian -= np.diag(inlet_mass_flow)

        transported_temperature = np.zeros((dimension, dimension), dtype=float)
        np.add.at(
            transported_temperature,
            (destination_rows, source_columns),
            outlet_temperature[edge_mask] * active_abs_flow,
        )
        transported_temperature -= np.diag(
            inlet_mass_flow * node_temperature[node_mask]
        )
        residual = transported_temperature.sum(axis=1)
        max_error = float(np.max(np.abs(residual)))
        residual_history.append(max_error)

        if verbose > 1:
            print(f"Thermal NR iteration {iteration}: {max_error:.6g}")
        if max_error <= error_threshold:
            converged = True
            break
        if iteration == max_iters:
            break

        step = np.linalg.solve(jacobian, -residual)
        node_temperature[node_mask] += damp * step

        if decreasing:
            damp = damping_factor * (1.0 - (iteration + 1) / (max_iters + 1))
        elif adaptive and iteration >= 2:
            damp = damping_factor if max_error <= residual_history[-2] else max(
                0.1 * damping_factor, 0.5 * damp
            )

    net.set_node_attributes(node_temperature, "temperature")
    delta_t = (outlet_temperature - inlet_temperature) * np.sign(mass_flow)
    net.set_edge_attributes(delta_t, "delta_t")

    if verbose > 0:
        status = "converged" if converged else "did not converge"
        print(
            f"Custom thermal Newton--Raphson {status} after {iteration} "
            f"iterations (error={max_error:.6g})"
        )
    if not converged and verbose > 0:
        warn("Thermal Newton--Raphson reached its iteration limit")

    return ThermalNewtonResult(
        converged=converged,
        iterations=iteration,
        residual_history=np.asarray(residual_history),
        node_temperature=node_temperature.copy(),
        inlet_temperature=np.asarray(inlet_temperature).copy(),
        outlet_temperature=np.asarray(outlet_temperature).copy(),
        edge_temperature=np.asarray(edge_temperature).copy(),
        outlet_temperature_derivative=np.asarray(outlet_derivative).copy(),
        delta_t=np.asarray(delta_t).copy(),
        delta_q=np.asarray(delta_q).copy(),
    )


def _demo() -> None:
    from ScenarioSynthesis_thermal.case_generator import generate_thermal_case

    case = generate_thermal_case(seed=7)
    result = solve_thermal_newton(
        case.net,
        case.fluid,
        case.soil,
        max_iters=50,
        error_threshold=1e-9,
        verbose=2,
    )
    print("node temperature [degC]:", result.node_temperature)
    print("edge heat exchange [Wh]:", result.delta_q)


if __name__ == "__main__":
    _demo()

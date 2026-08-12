"""Custom loop-based Newton--Raphson solver for PyDHN hydraulics.

Only PyDHN's component constitutive models are reused through ``compute_dp``.
The residual construction, cycle Jacobian, linear solve, damping, convergence,
mass-flow update, and pressure reconstruction are implemented here.
"""

from dataclasses import dataclass
from typing import Any
from warnings import warn

import networkx as nx
import numpy as np
from pydhn.solving.pressure import compute_dp


@dataclass
class HydraulicNewtonResult:
    """Numerical result returned by :func:`solve_hydraulics_newton`."""

    converged: bool
    iterations: int
    residual_history: np.ndarray
    cycle_mass_flow: np.ndarray
    mass_flow: np.ndarray
    delta_p: np.ndarray
    delta_p_friction: np.ndarray
    delta_p_hydrostatic: np.ndarray
    node_pressure: np.ndarray

    @property
    def final_error(self) -> float:
        return float(self.residual_history[-1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "residual_history": self.residual_history,
            "cycle_mass_flow": self.cycle_mass_flow,
            "mass_flow": self.mass_flow,
            "delta_p": self.delta_p,
            "delta_p_friction": self.delta_p_friction,
            "delta_p_hydrostatic": self.delta_p_hydrostatic,
            "node_pressure": self.node_pressure,
        }


def _assign_node_pressures(net, edges: np.ndarray, main_edge_index: int) -> np.ndarray:
    """Recover absolute node pressures from edge pressure differences."""

    edge_tuples = [tuple(edge) for edge in edges]
    main_u, main_v = edge_tuples[main_edge_index]
    static_pressure = float(net[(main_u, main_v)]["static_pressure"])
    if not np.isfinite(static_pressure):
        static_pressure = 1e6

    pressures = {main_v: static_pressure}
    edge_lookup = {edge: i for i, edge in enumerate(edge_tuples)}
    _, delta_p = net.edges("delta_p")

    for parent, child in nx.dfs_edges(net._graph.to_undirected(), source=main_v):
        forward = (parent, child)
        reverse = (child, parent)
        if forward in edge_lookup:
            pressures[child] = pressures[parent] - delta_p[edge_lookup[forward]]
        elif reverse in edge_lookup:
            pressures[child] = pressures[parent] + delta_p[edge_lookup[reverse]]
        else:  # pragma: no cover - guarded by the graph traversal itself
            raise RuntimeError(f"Missing edge between {parent!r} and {child!r}")

    net.set_node_attributes(pressures, "pressure")
    _, node_pressure = net.nodes("pressure")
    return np.asarray(node_pressure, dtype=float)


def solve_hydraulics_newton(
    net,
    fluid,
    *,
    max_iters: int = 100,
    error_threshold: float = 100.0,
    damping_factor: float = 1.0,
    decreasing: bool = False,
    adaptive: bool = False,
    line_search: bool = True,
    minimum_step: float = 1e-6,
    armijo_coefficient: float = 1e-4,
    compute_hydrostatic: bool = True,
    compute_singular: bool = False,
    initial_cycle_mass_flow: float = 1e-4,
    verbose: int = 1,
    ts_id: int | None = None,
) -> HydraulicNewtonResult:
    r"""Solve ``B phi(B.T m_cycle) = 0`` with Newton--Raphson.

    The implementation targets the same constrained two-line DHN topology as
    PyDHN.  Consumers normally impose mass flow, the main producer imposes a
    pressure difference, and remaining fundamental cycles are Newton unknowns.
    The input network is updated in place.
    """

    if max_iters < 0:
        raise ValueError("max_iters must be non-negative")
    if error_threshold < 0:
        raise ValueError("error_threshold must be non-negative")
    if not 0 < damping_factor <= 1:
        raise ValueError("damping_factor must be in (0, 1]")
    if not 0 < minimum_step <= 1:
        raise ValueError("minimum_step must be in (0, 1]")
    if not 0 <= armijo_coefficient < 1:
        raise ValueError("armijo_coefficient must be in [0, 1)")

    cycle_matrix = np.asarray(net.cycle_matrix, dtype=float)
    if cycle_matrix.size == 0:
        raise ValueError("The network has no hydraulic cycles to solve")

    leaves = np.asarray(net.leaf_components_mask, dtype=int)
    pressure_edges = np.intersect1d(leaves, net.pressure_setpoints_mask)
    mass_edges = np.intersect1d(leaves, net.mass_flow_setpoints_mask)
    if len(pressure_edges) == 0:
        raise ValueError("At least one leaf component needs a pressure setpoint")

    main_idx = int(pressure_edges[0])
    secondary = np.setdiff1d(leaves, main_idx)
    cycle_indices = np.arange(cycle_matrix.shape[0])
    pressure_cycle_indices = np.where(np.isin(secondary, pressure_edges))[0]
    mass_cycle_indices = np.where(np.isin(secondary, mass_edges))[0]
    unknown_cycle_indices = np.setdiff1d(
        cycle_indices,
        np.union1d(pressure_cycle_indices, mass_cycle_indices),
    )

    edges, setpoint_values = net.edges("setpoint_value_hyd")
    edges = np.asarray(edges)
    setpoint_values = np.asarray(setpoint_values, dtype=float)
    directions = cycle_matrix[:, secondary].sum(axis=1)
    if np.any(np.abs(directions[mass_cycle_indices]) < 0.5):
        raise ValueError("Could not orient one or more mass-flow setpoint cycles")

    cycle_mass_flow = np.zeros(cycle_matrix.shape[0], dtype=float)
    cycle_mass_flow[mass_cycle_indices] = (
        setpoint_values[mass_edges] * directions[mass_cycle_indices]
    )
    cycle_mass_flow[unknown_cycle_indices] = float(initial_cycle_mass_flow)
    mass_flow = cycle_matrix.T @ cycle_mass_flow

    def evaluate(candidate_mass_flow: np.ndarray):
        """Evaluate component laws and the constrained cycle residual."""

        net.set_edge_attributes(candidate_mass_flow, "mass_flow")
        values = compute_dp(
            net=net,
            fluid=fluid,
            set_values=False,
            compute_hydrostatic=compute_hydrostatic,
            compute_singular=compute_singular,
            mask=None,
            ts_id=ts_id,
        )
        candidate_dp, friction, hydrostatic, derivative = values
        candidate_dp[pressure_edges] = setpoint_values[pressure_edges]
        cycle_residual = cycle_matrix @ candidate_dp
        candidate_dp[mass_edges] = (
            -cycle_residual[mass_cycle_indices]
            * directions[mass_cycle_indices]
        )
        cycle_residual = cycle_matrix @ candidate_dp
        error = float(np.max(np.abs(cycle_residual)))
        return (
            candidate_dp,
            friction,
            hydrostatic,
            derivative,
            cycle_residual,
            error,
        )

    residual_history: list[float] = []
    converged = False
    damp = float(damping_factor)

    for iteration in range(max_iters + 1):
        (
            delta_p,
            delta_p_friction,
            delta_p_hydrostatic,
            dp_derivative,
            all_cycle_residuals,
            max_error,
        ) = evaluate(mass_flow)
        residual_history.append(max_error)

        if verbose > 1:
            print(f"Hydraulic NR iteration {iteration}: {max_error:.6g} Pa")
        if max_error <= error_threshold:
            converged = True
            break
        if iteration == max_iters:
            break

        if len(unknown_cycle_indices) == 0:
            break

        edge_jacobian = np.diag(dp_derivative)
        unknown_cycles = cycle_matrix[unknown_cycle_indices]
        jacobian = unknown_cycles @ edge_jacobian @ unknown_cycles.T
        residual = unknown_cycles @ delta_p
        active = np.any(np.abs(jacobian) > 0.0, axis=0)
        if not np.any(active):
            raise np.linalg.LinAlgError("Hydraulic cycle Jacobian is all zero")

        step = np.zeros(len(unknown_cycle_indices), dtype=float)
        reduced_jacobian = jacobian[np.ix_(active, active)]
        step[active] = np.linalg.solve(reduced_jacobian, -residual[active])

        step_scale = damp
        if line_search:
            accepted = False
            while step_scale >= minimum_step:
                candidate_cycles = cycle_mass_flow.copy()
                candidate_cycles[unknown_cycle_indices] += step_scale * step
                candidate_mass_flow = cycle_matrix.T @ candidate_cycles
                candidate_error = evaluate(candidate_mass_flow)[-1]
                sufficient_decrease = (
                    candidate_error
                    <= (1.0 - armijo_coefficient * step_scale) * max_error
                )
                if sufficient_decrease:
                    cycle_mass_flow = candidate_cycles
                    mass_flow = candidate_mass_flow
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                if verbose > 0:
                    warn("Hydraulic line search could not reduce the residual")
                break
        else:
            cycle_mass_flow[unknown_cycle_indices] += step_scale * step
            mass_flow = cycle_matrix.T @ cycle_mass_flow

        if decreasing:
            damp = damping_factor * (1.0 - (iteration + 1) / (max_iters + 1))
        elif adaptive and iteration >= 2:
            damp = damping_factor if max_error <= residual_history[-2] else max(
                0.1 * damping_factor, 0.5 * damp
            )

    net.set_edge_attributes(mass_flow, "mass_flow")
    net.set_edge_attributes(delta_p, "delta_p")
    net.set_edge_attributes(delta_p_friction, "delta_p_friction")
    net.set_edge_attributes(delta_p_hydrostatic, "delta_p_hydrostatic")
    node_pressure = _assign_node_pressures(net, edges, main_idx)

    if verbose > 0:
        status = "converged" if converged else "did not converge"
        print(
            f"Custom hydraulic Newton--Raphson {status} after {iteration} "
            f"iterations (error={max_error:.6g} Pa)"
        )
    if not converged and verbose > 0:
        warn("Hydraulic Newton--Raphson reached its iteration limit")

    return HydraulicNewtonResult(
        converged=converged,
        iterations=iteration,
        residual_history=np.asarray(residual_history),
        cycle_mass_flow=cycle_mass_flow.copy(),
        mass_flow=np.asarray(mass_flow).copy(),
        delta_p=np.asarray(delta_p).copy(),
        delta_p_friction=np.asarray(delta_p_friction).copy(),
        delta_p_hydrostatic=np.asarray(delta_p_hydrostatic).copy(),
        node_pressure=node_pressure.copy(),
    )


def _demo() -> None:
    from ScenarioSynthesis_hydraulic.case_generator import generate_hydraulic_case

    case = generate_hydraulic_case(seed=7)
    result = solve_hydraulics_newton(
        case.net,
        case.fluid,
        max_iters=50,
        error_threshold=100.0,
        compute_hydrostatic=False,
        verbose=2,
    )
    print("mass flow [kg/s]:", result.mass_flow)
    print("node pressure [Pa]:", result.node_pressure)


if __name__ == "__main__":
    _demo()

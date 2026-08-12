"""Generate thermal Newton--Raphson cases and store them in Parquet."""

import argparse
from functools import partial
import io
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .case_generator import generate_thermal_case
from .newton_raphson import solve_thermal_newton


def _array_bytes(value) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def generate_record(
    seed: int,
    max_iters: int,
    error_threshold: float,
    min_nodes: int,
    max_nodes: int,
) -> dict:
    """Generate and solve one serialisable thermal scenario."""

    n_nodes = min_nodes + seed % (max_nodes - min_nodes + 1)
    case = generate_thermal_case(seed=seed, n_nodes=n_nodes)
    net = case.net
    _, mass_flow = net.edges("mass_flow")
    result = solve_thermal_newton(
        net,
        case.fluid,
        case.soil,
        max_iters=max_iters,
        error_threshold=error_threshold,
        adaptive=False,
        verbose=0,
    )

    edges, edge_names, component_type = net.edges(["name", "component_type"])
    nodes, _ = net.nodes()
    # For fixed mass flow and boundary conditions every PyDHN component used
    # by this generator has the local affine map
    #
    #     T_out = dT_out/dT_in * T_in + intercept.
    #
    # Persisting both coefficients lets the surrogate evaluate the exact
    # differentiable heat-balance residual without rebuilding PyDHN objects.
    outlet_intercept = (
        result.outlet_temperature
        - result.outlet_temperature_derivative * result.inlet_temperature
    )
    return {
        "seed": seed,
        "n_nodes": n_nodes,
        "hydraulic_converged": case.hydraulic_result.converged,
        "hydraulic_iterations": case.hydraulic_result.iterations,
        "thermal_converged": result.converged,
        "thermal_iterations": result.iterations,
        "final_error": result.final_error,
        "edges": _array_bytes(np.asarray(edges, dtype=str)),
        "edge_names": _array_bytes(np.asarray(edge_names, dtype=str)),
        "component_type": _array_bytes(np.asarray(component_type, dtype=str)),
        "nodes": _array_bytes(np.asarray(nodes, dtype=str)),
        "incidence_matrix": _array_bytes(np.asarray(net.incidence_matrix)),
        "mass_flow": _array_bytes(mass_flow),
        "consumer_heat_demand_wh": _array_bytes(case.consumer_heat_demand),
        "producer_supply_temperature_c": case.producer_supply_temperature,
        "initial_node_temperature_c": _array_bytes(
            case.initial_node_temperature
        ),
        "node_temperature_newton_c": _array_bytes(result.node_temperature),
        "inlet_temperature_newton_c": _array_bytes(result.inlet_temperature),
        "outlet_temperature_newton_c": _array_bytes(result.outlet_temperature),
        "outlet_temperature_derivative": _array_bytes(
            result.outlet_temperature_derivative
        ),
        "outlet_temperature_intercept_c": _array_bytes(outlet_intercept),
        "edge_temperature_newton_c": _array_bytes(result.edge_temperature),
        "delta_t_newton_k": _array_bytes(result.delta_t),
        "delta_q_newton_wh": _array_bytes(result.delta_q),
        "residual_history": _array_bytes(result.residual_history),
    }


def generate_dataset(
    output: str | Path,
    runs: int,
    seed: int = 0,
    max_iters: int = 100,
    error_threshold: float = 1e-6,
    batch_size: int = 100,
    workers: int = 1,
    min_nodes: int = 4,
    max_nodes: int = 32,
    overwrite: bool = False,
) -> Path:
    """Generate ``runs`` cases and stream row groups to one Parquet file."""

    if runs <= 0:
        raise ValueError("runs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not 4 <= min_nodes <= max_nodes <= 32:
        raise ValueError("node range must satisfy 4 <= min_nodes <= max_nodes <= 32")
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass overwrite=True: {output}")

    worker = partial(
        generate_record,
        max_iters=max_iters,
        error_threshold=error_threshold,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
    )
    seeds = range(seed, seed + runs)
    pool = None
    if workers == 1:
        records = map(worker, seeds)
    else:
        start_method = "spawn" if sys.platform.startswith("win") else "fork"
        context = mp.get_context(start_method)
        pool = context.Pool(processes=workers)
        records = pool.imap(worker, seeds, chunksize=1)

    writer = None
    batch: list[dict] = []
    try:
        for record in records:
            batch.append(record)
            if len(batch) < batch_size:
                continue
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            batch.clear()
        if batch:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
            pool = None
        raise
    finally:
        if writer is not None:
            writer.close()
        if pool is not None:
            pool.close()
            pool.join()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--error-threshold", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("thermal_newton_cases.parquet")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = generate_dataset(
        args.output,
        runs=args.runs,
        seed=args.seed,
        max_iters=args.max_iters,
        error_threshold=args.error_threshold,
        batch_size=args.batch_size,
        workers=args.workers,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        overwrite=args.overwrite,
    )
    print(f"Wrote {args.runs} thermal cases to {output}")


if __name__ == "__main__":
    main()

"""Generate hydraulic Newton--Raphson cases and store them in Parquet."""

import argparse
from functools import partial
import io
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .case_generator import generate_hydraulic_case
from .newton_raphson import solve_hydraulics_newton


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
    """Generate and solve one serialisable hydraulic scenario."""

    n_nodes = min_nodes + seed % (max_nodes - min_nodes + 1)
    case = None
    result = None
    mass_flow_start = None
    for initial_cycle_mass_flow in (1e-4, 1e-3, -1e-3, 1e-2):
        case = generate_hydraulic_case(seed=seed, n_nodes=n_nodes)
        net = case.net
        _, mass_flow_start = net.edges("mass_flow")
        result = solve_hydraulics_newton(
            net,
            case.fluid,
            max_iters=max_iters,
            error_threshold=error_threshold,
            compute_hydrostatic=False,
            adaptive=False,
            initial_cycle_mass_flow=initial_cycle_mass_flow,
            verbose=0,
        )
        if result.converged:
            break

    edges, edge_names, component_type = net.edges(["name", "component_type"])
    _, hydraulic_setpoint_edge = net.edges("setpoint_value_hyd")
    nodes, _ = net.nodes()
    return {
        "seed": seed,
        "n_nodes": n_nodes,
        "converged": result.converged,
        "iterations": result.iterations,
        "final_error_pa": result.final_error,
        "edges": _array_bytes(np.asarray(edges, dtype=str)),
        "edge_names": _array_bytes(np.asarray(edge_names, dtype=str)),
        "component_type": _array_bytes(np.asarray(component_type, dtype=str)),
        "nodes": _array_bytes(np.asarray(nodes, dtype=str)),
        "incidence_matrix": _array_bytes(np.asarray(net.incidence_matrix)),
        "cycle_matrix": _array_bytes(np.asarray(net.cycle_matrix)),
        "consumer_mass_flow_setpoint": _array_bytes(case.consumer_mass_flows),
        # Edge-aligned values avoid relying on the internal ordering of
        # PyDHN's consumer/producer masks in downstream graph datasets.
        "hydraulic_setpoint_edge": _array_bytes(hydraulic_setpoint_edge),
        "pressure_lift_pa": case.pressure_lift,
        "static_pressure_pa": case.static_pressure,
        "pipe_length_m": _array_bytes(case.pipe_lengths),
        "pipe_diameter_m": _array_bytes(case.pipe_diameters),
        "pipe_roughness_mm": _array_bytes(case.pipe_roughness),
        "mass_flow_start": _array_bytes(mass_flow_start),
        "cycle_mass_flow_newton": _array_bytes(result.cycle_mass_flow),
        "mass_flow_newton": _array_bytes(result.mass_flow),
        "delta_p_newton": _array_bytes(result.delta_p),
        "node_pressure_newton": _array_bytes(result.node_pressure),
        "residual_history": _array_bytes(result.residual_history),
    }


def generate_dataset(
    output: str | Path,
    runs: int,
    seed: int = 0,
    max_iters: int = 100,
    error_threshold: float = 100.0,
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
    parser.add_argument("--error-threshold", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("hydraulic_newton_cases.parquet")
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
    print(f"Wrote {args.runs} hydraulic cases to {output}")


if __name__ == "__main__":
    main()

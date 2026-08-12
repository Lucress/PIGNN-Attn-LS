#!/usr/bin/env python3
"""Summarize V_start and V_newton RMSE/residual baselines for PPC parquets."""

import argparse
import math

import torch
from torch.utils.data import DataLoader

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from train_valid_test_gridfm import (
    angle_diff,
    compute_power_flow_residual_metrics,
    ensure_dense_y_for_metrics,
    residual_distribution,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--PARQUET", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--BATCH", type=int, default=32)
    p.add_argument("--target_S_base", type=float, default=1e8)
    p.add_argument("--lazy_parquet", action="store_true")
    p.add_argument("--row_group_cache_size", type=int, default=2)
    p.add_argument("--dataset_complex_dtype", choices=("complex64", "complex128"), default="complex128")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def summarize(loader, device, which):
    sum_mse = sum_mag = sum_ang = 0.0
    sum_dp = sum_dq = sum_dp_mva = sum_dq_mva = 0.0
    n_graphs = n_nodes = 0
    dp_values = []
    dq_values = []

    for batch in loader:
        B_eff = int(batch["sizes"].numel())
        n_graphs += B_eff
        n_nodes_batch = int(batch["V_newton"].numel() // 2)
        n_nodes += n_nodes_batch
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        Vpred = batch_dev[which]
        target = batch_dev["V_newton"]
        dmag = Vpred[..., 0] - target[..., 0]
        dang = angle_diff(Vpred[..., 1], target[..., 1])
        sum_mag += float((dmag * dmag).sum().item())
        sum_ang += float((dang * dang).sum().item())
        sum_mse += float(((dmag * dmag) + (dang * dang)).sum().item())

        Y = ensure_dense_y_for_metrics(
            batch_dev.get("Ybus", None),
            batch_dev["bus_type"],
            batch_dev["Branch_f_bus"],
            batch_dev["Branch_t_bus"],
            batch_dev["Branch_status"],
            batch_dev["Branch_tau"],
            batch_dev["Branch_shift_deg"],
            batch_dev["Branch_y_series_from"],
            batch_dev["Branch_y_series_to"],
            batch_dev["Branch_y_series_ft"],
            batch_dev["Branch_y_shunt_from"],
            batch_dev["Branch_y_shunt_to"],
            batch_dev["Y_shunt_bus"],
        )
        metrics = compute_power_flow_residual_metrics(
            Y,
            Vpred,
            batch_dev["S_start"],
            batch_dev["bus_type"],
            n_nodes_per_graph=batch_dev["sizes"],
            S_base=batch_dev.get("S_base", None),
        )
        sum_dp += metrics["max_dp_pu"].sum().item()
        sum_dq += metrics["max_dq_pu"].sum().item()
        sum_dp_mva += metrics.get("max_dp_mva", metrics["max_dp_pu"]).sum().item()
        sum_dq_mva += metrics.get("max_dq_mva", metrics["max_dq_pu"]).sum().item()
        dp_values.append(metrics["dp_abs_valid"].cpu())
        dq_values.append(metrics["dq_abs_valid"].cpu())

    dist = residual_distribution(dp_values, dq_values, tol_pu=1e-6)
    return {
        "rmse": math.sqrt(sum_mse / max(n_nodes, 1)),
        "mag": math.sqrt(sum_mag / max(n_nodes, 1)),
        "ang_deg": math.sqrt(sum_ang / max(n_nodes, 1)) * 180.0 / math.pi,
        "dPinf": sum_dp / max(n_graphs, 1),
        "dQinf": sum_dq / max(n_graphs, 1),
        "dP_mw": sum_dp_mva / max(n_graphs, 1),
        "dQ_mvar": sum_dq_mva / max(n_graphs, 1),
        "meanP": dist["mean_dp_pu"],
        "meanQ": dist["mean_dq_pu"],
        "p95P": dist["p95_dp_pu"],
        "p95Q": dist["p95_dq_pu"],
        "rmseP": dist["rmse_dp_pu"],
        "rmseQ": dist["rmse_dq_pu"],
        "n_graphs": n_graphs,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    ds = ChanghunDataset(
        args.PARQUET,
        per_unit=True,
        target_S_base=args.target_S_base,
        share_grid=True,
        share_ybus=True,
        lazy_row_groups=args.lazy_parquet,
        row_group_cache_size=args.row_group_cache_size,
        complex_dtype=args.dataset_complex_dtype,
    )
    loader = DataLoader(ds, batch_size=args.BATCH, shuffle=False, collate_fn=collate_blockdiag)
    print(f"LABEL\t{args.label}")
    print(f"ROWS\t{len(ds)}")
    for which in ("V_start", "V_newton"):
        m = summarize(loader, device, which)
        print(
            "METRIC\t{which}\tgraphs={n_graphs}\trmse={rmse:.6e}\tmag={mag:.6e}\t"
            "ang_deg={ang_deg:.6e}\tdPinf={dPinf:.6e}\tdQinf={dQinf:.6e}\t"
            "meanP={meanP:.6e}\tmeanQ={meanQ:.6e}\tp95P={p95P:.6e}\tp95Q={p95Q:.6e}\t"
            "rmseP={rmseP:.6e}\trmseQ={rmseQ:.6e}".format(which=which, **m)
        )


if __name__ == "__main__":
    main()

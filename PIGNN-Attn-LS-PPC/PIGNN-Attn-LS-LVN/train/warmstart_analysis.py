"""Model 3 warm-start analysis (O3).

Evaluates the practical value of using Model 2 (physics-informed PIGNN-OPF)
predictions as warm-start initial conditions for a classical Newton-Raphson
solver, rather than the standard flat start (V=1∠0, Pg nominal).

Metrics collected per test sample:
    - NR iterations to convergence from flat start   (nr_flat_iters)
    - NR iterations to convergence from model warm start (nr_warm_iters)
    - Iteration saving = nr_flat_iters - nr_warm_iters
    - Wall-clock time per solve (flat vs warm)

Usage (run from the project root with the LVN environment active):

    python -m train.warmstart_analysis \\
        --ckpt results/runs/<run_id>/ckpt/best.ckpt \\
        --config configs/opf_pignn_attn_ls.yaml \\
        --out results/warmstart/

The script does NOT require the full training pipeline — it loads the checkpoint,
runs inference once over the test set, then calls the lightweight NR solver.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch


# ── Lightweight NR solver (CPU-only, same as generate_synthetic.py) ───────────

def _nr_iters(Ybus: np.ndarray, P_set: np.ndarray, Q_set: np.ndarray,
              bus_typ: np.ndarray, v0: np.ndarray, th0: np.ndarray,
              tol: float = 1e-4, max_iters: int = 100) -> int:
    """Run NR power flow; return number of iterations to reach convergence."""
    v, th = v0.copy(), th0.copy()
    slack = (bus_typ == 1)
    pv    = (bus_typ == 2)
    for it in range(1, max_iters + 1):
        Vc = v * np.exp(1j * th)
        Sc = Vc * np.conj(Ybus @ Vc)
        dP = np.where(slack, 0.0, P_set - Sc.real)
        dQ = np.where(slack | pv, 0.0, Q_set - Sc.imag)
        res = np.max(np.abs(np.concatenate([dP, dQ])))
        if res < tol:
            return it
        th += 0.05 * dP
        v  += 0.05 * dQ
        v   = np.clip(v, 0.75, 1.25)
    return max_iters


def run_warmstart_analysis(
    ckpt_path: str,
    config_path: str,
    out_dir: str,
    device_str: str = "cpu",
) -> dict:
    """Load Model 2 checkpoint, run warm-start analysis on the test set."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from train.cli import parse_train_config
    from train.data import build_dataloaders
    from train.modeling import create_model, init_weights

    device = torch.device(device_str)

    # Load config and build dataloaders
    cfg, _ = parse_train_config(["--config", config_path])
    splits = build_dataloaders(
        parquet_paths=cfg.data.parquet_paths,
        per_unit=cfg.data.per_unit,
        device=device,
        batch_size=1,               # one sample at a time for per-sample stats
        block_diag=cfg.model.block_diag,
        seed=cfg.run.seed,
        split_mode=cfg.data.split_mode,
        train_ratio=cfg.data.train_ratio,
        valid_ratio=cfg.data.valid_ratio,
    )

    # Build and load model
    model = create_model(
        model_name=cfg.model.name,
        d=cfg.model.d, d_hi=cfg.model.d_hi, K=cfg.model.K,
        pinn=cfg.model.pinn, gamma=cfg.model.gamma,
        v_limit=cfg.model.vlimit, use_armijo=cfg.model.use_armijo,
        dtheta_max=cfg.model.dtheta_max, dvm_frac=cfg.model.dvm_frac,
        num_attn_layers=cfg.model.num_attn_layers, device=device,
        lambda_cost=cfg.model.lambda_cost, lambda_kcl=cfg.model.lambda_kcl,
        lambda_lim=cfg.model.lambda_lim, lambda_branch=cfg.model.lambda_branch,
        lambda_v=cfg.model.lambda_v, pg_step_frac=cfg.model.pg_step_frac,
        c1_default=cfg.model.c1_default, c2_default=cfg.model.c2_default,
        pg_lim_frac=cfg.model.pg_lim_frac, s_max_pu=cfg.model.s_max_pu,
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Test set: {splits.n_test} samples")

    flat_iters, warm_iters, flat_times, warm_times = [], [], [], []

    with torch.no_grad():
        for batch in splits.test_loader:
            n_nodes_per_graph = batch["sizes"].to(device) if cfg.model.block_diag else None

            bus_type = batch["bus_type"].to(device)
            Line = batch["Lines_connected"].to(device)
            Y    = batch.get("Ybus")
            if isinstance(Y, torch.Tensor):
                Y = Y.to(device)
            Ys = batch["Y_Lines"].to(device)
            Yc = batch["Y_C_Lines"].to(device)
            Sstart = batch["S_start"].to(device)
            Vstart = batch["V_start"].to(device)

            V_pred, Pg_pred, _ = model(
                bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph,
            )

            # Convert model predictions to numpy for NR
            V_pred_np = V_pred.squeeze(0).cpu().numpy()  # (N, 2): [mag, ang]
            v_warm = V_pred_np[:, 0]
            th_warm = V_pred_np[:, 1]

            # Reconstruct Ybus and P_set from batch for NR
            Ybus_np = batch["Ybus"].squeeze(0).cpu().numpy()
            bt_np   = batch["bus_type"].squeeze(0).cpu().numpy()
            S_np    = batch["S_start"].squeeze(0).cpu().numpy()
            P_set   = S_np.real.astype(np.float32)
            Q_set   = S_np.imag.astype(np.float32)
            N_grid  = len(bt_np)

            # Flat start
            v_flat  = np.ones(N_grid, dtype=np.float32)
            th_flat = np.zeros(N_grid, dtype=np.float32)
            v_flat[bt_np == 2] = 1.02

            t0 = time.time(); n_flat = _nr_iters(Ybus_np, P_set, Q_set, bt_np, v_flat, th_flat)
            flat_times.append(time.time() - t0)
            flat_iters.append(n_flat)

            t0 = time.time(); n_warm = _nr_iters(Ybus_np, P_set, Q_set, bt_np, v_warm.astype(np.float32), th_warm.astype(np.float32))
            warm_times.append(time.time() - t0)
            warm_iters.append(n_warm)

    flat_iters = np.array(flat_iters)
    warm_iters = np.array(warm_iters)
    savings    = flat_iters - warm_iters

    results = {
        "n_samples":         int(len(flat_iters)),
        "flat_iters_mean":   float(flat_iters.mean()),
        "flat_iters_median": float(np.median(flat_iters)),
        "warm_iters_mean":   float(warm_iters.mean()),
        "warm_iters_median": float(np.median(warm_iters)),
        "savings_mean":      float(savings.mean()),
        "savings_pct":       float(savings.mean() / flat_iters.mean() * 100.0),
        "flat_ms_mean":      float(np.mean(flat_times) * 1000.0),
        "warm_ms_mean":      float(np.mean(warm_times) * 1000.0),
    }

    print("\n=== Warm-Start Analysis Results ===")
    for k, v in results.items():
        print(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    import csv
    out_csv = Path(out_dir) / "warmstart_results.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in results.items():
            w.writerow([k, v])
    print(f"\nResults saved to {out_csv}")

    # Histogram plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(flat_iters, bins=20, alpha=0.7, label="flat start", color="tab:blue")
        axes[0].hist(warm_iters, bins=20, alpha=0.7, label="warm start", color="tab:orange")
        axes[0].set_xlabel("NR iterations"); axes[0].set_ylabel("Count")
        axes[0].set_title("NR Iterations to Convergence"); axes[0].legend()
        axes[1].hist(savings, bins=20, color="tab:green", alpha=0.8)
        axes[1].axvline(0, color="k", linestyle="--")
        axes[1].set_xlabel("Iteration savings (flat - warm)"); axes[1].set_ylabel("Count")
        axes[1].set_title(f"Iteration Savings (mean={savings.mean():.1f}, {results['savings_pct']:.1f}%)")
        fig.tight_layout()
        fig.savefig(str(Path(out_dir) / "warmstart_histogram.png"), dpi=120)
        plt.close(fig)
        print(f"Plot saved to {Path(out_dir) / 'warmstart_histogram.png'}")
    except Exception as e:
        print(f"Plot failed: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model 3 warm-start analysis")
    parser.add_argument("--ckpt",   required=True, help="Path to best.ckpt from Model 2 run")
    parser.add_argument("--config", required=True, help="Config YAML used for that run")
    parser.add_argument("--out",    default="results/warmstart", help="Output directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_warmstart_analysis(args.ckpt, args.config, args.out, args.device)

"""O4 comparison: Model 1 (supervised) vs Model 2 (physics-informed).

Loads best checkpoints from both runs and evaluates them on the same test
set, then generates a comparison table + violation-breakdown plots.

Metrics reported:
    - Objective gap: (cost_pred_pv - cost_opt_pv) / |cost_opt_pv|
    - Feasibility rate (all max violations < 0.05 pu)
    - KCL L2 and L∞ norms
    - Branch max, voltage max, generator limit max violations
    - Runtime per sample

Usage:
    python -m train.opf_comparison \\
        --ckpt1 results/runs/<supervised_run>/ckpt/best.ckpt \\
        --config1 configs/opf_supervised_pignn.yaml \\
        --ckpt2 results/runs/<physics_run>/ckpt/best.ckpt \\
        --config2 configs/opf_pignn_attn_ls.yaml \\
        --out results/comparison/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch


def _load_model_for_eval(ckpt_path: str, config_path: str, device: torch.device):
    """Load a trained model ready for evaluation."""
    from train.cli import parse_train_config
    from train.modeling import create_model

    cfg, _ = parse_train_config(["--config", config_path])
    model = create_model(
        model_name="GNSMsg_EdgeSelfAttn_OPF",   # both models use same arch
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
    return model, cfg


def run_comparison(
    ckpt1: str, config1: str,
    ckpt2: str, config2: str,
    out_dir: str,
    device_str: str = "cpu",
) -> dict:
    """Evaluate both models on the same test set and produce comparison report."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from train.data import build_dataloaders
    from train.opf_loop import evaluate_test_opf
    from train.opf_supervised_loop import evaluate_test_opf_supervised

    device = torch.device(device_str)

    print("Loading Model 1 (supervised)...")
    m1, cfg1 = _load_model_for_eval(ckpt1, config1, device)

    print("Loading Model 2 (physics-informed)...")
    m2, cfg2 = _load_model_for_eval(ckpt2, config2, device)

    # Use Model 2's config for the shared test loader (same dataset)
    splits = build_dataloaders(
        parquet_paths=cfg2.data.parquet_paths,
        per_unit=cfg2.data.per_unit,
        device=device,
        batch_size=cfg2.optim.batch_size,
        block_diag=cfg2.model.block_diag,
        seed=cfg2.run.seed,
        split_mode=cfg2.data.split_mode,
        train_ratio=cfg2.data.train_ratio,
        valid_ratio=cfg2.data.valid_ratio,
    )

    print(f"\nEvaluating Model 1 on test set ({splits.n_test} samples)...")
    met1 = evaluate_test_opf_supervised(
        model=m1, test_loader=splits.test_loader, device=device,
        pinn=True, block_diag=cfg1.model.block_diag,
    )

    print("Evaluating Model 2 on test set...")
    met2 = evaluate_test_opf(
        model=m2, test_loader=splits.test_loader, device=device,
        pinn=True, block_diag=cfg2.model.block_diag,
    )

    # Build comparison table
    rows = [
        ("metric",              "Model 1 (supervised)", "Model 2 (physics)"),
        ("total_loss",          f"{met1.total_loss:.4e}", f"{met2.total_loss:.4e}"),
        ("v_mse_or_gen_cost",   f"{met1.v_mse:.4e}",     f"{met2.gen_cost:.4e}"),
        ("pg_mse_or_kcl",       f"{met1.pg_mse:.4e}",    f"{met2.kcl_loss:.4e}"),
        ("obj_gap_pct",         f"{met1.obj_gap*100:.2f}%", "N/A (no OPF labels in loop)"),
        ("cost_pred_pv",        f"{met1.cost_pred:.4f}",  "N/A"),
        ("cost_opt_pv",         f"{met1.cost_opt:.4f}",   "N/A"),
        ("feasibility_rate",    "N/A",                    f"{met2.feasibility_rate*100:.1f}%"),
        ("kcl_inf_norm",        "N/A",                    f"{met2.kcl_inf_norm:.4e}"),
        ("branch_max",          "N/A",                    f"{met2.branch_max:.4e}"),
        ("lim_max",             "N/A",                    f"{met2.lim_max:.4e}"),
        ("v_inf_norm",          "N/A",                    f"{met2.v_inf_norm:.4e}"),
        ("runtime_ms_per_sample", f"{met1.runtime_ms_per_sample:.2f}ms", f"{met2.runtime_ms_per_sample:.2f}ms"),
    ]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_csv = Path(out_dir) / "model_comparison.csv"
    with out_csv.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    print("\n=== Model Comparison (O4) ===")
    for r in rows:
        print(f"  {r[0]:35s}  {r[1]:30s}  {r[2]}")
    print(f"\nSaved to {out_csv}")

    # Violation breakdown bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        labels  = ["KCL inf-norm", "Branch max", "Gen lim max", "Voltage max"]
        m2_vals = [met2.kcl_inf_norm, met2.branch_max, met2.lim_max, met2.v_inf_norm]
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], m2_vals, 0.4, label="Model 2 (physics)", color="tab:blue")
        ax.axhline(0.05, color="red", linestyle="--", label="Feasibility threshold (0.05 pu)")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.set_ylabel("Mean max violation (pu)")
        ax.set_title("Constraint Violation Breakdown — Model 2 vs Feasibility Threshold")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(Path(out_dir) / "violation_breakdown.png"), dpi=120)
        plt.close(fig)
        print(f"Plot saved to {Path(out_dir) / 'violation_breakdown.png'}")
    except Exception as e:
        print(f"Plot failed: {e}")

    return {"model1": met1, "model2": met2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="O4: Model 1 vs Model 2 comparison")
    parser.add_argument("--ckpt1",   required=True, help="Best ckpt for Model 1")
    parser.add_argument("--config1", required=True, help="Config for Model 1")
    parser.add_argument("--ckpt2",   required=True, help="Best ckpt for Model 2")
    parser.add_argument("--config2", required=True, help="Config for Model 2")
    parser.add_argument("--out",     default="results/comparison")
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()
    run_comparison(args.ckpt1, args.config1, args.ckpt2, args.config2, args.out, args.device)

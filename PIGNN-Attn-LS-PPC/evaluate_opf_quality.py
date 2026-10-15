"""
OPF quality evaluation: cost gap (%) and constraint violations on the test set.

For each trained checkpoint this script runs the model on the OPFData test
split and reports:
  - mean / std generation cost gap vs the true optimal (%)
  - voltage band violation rate (fraction of scenarios with any violation)
  - active / reactive power balance residual (free buses)
  - generation limit violation (controllable buses)

Usage (single model):
    python evaluate_opf_quality.py \
        --ckpt results/ckpt/tinyx/pignn_supervised_case118_best.pt \
        --opfdata_root ./opfdata \
        --case_name pglib_opf_case118_ieee \
        --n_scenarios 200 \
        --out_dir results/opf_quality

Usage (all case118 models from a directory — auto-discovers *_best.pt files):
    python evaluate_opf_quality.py \
        --ckpt_dir results/ckpt/tinyx \
        --opfdata_root ./opfdata \
        --case_name pglib_opf_case118_ieee \
        --n_scenarios 200 \
        --out_dir results/opf_quality

The model configurations below must match what was used in training.
When --ckpt_dir is used, model config is inferred from the filename.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# ── model config registry ─────────────────────────────────────────────────────
# Maps checkpoint name → (case_name, model_kwargs_overrides, use_chunks)
# use_chunks=True → case500 data via make_case500_loaders (vault chunks)
# use_chunks=False → standard make_opfdata_loaders
MODEL_REGISTRY = {
    "pignn_supervised_case118": ("pglib_opf_case118_ieee", dict(),                                                        False),
    "pignn_pinn_case118":       ("pglib_opf_case118_ieee", dict(),                                                        False),
    "pignn_global_case118":     ("pglib_opf_case118_ieee", dict(use_global_attn=True, global_attn_position="after"),      False),
    "pignn_khop2_case118":      ("pglib_opf_case118_ieee", dict(attn_k_hop=2),                                            False),
    "pignn_khop3_case118":      ("pglib_opf_case118_ieee", dict(attn_k_hop=3),                                            False),
    "pignn_pcost_case118":      ("pglib_opf_case118_ieee", dict(),                                                        False),
    "pignn_phys1_mse5_case118": ("pglib_opf_case118_ieee", dict(),                                                        False),
    "pignn_lowrank16_case118":  ("pglib_opf_case118_ieee", dict(attn_rank=16),                                            False),
    "pignn_physonly_case118":   ("pglib_opf_case118_ieee", dict(),                                                        False),
    "pignn_phys5_mse1_case118": ("pglib_opf_case118_ieee", dict(),                                                        False),
    "pignn_highlim_case118":    ("pglib_opf_case118_ieee", dict(),                                                        False),
    "pignn_highlim_lr1e4_case118": ("pglib_opf_case118_ieee", dict(), False),
    "pignn_highlim_lr3e4_case118": ("pglib_opf_case118_ieee", dict(), False),
    "pignn_highlim_lr1e3_case118": ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pinn_b8_case118":       ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pinn_b32_case118":      ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pinn_cost_b8_case118":  ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pinn_cost_b32_case118": ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pinn_cost_cw1e4_case118": ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pinn_cost_cw1e2_case118": ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pinn_cost_cw1e1_case118": ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pinn_mse05_case118":      ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pinn_mse1_case118":       ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pdl_case118":             ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pdl2_case118":            ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pdl3_case118":            ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pdl4_case118":            ("pglib_opf_case118_ieee", dict(), False),
    "pignn_pdl_case500":             ("pglib_opf_case500_goc",  dict(), True),
    "pignn_pdl2_case500":            ("pglib_opf_case500_goc",  dict(), True),
    "pignn_supervised_case500": ("pglib_opf_case500_goc",  dict(),                                                        True),
    "pignn_global_case500":     ("pglib_opf_case500_goc",  dict(use_global_attn=True, global_attn_position="after"),      True),
}
DEFAULT_MODEL_KWARGS = dict(
    K=30, d=4, d_hi=16, n_heads=4, num_attn_layers=1,
    pinn=True, use_armijo=True, armijo_mode="geometric_safe",
    use_global_attn=False, global_attn_position="after", attn_k_hop=1, attn_rank=0,
)
SKIP_NAMES = {"pignn_pinn_case14", "pignn_supervised_case14"}   # unrelated experiments


def load_model(ckpt_path: str, overrides: dict = None):
    from GNSMsg_SelfAttention_armijo_opf import GNSMsg_EdgeSelfAttn_OPF
    kwargs = {**DEFAULT_MODEL_KWARGS, **(overrides or {})}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = GNSMsg_EdgeSelfAttn_OPF(**kwargs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}")
    model.eval()
    return model


@torch.no_grad()
def evaluate_one(model, test_loader, n_scenarios: int, device):
    from opf_task import opf_decision_space, opf_metrics
    from train_valid_test_pignn_opf import branch_rows_from_opfdata

    all_metrics: List[Dict] = []
    FEAS_TOL = 1e-2   # pu — threshold for "feasible" on each constraint

    for i, batch in enumerate(test_loader):
        if i >= n_scenarios:
            break
        try:
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in batch.items() if k != "hetero"}
            hb = batch["hetero"].to(device)

            space = opf_decision_space(bd, batch["sizes"], device)
            Y        = bd["Ybus"]
            S_load   = bd["S_start"]
            V0       = bd["V_start"].float()
            bus_type = bd["bus_type"].reshape(1, -1)
            ctrl     = space["Gen_controllable"].to(torch.bool)
            vmin     = space.get("Bus_vmin")
            vmax     = space.get("Bus_vmax")

            br = branch_rows_from_opfdata(hb, device)

            result = model(
                bus_type=bus_type,
                Branch_f_bus=br["Branch_f_bus"].unsqueeze(0),
                Branch_t_bus=br["Branch_t_bus"].unsqueeze(0),
                Branch_status=br["Branch_status"].unsqueeze(0),
                Branch_tau=br["Branch_tau"].unsqueeze(0),
                Branch_shift_deg=br["Branch_shift_deg"].unsqueeze(0),
                Branch_y_series_from=br["Branch_y_series_from"].unsqueeze(0),
                Branch_y_series_to=br["Branch_y_series_to"].unsqueeze(0),
                Branch_y_series_ft=br["Branch_y_series_ft"].unsqueeze(0),
                Branch_y_shunt_from=br["Branch_y_shunt_from"].unsqueeze(0),
                Branch_y_shunt_to=br["Branch_y_shunt_to"].unsqueeze(0),
                Is_trafo=br["Is_trafo"].unsqueeze(0),
                Y=Y, S=S_load, V0=V0,
                ctrl_mask=ctrl.reshape(1, -1),
                ref_mask=(bus_type == 3),
                n_nodes_per_graph=batch["sizes"].to(device),
                v_min=(vmin.reshape(1, -1).float() if vmin is not None else None),
                v_max=(vmax.reshape(1, -1).float() if vmax is not None else None),
            )

            V_pred = result[0] if isinstance(result, (tuple, list)) else result
            V_true = bd.get("V_newton")
            if V_true is None:
                V_true = bd.get("V_target", V_pred)

            m = opf_metrics(Y, V_pred, V_true, S_load, space, batch["sizes"].to(device))

            # voltage RMSE and MSE (magnitude only, over all buses)
            vp = V_pred[0, :, 0].float()
            vt = V_true[0, :, 0].float()
            err = (vp - vt) ** 2
            m["vmag_mse"]  = float(err.mean().item())
            m["vmag_rmse"] = float(err.mean().sqrt().item())

            # angle RMSE
            ap = V_pred[0, :, 1].float()
            at = V_true[0, :, 1].float()
            m["ang_rmse"] = float(((ap - at) ** 2).mean().sqrt().item())

            # joint feasibility: ALL constraints within tolerance simultaneously
            feasible = (
                m["vband_max_pu"]      < FEAS_TOL and
                m["p_limit_max_pu"]    < FEAS_TOL and
                m["q_limit_max_pu"]    < FEAS_TOL and
                m["dp_inf_pu"]         < FEAS_TOL
            )
            m["feasible"] = int(feasible)

            all_metrics.append(m)

        except Exception as e:
            print(f"  scenario {i}: ERROR {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_scenarios}]")

    return all_metrics


def aggregate(metrics: List[Dict]) -> Dict:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics if k in m]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"]  = float(np.std(vals))
    out["n_scenarios"] = len(metrics)

    # convenience: cost gap in %
    gaps = [m["cost_gap_rel"] * 100 for m in metrics]
    out["cost_gap_pct_mean"]   = float(np.mean(gaps))
    out["cost_gap_pct_std"]    = float(np.std(gaps))
    out["cost_gap_pct_median"] = float(np.median(gaps))

    # feasibility-conditioned cost gap (only on truly feasible scenarios)
    feasible_gaps = [m["cost_gap_rel"] * 100 for m in metrics if m.get("feasible", 0) == 1]
    if feasible_gaps:
        out["cost_gap_feasible_pct_mean"]   = float(np.mean(feasible_gaps))
        out["cost_gap_feasible_pct_std"]    = float(np.std(feasible_gaps))
        out["cost_gap_feasible_n"]          = len(feasible_gaps)
    else:
        out["cost_gap_feasible_pct_mean"]   = float("nan")
        out["cost_gap_feasible_pct_std"]    = float("nan")
        out["cost_gap_feasible_n"]          = 0

    # voltage violation rate: fraction of scenarios with any bus violation
    vband = [m["vband_frac"] > 1e-9 for m in metrics]
    out["vband_violation_rate"] = float(np.mean(vband))

    return out


def print_summary(name: str, agg: Dict):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    feas_gap = agg.get("cost_gap_feasible_pct_mean", float("nan"))
    feas_n   = agg.get("cost_gap_feasible_n", 0)
    print(f"  Val MSE  (vmag):   {agg['vmag_mse_mean']:.4e} ± {agg['vmag_mse_std']:.4e}")
    print(f"  Val RMSE (vmag):   {agg['vmag_rmse_mean']:.4e} ± {agg['vmag_rmse_std']:.4e}")
    print(f"  Angle RMSE:        {agg['ang_rmse_mean']:.4e} ± {agg['ang_rmse_std']:.4e}")
    print(f"  Cost gap (all):    {agg['cost_gap_pct_mean']:+.3f}% ± {agg['cost_gap_pct_std']:.3f}%  (median {agg['cost_gap_pct_median']:+.3f}%)")
    print(f"  Cost gap (feasib): {feas_gap:+.3f}%  over {feas_n}/{agg['n_scenarios']} feasible scenarios")
    print(f"  Feasibility rate:  {agg['feasible_mean']*100:.1f}% of scenarios fully feasible")
    print(f"  Voltage violations:{agg['vband_violation_rate']*100:.1f}% of scenarios")
    print(f"  Vband max (pu):    {agg['vband_max_pu_mean']:.4f} ± {agg['vband_max_pu_std']:.4f}")
    print(f"  P-balance (pu):    {agg['dp_inf_pu_mean']:.4e} ± {agg['dp_inf_pu_std']:.4e}")
    print(f"  P-limit viol (pu): {agg['p_limit_max_pu_mean']:.4e}")
    print(f"  Q-limit viol (pu): {agg['q_limit_max_pu_mean']:.4e}")
    print(f"  N scenarios:       {agg['n_scenarios']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         default="", help="single checkpoint path")
    p.add_argument("--ckpt_dir",     default="", help="directory; auto-discovers *_best.pt files")
    p.add_argument("--opfdata_root", required=True)
    p.add_argument("--case_name",    default="pglib_opf_case118_ieee")
    p.add_argument("--n_scenarios",  type=int, default=200)
    p.add_argument("--out_dir",      default="results/opf_quality")
    p.add_argument("--chunk_dir",    default="", help="for case500 chunked data")
    # manual model config overrides (used when --ckpt is passed and name not in registry)
    p.add_argument("--K",            type=int, default=30)
    p.add_argument("--d",            type=int, default=4)
    p.add_argument("--d_hi",         type=int, default=16)
    p.add_argument("--n_heads",      type=int, default=4)
    p.add_argument("--num_attn_layers", type=int, default=1)
    p.add_argument("--use_global_attn", action="store_true", default=False)
    p.add_argument("--global_attn_position", default="after")
    p.add_argument("--attn_k_hop",   type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    # build list of (name, ckpt_path, case_name, overrides, use_chunks)
    jobs = []
    if args.ckpt_dir:
        for f in sorted(Path(args.ckpt_dir).glob("*_best.pt")):
            name = f.stem.replace("_best", "")
            if any(s in name for s in SKIP_NAMES):
                continue
            reg = next((v for k, v in MODEL_REGISTRY.items() if k == name), None)
            if reg is None:
                print(f"  [skip] {name} not in registry")
                continue
            case_name, overrides, use_chunks = reg
            jobs.append((name, str(f), case_name, overrides, use_chunks))
    elif args.ckpt:
        name = Path(args.ckpt).stem.replace("_best", "")
        reg  = next((v for k, v in MODEL_REGISTRY.items() if k == name), None)
        if reg:
            case_name, overrides, use_chunks = reg
        else:
            case_name, use_chunks = args.case_name, False
            overrides = dict(K=args.K, d=args.d, d_hi=args.d_hi, n_heads=args.n_heads,
                             num_attn_layers=args.num_attn_layers,
                             use_global_attn=args.use_global_attn,
                             global_attn_position=args.global_attn_position,
                             attn_k_hop=args.attn_k_hop)
        jobs.append((name, args.ckpt, case_name, overrides, use_chunks))
    else:
        print("ERROR: provide --ckpt or --ckpt_dir"); sys.exit(1)

    # cache loaders; case500 uses chunked vault loader
    loaders: Dict[str, object] = {}
    def get_loader(case_name: str, use_chunks: bool):
        key = f"{case_name}_{'chunks' if use_chunks else 'std'}"
        if key not in loaders:
            print(f"[eval] Loading OPFData: {case_name} ({'chunks' if use_chunks else 'std'}, {args.n_scenarios} scenarios)")
            try:
                if use_chunks and args.chunk_dir:
                    from opfdata_pipeline import make_case500_loaders
                    _, _, _, tl = make_case500_loaders(
                        args.chunk_dir, batch_size=1, n_train_groups=1,
                        max_train=1, max_valid=1, max_test=args.n_scenarios, num_workers=0,
                    )
                else:
                    from opfdata_pipeline import make_opfdata_loaders
                    _, _, _, tl = make_opfdata_loaders(
                        args.opfdata_root, case_name, batch_size=1,
                        max_train=1, max_valid=1, max_test=args.n_scenarios, num_workers=0,
                    )
                loaders[key] = tl
            except Exception as e:
                print(f"  [error] could not load data for {case_name}: {e}")
                loaders[key] = None
        return loaders[key]

    all_results: Dict[str, Dict] = {}
    for name, ckpt_path, case_name, overrides, use_chunks in jobs:
        print(f"\n[eval] Model: {name}  case: {case_name}")
        print(f"       Checkpoint: {ckpt_path}")
        print(f"       Overrides:  {overrides}")
        try:
            model = load_model(ckpt_path, overrides)
        except Exception as e:
            print(f"  [error] could not load model: {e}"); continue
        test_loader = get_loader(case_name, use_chunks)
        if test_loader is None:
            print(f"  [skip] no data loader for {case_name}"); continue
        raw = evaluate_one(model, test_loader, args.n_scenarios, device)
        if not raw:
            print(f"  [warn] no scenarios evaluated for {name}"); continue
        agg = aggregate(raw)
        agg["case_name"] = case_name
        print_summary(name, agg)
        all_results[name] = agg

    # save full results split by case
    for case in set(v["case_name"] for v in all_results.values()):
        subset = {k: v for k, v in all_results.items() if v["case_name"] == case}
        out_path = os.path.join(args.out_dir, f"opf_quality_{case}.json")
        with open(out_path, "w") as f:
            json.dump(subset, f, indent=2)
        print(f"\n[eval] Results saved -> {out_path}")

    # print comparison table
    print(f"\n{'='*90}")
    print(f"{'Model':<35} {'Case':>10} {'RMSE':>10} {'Cost Gap':>10} {'Feasible':>10} {'P-balance':>12}")
    print(f"{'='*90}")
    for name, agg in all_results.items():
        case_short = agg['case_name'].replace("pglib_opf_", "").replace("_ieee","").replace("_goc","")
        print(f"{name:<35} {case_short:>10} {agg['vmag_rmse_mean']:>10.4e} "
              f"{agg['cost_gap_pct_mean']:>+9.3f}% "
              f"{agg['feasible_mean']*100:>9.1f}% "
              f"{agg['dp_inf_pu_mean']:>12.4e}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()

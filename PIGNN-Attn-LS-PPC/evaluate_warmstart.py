"""
evaluate_warmstart.py

Measures the warm-start benefit of PIGNN-Attn-LS.

Key claim for the paper:
  The model's predicted V_pred is much closer to the feasible OPF solution
  than the flat-start V_flat, so a standard Newton-Raphson power flow
  converges in far fewer iterations when initialized from V_pred.

For each test scenario this script:
  1. Gets V_pred from the model (K=30 GNN-Newton steps from flat start)
  2. Gets V_flat  (the flat start: |V|=1 pu, angle=0)
  3. Gets V_true  (the reference OPF solution from the dataset)
  4. Runs standard Newton-Raphson from V_pred → records iterations to convergence
  5. Runs standard Newton-Raphson from V_flat → records iterations to convergence
  6. Computes:
       - Initial mismatch at V_pred vs V_flat (how good is the warm start?)
       - Warm-start iterations vs cold-start iterations
       - Speed-up factor: N_cold / N_warm
       - Cost gap before and after NR correction

Usage:
    python evaluate_warmstart.py \
        --ckpt results/ckpt/tinyx/pignn_supervised_case118_best.pt \
        --opfdata_root ./opfdata \
        --case_name pglib_opf_case118_ieee \
        --n_scenarios 200 \
        --out_dir results/warmstart

Batch sweep:
    python evaluate_warmstart.py \
        --ckpt_dir results/ckpt/tinyx \
        --opfdata_root ./opfdata \
        --n_scenarios 200 \
        --out_dir results/warmstart
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from evaluate_opf_quality import (
    MODEL_REGISTRY, SKIP_NAMES, DEFAULT_MODEL_KWARGS, load_model
)

NR_TOL      = 1e-6   # convergence tolerance (pu)
NR_MAX_ITER = 50     # max Newton-Raphson iterations


# ── Newton-Raphson power flow ─────────────────────────────────────────────────

def _power_mismatch(V: torch.Tensor, Y: torch.Tensor,
                    P_spec: torch.Tensor, Q_spec: torch.Tensor,
                    ref_mask: torch.Tensor, pq_mask: torch.Tensor
                    ) -> Tuple[torch.Tensor, float]:
    """Returns mismatch vector and max|mismatch|."""
    I   = Y @ V
    S   = V * I.conj()
    dP  = (P_spec - S.real)[~ref_mask]
    dQ  = (Q_spec - S.imag)[pq_mask]
    mis = torch.cat([dP, dQ])
    return mis, float(mis.abs().max().item())


def _build_jacobian(V: torch.Tensor, Y: torch.Tensor,
                    P: torch.Tensor, Q: torch.Tensor,
                    non_slack_idx: torch.Tensor,
                    pq_idx: torch.Tensor) -> torch.Tensor:
    """Polar-form Newton-Raphson Jacobian (H N / J L blocks)."""
    vmag = V.abs()
    ang  = V.angle()
    G    = Y.real
    B    = Y.imag
    N    = vmag.shape[0]

    # Angle difference matrix (N,N)
    dang   = ang.unsqueeze(1) - ang.unsqueeze(0)
    cos_d  = dang.cos()
    sin_d  = dang.sin()
    vmij   = vmag.unsqueeze(1) * vmag.unsqueeze(0)  # |V_i||V_j|

    # H and N matrices (full N×N, diagonals corrected)
    H_mat = vmij * (G * sin_d - B * cos_d)
    H_mat.diagonal().copy_(-Q - B.diagonal() * vmag ** 2)

    Nmat = vmij * (G * cos_d + B * sin_d)
    Nmat.diagonal().copy_(P + G.diagonal() * vmag ** 2)

    # J = -N (off-diag), diagonal = P_i - G_ii|V_i|²
    J_mat = -Nmat.clone()
    J_mat.diagonal().copy_(P - G.diagonal() * vmag ** 2)

    # L = H (off-diag), diagonal = Q_i - B_ii|V_i|²
    L_mat = H_mat.clone()
    L_mat.diagonal().copy_(Q - B.diagonal() * vmag ** 2)

    # Submatrices for reduced system
    ns  = non_slack_idx
    pq  = pq_idx
    n_ns = len(ns)
    n_pq = len(pq)
    Jac  = torch.zeros(n_ns + n_pq, n_ns + n_pq, dtype=torch.float64)
    Jac[:n_ns, :n_ns] = H_mat[ns][:, ns]
    Jac[:n_ns, n_ns:] = Nmat[ns][:, pq]
    Jac[n_ns:, :n_ns] = J_mat[pq][:, ns]
    Jac[n_ns:, n_ns:] = L_mat[pq][:, pq]
    return Jac


def newton_pf(Y: torch.Tensor, P_spec: torch.Tensor, Q_spec: torch.Tensor,
              V_init: torch.Tensor, bus_type: torch.Tensor,
              max_iter: int = NR_MAX_ITER, tol: float = NR_TOL
              ) -> Tuple[torch.Tensor, int, bool, List[float]]:
    """
    Standard Newton-Raphson power flow (polar form, double precision).

    Returns:
        V_final    : converged complex voltage vector
        n_iters    : number of NR iterations performed
        converged  : True if |mismatch| < tol
        mis_history: list of max|mismatch| at each iteration
    """
    V   = V_init.squeeze().to(torch.complex128)
    Y64 = Y.squeeze().to(torch.complex128)
    P64 = P_spec.squeeze().to(torch.float64)
    Q64 = Q_spec.squeeze().to(torch.float64)

    ref_mask      = (bus_type.squeeze() == 3)
    pq_mask       = (bus_type.squeeze() == 1)
    non_slack_idx = torch.where(~ref_mask)[0]
    pq_idx        = torch.where(pq_mask)[0]

    vmag = V.abs().clone()
    ang  = V.angle().clone()
    mis_hist: List[float] = []

    for it in range(max_iter):
        V  = vmag * torch.exp(1j * ang)
        mis, max_mis = _power_mismatch(V, Y64, P64, Q64, ref_mask, pq_mask)
        mis_hist.append(max_mis)

        if max_mis < tol:
            return V, it, True, mis_hist

        I   = Y64 @ V
        S   = V * I.conj()
        Jac = _build_jacobian(V, Y64, S.real, S.imag, non_slack_idx, pq_idx)

        try:
            dx = torch.linalg.solve(Jac, mis.to(torch.float64))
        except torch.linalg.LinAlgError:
            break  # singular Jacobian — stop

        n_ns = len(non_slack_idx)
        ang[non_slack_idx] += dx[:n_ns]
        vmag[pq_idx]       += dx[n_ns:] * vmag[pq_idx]

    V = vmag * torch.exp(1j * ang)
    _, max_mis = _power_mismatch(V, Y64, P64, Q64, ref_mask, pq_mask)
    mis_hist.append(max_mis)
    return V, max_iter, False, mis_hist


# ── main evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_warmstart(model, test_loader, n_scenarios: int, device
                       ) -> List[Dict]:
    from opf_task import opf_decision_space, dispatch_cost
    from train_valid_test_pignn_opf import branch_rows_from_opfdata

    results = []
    for i, batch in enumerate(test_loader):
        if i >= n_scenarios:
            break
        try:
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in batch.items() if k != "hetero"}
            hb = batch["hetero"].to(device)

            space    = opf_decision_space(bd, batch["sizes"], device)
            Y        = bd["Ybus"]
            S_load   = bd["S_start"]
            V0       = bd["V_start"].float()
            bus_type = bd["bus_type"].reshape(1, -1)
            ctrl     = space["Gen_controllable"].to(torch.bool)
            vmin     = space.get("Bus_vmin")
            vmax     = space.get("Bus_vmax")

            br = branch_rows_from_opfdata(hb, device)

            # ── 1. Model prediction (warm start) ─────────────────────────────
            result  = model(
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
            V_true = bd.get("V_newton", bd.get("V_target", V_pred))

            # Convert to complex (magnitude, angle) → complex polar
            V_pred_np = V_pred[0].cpu()  # (N, 2): col0=vmag, col1=angle_rad
            V_true_np = V_true[0].cpu()
            V0_np     = V0[0].cpu()

            to_complex = lambda t: (t[:, 0] * torch.exp(1j * t[:, 1])).to(torch.complex128)
            Vc_pred = to_complex(V_pred_np)
            Vc_true = to_complex(V_true_np)
            Vc_flat = to_complex(V0_np)

            # Y-bus: may be (1,N,N) or (N,N), complex128
            Y_bus   = Y[0].cpu() if Y.dim() == 3 else Y.cpu()

            # ── 2. Scheduled injections from V_true ───────────────────────────
            Y64   = Y_bus.to(torch.complex128)
            I_true = Y64 @ Vc_true
            S_inj  = Vc_true * I_true.conj()
            P_spec = S_inj.real
            Q_spec = S_inj.imag

            bt = bus_type.squeeze().cpu()

            # ── 3. Initial mismatch (no NR iterations) ────────────────────────
            ref_mask = (bt == 3)
            pq_mask  = (bt == 1)

            def mis_val(Vc):
                _, m = _power_mismatch(Vc.to(torch.complex128), Y64, P_spec, Q_spec, ref_mask, pq_mask)
                return m

            mis_pred = mis_val(Vc_pred)   # warm start mismatch
            mis_flat = mis_val(Vc_flat)   # cold start mismatch
            mis_true = mis_val(Vc_true)   # reference (should be ~0)

            # ── 4. Newton-Raphson warm start (from V_pred) ────────────────────
            V_warm, n_warm, conv_warm, hist_warm = newton_pf(
                Y64, P_spec, Q_spec, Vc_pred, bt)

            # ── 5. Newton-Raphson cold start (from V_flat) ────────────────────
            V_cold, n_cold, conv_cold, hist_cold = newton_pf(
                Y64, P_spec, Q_spec, Vc_flat, bt)

            # ── 6. Cost gap after NR correction ───────────────────────────────
            def to_model_fmt(Vc):
                vmag = Vc.abs().float()
                ang  = Vc.angle().float()
                return torch.stack([vmag, ang], dim=-1).unsqueeze(0).to(device)

            def cost_gap(Vc):
                try:
                    from opf_task import dispatch_cost, opf_residual_and_violation
                    V_fmt   = to_model_fmt(Vc)
                    V_t_fmt = V_true.to(device) if V_true.dim() == 3 else V_true.unsqueeze(0).to(device)
                    parts_p = opf_residual_and_violation(Y.to(device), V_fmt,   S_load.to(device), space)
                    parts_t = opf_residual_and_violation(Y.to(device), V_t_fmt, S_load.to(device), space)
                    cost_pred = float(dispatch_cost(parts_p["gen_p"], space).item())
                    cost_true = float(dispatch_cost(parts_t["gen_p"], space).item())
                    gap = (cost_pred - cost_true) / (abs(cost_true) + 1e-9) * 100
                except Exception:
                    gap = float("nan")
                return gap

            gap_pred  = cost_gap(Vc_pred)
            gap_warm  = cost_gap(V_warm)
            gap_cold  = cost_gap(V_cold) if conv_cold else float("nan")

            results.append({
                "scenario":    i,
                # mismatch before any NR
                "mis_pred":    mis_pred,
                "mis_flat":    mis_flat,
                "mis_true":    mis_true,
                "mis_ratio":   mis_pred / (mis_flat + 1e-12),  # <1 → warm start better
                # NR iterations
                "n_warm":      n_warm,
                "n_cold":      n_cold,
                "conv_warm":   int(conv_warm),
                "conv_cold":   int(conv_cold),
                "speedup":     n_cold / (n_warm + 1e-3),
                # cost gaps
                "gap_pred_pct":   gap_pred,
                "gap_warm_pct":   gap_warm,
                "gap_cold_pct":   gap_cold,
            })

        except Exception as e:
            print(f"  scenario {i}: ERROR {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_scenarios}]  "
                  f"n_warm={results[-1]['n_warm']}  "
                  f"n_cold={results[-1]['n_cold']}  "
                  f"mis_ratio={results[-1]['mis_ratio']:.3f}")

    return results


def print_warmstart_summary(name: str, results: List[Dict]):
    arr = lambda k: np.array([r[k] for r in results if not np.isnan(r[k])])

    n_warm = arr("n_warm"); n_cold = arr("n_cold")
    conv_w = arr("conv_warm"); conv_c = arr("conv_cold")
    ratio  = arr("mis_ratio")
    su     = arr("speedup")

    gap_p = arr("gap_pred_pct")
    gap_w = arr("gap_warm_pct")

    print(f"\n{'='*65}")
    print(f"  WARM-START RESULTS — {name}")
    print(f"{'='*65}")
    print(f"  N scenarios evaluated:   {len(results)}")
    print(f"  NR convergence (warm):   {conv_w.mean()*100:.1f}%  avg iters = {n_warm.mean():.1f}")
    print(f"  NR convergence (cold):   {conv_c.mean()*100:.1f}%  avg iters = {n_cold.mean():.1f}")
    print(f"  Speed-up factor:         {su.mean():.2f}×  (cold/warm iters)")
    print(f"  Mismatch ratio (W/C):    {ratio.mean():.4f}  (1.0 = no improvement)")
    print(f"  Cost gap before NR:      {gap_p.mean():+.3f}%")
    print(f"  Cost gap after warm NR:  {gap_w.mean():+.3f}%")
    print(f"{'='*65}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         default="")
    p.add_argument("--ckpt_dir",     default="")
    p.add_argument("--opfdata_root", required=True)
    p.add_argument("--case_name",    default="pglib_opf_case118_ieee")
    p.add_argument("--n_scenarios",  type=int, default=200)
    p.add_argument("--out_dir",      default="results/warmstart")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    # Build job list
    jobs = []
    if args.ckpt_dir:
        for f in sorted(Path(args.ckpt_dir).glob("*_best.pt")):
            name = f.stem.replace("_best", "")
            if any(s in name for s in SKIP_NAMES):
                continue
            reg = next(((c, v) for k, (c, v) in MODEL_REGISTRY.items() if k == name), None)
            if reg is None:
                continue
            case_name, overrides = reg
            jobs.append((name, str(f), case_name, overrides))
    elif args.ckpt:
        name = Path(args.ckpt).stem.replace("_best", "")
        reg  = next(((c, v) for k, (c, v) in MODEL_REGISTRY.items() if k == name), None)
        case_name, overrides = reg if reg else (args.case_name, {})
        jobs.append((name, args.ckpt, case_name, overrides))
    else:
        print("ERROR: provide --ckpt or --ckpt_dir"); sys.exit(1)

    # Cache loaders
    loaders = {}
    def get_loader(case_name):
        if case_name not in loaders:
            from opfdata_pipeline import make_opfdata_loaders
            _, _, _, tl = make_opfdata_loaders(
                args.opfdata_root, case_name, batch_size=1,
                max_train=1, max_valid=1, max_test=args.n_scenarios, num_workers=0,
            )
            loaders[case_name] = tl
        return loaders[case_name]

    all_summary = {}
    for name, ckpt_path, case_name, overrides in jobs:
        print(f"\n[warmstart] {name}  ({case_name})")
        try:
            model = load_model(ckpt_path, overrides)
        except Exception as e:
            print(f"  [error] {e}"); continue

        loader  = get_loader(case_name)
        results = evaluate_warmstart(model, loader, args.n_scenarios, device)
        if not results:
            print("  [warn] no results"); continue

        print_warmstart_summary(name, results)

        out_path = os.path.join(args.out_dir, f"warmstart_{name}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  saved → {out_path}")

        # aggregate for comparison table
        arr = lambda k: np.array([r[k] for r in results if not np.isnan(r[k])])
        all_summary[name] = {
            "n_warm_mean":    float(arr("n_warm").mean()),
            "n_cold_mean":    float(arr("n_cold").mean()),
            "speedup_mean":   float(arr("speedup").mean()),
            "mis_ratio_mean": float(arr("mis_ratio").mean()),
            "conv_warm_pct":  float(arr("conv_warm").mean() * 100),
            "conv_cold_pct":  float(arr("conv_cold").mean() * 100),
            "gap_pred_mean":  float(arr("gap_pred_pct").mean()),
            "gap_warm_mean":  float(arr("gap_warm_pct").mean()),
        }

    # Comparison table
    print(f"\n{'='*90}")
    print(f"{'Model':<35} {'NR(warm)':>9} {'NR(cold)':>9} {'Speed-up':>9} "
          f"{'Mis ratio':>10} {'Gap(pred)':>10} {'Gap(NR)':>9}")
    print(f"{'='*90}")
    for name, s in all_summary.items():
        print(f"{name:<35} {s['n_warm_mean']:>9.1f} {s['n_cold_mean']:>9.1f} "
              f"{s['speedup_mean']:>9.2f}× {s['mis_ratio_mean']:>10.4f} "
              f"{s['gap_pred_mean']:>+9.3f}% {s['gap_warm_mean']:>+8.3f}%")
    print(f"{'='*90}")

    summary_path = os.path.join(args.out_dir, "warmstart_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\n[warmstart] Summary saved → {summary_path}")


if __name__ == "__main__":
    main()

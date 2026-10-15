"""
Per-bus P-limit violation analysis for PDL-v6 on the case118 test set.

For each test scenario, computes which ctrl buses have gen_p > gen_p_max
(upper P-limit breach) and accumulates across scenarios:
  - violation frequency  : fraction of scenarios where bus violates
  - mean violation       : mean of max(0, gen_p - gen_p_max) over all scenarios
  - max violation        : worst-case violation across all scenarios

Also extracts attention weights for violated vs. feasible scenarios to test
whether the model mis-attends on scenarios that end up infeasible.

Usage (run from PIGNN-Attn-LS-PPC/):
    python analyze_violations_pdl6.py \
        --ckpt results/ckpt/tinyx/pignn_pdl6_case118_best_feas.pt \
        --opfdata_root ./opfdata \
        --out_dir results/violations/pdl6

Outputs:
    violation_stats.pt   -- dict: freq[n_ctrl], mean_viol[n_ctrl], max_viol[n_ctrl]
    violation_stats.json -- human-readable sorted ranking of worst buses
    attn_feas.pt         -- mean attention weights on feasible scenarios
    attn_viol.pt         -- mean attention weights on violated scenarios
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import numpy as np


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path, K, d, d_hi, n_heads):
    from GNSMsg_SelfAttention_armijo_opf import GNSMsg_EdgeSelfAttn_OPF
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = GNSMsg_EdgeSelfAttn_OPF(
        d=d, d_hi=d_hi, K=K, n_heads=n_heads, num_attn_layers=1,
        pinn=True, use_armijo=True, armijo_mode="geometric_safe",
        use_global_attn=False,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_batch(model, batch, device):
    from opf_task import opf_decision_space, opf_residual_and_violation
    from train_valid_test_pignn_opf import branch_rows_from_opfdata

    bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
          for k, v in batch.items() if k != "hetero"}
    hb = batch["hetero"].to(device)

    space  = opf_decision_space(bd, batch["sizes"], device)
    Y      = bd["Ybus"]
    S_load = bd["S_start"]
    V0     = bd["V_start"].float()
    bus_type = bd["bus_type"].reshape(1, -1)
    ctrl   = space["Gen_controllable"].to(torch.bool)
    vmin   = space.get("Bus_vmin")
    vmax   = space.get("Bus_vmax")
    br     = branch_rows_from_opfdata(hb, device)

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
        return_attn=True,
    )

    V, _, attn_list, edge_index = result
    parts = opf_residual_and_violation(Y, V, S_load, space)
    return parts, attn_list, edge_index, space, batch["sizes"]


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--opfdata_root", required=True)
    p.add_argument("--case_name",    default="pglib_opf_case118_ieee")
    p.add_argument("--n_scenarios",  type=int, default=0,
                   help="0 = full test set")
    p.add_argument("--out_dir",      default="results/violations/pdl6")
    p.add_argument("--K",   type=int, default=30)
    p.add_argument("--d",   type=int, default=4)
    p.add_argument("--d_hi",type=int, default=16)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1,
                   help="1 = one scenario at a time for clean per-bus stats")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[viol] loading model from {args.ckpt}")
    model = load_model(args.ckpt, args.K, args.d, args.d_hi, args.n_heads)
    model.to(device)

    # ── data loader ───────────────────────────────────────────────────────────
    from opfdata_pipeline import make_opfdata_loaders

    _, _, _, loader = make_opfdata_loaders(
        root=args.opfdata_root,
        case_name=args.case_name,
        batch_size=args.batch_size,
        num_groups=1,
    )
    print(f"[viol] running on test set (batch_size={args.batch_size})")

    # ── accumulators ──────────────────────────────────────────────────────────
    # We discover n_nodes from first batch; pre-allocate after
    freq_viol  = None   # (n_nodes,) — count of violated scenarios
    sum_viol   = None   # (n_nodes,) — sum of violation magnitude
    max_viol   = None   # (n_nodes,) — worst violation seen
    n_total    = 0

    # attention accumulators (edge-level, averaged over K steps)
    attn_feas_sum  = None
    attn_viol_sum  = None
    n_feas_sc = 0
    n_viol_sc = 0
    edge_index_saved = None

    _THRESH = 0.01  # feasibility threshold (pu)

    for batch in loader:
        parts, attn_list, edge_index, space, sizes = run_batch(model, batch, device)

        # n_nodes for this case (single scenario, batch_size=1)
        n_nodes = int(sizes[0].item())
        B = sizes.numel()   # should be 1

        # ── initialise accumulators on first batch ─────────────────────────
        if freq_viol is None:
            freq_viol = torch.zeros(n_nodes, device=device)
            sum_viol  = torch.zeros(n_nodes, device=device)
            max_viol  = torch.zeros(n_nodes, device=device)

        # vp: P-limit violation per bus, shape (B*n_nodes,), already normalised pu
        # 0 for non-ctrl buses; positive where gen_p > gen_p_max
        vp   = parts["vp"].float()                             # (B*n_nodes,)
        ctrl = space["Gen_controllable"].to(torch.bool)        # (B*n_nodes,)

        # per-scenario loop
        offset = 0
        for i in range(B):
            sc_vp   = vp[offset:offset + n_nodes]              # (n_nodes,)
            sc_ctrl = ctrl[offset:offset + n_nodes]

            # use vp directly — already the P-limit violation in pu
            upper_viol = sc_vp.clamp(min=0.0)

            freq_viol += (upper_viol > _THRESH).float()
            sum_viol  += upper_viol
            max_viol   = torch.maximum(max_viol, upper_viol)

            # scenario-level feasibility (use vp from parts)
            sc_vp = parts["vp"][offset:offset + n_nodes][sc_ctrl]
            sc_vq = parts["vq"][offset:offset + n_nodes][sc_ctrl]
            sc_dp = parts["dp"][offset:offset + n_nodes][~sc_ctrl] if (~sc_ctrl).any() else torch.zeros(1, device=device)
            sc_dq = parts["dq"][offset:offset + n_nodes][~sc_ctrl] if (~sc_ctrl).any() else torch.zeros(1, device=device)
            worst = max(
                sc_vp.max().item() if sc_vp.numel() else 0,
                sc_vq.max().item() if sc_vq.numel() else 0,
                sc_dp.abs().max().item() if sc_dp.numel() else 0,
                sc_dq.abs().max().item() if sc_dq.numel() else 0,
            )
            feasible = worst < _THRESH

            # attention: mean over K steps → (E,)
            if attn_list:
                attn_mean = torch.stack(attn_list, dim=0).mean(dim=0)  # (E,)
                if edge_index_saved is None:
                    edge_index_saved = edge_index.cpu()
                # skip if edge count mismatches (variable last batch)
                if attn_feas_sum is not None and attn_mean.shape != attn_feas_sum.shape:
                    pass
                elif attn_viol_sum is not None and attn_mean.shape != attn_viol_sum.shape:
                    pass
                elif feasible:
                    if attn_feas_sum is None:
                        attn_feas_sum = torch.zeros_like(attn_mean)
                    attn_feas_sum += attn_mean
                    n_feas_sc += 1
                else:
                    if attn_viol_sum is None:
                        attn_viol_sum = torch.zeros_like(attn_mean)
                    attn_viol_sum += attn_mean
                    n_viol_sc += 1

            offset += n_nodes
        n_total += B

    print(f"[viol] processed {n_total} scenarios | feasible={n_feas_sc} | violated={n_viol_sc}")

    # ── compute final stats ───────────────────────────────────────────────────
    freq_viol = (freq_viol / n_total).cpu()   # fraction
    mean_viol = (sum_viol  / n_total).cpu()
    max_viol  = max_viol.cpu()

    # sorted ranking: worst buses first by violation frequency
    order = torch.argsort(freq_viol, descending=True)
    ranking = []
    for rank, bus_0 in enumerate(order.tolist()):
        fq = freq_viol[bus_0].item()
        if fq < 1e-6:
            break  # no more violating buses
        ranking.append({
            "rank":       rank + 1,
            "bus_1idx":   bus_0 + 1,   # 1-indexed for MATPOWER convention
            "freq":       round(fq, 4),
            "mean_viol":  round(mean_viol[bus_0].item(), 6),
            "max_viol":   round(max_viol[bus_0].item(), 6),
        })

    print(f"\n[viol] Top violating buses (1-indexed, P-limit breach > {_THRESH} pu):")
    print(f"  {'rank':>4}  {'bus':>5}  {'freq':>8}  {'mean_viol':>10}  {'max_viol':>10}")
    for r in ranking[:20]:
        print(f"  {r['rank']:>4}  {r['bus_1idx']:>5}  {r['freq']:>8.1%}  "
              f"{r['mean_viol']:>10.4f}  {r['max_viol']:>10.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    torch.save({"freq": freq_viol, "mean": mean_viol, "max": max_viol},
               os.path.join(args.out_dir, "violation_stats.pt"))

    with open(os.path.join(args.out_dir, "violation_stats.json"), "w") as f:
        json.dump({"n_scenarios": n_total, "n_feasible": n_feas_sc,
                   "n_violated": n_viol_sc, "ranking": ranking}, f, indent=2)

    if attn_feas_sum is not None and n_feas_sc > 0:
        torch.save((attn_feas_sum / n_feas_sc).cpu(),
                   os.path.join(args.out_dir, "attn_feas.pt"))
    if attn_viol_sum is not None and n_viol_sc > 0:
        torch.save((attn_viol_sum / n_viol_sc).cpu(),
                   os.path.join(args.out_dir, "attn_viol.pt"))
    if edge_index_saved is not None:
        torch.save(edge_index_saved, os.path.join(args.out_dir, "edge_index.pt"))

    print(f"\n[viol] results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

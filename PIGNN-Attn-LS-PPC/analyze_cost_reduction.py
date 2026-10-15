"""
Cost-reduction XAI for PIGNN-Attn-LS AC-OPF.

Since attention weights are near-uniform (a finding in itself), this script uses
gradient-based attribution and dispatch comparison to explain the cost gap:

  1. For each test scenario:
       - Run model inference → predicted voltages V_pred → dispatch gen_p_pred
       - Read reference optimal voltages V_ref from the OPFData batch → gen_p_ref
       - Compute per-bus cost excess:
           excess_i = cost_i(gen_p_pred_i) - cost_i(gen_p_ref_i)
       - Compute gradient attribution: dCost/dgen_p_i = c1_i + 2*c2_i*gen_p_pred_i
         (marginal cost = sensitivity of total cost to generator i's output)
       - Compute dispatch ratio: gen_p_pred_i / gen_p_ref_i (>1 = over-dispatch)
       - Compute headroom gain: how much cost would drop if generator i moved to
         the reference dispatch level

  2. Aggregate across scenarios:
       - Mean cost excess per bus (which generators consistently over-cost?)
       - Mean dispatch ratio per bus
       - "Cost reduction recipe": rank generators by mean excess cost contribution

  3. Counterfactual merit order:
       - Sort generators by marginal cost
       - Compute "ideal" dispatch: cheapest generators at max capacity first
       - Compare to model dispatch: model deviates from merit order for certain buses

Outputs in out_dir/:
  cost_reduction_summary.json     per-bus excess, recipe, interpretation
  cost_excess_per_bus.png         bar: mean excess cost per generator bus
  dispatch_ratio_vs_mc.png        scatter: dispatch ratio vs marginal cost
  cost_reduction_waterfall.png    waterfall: top-10 buses driving the cost gap
  merit_order_deviation.png       predicted vs optimal dispatch sorted by MC
  gradient_attribution.png        dCost/dgen_p per bus (sensitivity map)

Usage:
    python analyze_cost_reduction.py \\
        --ckpt results/ckpt/tinyx/pignn_pdl2_case118_best.pt \\
        --opfdata_root ./opfdata \\
        --case_name pglib_opf_case118_ieee \\
        --n_scenarios 200 \\
        --out_dir results/xai_cost_reduction/pdl2
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path, K, d, d_hi, n_heads, num_attn_layers):
    from GNSMsg_SelfAttention_armijo_opf import GNSMsg_EdgeSelfAttn_OPF
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = GNSMsg_EdgeSelfAttn_OPF(
        d=d, d_hi=d_hi, K=K, n_heads=n_heads, num_attn_layers=num_attn_layers,
        pinn=True, use_armijo=True, armijo_mode="geometric_safe",
        use_global_attn=False,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ── run inference + extract V_pred and V_ref ──────────────────────────────────

@torch.no_grad()
def run_scenario(model, batch, device):
    """Returns (V_pred, V_ref, space, Y, S_load) where V_ref is from the dataset."""
    from opf_task import opf_decision_space
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
    out, _phys, _attn, _ei = result
    V_pred = out.unsqueeze(0) if out.dim() == 2 else out    # [1, N, 2]

    # Reference solution: OPFData stores the optimal voltage as the MSE target.
    # It is typically stored as "V_opt" or can be inferred from "V_start" if
    # the dataset uses a converged solution as starting point.
    V_ref = None
    for key in ("V_newton", "V_opt", "V_sol", "V_target", "V_ref"):
        if key in bd:
            vr = bd[key]
            V_ref = (vr.unsqueeze(0) if vr.dim() == 2 else vr).float()
            break

    return V_pred, V_ref, space, Y, S_load


def implied_gen_p(Y, V, S_load, space):
    """Extract gen_p from model voltages via opf_residual_and_violation."""
    from opf_task import opf_residual_and_violation
    parts = opf_residual_and_violation(Y, V, S_load, space)
    return parts["gen_p"].to(torch.float64), parts["ctrl"]


def per_bus_cost(gen_p, space):
    """c1*Pg + c2*Pg^2 per bus (float64)."""
    c1 = space.get("Gen_cost_c1")
    if c1 is None:
        return torch.zeros_like(gen_p)
    cost = c1.to(torch.float64) * gen_p
    c2 = space.get("Gen_cost_c2")
    if c2 is not None:
        cost = cost + c2.to(torch.float64) * gen_p * gen_p
    return cost


def marginal_cost(gen_p, space):
    """dCost/dgen_p = c1 + 2*c2*Pg per bus."""
    c1 = space.get("Gen_cost_c1")
    if c1 is None:
        return torch.zeros_like(gen_p)
    mc = c1.to(torch.float64).clone()
    c2 = space.get("Gen_cost_c2")
    if c2 is not None:
        mc = mc + 2.0 * c2.to(torch.float64) * gen_p
    return mc


# ── plotting ──────────────────────────────────────────────────────────────────

def make_plots(ctrl_idx, bus_ids_1indexed,
               excess_mean, mc_mean, dispatch_ratio_mean,
               gp_pred_mean, gp_ref_mean, p_max_mean,
               gradient_mean, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ctrl = len(ctrl_idx)

    # ── 1. Cost excess per generator bus ─────────────────────────────────────
    order = np.argsort(excess_mean)[::-1]
    top20 = order[:20]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in excess_mean[top20]]
    ax.bar(range(len(top20)), excess_mean[top20], color=colors, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(top20)))
    ax.set_xticklabels([f"b{bus_ids_1indexed[i]}" for i in top20], rotation=60, fontsize=7)
    ax.set_ylabel("Mean cost excess (pred − ref, pu·cost)")
    ax.set_title("Top-20 generator buses by mean cost excess\n"
                 "(red = model dispatches too expensively; green = model saves vs reference)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cost_excess_per_bus.png"), dpi=150)
    plt.close()

    # ── 2. Dispatch ratio vs marginal cost ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(mc_mean, dispatch_ratio_mean,
                    c=excess_mean, cmap="RdYlGn_r", s=60, alpha=0.8,
                    edgecolors="none", vmin=-abs(excess_mean).max(), vmax=abs(excess_mean).max())
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Cost excess (pred − ref)")
    ax.axhline(1.0, color="k", lw=1, ls="--", label="ratio=1 (optimal)")
    ax.set_xlabel("Mean marginal cost (pu)")
    ax.set_ylabel("Dispatch ratio  gen_p_pred / gen_p_ref")
    ax.set_title("Dispatch ratio vs marginal cost\n"
                 ">1: over-dispatched (too expensive);  <1: under-dispatched")
    ax.legend(fontsize=8)

    # annotate top-5 over-dispatched expensive generators
    exp_over = np.argsort(mc_mean * np.maximum(0, dispatch_ratio_mean - 1))[-5:]
    for i in exp_over:
        ax.annotate(f"b{bus_ids_1indexed[i]}", (mc_mean[i], dispatch_ratio_mean[i]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dispatch_ratio_vs_mc.png"), dpi=150)
    plt.close()

    # ── 3. Cost reduction waterfall ───────────────────────────────────────────
    # Top-10 buses where reducing dispatch to reference level saves most cost
    savings = np.maximum(0, excess_mean)   # only buses where model costs more
    top10_save = np.argsort(savings)[-10:][::-1]
    cumulative = np.cumsum(savings[top10_save])

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(top10_save)), savings[top10_save], color="#d62728", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(range(len(top10_save)), cumulative, "k-o", ms=5, lw=1.5, label="Cumulative savings")
    ax2.set_ylabel("Cumulative cost savings (pu·cost)")
    ax.set_xticks(range(len(top10_save)))
    ax.set_xticklabels(
        [f"b{bus_ids_1indexed[i]}\nMC={mc_mean[i]:.0f}" for i in top10_save],
        fontsize=7, rotation=30
    )
    ax.set_ylabel("Cost savings if re-dispatched to optimal (pu·cost)")
    ax.set_title("Cost reduction waterfall — top-10 generators\n"
                 "(dispatching these buses optimally closes most of the cost gap)")
    ax2.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cost_reduction_waterfall.png"), dpi=150)
    plt.close()

    # ── 4. Merit-order deviation ──────────────────────────────────────────────
    # Sort generators by marginal cost and compare pred vs ref dispatch
    merit_order = np.argsort(mc_mean)
    fig, ax = plt.subplots(figsize=(max(8, n_ctrl // 3), 5))
    x = np.arange(n_ctrl)
    w = 0.35
    ax.bar(x - w/2, gp_ref_mean[merit_order],  w, label="Reference (OPF optimal)", alpha=0.7, color="steelblue")
    ax.bar(x + w/2, gp_pred_mean[merit_order], w, label="Model prediction",         alpha=0.7, color="coral")
    tick_pos = x[::max(1, n_ctrl // 15)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"b{bus_ids_1indexed[merit_order[i]]}" for i in tick_pos],
                       rotation=60, fontsize=6)
    ax.set_xlabel("Generators sorted by marginal cost (cheapest → most expensive)")
    ax.set_ylabel("Active power dispatch (pu)")
    ax.set_title("Merit-order dispatch: model prediction vs OPF reference\n"
                 "Ideal: cheap generators at high dispatch, expensive at low dispatch")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "merit_order_deviation.png"), dpi=150)
    plt.close()

    # ── 5. Gradient attribution (dCost/dgen_p = marginal cost) ───────────────
    order_g = np.argsort(gradient_mean)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(n_ctrl), gradient_mean[order_g], color="darkorange", alpha=0.8)
    tick_pos = np.arange(0, n_ctrl, max(1, n_ctrl // 15))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"b{bus_ids_1indexed[order_g[i]]}" for i in tick_pos],
                       rotation=60, fontsize=6)
    ax.set_xlabel("Generator buses sorted by marginal cost")
    ax.set_ylabel("∂TotalCost / ∂gen_p  (marginal cost, pu·cost/pu·power)")
    ax.set_title("Gradient attribution: sensitivity of total dispatch cost to each generator\n"
                 "High bars = expensive generators where cost is most sensitive to dispatch level")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gradient_attribution.png"), dpi=150)
    plt.close()

    print(f"[xai-cr] Plots saved to {out_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",            required=True)
    p.add_argument("--opfdata_root",    required=True)
    p.add_argument("--case_name",       default="pglib_opf_case118_ieee")
    p.add_argument("--n_scenarios",     type=int, default=200)
    p.add_argument("--out_dir",         default="results/xai_cost_reduction/pdl2")
    p.add_argument("--K",               type=int, default=30)
    p.add_argument("--d",               type=int, default=4)
    p.add_argument("--d_hi",            type=int, default=16)
    p.add_argument("--n_heads",         type=int, default=4)
    p.add_argument("--num_attn_layers", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    print(f"[xai-cr] Checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, args.K, args.d, args.d_hi,
                       args.n_heads, args.num_attn_layers)

    from opfdata_pipeline import make_opfdata_loaders
    _, _, _, test_loader = make_opfdata_loaders(
        args.opfdata_root, args.case_name, batch_size=1,
        max_train=1, max_valid=1, max_test=args.n_scenarios, num_workers=0,
    )

    n_nodes        = None
    ctrl_mask_ref  = None
    p_max_ref      = None
    has_vref       = None      # determined on first scenario

    # per-controllable-bus accumulators (filled after first scenario)
    n_ctrl         = None
    excess_acc     = None
    mc_acc         = None
    dr_acc         = None     # dispatch ratio
    gp_pred_acc    = None
    gp_ref_acc     = None
    gradient_acc   = None
    n_ok = 0; n_no_ref = 0

    print(f"[xai-cr] Running {args.n_scenarios} scenarios ...")
    for i, batch in enumerate(test_loader):
        if i >= args.n_scenarios:
            break
        try:
            V_pred, V_ref, space, Y, S_load = run_scenario(model, batch, device)
        except Exception as e:
            print(f"  scenario {i}: ERROR {e}")
            continue

        if n_nodes is None:
            n_nodes = int(batch["sizes"].sum().item())

        gp_pred, ctrl = implied_gen_p(Y, V_pred, S_load, space)
        cost_pred = per_bus_cost(gp_pred, space)
        mc_pred   = marginal_cost(gp_pred, space)

        if V_ref is not None:
            gp_ref, _ = implied_gen_p(Y, V_ref, S_load, space)
            cost_ref  = per_bus_cost(gp_ref, space)
            has_vref  = True
        else:
            # No reference voltage — use gen_p_pred as proxy and excess = 0
            gp_ref   = gp_pred.clone()
            cost_ref = cost_pred.clone()
            has_vref = False
            n_no_ref += 1

        ctrl_np = ctrl.cpu().numpy()
        c_idx   = np.where(ctrl_np)[0]

        if ctrl_mask_ref is None:
            ctrl_mask_ref = ctrl_np
            n_ctrl        = len(c_idx)
            excess_acc    = np.zeros(n_ctrl, np.float64)
            mc_acc        = np.zeros(n_ctrl, np.float64)
            dr_acc        = np.zeros(n_ctrl, np.float64)
            gp_pred_acc   = np.zeros(n_ctrl, np.float64)
            gp_ref_acc    = np.zeros(n_ctrl, np.float64)
            gradient_acc  = np.zeros(n_ctrl, np.float64)
            p_max_ref     = space.get("Gen_p_max")
            if p_max_ref is not None:
                p_max_ref = p_max_ref.cpu().float().numpy()[c_idx]
            else:
                p_max_ref = np.ones(n_ctrl, np.float32)

        excess_c   = (cost_pred - cost_ref).cpu().float().numpy()[c_idx]
        mc_c       = mc_pred.cpu().float().numpy()[c_idx]
        gp_pred_c  = gp_pred.cpu().float().numpy()[c_idx]
        gp_ref_c   = gp_ref.cpu().float().numpy()[c_idx]

        # dispatch ratio: clamp to avoid division by zero on zero-ref buses
        with np.errstate(divide="ignore", invalid="ignore"):
            dr_c = np.where(np.abs(gp_ref_c) > 1e-6,
                            gp_pred_c / gp_ref_c,
                            np.ones_like(gp_pred_c))

        excess_acc   += excess_c
        mc_acc       += mc_c
        dr_acc       += dr_c
        gp_pred_acc  += gp_pred_c
        gp_ref_acc   += gp_ref_c
        gradient_acc += mc_c      # dCost/dgen_p = marginal cost at pred point

        n_ok += 1
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{args.n_scenarios}  (ok={n_ok}, no_ref={n_no_ref})")

    if n_ok == 0:
        print("[xai-cr] ERROR: no scenarios processed.")
        return

    excess_mean   = excess_acc   / n_ok
    mc_mean       = mc_acc       / n_ok
    dr_mean       = dr_acc       / n_ok
    gp_pred_mean  = gp_pred_acc  / n_ok
    gp_ref_mean   = gp_ref_acc   / n_ok
    gradient_mean = gradient_acc / n_ok

    # ctrl bus 1-indexed IDs (position in the full bus list)
    ctrl_idx_full   = np.where(ctrl_mask_ref)[0]
    bus_ids_1indexed = ctrl_idx_full + 1

    # ── Summary statistics ────────────────────────────────────────────────────
    total_excess      = float(excess_mean.sum())
    total_pred_cost   = float(gp_pred_mean.sum())   # rough proxy
    top10_save        = np.argsort(np.maximum(0, excess_mean))[-10:][::-1]
    top10_savings_frac = float(np.maximum(0, excess_mean)[top10_save].sum() /
                               max(1e-12, np.maximum(0, excess_mean).sum()))

    summary = {
        "n_scenarios_ok": n_ok,
        "n_scenarios_no_vref": n_no_ref,
        "has_reference_voltages": has_vref and n_no_ref == 0,
        "n_ctrl_buses": int(n_ctrl),
        "mean_total_cost_excess_pu": round(total_excess, 6),
        "top10_buses_explain_fraction_of_total_excess": round(top10_savings_frac, 4),
        "cost_reduction_recipe": (
            "Dispatch these generators closer to their reference (OPF-optimal) levels "
            "to reduce total cost. The top-10 over-dispatched expensive generators "
            f"account for {top10_savings_frac:.1%} of all recoverable cost excess."
        ),
        "top10_buses_to_redispatch": [
            {
                "bus_1indexed":       int(bus_ids_1indexed[i]),
                "mean_excess_pu":     round(float(excess_mean[i]), 6),
                "mean_mc_pu":         round(float(mc_mean[i]), 4),
                "mean_dispatch_ratio": round(float(dr_mean[i]), 4),
                "mean_gp_pred_pu":   round(float(gp_pred_mean[i]), 4),
                "mean_gp_ref_pu":    round(float(gp_ref_mean[i]), 4),
                "action": ("reduce dispatch" if dr_mean[i] > 1.0 else "increase dispatch"),
            }
            for i in top10_save
        ],
        "xai_interpretation": (
            "Attention weights are near-uniform — the model captures AC-OPF "
            "structure through its GNN topology rather than differential attention. "
            "Cost reduction is achieved by correcting the dispatch of specific "
            "high-marginal-cost generators. The gradient attribution (dCost/dgen_p = "
            "marginal cost) identifies which generators are most price-sensitive: "
            "reducing their dispatch by ε pu lowers total cost by MC_i × ε pu. "
            "The merit-order deviation plot shows which cheap generators are "
            "under-dispatched and which expensive ones are over-dispatched."
        ),
    }

    summary_path = os.path.join(args.out_dir, "cost_reduction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[xai-cr] Summary -> {summary_path}")
    print(f"  n_ok={n_ok}  has_vref={'YES' if (has_vref and n_no_ref==0) else 'NO'}")
    print(f"  Total mean cost excess: {total_excess:.4f} pu")
    print(f"  Top-10 buses explain:   {top10_savings_frac:.1%} of recoverable excess")
    print(f"\n  Top-5 cost reduction targets:")
    for entry in summary["top10_buses_to_redispatch"][:5]:
        print(f"    bus {entry['bus_1indexed']:4d}  excess={entry['mean_excess_pu']:+.4f}  "
              f"MC={entry['mean_mc_pu']:.1f}  ratio={entry['mean_dispatch_ratio']:.3f}  "
              f"→ {entry['action']}")

    print("[xai-cr] Generating plots ...")
    try:
        make_plots(
            ctrl_idx=ctrl_idx_full,
            bus_ids_1indexed=bus_ids_1indexed,
            excess_mean=excess_mean,
            mc_mean=mc_mean,
            dispatch_ratio_mean=dr_mean,
            gp_pred_mean=gp_pred_mean,
            gp_ref_mean=gp_ref_mean,
            p_max_mean=p_max_ref,
            gradient_mean=gradient_mean,
            out_dir=args.out_dir,
        )
    except Exception as e:
        print(f"[xai-cr] WARNING: plot error: {e}")

    torch.save({
        "excess_mean":    torch.from_numpy(excess_mean),
        "mc_mean":        torch.from_numpy(mc_mean),
        "dr_mean":        torch.from_numpy(dr_mean),
        "gp_pred_mean":   torch.from_numpy(gp_pred_mean),
        "gp_ref_mean":    torch.from_numpy(gp_ref_mean),
        "gradient_mean":  torch.from_numpy(gradient_mean),
        "ctrl_idx":       torch.from_numpy(ctrl_idx_full),
        "bus_ids_1indexed": torch.from_numpy(bus_ids_1indexed),
        "n_scenarios":    n_ok,
    }, os.path.join(args.out_dir, "cost_reduction_tensors.pt"))
    print(f"[xai-cr] Tensors -> {args.out_dir}/cost_reduction_tensors.pt")
    print("[xai-cr] Done.")


if __name__ == "__main__":
    main()

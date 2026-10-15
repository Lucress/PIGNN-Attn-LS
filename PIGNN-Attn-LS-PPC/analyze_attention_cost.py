"""
Cost-focused XAI: connecting PIGNN-Attn-LS attention weights to AC-OPF dispatch economics.

For each test scenario we:
  1. Run inference with return_attn=True and capture voltages V
  2. Extract gen_p via opf_residual_and_violation(Y, V, S_load, space)
  3. Compute per-bus economic quantities:
       cost_contribution_i = c1_i * gen_p_i + c2_i * gen_p_i^2
       marginal_cost_i     = c1_i + 2 * c2_i * gen_p_i
       p_utilization_i     = gen_p_i / p_max_i   (controllable buses only)
  4. Aggregate per-bus attention (sum incoming-edge weights, normalized)

Cross-scenario aggregates:
  - Pearson correlation: bus attention vs cost_contribution
  - Pearson correlation: bus attention vs marginal_cost
  - Spearman rank: attention rank vs merit-order rank (sorted by marginal cost)
  - "Cost reduction map": high-MC + high-utilization buses = dispatch candidates

Key question: Does the model's attention reflect economic dispatch structure?
In optimal AC-OPF:
  - Price-setting generators (highest marginal cost that is still dispatched) should
    receive the most attention — they determine the shadow price of all constraints.
  - Congested corridors create LMP spreads; edge attention there implies implicit
    LMP awareness.

Outputs in out_dir/:
  xai_cost_summary.json          correlations, top-10 buses, interpretation
  attn_vs_cost_contribution.png  scatter: attention vs cost contribution
  attn_vs_marginal_cost.png      scatter: attention vs marginal cost
  merit_order_attn.png           bar: generators sorted by MC, attention overlay
  cost_reduction_map.png         scatter: MC vs P-headroom, size=attention
  edge_attn_heatmap.png          per-edge attention magnitude (sorted by bus pair)

Usage (run from PIGNN-Attn-LS-PPC/):
    python analyze_attention_cost.py \\
        --ckpt results/ckpt/tinyx/pignn_pdl2_case118_best.pt \\
        --opfdata_root ./opfdata \\
        --case_name pglib_opf_case118_ieee \\
        --n_scenarios 200 \\
        --out_dir results/xai_cost/pdl2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch


# ── model loading (same interface as analyze_attention_opfdata.py) ─────────────

def load_model(ckpt_path: str, K: int, d: int, d_hi: int, n_heads: int,
               num_attn_layers: int) -> "GNSMsg_EdgeSelfAttn_OPF":
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


# ── inference returning voltages + cost data ──────────────────────────────────

@torch.no_grad()
def run_scenario(model, batch, device):
    """Return (V, attn_list, edge_index, space, Y, S_load)."""
    from opf_task import opf_decision_space
    from train_valid_test_pignn_opf import branch_rows_from_opfdata

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

    out, _phys, attn_list, edge_index = result
    V = out.unsqueeze(0) if out.dim() == 2 else out          # [1, N, 2]
    return V, attn_list, edge_index, space, Y, S_load


def compute_bus_economics(V, space, Y, S_load, device):
    """
    Returns per-bus dicts (length N, float32 numpy):
      cost_contribution  c1*Pg + c2*Pg^2
      marginal_cost      c1 + 2*c2*Pg
      p_utilization      Pg / Pg_max  (0 where not controllable)
      gen_p              implied dispatch (0 where not controllable)
      is_ctrl            boolean
    """
    from opf_task import opf_residual_and_violation

    parts = opf_residual_and_violation(Y, V, S_load, space)
    gen_p = parts["gen_p"].to(torch.float64)                 # (N,)
    ctrl  = parts["ctrl"]                                    # (N,) bool

    c1 = space.get("Gen_cost_c1")
    c2 = space.get("Gen_cost_c2")

    if c1 is None:
        N = gen_p.shape[0]
        zeros = np.zeros(N, dtype=np.float32)
        return {"cost_contribution": zeros, "marginal_cost": zeros,
                "p_utilization": zeros, "gen_p": zeros, "is_ctrl": ctrl.cpu().numpy()}

    c1 = c1.to(torch.float64)
    cost_contr = (c1 * gen_p)
    if c2 is not None:
        cost_contr = cost_contr + c2.to(torch.float64) * gen_p * gen_p

    mc = c1.clone()
    if c2 is not None:
        mc = mc + 2.0 * c2.to(torch.float64) * gen_p

    p_max = space.get("Gen_p_max")
    if p_max is not None:
        p_util = torch.zeros_like(gen_p)
        active = ctrl & (p_max.to(torch.float64) > 1e-6)
        p_util[active] = (gen_p[active] / p_max.to(torch.float64)[active]).clamp(0, 1)
    else:
        p_util = torch.zeros_like(gen_p)

    # zero out non-controllable buses
    mask_f = ctrl.to(torch.float64)
    return {
        "cost_contribution": (cost_contr * mask_f).cpu().float().numpy(),
        "marginal_cost":     (mc         * mask_f).cpu().float().numpy(),
        "p_utilization":     p_util.cpu().float().numpy(),
        "gen_p":             gen_p.cpu().float().numpy(),
        "is_ctrl":           ctrl.cpu().numpy(),
    }


def aggregate_bus_attention(attn_list, edge_index, n_nodes):
    """Mean over K steps → per-edge weight → sum-to-bus."""
    attn_stack = torch.stack(attn_list, dim=0)    # (K, E)
    mean_edge  = attn_stack.mean(dim=0)           # (E,)
    bus_attn   = torch.zeros(n_nodes)
    for e_idx, (_, dst) in enumerate(edge_index.tolist()):
        bus_attn[dst] += mean_edge[e_idx]
    denom = bus_attn.sum().clamp_min(1e-12)
    return (bus_attn / denom).numpy(), mean_edge.numpy()


# ── plotting ──────────────────────────────────────────────────────────────────

def _pearson(a, b):
    a, b = np.array(a), np.array(b)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return float(r)


def make_plots(
    bus_attn_mean: np.ndarray,
    cost_mean: np.ndarray,
    mc_mean: np.ndarray,
    p_util_mean: np.ndarray,
    p_max_mean: np.ndarray,
    gen_p_mean: np.ndarray,
    ctrl_mask: np.ndarray,
    edge_index: np.ndarray,
    edge_attn_mean: np.ndarray,
    out_dir: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    ctrl_idx = np.where(ctrl_mask)[0]

    def ctrl_vals(arr):
        return arr[ctrl_idx]

    attn_c  = ctrl_vals(bus_attn_mean)
    cost_c  = ctrl_vals(cost_mean)
    mc_c    = ctrl_vals(mc_mean)
    util_c  = ctrl_vals(p_util_mean)

    # ── 1. Attention vs cost contribution ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(cost_c, attn_c, c=mc_c, cmap="plasma", s=60, alpha=0.8, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Marginal cost (pu)")
    ax.set_xlabel("Cost contribution per bus (pu·cost)")
    ax.set_ylabel("Normalised bus attention")
    r = _pearson(cost_c, attn_c)
    ax.set_title(f"Attention vs dispatch cost contribution  (r={r:+.3f})")

    # annotate top-5 by cost
    top5 = np.argsort(cost_c)[-5:]
    for i in top5:
        ax.annotate(f"bus {ctrl_idx[i]+1}", (cost_c[i], attn_c[i]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "attn_vs_cost_contribution.png"), dpi=150)
    plt.close()

    # ── 2. Attention vs marginal cost ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(mc_c, attn_c, c=util_c, cmap="RdYlGn_r", s=60, alpha=0.8, edgecolors="none",
                    vmin=0, vmax=1)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("P utilisation (Pg/Pg_max)")
    ax.set_xlabel("Marginal cost (pu·cost / pu·power)")
    ax.set_ylabel("Normalised bus attention")
    r2 = _pearson(mc_c, attn_c)
    ax.set_title(f"Attention vs marginal cost  (r={r2:+.3f})")
    top5 = np.argsort(mc_c)[-5:]
    for i in top5:
        ax.annotate(f"bus {ctrl_idx[i]+1}", (mc_c[i], attn_c[i]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "attn_vs_marginal_cost.png"), dpi=150)
    plt.close()

    # ── 3. Merit-order bar chart ───────────────────────────────────────────────
    # Sort controllable generators by mean marginal cost (merit order)
    order = np.argsort(mc_c)
    merit_mc   = mc_c[order]
    merit_attn = attn_c[order]
    merit_bus  = ctrl_idx[order] + 1    # 1-indexed bus numbers

    fig, ax1 = plt.subplots(figsize=(max(8, len(order) // 3), 5))
    x = np.arange(len(order))
    ax1.bar(x, merit_mc, color="steelblue", alpha=0.6, label="Marginal cost")
    ax1.set_xlabel("Generators sorted by marginal cost (merit order)")
    ax1.set_ylabel("Marginal cost (pu)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(x, merit_attn, color="crimson", marker="o", ms=4, lw=1.5,
             label="Bus attention")
    ax2.set_ylabel("Bus attention (normalised)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    # x-tick every 5 generators
    tick_pos = x[::5]
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels([f"b{merit_bus[i]}" for i in tick_pos], rotation=60, fontsize=7)

    ax1.set_title("Merit-order dispatch vs model attention")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "merit_order_attn.png"), dpi=150)
    plt.close()

    # ── 4. Cost reduction map: MC vs headroom, sized by attention ─────────────
    # Headroom = P_max - P_dispatch (how much more generation is available)
    p_max_c  = ctrl_vals(p_max_mean)
    gp_c     = ctrl_vals(gen_p_mean)
    headroom = np.maximum(0, p_max_c - gp_c)

    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = 20 + 400 * (attn_c / (attn_c.max() + 1e-12))
    sc = ax.scatter(mc_c, headroom, s=sizes, c=cost_c, cmap="YlOrRd", alpha=0.75,
                    edgecolors="k", linewidths=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Cost contribution (pu·cost)")
    ax.set_xlabel("Marginal cost (pu·cost / pu·power)")
    ax.set_ylabel("P headroom = Pg_max − Pg (pu)  [lower MC = cheaper slack]")
    ax.set_title("Cost reduction map  (size ∝ model attention)\n"
                 "Low-MC + high headroom = priority for dispatch increase")

    # annotate the cheapest generators with spare headroom > 0.05 pu
    cheap_with_headroom = np.where((mc_c < np.percentile(mc_c, 30)) & (headroom > 0.05))[0]
    for i in cheap_with_headroom[:5]:
        ax.annotate(f"b{ctrl_idx[i]+1}", (mc_c[i], headroom[i]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cost_reduction_map.png"), dpi=150)
    plt.close()

    # ── 5. Edge attention heatmap (top-50 edges by mean attention) ────────────
    top_e = np.argsort(edge_attn_mean)[-50:]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(top_e)), edge_attn_mean[top_e], color="darkorange", alpha=0.8)
    ax.set_yticks(range(len(top_e)))
    ax.set_yticklabels(
        [f"{edge_index[e,0]+1}→{edge_index[e,1]+1}" for e in top_e[::-1]],
        fontsize=6
    )
    # reverse for top-to-bottom display
    ax.invert_yaxis()
    ax.set_xlabel("Mean attention weight")
    ax.set_title("Top-50 edges by attention (bus_src → bus_dst, 1-indexed)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "edge_attn_heatmap.png"), dpi=150)
    plt.close()

    print(f"[xai] Plots saved to {out_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",           required=True)
    p.add_argument("--opfdata_root",   required=True)
    p.add_argument("--case_name",      default="pglib_opf_case118_ieee")
    p.add_argument("--n_scenarios",    type=int, default=200)
    p.add_argument("--out_dir",        default="results/xai_cost/pdl2")
    p.add_argument("--K",              type=int, default=30)
    p.add_argument("--d",              type=int, default=4)
    p.add_argument("--d_hi",           type=int, default=16)
    p.add_argument("--n_heads",        type=int, default=4)
    p.add_argument("--num_attn_layers", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    print(f"[xai-cost] Checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, args.K, args.d, args.d_hi,
                       args.n_heads, args.num_attn_layers)

    print(f"[xai-cost] Loading OPFData: {args.case_name}")
    from opfdata_pipeline import make_opfdata_loaders
    _, _, _, test_loader = make_opfdata_loaders(
        args.opfdata_root, args.case_name, batch_size=1,
        max_train=1, max_valid=1, max_test=args.n_scenarios,
        num_workers=0,
    )

    # Per-scenario accumulators
    bus_attn_acc  = None
    cost_acc      = None
    mc_acc        = None
    util_acc      = None
    p_max_acc     = None
    gp_acc        = None
    ctrl_mask_ref = None
    edge_index_np = None
    edge_attn_acc = None
    n_ok          = 0
    n_nodes       = None

    print(f"[xai-cost] Running {args.n_scenarios} scenarios ...")
    for i, batch in enumerate(test_loader):
        if i >= args.n_scenarios:
            break
        try:
            V, attn_list, edge_index, space, Y, S_load = run_scenario(model, batch, device)
        except Exception as e:
            print(f"  scenario {i}: ERROR {e}")
            continue

        if n_nodes is None:
            n_nodes = int(batch["sizes"].sum().item())

        econ = compute_bus_economics(V, space, Y, S_load, device)
        bus_attn_np, edge_attn_np = aggregate_bus_attention(attn_list, edge_index, n_nodes)

        if bus_attn_acc is None:
            bus_attn_acc  = bus_attn_np.copy()
            cost_acc      = econ["cost_contribution"].copy()
            mc_acc        = econ["marginal_cost"].copy()
            util_acc      = econ["p_utilization"].copy()
            gp_acc        = econ["gen_p"].copy()
            ctrl_mask_ref = econ["is_ctrl"].copy()
            edge_index_np = edge_index.cpu().numpy()
            edge_attn_acc = edge_attn_np.copy()

            # p_max is scenario-independent (network topology), grab once
            pm = space.get("Gen_p_max")
            p_max_acc = pm.cpu().float().numpy() if pm is not None else np.zeros(n_nodes, np.float32)
        else:
            bus_attn_acc  += bus_attn_np
            cost_acc      += econ["cost_contribution"]
            mc_acc        += econ["marginal_cost"]
            util_acc      += econ["p_utilization"]
            gp_acc        += econ["gen_p"]
            edge_attn_acc += edge_attn_np

        n_ok += 1
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{args.n_scenarios}  (ok={n_ok})")

    if n_ok == 0:
        print("[xai-cost] ERROR: no scenarios processed.")
        return

    # ── Normalise accumulators ─────────────────────────────────────────────────
    bus_attn_mean  = bus_attn_acc  / n_ok
    cost_mean      = cost_acc      / n_ok
    mc_mean        = mc_acc        / n_ok
    util_mean      = util_acc      / n_ok
    gp_mean        = gp_acc        / n_ok
    edge_attn_mean = edge_attn_acc / n_ok

    # ── Correlations (controllable buses only) ─────────────────────────────────
    ctrl_idx  = np.where(ctrl_mask_ref)[0]
    attn_c    = bus_attn_mean[ctrl_idx]
    cost_c    = cost_mean[ctrl_idx]
    mc_c      = mc_mean[ctrl_idx]

    try:
        from scipy.stats import spearmanr
        r_sp_cost, _ = spearmanr(attn_c, cost_c)
        r_sp_mc, _   = spearmanr(attn_c, mc_c)
    except ImportError:
        r_sp_cost = r_sp_mc = float("nan")

    r_pe_cost = _pearson(attn_c, cost_c)
    r_pe_mc   = _pearson(attn_c, mc_c)

    # top-10 generator buses by cost contribution
    top10_cost = np.argsort(cost_c)[-10:][::-1]
    top10_mc   = np.argsort(mc_c)[-10:][::-1]

    # merit-order alignment: fraction of top-50%-MC generators that are also top-50%-attn
    half = len(ctrl_idx) // 2
    top_mc_mask   = np.argsort(mc_c)[-half:]
    top_attn_mask = np.argsort(attn_c)[-half:]
    overlap_frac  = len(np.intersect1d(top_mc_mask, top_attn_mask)) / max(1, half)

    summary = {
        "n_scenarios_ok": n_ok,
        "n_buses": int(n_nodes),
        "n_ctrl_buses": int(len(ctrl_idx)),
        "correlation_attn_vs_cost_contribution": {
            "pearson":  round(r_pe_cost, 4),
            "spearman": round(float(r_sp_cost), 4),
        },
        "correlation_attn_vs_marginal_cost": {
            "pearson":  round(r_pe_mc, 4),
            "spearman": round(float(r_sp_mc), 4),
        },
        "merit_order_top50pct_overlap": round(overlap_frac, 4),
        "interpretation": (
            "A positive Pearson/Spearman correlation between attention and "
            "marginal cost indicates that the model implicitly learns the "
            "economic merit order: buses with expensive generators attract more "
            "message-passing attention. "
            f"Top-50%%-MC vs top-50%%-attn overlap: {overlap_frac:.1%}. "
            "Overlap > 50%% means attention is non-random with respect to cost."
        ),
        "top10_buses_by_cost_contribution": [
            {
                "bus_1indexed": int(ctrl_idx[i] + 1),
                "mean_cost_contribution_pu": float(cost_c[i]),
                "mean_marginal_cost_pu":    float(mc_c[i]),
                "mean_attention":           float(attn_c[i]),
                "mean_p_utilization":       float(util_mean[ctrl_idx[i]]),
            }
            for i in top10_cost
        ],
        "top10_buses_by_marginal_cost": [
            {
                "bus_1indexed": int(ctrl_idx[i] + 1),
                "mean_marginal_cost_pu":    float(mc_c[i]),
                "mean_cost_contribution_pu": float(cost_c[i]),
                "mean_attention":           float(attn_c[i]),
            }
            for i in top10_mc
        ],
        "cost_reduction_candidates": (
            "Generator buses with MC below the 30th-percentile that still have "
            "P-headroom > 0.05 pu should be dispatched more to lower total cost. "
            "If the model attends more to expensive generators, it has learned the "
            "shadow prices but may not yet fully exploit cheap generation slack."
        ),
        "ac_opf_interpretation": (
            "In AC-OPF, cost is minimised subject to power-flow feasibility and "
            "generator limits.  The price-setting generator (the one with the "
            "highest MC still dispatched) sets the marginal system cost.  "
            "If attention concentrates on high-MC buses, the model has captured "
            "the 'price-setter' role without being trained on LMPs directly. "
            "Edge attention on corridors connecting low-cost and high-cost buses "
            "reflects implicit awareness of transmission constraints (congestion)."
        ),
    }

    summary_path = os.path.join(args.out_dir, "xai_cost_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[xai-cost] Summary -> {summary_path}")

    # Print key numbers
    print(f"\n[xai-cost] === Key findings (n_ok={n_ok}) ===")
    print(f"  Attn vs cost contribution:  Pearson={r_pe_cost:+.3f}  Spearman={float(r_sp_cost):+.3f}")
    print(f"  Attn vs marginal cost:      Pearson={r_pe_mc:+.3f}  Spearman={float(r_sp_mc):+.3f}")
    print(f"  Merit-order top-50%% overlap: {overlap_frac:.1%}")
    print(f"\n  Top-5 buses by cost contribution:")
    for entry in summary["top10_buses_by_cost_contribution"][:5]:
        print(f"    bus {entry['bus_1indexed']:4d}  cost={entry['mean_cost_contribution_pu']:.4f}"
              f"  MC={entry['mean_marginal_cost_pu']:.4f}"
              f"  attn={entry['mean_attention']:.5f}"
              f"  util={entry['mean_p_utilization']:.2%}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("[xai-cost] Generating plots ...")
    try:
        make_plots(
            bus_attn_mean=bus_attn_mean,
            cost_mean=cost_mean,
            mc_mean=mc_mean,
            p_util_mean=util_mean,
            p_max_mean=p_max_acc,
            gen_p_mean=gp_mean,
            ctrl_mask=ctrl_mask_ref,
            edge_index=edge_index_np,
            edge_attn_mean=edge_attn_mean,
            out_dir=args.out_dir,
        )
    except Exception as e:
        print(f"[xai-cost] WARNING: plot generation failed: {e}")

    # ── Save raw tensors for further analysis ──────────────────────────────────
    torch.save({
        "bus_attn_mean":  torch.from_numpy(bus_attn_mean),
        "cost_mean":      torch.from_numpy(cost_mean),
        "mc_mean":        torch.from_numpy(mc_mean),
        "util_mean":      torch.from_numpy(util_mean),
        "gp_mean":        torch.from_numpy(gp_mean),
        "edge_attn_mean": torch.from_numpy(edge_attn_mean),
        "edge_index":     torch.from_numpy(edge_index_np),
        "ctrl_mask":      torch.from_numpy(ctrl_mask_ref),
        "n_scenarios":    n_ok,
    }, os.path.join(args.out_dir, "xai_cost_tensors.pt"))
    print(f"[xai-cost] Raw tensors -> {args.out_dir}/xai_cost_tensors.pt")
    print("[xai-cost] Done.")


if __name__ == "__main__":
    main()

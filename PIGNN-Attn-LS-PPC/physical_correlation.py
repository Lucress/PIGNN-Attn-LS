"""
Physical correlation analysis: do edge self-attention weights encode physically
meaningful bus influence?

For each test scenario we collect:
  - edge attention weights (already in attn_weights.pt)
  - line power flow magnitude |S_ij| (computed from predicted V)
  - bus voltage deviation |V_i - 1.0|

Then we compute Spearman correlation across scenarios and edges/buses.

Usage (run from PIGNN-Attn-LS-PPC/):
    python physical_correlation.py \
        --ckpt results/ckpt/pignn_opf_pglib_opf_case14_ieee_best.pt \
        --opfdata_root ./opfdata \
        --case_name pglib_opf_case14_ieee \
        --attn_dir results/attention_supervised \
        --out_dir results/correlation_supervised \
        --K 15 --d 4 --d_hi 16 --n_heads 4 \
        --label "Supervised (AC-OPF)"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats


# ── IEEE 14-bus bus positions and types ──────────────────────────────────────

CASE14_POS = {
    0:  (0.0,  2.0), 1:  (1.5,  3.0), 2:  (3.5,  3.5), 3:  (4.5,  2.5),
    4:  (4.0,  1.5), 5:  (2.5,  3.8), 6:  (3.2,  4.5), 7:  (2.8,  4.8),
    8:  (3.8,  4.8), 9:  (4.5,  4.2), 10: (4.2,  3.8), 11: (4.8,  3.5),
    12: (5.2,  3.8), 13: (5.0,  3.2),
}
CASE14_BRANCHES = [
    (0,1),(0,4),(1,2),(1,3),(1,4),(2,3),(3,4),(3,6),(3,8),
    (4,5),(5,10),(5,11),(5,12),(6,7),(6,8),(8,9),(9,10),(10,11),(11,12),(12,13)
]
CASE14_BUS_TYPE = {0:"slack",1:"gen",2:"gen",5:"gen",7:"gen"}


# ── model loading (same as analyze_attention_opfdata.py) ─────────────────────

def load_model(ckpt_path, K, d, d_hi, n_heads, num_attn_layers):
    from GNSMsg_SelfAttention_armijo_opf import GNSMsg_EdgeSelfAttn_OPF
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = GNSMsg_EdgeSelfAttn_OPF(
        d=d, d_hi=d_hi, K=K, n_heads=n_heads, num_attn_layers=num_attn_layers,
        pinn=True, use_armijo=True, armijo_mode="geometric_safe",
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ── inference: attention + voltage output ────────────────────────────────────

@torch.no_grad()
def run_batch_full(model, batch, device):
    from opf_task import opf_decision_space
    from train_valid_test_pignn_opf import branch_rows_from_opfdata

    bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
          for k, v in batch.items() if k != "hetero"}
    hb  = batch["hetero"].to(device)
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

    out, _, attn_list, edge_index = result
    # out: (1, N, 2) — [v_mag, v_ang]
    vm  = out[0, :, 0].cpu()   # (N,)
    va  = out[0, :, 1].cpu()   # (N,)
    Y_c = Y.squeeze(0).cpu() if Y.dim() == 3 else Y.cpu()
    return vm, va, Y_c.to(torch.complex128), attn_list, edge_index


def compute_line_loading(vm, va, Y_dense, edge_index):
    """Compute |S_ij| for each directed edge from predicted voltages."""
    Vc = (vm * torch.exp(1j * va.to(torch.float64))).to(torch.complex128)
    Ic = Y_dense @ Vc          # (N,)
    # For each directed edge (src -> dst), apparent power leaving src
    line_flows = []
    for src, dst in edge_index.tolist():
        # Current flowing from src to dst via Y[dst, src] admittance
        # Simple approximation: |V_src - V_dst| * |Y_ij|
        Y_ij = Y_dense[dst, src]
        S_ij = abs(complex(Vc[src]) * complex(Y_ij).conjugate() *
                   (complex(Vc[src]) - complex(Vc[dst])).conjugate())
        line_flows.append(float(S_ij))
    return torch.tensor(line_flows)   # (E,)


def compute_bus_voltage_deviation(vm):
    """|V_i - 1.0| for each bus."""
    return (vm - 1.0).abs()   # (N,)


# ── correlation analysis ─────────────────────────────────────────────────────

def spearman(x, y):
    """Spearman r and p-value between two 1-D arrays."""
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


# ── plots ────────────────────────────────────────────────────────────────────

def plot_scatter(x, y, xlabel, ylabel, title, out_path, annotate_buses=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.6, s=60, color="#1565C0", edgecolors="white", lw=0.5)
    if annotate_buses is not None:
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(str(i + 1), (xi, yi), fontsize=7, ha="left", va="bottom",
                        color="#555", xytext=(3, 3), textcoords="offset points")
    r, p = spearman(x, y)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{title}\nSpearman r={r:.3f}, p={p:.3f}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return r, p


def plot_attention_map_with_physics(bus_attn, bus_vdev, n_nodes, label, out_path):
    """Side-by-side: attention strength vs voltage deviation, colored by bus type."""
    pos      = CASE14_POS if n_nodes == 14 else {i: (i % 5, i // 5) for i in range(n_nodes)}
    branches = CASE14_BRANCHES if n_nodes == 14 else []

    type_colors = {"slack": "#1565C0", "gen": "#6A1B9A", "load": "#E65100"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{label} — Attention vs Physics", fontsize=10, fontweight="bold")
    cmap = cm.YlOrRd

    def draw(ax, values, title):
        vmin_, vmax_ = float(values.min()), float(values.max())
        norm_vals = (values - vmin_) / max(vmax_ - vmin_, 1e-12)
        for (u, v) in branches:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color="#999", lw=1.0, alpha=0.6, zorder=1)
        xs = [pos[i][0] for i in range(n_nodes)]
        ys = [pos[i][1] for i in range(n_nodes)]
        colors = list(norm_vals.numpy())
        sizes  = [150 + 350 * c for c in colors]
        sc = ax.scatter(xs, ys, c=colors, s=sizes, cmap=cmap, vmin=0, vmax=1,
                        zorder=2, edgecolors="white", lw=1.0)
        for i in range(n_nodes):
            ax.annotate(str(i + 1), (pos[i][0], pos[i][1]), fontsize=7,
                        ha="center", va="center", color="white", fontweight="bold", zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")
        return sc

    sc1 = draw(axes[0], bus_attn, "Attention weight per bus")
    sc2 = draw(axes[1], bus_vdev,  "Voltage deviation |V-1| per bus")

    fig.colorbar(sc1, ax=axes[0], shrink=0.7, label="normalized")
    fig.colorbar(sc2, ax=axes[1], shrink=0.7, label="normalized")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_bustype_attention(bus_attn, n_nodes, label, out_path):
    """Bar chart: mean attention by bus type."""
    type_groups = {"slack": [], "gen": [], "load": []}
    for i in range(n_nodes):
        bt = CASE14_BUS_TYPE.get(i, "load")
        type_groups[bt].append(float(bus_attn[i]))

    fig, ax = plt.subplots(figsize=(5, 4))
    means = {k: np.mean(v) if v else 0 for k, v in type_groups.items()}
    stds  = {k: np.std(v)  if v else 0 for k, v in type_groups.items()}
    colors = {"slack": "#1565C0", "gen": "#6A1B9A", "load": "#E65100"}
    for i, (bt, m) in enumerate(means.items()):
        ax.bar(bt, m, yerr=stds[bt], color=colors[bt], alpha=0.85,
               capsize=4, edgecolor="white")
    ax.set_ylabel("Mean attention weight", fontsize=10)
    ax.set_title(f"{label}\nAttention by bus type", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return means


def plot_correlation_summary(results: dict, out_path: str):
    """Bar chart comparing Spearman r values across models."""
    models  = list(results.keys())
    metrics = ["r_attn_vdev", "r_attn_flow"]
    labels  = ["Attn vs |V-1|", "Attn vs line flow"]
    colors  = ["#1565C0", "#E65100"]

    x = np.arange(len(models))
    w = 0.3
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, (m, lbl, c) in enumerate(zip(metrics, labels, colors)):
        vals = [results[mod].get(m, 0) for mod in models]
        ax.bar(x + j * w, vals, w, label=lbl, color=c, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Spearman r", fontsize=10)
    ax.set_title("Physical Interpretability: Attention vs Physics\n"
                 "(higher r = attention tracks physical stress)", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--opfdata_root", required=True)
    p.add_argument("--case_name",    default="pglib_opf_case14_ieee")
    p.add_argument("--attn_dir",     required=True)
    p.add_argument("--out_dir",      required=True)
    p.add_argument("--label",        default="Model")
    p.add_argument("--n_scenarios",  type=int, default=50)
    p.add_argument("--K",            type=int, default=30)
    p.add_argument("--d",            type=int, default=10)
    p.add_argument("--d_hi",         type=int, default=32)
    p.add_argument("--n_heads",      type=int, default=4)
    p.add_argument("--num_attn_layers", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    # load pre-computed attention
    attn_weights = torch.load(os.path.join(args.attn_dir, "attn_weights.pt"),
                              map_location="cpu", weights_only=False)  # (S, K, E)
    edge_index   = torch.load(os.path.join(args.attn_dir, "edge_index.pt"),
                              map_location="cpu", weights_only=False)  # (E, 2)
    bus_data     = torch.load(os.path.join(args.attn_dir, "bus_attention.pt"),
                              map_location="cpu", weights_only=False)
    bus_attn     = bus_data["bus_attn"]    # (N,)
    n_nodes      = bus_data["n_nodes"]
    K            = bus_data["K"]

    print(f"[corr] {args.label}: {attn_weights.shape[0]} scenarios, K={K}, "
          f"{n_nodes} buses, {edge_index.shape[0]} directed edges")

    # re-run model to collect voltage predictions per scenario
    print(f"[corr] Loading model and re-running inference for physical quantities ...")
    model = load_model(args.ckpt, args.K, args.d, args.d_hi,
                       args.n_heads, args.num_attn_layers)

    from opfdata_pipeline import make_opfdata_loaders
    _, _, _, test_loader = make_opfdata_loaders(
        args.opfdata_root, args.case_name, batch_size=1,
        max_train=1, max_valid=1, max_test=args.n_scenarios, num_workers=0,
    )

    all_vdev   = []   # (N,) per scenario — bus voltage deviation
    all_flow   = []   # (E,) per scenario — directed edge line flow
    all_attn_e = []   # (E,) per scenario — mean attention over K steps

    for i, batch in enumerate(test_loader):
        if i >= attn_weights.shape[0]:
            break
        try:
            vm, va, Y_dense, attn_list, ei = run_batch_full(model, batch, device)
        except Exception as e:
            print(f"  scenario {i}: ERROR {e}")
            continue

        vdev = compute_bus_voltage_deviation(vm)           # (N,)
        flow = compute_line_loading(vm, va, Y_dense, edge_index)  # (E,)
        a_e  = attn_weights[i].mean(dim=0)                # (E,) mean over K

        all_vdev.append(vdev)
        all_flow.append(flow)
        all_attn_e.append(a_e)

        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{attn_weights.shape[0]}")

    S = len(all_vdev)
    print(f"[corr] Collected {S} scenarios")

    # ── aggregate to bus level: sum incoming attention ────────────────────────
    all_bus_attn = []
    for a_e in all_attn_e:
        ba = torch.zeros(n_nodes)
        for e_idx, (_, dst) in enumerate(edge_index.tolist()):
            ba[dst] += float(a_e[e_idx])
        all_bus_attn.append(ba)

    # flatten across scenarios for correlation
    flat_attn_e  = torch.stack(all_attn_e,  dim=0).reshape(-1).numpy()   # (S*E,)
    flat_flow    = torch.stack(all_flow,    dim=0).reshape(-1).numpy()   # (S*E,)
    flat_bus_a   = torch.stack(all_bus_attn, dim=0).reshape(-1).numpy()  # (S*N,)
    flat_vdev    = torch.stack(all_vdev,    dim=0).reshape(-1).numpy()   # (S*N,)

    # ── Spearman correlations ─────────────────────────────────────────────────
    r_flow,  p_flow  = spearman(flat_attn_e, flat_flow)
    r_vdev,  p_vdev  = spearman(flat_bus_a,  flat_vdev)

    print(f"[corr] Edge attention vs line flow:   r={r_flow:.3f}  p={p_flow:.4f}")
    print(f"[corr] Bus attention  vs voltage dev: r={r_vdev:.3f}  p={p_vdev:.4f}")

    results = {
        "label":       args.label,
        "n_scenarios": S,
        "n_nodes":     n_nodes,
        "K":           K,
        "r_attn_flow": r_flow,
        "p_attn_flow": p_flow,
        "r_attn_vdev": r_vdev,
        "p_attn_vdev": p_vdev,
    }
    with open(os.path.join(args.out_dir, "correlation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── mean per-bus statistics across scenarios ──────────────────────────────
    mean_vdev    = torch.stack(all_vdev,     dim=0).mean(dim=0)    # (N,)
    mean_bus_a   = torch.stack(all_bus_attn, dim=0).mean(dim=0)    # (N,)
    mean_flow_e  = torch.stack(all_flow,     dim=0).mean(dim=0)    # (E,)
    mean_attn_e  = torch.stack(all_attn_e,   dim=0).mean(dim=0)    # (E,)

    # ── scatter: edge attention vs line flow (per-edge, mean across scenarios) -
    plot_scatter(
        mean_attn_e.numpy(), mean_flow_e.numpy(),
        "Mean edge attention", "Mean line flow |S_ij|",
        f"{args.label}\nEdge attention vs line loading",
        os.path.join(args.out_dir, "scatter_attn_vs_flow.png"),
    )

    # ── scatter: bus attention vs voltage deviation ───────────────────────────
    plot_scatter(
        mean_bus_a.numpy(), mean_vdev.numpy(),
        "Mean bus attention", "Mean voltage deviation |V-1|",
        f"{args.label}\nBus attention vs voltage deviation",
        os.path.join(args.out_dir, "scatter_attn_vs_vdev.png"),
        annotate_buses=list(range(n_nodes)),
    )

    # ── grid map: attention vs physics side by side ───────────────────────────
    plot_attention_map_with_physics(
        mean_bus_a, mean_vdev, n_nodes, args.label,
        os.path.join(args.out_dir, "attn_vs_physics_map.png"),
    )

    # ── bar chart: attention by bus type ─────────────────────────────────────
    means = plot_bustype_attention(
        mean_bus_a, n_nodes, args.label,
        os.path.join(args.out_dir, "bustype_attention.png"),
    )
    results["mean_attn_by_type"] = means
    with open(os.path.join(args.out_dir, "correlation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"[corr] Mean attention by type: {means}")
    print(f"[corr] Figures saved to {args.out_dir}/")
    print(f"[corr] Done.")
    return results


if __name__ == "__main__":
    main()

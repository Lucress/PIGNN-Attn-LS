"""
Visualization of receptive field and attention weights on the power grid graph.

Usage (run from PIGNN-Attn-LS-PPC/):
    python visualize_attention.py \
        --attn_dir results/attention \
        --case_name pglib_opf_case14_ieee \
        --out_dir results/figures

Produces:
    receptive_field_case14.png  -- graph diameter vs K bar chart
    attention_map_case14.png    -- grid graph colored by bus attention strength
    attention_evolution.png     -- how attention changes across K steps
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# ── IEEE 14-bus fixed layout (approximate geographic positions) ──────────────

CASE14_POS = {
    0:  (0.0,  2.0),   # bus 1  — slack (generator)
    1:  (1.5,  3.0),   # bus 2  — generator (PV)
    2:  (3.5,  3.5),   # bus 3  — generator (PV)
    3:  (4.5,  2.5),   # bus 4  — load
    4:  (4.0,  1.5),   # bus 5  — load
    5:  (2.5,  3.8),   # bus 6  — generator (PV)
    6:  (3.2,  4.5),   # bus 7  — transit
    7:  (2.8,  4.8),   # bus 8  — generator (PV)
    8:  (3.8,  4.8),   # bus 9  — load
    9:  (4.5,  4.2),   # bus 10 — load
    10: (4.2,  3.8),   # bus 11 — load
    11: (4.8,  3.5),   # bus 12 — load
    12: (5.2,  3.8),   # bus 13 — load
    13: (5.0,  3.2),   # bus 14 — load
}

# IEEE 14-bus branch list (0-indexed)
CASE14_BRANCHES = [
    (0,1),(0,4),(1,2),(1,3),(1,4),(2,3),(3,4),(3,6),(3,8),
    (4,5),(5,10),(5,11),(5,12),(6,7),(6,8),(8,9),(9,10),(10,11),(11,12),(12,13)
]

# Bus types for case14 (0=slack, 1=PV gen, 2=PQ load)
CASE14_BUS_TYPES = {
    0: "slack", 1: "gen", 2: "gen", 5: "gen", 7: "gen",
}


def bus_label(i):
    return str(i + 1)  # 1-indexed label


# ── helpers ──────────────────────────────────────────────────────────────────

def draw_grid(ax, pos, branches, node_colors, node_sizes, title, cmap, vmin, vmax,
              show_colorbar=True, fig=None):
    """Draw the power grid graph with colored buses."""
    # edges
    for (u, v) in branches:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#888", lw=1.2, zorder=1, alpha=0.7)

    # nodes
    xs = [pos[i][0] for i in sorted(pos)]
    ys = [pos[i][1] for i in sorted(pos)]
    sc = ax.scatter(xs, ys, c=node_colors, s=node_sizes, cmap=cmap,
                    vmin=vmin, vmax=vmax, zorder=2, edgecolors="white", linewidths=1.0)

    # labels
    for i in sorted(pos):
        x, y = pos[i]
        ax.annotate(bus_label(i), (x, y), fontsize=7.5, ha="center", va="center",
                    color="white", fontweight="bold", zorder=3)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    if show_colorbar and fig is not None:
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
        cb.set_label("attention weight", fontsize=8)
    return sc


# ── plot 1: receptive field analysis ─────────────────────────────────────────

def plot_receptive_field(rf_info: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Receptive Field Analysis — {rf_info['case']}", fontsize=11, fontweight="bold")

    # left: eccentricity per bus
    ax = axes[0]
    ecc = rf_info["eccentricity_per_bus"]
    buses = list(range(len(ecc)))
    colors = ["#E53935" if e > rf_info["K"] else "#43A047" for e in ecc]
    ax.bar([bus_label(b) for b in buses], ecc, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(rf_info["K"], color="#1565C0", lw=2, ls="--", label=f"K={rf_info['K']}")
    ax.axhline(rf_info["graph_diameter"], color="#E65100", lw=1.5, ls=":",
               label=f"diameter={rf_info['graph_diameter']}")
    ax.set_xlabel("Bus", fontsize=9)
    ax.set_ylabel("Eccentricity (max shortest path)", fontsize=9)
    ax.set_title("Eccentricity per bus vs K", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelsize=7)

    # right: summary table as text
    ax2 = axes[1]
    ax2.axis("off")
    summary = [
        ["Metric", "Value"],
        ["Grid", rf_info["case"]],
        ["Buses", str(rf_info["n_buses"])],
        ["Branches", str(rf_info["n_edges_undirected"])],
        ["Graph diameter", str(rf_info["graph_diameter"])],
        ["Graph radius", str(rf_info["graph_radius"])],
        ["K (correction steps)", str(rf_info["K"])],
        ["K ≥ diameter?", "YES ✓" if rf_info["K_covers_diameter"] else "NO ✗"],
    ]
    tbl = ax2.table(cellText=summary[1:], colLabels=summary[0],
                    loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif rf_info["K_covers_diameter"] and r == len(summary) - 1:
            cell.set_facecolor("#E8F5E9")
        elif not rf_info["K_covers_diameter"] and r == len(summary) - 1:
            cell.set_facecolor("#FFEBEE")
        cell.set_edgecolor("#CCCCCC")

    ax2.set_title("Receptive Field Summary", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Receptive field plot → {out_path}")


# ── plot 2: attention heat map on grid graph ──────────────────────────────────

def plot_attention_map(bus_attn: torch.Tensor, case_name: str, out_path: str,
                       rf_info: dict = None):
    n = bus_attn.shape[0]
    pos = CASE14_POS if n == 14 else {i: (i % 5, i // 5) for i in range(n)}
    branches = CASE14_BRANCHES if n == 14 else []

    attn_np = bus_attn.numpy()
    # normalize 0-1
    attn_norm = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Attention Weight Map — {case_name}\n"
                 f"(averaged over {rf_info.get('n_scenarios', '?') if rf_info else '?'} scenarios "
                 f"and K={rf_info.get('K','?') if rf_info else '?'} steps)",
                 fontsize=10, fontweight="bold")

    cmap = cm.YlOrRd

    # ── left: raw attention strength ──
    ax = axes[0]
    colors = [attn_norm[i] for i in sorted(pos)]
    sizes  = [200 + 500 * attn_norm[i] for i in sorted(pos)]
    draw_grid(ax, pos, branches, colors, sizes,
              "Attention strength per bus\n(larger + darker = more attended)",
              cmap, 0, 1, show_colorbar=True, fig=fig)

    # mark bus types
    for i, btype in CASE14_BUS_TYPES.items():
        x, y = pos[i]
        marker = "★" if btype == "slack" else "●"
        ax.annotate(marker, (x, y + 0.25), fontsize=9, ha="center",
                    color="#1565C0" if btype == "slack" else "#6A1B9A")

    ax.annotate("★ slack  ● generator", xy=(0.01, 0.01), xycoords="axes fraction",
                fontsize=7, color="#555")

    # ── right: attention vs bus type ──
    ax2 = axes[1]
    types  = []
    values = []
    for i in range(n):
        bt = CASE14_BUS_TYPES.get(i, "load")
        types.append(bt)
        values.append(float(attn_np[i]))

    type_colors = {"slack": "#1565C0", "gen": "#6A1B9A", "load": "#E65100"}
    for i, (bt, v) in enumerate(zip(types, values)):
        ax2.bar(bus_label(i), v, color=type_colors[bt], alpha=0.85, edgecolor="white", lw=0.5)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=lbl)
                  for lbl, c in type_colors.items()]
    ax2.legend(handles=legend_els, fontsize=8, title="Bus type")
    ax2.set_xlabel("Bus", fontsize=9)
    ax2.set_ylabel("Mean incoming attention", fontsize=9)
    ax2.set_title("Attention by bus (colored by type)", fontsize=9)
    ax2.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Attention map      → {out_path}")


# ── plot 3: attention evolution across K steps ────────────────────────────────

def plot_attention_evolution(attn_weights: torch.Tensor, edge_index: torch.Tensor,
                             n_nodes: int, case_name: str, out_path: str):
    """attn_weights: (N_scenarios, K, E). Show how bus attention evolves over K steps."""
    K = attn_weights.shape[1]
    E = attn_weights.shape[2]

    # aggregate to buses per step: mean over scenarios, then sum incoming per bus
    mean_over_scenarios = attn_weights.mean(dim=0)  # (K, E)

    bus_attn_per_step = torch.zeros(K, n_nodes)
    for e_idx, (src, dst) in enumerate(edge_index.tolist()):
        bus_attn_per_step[:, dst] += mean_over_scenarios[:, e_idx]

    # normalize each step independently
    row_sums = bus_attn_per_step.sum(dim=1, keepdim=True).clamp_min(1e-12)
    bus_attn_norm = bus_attn_per_step / row_sums  # (K, N)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Attention Evolution over K Steps — {case_name}", fontsize=10, fontweight="bold")

    step_indices = [0, K // 2, K - 1]
    labels = ["Step 1 (early)", f"Step {K//2+1} (mid)", f"Step {K} (final)"]
    cmap = cm.YlOrRd

    pos = CASE14_POS if n_nodes == 14 else {i: (i % 5, i // 5) for i in range(n_nodes)}
    branches = CASE14_BRANCHES if n_nodes == 14 else []

    for ax, step_i, lbl in zip(axes, step_indices, labels):
        a = bus_attn_norm[step_i].numpy()
        colors = list(a)
        sizes  = [150 + 400 * a[i] for i in range(n_nodes)]
        draw_grid(ax, pos, branches, colors, sizes, lbl,
                  cmap, 0, float(a.max()), show_colorbar=False)

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01, label="attention (normalized)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Attention evolution → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attn_dir",  default="results/attention")
    p.add_argument("--case_name", default="pglib_opf_case14_ieee")
    p.add_argument("--out_dir",   default="results/figures")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.case_name.replace("pglib_opf_", "").replace("_ieee", "")

    # load receptive field info
    rf_path = os.path.join(args.attn_dir, "receptive_field.json")
    rf_info = {}
    if os.path.exists(rf_path):
        with open(rf_path) as f:
            rf_info = json.load(f)
        plot_receptive_field(rf_info, os.path.join(args.out_dir, f"receptive_field_{tag}.png"))
    else:
        print(f"[viz] WARNING: {rf_path} not found — skipping receptive field plot")

    # load attention data
    attn_path = os.path.join(args.attn_dir, "attn_weights.pt")
    ei_path   = os.path.join(args.attn_dir, "edge_index.pt")
    bus_path  = os.path.join(args.attn_dir, "bus_attention.pt")

    if not os.path.exists(attn_path):
        print(f"[viz] WARNING: {attn_path} not found — run analyze_attention_opfdata.py first")
        return

    attn_weights = torch.load(attn_path, map_location="cpu", weights_only=False)
    edge_index   = torch.load(ei_path,   map_location="cpu", weights_only=False)
    bus_data     = torch.load(bus_path,  map_location="cpu", weights_only=False)

    bus_attn = bus_data["bus_attn"]
    n_nodes  = bus_data["n_nodes"]
    rf_info["n_scenarios"] = bus_data["n_scenarios"]
    rf_info["K"]           = bus_data["K"]

    plot_attention_map(bus_attn, args.case_name,
                       os.path.join(args.out_dir, f"attention_map_{tag}.png"), rf_info)

    plot_attention_evolution(attn_weights, edge_index, n_nodes, args.case_name,
                             os.path.join(args.out_dir, f"attention_evolution_{tag}.png"))

    print("[viz] All figures done.")


if __name__ == "__main__":
    main()

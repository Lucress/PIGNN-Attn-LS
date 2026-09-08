"""
Analysis 2 — Source-bus attention for AC-OPF.

The edge self-attention in PIGNN-Attn-LS is normalised PER DESTINATION bus
(segmented softmax), so every bus receives total attention = 1.0 by construction.
The bus-level destination story is therefore uniform.

The SOURCE side is not constrained this way: a bus that is a high-weight SENDER
across many of its outgoing edges is a genuine influential information hub for
the OPF correction process.

This script computes, for each bus b:
  source_attn[b] = mean over (scenarios, steps) of the attention weights
                   on all OUTGOING edges from b

Then compares source attention to destination attention, and to bus type (slack /
generator / load) in the context of AC-OPF.

No model inference needed — uses existing attn_weights.pt files.

Usage (from PIGNN-Attn-LS-PPC/):
    python project_source_attention.py \\
        --attn_dir_sup results/attention_supervised \\
        --attn_dir_pinn results/attention_pinn \\
        --out_dir results/source_attention
"""

from __future__ import annotations
import argparse, json, os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# ── IEEE 14-bus layout (OPF context) ─────────────────────────────────────────

CASE14_POS = {
    0: (0.0,2.0), 1: (1.5,3.0), 2: (3.5,3.5), 3: (4.5,2.5),
    4: (4.0,1.5), 5: (2.5,3.8), 6: (3.2,4.5), 7: (2.8,4.8),
    8: (3.8,4.8), 9: (4.5,4.2), 10:(4.2,3.8), 11:(4.8,3.5),
    12:(5.2,3.8), 13:(5.0,3.2),
}
CASE14_BRANCHES = [
    (0,1),(0,4),(1,2),(1,3),(1,4),(2,3),(3,4),(3,6),(3,8),
    (4,5),(5,10),(5,11),(5,12),(6,7),(6,8),(8,9),(9,10),(10,11),(11,12),(12,13)
]
# OPF bus types: who can redispatch?
CASE14_BUS_TYPE = {0:"slack", 1:"gen", 2:"gen", 5:"gen", 7:"gen"}
TYPE_COLOR = {"slack":"#1565C0", "gen":"#7B2FBE", "load":"#E65100"}
TYPE_LABEL = {"slack":"Slack (ref bus)", "gen":"Generator (PV)", "load":"Load (PQ)"}


def bus_type(idx):
    return CASE14_BUS_TYPE.get(idx, "load")


# ── core computation ──────────────────────────────────────────────────────────

def compute_source_attention(attn_weights, edge_index, n_nodes):
    """
    attn_weights: (S, K, E) — mean-head attention, all steps and scenarios
    edge_index:   (E, 2)    — [src, dst]

    Returns source_attn (N,): mean outgoing attention per source bus,
    normalised globally to sum to 1.
    """
    mean_attn = attn_weights.mean(dim=0).mean(dim=0)   # (E,) mean over S and K
    src_attn  = torch.zeros(n_nodes)
    src_count = torch.zeros(n_nodes)
    for e_idx, (src, _) in enumerate(edge_index.tolist()):
        src_attn[src]  += float(mean_attn[e_idx])
        src_count[src] += 1
    src_count = src_count.clamp_min(1)
    src_attn  = src_attn / src_count          # mean per outgoing edge
    return src_attn / src_attn.sum().clamp_min(1e-12)


def compute_dest_attention(attn_weights, edge_index, n_nodes):
    """Same but for destination buses (for comparison)."""
    mean_attn = attn_weights.mean(dim=0).mean(dim=0)
    dst_attn  = torch.zeros(n_nodes)
    dst_count = torch.zeros(n_nodes)
    for e_idx, (_, dst) in enumerate(edge_index.tolist()):
        dst_attn[dst]  += float(mean_attn[e_idx])
        dst_count[dst] += 1
    dst_count = dst_count.clamp_min(1)
    dst_attn  = dst_attn / dst_count
    return dst_attn / dst_attn.sum().clamp_min(1e-12)


# ── plotting ──────────────────────────────────────────────────────────────────

def draw_grid(ax, attn, pos, branches, title, cmap="YlOrRd", annotate_type=True):
    n = len(pos)
    a = attn.numpy()
    norm = (a - a.min()) / max(a.max() - a.min(), 1e-12)

    for (u, v) in branches:
        x0,y0=pos[u]; x1,y1=pos[v]
        ax.plot([x0,x1],[y0,y1], color="#666", lw=1.2, alpha=0.6, zorder=1)

    xs=[pos[i][0] for i in range(n)]; ys=[pos[i][1] for i in range(n)]
    sc=ax.scatter(xs,ys, c=list(norm), s=[120+380*v for v in norm],
                  cmap=cmap, vmin=0, vmax=1, zorder=2, edgecolors="white", lw=1.0)
    for i in range(n):
        ax.annotate(str(i+1),(pos[i][0],pos[i][1]), fontsize=7.5,
                    ha="center",va="center",color="white",fontweight="bold",zorder=3)

    if annotate_type:
        for i, bt in CASE14_BUS_TYPE.items():
            m = "★" if bt=="slack" else "●"
            ax.annotate(m,(pos[i][0],pos[i][1]+0.28), fontsize=9,
                        ha="center", color=TYPE_COLOR[bt])
    ax.set_title(title, fontsize=9); ax.set_aspect("equal"); ax.axis("off")
    return sc


def plot_source_vs_dest(sup_src, sup_dst, pinn_src, pinn_dst, n, out_dir):
    """Four-panel: source and destination attention for both models."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "AC-OPF Attention: Source vs Destination Bus Perspective\n"
        "Source = which bus SENDS influential information | "
        "Destination = which bus RECEIVES it",
        fontsize=10, fontweight="bold"
    )

    cmap = "YlOrRd"
    combos = [
        (axes[0,0], sup_src,  "Supervised — Source attention\n(information senders)"),
        (axes[0,1], sup_dst,  "Supervised — Destination attention\n(information receivers)"),
        (axes[1,0], pinn_src, "PINN — Source attention\n(information senders)"),
        (axes[1,1], pinn_dst, "PINN — Destination attention\n(information receivers)"),
    ]
    scs = []
    for ax, attn, title in combos:
        sc = draw_grid(ax, attn, CASE14_POS, CASE14_BRANCHES, title, cmap)
        scs.append(sc)

    # shared legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=TYPE_LABEL[t])
                  for t, c in TYPE_COLOR.items()]
    legend_els.append(Patch(facecolor="none", edgecolor="none", label="★ slack  ● generator"))
    fig.legend(handles=legend_els, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))
    plt.colorbar(scs[0], ax=axes.ravel().tolist(), shrink=0.35, pad=0.01,
                 label="attention (normalised)", location="right")
    plt.tight_layout(rect=[0,0.05,1,1])
    path = os.path.join(out_dir, "source_vs_dest_maps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def plot_source_bar(sup_src, pinn_src, n, out_dir):
    """Bar chart of source attention per bus, coloured by OPF bus type."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        "Source-Bus Attention per Bus — AC-OPF\n"
        "Which bus's voltage/angle information does the model rely on most?",
        fontsize=10, fontweight="bold"
    )

    for ax, src, label, col_hi in [
        (axes[0], sup_src,  "Supervised (K=15)", "#00CFA8"),
        (axes[1], pinn_src, "PINN (K=30)",       "#FF6B35"),
    ]:
        vals = src.numpy()
        colors = [TYPE_COLOR[bus_type(i)] for i in range(n)]
        bars = ax.bar(range(1, n+1), vals, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        # highlight top 3
        top3 = np.argsort(vals)[::-1][:3]
        for idx in top3:
            bars[idx].set_edgecolor(col_hi)
            bars[idx].set_linewidth(2.5)
            ax.text(idx+1, vals[idx]+0.001, f"#{list(np.argsort(vals)[::-1]).index(idx)+1}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=col_hi)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=c, label=TYPE_LABEL[t])
                            for t, c in TYPE_COLOR.items()], fontsize=8)
        ax.set_xlabel("Bus (1-indexed)", fontsize=10)
        ax.set_ylabel("Normalised source attention", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(1, n+1))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out_dir, "source_attention_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def plot_source_type_breakdown(sup_src, pinn_src, n, out_dir):
    """Mean source attention by OPF bus type — slack / generator / load."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    types = ["slack", "gen", "load"]
    type_buses = {t: [i for i in range(n) if bus_type(i) == t] for t in types}

    sup_means  = {t: float(sup_src[type_buses[t]].mean())  for t in types}
    pinn_means = {t: float(pinn_src[type_buses[t]].mean()) for t in types}

    x = np.arange(len(types)); w = 0.35
    b1 = ax.bar(x - w/2, [sup_means[t]  for t in types], w,
                label="Supervised", color="#00CFA8", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, [pinn_means[t] for t in types], w,
                label="PINN",       color="#FF6B35", alpha=0.85, edgecolor="white")

    for bars, vals in [(b1, sup_means), (b2, pinn_means)]:
        for bar, t in zip(bars, types):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.0003, f"{h:.4f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABEL[t] for t in types], fontsize=9)
    ax.set_ylabel("Mean source attention (normalised)", fontsize=10)
    ax.set_title(
        "Source attention by OPF bus type\n"
        "Do generators or load buses drive OPF information flow?",
        fontsize=9.5,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out_dir, "source_type_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")

    return sup_means, pinn_means


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attn_dir_sup",  required=True)
    p.add_argument("--attn_dir_pinn", required=True)
    p.add_argument("--out_dir",       required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def load(attn_dir):
        aw = torch.load(os.path.join(attn_dir,"attn_weights.pt"), weights_only=False)
        ei = torch.load(os.path.join(attn_dir,"edge_index.pt"),   weights_only=False)
        bd = torch.load(os.path.join(attn_dir,"bus_attention.pt"),weights_only=False)
        return aw, ei, bd["n_nodes"]

    print("[src] Loading saved OPF attention tensors ...")
    sup_aw,  sup_ei,  n = load(args.attn_dir_sup)
    pinn_aw, pinn_ei, _ = load(args.attn_dir_pinn)

    print(f"[src] Computing source-bus attention (n={n} buses) ...")
    sup_src  = compute_source_attention(sup_aw,  sup_ei,  n)
    pinn_src = compute_source_attention(pinn_aw, pinn_ei, n)
    sup_dst  = compute_dest_attention(sup_aw,   sup_ei,  n)
    pinn_dst = compute_dest_attention(pinn_aw,  pinn_ei, n)

    # print top-3 for each
    for label, src in [("Supervised source", sup_src), ("PINN source", pinn_src),
                        ("Supervised dest",  sup_dst),  ("PINN dest",   pinn_dst)]:
        top = src.topk(3)
        print(f"  {label}: top-3 buses (1-idx) "
              f"{[b+1 for b in top.indices.tolist()]} "
              f"vals {[round(float(v),4) for v in top.values]}")

    plot_source_vs_dest(sup_src, sup_dst, pinn_src, pinn_dst, n, args.out_dir)
    plot_source_bar(sup_src, pinn_src, n, args.out_dir)
    sup_means, pinn_means = plot_source_type_breakdown(sup_src, pinn_src, n, args.out_dir)

    results = {
        "supervised": {
            "top3_source_buses_1idx": [b+1 for b in sup_src.topk(3).indices.tolist()],
            "top3_dest_buses_1idx":   [b+1 for b in sup_dst.topk(3).indices.tolist()],
            "mean_source_by_type":    sup_means,
        },
        "pinn": {
            "top3_source_buses_1idx": [b+1 for b in pinn_src.topk(3).indices.tolist()],
            "top3_dest_buses_1idx":   [b+1 for b in pinn_dst.topk(3).indices.tolist()],
            "mean_source_by_type":    pinn_means,
        },
    }
    with open(os.path.join(args.out_dir, "source_attention_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"[src] Done -> {args.out_dir}/")


if __name__ == "__main__":
    main()

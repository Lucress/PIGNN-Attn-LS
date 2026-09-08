"""
Generator-centric attention spread analysis for AC-OPF.

Key question: For each generator bus in the 14-bus system, how does the
model distribute its outgoing edge attention, and does this distribution
concentrate onto specific physical paths as correction steps progress?

Metrics per generator bus g, per step k:
  - attention vector: alpha_g(k) = softmax-like weights on outgoing edges of g
                      (raw softmax values — unconstrained from the source side)
  - entropy H_g(k) = -sum(p * log(p))  where p = normalised outgoing attention
  - mean outgoing attention (vs other bus types)
  - Spearman r between outgoing edge attention and edge line loading

Main outputs:
  - entropy_evolution.png : H(k) per generator bus, both models
  - outgoing_spread_gen1.png : edge-by-edge attention trajectory for bus 1 (most connected gen)
  - generator_spread_results.json
"""
from __future__ import annotations
import argparse, json, os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Case14 metadata (0-indexed) ───────────────────────────────────────────────
CASE14_BUS_TYPE = {0:"slack", 1:"gen", 2:"gen", 5:"gen", 7:"gen"}
TYPE_LABEL = {"slack":"Slack (Bus 1)", "gen":"Generator", "load":"Load"}
BUS_NAMES = {
    0: "Bus 1\n(slack)",
    1: "Bus 2\n(gen)",
    2: "Bus 3\n(gen)",
    5: "Bus 6\n(gen)",
    7: "Bus 8\n(gen)",
}

GEN_BUSES = [0, 1, 2, 5, 7]   # 0-indexed generator buses
COLORS_GEN = ["#4DA6FF", "#9F7AEA", "#34D399", "#FBBF24", "#FF6B35"]


def entropy(p):
    """Shannon entropy (bits) of a probability distribution."""
    p = np.array(p, dtype=float)
    p = p / max(p.sum(), 1e-12)
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))


def outgoing_edges(ei_array, bus):
    """Return indices of all directed edges where src == bus."""
    return [e for e, (src, dst) in enumerate(ei_array) if src == bus]


def normalize_outgoing(attn_e):
    """Normalize a vector to probability distribution."""
    s = attn_e.sum()
    return attn_e / max(s, 1e-12)


# ── entropy evolution analysis ────────────────────────────────────────────────

def compute_entropy_curves(attn, ei_array, gen_buses):
    """
    attn: (S, K, E)
    Returns dict: bus_idx -> (K,) entropy curve (averaged over S scenarios)
    """
    S, K, E = attn.shape
    curves = {}
    for g in gen_buses:
        oe = outgoing_edges(ei_array, g)
        if not oe:
            continue
        oe_t = torch.tensor(oe)
        attn_out = attn[:, :, oe_t]  # (S, K, len(oe))
        # normalise each scenario-step vector, compute entropy
        H = np.zeros((S, K))
        for s in range(S):
            for k in range(K):
                p = attn_out[s, k].numpy()
                H[s, k] = entropy(normalize_outgoing(p))
        curves[g] = H.mean(axis=0)  # (K,)
    return curves


def plot_entropy_evolution(sup_curves, pinn_curves, gen_buses, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "AC-OPF: Generator Bus Outgoing Attention Entropy over Correction Steps\n"
        "Low entropy = attention concentrated on few paths · High entropy = uniform spread",
        fontsize=10, fontweight="bold"
    )

    for ax, curves, label, base_col, K_label in [
        (axes[0], sup_curves,  "Supervised", "#00CFA8", "(K=15)"),
        (axes[1], pinn_curves, "PINN",       "#FF6B35", "(K=30)"),
    ]:
        for i, g in enumerate(gen_buses):
            if g not in curves: continue
            H = curves[g]
            K = len(H)
            col = COLORS_GEN[i]
            ax.plot(range(1, K+1), H, lw=2.0, color=col,
                    label=BUS_NAMES.get(g, f"Bus {g+1}"), marker="o",
                    markersize=3.5, markevery=max(1, K//10))

        ax.set_xlabel("Correction step k", fontsize=11)
        ax.set_ylabel("Outgoing attention entropy H (nats)", fontsize=10)
        ax.set_title(f"{label} {K_label}", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(out_dir, "entropy_evolution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ── edge-by-edge attention trajectory for most-connected generator ────────────

def plot_edge_trajectory(attn, ei_array, bus, label, out_dir, col):
    """
    For a single generator bus, plot each outgoing edge's attention over steps k.
    """
    oe = outgoing_edges(ei_array, bus)
    if not oe:
        print(f"  [warn] no outgoing edges for bus {bus}")
        return

    S, K, E = attn.shape
    oe_t = torch.tensor(oe)
    attn_out = attn[:, :, oe_t].mean(dim=0).numpy()  # (K, n_out)

    n_out = len(oe)
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.get_cmap("tab10", n_out)
    for j, e_id in enumerate(oe):
        dst = ei_array[e_id][1]
        dst_type = CASE14_BUS_TYPE.get(dst, "load")
        lbl = f"Edge {bus+1}→{dst+1} ({dst_type})"
        ax.plot(range(1, K+1), attn_out[:, j], lw=2.0,
                color=cmap(j), label=lbl, marker="o", markersize=3.5,
                markevery=max(1, K//10))

    ax.set_xlabel("Correction step k", fontsize=11)
    ax.set_ylabel("Outgoing edge attention (mean over scenarios)", fontsize=10)
    ax.set_title(
        f"{label} — Bus {bus+1} (generator) outgoing attention per edge\n"
        f"How does the generator node route its information across K correction steps?",
        fontsize=9.5,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(1, K)
    plt.tight_layout()
    path = os.path.join(out_dir, f"outgoing_edge_bus{bus+1}_{label.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")
    return attn_out


def plot_edge_trajectory_both(sup_attn, pinn_attn, ei_array, bus, out_dir):
    """Side-by-side supervised vs PINN outgoing edge trajectories for one generator."""
    oe = outgoing_edges(ei_array, bus)
    if not oe:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"AC-OPF — Generator Bus {bus+1}: outgoing attention allocation across correction steps\n"
        f"Which downstream buses does the generator 'talk to' most at each step?",
        fontsize=10, fontweight="bold"
    )

    cmap = plt.cm.get_cmap("tab10", len(oe))
    for ax, attn, label, lw_main in [
        (axes[0], sup_attn,  "Supervised (K=15)", 2.2),
        (axes[1], pinn_attn, "PINN (K=30)",       2.2),
    ]:
        S, K, E = attn.shape
        attn_out = attn[:, :, torch.tensor(oe)].mean(dim=0).numpy()  # (K, n_out)
        for j, e_id in enumerate(oe):
            dst = ei_array[e_id][1]
            dst_type = CASE14_BUS_TYPE.get(dst, "load")
            lbl = f"→ Bus {dst+1} ({dst_type})"
            ax.plot(range(1, K+1), attn_out[:, j], lw=lw_main, color=cmap(j),
                    label=lbl, marker="o", markersize=4, markevery=max(1, K//8))
        # also show entropy as shaded background
        H_vals = []
        for k in range(K):
            H_vals.append(entropy(normalize_outgoing(attn_out[k])))
        ax_r = ax.twinx()
        ax_r.fill_between(range(1, K+1), H_vals, alpha=0.08, color="#FBBF24")
        ax_r.plot(range(1, K+1), H_vals, lw=1.5, color="#FBBF24", ls="--", alpha=0.6)
        ax_r.set_ylabel("Entropy H (nats)", color="#FBBF24", fontsize=9)
        ax_r.tick_params(axis='y', colors="#FBBF24")

        ax.set_xlabel("Correction step k", fontsize=11)
        ax.set_ylabel("Attention weight (mean over scenarios)", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=9, loc="right")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(1, K)

    plt.tight_layout()
    path = os.path.join(out_dir, f"outgoing_edge_bus{bus+1}_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ── mean outgoing attention by bus type ───────────────────────────────────────

def compute_mean_outgoing_by_type(attn, ei_array, n_nodes):
    """
    For each source bus, compute mean outgoing attention (averaged over S, K, outgoing edges).
    Then group by bus type.
    Returns dict: type -> mean
    """
    S, K, E = attn.shape
    mean_attn = attn.mean(dim=0).mean(dim=0)  # (E,)
    src_attn = {i: [] for i in range(n_nodes)}
    for e_id, (src, dst) in enumerate(ei_array):
        src_attn[src].append(float(mean_attn[e_id]))
    by_type = {"slack": [], "gen": [], "load": []}
    for b in range(n_nodes):
        btype = CASE14_BUS_TYPE.get(b, "load")
        if src_attn[b]:
            by_type[btype].append(np.mean(src_attn[b]))
    return {t: float(np.mean(v)) if v else 0.0 for t, v in by_type.items()}


def plot_type_comparison(sup_by_type, pinn_by_type, out_dir):
    types = ["slack", "gen", "load"]
    labels = ["Slack\n(Bus 1)", "Generators\n(Bus 2,3,6,8)", "Load buses\n(Bus 4,5,7,9–14)"]
    x = np.arange(len(types))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w/2, [sup_by_type[t]  for t in types], w,
                label="Supervised", color="#00CFA8", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, [pinn_by_type[t] for t in types], w,
                label="PINN",       color="#FF6B35", alpha=0.85, edgecolor="white")

    for bars, by_type in [(b1, sup_by_type), (b2, pinn_by_type)]:
        for bar, t in zip(bars, types):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.001, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean outgoing edge attention", fontsize=10)
    ax.set_title(
        "AC-OPF Outgoing Attention by Bus Type\n"
        "Do generator nodes send more information than load nodes?",
        fontsize=9.5
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out_dir, "outgoing_by_type.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ── attention-flow correlation (outgoing edge attention vs line loading) ───────

def load_flows(deep_dir):
    fp = os.path.join(deep_dir, "line_flows.pt")
    if not os.path.exists(fp):
        return None
    return torch.load(fp, weights_only=False)  # (S, E)


def generator_flow_correlation(attn, flows, ei_array, bus, label):
    """
    For a generator bus g, correlate its outgoing edge attention with line loading on those edges.
    Pooled over all S scenarios.
    """
    oe = outgoing_edges(ei_array, bus)
    if not oe or flows is None:
        return None

    S, K, E = attn.shape
    attn_mean = attn.mean(dim=1)  # (S, E) — mean over K
    results = {}
    for e_id in oe:
        dst = ei_array[e_id][1]
        a_e = attn_mean[:, e_id].numpy()     # (S,)
        f_e = flows[:, e_id].numpy()         # (S,)
        if np.std(a_e) > 1e-9 and np.std(f_e) > 1e-9:
            r, p = stats.spearmanr(a_e, f_e)
            results[f"{bus+1}→{dst+1}"] = {"r": float(r), "p": float(p)}
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attn_sup",   default="results/attention_supervised")
    p.add_argument("--attn_pinn",  default="results/attention_pinn")
    p.add_argument("--deep_sup",   default="results/deep_supervised")
    p.add_argument("--deep_pinn",  default="results/deep_pinn")
    p.add_argument("--out_dir",    default="results/generator_spread")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[gen] Loading attention tensors ...")
    aw_sup  = torch.load(os.path.join(args.attn_sup,  "attn_weights.pt"), weights_only=False)
    aw_pinn = torch.load(os.path.join(args.attn_pinn, "attn_weights.pt"), weights_only=False)
    ei      = torch.load(os.path.join(args.attn_sup,  "edge_index.pt"),   weights_only=False)

    ei_array = ei.tolist()  # list of [src, dst]
    S_sup, K_sup, E = aw_sup.shape
    S_pinn, K_pinn, _ = aw_pinn.shape
    print(f"  Sup:  S={S_sup} K={K_sup} E={E}")
    print(f"  PINN: S={S_pinn} K={K_pinn} E={E}")

    n_nodes = 14
    gen_buses = [g for g in GEN_BUSES if outgoing_edges(ei_array, g)]
    print(f"  Generator buses with outgoing edges: {[g+1 for g in gen_buses]} (1-indexed)")
    for g in gen_buses:
        oe = outgoing_edges(ei_array, g)
        print(f"    Bus {g+1}: outgoing edges -> {[(ei_array[e][0]+1, ei_array[e][1]+1) for e in oe]}")

    # 1. Entropy evolution
    print("[gen] Computing entropy curves ...")
    sup_H  = compute_entropy_curves(aw_sup,  ei_array, gen_buses)
    pinn_H = compute_entropy_curves(aw_pinn, ei_array, gen_buses)
    plot_entropy_evolution(sup_H, pinn_H, gen_buses, args.out_dir)

    # 2. Edge trajectory for bus with most outgoing edges
    deg = {g: len(outgoing_edges(ei_array, g)) for g in gen_buses}
    top_gen = max(deg, key=deg.get)
    print(f"[gen] Most-connected generator: Bus {top_gen+1} ({deg[top_gen]} outgoing edges)")
    plot_edge_trajectory_both(aw_sup, aw_pinn, ei_array, top_gen, args.out_dir)

    # also do bus 7 (0-indexed) = bus 8 (top source) if different
    for g in gen_buses:
        if g != top_gen:
            oe = outgoing_edges(ei_array, g)
            if len(oe) >= 1:
                plot_edge_trajectory_both(aw_sup, aw_pinn, ei_array, g, args.out_dir)

    # 3. Mean outgoing by bus type
    print("[gen] Computing mean outgoing attention by bus type ...")
    sup_btype  = compute_mean_outgoing_by_type(aw_sup,  ei_array, n_nodes)
    pinn_btype = compute_mean_outgoing_by_type(aw_pinn, ei_array, n_nodes)
    plot_type_comparison(sup_btype, pinn_btype, args.out_dir)
    print(f"  Supervised by type: {sup_btype}")
    print(f"  PINN by type:       {pinn_btype}")

    # 4. Generator-specific flow correlation
    flows_sup  = load_flows(args.deep_sup)
    flows_pinn = load_flows(args.deep_pinn)
    gen_corr = {}
    for g in gen_buses:
        sup_r  = generator_flow_correlation(aw_sup,  flows_sup,  ei_array, g, "Supervised")
        pinn_r = generator_flow_correlation(aw_pinn, flows_pinn, ei_array, g, "PINN")
        gen_corr[f"bus{g+1}"] = {"supervised": sup_r, "pinn": pinn_r}
        if sup_r:
            print(f"  Bus {g+1} edge correlations (Supervised): {sup_r}")
        if pinn_r:
            print(f"  Bus {g+1} edge correlations (PINN):       {pinn_r}")

    # 5. Entropy summary: first vs last step
    print("\n[gen] Entropy summary (first vs last step):")
    entropy_summary = {}
    for g in gen_buses:
        name = f"bus{g+1}"
        entry = {}
        for label, curves in [("supervised", sup_H), ("pinn", pinn_H)]:
            if g in curves:
                H = curves[g]
                entry[label] = {
                    "first": float(H[0]),
                    "last":  float(H[-1]),
                    "delta": float(H[-1] - H[0]),
                    "mean":  float(H.mean()),
                }
                print(f"  {name} {label}: H_1={H[0]:.4f}  H_K={H[-1]:.4f}  "
                      f"Δ={H[-1]-H[0]:+.4f} "
                      f"({'concentrating' if H[-1]<H[0] else 'spreading'})")
        entropy_summary[name] = entry

    results = {
        "entropy_summary": entropy_summary,
        "mean_outgoing_by_type": {
            "supervised": sup_btype,
            "pinn": pinn_btype,
        },
        "generator_edge_flow_correlation": gen_corr,
    }
    with open(os.path.join(args.out_dir, "generator_spread_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[gen] Done -> {args.out_dir}/")


if __name__ == "__main__":
    main()

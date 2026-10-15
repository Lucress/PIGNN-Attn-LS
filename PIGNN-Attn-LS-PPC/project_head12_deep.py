"""
Deep analysis of Head 1 and Head 2 attention patterns.

Investigates:
  1. Per-step correlation of H1 and H2 vs physics quantities across K=30 steps
  2. Engineering units: what are the actual MW / pu scales of the quantities
  3. Partial correlations: r(attn, P | Δθ) and r(attn, Δθ | P) to separate
     the P-angle coupling (P ≈ B * Δθ for small angles, so they are not independent)
  4. Head alignment: how similar H1 and H2 are to each other (dot product / cosine)

Requires:
  - attn_weights.pt from Step 1 of XAI pipeline   (N_scen, K, E, H) or (N_scen, K, E)
  - edge_index.pt from Step 1
  - OPFData test scenarios for physics quantities

Usage:
    python project_head12_deep.py \
        --attn_dir  results/attention/pignn_supervised_case118 \
        --attn_dir2 results/attention/pignn_global_case118 \
        --opfdata_root ./opfdata \
        --case_name pglib_opf_case118_ieee \
        --out_dir results/head12_analysis
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── physical quantities on edges ─────────────────────────────────────────────

@torch.no_grad()
def compute_edge_physics(model, test_loader, edge_index, n_scenarios: int, device):
    """Compute per-edge physics quantities for each test scenario.

    Returns dict with arrays (N_scen, E):
      - P_ij:     active power flow (pu)
      - Q_ij:     reactive power flow (pu)
      - dtheta:   |voltage angle difference| between endpoint buses (rad)
      - dV:       |voltage magnitude difference| (pu)
      - P_MW:     active power in engineering units (to check scaling)
    """
    from opf_task import opf_decision_space, implied_injection
    from train_valid_test_pignn_opf import branch_rows_from_opfdata

    src = edge_index[:, 0].numpy()
    dst = edge_index[:, 1].numpy()
    E = len(src)

    all_P, all_Q, all_dth, all_dV = [], [], [], []

    for i, batch in enumerate(test_loader):
        if i >= n_scenarios:
            break
        try:
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in batch.items() if k != "hetero"}
            hb = batch["hetero"].to(device)

            # use the TRUE optimal voltage (V_newton = target)
            V_true = bd.get("V_newton", bd.get("V_target"))
            if V_true is None:
                continue

            # V_true shape: (1, N, 2) — [mag, angle]
            mag = V_true[0, :, 0].numpy()  # (N,)
            ang = V_true[0, :, 1].numpy()  # (N,) in radians

            dth = np.abs(ang[src] - ang[dst])     # (E,)
            dV  = np.abs(mag[src] - mag[dst])      # (E,)

            # Apparent power on each edge from Y-bus
            Y = bd["Ybus"]  # (1, N, N) complex
            Vc = (mag * np.exp(1j * ang))          # (N,) complex
            Vc_t = torch.tensor(Vc, dtype=torch.complex128)
            if Y.is_sparse:
                Yu = torch.sparse.mm(Y.squeeze(0), Vc_t.unsqueeze(1)).squeeze(1)
            else:
                Yu = (Y.squeeze(0).to(torch.complex128) @ Vc_t)
            S_bus = Vc_t * Yu.conj()    # (N,) complex injection at each bus

            # branch power: approximate from bus injections at endpoints
            P_edge = (S_bus.real[dst] - S_bus.real[src]).abs().numpy()
            Q_edge = (S_bus.imag[dst] - S_bus.imag[src]).abs().numpy()

            all_P.append(P_edge)
            all_Q.append(Q_edge)
            all_dth.append(dth)
            all_dV.append(dV)
        except Exception as e:
            print(f"  scenario {i}: {e}")
            continue

    return {
        "P_ij":   np.stack(all_P),    # (N_scen, E)
        "Q_ij":   np.stack(all_Q),
        "dtheta": np.stack(all_dth),
        "dV":     np.stack(all_dV),
    }


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r over flattened x, y."""
    x = x.ravel(); y = y.ravel()
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).mean()
    denom = x.std() * y.std()
    return float(num / (denom + 1e-12))


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson r(x, y | z) — correlation of x and y with z removed."""
    def resid(a, b):
        b = b.ravel()
        a = a.ravel()
        slope = np.cov(a, b)[0, 1] / (b.var() + 1e-12)
        return a - slope * b
    return pearson_r(resid(x, z), resid(y, z))


def load_attn(attn_dir: str):
    """Load attn_weights.pt. Returns (N_scen, K, E, H) or expands to H=4 if needed."""
    path = os.path.join(attn_dir, "attn_weights.pt")
    attn = torch.load(path, map_location="cpu", weights_only=False)
    # may be (N, K, E) or (N, K, E, H)
    if attn.dim() == 3:
        attn = attn.unsqueeze(-1)   # treat as 1-head average → (N, K, E, 1)
    return attn.float().numpy()     # (N, K, E, H)


def analyse_model(name: str, attn: np.ndarray, phys: dict, out_dir: str):
    """Full H1/H2 analysis for one model. attn: (N, K, E_attn, H)."""
    N, K, E_attn, H = attn.shape
    E_phys = phys["P_ij"].shape[1]
    E = min(E_attn, E_phys)    # align edge counts in case of mismatch
    if E_attn != E_phys:
        print(f"  [warn] E_attn={E_attn} != E_phys={E_phys}; truncating both to {E}")
    print(f"\n[{name}] shape: N={N}, K={K}, E_attn={E_attn}, E_phys={E_phys}, E_used={E}, H={H}")

    qty_names = ["P_ij", "Q_ij", "dtheta", "dV"]
    qty_labels = ["|Active power| (pu)", "|Reactive power| (pu)",
                  "|Angle diff| (rad)", "|Voltage diff| (pu)"]
    qtys = [phys["P_ij"][:, :E], phys["Q_ij"][:, :E],
            phys["dtheta"][:, :E], phys["dV"][:, :E]]

    # ── engineering units summary ──────────────────────────────────────────
    print(f"  Engineering units (test set statistics):")
    S_base_MVA = 100.0  # standard per-unit base for pglib cases
    print(f"    |P_ij| mean={phys['P_ij'].mean():.4f} pu  = {phys['P_ij'].mean()*S_base_MVA:.1f} MW")
    print(f"    |P_ij| max ={phys['P_ij'].max():.4f} pu  = {phys['P_ij'].max()*S_base_MVA:.1f} MW")
    print(f"    |Q_ij| mean={phys['Q_ij'].mean():.4f} pu  = {phys['Q_ij'].mean()*S_base_MVA:.1f} MVAR")
    print(f"    |Δθ|  mean={phys['dtheta'].mean():.4f} rad = {np.degrees(phys['dtheta'].mean()):.2f} deg")
    print(f"    |ΔV|  mean={phys['dV'].mean():.4f} pu")
    print(f"    NOTE: P and Δθ correlation (expected coupling): r = {pearson_r(phys['P_ij'], phys['dtheta']):.3f}")

    # ── per-step correlation for H1 and H2 ────────────────────────────────
    results = {}
    for h in [0, 1, 2, 3]:
        if h >= H:
            continue
        label = f"H{h}"
        step_corr = {}
        for qname, q in zip(qty_names, qtys):
            step_rs = []
            for k in range(K):
                a_k = attn[:, k, :E, h].ravel()  # (N*E,) — truncated to shared edge set
                q_k = q.ravel()
                step_rs.append(pearson_r(a_k, q_k))
            step_corr[qname] = step_rs
        results[label] = step_corr

    # ── partial correlations (only for H1 and H2) ─────────────────────────
    # Ask: does H1/H2 track P independently of Δθ, and vice versa?
    partial = {}
    for h in [1, 2]:
        if h >= H:
            continue
        label = f"H{h}"
        # average attention over all steps and scenarios for simplicity
        a_avg = attn[:, :, :E, h].mean(axis=1).ravel()  # (N*E,) — truncated
        P_flat = phys["P_ij"][:, :E].ravel()
        T_flat = phys["dtheta"][:, :E].ravel()

        r_P_given_T = partial_corr(a_avg, P_flat, T_flat)
        r_T_given_P = partial_corr(a_avg, T_flat, P_flat)
        partial[label] = {
            "r(attn,P)": pearson_r(a_avg, P_flat),
            "r(attn,dθ)": pearson_r(a_avg, T_flat),
            "r(attn,P|dθ)": r_P_given_T,
            "r(attn,dθ|P)": r_T_given_P,
        }
        print(f"\n  {label} partial correlations:")
        for k, v in partial[label].items():
            print(f"    {k:20s}: {v:+.4f}")
        if abs(r_T_given_P) > abs(r_P_given_T):
            print(f"    → {label} primarily tracks Δθ (angle difference drives the signal)")
        else:
            print(f"    → {label} primarily tracks P (active power drives the signal)")

    # ── head alignment: H1 vs H2 cosine similarity per step ───────────────
    if H >= 3:
        cosines = []
        for k in range(K):
            h1 = attn[:, k, :, 1].ravel()
            h2 = attn[:, k, :, 2].ravel()
            cos = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-12)
            cosines.append(float(cos))
        print(f"\n  H1 vs H2 cosine similarity: mean={np.mean(cosines):.3f}, "
              f"min={np.min(cosines):.3f}, max={np.max(cosines):.3f}")
        if np.mean(cosines) > 0.9:
            print("  → H1 and H2 are highly aligned — they may be redundant")
        else:
            print("  → H1 and H2 are distinct — they track different patterns")

    # ── plots ──────────────────────────────────────────────────────────────
    safe_name = name.replace("/", "_").replace(" ", "_")
    steps = np.arange(1, K + 1)

    # Plot 1: per-step correlation for H1 and H2 across all 4 physics quantities
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{name} — Head 1 & Head 2 per-step correlation", fontsize=13)
    colors = {"H1": "#8b5cf6", "H2": "#0dab76", "H0": "#3478e0", "H3": "#e04040"}
    for ax, (qname, qlabel) in zip(axes.flat, zip(qty_names, qty_labels)):
        for h in [1, 2]:
            if f"H{h}" not in results:
                continue
            ax.plot(steps, results[f"H{h}"][qname], label=f"H{h}",
                    color=colors[f"H{h}"], linewidth=2)
        # also show H0 and H3 faded for reference
        for h in [0, 3]:
            if f"H{h}" not in results:
                continue
            ax.plot(steps, results[f"H{h}"][qname], label=f"H{h}",
                    color=colors[f"H{h}"], linewidth=1, alpha=0.35, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_xlabel("Newton step k", fontsize=10)
        ax.set_ylabel("Pearson r", fontsize=10)
        ax.set_title(qlabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{safe_name}_head12_step_corr.png"), dpi=150)
    plt.close(fig)

    # Plot 2: partial correlation bar chart for H1 and H2
    if partial:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fig2.suptitle(f"{name} — Partial correlations (H1, H2)", fontsize=13)
        heads = [f"H{h}" for h in [1, 2] if f"H{h}" in partial]
        bar_labels = ["r(attn,P)", "r(attn,Δθ)", "r(attn,P|Δθ)", "r(attn,Δθ|P)"]
        x = np.arange(len(bar_labels))
        width = 0.35
        for i, hl in enumerate(heads):
            vals = [partial[hl][k] for k in bar_labels]
            ax2.bar(x + i * width, vals, width, label=hl,
                    color=colors[hl], alpha=0.85)
        ax2.axhline(0, color="gray", linewidth=0.8)
        ax2.set_xticks(x + width / 2)
        ax2.set_xticklabels(bar_labels, fontsize=10)
        ax2.set_ylabel("Pearson r", fontsize=10)
        ax2.legend()
        ax2.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig2.savefig(os.path.join(out_dir, f"{safe_name}_partial_corr.png"), dpi=150)
        plt.close(fig2)

    # save JSON summary
    summary = {"name": name, "results": results, "partial": partial}
    with open(os.path.join(out_dir, f"{safe_name}_head12.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attn_dirs",     nargs="+", required=True,
                   help="one or more attn_dir paths (each contains attn_weights.pt + edge_index.pt)")
    p.add_argument("--model_names",   nargs="+", default=None,
                   help="display names matching --attn_dirs; defaults to directory names")
    p.add_argument("--opfdata_root",  required=True)
    p.add_argument("--case_name",     default="pglib_opf_case118_ieee")
    p.add_argument("--n_scenarios",   type=int, default=50)
    p.add_argument("--out_dir",       default="results/head12_analysis")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    names = args.model_names or [Path(d).name for d in args.attn_dirs]

    # load physics from OPFData (computed once, shared across models)
    print(f"[head12] Loading OPFData: {args.case_name}")
    from opfdata_pipeline import make_opfdata_loaders
    _, _, _, test_loader = make_opfdata_loaders(
        args.opfdata_root, args.case_name, batch_size=1,
        max_train=1, max_valid=1, max_test=args.n_scenarios, num_workers=0,
    )

    # load edge_index from first model (same graph topology across all models)
    edge_index = torch.load(
        os.path.join(args.attn_dirs[0], "edge_index.pt"),
        map_location="cpu", weights_only=False
    )

    print("[head12] Computing edge physics quantities ...")
    phys = compute_edge_physics(None, test_loader, edge_index, args.n_scenarios, device)
    print(f"  Got physics for {len(phys['P_ij'])} scenarios, {phys['P_ij'].shape[1]} edges")

    # analyse each model
    all_summaries = {}
    for attn_dir, name in zip(args.attn_dirs, names):
        attn = load_attn(attn_dir)
        # align scenarios
        n = min(attn.shape[0], len(phys["P_ij"]))
        attn_trim = attn[:n]
        phys_trim = {k: v[:n] for k, v in phys.items()}
        summary = analyse_model(name, attn_trim, phys_trim, args.out_dir)
        all_summaries[name] = summary

    # comparative plot: H1 and H2 dθ-correlation across models at each step
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("H1 and H2 — |Angle diff| correlation per Newton step", fontsize=13)
    model_colors = {
        "Supervised": "#3478e0", "Global-Attn": "#8b5cf6", "k-hop=2": "#0dab76",
        "Partial-Cost": "#e04040", "Phys+MSE": "#0aa5d4",
    }
    for ax, hname in zip(axes, ["H1", "H2"]):
        for name, summary in all_summaries.items():
            if hname not in summary["results"]:
                continue
            color = next((v for k, v in model_colors.items() if k.lower() in name.lower()),
                         "#888888")
            steps = np.arange(1, len(summary["results"][hname]["dtheta"]) + 1)
            ax.plot(steps, summary["results"][hname]["dtheta"],
                    label=name, color=color, linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_title(f"{hname} correlation with |Δθ| across Newton steps", fontsize=11)
        ax.set_xlabel("Newton step k", fontsize=10)
        ax.set_ylabel("Pearson r", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "comparison_h1h2_dtheta.png"), dpi=150)
    plt.close(fig)

    print(f"\n[head12] Done. Results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

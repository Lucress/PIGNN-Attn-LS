"""
Per-head attention analysis + loss landscape interpretability for PDL-v2/v3.

Part 1 — Head specialization (4 heads, n_heads=4):
  Uses model's built-in return_attn_heads=True to extract (E, H) per Newton step.
  For each head independently: per-bus incoming attention, entropy, cross-head
  correlation, and Pearson correlation with predicted voltage magnitude.
  Key question: do individual heads specialise even when the aggregate is uniform?

Part 2 — Loss landscape / training dynamics:
  Parse training log (--log_file) to plot:
    loss components, lambda trajectories, constraint phase portrait, feasibility curve.

Usage (from PIGNN-Attn-LS-PPC/):
    python analyze_attention_heads.py \\
        --ckpt results/ckpt/tinyx/pignn_pdl3_case118_best.pt \\
        --opfdata_root ./opfdata \\
        --case_name pglib_opf_case118_ieee \\
        --n_scenarios 200 \\
        --log_file results/logs/tinyx/pignn_pdl3_case118.log \\
        --out_dir results/xai_heads/pdl3
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ── model loading (identical to analyze_attention_cost.py) ─────────────────────

def load_model(ckpt_path, K, d, d_hi, n_heads, num_attn_layers):
    from GNSMsg_SelfAttention_armijo_opf import GNSMsg_EdgeSelfAttn_OPF
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = (ckpt["model_state_dict"]
             if isinstance(ckpt, dict) and "model_state_dict" in ckpt
             else ckpt)
    model = GNSMsg_EdgeSelfAttn_OPF(
        d=d, d_hi=d_hi, K=K, n_heads=n_heads,
        num_attn_layers=num_attn_layers,
        pinn=True, use_armijo=True,
        armijo_mode="geometric_safe", use_global_attn=False,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ── run_scenario with per-head attention (return_attn_heads=True) ─────────────

@torch.no_grad()
def run_scenario_heads(model, batch, device):
    """
    Calls the model with return_attn=True, return_attn_heads=True.
    The OPF model (GNSMsg_EdgeSelfAttn_OPF) natively supports this flag:
    it returns collected_attn as list of K tensors each (E, H) — one per
    Newton step — instead of head-averaged (E,).
    """
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
        return_attn_heads=True,   # ← (E, H) per Newton step instead of (E,)
    )

    out, _phys, attn_list, edge_index = result
    # attn_list: list of K tensors, each (E, H)
    V       = out.unsqueeze(0) if out.dim() == 2 else out
    n_nodes = int(batch["sizes"].sum().item())
    vm      = V[0, :, 0].cpu().numpy()
    return attn_list, edge_index, vm, n_nodes


# ── collect per-head attention across scenarios ───────────────────────────────

def collect_head_attention(model, loader, device, n_scenarios, n_heads):
    """
    Returns:
        alpha_mean_bus: (N, H) — mean incoming attention per bus per head
        vm_mean:        (N,)   — mean predicted voltage magnitude
        n_nodes:        int
        edge_index:     (E, 2) tensor
    """
    alpha_sum         = None   # (N, H)
    alpha_cnt         = 0
    vm_sum            = None
    n_nodes           = None
    edge_index_stored = None

    n_done = 0
    for batch in loader:
        if n_done >= n_scenarios:
            break

        try:
            attn_list, edge_index, vm, nn = run_scenario_heads(model, batch, device)
        except Exception as e:
            print(f"  scenario {n_done}: ERROR {e}")
            continue

        if not attn_list:
            print(f"  scenario {n_done}: empty attn_list, skipping")
            n_done += 1
            continue

        # attn_list: K × (E, H)  — stack and mean over K
        alpha_keh = torch.stack(attn_list, dim=0)   # (K, E, H)
        alpha_eh  = alpha_keh.mean(dim=0)            # (E, H)

        if n_nodes is None:
            n_nodes = nn
            vm_sum  = vm.copy()
            if isinstance(edge_index, torch.Tensor):
                edge_index_stored = edge_index.cpu()
        else:
            vm_sum += vm

        # dst node for each edge
        if edge_index_stored is not None:
            ei = edge_index_stored
            dst_e = ei[1] if (ei.dim() == 2 and ei.shape[0] == 2) else ei[:, 1]
        else:
            n_done += 1
            continue

        if alpha_sum is None:
            alpha_sum = np.zeros((n_nodes, n_heads))

        alpha_np = alpha_eh.numpy()   # (E, H)
        for h in range(n_heads):
            np.add.at(alpha_sum[:, h], dst_e.numpy(), alpha_np[:, h])
        alpha_cnt += 1

        n_done += 1
        print(f"  scenario {n_done}/{n_scenarios}", end="\r", flush=True)

    print()

    if alpha_sum is None or alpha_cnt == 0:
        raise RuntimeError(
            "No per-head attention captured. Check that the model supports "
            "return_attn_heads=True (GNSMsg_EdgeSelfAttn_OPF with num_attn_layers>=1)."
        )

    alpha_mean_bus = alpha_sum / alpha_cnt
    vm_mean        = vm_sum / n_done if vm_sum is not None else np.ones(n_nodes)
    return alpha_mean_bus, vm_mean, n_nodes, edge_index_stored


# ── Part 1: head specialization plots ─────────────────────────────────────────

def analyze_heads(alpha_mean, vm_mean, out_dir, n_heads):
    N = alpha_mean.shape[0]
    uniform_ref = 1.0 / N
    colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]
    head_stats = {}

    # ── 1. Per-head top-20 bar chart ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("PDL-v2: Per-Head Attention Distribution (case118, 200 scenarios)",
                 fontsize=13, fontweight="bold")

    for h in range(n_heads):
        ax = axes[h // 2][h % 2]
        attn_h = alpha_mean[:, h]
        order  = np.argsort(attn_h)[::-1]
        top20  = order[:20]

        # Entropy
        p            = attn_h / (attn_h.sum() + 1e-12)
        entropy      = float(-np.sum(p * np.log(p + 1e-12)))
        max_entropy  = float(np.log(N))
        norm_entropy = entropy / max_entropy

        head_stats[f"head_{h+1}"] = {
            "entropy":      entropy,
            "norm_entropy": norm_entropy,
            "max_bus":      int(order[0]),
            "max_attn":     float(attn_h[order[0]]),
            "top5_buses":   order[:5].tolist(),
        }

        ax.bar(range(20), attn_h[top20], color=colors[h], alpha=0.8)
        ax.axhline(uniform_ref, color="gray", ls="--", lw=1,
                   label=f"uniform={uniform_ref:.5f}")
        ax.set_title(f"Head {h+1}  (H/H_max={norm_entropy:.4f})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Top-20 buses (ranked by attention)")
        ax.set_ylabel("Mean incoming attention")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_attn_per_head.png"), dpi=150)
    plt.close()
    print("  saved head_attn_per_head.png")

    # ── 2. Cross-head Pearson correlation matrix ───────────────────────────────
    corr_mat = np.corrcoef(alpha_mean.T)   # (H, H)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n_heads)); ax.set_xticklabels([f"H{h+1}" for h in range(n_heads)])
    ax.set_yticks(range(n_heads)); ax.set_yticklabels([f"H{h+1}" for h in range(n_heads)])
    for i in range(n_heads):
        for j in range(n_heads):
            col = "white" if abs(corr_mat[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{corr_mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=11, color=col)
    plt.colorbar(im, ax=ax, label="Pearson r (bus attn)")
    ax.set_title("Cross-Head Correlation\n(r≈1 → redundant; r<0.5 → specialised)",
                 fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_diversity.png"), dpi=150)
    plt.close()
    print("  saved head_diversity.png")

    # Add cross-head corr to stats
    for h in range(n_heads):
        head_stats[f"head_{h+1}"]["cross_head_corr"] = {
            f"vs_H{h2+1}": float(corr_mat[h, h2])
            for h2 in range(n_heads) if h2 != h
        }

    # ── 3. Each head vs predicted voltage magnitude ────────────────────────────
    fig, axes = plt.subplots(1, n_heads, figsize=(14, 4), sharey=False)
    for h in range(n_heads):
        ax = axes[h]
        attn_h = alpha_mean[:, h]
        r, pv  = pearsonr(vm_mean, attn_h)
        ax.scatter(vm_mean, attn_h, s=18, alpha=0.7, color=colors[h])
        ax.set_xlabel("|V| mean predicted (pu)")
        ax.set_ylabel(f"Head {h+1} attention")
        ax.set_title(f"Head {h+1}\nr={r:.3f}, p={pv:.3f}", fontsize=9)
        head_stats[f"head_{h+1}"]["r_vs_vm"]  = float(r)
        head_stats[f"head_{h+1}"]["p_vs_vm"]  = float(pv)
    plt.suptitle("Head Attention vs Predicted Voltage Magnitude", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_vs_voltage.png"), dpi=150)
    plt.close()
    print("  saved head_vs_voltage.png")

    # ── 4. Entropy bar chart ───────────────────────────────────────────────────
    entropies = [head_stats[f"head_{h+1}"]["norm_entropy"] for h in range(n_heads)]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(1, n_heads + 1), entropies, color=colors, alpha=0.85)
    ax.axhline(1.0, color="gray", ls="--", lw=1.2, label="max entropy (uniform)")
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Normalised Entropy H/H_max")
    ax.set_title("Head Entropy: Uniform vs Focused Attention", fontweight="bold")
    ax.set_xticks(range(1, n_heads + 1))
    ax.set_xticklabels([f"Head {h+1}" for h in range(n_heads)])
    ax.set_ylim(0, 1.15)
    for bar, v in zip(bars, entropies):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_entropy.png"), dpi=150)
    plt.close()
    print("  saved head_entropy.png")

    return head_stats


# ── Part 2: training log parsing + plots ──────────────────────────────────────

LOG_EP_RE   = re.compile(
    r"Epoch\s+(\d+)\s*\|.*?loss\s+([\d.eE+\-]+).*?mse\s+([\d.eE+\-]+)"
    r".*?balance\(dPinf\s+([\d.eE+\-]+).*?limits\(P\s+([\d.eE+\-]+)",
    re.IGNORECASE,
)
LAM_RE      = re.compile(r"λbal=([\d.eE+\-]+)\s+λlim=([\d.eE+\-]+)", re.IGNORECASE)
VAL_BAL_RE  = re.compile(r"valid.*?balance\(dPinf\s+([\d.eE+\-]+).*?dQinf\s+([\d.eE+\-]+)",
                          re.IGNORECASE)
VAL_LIM_RE  = re.compile(r"valid.*?limits\(P\s+([\d.eE+\-]+).*?Q\s+([\d.eE+\-]+)",
                          re.IGNORECASE)
FEASIB_RE   = re.compile(r"feasib.*?([\d.]+)%", re.IGNORECASE)


def parse_log(log_path):
    epochs, losses, mses = [], [], []
    bal_train, lim_train = [], []
    lam_bal, lam_lim     = [], []
    val_ep, val_dP, val_dQ, val_Pl, val_Ql = [], [], [], [], []
    feasib_ep, feasib_pct = [], []

    current_ep = None
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = LOG_EP_RE.search(line)
            if m:
                current_ep = int(m.group(1))
                epochs.append(current_ep)
                losses.append(float(m.group(2)))
                mses.append(float(m.group(3)))
                bal_train.append(float(m.group(4)))
                lim_train.append(float(m.group(5)))

                lm = LAM_RE.search(line)
                if lm:
                    lam_bal.append(float(lm.group(1)))
                    lam_lim.append(float(lm.group(2)))

                vb = VAL_BAL_RE.search(line)
                vl = VAL_LIM_RE.search(line)
                if vb and vl:
                    val_ep.append(current_ep)
                    val_dP.append(float(vb.group(1)))
                    val_dQ.append(float(vb.group(2)))
                    val_Pl.append(float(vl.group(1)))
                    val_Ql.append(float(vl.group(2)))

            fm = FEASIB_RE.search(line)
            if fm and current_ep is not None:
                feasib_ep.append(current_ep)
                feasib_pct.append(float(fm.group(1)))

    return dict(epochs=epochs, losses=losses, mses=mses,
                bal_train=bal_train, lim_train=lim_train,
                lam_bal=lam_bal, lam_lim=lam_lim,
                val_ep=val_ep, val_dP=val_dP, val_dQ=val_dQ,
                val_Pl=val_Pl, val_Ql=val_Ql,
                feasib_ep=feasib_ep, feasib_pct=feasib_pct)


def plot_loss_landscape(log_data, out_dir):
    ep  = np.array(log_data["epochs"])
    if len(ep) == 0:
        print("  WARNING: no epochs parsed from log — skipping Part 2")
        return {}

    # ── 1. Loss components + constraint violations ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("PDL-v2 Training Dynamics — Loss Landscape (case118)",
                 fontweight="bold", fontsize=13)

    ax = axes[0, 0]
    ax.semilogy(ep, log_data["losses"],   lw=1.5, label="Total",   color="#333")
    ax.semilogy(ep, log_data["mses"],     lw=1,   label="MSE",     color="#2196F3", ls="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)"); ax.set_title("Reconstruction Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(ep, log_data["bal_train"], lw=1.5, label="Balance (dP_inf)", color="#E91E63")
    ax.semilogy(ep, log_data["lim_train"], lw=1.5, label="Limit (P_lim)",    color="#FF9800")
    ax.axhline(0.01, color="black", ls=":", lw=1, label="feasible threshold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Constraint violation (pu)")
    ax.set_title("Train Constraint Violations")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ep_lam = ep[:len(log_data["lam_bal"])]
    if log_data["lam_bal"]:
        ax.plot(ep_lam, log_data["lam_bal"], lw=2, label="λ_bal", color="#9C27B0")
    if log_data["lam_lim"]:
        ax.plot(ep_lam, log_data["lam_lim"], lw=2, label="λ_lim", color="#F44336")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Lagrange multiplier")
    ax.set_title("PDL Dual Variables (Lagrange Multipliers)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    vep = np.array(log_data["val_ep"])
    if len(vep):
        ax.plot(vep, log_data["val_dP"], label="dP_inf", color="#2196F3")
        ax.plot(vep, log_data["val_dQ"], label="dQ_inf", color="#E91E63")
        ax.plot(vep, log_data["val_Pl"], label="P_limit", color="#FF9800")
        ax.plot(vep, log_data["val_Ql"], label="Q_limit", color="#4CAF50")
        ax.axhline(0.01, color="black", ls=":", lw=1, label="threshold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Violation (pu)")
    ax.set_title("Validation Feasibility Metrics")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_components.png"), dpi=150)
    plt.close()
    print("  saved loss_components.png")

    # ── 2. Lambda trajectory (standalone) ─────────────────────────────────────
    if log_data["lam_bal"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(ep_lam, log_data["lam_bal"], lw=2.5, label="λ_bal (power balance)", color="#9C27B0")
        ax.plot(ep_lam, log_data["lam_lim"], lw=2.5, label="λ_lim (gen limits)",    color="#F44336")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Lagrange multiplier value")
        ax.set_title("PDL Dual Ascent: λ trajectories — multipliers penalise infeasible solutions",
                     fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "lambda_trajectory.png"), dpi=150)
        plt.close()
        print("  saved lambda_trajectory.png")

    # ── 3. Constraint phase portrait ───────────────────────────────────────────
    if len(vep) > 1:
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(log_data["val_dP"], log_data["val_Pl"],
                        c=vep, cmap="plasma", s=40, alpha=0.85, zorder=3)
        ax.axvline(0.01, color="gray", ls="--", lw=1)
        ax.axhline(0.01, color="gray", ls="--", lw=1)
        ax.fill_between([0, 0.01], 0, 0.01, color="green", alpha=0.08, label="feasible region")
        ax.set_xlabel("dP_inf  (power balance violation, pu)")
        ax.set_ylabel("P_limit  (generation limit violation, pu)")
        ax.set_title("Constraint Phase Portrait\n(colour = epoch, target = bottom-left corner)",
                     fontweight="bold")
        ax.legend(fontsize=9)
        plt.colorbar(sc, ax=ax, label="Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "constraint_phase_portrait.png"), dpi=150)
        plt.close()
        print("  saved constraint_phase_portrait.png")

    # ── 4. Feasibility curve ───────────────────────────────────────────────────
    if log_data["feasib_ep"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(log_data["feasib_ep"], log_data["feasib_pct"],
                lw=2.5, color="#009688", marker="o", ms=5)
        ax.axhline(88.5, color="red", ls="--", lw=1.2, label="PDL-v2 best (88.5%)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Feasibility rate (%)")
        ax.set_title("Feasibility Over Training (all 4 constraints < 0.01 pu)",
                     fontweight="bold")
        ax.set_ylim(0, 105); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feasibility_curve.png"), dpi=150)
        plt.close()
        print("  saved feasibility_curve.png")

    lb = log_data["lam_bal"]
    ll = log_data["lam_lim"]
    return {
        "n_epochs_logged":   len(ep),
        "final_lambda_bal":  float(lb[-1]) if lb else None,
        "final_lambda_lim":  float(ll[-1]) if ll else None,
        "best_feasibility":  max(log_data["feasib_pct"]) if log_data["feasib_pct"] else None,
        "min_dP_inf":        min(log_data["val_dP"]) if log_data["val_dP"] else None,
        "min_P_limit":       min(log_data["val_Pl"]) if log_data["val_Pl"] else None,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",            required=True)
    ap.add_argument("--opfdata_root",    required=True)
    ap.add_argument("--case_name",       default="pglib_opf_case118_ieee")
    ap.add_argument("--n_scenarios",     type=int, default=200)
    ap.add_argument("--log_file",        default=None)
    ap.add_argument("--out_dir",         default="results/xai_heads/pdl2")
    ap.add_argument("--K",               type=int, default=30)
    ap.add_argument("--d",               type=int, default=4)
    ap.add_argument("--d_hi",            type=int, default=16)
    ap.add_argument("--n_heads",         type=int, default=4)
    ap.add_argument("--num_attn_layers", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")   # CPU like analyze_attention_cost.py
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.ckpt}")
    model = load_model(args.ckpt, args.K, args.d, args.d_hi,
                       args.n_heads, args.num_attn_layers)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading data: {args.case_name}")
    from opfdata_pipeline import make_opfdata_loaders
    _, _, _, test_loader = make_opfdata_loaders(
        args.opfdata_root, args.case_name, batch_size=1,
        max_train=1, max_valid=1, max_test=args.n_scenarios,
        num_workers=0,
    )

    # ── Part 1: per-head attention ─────────────────────────────────────────────
    print(f"\n=== Part 1: Per-head attention analysis ({args.n_scenarios} scenarios) ===")
    print(f"  Using return_attn_heads=True — no monkey-patching needed")
    alpha_mean, vm_mean, n_nodes, edge_index = collect_head_attention(
        model, test_loader, device, args.n_scenarios, args.n_heads
    )
    print(f"  alpha_mean shape: {alpha_mean.shape}  (N={n_nodes}, H={args.n_heads})")
    head_stats = analyze_heads(alpha_mean, vm_mean, args.out_dir, args.n_heads)

    # ── Part 2: loss landscape ─────────────────────────────────────────────────
    loss_summary = {}
    if args.log_file:
        if os.path.exists(args.log_file):
            print(f"\n=== Part 2: Loss landscape from {args.log_file} ===")
            log_data = parse_log(args.log_file)
            print(f"  parsed {len(log_data['epochs'])} train epochs, "
                  f"{len(log_data['val_ep'])} val checkpoints, "
                  f"{len(log_data['feasib_ep'])} feasibility records")
            loss_summary = plot_loss_landscape(log_data, args.out_dir)
        else:
            print(f"  WARNING: log_file not found: {args.log_file}")
    else:
        print("  (--log_file not provided, skipping Part 2)")

    # ── Save JSON summary ──────────────────────────────────────────────────────
    summary = {
        "model":         args.ckpt,
        "n_scenarios":   args.n_scenarios,
        "n_heads":       args.n_heads,
        "n_nodes":       n_nodes,
        "head_stats":    head_stats,
        "loss_summary":  loss_summary,
        "interpretation": {
            "norm_entropy=1.0":   "Perfectly uniform attention (no specialisation)",
            "cross_head_r≈1.0":   "Heads are redundant — same attention pattern",
            "cross_head_r<0.5":   "Heads specialise — different buses get attention per head",
            "lambda_trajectory":  "Dual ascent: λ grows until constraint satisfied, then stabilises",
            "phase_portrait":     "Convergence path toward feasible region (bottom-left corner)",
        },
    }
    out_json = os.path.join(args.out_dir, "attn_heads_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Print key findings
    print("\n=== Key findings ===")
    for h in range(args.n_heads):
        hs = head_stats[f"head_{h+1}"]
        print(f"  Head {h+1}: H/H_max={hs['norm_entropy']:.4f}  "
              f"max_bus={hs['max_bus']}  max_attn={hs['max_attn']:.5f}  "
              f"r_vs_vm={hs.get('r_vs_vm','N/A')}")
    if loss_summary.get("final_lambda_lim"):
        print(f"  Final λ_lim={loss_summary['final_lambda_lim']:.3f}  "
              f"λ_bal={loss_summary['final_lambda_bal']:.3f}")
    if loss_summary.get("best_feasibility"):
        print(f"  Best feasibility: {loss_summary['best_feasibility']:.1f}%")


if __name__ == "__main__":
    main()

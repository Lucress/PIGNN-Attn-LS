"""
Deep OPF attention analysis:
  Analysis 1 — r(k): Spearman correlation between edge attention and AC-OPF
               line loading at each correction step k (1..K).
  Analysis 3 — Head specialisation: per-head Spearman r, testing whether
               individual attention heads specialise in tracking OPF line stress.

Runs model inference once per scenario, collects:
  attn_perhead.pt  : (S, K, E, H) — per-head attention at every correction step
  line_flows.pt    : (S, E)        — |S_ij| from final AC-OPF voltage predictions

Usage (from PIGNN-Attn-LS-PPC/):
    python project_deep_analysis.py \\
        --ckpt results/ckpt/pignn_opf_pglib_opf_case14_ieee_best.pt \\
        --opfdata_root ./opfdata \\
        --attn_dir results/attention_supervised \\
        --out_dir results/deep_supervised \\
        --K 15 --d 4 --d_hi 16 --n_heads 4 --label "Supervised"
"""

from __future__ import annotations
import argparse, json, os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# ── physical helpers ──────────────────────────────────────────────────────────

def compute_line_flows(vm, va, Y_dense, edge_index):
    """|S_ij| for each directed edge from AC-OPF predicted voltages."""
    Vc = (vm * torch.exp(1j * va.to(torch.float64))).to(torch.complex128)
    flows = []
    for src, dst in edge_index.tolist():
        Y_ij = Y_dense[dst, src]
        S_ij = abs(complex(Vc[src]) * complex(Y_ij).conjugate() *
                   (complex(Vc[src]) - complex(Vc[dst])).conjugate())
        flows.append(float(S_ij))
    return torch.tensor(flows)


def spearman(x, y):
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


# ── model + inference ─────────────────────────────────────────────────────────

def load_model(ckpt, K, d, d_hi, n_heads, num_attn_layers=1):
    from GNSMsg_SelfAttention_armijo_opf import GNSMsg_EdgeSelfAttn_OPF
    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ckpt_data.get("model_state_dict", ckpt_data)
    model = GNSMsg_EdgeSelfAttn_OPF(
        d=d, d_hi=d_hi, K=K, n_heads=n_heads,
        num_attn_layers=num_attn_layers,
        pinn=True, use_armijo=True, armijo_mode="geometric_safe",
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def run_scenario(model, batch, device):
    """Returns (vm, va, Y_dense, attn_perhead, edge_index).
    attn_perhead: list of K tensors, each (E, H).
    """
    from opf_task import opf_decision_space
    from train_valid_test_pignn_opf import branch_rows_from_opfdata

    bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
          for k, v in batch.items() if k != "hetero"}
    hb  = batch["hetero"].to(device)
    sp  = opf_decision_space(bd, batch["sizes"], device)
    Y   = bd["Ybus"]; S = bd["S_start"]; V0 = bd["V_start"].float()
    bus_type = bd["bus_type"].reshape(1, -1)
    ctrl = sp["Gen_controllable"].to(torch.bool)
    vmin = sp.get("Bus_vmin"); vmax = sp.get("Bus_vmax")
    br   = branch_rows_from_opfdata(hb, device)

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
        Y=Y, S=S, V0=V0,
        ctrl_mask=ctrl.reshape(1, -1),
        ref_mask=(bus_type == 3),
        n_nodes_per_graph=batch["sizes"].to(device),
        v_min=(vmin.reshape(1,-1).float() if vmin is not None else None),
        v_max=(vmax.reshape(1,-1).float() if vmax is not None else None),
        return_attn=True,
        return_attn_heads=True,   # keep (E, H) per step
    )
    out, _, attn_list, edge_index = result
    vm = out[0, :, 0].cpu()
    va = out[0, :, 1].cpu()
    Y_c = Y.squeeze(0).cpu() if Y.dim() == 3 else Y.cpu()
    return vm, va, Y_c.to(torch.complex128), attn_list, edge_index


# ── Analysis 1: r(k) — correlation per correction step ───────────────────────

def analysis_step_correlation(attn_ph, flows, edge_index, K, H, label, out_dir):
    """
    attn_ph : (S, K, E, H)
    flows   : (S, E)
    For each step k: pool all S*E pairs, compute Spearman r (mean over H,
    and also per head).
    """
    S, _, E, _ = attn_ph.shape
    r_by_step   = []   # mean-head r at each k
    r_by_step_h = [[] for _ in range(H)]  # per-head r at each k

    flat_flows = flows.reshape(-1).numpy()  # (S*E,)

    for k in range(K):
        # mean over heads
        a_mean = attn_ph[:, k, :, :].mean(dim=-1).reshape(-1).numpy()  # (S*E,)
        r, _ = spearman(a_mean, flat_flows)
        r_by_step.append(r)

        for h in range(H):
            a_h = attn_ph[:, k, :, h].reshape(-1).numpy()
            r_h, _ = spearman(a_h, flat_flows)
            r_by_step_h[h].append(r_h)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = list(range(1, K + 1))
    colors_h = plt.cm.plasma(np.linspace(0.1, 0.9, H))

    for h in range(H):
        ax.plot(steps, r_by_step_h[h], lw=1.2, alpha=0.55,
                color=colors_h[h], label=f"Head {h+1}")

    ax.plot(steps, r_by_step, lw=2.5, color="white" if True else "black",
            label="Mean (all heads)", zorder=5)
    ax.fill_between(steps, r_by_step, alpha=0.12, color="#aaa")

    ax.axhline(0, color="#555", lw=0.8, ls="--")
    ax.set_xlabel("Correction step k", fontsize=11)
    ax.set_ylabel("Spearman r  (attention vs |S_ij|)", fontsize=11)
    ax.set_title(
        f"{label} — AC-OPF: Edge attention vs line loading at each correction step\n"
        f"Negative = model attends MORE to lightly loaded lines (available capacity)",
        fontsize=9.5,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(1, K)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out_dir, "step_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")

    results = {
        "r_by_step": r_by_step,
        "r_by_step_per_head": [list(r_by_step_h[h]) for h in range(H)],
    }
    with open(os.path.join(out_dir, "step_correlation.json"), "w") as f:
        json.dump(results, f, indent=2)
    return r_by_step, r_by_step_h


# ── Analysis 3: head specialisation ──────────────────────────────────────────

def analysis_head_specialization(attn_ph, flows, K, H, label, out_dir):
    """
    Compute Spearman r between each head's attention and line flow,
    averaged over all K steps.
    Also: r when only using early (first K//3) or late (last K//3) steps.
    """
    S, _, E, _ = attn_ph.shape
    flat_flows = flows.reshape(-1).numpy()  # (S*E,)

    # overall r per head (pooled over all K steps and S scenarios)
    r_head_all  = []
    for h in range(H):
        a = attn_ph[:, :, :, h].reshape(-1).numpy()   # (S*K*E,)
        f = flows.unsqueeze(1).expand(S, K, E).reshape(-1).numpy()
        r, p = spearman(a, f)
        r_head_all.append((r, p))

    # r for early steps (first K//3)
    k_early = max(1, K // 3)
    r_head_early = []
    for h in range(H):
        a = attn_ph[:, :k_early, :, h].reshape(-1).numpy()
        f = flows.unsqueeze(1).expand(S, k_early, E).reshape(-1).numpy()
        r, _ = spearman(a, f)
        r_head_early.append(r)

    # r for late steps (last K//3)
    r_head_late = []
    for h in range(H):
        a = attn_ph[:, -k_early:, :, h].reshape(-1).numpy()
        f = flows.unsqueeze(1).expand(S, k_early, E).reshape(-1).numpy()
        r, _ = spearman(a, f)
        r_head_late.append(r)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(H)
    w = 0.25
    colors = ["#4DA6FF", "#00CFA8", "#FF6B35"]

    bars_all   = [r for r, _ in r_head_all]
    ax.bar(x - w,   r_head_early, w, label=f"Early steps (1–{k_early})",  color=colors[0], alpha=0.85)
    ax.bar(x,       bars_all,     w, label="All steps (mean)",             color=colors[1], alpha=0.85)
    ax.bar(x + w,   r_head_late,  w, label=f"Late steps ({K-k_early+1}–{K})", color=colors[2], alpha=0.85)

    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Head {h+1}" for h in range(H)], fontsize=10)
    ax.set_ylabel("Spearman r  (attention vs |S_ij|)", fontsize=10)
    ax.set_title(
        f"{label} — AC-OPF head specialisation\n"
        f"Does any head specifically track line loading?",
        fontsize=9.5,
    )
    ax.legend(fontsize=8)
    ax.set_ylim(-0.75, 0.25)
    ax.grid(axis="y", alpha=0.25)

    # annotate significance
    for h, (r, p) in enumerate(r_head_all):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(h, bars_all[h] - 0.04, sig, ha="center", va="top", fontsize=9,
                color="white" if bars_all[h] < -0.1 else "#888")

    plt.tight_layout()
    path = os.path.join(out_dir, "head_specialization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")

    results = {
        "r_head_all":   [(r, p) for r, p in r_head_all],
        "r_head_early": r_head_early,
        "r_head_late":  r_head_late,
        "K_early":      k_early,
    }
    with open(os.path.join(out_dir, "head_specialization.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Head r (all steps): " +
          " | ".join(f"H{h+1}={r:.3f}({'***' if p<0.001 else 'ns'})"
                     for h, (r, p) in enumerate(r_head_all)))
    return r_head_all


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--opfdata_root", required=True)
    p.add_argument("--case_name",    default="pglib_opf_case14_ieee")
    p.add_argument("--attn_dir",     required=True)   # has existing attn_weights.pt
    p.add_argument("--out_dir",      required=True)
    p.add_argument("--label",        default="Model")
    p.add_argument("--n_scenarios",  type=int, default=30)
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

    print(f"[deep] {args.label} — loading model")
    model = load_model(args.ckpt, args.K, args.d, args.d_hi,
                       args.n_heads, args.num_attn_layers)

    print(f"[deep] Loading OPFData: {args.case_name}")
    from opfdata_pipeline import make_opfdata_loaders
    _, _, _, test_loader = make_opfdata_loaders(
        args.opfdata_root, args.case_name, batch_size=1,
        max_train=1, max_valid=1, max_test=args.n_scenarios, num_workers=0,
    )

    all_attn_ph = []   # list of (K, E, H) per scenario
    all_flows   = []   # list of (E,) per scenario
    edge_index_saved = None

    print(f"[deep] Running {args.n_scenarios} AC-OPF scenarios ...")
    for i, batch in enumerate(test_loader):
        if i >= args.n_scenarios:
            break
        try:
            vm, va, Y_dense, attn_list, ei = run_scenario(model, batch, device)
        except Exception as e:
            print(f"  scenario {i}: ERROR {e}")
            continue

        if edge_index_saved is None:
            edge_index_saved = ei  # (E, 2)

        # attn_list: list of K tensors (E, H)
        stacked = torch.stack(attn_list, dim=0)   # (K, E, H)
        all_attn_ph.append(stacked)

        flows = compute_line_flows(vm, va, Y_dense, edge_index_saved)
        all_flows.append(flows)

        if (i + 1) % 10 == 0:
            print(f"  done {i+1}/{args.n_scenarios}")

    S  = len(all_attn_ph)
    K  = all_attn_ph[0].shape[0]
    E  = all_attn_ph[0].shape[1]
    H  = all_attn_ph[0].shape[2]
    print(f"[deep] Collected {S} scenarios | K={K} | E={E} edges | H={H} heads")

    attn_ph = torch.stack(all_attn_ph, dim=0)    # (S, K, E, H)
    flows   = torch.stack(all_flows,   dim=0)    # (S, E)

    torch.save(attn_ph, os.path.join(args.out_dir, "attn_perhead.pt"))
    torch.save(flows,   os.path.join(args.out_dir, "line_flows.pt"))
    torch.save(edge_index_saved, os.path.join(args.out_dir, "edge_index.pt"))

    print(f"[deep] Analysis 1: r(k) per correction step")
    analysis_step_correlation(attn_ph, flows, edge_index_saved, K, H,
                              args.label, args.out_dir)

    print(f"[deep] Analysis 3: head specialisation")
    analysis_head_specialization(attn_ph, flows, K, H, args.label, args.out_dir)

    print(f"[deep] Done. Results -> {args.out_dir}/")


if __name__ == "__main__":
    main()

"""
Head-physics correlation matrix for AC-OPF.

For each attention head h and each directed edge (i,j), compute Spearman r
between the head's attention weight and four AC-OPF physical quantities:

  |P_ij|  — active power flow magnitude (MW)
  |Q_ij|  — reactive power flow magnitude (MVAr)
  |Dth_ij| — voltage angle difference |theta_i - theta_j| (rad)
  |DV_ij|  — voltage magnitude difference |V_i - V_j| (pu)

Output: 4 x 4 heatmap (heads x quantities) for each model.
If heads 2/4 in PINN track Q or angle rather than P, that reveals
multi-head specialisation by AC power component.

Uses attn_perhead.pt already saved by project_deep_analysis.py.
Runs inference once per scenario to get predicted voltages.

Usage (from PIGNN-Attn-LS-PPC/):
    python project_head_physics.py \\
        --ckpt results/ckpt/pignn_opf_pglib_opf_case14_ieee_best.pt \\
        --opfdata_root ./opfdata \\
        --deep_dir results/deep_supervised \\
        --out_dir results/head_physics_supervised \\
        --K 15 --d 4 --d_hi 16 --n_heads 4 --label "Supervised"
"""

from __future__ import annotations
import argparse, json, os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats


# ── AC-OPF physical quantities from predicted voltages ───────────────────────

def compute_edge_physics(vm, va, Y_dense, edge_index):
    """
    Returns dict of (E,) tensors:
      P    : |Re(S_ij)| — active power flow magnitude
      Q    : |Im(S_ij)| — reactive power flow magnitude
      Dth  : |theta_i - theta_j| — angle difference
      DV   : |V_i - V_j| — voltage magnitude difference
    """
    Vc = (vm * torch.exp(1j * va.to(torch.float64))).to(torch.complex128)
    P, Q, Dth, DV = [], [], [], []
    for src, dst in edge_index.tolist():
        Y_ij  = complex(Y_dense[dst, src])
        V_i   = complex(Vc[src])
        V_j   = complex(Vc[dst])
        I_ij  = Y_ij * (V_i - V_j)
        S_ij  = V_i * I_ij.conjugate()
        P.append(abs(S_ij.real))
        Q.append(abs(S_ij.imag))
        Dth.append(abs(float(va[src]) - float(va[dst])))
        DV.append(abs(float(vm[src]) - float(vm[dst])))
    return {
        "|P_ij|":   torch.tensor(P),
        "|Q_ij|":   torch.tensor(Q),
        "|Dθ_ij|":  torch.tensor(Dth),
        "|ΔV_ij|":  torch.tensor(DV),
    }


def spearman(x, y):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), 1.0
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


# ── model + inference ─────────────────────────────────────────────────────────

def load_model(ckpt, K, d, d_hi, n_heads, num_attn_layers=1):
    from GNSMsg_SelfAttention_armijo_opf import GNSMsg_EdgeSelfAttn_OPF
    data  = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = data.get("model_state_dict", data)
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
    from opf_task import opf_decision_space
    from train_valid_test_pignn_opf import branch_rows_from_opfdata
    bd  = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
           for k, v in batch.items() if k != "hetero"}
    hb  = batch["hetero"].to(device)
    sp  = opf_decision_space(bd, batch["sizes"], device)
    Y   = bd["Ybus"]; S = bd["S_start"]; V0 = bd["V_start"].float()
    bt  = bd["bus_type"].reshape(1, -1)
    ctrl = sp["Gen_controllable"].to(torch.bool)
    vmin = sp.get("Bus_vmin"); vmax = sp.get("Bus_vmax")
    br  = branch_rows_from_opfdata(hb, device)
    result = model(
        bus_type=bt,
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
        ref_mask=(bt == 3),
        n_nodes_per_graph=batch["sizes"].to(device),
        v_min=(vmin.reshape(1,-1).float() if vmin is not None else None),
        v_max=(vmax.reshape(1,-1).float() if vmax is not None else None),
        return_attn=False,
    )
    # return_attn=False since we load attn_perhead from disk
    if isinstance(result, tuple):
        out = result[0]
    else:
        out = result
    vm = out[0, :, 0].cpu()
    va = out[0, :, 1].cpu()
    Y_c = Y.squeeze(0).cpu() if Y.dim() == 3 else Y.cpu()
    return vm, va, Y_c.to(torch.complex128)


# ── correlation matrix ────────────────────────────────────────────────────────

def compute_corr_matrix(attn_ph, phys_dict, K, H):
    """
    attn_ph  : (S, K, E, H)
    phys_dict: {qty_name: (S, E)}
    Returns  : {qty_name: [(r_h, p_h) for h in H]}  — averaged over K steps
    """
    S, _, E, _ = attn_ph.shape
    result = {}
    for qty, phys in phys_dict.items():
        # pool all K steps: (S*K, E)
        a_flat = attn_ph.permute(0, 1, 3, 2).reshape(S * K, H, E)  # (S*K, H, E)
        p_flat = phys.unsqueeze(1).expand(S, K, E).reshape(S * K, E)  # (S*K, E)
        head_rs = []
        for h in range(H):
            a_h = a_flat[:, h, :].reshape(-1).numpy()   # (S*K*E,)
            p_h = p_flat.reshape(-1).numpy()              # (S*K*E,)
            r, pv = spearman(a_h, p_h)
            head_rs.append((r, pv))
        result[qty] = head_rs
    return result


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(corr_matrix, H, label, out_dir):
    """4 x H heatmap: rows=quantities, cols=heads."""
    qtys  = list(corr_matrix.keys())
    nqty  = len(qtys)
    r_mat = np.array([[corr_matrix[q][h][0] for h in range(H)] for q in qtys])
    p_mat = np.array([[corr_matrix[q][h][1] for h in range(H)] for q in qtys])

    fig, ax = plt.subplots(figsize=(7, 5))
    vmax = max(abs(r_mat[~np.isnan(r_mat)]).max(), 0.05)
    cmap = "RdYlGn_r"   # red=negative (attends to lightly-loaded), green=positive
    im   = ax.imshow(r_mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    # annotate each cell
    for i in range(nqty):
        for j in range(H):
            r  = r_mat[i, j]
            pv = p_mat[i, j]
            if np.isnan(r):
                txt = "nan"
                col = "grey"
            else:
                sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
                txt = f"{r:.3f}\n{sig}"
                col = "white" if abs(r) > vmax * 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color=col, fontweight="bold" if pv < 0.001 else "normal")

    ax.set_xticks(range(H))
    ax.set_xticklabels([f"Head {h+1}" for h in range(H)], fontsize=10)
    ax.set_yticks(range(nqty))
    ax.set_yticklabels(qtys, fontsize=10)
    ax.set_title(
        f"{label} — AC-OPF Head × Physical Quantity Correlation\n"
        f"Spearman r pooled over {'{S}' if 'S' not in label else ''} scenarios × K steps · "
        f"*** p<0.001 · ns = not significant",
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, label="Spearman r", shrink=0.8)

    # highlight strongest per row
    for i in range(nqty):
        row = r_mat[i]
        if not np.all(np.isnan(row)):
            j_max = np.nanargmax(np.abs(row))
            ax.add_patch(plt.Rectangle((j_max - 0.48, i - 0.48), 0.96, 0.96,
                                       fill=False, edgecolor="yellow", lw=2.5))

    plt.tight_layout()
    path = os.path.join(out_dir, "head_physics_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")
    return r_mat, p_mat, qtys


def plot_head_bar(corr_matrix, H, label, out_dir):
    """Bar chart per head, 4 grouped bars (one per quantity)."""
    qtys   = list(corr_matrix.keys())
    colors = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"]
    x = np.arange(H)
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))

    for j, (qty, col) in enumerate(zip(qtys, colors)):
        rs = [corr_matrix[qty][h][0] for h in range(H)]
        ps = [corr_matrix[qty][h][1] for h in range(H)]
        bars = ax.bar(x + (j - 1.5) * w, rs, w, label=qty, color=col, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, r, p in zip(bars, rs, ps):
            if np.isnan(r): continue
            sig = "***" if p < 0.001 else ("*" if p < 0.05 else "")
            if sig:
                ax.text(bar.get_x() + bar.get_width()/2,
                        r + (0.012 if r >= 0 else -0.025),
                        sig, ha="center", va="bottom" if r >= 0 else "top",
                        fontsize=9, color=col, fontweight="bold")

    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Head {h+1}" for h in range(H)], fontsize=11)
    ax.set_ylabel("Spearman r", fontsize=10)
    ax.set_title(
        f"{label} — What does each attention head track?\n"
        f"Spearman r between per-head attention and AC-OPF physical quantities",
        fontsize=9.5,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-0.65, 0.35)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = os.path.join(out_dir, "head_physics_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--opfdata_root", required=True)
    p.add_argument("--case_name",    default="pglib_opf_case14_ieee")
    p.add_argument("--deep_dir",     required=True)   # has attn_perhead.pt + edge_index.pt
    p.add_argument("--out_dir",      required=True)
    p.add_argument("--label",        default="Model")
    p.add_argument("--K",            type=int, default=30)
    p.add_argument("--d",            type=int, default=10)
    p.add_argument("--d_hi",         type=int, default=32)
    p.add_argument("--n_heads",      type=int, default=4)
    p.add_argument("--num_attn_layers", type=int, default=1)
    p.add_argument("--chunk_dir", default="",
                   help="case500 chunk dir; if set, uses ChunkedOPFDataset")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    # load saved per-head attention
    attn_ph = torch.load(os.path.join(args.deep_dir, "attn_perhead.pt"),
                         weights_only=False)           # (S, K, E, H)
    ei      = torch.load(os.path.join(args.deep_dir, "edge_index.pt"),
                         weights_only=False)           # (E, 2)
    S, K, E, H = attn_ph.shape
    print(f"[hp] {args.label}: S={S} scenarios, K={K}, E={E} edges, H={H} heads")

    # run inference to get predicted voltages per scenario
    print(f"[hp] Loading model + running AC-OPF inference ...")
    model = load_model(args.ckpt, args.K, args.d, args.d_hi,
                       args.n_heads, args.num_attn_layers)

    if args.chunk_dir:
        from opfdata_pipeline import make_case500_loaders
        _, _, _, test_loader = make_case500_loaders(
            args.chunk_dir, batch_size=1,
            n_train_groups=1,
            max_train=1, max_valid=1, max_test=S, num_workers=0,
        )
    else:
        from opfdata_pipeline import make_opfdata_loaders
        _, _, _, test_loader = make_opfdata_loaders(
            args.opfdata_root, args.case_name, batch_size=1,
            max_train=1, max_valid=1, max_test=S, num_workers=0,
        )

    all_phys = {"|P_ij|": [], "|Q_ij|": [], "|Dθ_ij|": [], "|ΔV_ij|": []}
    for i, batch in enumerate(test_loader):
        if i >= S: break
        try:
            vm, va, Y_dense = run_scenario(model, batch, device)
            phys = compute_edge_physics(vm, va, Y_dense, ei)
            for qty in all_phys:
                all_phys[qty].append(phys[qty])
        except Exception as e:
            print(f"  scenario {i}: ERROR {e}")
            # fill with zeros to keep indexing aligned
            for qty in all_phys:
                all_phys[qty].append(torch.zeros(E))
        if (i + 1) % 10 == 0:
            print(f"  done {i+1}/{S}")

    # stack to (S, E)
    phys_tensors = {qty: torch.stack(v, dim=0) for qty, v in all_phys.items()}

    # compute correlation matrix
    print(f"[hp] Computing head × physics correlation matrix ...")
    corr = compute_corr_matrix(attn_ph, phys_tensors, K, H)

    # print full matrix
    qtys = list(corr.keys())
    print(f"\n{'':12s}" + "  ".join(f"Head {h+1:1d}   " for h in range(H)))
    for qty in qtys:
        row = "  ".join(f"{r:+.3f}({'***' if p<0.001 else 'ns ':3s})"
                        for r, p in corr[qty])
        print(f"{qty:12s}  {row}")

    # save JSON
    results = {qty: [(float(r), float(p)) for r, p in corr[qty]] for qty in qtys}
    with open(os.path.join(args.out_dir, "head_physics_corr.json"), "w") as f:
        json.dump(results, f, indent=2)

    # generate figures
    r_mat, p_mat, _ = plot_heatmap(corr, H, args.label, args.out_dir)
    plot_head_bar(corr, H, args.label, args.out_dir)

    # highlight winner per head
    print(f"\n[hp] Dominant quantity per head:")
    for h in range(H):
        col_r = [corr[q][h][0] for q in qtys]
        col_p = [corr[q][h][1] for q in qtys]
        best_i = int(np.nanargmax(np.abs(col_r)))
        print(f"  Head {h+1}: {qtys[best_i]}  r={col_r[best_i]:.3f}  "
              f"p={'<0.001' if col_p[best_i]<0.001 else f'{col_p[best_i]:.3f}'}")

    print(f"\n[hp] Done -> {args.out_dir}/")


if __name__ == "__main__":
    main()

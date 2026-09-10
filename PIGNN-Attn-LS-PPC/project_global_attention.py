"""
Global attention XAI analysis for PIGNN-Attn-LS.

Answers the supervisor question: "what does the global attention head attend to?"

For each Newton step k, the global attention weight matrix W_k ∈ R^{H×N×N}
says: for query bus i, head h puts weight W_k[h,i,j] on key bus j.

This script:
  1. Runs N_scenarios through the model (global_attn enabled), collecting
     W_k tensors (K steps × H heads × N×N).
  2. Computes:
     a. Mean attention vs graph-hop distance — does global attention attend
        to distant nodes more than local attention?
     b. Top-attended (i→j) pairs per head per step — are they physical
        (high-impedance paths, islanded sub-systems)?
     c. Entropy of each head's distribution — are heads specialised?
  3. Saves:
     - global_attn.pt        : (S, K, H, N, N) stacked weights
     - global_attn_stats.json: per-step, per-head summary statistics
     - global_attn_vs_dist.png : mean attention weight vs graph hop distance
     - global_attn_entropy.png : per-head entropy across Newton steps
     - global_attn_heatmap_k*.png : N×N heatmap for selected steps

Usage (run from PIGNN-Attn-LS-PPC/):
    python project_global_attention.py \\
        --ckpt results/ckpt/tinyx/pignn_supervised_case118_global_best.pt \\
        --opfdata_root ./opfdata \\
        --case_name pglib_opf_case118_ieee \\
        --K 30 --d 4 --d_hi 16 --n_heads 4 \\
        --global_attn_position after \\
        --n_scenarios 30 \\
        --out_dir results/xai/global_attn_case118_supervised
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import torch


# ── graph utilities ──────────────────────────────────────────────────────────

def hop_distance_matrix(edge_index: torch.Tensor, n_nodes: int) -> np.ndarray:
    """BFS hop distances for all pairs. Disconnected pairs → -1."""
    adj = collections.defaultdict(list)
    for s, d in edge_index.tolist():
        adj[s].append(d)
        adj[d].append(s)

    D = np.full((n_nodes, n_nodes), -1, dtype=np.int32)
    for start in range(n_nodes):
        dist = [-1] * n_nodes
        dist[start] = 0
        q = collections.deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        D[start] = dist
    return D


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path, K, d, d_hi, n_heads, num_attn_layers,
               global_attn_position):
    from GNSMsg_SelfAttention_armijo_opf import GNSMsg_EdgeSelfAttn_OPF
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = GNSMsg_EdgeSelfAttn_OPF(
        d=d, d_hi=d_hi, K=K, n_heads=n_heads, num_attn_layers=num_attn_layers,
        pinn=True, use_armijo=True, armijo_mode="geometric_safe",
        use_global_attn=True, global_attn_position=global_attn_position,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ── inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_batch_global(model, batch, device):
    """Returns (local_attn_list, global_attn_list, edge_index)."""
    from opf_task import opf_decision_space
    from train_valid_test_pignn_opf import branch_rows_from_opfdata

    bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
          for k, v in batch.items() if k != "hetero"}
    hb = batch["hetero"].to(device)

    space    = opf_decision_space(bd, batch["sizes"], device)
    br       = branch_rows_from_opfdata(hb, device)

    result = model(
        bus_type=bd["bus_type"].reshape(1, -1),
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
        Y=bd["Ybus"], S=bd["S_start"], V0=bd["V_start"].float(),
        ctrl_mask=space["Gen_controllable"].to(torch.bool).reshape(1, -1),
        ref_mask=(bd["bus_type"].reshape(1, -1) == 3),
        n_nodes_per_graph=batch["sizes"].to(device),
        v_min=(space.get("Bus_vmin", None) if space.get("Bus_vmin") is not None
               else None),
        v_max=(space.get("Bus_vmax", None) if space.get("Bus_vmax") is not None
               else None),
        return_attn=True,
        return_attn_heads=True,
        return_global_attn=True,
    )
    # result: (out, phys, local_attn_list, edge_index, global_attn_list)
    if len(result) == 5:
        _, _, local_attn, edge_index, global_attn = result
    else:
        _, _, local_attn, edge_index = result
        global_attn = []
    return local_attn, global_attn, edge_index


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",              required=True)
    p.add_argument("--opfdata_root",      required=True)
    p.add_argument("--case_name",         default="pglib_opf_case118_ieee")
    p.add_argument("--chunk_dir",         default="")
    p.add_argument("--n_scenarios",       type=int, default=30)
    p.add_argument("--out_dir",           default="results/xai/global_attn")
    p.add_argument("--K",                 type=int, default=30)
    p.add_argument("--d",                 type=int, default=4)
    p.add_argument("--d_hi",              type=int, default=16)
    p.add_argument("--n_heads",           type=int, default=4)
    p.add_argument("--num_attn_layers",   type=int, default=1)
    p.add_argument("--global_attn_position", default="after",
                   choices=("before", "after", "both"))
    p.add_argument("--plot_steps",        type=int, nargs="+",
                   default=[0, 4, 9, 19, 29],
                   help="Newton steps for which to save N×N heatmaps")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    print(f"[global_attn] Loading checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, args.K, args.d, args.d_hi,
                       args.n_heads, args.num_attn_layers,
                       args.global_attn_position)

    print(f"[global_attn] Loading data: {args.case_name}")
    if args.chunk_dir:
        from opfdata_pipeline import make_case500_loaders
        _, _, _, test_loader = make_case500_loaders(
            args.chunk_dir, batch_size=1, n_train_groups=1,
            max_train=1, max_valid=1, max_test=args.n_scenarios)
    else:
        from opfdata_pipeline import make_opfdata_loaders
        _, _, _, test_loader = make_opfdata_loaders(
            args.opfdata_root, args.case_name, batch_size=1,
            max_train=1, max_valid=1, max_test=args.n_scenarios)

    all_global_attn = []   # list[S] of (K, H, N, N)
    edge_index_saved = None
    n_nodes = None

    print(f"[global_attn] Running {args.n_scenarios} scenarios ...")
    for i, batch in enumerate(test_loader):
        if i >= args.n_scenarios:
            break
        try:
            _, global_attn, edge_index = run_batch_global(model, batch, device)
        except Exception as e:
            print(f"  scenario {i}: ERROR {e}")
            continue

        if not global_attn:
            print(f"  scenario {i}: no global attention returned — is use_global_attn=True?")
            continue

        if edge_index_saved is None:
            edge_index_saved = edge_index
            n_nodes = int(batch["sizes"].sum().item())

        # global_attn: list[K] of (H, N, N)
        stacked = torch.stack(global_attn, dim=0)   # (K, H, N, N)
        all_global_attn.append(stacked)

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{args.n_scenarios} done")

    if not all_global_attn:
        print("[global_attn] ERROR: no global attention collected. Check model flags.")
        return

    attn_tensor = torch.stack(all_global_attn, dim=0)  # (S, K, H, N, N)
    S, K, H, N, _ = attn_tensor.shape
    print(f"[global_attn] Collected: S={S} K={K} H={H} N={N}")

    # Save raw weights
    torch.save(attn_tensor, os.path.join(args.out_dir, "global_attn.pt"))
    torch.save(edge_index_saved, os.path.join(args.out_dir, "edge_index.pt"))
    print(f"[global_attn] Saved global_attn.pt ({S}×{K}×{H}×{N}×{N})")

    # ── hop-distance analysis ─────────────────────────────────────────────────
    print("[global_attn] Computing hop-distance matrix ...")
    hop_mat = hop_distance_matrix(edge_index_saved, N)  # (N, N)
    max_hop = int(hop_mat.max())

    # mean attention weight per hop distance, averaged over S, K, H
    mean_attn = attn_tensor.mean(dim=(0, 1, 2)).numpy()  # (N, N)
    hop_means = {}
    for h in range(1, max_hop + 1):
        mask = (hop_mat == h)
        if mask.sum() > 0:
            hop_means[h] = float(mean_attn[mask].mean())
    hop_means[0] = float(mean_attn[hop_mat == 0].mean())

    print("[global_attn] Hop-distance → mean attention weight:")
    for d, w in sorted(hop_means.items()):
        print(f"  hop {d}: {w:.5f}")

    # ── per-head entropy ──────────────────────────────────────────────────────
    # entropy of each head's attention distribution per step, averaged over S
    eps = 1e-12
    # attn_tensor: (S, K, H, N, N), rows (N dim -1) sum to ~1 per query node
    ent = -(attn_tensor * (attn_tensor + eps).log()).sum(dim=-1)  # (S, K, H, N)
    mean_ent = ent.mean(dim=(0, 3)).numpy()   # (K, H)

    # ── statistics JSON ───────────────────────────────────────────────────────
    stats = {
        "n_scenarios": S, "K": K, "n_heads": H, "n_nodes": N,
        "hop_distance_mean_attn": {str(k): v for k, v in hop_means.items()},
        "per_step_per_head_mean_entropy": mean_ent.tolist(),
        "max_hop": max_hop,
    }
    with open(os.path.join(args.out_dir, "global_attn_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ── plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Mean attention vs hop distance
        hops = sorted(hop_means)
        vals = [hop_means[h] for h in hops]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(hops, vals, color="#3a7ebf", edgecolor="k", linewidth=0.5)
        ax.set_xlabel("Graph hop distance (i→j)")
        ax.set_ylabel("Mean global attention weight")
        ax.set_title(f"Global attention vs. hop distance\n({args.case_name}, K={K}, H={H}, S={S})")
        ax.axvline(x=1, color="red", linestyle="--", linewidth=1, label="local (1-hop)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "global_attn_vs_dist.png"), dpi=150)
        plt.close()
        print("[global_attn] Saved global_attn_vs_dist.png")

        # 2. Per-head entropy across Newton steps
        fig, ax = plt.subplots(figsize=(9, 4))
        for h in range(H):
            ax.plot(range(K), mean_ent[:, h], label=f"Head {h+1}", linewidth=1.5)
        ax.set_xlabel("Newton step k")
        ax.set_ylabel("Mean entropy of global attention distribution")
        ax.set_title(f"Global attention entropy per head — {args.case_name}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "global_attn_entropy.png"), dpi=150)
        plt.close()
        print("[global_attn] Saved global_attn_entropy.png")

        # 3. N×N heatmaps for selected Newton steps
        mean_over_s = attn_tensor.mean(dim=0).numpy()  # (K, H, N, N)
        for k_step in args.plot_steps:
            if k_step >= K:
                continue
            fig, axes = plt.subplots(1, H, figsize=(5 * H, 4))
            if H == 1:
                axes = [axes]
            for h, ax in enumerate(axes):
                im = ax.imshow(mean_over_s[k_step, h], aspect="auto",
                               cmap="hot", interpolation="nearest")
                ax.set_title(f"Head {h+1}")
                ax.set_xlabel("Key bus j")
                ax.set_ylabel("Query bus i")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"Global attention — step k={k_step} — {args.case_name}")
            plt.tight_layout()
            fn = os.path.join(args.out_dir, f"global_attn_heatmap_k{k_step:02d}.png")
            plt.savefig(fn, dpi=130)
            plt.close()
        print(f"[global_attn] Saved heatmaps for steps {args.plot_steps}")

    except ImportError:
        print("[global_attn] matplotlib not available — skipping plots")

    print(f"\n[global_attn] Done. Outputs in {args.out_dir}/")
    print(f"  global_attn.pt           — ({S},{K},{H},{N},{N}) weight tensor")
    print(f"  global_attn_stats.json   — hop-distance means + entropy")
    print(f"  global_attn_vs_dist.png  — bar chart: attn weight vs hop distance")
    print(f"  global_attn_entropy.png  — head entropy across Newton steps")
    print(f"  global_attn_heatmap_k*.png — N×N heatmaps for selected steps")


if __name__ == "__main__":
    main()

"""
Receptive field + attention weight analysis for PIGNN-Attn-LS on OPFData.

Usage (run from PIGNN-Attn-LS-PPC/):
    python analyze_attention_opfdata.py \
        --ckpt results/ckpt/pignn_opf_pinn_opfdata_5k_best.pt \
        --opfdata_root ./opfdata \
        --case_name pglib_opf_case14_ieee \
        --n_scenarios 50 \
        --out_dir results/attention

Outputs (in out_dir/):
    receptive_field.json   -- graph diameter vs K analysis
    attn_weights.pt        -- stacked attention tensors (N_scenarios, K, E)
    edge_index.pt          -- directed edge list (E, 2)  [src, dst] 0-indexed
    bus_attention.pt       -- aggregated bus attention scores
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import torch
import numpy as np


# ── graph diameter (receptive field) ────────────────────────────────────────

def graph_diameter_and_eccentricity(edge_index: torch.Tensor, n_nodes: int):
    """BFS-based diameter and per-node eccentricity from an undirected edge list."""
    adj = [[] for _ in range(n_nodes)]
    for s, d in edge_index.tolist():
        adj[s].append(d)
        adj[d].append(s)

    eccentricities = []
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
        reachable = [d for d in dist if d >= 0]
        eccentricities.append(max(reachable))

    return max(eccentricities), min(eccentricities), eccentricities


# ── model loading ────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, K: int, d: int, d_hi: int, n_heads: int,
               num_attn_layers: int) -> "GNSMsg_EdgeSelfAttn_OPF":
    from GNSMsg_SelfAttention_armijo_opf import GNSMsg_EdgeSelfAttn_OPF

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model = GNSMsg_EdgeSelfAttn_OPF(
        d=d, d_hi=d_hi, K=K, n_heads=n_heads, num_attn_layers=num_attn_layers,
        pinn=True, use_armijo=True, armijo_mode="geometric_safe",
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ── inference with attention extraction ─────────────────────────────────────

@torch.no_grad()
def run_batch(model, batch, device):
    """Run one collated OPFData batch with return_attn=True.

    Returns (attn_list, edge_index) where attn_list is a list of K tensors (E,).
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
        ref_mask=(bus_type == 3),   # OPFData/MATPOWER: 3=ref bus
        n_nodes_per_graph=batch["sizes"].to(device),
        v_min=(vmin.reshape(1, -1).float() if vmin is not None else None),
        v_max=(vmax.reshape(1, -1).float() if vmax is not None else None),
        return_attn=True,
    )

    # result = (out, phys_loss_or_None, collected_attn, edge_index)
    _, _, attn_list, edge_index = result
    return attn_list, edge_index


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--opfdata_root", required=True)
    p.add_argument("--case_name",    default="pglib_opf_case14_ieee")
    p.add_argument("--n_scenarios",  type=int, default=50)
    p.add_argument("--out_dir",      default="results/attention")
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

    print(f"[attn] Loading checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, args.K, args.d, args.d_hi,
                       args.n_heads, args.num_attn_layers)

    print(f"[attn] Loading OPFData: {args.case_name}")
    from opfdata_pipeline import make_opfdata_loaders
    _, _, _, test_loader = make_opfdata_loaders(
        args.opfdata_root, args.case_name, batch_size=1,
        max_train=1, max_valid=1, max_test=args.n_scenarios,
        num_workers=0,
    )

    all_attn       = []
    edge_index_saved = None
    n_nodes        = None

    print(f"[attn] Running {args.n_scenarios} test scenarios ...")
    for i, batch in enumerate(test_loader):
        if i >= args.n_scenarios:
            break
        try:
            attn_list, edge_index = run_batch(model, batch, device)
        except Exception as e:
            print(f"  scenario {i}: ERROR {e}")
            continue

        if edge_index_saved is None:
            edge_index_saved = edge_index            # (E, 2)
            n_nodes = int(batch["sizes"].sum().item())

        # attn_list: K tensors each (E,) — stack to (K, E)
        attn_stack = torch.stack(attn_list, dim=0)
        all_attn.append(attn_stack)

        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{args.n_scenarios}")

    if not all_attn:
        print("[attn] ERROR: no scenarios processed. Check checkpoint and data paths.")
        return

    # ── receptive field ──────────────────────────────────────────────────────
    print("[attn] Computing receptive field ...")
    diameter, radius, eccentricities = graph_diameter_and_eccentricity(
        edge_index_saved, n_nodes
    )
    rf_info = {
        "case":              args.case_name,
        "n_buses":           n_nodes,
        "n_edges_directed":  int(edge_index_saved.shape[0]),
        "n_edges_undirected": int(edge_index_saved.shape[0] // 2),
        "graph_diameter":    diameter,
        "graph_radius":      radius,
        "K":                 args.K,
        "K_covers_diameter": args.K >= diameter,
        "eccentricity_per_bus": eccentricities,
        "interpretation": (
            f"K={args.K} correction steps vs graph diameter={diameter}. "
            + ("K >= diameter: every bus pair can exchange information."
               if args.K >= diameter
               else f"K < diameter: buses >K hops apart are out of reach.")
        ),
    }
    rf_path = os.path.join(args.out_dir, "receptive_field.json")
    with open(rf_path, "w") as f:
        json.dump(rf_info, f, indent=2)
    print(f"[attn] Receptive field -> {rf_path}")
    print(f"       diameter={diameter}, radius={radius}, K={args.K}, "
          f"covers={'YES' if args.K >= diameter else 'NO'}")

    # ── save attention tensors ────────────────────────────────────────────────
    attn_tensor = torch.stack(all_attn, dim=0)          # (N_scenarios, K, E)
    torch.save(attn_tensor, os.path.join(args.out_dir, "attn_weights.pt"))
    torch.save(edge_index_saved, os.path.join(args.out_dir, "edge_index.pt"))

    # aggregate to bus level: mean over scenarios and steps, sum incoming edges
    mean_edge = attn_tensor.mean(dim=0).mean(dim=0)     # (E,)
    bus_attn  = torch.zeros(n_nodes)
    for e_idx, (_, dst) in enumerate(edge_index_saved.tolist()):
        bus_attn[dst] += mean_edge[e_idx]
    bus_attn = bus_attn / bus_attn.sum().clamp_min(1e-12)

    torch.save({"bus_attn": bus_attn, "n_nodes": n_nodes,
                "n_scenarios": len(all_attn), "K": args.K},
               os.path.join(args.out_dir, "bus_attention.pt"))

    print(f"[attn] Attention saved  -> {args.out_dir}/attn_weights.pt")
    print(f"[attn] Bus attention    -> {args.out_dir}/bus_attention.pt")
    top3 = bus_attn.topk(3).indices.tolist()
    print(f"[attn] Top-3 attended buses (0-indexed): {top3} "
          f"(1-indexed: {[b+1 for b in top3]})")
    print("[attn] Done.")


if __name__ == "__main__":
    main()

"""
Three-part XAI analysis on PDL-v6 (best_feas checkpoint), case118 test set.

XAI-1  Scenario-level analysis
       Compare input features (total load, max load, per-bus load at the
       known violating buses 46/49/31/59) between feasible and failed scenarios.

XAI-2  Attention analysis
       For feasible vs. failed scenarios, compare mean edge attention weights
       and aggregate them to bus-level. Check whether buses 46/49/31/59
       receive less attention on failed scenarios (model "misses" tight buses).

XAI-3  Topology analysis
       For buses 46/49/31/59: bus degree, network distance to slack, load
       magnitude, generator capacity — characterise why these buses are hard.

Usage (run from PIGNN-Attn-LS-PPC/):
    python analyze_xai_pdl6.py \
        --ckpt results/ckpt/tinyx/pignn_pdl6_case118_best_feas.pt \
        --opfdata_root ./opfdata \
        --out_dir results/xai/pdl6

Outputs in out_dir/:
    xai1_scenario_stats.json   -- load statistics feasible vs. failed
    xai2_attention.pt          -- bus-level attention: feasible, failed, ratio
    xai2_attention.json        -- top-20 bus attention differences
    xai3_topology.json         -- topology profile of violating buses
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path, K, d, d_hi, n_heads):
    from GNSMsg_SelfAttention_armijo_opf import GNSMsg_EdgeSelfAttn_OPF
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = GNSMsg_EdgeSelfAttn_OPF(
        d=d, d_hi=d_hi, K=K, n_heads=n_heads, num_attn_layers=1,
        pinn=True, use_armijo=True, armijo_mode="geometric_safe",
        use_global_attn=False,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def run_batch(model, batch, device):
    from opf_task import opf_decision_space, opf_residual_and_violation
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
    br       = branch_rows_from_opfdata(hb, device)

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

    V, _, attn_list, edge_index = result
    parts = opf_residual_and_violation(Y, V, S_load, space)
    return parts, attn_list, edge_index, space, bd, batch["sizes"]


def bfs_distances(edge_index, n_nodes, source):
    """BFS shortest-path distances from source (0-indexed). edge_index: (E,2)."""
    adj = [[] for _ in range(n_nodes)]
    for s, d in edge_index.tolist():
        adj[s].append(d)
        adj[d].append(s)
    dist = [-1] * n_nodes
    dist[source] = 0
    q = collections.deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--opfdata_root", required=True)
    p.add_argument("--case_name",    default="pglib_opf_case118_ieee")
    p.add_argument("--out_dir",      default="results/xai/pdl6")
    p.add_argument("--K",            type=int, default=30)
    p.add_argument("--d",            type=int, default=4)
    p.add_argument("--d_hi",         type=int, default=16)
    p.add_argument("--n_heads",      type=int, default=4)
    p.add_argument("--batch_size",   type=int, default=32)
    # 1-indexed buses identified as violating in b32 violation map
    p.add_argument("--viol_buses", type=int, nargs="+", default=[46, 49, 31, 59])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    THRESH = 0.01
    viol_0idx = [b - 1 for b in args.viol_buses]  # 0-indexed

    print(f"[xai] loading model: {args.ckpt}")
    model = load_model(args.ckpt, args.K, args.d, args.d_hi, args.n_heads)
    model.to(device)

    from opfdata_pipeline import make_opfdata_loaders
    _, _, _, loader = make_opfdata_loaders(
        root=args.opfdata_root, case_name=args.case_name,
        batch_size=args.batch_size, num_groups=1,
    )

    # ── accumulators ──────────────────────────────────────────────────────────
    # XAI-1: load statistics per scenario
    feas_loads, fail_loads   = [], []   # list of (total_P, max_P, viol_bus_Ps)
    # XAI-2: attention per bus
    attn_feas_bus = None   # (n_nodes,)
    attn_fail_bus = None
    n_feas = n_fail = 0
    n_nodes_saved = None
    # XAI-3: topology (from first batch)
    topo_done = False
    topo_info = {}

    for batch in loader:
        parts, attn_list, edge_index, space, bd, sizes = run_batch(model, batch, device)

        B       = sizes.numel()
        n_nodes = int(sizes[0].item())

        if n_nodes_saved is None:
            n_nodes_saved = n_nodes

        # S_start: complex injections, shape (1, B*n_nodes) or (B*n_nodes,)
        S = bd["S_start"].squeeze(0)          # (B*n_nodes,) complex or real
        if S.is_complex():
            P_inj = S.real.cpu()
        else:
            P_inj = S.cpu()

        vp  = parts["vp"].float().cpu()       # (B*n_nodes,)
        vq  = parts["vq"].float().cpu()
        dp  = parts["dp"].float().cpu()
        dq  = parts["dq"].float().cpu()
        ctrl = space["Gen_controllable"].to(torch.bool).cpu()

        # edge-level attention averaged over K steps
        attn_valid_size = (attn_list[0].shape[0] == n_nodes_saved * args.batch_size or
                           attn_list[0].shape[0] == n_nodes_saved * B) if attn_list else False
        if attn_list:
            attn_edge = torch.stack(attn_list, 0).mean(0).cpu()  # (E,)
        else:
            attn_edge = None

        # XAI-3 topology from first batch (edge_index is (E, 2))
        if not topo_done and edge_index is not None:
            ei_full = edge_index.cpu()                          # (E, 2)
            # first-scenario subgraph: both endpoints < n_nodes
            mask   = (ei_full[:, 0] < n_nodes) & (ei_full[:, 1] < n_nodes)
            ei0    = ei_full[mask]                              # (E_single, 2)
            # bus degrees (count source appearances)
            deg    = torch.zeros(n_nodes, dtype=torch.long)
            for idx in ei0[:, 0].tolist():
                deg[idx] += 1
            # find slack bus (bus_type==3)
            bt        = bd["bus_type"].view(-1)[:n_nodes].cpu()
            slack_idx = int((bt == 3).nonzero(as_tuple=False)[0, 0].item())
            dists     = bfs_distances(ei0, n_nodes, slack_idx)  # ei0 is (E_single,2)
            # per-viol-bus info
            gp_max = space.get("Gen_p_max")
            for b0 in viol_0idx:
                dist_to_slack = dists[b0] if dists[b0] >= 0 else -1
                gen_cap = float(gp_max[b0].item()) if gp_max is not None else float("nan")
                p_load  = float(P_inj[b0].item())
                topo_info[b0 + 1] = {
                    "bus_1idx":      b0 + 1,
                    "degree":        int(deg[b0].item()),
                    "dist_to_slack": dist_to_slack,
                    "gen_p_max_pu":  round(gen_cap, 4),
                    "p_load_sample": round(p_load, 4),
                    "bus_type":      int(bt[b0].item()),
                }
            topo_done = True

        # per-scenario loop
        offset = 0
        for i in range(B):
            sc_vp = vp[offset:offset + n_nodes][ctrl[offset:offset + n_nodes]]
            sc_vq = vq[offset:offset + n_nodes][ctrl[offset:offset + n_nodes]]
            sc_dp = dp[offset:offset + n_nodes]
            sc_dq = dq[offset:offset + n_nodes]
            worst = max(
                sc_vp.max().item() if sc_vp.numel() else 0.0,
                sc_vq.max().item() if sc_vq.numel() else 0.0,
                sc_dp.abs().max().item() if sc_dp.numel() else 0.0,
                sc_dq.abs().max().item() if sc_dq.numel() else 0.0,
            )
            feasible = worst < THRESH

            # XAI-1: load features
            sc_P    = P_inj[offset:offset + n_nodes]
            total_P = float(sc_P.sum().item())
            max_P   = float(sc_P.abs().max().item())
            vbus_P  = [float(sc_P[b0].item()) for b0 in viol_0idx]
            feat    = {"total_P": total_P, "max_P": max_P, "vbus_P": vbus_P}
            if feasible:
                feas_loads.append(feat)
            else:
                fail_loads.append(feat)

            # XAI-2: bus-level attention (aggregate incident edges → bus)
            # edge_index is (E, 2) with global node indices for this batch
            if attn_edge is not None and attn_feas_bus is None:
                attn_feas_bus = torch.zeros(n_nodes)
                attn_fail_bus = torch.zeros(n_nodes)

            if attn_edge is not None and edge_index is not None:
                ei_curr = edge_index.cpu()   # (E, 2) — current batch
                lo, hi  = i * n_nodes, (i + 1) * n_nodes
                e_mask  = (ei_curr[:, 0] >= lo) & (ei_curr[:, 0] < hi)
                if e_mask.any():
                    sc_attn  = attn_edge[e_mask]
                    dst      = ei_curr[:, 1][e_mask] - lo   # 0..n_nodes-1
                    bus_attn = torch.zeros(n_nodes)
                    bus_attn.scatter_add_(0, dst, sc_attn)
                    if feasible:
                        attn_feas_bus += bus_attn
                        n_feas += 1
                    else:
                        attn_fail_bus += bus_attn
                        n_fail += 1
                else:
                    if feasible: n_feas += 1
                    else:        n_fail += 1
            else:
                if feasible: n_feas += 1
                else:        n_fail += 1

            offset += n_nodes

    # ── XAI-1: statistics ─────────────────────────────────────────────────────
    def stats(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90))}

    xai1 = {"n_feasible": len(feas_loads), "n_failed": len(fail_loads)}
    for key in ["total_P", "max_P"]:
        xai1[key] = {
            "feasible": stats([f[key] for f in feas_loads]),
            "failed":   stats([f[key] for f in fail_loads]),
        }
    for j, bus in enumerate(args.viol_buses):
        xai1[f"P_bus{bus}"] = {
            "feasible": stats([f["vbus_P"][j] for f in feas_loads]),
            "failed":   stats([f["vbus_P"][j] for f in fail_loads]),
        }

    print("\n=== XAI-1: Scenario load statistics (feasible vs. failed) ===")
    for key in ["total_P", "max_P"] + [f"P_bus{b}" for b in args.viol_buses]:
        fv = xai1[key]["feasible"]["mean"]
        fk = xai1[key]["failed"]["mean"]
        diff = fk - fv
        print(f"  {key:<12}  feas={fv:+.4f}  fail={fk:+.4f}  diff={diff:+.4f}")

    with open(os.path.join(args.out_dir, "xai1_scenario_stats.json"), "w") as f:
        json.dump(xai1, f, indent=2)

    # ── XAI-2: attention analysis ─────────────────────────────────────────────
    print("\n=== XAI-2: Bus-level attention (feasible vs. failed scenarios) ===")
    if attn_feas_bus is not None and n_feas > 0 and n_fail > 0:
        avg_feas = attn_feas_bus / n_feas
        avg_fail = attn_fail_bus / n_fail
        ratio    = avg_fail / (avg_feas + 1e-12)   # >1 = more attention on fail

        # top buses by attention difference (fail - feas)
        diff_bus = avg_fail - avg_feas
        order    = diff_bus.argsort(descending=True)
        print(f"  {'bus':>5}  {'attn_feas':>10}  {'attn_fail':>10}  {'ratio':>8}  {'diff':>10}")
        for b0 in order[:20].tolist():
            marker = " <<< VIOL" if (b0 + 1) in args.viol_buses else ""
            print(f"  {b0+1:>5}  {avg_feas[b0].item():>10.4f}  "
                  f"{avg_fail[b0].item():>10.4f}  {ratio[b0].item():>8.2f}  "
                  f"{diff_bus[b0].item():>10.4f}{marker}")

        print(f"\n  Attention at violating buses specifically:")
        for b0 in viol_0idx:
            print(f"    bus {b0+1:>3}: feas={avg_feas[b0]:.4f}  fail={avg_fail[b0]:.4f}  "
                  f"ratio={ratio[b0]:.2f}")

        torch.save({"feas": avg_feas, "fail": avg_fail, "ratio": ratio, "diff": diff_bus},
                   os.path.join(args.out_dir, "xai2_attention.pt"))

        attn_json = [{"bus_1idx": int(b0+1),
                      "attn_feas": round(avg_feas[b0].item(), 6),
                      "attn_fail": round(avg_fail[b0].item(), 6),
                      "ratio": round(ratio[b0].item(), 4),
                      "diff": round(diff_bus[b0].item(), 6)}
                     for b0 in order[:30].tolist()]
        with open(os.path.join(args.out_dir, "xai2_attention.json"), "w") as f:
            json.dump({"n_feasible": n_feas, "n_failed": n_fail,
                       "violating_buses": args.viol_buses,
                       "top_attention_diff": attn_json}, f, indent=2)
    else:
        print("  (attention not available)")

    # ── XAI-3: topology ───────────────────────────────────────────────────────
    print("\n=== XAI-3: Topology profile of violating buses ===")
    print(f"  {'bus':>5}  {'degree':>6}  {'dist_slack':>10}  {'Pmax_pu':>8}  "
          f"{'P_load':>8}  {'type':>5}")
    for bus, info in sorted(topo_info.items()):
        print(f"  {info['bus_1idx']:>5}  {info['degree']:>6}  "
              f"{info['dist_to_slack']:>10}  {info['gen_p_max_pu']:>8.4f}  "
              f"{info['p_load_sample']:>8.4f}  {info['bus_type']:>5}")

    with open(os.path.join(args.out_dir, "xai3_topology.json"), "w") as f:
        json.dump(topo_info, f, indent=2)

    print(f"\n[xai] all results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

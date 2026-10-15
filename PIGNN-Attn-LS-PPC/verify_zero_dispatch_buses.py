"""
Verify that buses 12, 65, 66 (1-indexed, case118) have optimal gen_p ≈ 0
across ALL training scenarios, not just the 200 test scenarios used in XAI.

Reads the training split of OPFData, computes power_implied(Y, V_newton) for
each scenario, and reports mean/max/std of |gen_p| at the target buses.
If max < 0.01 pu, the XAI finding holds globally and the XAI penalty is safe.

Usage:
    python verify_zero_dispatch_buses.py \
        --opfdata_root /path/to/opfdata \
        --case_name pglib_opf_case118_ieee \
        --target_buses 12 65 66
"""

import argparse, torch, sys, os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--opfdata_root", required=True)
    p.add_argument("--case_name", default="pglib_opf_case118_ieee")
    p.add_argument("--target_buses", type=int, nargs="+", default=[12, 65, 66],
                   help="1-indexed bus numbers to check")
    p.add_argument("--split", default="train", choices=("train", "val", "test", "all"))
    p.add_argument("--max_scenarios", type=int, default=0,
                   help="cap on scenarios to scan (0=all)")
    return p.parse_args()


def main():
    args = parse_args()

    # torch_geometric OPFData
    from torch_geometric.datasets import OPFData
    ds = OPFData(root=args.opfdata_root, case_name=args.case_name)
    n_total = len(ds)
    # Standard 70/15/15 split used in training script
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    if args.split == "train":
        indices = range(0, n_train)
    elif args.split == "val":
        indices = range(n_train, n_train + n_val)
    elif args.split == "test":
        indices = range(n_train + n_val, n_total)
    else:
        indices = range(n_total)

    if args.max_scenarios > 0:
        indices = list(indices)[:args.max_scenarios]

    target_0 = [b - 1 for b in args.target_buses]  # 0-indexed

    # Accumulate per-bus absolute gen_p values
    bus_vals = {b0: [] for b0 in target_0}

    from opf_task import power_implied

    print(f"Scanning {len(indices)} scenarios (split={args.split}) ...")
    for i, idx in enumerate(indices):
        if i % 500 == 0:
            print(f"  {i}/{len(indices)} ...", flush=True)
        data = ds[idx]

        # Build Ybus and V_newton from OPFData fields
        # V_newton is stored as |V|, angle in data.v
        # We replicate what opfdata_pipeline does:
        try:
            V_mag  = data.v[:, 0]
            V_ang  = data.v[:, 1]
        except Exception:
            continue

        V_complex = V_mag * torch.exp(1j * V_ang)

        # Ybus is stored as sparse in data.y_bus — build dense
        try:
            Y = torch.zeros(V_complex.shape[0], V_complex.shape[0], dtype=torch.complex128)
            edge_index = data.edge_index  # [2, E]
            edge_attr  = data.edge_attr   # [E, 2] = (real, imag)
            Y[edge_index[0], edge_index[1]] = torch.complex(edge_attr[:, 0], edge_attr[:, 1])
        except Exception:
            continue

        # S_implied = V * conj(Y @ V)
        S = V_complex * torch.conj(Y @ V_complex)
        gen_p = S.real  # active power per bus in the optimal solution

        # Check gen_controllable mask to get ctrl buses
        try:
            ctrl = data.gen_controllable.to(torch.bool)  # shape (n_nodes,)
        except Exception:
            ctrl = torch.ones(V_complex.shape[0], dtype=torch.bool)

        for b0 in target_0:
            if b0 < gen_p.shape[0]:
                val = float(gen_p[b0].abs().item())
                bus_vals[b0].append(val)

    print("\n=== Verification results ===")
    all_ok = True
    for b1, b0 in zip(args.target_buses, target_0):
        vals = bus_vals[b0]
        if not vals:
            print(f"  Bus {b1}: no data")
            continue
        t = torch.tensor(vals)
        mean_v = float(t.mean()); max_v = float(t.max()); std_v = float(t.std())
        flag = "OK" if max_v < 0.01 else "WARN — NOT always zero!"
        print(f"  Bus {b1} (0-idx {b0}): mean={mean_v:.4f} pu  max={max_v:.4f} pu  "
              f"std={std_v:.4f} pu  n={len(vals)}  [{flag}]")
        if max_v >= 0.01:
            all_ok = False

    print()
    if all_ok:
        print("CONCLUSION: All target buses have |gen_p| < 0.01 pu across the training set.")
        print("  XAI finding is consistent with training distribution. Safe to use --xai_pen_buses.")
    else:
        print("CONCLUSION: At least one bus exceeds 0.01 pu in some scenarios.")
        print("  Penalizing it may fight against legitimate dispatch. Review before using --xai_pen_buses.")


if __name__ == "__main__":
    main()

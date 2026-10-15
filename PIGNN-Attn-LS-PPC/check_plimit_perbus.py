"""
Per-bus breakdown of P-limit violations.
Shows which specific generator buses are violating and by how much,
for both V_true (oracle) and V_pred (model).
"""

import argparse
import torch
import numpy as np
from collections import defaultdict

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opfdata_root", required=True)
    p.add_argument("--case_name", default="pglib_opf_case118_ieee")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_scenarios", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from opfdata_pipeline import make_opfdata_loaders
    from opf_task import opf_decision_space, opf_residual_and_violation, implied_injection
    from train_valid_test_pignn_opf import branch_rows_from_opfdata

    _, _, _, test_loader = make_opfdata_loaders(
        args.opfdata_root, args.case_name,
        batch_size=1, max_test=args.n_scenarios, num_workers=0,
    )

    # load model
    from GNSMsg_SelfAttention_armijo_opf import GNSMsg_EdgeSelfAttn_OPF
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = GNSMsg_EdgeSelfAttn_OPF(K=30, d=4, d_hi=16, n_heads=4, num_attn_layers=1,
                                     pinn=True, use_armijo=True, armijo_mode="geometric_safe")
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    # per-bus accumulator: bus_idx → list of (vp_true, vp_pred, gen_p_true, gen_p_pred)
    bus_stats = defaultdict(lambda: {"vp_true": [], "vp_pred": [], "gp_true": [], "gp_pred": []})

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= args.n_scenarios:
                break
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in batch.items() if k != "hetero"}
            hb = batch["hetero"].to(device)

            space  = opf_decision_space(bd, batch["sizes"], device)
            Y      = bd["Ybus"]
            S_load = bd["S_start"]
            V0     = bd["V_start"].float()
            bus_type = bd["bus_type"].reshape(1, -1)
            ctrl   = space["Gen_controllable"].to(torch.bool)
            vmin   = space.get("Bus_vmin")
            vmax   = space.get("Bus_vmax")
            p_min  = space.get("Gen_p_min")
            p_max  = space.get("Gen_p_max")

            V_true = bd.get("V_newton")
            if V_true is None:
                V_true = bd.get("V_target")
            if V_true is None:
                continue

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
            )
            V_pred = result[0] if isinstance(result, (tuple, list)) else result

            # compute per-bus quantities
            for V, tag in [(V_true, "true"), (V_pred, "pred")]:
                S_impl = implied_injection(Y, V)
                resid  = S_impl - S_load.squeeze(0).to(torch.complex128)
                gen_p  = resid.real  # gen_box_on_injection=False for case118
                vp     = (torch.clamp(p_min - gen_p, min=0) +
                          torch.clamp(gen_p - p_max, min=0))
                ctrl_idx = ctrl.nonzero(as_tuple=True)[0]
                for b in ctrl_idx:
                    b = b.item()
                    bus_stats[b][f"vp_{tag}"].append(float(vp[b].item()))
                    bus_stats[b][f"gp_{tag}"].append(float(gen_p[b].item()))

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{args.n_scenarios}]")

    # report: sort by mean vp_pred descending
    print(f"\n{'='*80}")
    print(f"  Per-bus P-limit analysis over {args.n_scenarios} scenarios")
    print(f"  Columns: bus | p_min | p_max | oracle_vp | model_vp | oracle_gp | model_gp")
    print(f"{'='*80}")

    # need p_min/p_max — grab from last batch
    pmin_all = p_min.cpu().numpy()
    pmax_all = p_max.cpu().numpy()
    ctrl_buses = sorted(bus_stats.keys())

    rows = []
    for b in ctrl_buses:
        s = bus_stats[b]
        mean_vp_true = np.mean(s["vp_true"])
        mean_vp_pred = np.mean(s["vp_pred"])
        mean_gp_true = np.mean(s["gp_true"])
        mean_gp_pred = np.mean(s["gp_pred"])
        rows.append((b, mean_vp_true, mean_vp_pred, mean_gp_true, mean_gp_pred,
                     pmin_all[b], pmax_all[b]))

    rows.sort(key=lambda r: -r[2])  # sort by model vp descending
    print(f"{'Bus':>5}  {'p_min':>6}  {'p_max':>6}  {'oracle_vp':>10}  {'model_vp':>10}  {'oracle_gp':>10}  {'model_gp':>10}")
    for b, vp_t, vp_p, gp_t, gp_p, pm, pM in rows[:20]:
        flag = " <<" if vp_p > 0.01 else ""
        print(f"{b:>5}  {pm:>6.3f}  {pM:>6.3f}  {vp_t:>10.4f}  {vp_p:>10.4f}  {gp_t:>10.4f}  {gp_p:>10.4f}{flag}")

    total_vp = np.mean([np.mean(bus_stats[b]["vp_pred"]) for b in ctrl_buses])
    max_vp   = max(np.max(bus_stats[b]["vp_pred"]) for b in ctrl_buses)
    n_violating = sum(1 for b in ctrl_buses if np.mean(bus_stats[b]["vp_pred"]) > 0.01)
    print(f"\n  {n_violating}/{len(ctrl_buses)} generator buses violating on average")
    print(f"  Mean vp across all ctrl buses : {total_vp:.4f} pu")
    print(f"  Max  vp across all ctrl buses : {max_vp:.4f} pu")

if __name__ == "__main__":
    main()

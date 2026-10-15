"""
Diagnose why P-limit violations are ~0.4 pu despite opf_limit_weight=10.

Checks:
1. What is Gen_box_injection for case118?
2. What are p_min/p_max values at generator buses?
3. What is gen_p (the quantity being boxed) for the TRUE solution?
4. Does the TRUE solution show vp > 0? (If yes → wrong convention or wrong limits)
5. What is gen_p for V_pred from the physonly/highlim model?

Usage:
    python check_plimit_convention.py \
        --opfdata_root ./opfdata \
        --case_name pglib_opf_case118_ieee \
        --ckpt results/ckpt/tinyx/pignn_highlim_lr3e4_case118_best.pt \
        --n_scenarios 5
"""

import argparse
import torch
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opfdata_root", required=True)
    p.add_argument("--case_name", default="pglib_opf_case118_ieee")
    p.add_argument("--ckpt", default="")
    p.add_argument("--n_scenarios", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from opfdata_pipeline import make_opfdata_loaders
    from opf_task import opf_decision_space, opf_residual_and_violation, implied_injection

    _, _, _, test_loader = make_opfdata_loaders(
        args.opfdata_root, args.case_name,
        batch_size=1, max_test=args.n_scenarios, num_workers=0,
    )

    for i, batch in enumerate(test_loader):
        if i >= args.n_scenarios:
            break

        bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in batch.items() if k != "hetero"}

        space = opf_decision_space(bd, batch["sizes"], device)
        Y      = bd["Ybus"]
        S_load = bd["S_start"]
        V_true = bd.get("V_newton")
        if V_true is None:
            V_true = bd.get("V_target")
        if V_true is None:
            print(f"[{i}] no V_true"); continue

        ctrl   = space.get("Gen_controllable")
        p_min  = space.get("Gen_p_min")
        p_max  = space.get("Gen_p_max")
        box_on_inj = bool(space.get("gen_box_on_injection", False))

        # Check raw dataset flag
        raw_flag = batch.get("Gen_box_injection")
        print(f"\n[scenario {i}]")
        print(f"  gen_box_on_injection flag : {box_on_inj}")
        print(f"  Gen_box_injection raw     : {raw_flag}")

        # Implied injection from TRUE voltages
        S_true = implied_injection(Y, V_true)
        resid  = S_true - S_load.squeeze(0).to(torch.complex128)

        n_ctrl = int(ctrl.sum().item()) if ctrl is not None else 0
        print(f"  N generator (ctrl) buses  : {n_ctrl}")

        if ctrl is not None and p_min is not None:
            ctrl_idx = ctrl.nonzero(as_tuple=True)[0]

            if box_on_inj:
                gen_p_true = S_true.real[ctrl_idx]
            else:
                gen_p_true = resid.real[ctrl_idx]

            pmin_ctrl = p_min[ctrl_idx]
            pmax_ctrl = p_max[ctrl_idx]

            vp_true = (torch.clamp(pmin_ctrl - gen_p_true, min=0) +
                       torch.clamp(gen_p_true - pmax_ctrl, min=0))

            print(f"  --- TRUE SOLUTION (oracle) ---")
            print(f"  gen_p range (boxed qty)   : [{gen_p_true.min():.4f}, {gen_p_true.max():.4f}] pu")
            print(f"  p_min range               : [{pmin_ctrl.min():.4f}, {pmin_ctrl.max():.4f}] pu")
            print(f"  p_max range               : [{pmax_ctrl.min():.4f}, {pmax_ctrl.max():.4f}] pu")
            print(f"  vp_true (P-limit viol)    : max={vp_true.max():.6f}, mean={vp_true.mean():.6f} pu")
            print(f"  Oracle P-feasible?        : {'YES' if vp_true.max() < 1e-4 else 'NO — convention bug!'}")

            # Show worst 3 buses if oracle fails
            if vp_true.max() > 1e-4:
                worst = vp_true.topk(min(3, len(vp_true))).indices
                for w in worst:
                    bus = ctrl_idx[w].item()
                    print(f"    bus {bus}: gen_p={gen_p_true[w]:.4f}  pmin={pmin_ctrl[w]:.4f}  pmax={pmax_ctrl[w]:.4f}  vp={vp_true[w]:.4f}")

        if i == 0:
            # Also print S_load stats for reference
            sl = S_load.squeeze(0)
            print(f"\n  S_load.real range         : [{sl.real.min():.4f}, {sl.real.max():.4f}] pu")
            print(f"  S_implied.real range      : [{S_true.real.min():.4f}, {S_true.real.max():.4f}] pu")
            print(f"  resid.real range          : [{resid.real.min():.4f}, {resid.real.max():.4f}] pu")

    print("\n\nDIAGNOSIS:")
    print("  If oracle P-feasible = NO  → convention is WRONG (wrong gen_p quantity or wrong limits)")
    print("  If oracle P-feasible = YES → convention is correct; model genuinely can't dispatch within limits")

if __name__ == "__main__":
    main()

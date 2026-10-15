"""
Quick diagnostic: are the ground-truth OPF solutions (V_true from dataset)
feasible according to our opf_metrics checker?

If V_true shows 0% feasibility, the checker is wrong.
If V_true shows ~100% feasibility, the checker is correct and the model is bad.

Usage:
    python check_oracle_feasibility.py \
        --opfdata_root ./opfdata \
        --case_name pglib_opf_case118_ieee \
        --n_scenarios 50
"""

import argparse
import torch

FEAS_TOL = 1e-2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opfdata_root", required=True)
    p.add_argument("--case_name", default="pglib_opf_case118_ieee")
    p.add_argument("--n_scenarios", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from opfdata_pipeline import make_opfdata_loaders
    from opf_task import opf_decision_space, opf_metrics
    from train_valid_test_pignn_opf import branch_rows_from_opfdata

    _, _, _, test_loader = make_opfdata_loaders(
        args.opfdata_root, args.case_name,
        batch_size=1, max_test=args.n_scenarios, num_workers=0,
    )

    results = []
    for i, batch in enumerate(test_loader):
        if i >= args.n_scenarios:
            break
        bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in batch.items() if k != "hetero"}
        hb = batch["hetero"].to(device)

        space  = opf_decision_space(bd, batch["sizes"], device)
        Y      = bd["Ybus"]
        S_load = bd["S_start"]

        V_true = bd.get("V_newton")
        if V_true is None:
            V_true = bd.get("V_target")
        if V_true is None:
            print(f"  scenario {i}: no V_true in batch — skipping")
            continue

        # evaluate ground-truth voltages through the SAME checker used for models
        m = opf_metrics(Y, V_true, V_true, S_load, space, batch["sizes"].to(device))

        feasible = (
            m["vband_max_pu"]   < FEAS_TOL and
            m["p_limit_max_pu"] < FEAS_TOL and
            m["q_limit_max_pu"] < FEAS_TOL and
            m["dp_inf_pu"]      < FEAS_TOL
        )

        results.append({
            "feasible":       feasible,
            "dp_inf_pu":      m["dp_inf_pu"],
            "p_limit_max_pu": m["p_limit_max_pu"],
            "q_limit_max_pu": m["q_limit_max_pu"],
            "vband_max_pu":   m["vband_max_pu"],
        })

        if i < 5 or not feasible:
            tag = "OK" if feasible else "FAIL"
            print(f"  [{i:3d}] {tag}  dp={m['dp_inf_pu']:.4e}  Plim={m['p_limit_max_pu']:.4e}"
                  f"  Qlim={m['q_limit_max_pu']:.4e}  Vband={m['vband_max_pu']:.4e}")

    n = len(results)
    n_feas = sum(r["feasible"] for r in results)
    import numpy as np
    print(f"\n{'='*60}")
    print(f"  Oracle feasibility (V_true through opf_metrics checker)")
    print(f"{'='*60}")
    print(f"  Feasible:       {n_feas}/{n}  ({100*n_feas/n:.1f}%)")
    print(f"  dp_inf mean:    {np.mean([r['dp_inf_pu']      for r in results]):.4e} pu")
    print(f"  P-limit mean:   {np.mean([r['p_limit_max_pu'] for r in results]):.4e} pu")
    print(f"  Q-limit mean:   {np.mean([r['q_limit_max_pu'] for r in results]):.4e} pu")
    print(f"  Vband mean:     {np.mean([r['vband_max_pu']   for r in results]):.4e} pu")
    print()
    if n_feas == n:
        print("  VERDICT: checker is CORRECT — ground truth is 100% feasible.")
        print("  The model outputs are genuinely infeasible; feasibility must be improved.")
    elif n_feas == 0:
        print("  VERDICT: checker is BROKEN — ground truth shows 0% feasibility.")
        print("  opf_metrics is computing violations incorrectly. Must fix the checker.")
    else:
        print(f"  VERDICT: PARTIAL — {n_feas}/{n} ground-truth scenarios pass.")
        print("  Some checker bug or dataset quality issue. Inspect the FAILing rows above.")


if __name__ == "__main__":
    main()

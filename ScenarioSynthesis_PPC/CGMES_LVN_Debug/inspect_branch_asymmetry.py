#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = ROOT / "ScenarioSynthesis_PPC"
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from case_generator_all_test_cases_pandapower_consider_ppc_branch_row import (  # noqa: E402
    _apply_model_a_cgmes_cleanup,
    _build_net_from_source,
)


DEFAULT_CGMES = (
    ROOT / "CGMES_to_PandaPower_clean" / "LVN_PowerFactory_fixed.zip"
)


def _branch_index(name):
    try:
        idx_brch = __import__("pandapower.pypower.idx_brch", fromlist=[name])
        return getattr(idx_brch, name)
    except Exception:
        return None


def _compile_internal_ppc(net):
    import pandapower as pp

    try:
        pp.runpp(
            net,
            init="flat",
            calculate_voltage_angles=True,
            max_iteration=1,
            enforce_q_lims=False,
            tolerance_mva=1e9,
        )
    except Exception:
        pass

    if not hasattr(net, "_ppc") or "internal" not in net._ppc:
        raise RuntimeError("pandapower did not build net._ppc['internal']")

    return net._ppc["internal"]


def _stats(name, values, tol):
    abs_values = np.abs(values)
    nz = abs_values > tol
    print(f"{name}:")
    print(f"  index              = {_branch_index(name)}")
    print(f"  nonzero(|x|>{tol:g}) = {int(nz.sum())} / {values.size}")
    print(f"  min                = {values.min():.12g}")
    print(f"  max                = {values.max():.12g}")
    print(f"  max_abs            = {abs_values.max():.12g}")
    return nz


def main():
    parser = argparse.ArgumentParser(
        description="Inspect pandapower internal PPC asymmetric branch columns."
    )
    parser.add_argument(
        "--cgmes_path",
        default=str(DEFAULT_CGMES),
        help="CGMES zip/folder path. Default: local LVN_PowerFactory_fixed.zip",
    )
    parser.add_argument("--case_name", default="LVN")
    parser.add_argument("--cgmes_version", default="2.4.15")
    parser.add_argument("--no_model_a_cleanup", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-14)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    case = {
        "cgmes_files": os.path.abspath(os.path.expanduser(args.cgmes_path)),
        "case_name": args.case_name,
        "converter_kwargs": {
            "cgmes_version": args.cgmes_version,
            "ignore_errors": True,
        },
    }

    net, loaded_name = _build_net_from_source(case, case_kwargs={})
    if not args.no_model_a_cleanup:
        _apply_model_a_cgmes_cleanup(net)

    ppc_int = _compile_internal_ppc(net)
    branch = np.asarray(ppc_int["branch"], dtype=float)

    from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X

    br_r_asym = _branch_index("BR_R_ASYM")
    br_x_asym = _branch_index("BR_X_ASYM")
    if br_r_asym is None or br_x_asym is None:
        raise RuntimeError("This pandapower version has no BR_R_ASYM/BR_X_ASYM columns.")
    if max(br_r_asym, br_x_asym) >= branch.shape[1]:
        raise RuntimeError(
            f"Internal branch table has shape {branch.shape}, but "
            f"BR_R_ASYM={br_r_asym}, BR_X_ASYM={br_x_asym}."
        )

    r = branch[:, BR_R]
    x = branch[:, BR_X]
    r_asym = branch[:, br_r_asym]
    x_asym = branch[:, br_x_asym]

    print("=" * 80)
    print("CASE")
    print("=" * 80)
    print(f"loaded_name         = {loaded_name}")
    print(f"cgmes_path          = {case['cgmes_files']}")
    print(f"branch shape        = {branch.shape}")
    print(f"BR_R index          = {BR_R}")
    print(f"BR_X index          = {BR_X}")
    print()

    print("=" * 80)
    print("ASYMMETRY SUMMARY")
    print("=" * 80)
    nz_r = _stats("BR_R_ASYM", r_asym, args.tol)
    nz_x = _stats("BR_X_ASYM", x_asym, args.tol)
    nz_any = nz_r | nz_x
    print(f"any asymmetric row  = {int(nz_any.sum())} / {branch.shape[0]}")
    print()

    print("=" * 80)
    print(f"TOP {args.top} ASYMMETRIC BRANCH ROWS")
    print("=" * 80)
    score = np.maximum(np.abs(r_asym), np.abs(x_asym))
    rows = np.where(nz_any)[0]
    if rows.size == 0:
        print("No nonzero asymmetric branch rows.")
        return

    rows = rows[np.argsort(score[rows])[::-1]][: max(args.top, 0)]
    header = (
        "row  f_bus  t_bus  "
        "R  X  R_ASYM  X_ASYM  "
        "Z_from=(R+jX)  Z_to=(R+R_ASYM+j(X+X_ASYM))"
    )
    print(header)
    for k in rows:
        z_from = complex(r[k], x[k])
        z_to = complex(r[k] + r_asym[k], x[k] + x_asym[k])
        print(
            f"{k:4d} {int(branch[k, F_BUS]):6d} {int(branch[k, T_BUS]):6d} "
            f"{r[k]: .12g} {x[k]: .12g} "
            f"{r_asym[k]: .12g} {x_asym[k]: .12g} "
            f"{z_from.real:+.8e}{z_from.imag:+.8e}j "
            f"{z_to.real:+.8e}{z_to.imag:+.8e}j"
        )


if __name__ == "__main__":
    main()

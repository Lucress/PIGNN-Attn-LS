#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import zipfile
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


DEFAULT_CGMES = ROOT / "CGMES_to_PandaPower_clean" / "LVN_PowerFactory_fixed.zip"


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


def _lookup_pp_element(net, ppc_branch_row):
    branch_lookup = getattr(net, "_pd2ppc_lookups", {}).get("branch", {})
    for element_type, span in branch_lookup.items():
        if not isinstance(span, tuple) or len(span) != 2:
            continue
        start, end = map(int, span)
        if start <= int(ppc_branch_row) < end:
            return element_type, int(ppc_branch_row) - start, start, end
    return None, None, None, None


def _safe_value(row, col):
    if col not in row.index:
        return ""
    value = row[col]
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _row_identifiers(net, element_type, element_index):
    if element_type is None or not hasattr(net, element_type):
        return {}
    table = getattr(net, element_type)
    if element_index not in table.index:
        return {}

    row = table.loc[element_index]
    preferred_cols = [
        "origin_id",
        "origin_class",
        "name",
        "description",
        "terminal_from",
        "terminal_to",
        "terminal_hv",
        "terminal_lv",
        "PowerTransformerEnd_id_hv",
        "PowerTransformerEnd_id_lv",
        "PowerTransformerEnd_id_mv",
        "PowerTransformerEnd_id",
        "EquipmentContainer_id",
    ]
    return {col: _safe_value(row, col) for col in preferred_cols if col in row.index}


def _search_zip_for_ids(zip_path, ids, max_hits_per_id=5):
    ids = [x for x in dict.fromkeys(ids) if x]
    hits = {x: [] for x in ids}
    if not ids:
        return hits

    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m for m in zf.namelist()
            if not m.endswith("/") and not m.startswith("__MACOSX/")
        ]
        for member in members:
            try:
                text = zf.read(member).decode("utf-8", errors="ignore")
            except Exception:
                continue
            for ident in ids:
                if len(hits[ident]) >= max_hits_per_id:
                    continue
                if ident in text:
                    hits[ident].append(member)
    return hits


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Locate pandapower elements/CGMES IDs for PPC branch rows with "
            "nonzero BR_R_ASYM or BR_X_ASYM."
        )
    )
    parser.add_argument("--cgmes_path", default=str(DEFAULT_CGMES))
    parser.add_argument("--case_name", default="LVN")
    parser.add_argument("--cgmes_version", default="2.4.15")
    parser.add_argument("--no_model_a_cleanup", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-14)
    parser.add_argument("--search_xml", action="store_true")
    parser.add_argument("--max_xml_hits_per_id", type=int, default=5)
    args = parser.parse_args()

    cgmes_path = os.path.abspath(os.path.expanduser(args.cgmes_path))
    case = {
        "cgmes_files": cgmes_path,
        "case_name": args.case_name,
        "converter_kwargs": {
            "cgmes_version": args.cgmes_version,
            "ignore_errors": True,
        },
    }

    net, loaded_name = _build_net_from_source(case, case_kwargs={})
    if not args.no_model_a_cleanup:
        _apply_model_a_cgmes_cleanup(net)

    _compile_internal_ppc(net)
    # Use the full external PPC branch table for element lookup. The internal
    # table can be filtered/reordered after out-of-service branches are removed.
    branch = np.asarray(net._ppc["branch"], dtype=float)

    from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X

    br_r_asym = _branch_index("BR_R_ASYM")
    br_x_asym = _branch_index("BR_X_ASYM")
    if br_r_asym is None or br_x_asym is None:
        raise RuntimeError("This pandapower version has no BR_R_ASYM/BR_X_ASYM columns.")

    r_asym = branch[:, br_r_asym]
    x_asym = branch[:, br_x_asym]
    asymmetric_rows = np.where((np.abs(r_asym) > args.tol) | (np.abs(x_asym) > args.tol))[0]

    print("=" * 100)
    print("ASYMMETRIC PPC BRANCH ROW LOCATOR")
    print("=" * 100)
    print(f"loaded_name      = {loaded_name}")
    print(f"cgmes_path       = {cgmes_path}")
    print(f"branch table     = net._ppc['branch'] (full external PPC)")
    print(f"branch shape     = {branch.shape}")
    print(f"BR_R_ASYM index  = {br_r_asym}")
    print(f"BR_X_ASYM index  = {br_x_asym}")
    print(f"nonzero rows     = {len(asymmetric_rows)}")
    print(f"branch lookups   = {getattr(net, '_pd2ppc_lookups', {}).get('branch', {})}")
    print()

    ids_for_xml_search = []
    records = []
    for ppc_row in asymmetric_rows:
        element_type, element_index, start, end = _lookup_pp_element(net, int(ppc_row))
        identifiers = _row_identifiers(net, element_type, element_index)
        for key in (
            "origin_id",
            "terminal_from",
            "terminal_to",
            "terminal_hv",
            "terminal_lv",
            "PowerTransformerEnd_id_hv",
            "PowerTransformerEnd_id_lv",
            "PowerTransformerEnd_id_mv",
            "PowerTransformerEnd_id",
        ):
            value = identifiers.get(key, "")
            if value:
                ids_for_xml_search.append(value)

        records.append((int(ppc_row), element_type, element_index, identifiers))

    xml_hits = {}
    if args.search_xml:
        if not zipfile.is_zipfile(cgmes_path):
            print("[WARN] --search_xml currently supports zip files only.")
        else:
            xml_hits = _search_zip_for_ids(
                cgmes_path,
                ids_for_xml_search,
                max_hits_per_id=max(args.max_xml_hits_per_id, 1),
            )

    for ppc_row, element_type, element_index, identifiers in records:
        print("-" * 100)
        print(
            f"ppc_branch_row={ppc_row} "
            f"f_bus={int(branch[ppc_row, F_BUS])} "
            f"t_bus={int(branch[ppc_row, T_BUS])} "
            f"element={element_type}[{element_index}]"
        )
        print(
            f"  R={branch[ppc_row, BR_R]:.12g} "
            f"X={branch[ppc_row, BR_X]:.12g} "
            f"BR_R_ASYM={branch[ppc_row, br_r_asym]:.12g} "
            f"BR_X_ASYM={branch[ppc_row, br_x_asym]:.12g}"
        )
        for key, value in identifiers.items():
            if value:
                print(f"  {key} = {value}")
                if xml_hits.get(value):
                    print(f"    XML files containing this id: {', '.join(xml_hits[value])}")


if __name__ == "__main__":
    main()

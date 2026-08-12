#!/usr/bin/env python
"""Aggregate rescore_pf.py JSON output into a table.

With one seed per grid this prints the values as-is. With several -- see the
SEEDS knob in dispatch_pf_gridfm.sh, following PFDelta's three-run protocol --
it prints mean +/- sample standard deviation across seeds, which is what the
error bars in that benchmark are.

    python summarize_rescore.py results/rescore/<group> [--tex]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict


def load(d):
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(p) as f:
            r = json.load(f)
        # A seed suffix is only present when the dispatcher was sweeping.
        grid = re.sub(r"_s\d+$", "", r.get("grid", os.path.basename(p)[:-5]))
        runs[grid].append(r)
    return runs


def agg(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, None
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, var ** 0.5


FIELDS = [
    ("mae_vmag_pu", "MAE |V|", "%.3e"),
    ("rmse_vmag_pu", "RMSE |V|", "%.3e"),
    ("mae_theta_deg", "MAE th", "%.3f"),
    ("rmse_theta_deg", "RMSE th", "%.3f"),
    ("slope_vmag", "slope|V|", "%.3f"),
    ("R2_vmag", "R2|V|", "%.4f"),
    ("slope_sin", "slope sin", "%.3f"),
    ("R2_sin", "R2 sin", "%.4f"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tex", action="store_true")
    ap.add_argument("--sort", default="mae_vmag_pu")
    a = ap.parse_args()

    runs = load(a.dir)
    if not runs:
        raise SystemExit(f"no rescore JSON under {a.dir}")
    rows = []
    for grid, rs in runs.items():
        row = {"grid": grid, "seeds": len(rs)}
        for k, _, _ in FIELDS:
            vals = [r[k] for r in rs if k in r and r[k] == r[k]]
            row[k] = agg(vals) if vals else (float("nan"), None)
        rows.append(row)
    rows.sort(key=lambda r: r[a.sort][0])

    nseed = max(r["seeds"] for r in rows)
    if a.tex:
        for r in rows:
            cells = []
            for k, _, fmt in FIELDS:
                m, sd = r[k]
                cells.append((fmt % m) if sd is None else f"{fmt % m} $\\pm$ {fmt % sd}")
            print(f"{r['grid'].replace('_', chr(92)+'_')} & " + " & ".join(cells) + r" \\")
        return
    hdr = f"{'grid':<22}" + "".join(f"{lbl:>13}" for _, lbl, _ in FIELDS)
    print(hdr + ("   seeds" if nseed > 1 else ""))
    for r in rows:
        line = f"{r['grid']:<22}"
        for k, _, fmt in FIELDS:
            m, sd = r[k]
            line += f"{(fmt % m):>13}" if sd is None else f"{(fmt % m)+'±'+(fmt % sd):>13}"
        print(line + (f"   {r['seeds']}" if nseed > 1 else ""))


if __name__ == "__main__":
    main()

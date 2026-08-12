#!/usr/bin/env python3
"""Summarize --task opf training logs (balance / limits / band / cost)."""
import re, sys, glob, math
from pathlib import Path
NUM = r"[-+0-9.eE]+"
FINAL = re.compile(rf"Final test-set RMSE : ({NUM}) \(\|V\|: ({NUM}), theta: ({NUM})deg\)")
EP = re.compile(rf"^Epoch\s+(\d+)\s*\|(.*)$", re.M)
PH = re.compile(
    rf"(?P<phase>train|valid|test) loss (?P<loss>{NUM}) mse (?P<mse>{NUM}) phys (?P<phys>{NUM}) "
    rf"rmse (?P<rmse>{NUM}) \(mag (?P<mag>{NUM}), ang (?P<ang>{NUM})deg\) "
    rf"balance\(dPinf (?P<dp>{NUM}), dQinf (?P<dq>{NUM}) pu over (?P<nfree>{NUM}) free buses\) "
    rf"limits\(P (?P<lp>{NUM}), Q (?P<lq>{NUM}) pu over (?P<nctrl>{NUM}) ctrl buses\) "
    rf"vband\(max (?P<vb>{NUM}), frac (?P<vbf>{NUM})\) cost\(gap (?P<gap>{NUM})%\)")
def rows(paths):
    out = []
    for p in paths:
        t = Path(p).read_text(errors="replace")
        name = Path(p).stem.replace("_training_log", "").replace("opf_", "")
        best = None; zero = None
        for m in EP.finditer(t):
            ep = int(m.group(1))
            for ph in PH.finditer(m.group(2)):
                d = ph.groupdict()
                if ep == 0 and d["phase"] == "test":
                    zero = d
                if d["phase"] == "valid" and ep > 0:
                    v = float(d["loss"])
                    if best is None or v < best[1]:
                        best = (ep, v, d)
        fin = FINAL.search(t)
        src = best[2] if best else zero
        if src is None: continue
        out.append({
            "run": name,
            "row": (f"final test (best ep{best[0]})" if (fin and best) else
                    ("zero-shot" if zero and not best else f"best valid ep{best[0]}" if best else "?")),
            "rmse": float(fin.group(1)) if fin else float(src["rmse"]),
            "mag": float(fin.group(2)) if fin else float(src["mag"]),
            "ang": float(fin.group(3)) if fin else float(src["ang"]),
            "dp": float(src["dp"]), "dq": float(src["dq"]),
            "lp": float(src["lp"]), "lq": float(src["lq"]),
            "vb": float(src["vb"]), "gap": float(src["gap"]),
        })
    return out
paths = []
for a in sys.argv[1:]:
    paths.extend(sorted(glob.glob(a)))
r = rows(paths)
hdr = f"{'run':<26} {'row':<24} {'RMSE':>10} {'|V|':>10} {'theta':>8} {'dPinf':>10} {'dQinf':>10} {'limP':>9} {'limQ':>9} {'vband':>9} {'cost%':>9}"
print(hdr); print("-"*len(hdr))
for x in sorted(r, key=lambda z: z["run"]):
    print(f"{x['run']:<26} {x['row']:<24} {x['rmse']:10.4e} {x['mag']:10.4e} {x['ang']:8.3f} "
          f"{x['dp']:10.3e} {x['dq']:10.3e} {x['lp']:9.3e} {x['lq']:9.3e} {x['vb']:9.2e} {x['gap']:9.2f}")

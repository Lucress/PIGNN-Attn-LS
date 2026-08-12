#!/usr/bin/env python3
"""
Summarize GridFM / GridSFM / LUMINA training logs into report-ready rows.

The three surrogate scripts (`train_valid_test_gridfm.py`,
`train_valid_test_gridsfm.py`, `train_valid_test_lumina.py`) share the same
log format, so one parser covers all of them:

  Epoch  12 | train ... | valid loss ... rmse ... (mag ..., ang ...deg)
             (dPinf ... pu, dQinf ... pu; ... MW, ... MVAr) mean|dP| ..., ...
  Final test-set RMSE : ... (|V|: ..., theta: ...deg) | dPinf : ... | dQinf : ...
  mean|dP| ..., mean|dQ| ..., p95|dP| ..., p95|dQ| ..., rmse|dP| ..., rmse|dQ| ...

For every log the script reports the completed final-test block when present,
otherwise the best-validation epoch (lowest validation loss, i.e. the epoch the
training script checkpointed).

Usage:
  python summarize_model_logs.py --log-dir results/logs/lumina_lvn_a100_*/
  python summarize_model_logs.py --log-dir DIR --format latex
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

NUM = r"[-+0-9.eE]+"

EPOCH_RE = re.compile(rf"^Epoch\s+(?P<epoch>\d+)\s*\|(?P<body>.*)$", re.M)
PHASE_RE = re.compile(
    rf"(?P<phase>train|valid|test) loss (?P<loss>{NUM}) mse (?P<mse>{NUM}) "
    rf"phys (?P<phys>{NUM}) rmse (?P<rmse>{NUM}) \(mag (?P<rmse_mag>{NUM}), "
    rf"ang (?P<rmse_ang>{NUM})deg\) \(dPinf (?P<dp_inf>{NUM}) pu, "
    rf"dQinf (?P<dq_inf>{NUM}) pu; (?P<dp_mw>{NUM}) MW, (?P<dq_mvar>{NUM}) MVAr\) "
    rf"(?P<dist>mean\|dP\|.*?)(?=\s+\|\s+(?:train|valid|test|time)\b|$)"
)
DIST_RE = re.compile(
    rf"mean\|dP\| (?P<mean_dp>{NUM}), mean\|dQ\| (?P<mean_dq>{NUM}), "
    rf"p95\|dP\| (?P<p95_dp>{NUM}), p95\|dQ\| (?P<p95_dq>{NUM})"
    rf"(?:, rmse\|dP\| (?P<rmse_dp>{NUM}), rmse\|dQ\| (?P<rmse_dq>{NUM}))?"
)
FINAL_RE = re.compile(
    rf"Final test-set RMSE : (?P<rmse>{NUM}) \(\|V\|: (?P<rmse_mag>{NUM}), "
    rf"theta: (?P<rmse_ang>{NUM})deg\) \| dPinf : (?P<dp_inf>{NUM}) pu "
    rf"\((?P<dp_mw>{NUM}) MW\) \| dQinf : (?P<dq_inf>{NUM}) pu "
    rf"\((?P<dq_mvar>{NUM}) MVAr\)\s*\n(?P<dist>mean\|dP\|.*)"
)
RUN_RE = re.compile(r"^\[run\] (?P<run>.+)$", re.M)
MODEL_RE = re.compile(r"^\[model\] (?P<model>\w+) init_mode=(?P<init_mode>\w+)", re.M)
PARAMS_RE = re.compile(r"^\[model\] trainable_params=(?P<params>[\d,]+)", re.M)


def _f(value: Optional[str]) -> float:
    return float(value) if value not in (None, "") else math.nan


def _dist(text: str) -> Dict[str, float]:
    m = DIST_RE.search(text or "")
    if m is None:
        return {}
    return {k: _f(v) for k, v in m.groupdict().items()}


def parse_log(path: Path) -> Optional[Dict[str, object]]:
    text = path.read_text(errors="replace")
    run_m = RUN_RE.search(text)
    model_m = MODEL_RE.search(text)
    params_m = PARAMS_RE.search(text)

    row: Dict[str, object] = {
        "log": path.name,
        "run": run_m.group("run") if run_m else path.stem,
        "model": model_m.group("model") if model_m else "",
        "init_mode": model_m.group("init_mode") if model_m else "",
        "params": params_m.group("params") if params_m else "",
    }

    final_m = FINAL_RE.search(text)
    if final_m is not None:
        row["reported_row"] = "final test"
        row.update({k: _f(v) for k, v in final_m.groupdict().items() if k != "dist"})
        row.update(_dist(final_m.group("dist")))
        row["epoch"] = _last_epoch(text)
        return row

    best = _best_valid(text)
    if best is None:
        zero_shot = _zero_shot_test(text)
        if zero_shot is None:
            row["reported_row"] = "no metrics yet"
            return row
        row["reported_row"] = "valid/test ep0"
        row.update(zero_shot)
        return row
    row["reported_row"] = f"best valid ep{int(best['epoch'])}"
    row.update(best)
    return row


def _epoch_phases(text: str):
    for m in EPOCH_RE.finditer(text):
        epoch = int(m.group("epoch"))
        for p in PHASE_RE.finditer(m.group("body")):
            yield epoch, p


def _last_epoch(text: str) -> float:
    epochs = [e for e, _ in _epoch_phases(text)]
    return float(max(epochs)) if epochs else math.nan


def _phase_row(epoch: int, match: re.Match) -> Dict[str, float]:
    g = match.groupdict()
    row = {k: _f(v) for k, v in g.items() if k not in ("phase", "dist")}
    row["epoch"] = float(epoch)
    row.update(_dist(g.get("dist", "")))
    return row


def _best_valid(text: str) -> Optional[Dict[str, float]]:
    best = None
    for epoch, m in _epoch_phases(text):
        if m.group("phase") != "valid" or epoch == 0:
            continue
        row = _phase_row(epoch, m)
        if best is None or row["loss"] < best["loss"]:
            best = row
    return best


def _zero_shot_test(text: str) -> Optional[Dict[str, float]]:
    for epoch, m in _epoch_phases(text):
        if epoch == 0 and m.group("phase") == "test":
            return _phase_row(epoch, m)
    for epoch, m in _epoch_phases(text):
        if epoch == 0 and m.group("phase") == "valid":
            return _phase_row(epoch, m)
    return None


COLUMNS = [
    ("run", "Run", "s"),
    ("reported_row", "Reported row", "s"),
    ("rmse", "RMSE", "e"),
    ("rmse_mag", "|V| RMSE", "e"),
    ("rmse_ang", "theta [deg]", "f"),
    ("dp_inf", "dPinf", "e"),
    ("dq_inf", "dQinf", "e"),
    ("mean_dp", "mean|dP|", "e"),
    ("mean_dq", "mean|dQ|", "e"),
    ("p95_dp", "p95|dP|", "e"),
    ("p95_dq", "p95|dQ|", "e"),
]


def _cell(row: Dict[str, object], key: str, kind: str) -> str:
    value = row.get(key, "")
    if kind == "s":
        return str(value)
    if not isinstance(value, float) or math.isnan(value):
        return "--"
    return f"{value:.4f}" if kind == "f" else f"{value:.4e}"


def render(rows: List[Dict[str, object]], fmt: str) -> str:
    header = [label for _, label, _ in COLUMNS]
    body = [[_cell(r, k, t) for k, _, t in COLUMNS] for r in rows]
    if fmt == "latex":
        lines = [" & ".join(header) + r" \\", r"\midrule"]
        lines += [" & ".join(c.replace("_", r"\_") for c in r) + r" \\" for r in body]
        return "\n".join(lines)
    if fmt == "csv":
        return "\n".join([",".join(header)] + [",".join(r) for r in body])
    widths = [max(len(header[i]), *(len(r[i]) for r in body)) if body else len(header[i])
              for i in range(len(header))]
    out = [" | ".join(h.ljust(w) for h, w in zip(header, widths))]
    out.append("-|-".join("-" * w for w in widths))
    out += [" | ".join(c.ljust(w) for c, w in zip(r, widths)) for r in body]
    return "\n".join(out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-dir", nargs="+", required=True,
                   help="Directories (globs allowed) holding *_training_log.txt files")
    p.add_argument("--pattern", default="*_training_log.txt")
    p.add_argument("--format", choices=("table", "csv", "latex"), default="table")
    return p.parse_args()


def main():
    args = parse_args()
    paths: List[Path] = []
    for entry in args.log_dir:
        for d in sorted(glob.glob(entry)) or [entry]:
            paths.extend(sorted(Path(d).glob(args.pattern)))
    if not paths:
        raise SystemExit(f"No logs matching {args.pattern} under {args.log_dir}")

    rows = [r for r in (parse_log(p) for p in paths) if r is not None]
    print(render(rows, args.format))


if __name__ == "__main__":
    main()

"""Phase A ablation sweep — lambda_kcl_inf sweep for KCL L∞ feasibility.

Runs each ablation config sequentially, collects test metrics from each run's
final_metrics.json, and writes a ranked ablation table to results/ablation/.

Usage (from the project root with the LVN environment active):

    python -m train.ablation_runner

Options:
    --configs   Glob pattern for ablation config files (default: configs/ablation/linf_*.yaml)
    --out       Output directory for the summary table (default: results/ablation)
    --epochs    Override the number of training epochs (optional)

The ablation runner also prints a formatted table to stdout at the end, ranked
by feasibility_rate (primary) then kcl_inf_norm (secondary).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


# ── Config discovery ──────────────────────────────────────────────────────────

_DEFAULT_CONFIGS = [
    "configs/ablation/linf_w0.yaml",
    "configs/ablation/linf_w1.yaml",
    "configs/ablation/linf_w5.yaml",
    "configs/ablation/linf_w10.yaml",
    "configs/ablation/linf_w20.yaml",
    "configs/ablation/linf_w50.yaml",
]

_KEY_METRICS = [
    "test/feasibility_rate",
    "test/kcl_inf_norm",
    "test/kcl_loss",
    "test/gen_cost",
    "test/branch_max",
    "test/lim_max",
    "best/val_kcl",
    "best/epoch",
]


# ── Run helpers ───────────────────────────────────────────────────────────────

def _existing_run_dirs(base: Path) -> set[Path]:
    if not base.exists():
        return set()
    return {p for p in base.iterdir() if p.is_dir()}


def _find_new_run_dir(base: Path, known: set[Path]) -> Path | None:
    current = {p for p in base.iterdir() if p.is_dir()}
    new = current - known
    if not new:
        return None
    return max(new, key=lambda p: p.stat().st_mtime)


def _read_final_metrics(run_dir: Path) -> dict:
    candidate = run_dir / "artifacts" / "final_metrics.json"
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _extract_lambda_kcl_inf(config_path: str) -> float:
    """Parse lambda_kcl_inf from a YAML config file without a full YAML parser."""
    try:
        import yaml  # optional
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return float(raw.get("model", {}).get("lambda_kcl_inf", 0.0))
    except ImportError:
        pass
    # Fallback: grep the file
    with open(config_path, encoding="utf-8") as f:
        for line in f:
            if "lambda_kcl_inf" in line and ":" in line:
                try:
                    return float(line.split(":")[1].strip())
                except ValueError:
                    pass
    return 0.0


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_ablation(
    configs: list[str],
    out_dir: str = "results/ablation",
    epoch_override: int | None = None,
) -> list[dict]:
    """Run each config, collect metrics, return sorted results list."""
    from train.entrypoint import main as train_main

    runs_base = Path("results/runs")
    out_path  = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, cfg_path in enumerate(configs):
        if not Path(cfg_path).exists():
            print(f"  [SKIP] Config not found: {cfg_path}", flush=True)
            continue

        lval = _extract_lambda_kcl_inf(cfg_path)
        label = f"lambda_kcl_inf={lval:.1f}"
        print(f"\n{'='*60}", flush=True)
        print(f"  Run {idx+1}/{len(configs)}: {label}  ({cfg_path})", flush=True)
        print(f"{'='*60}", flush=True)

        known_dirs = _existing_run_dirs(runs_base)
        t0 = time.time()

        argv = ["--config", cfg_path]
        if epoch_override is not None:
            argv += ["--EPOCHS", str(epoch_override)]

        try:
            train_main(argv)
        except SystemExit:
            pass
        except Exception as exc:
            print(f"  [ERROR] Training failed: {exc}", flush=True)
            results.append({"label": label, "lambda_kcl_inf": lval, "error": str(exc)})
            continue

        elapsed = time.time() - t0

        new_dir = _find_new_run_dir(runs_base, known_dirs)
        if new_dir is None:
            print(f"  [WARN] Could not locate new run directory.", flush=True)
            results.append({"label": label, "lambda_kcl_inf": lval, "error": "run_dir_not_found"})
            continue

        metrics = _read_final_metrics(new_dir)
        row = {
            "label":           label,
            "lambda_kcl_inf":  lval,
            "run_dir":         str(new_dir),
            "elapsed_min":     round(elapsed / 60.0, 1),
        }
        for k in _KEY_METRICS:
            row[k] = metrics.get(k, float("nan"))

        results.append(row)
        feas = row.get("test/feasibility_rate", float("nan"))
        linf = row.get("test/kcl_inf_norm", float("nan"))
        print(
            f"  Done in {row['elapsed_min']:.1f} min | "
            f"feasibility={feas*100:.1f}%  kcl_inf={linf:.4f} pu",
            flush=True,
        )

    return results


# ── Table rendering ───────────────────────────────────────────────────────────

def _fmt(val, fmt=".4f", pct=False):
    if isinstance(val, float) and val != val:  # NaN
        return "—"
    if pct:
        return f"{val*100:.1f}%"
    return format(val, fmt)


def print_table(results: list[dict]) -> None:
    sorted_rows = sorted(
        [r for r in results if "error" not in r],
        key=lambda r: (-r.get("test/feasibility_rate", -1), r.get("test/kcl_inf_norm", 999)),
    )
    if "error" in (results[0] if results else {}):
        sorted_rows = results

    cols = [
        ("λ_kcl_inf", "lambda_kcl_inf", ".1f", False),
        ("feasibility", "test/feasibility_rate", ".3f", True),
        ("kcl_L∞ (pu)", "test/kcl_inf_norm", ".4f", False),
        ("kcl_L2", "test/kcl_loss", ".4f", False),
        ("gen_cost", "test/gen_cost", ".4f", False),
        ("branch_max", "test/branch_max", ".4f", False),
        ("lim_max", "test/lim_max", ".4f", False),
        ("best_ep", "best/epoch", ".0f", False),
    ]

    header = "  ".join(f"{h:<12}" for h, *_ in cols)
    sep    = "  ".join("-"*12 for _ in cols)
    print(f"\n{'─'*80}")
    print("  Phase A Ablation — KCL L∞ sweep (ranked by feasibility rate)")
    print(f"{'─'*80}")
    print(header)
    print(sep)
    for row in sorted_rows:
        if "error" in row:
            print(f"  {row['label']}  ERROR: {row['error']}")
            continue
        line = "  ".join(
            f"{_fmt(row.get(k, float('nan')), fmt, pct):<12}"
            for _, k, fmt, pct in cols
        )
        print(line)
    print(f"{'─'*80}\n")


def save_csv(results: list[dict], out_dir: str) -> Path:
    out = Path(out_dir) / "ablation_kcl_linf.csv"
    if not results:
        return out
    fieldnames = list(results[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in results:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    return out


def recommend_best(results: list[dict]) -> dict | None:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None
    return max(valid, key=lambda r: (r.get("test/feasibility_rate", -1), -r.get("test/kcl_inf_norm", 999)))


# ── CLI entry ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase A: λ_kcl_inf ablation sweep")
    parser.add_argument(
        "--configs", nargs="+", default=_DEFAULT_CONFIGS,
        help="Config YAML paths to sweep (default: configs/ablation/linf_w*.yaml)",
    )
    parser.add_argument("--out",    default="results/ablation", help="Output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    args = parser.parse_args(argv)

    # Ensure we can import train modules
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    print(f"\nPhase A — KCL L∞ ablation sweep")
    print(f"Configs: {args.configs}")
    print(f"Output:  {args.out}\n")

    results = run_ablation(
        configs=args.configs,
        out_dir=args.out,
        epoch_override=args.epochs,
    )

    print_table(results)

    csv_path = save_csv(results, args.out)
    print(f"Saved: {csv_path}")

    best = recommend_best(results)
    if best:
        print(
            f"\nBest config: lambda_kcl_inf={best['lambda_kcl_inf']:.1f}  "
            f"→ feasibility={best.get('test/feasibility_rate', 0)*100:.1f}%  "
            f"kcl_inf={best.get('test/kcl_inf_norm', 0):.4f} pu"
        )
        print(f"Run dir: {best.get('run_dir', '?')}")
        print(
            "\nNext: update configs/opf_pignn_attn_ls.yaml with the best lambda_kcl_inf "
            "and run 100-epoch training to produce the improved Model 2 checkpoint."
        )


if __name__ == "__main__":
    main()

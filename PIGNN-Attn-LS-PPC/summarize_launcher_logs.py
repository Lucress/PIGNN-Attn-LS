#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


PARAM_RE = re.compile(r"Total number of parameters: (\d+)")
TEST_RE = re.compile(
    r"Test physics-loss : ([0-9.e+-]+) \| total RMSE : ([0-9.e+-]+) "
    r"\| \|V\| RMSE : ([0-9.e+-]+) \| θ RMSE : ([0-9.e+-]+)°"
    r"(?: \| ΔP∞ : ([0-9.e+-]+) pu .*? \| ΔQ∞ : ([0-9.e+-]+) pu .*?)?$",
    re.M,
)
OLD_NAME_RE = re.compile(
    r"(?P<idx>\d+)_nhead(?P<nheads>\d+)_attn(?P<layers>\d+)(?:_.*)?_launcher\.log$"
)
NEW_NAME_RE = re.compile(
    r"(?P<idx>\d+)_nhead(?P<nheads>\d+)_dhi(?P<dhi>\d+)_attn(?P<layers>\d+)(?:_.*)?_launcher\.log$"
)
EPOCH_RE = re.compile(r"^Epoch\s+(\d+)\s+\|", re.M)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize launcher.log sweep results.")
    parser.add_argument("--log-dir", required=True, help="Directory containing *_launcher.log files")
    return parser.parse_args()


def parse_name(path: Path):
    for regex in (NEW_NAME_RE, OLD_NAME_RE):
        match = regex.search(path.name)
        if match is not None:
            groups = match.groupdict()
            return {
                "run_idx": int(groups["idx"]),
                "n_heads": int(groups["nheads"]),
                "layers": int(groups["layers"]),
                "d_hi": None if "dhi" not in groups else int(groups["dhi"]),
            }
    return {"run_idx": None, "n_heads": None, "layers": None, "d_hi": None}


def classify_status(text: str):
    if TEST_RE.search(text):
        return "completed"
    if "OutOfMemoryError" in text or "CUDA out of memory" in text:
        return "oom"
    if "Traceback" in text or "RuntimeError" in text:
        return "error"
    return "incomplete"


def load_rows(log_dir: Path):
    rows = []
    for path in sorted(log_dir.glob("*_launcher.log")):
        text = path.read_text(errors="replace")
        name_info = parse_name(path)
        test_match = TEST_RE.search(text)
        param_match = PARAM_RE.search(text)
        epochs = EPOCH_RE.findall(text)

        row = {
            "file": path.name,
            "label": path.name.replace("_launcher.log", ""),
            "status": classify_status(text),
            "params": None if param_match is None else int(param_match.group(1)),
            "physics_loss": None,
            "total_rmse": None,
            "v_rmse": None,
            "theta_rmse_deg": None,
            "dp_inf_pu": None,
            "dq_inf_pu": None,
            "last_epoch": None if not epochs else int(epochs[-1]),
        }
        row.update(name_info)

        if test_match is not None:
            row["physics_loss"] = float(test_match.group(1))
            row["total_rmse"] = float(test_match.group(2))
            row["v_rmse"] = float(test_match.group(3))
            row["theta_rmse_deg"] = float(test_match.group(4))
            if test_match.group(5) is not None:
                row["dp_inf_pu"] = float(test_match.group(5))
                row["dq_inf_pu"] = float(test_match.group(6))

        if row["run_idx"] is None:
            row["run_idx"] = len(rows) + 1
        rows.append(row)

    rows.sort(key=lambda r: r["run_idx"])
    return rows


def write_status_tsv(rows, out_path: Path):
    header = [
        "run_idx",
        "label",
        "status",
        "n_heads",
        "d_hi",
        "layers",
        "params",
        "last_epoch",
        "physics_loss",
        "total_rmse",
        "v_rmse_pu",
        "theta_rmse_deg",
        "dp_inf_pu",
        "dq_inf_pu",
        "file",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["run_idx"]),
                    row["label"],
                    row["status"],
                    "" if row["n_heads"] is None else str(row["n_heads"]),
                    "" if row["d_hi"] is None else str(row["d_hi"]),
                    "" if row["layers"] is None else str(row["layers"]),
                    "" if row["params"] is None else str(row["params"]),
                    "" if row["last_epoch"] is None else str(row["last_epoch"]),
                    "" if row["physics_loss"] is None else f"{row['physics_loss']:.6e}",
                    "" if row["total_rmse"] is None else f"{row['total_rmse']:.6e}",
                    "" if row["v_rmse"] is None else f"{row['v_rmse']:.6e}",
                    "" if row["theta_rmse_deg"] is None else f"{row['theta_rmse_deg']:.6f}",
                    "" if row["dp_inf_pu"] is None else f"{row['dp_inf_pu']:.6e}",
                    "" if row["dq_inf_pu"] is None else f"{row['dq_inf_pu']:.6e}",
                    row["file"],
                ]
            )
        )
    out_path.write_text("\n".join(lines) + "\n")


def write_ranking_tsv(rows, out_path: Path):
    completed = [row.copy() for row in rows if row["status"] == "completed"]
    if not completed:
        out_path.write_text(
            "run_idx\tlabel\tstatus\tn_heads\td_hi\tlayers\tparams\tphysics_loss\tv_rmse_pu\t"
            "theta_rmse_deg\tdp_inf_pu\tdq_inf_pu\n"
        )
        return

    for metric_name, key in (
        ("theta_rank", "theta_rmse_deg"),
        ("v_rank", "v_rmse"),
        ("physics_rank", "physics_loss"),
        ("dp_rank", "dp_inf_pu"),
        ("dq_rank", "dq_inf_pu"),
        ("param_rank", "params"),
    ):
        sortable = [row for row in completed if row[key] is not None]
        for rank, row in enumerate(sorted(sortable, key=lambda r: r[key]), start=1):
            row[metric_name] = rank

    for row in completed:
        row["performance_rank_sum"] = row["theta_rank"] + row["v_rank"] + row["physics_rank"]

    for i, row_a in enumerate(completed):
        dominated = False
        for j, row_b in enumerate(completed):
            if i == j:
                continue
            if (
                row_b["theta_rmse_deg"] <= row_a["theta_rmse_deg"]
                and row_b["v_rmse"] <= row_a["v_rmse"]
                and row_b["physics_loss"] <= row_a["physics_loss"]
                and (
                    row_b["theta_rmse_deg"] < row_a["theta_rmse_deg"]
                    or row_b["v_rmse"] < row_a["v_rmse"]
                    or row_b["physics_loss"] < row_a["physics_loss"]
                )
            ):
                dominated = True
                break
        row_a["pareto"] = "yes" if not dominated else "no"

    header = [
        "run_idx",
        "label",
        "status",
        "n_heads",
        "d_hi",
        "layers",
        "params",
        "physics_loss",
        "v_rmse_pu",
        "theta_rmse_deg",
        "dp_inf_pu",
        "dq_inf_pu",
        "theta_rank",
        "v_rank",
        "physics_rank",
        "dp_rank",
        "dq_rank",
        "param_rank",
        "performance_rank_sum",
        "pareto",
    ]
    lines = ["\t".join(header)]
    for row in sorted(completed, key=lambda r: (r["theta_rmse_deg"], r["v_rmse"], r["physics_loss"])):
        lines.append(
            "\t".join(
                [
                    str(row["run_idx"]),
                    row["label"],
                    row["status"],
                    str(row["n_heads"]),
                    "" if row["d_hi"] is None else str(row["d_hi"]),
                    str(row["layers"]),
                    "" if row["params"] is None else str(row["params"]),
                    f"{row['physics_loss']:.6e}",
                    f"{row['v_rmse']:.6e}",
                    f"{row['theta_rmse_deg']:.6f}",
                    "" if row["dp_inf_pu"] is None else f"{row['dp_inf_pu']:.6e}",
                    "" if row["dq_inf_pu"] is None else f"{row['dq_inf_pu']:.6e}",
                    str(row["theta_rank"]),
                    str(row["v_rank"]),
                    str(row["physics_rank"]),
                    "" if "dp_rank" not in row else str(row["dp_rank"]),
                    "" if "dq_rank" not in row else str(row["dq_rank"]),
                    "" if "param_rank" not in row else str(row["param_rank"]),
                    str(row["performance_rank_sum"]),
                    row["pareto"],
                ]
            )
        )
    out_path.write_text("\n".join(lines) + "\n")


def write_plot(rows, out_path: Path, title: str):
    completed = [row for row in rows if row["status"] == "completed"]
    if not completed:
        raise SystemExit("No completed runs with test metrics were found.")

    completed.sort(key=lambda r: r["run_idx"])
    x = list(range(len(completed)))
    labels = [f"{row['run_idx']:02d}\nh{row['n_heads']}\nd{row['d_hi']}\nL{row['layers']}" for row in completed]
    colors = []
    for row in completed:
        if row["n_heads"] == 8:
            colors.append("#1f6aa5")
        elif row["n_heads"] == 12:
            colors.append("#2a9d8f")
        elif row["n_heads"] == 16:
            colors.append("#c26d22")
        else:
            colors.append("#666666")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)

    metric_specs = [
        ("v_rmse", "|V| RMSE (p.u.)", True),
        ("theta_rmse_deg", "theta RMSE (deg)", False),
        ("dp_inf_pu", "ΔP∞ (p.u.)", True),
        ("dq_inf_pu", "ΔQ∞ (p.u.)", True),
    ]

    for ax, (key, ylabel, use_log) in zip(axes, metric_specs):
        values = [row[key] for row in completed]
        ax.bar(x, values, color=colors, alpha=0.9)
        ax.plot(x, values, color="#222222", linewidth=1, alpha=0.6)
        ax.set_ylabel(ylabel)
        if use_log:
            ax.set_yscale("log")
        best_idx = min(range(len(completed)), key=lambda i: completed[i][key])
        ax.scatter([x[best_idx]], [values[best_idx]], color="#d62828", s=70, zorder=4)
        ax.annotate(
            f"{values[best_idx]:.3e}" if use_log else f"{values[best_idx]:.3f}",
            xy=(x[best_idx], values[best_idx]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
        )

    axes[0].set_title(title)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    axes[-1].set_xlabel("completed runs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def main():
    args = parse_args()
    log_dir = Path(args.log_dir).resolve()
    rows = load_rows(log_dir)
    if not rows:
        raise SystemExit(f"No *_launcher.log files found in {log_dir}")

    stem = log_dir.name
    status_tsv = log_dir / f"{stem}_status.tsv"
    ranking_tsv = log_dir / f"{stem}_ranking.tsv"
    plot_png = log_dir / f"{stem}_test_metrics_by_run.png"

    write_status_tsv(rows, status_tsv)
    write_ranking_tsv(rows, ranking_tsv)
    write_plot(rows, plot_png, f"{stem}: completed test metrics from launcher logs")

    print(plot_png)
    print(status_tsv)
    print(ranking_tsv)


if __name__ == "__main__":
    main()

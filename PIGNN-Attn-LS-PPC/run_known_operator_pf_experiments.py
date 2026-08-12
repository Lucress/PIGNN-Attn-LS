#!/usr/bin/env python3
"""Build or execute the first Known-Operator AC-PF experiment matrix.

The default is a dry run.  Pass ``--execute`` after inspecting the emitted
commands.  Every selected backbone receives the same PF correction settings;
the GridFM mirror and released gridfm-graphkit implementation are separate
backbones in the matrix.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=("pignn", "gridfm-mirror", "gridfm-graphkit", "gridsfm", "lumina"),
        default=("pignn", "gridfm-mirror", "gridfm-graphkit", "gridsfm", "lumina"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("raw", "dpf-1", "dpf-3"),
        default=("raw", "dpf-1", "dpf-3"),
        help="raw baseline, one-step KOL, and three-step KOL training variants.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.3333)
    parser.add_argument("--valid-ratio", type=float, default=0.3333)
    parser.add_argument("--kol-eval-steps", type=int, default=100)
    parser.add_argument("--kol-lr", type=float, default=1e-3)
    parser.add_argument("--kol-tol", type=float, default=1e-6)
    parser.add_argument("--target-s-base", type=float, default=1e8)
    parser.add_argument("--gridsfm-checkpoint", type=Path)
    parser.add_argument("--lumina-checkpoint", type=Path)
    parser.add_argument("--lumina-config", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "known_operator_pf")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=(),
        help="Arguments appended verbatim to every generated training command.",
    )
    return parser.parse_args()


def common_args(args, run_name):
    return [
        "--PARQUET", str(args.parquet),
        "--task", "pf",
        "--PER_UNIT",
        "--target_S_base", str(args.target_s_base),
        "--dataset_complex_dtype", "complex128",
        "--share_grid",
        "--lazy_parquet",
        "--BATCH", str(args.batch),
        "--EPOCHS", str(args.epochs),
        "--train_ratio", str(args.train_ratio),
        "--valid_ratio", str(args.valid_ratio),
        "--run_name", run_name,
        "--log_to_file",
        "--log_dir", str(args.output_root / "logs"),
        "--ckpt_dir", str(args.output_root / "ckpt"),
    ]


def known_operator_args(args, variant):
    if variant == "raw":
        return ["--kol_pf_mode", "off"]
    train_steps = {"dpf-1": 1, "dpf-3": 3}[variant]
    return [
        "--kol_pf_mode", "dpf",
        "--kol_train_steps", str(train_steps),
        "--kol_eval_steps", str(args.kol_eval_steps),
        "--kol_lr", str(args.kol_lr),
        "--kol_tol", str(args.kol_tol),
        "--kol_optimizer", "adam",
        "--kol_raw_loss_weight", "1.0",
    ]


def command_for(args, backbone, variant):
    run_name = f"kol_{backbone.replace('-', '_')}_{variant.replace('-', '_')}_{args.parquet.stem}"
    kol = known_operator_args(args, variant)

    if backbone == "pignn":
        # train_valid_test.py is the older CLI and has no --task argument.
        common = common_args(args, run_name)
        task_pos = common.index("--task")
        del common[task_pos:task_pos + 2]
        cmd = [
            args.python,
            str(ROOT / "train_valid_test.py"),
            *common,
            "--PINN",
            "--mse_weight", "1.0",
            "--model", "GNSMsg_EdgeSelfAttn",
            "--use_armijo",
            "--preserve_zero_heads",
            *kol,
        ]
    elif backbone.startswith("gridfm-"):
        impl = backbone.split("-", 1)[1]
        cmd = [
            args.python,
            str(ROOT / "train_valid_test_gridfm.py"),
            *common_args(args, run_name),
            "--gridfm_impl", impl,
            "--mse_weight", "1.0",
            "--physics_weight", "0.01",
            *kol,
        ]
    elif backbone == "gridsfm":
        init = ["--init_mode", "scratch"]
        if args.gridsfm_checkpoint:
            init = [
                "--init_mode", "pretrained",
                "--pretrained_checkpoint", str(args.gridsfm_checkpoint),
            ]
        cmd = [
            args.python,
            str(ROOT / "train_valid_test_gridsfm.py"),
            *common_args(args, run_name),
            *init,
            "--mse_weight", "1.0",
            "--physics_weight", "0.01",
            *kol,
        ]
    else:
        init = ["--init_mode", "scratch"]
        if args.lumina_checkpoint:
            init = [
                "--init_mode", "pretrained",
                "--pretrained_checkpoint", str(args.lumina_checkpoint),
            ]
        if args.lumina_config:
            init += ["--model_config", str(args.lumina_config)]
        cmd = [
            args.python,
            str(ROOT / "train_valid_test_lumina.py"),
            *common_args(args, run_name),
            *init,
            "--mse_weight", "1.0",
            "--physics_weight", "0.01",
            *kol,
        ]
    return cmd + list(args.extra)


def main():
    args = parse_args()
    if not args.parquet.exists():
        raise FileNotFoundError(args.parquet)
    args.output_root.mkdir(parents=True, exist_ok=True)

    commands = [
        command_for(args, backbone, variant)
        for backbone in args.backbones
        for variant in args.variants
    ]
    for index, command in enumerate(commands, start=1):
        print(f"[{index:02d}/{len(commands):02d}] {shlex.join(command)}", flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

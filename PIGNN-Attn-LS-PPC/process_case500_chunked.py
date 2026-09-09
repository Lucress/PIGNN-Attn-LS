"""
Process case500 OPFData one group at a time to stay within vault file-count quota.

Each of the 20 raw groups (~8K JSON files, ~23 GB) is converted to a single
PyG .pt file, then the source JSONs are deleted to free file-count slots.

Usage:
    python process_case500_chunked.py \
        --vault_root /home/vault/iwso/iwso230h/opfdata \
        --case_name  pglib_opf_case500_goc \
        --out_dir    /home/vault/iwso/iwso230h/opfdata/case500_processed \
        --delete_json          # remove source JSONs after each group is saved
        --n_groups 20          # default 20
"""

from __future__ import annotations
import argparse
import gc
import glob
import json
import os
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vault_root", required=True,
                   help="Root under which pglib_opf_case500_goc/raw/... lives")
    p.add_argument("--case_name", default="pglib_opf_case500_goc")
    p.add_argument("--out_dir", required=True,
                   help="Directory to write group_N.pt files")
    p.add_argument("--n_groups", type=int, default=20)
    p.add_argument("--delete_json", action="store_true",
                   help="Delete source JSON files after each group is saved")
    p.add_argument("--skip_existing", action="store_true", default=True,
                   help="Skip groups whose .pt already exists (resume)")
    return p.parse_args()


def find_json_root(vault_root: str, case_name: str) -> str:
    """Locate the directory that contains group_0/ … group_N/ folders."""
    candidates = [
        os.path.join(vault_root, case_name, "raw",
                     "gridopt-dataset-tmp", "dataset_release_1", case_name),
        os.path.join(vault_root, "dataset_release_1", case_name,
                     "raw", "gridopt-dataset-tmp", "dataset_release_1", case_name),
        os.path.join(vault_root, case_name),
    ]
    for c in candidates:
        if os.path.isdir(c) and any(
            d.startswith("group_") for d in os.listdir(c)
        ):
            return c
    raise FileNotFoundError(
        f"Cannot find group_N/ directories under {vault_root}/{case_name}. "
        f"Tried: {candidates}"
    )


def load_one_json(path: str):
    """Load a single OPFData JSON example and convert to a simple dict of tensors.

    We store only what the training pipeline needs:
      - bus features, branch features, admittance, labels
    rather than re-implementing the full PyG OPFDataset conversion.
    We store the raw dict so the existing opfdata_pipeline can still convert it.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


def process_group(group_dir: str, out_path: str, delete_json: bool):
    """Convert all JSON files in group_dir to a list of raw dicts and save as .pt."""
    json_files = sorted(glob.glob(os.path.join(group_dir, "*.json")))
    if not json_files:
        print(f"  [warn] no JSON files found in {group_dir}", flush=True)
        return 0

    print(f"  Loading {len(json_files)} JSON files from {group_dir} ...", flush=True)
    records = []
    for i, jf in enumerate(json_files):
        records.append(load_one_json(jf))
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(json_files)}", flush=True)

    print(f"  Saving {len(records)} records -> {out_path}", flush=True)
    torch.save(records, out_path)

    if delete_json:
        print(f"  Deleting {len(json_files)} JSON files ...", flush=True)
        for jf in json_files:
            os.remove(jf)
        # remove the now-empty group dir
        try:
            os.rmdir(group_dir)
            print(f"  Removed directory {group_dir}", flush=True)
        except OSError:
            pass  # not empty (e.g. other files) — leave it

    del records
    gc.collect()
    return len(json_files)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    json_root = find_json_root(args.vault_root, args.case_name)
    print(f"[chunked] JSON root: {json_root}", flush=True)
    print(f"[chunked] Output  : {args.out_dir}", flush=True)
    print(f"[chunked] Groups  : 0..{args.n_groups - 1}", flush=True)
    print(f"[chunked] Delete JSON after save: {args.delete_json}", flush=True)

    total_saved = 0
    for g in range(args.n_groups):
        group_dir = os.path.join(json_root, f"group_{g}")
        out_path  = os.path.join(args.out_dir, f"group_{g}.pt")

        if not os.path.isdir(group_dir):
            print(f"[group {g}] directory not found, skipping: {group_dir}", flush=True)
            continue

        if args.skip_existing and os.path.exists(out_path):
            print(f"[group {g}] {out_path} already exists — skipping", flush=True)
            continue

        print(f"\n[group {g}/{args.n_groups - 1}] processing ...", flush=True)
        n = process_group(group_dir, out_path, args.delete_json)
        total_saved += n
        print(f"[group {g}] done — {n} examples saved", flush=True)

    print(f"\n[chunked] All done. Total examples saved: {total_saved}", flush=True)
    print(f"[chunked] Output files in: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

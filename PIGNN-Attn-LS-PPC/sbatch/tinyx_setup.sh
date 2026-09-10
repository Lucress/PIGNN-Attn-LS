#!/bin/bash -l
# Run this ONCE on the tinyx LOGIN NODE (not as a job).
# The -l shebang makes it a login shell so `module` commands work.
#
#   ssh tinyx
#   bash -l ~/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/tinyx_setup.sh 2>&1 | tee ~/setup.log
#
# To run in background (survives SSH disconnect):
#   nohup bash -l ~/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/tinyx_setup.sh > ~/setup.log 2>&1 &
#   tail -f ~/setup.log   # watch progress

set -euo pipefail

BASE="/home/hpc/iwso/iwso230h/PIGNN-Attn-LS"
PPC="${BASE}/PIGNN-Attn-LS-PPC"
ENV="/home/hpc/iwso/iwso230h/conda-envs/pignn"
OPFDATA="${PPC}/opfdata"
PY="${ENV}/bin/python"
PIP="${ENV}/bin/pip"

echo "[1] Loading conda module..."
module load python/3.12-conda

echo "[2] Creating conda env (skipped if already exists)..."
if [ ! -d "${ENV}" ]; then
  conda create -y -p "${ENV}" python=3.12
  echo "    env created at ${ENV}"
else
  echo "    env already exists"
fi

echo "[3] Installing PyTorch 2.7.1 cu126..."
"${PIP}" install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 -q

echo "[4] Installing torch_geometric..."
"${PIP}" install torch_geometric -q

echo "[5] Installing torch_scatter + torch_sparse..."
"${PIP}" install torch_scatter torch_sparse \
  -f https://data.pyg.org/whl/torch-2.7.1+cu126.html -q

echo "[6] Installing scipy, matplotlib, pandapower, tqdm..."
"${PIP}" install scipy matplotlib pandapower tqdm -q

echo "[7] Verifying install..."
"${PY}" -c "import torch, torch_geometric, scipy; print('torch', torch.__version__, '| pyg', torch_geometric.__version__)"

echo "[8] Downloading OPFData: case14 + case118..."
mkdir -p "${OPFDATA}"
"${PY}" - <<'PY'
from torch_geometric.datasets import OPFDataset
import os, sys
root = "/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/opfdata"
for case in ["pglib_opf_case14_ieee", "pglib_opf_case118_ieee"]:
    for split in ("train", "val", "test"):
        ds = OPFDataset(root, case_name=case, split=split, num_groups=1)
        print(f"  {case}  {split}: {len(ds)} scenarios", flush=True)
print("Download complete.")
PY

echo ""
echo "=== SETUP COMPLETE ==="
echo "Submit jobs with:"
echo "  sbatch ${PPC}/sbatch/tinyx_pignn_case14.sh"
echo "  sbatch ${PPC}/sbatch/tinyx_pignn_case118.sh"

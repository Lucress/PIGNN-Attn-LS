#!/bin/bash -l
#SBATCH --job-name=proc_case500
#SBATCH --output=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/process_case500_chunked.out
#SBATCH --error=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/process_case500_chunked.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

BASE="/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC"
PY="/home/hpc/iwso/iwso230h/conda-envs/pignn/bin/python"
VAULT="/home/vault/iwso/iwso230h"

export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${BASE}/sbatch/Job_out"

echo "========================================================"
echo " case500 chunked processing — vault file-count rescue"
echo " $(date)"
echo "========================================================"

# ── Step 1: delete the redundant tar.gz shards ───────────────────────────────
# Raw JSONs are already extracted; tarballs are 24 GB of pure waste.
TARDIR="${VAULT}/opfdata/dataset_release_1/pglib_opf_case500_goc/raw/gridopt-dataset-tmp/dataset_release_1/pglib_opf_case500_goc"
echo ""
echo "--- Step 1: Removing redundant tar.gz shards ---"
ls "${VAULT}/opfdata/dataset_release_1/pglib_opf_case500_goc/raw/"*.tar.gz 2>/dev/null \
  | wc -l | xargs -I{} echo "  Found {} tar.gz files to remove"
rm -f "${VAULT}/opfdata/dataset_release_1/pglib_opf_case500_goc/raw/"*.tar.gz
echo "  Tarballs removed."

# ── Step 2: chunked JSON → .pt conversion (one group at a time) ──────────────
# Each group: ~8K JSON files (~23 GB) → 1 .pt file (~500 MB), then JSONs deleted.
# 160K JSON files will be replaced by 20 .pt files → file count drops ~140K.
OUT_DIR="${VAULT}/opfdata/case500_chunks"
mkdir -p "${OUT_DIR}"

echo ""
echo "--- Step 2: Processing 20 groups (JSON → .pt, delete JSONs after each) ---"
echo "  Output: ${OUT_DIR}"

srun "${PY}" -u "${BASE}/process_case500_chunked.py" \
  --vault_root "${VAULT}/opfdata/dataset_release_1" \
  --case_name  "pglib_opf_case500_goc" \
  --out_dir    "${OUT_DIR}" \
  --n_groups   20 \
  --delete_json \
  --skip_existing

echo ""
echo "--- Step 3: Final file count check ---"
echo "  Chunks written:"
ls -lh "${OUT_DIR}/"
echo ""
echo "  Remaining JSON files (should be 0):"
find "${VAULT}/opfdata/dataset_release_1/pglib_opf_case500_goc" -name "*.json" 2>/dev/null | wc -l

echo ""
echo "========================================================"
echo " Done. $(date)"
echo "========================================================"

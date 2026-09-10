#!/bin/bash -l
#SBATCH --job-name=pignn_sup_case500
#SBATCH --output=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pignn_supervised_case500.out
#SBATCH --error=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pignn_supervised_case500.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

BASE="/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC"
PY="/home/hpc/iwso/iwso230h/conda-envs/pignn/bin/python"
VAULT="/home/vault/iwso/iwso230h"
CHUNK_DIR="${VAULT}/opfdata/case500_chunks"
CKPT="${BASE}/results/ckpt/tinyx"

export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p "${BASE}/sbatch/Job_out" "${CKPT}"

cd "${BASE}"

echo "========================================================"
echo " PIGNN-Attn-LS supervised — case500 (chunked loader)"
echo " chunk_dir : ${CHUNK_DIR}"
echo " n_train_groups: 4  (60K examples)"
echo " K=30  d=4  d_hi=16  n_heads=4  BATCH=4  EPOCHS=120"
echo " $(date)"
echo "========================================================"

srun "${PY}" -u train_valid_test_pignn_opf.py \
  --chunk_dir     "${CHUNK_DIR}" \
  --n_train_groups 4 \
  --case_name     pglib_opf_case500_goc \
  --K 30 --d 4 --d_hi 16 --n_heads 4 \
  --BATCH 4 --EPOCHS 120 --LR 5e-5 \
  --mse_weight 1.0 --pinn_weight 0.0 --physics_weight 0.0 \
  --run_name pignn_supervised_case500 \
  --log_to_file --log_dir "${BASE}/results/logs/tinyx" \
  --ckpt_dir "${CKPT}"

echo ""
echo "========================================================"
echo " Done. $(date)"
echo "========================================================"

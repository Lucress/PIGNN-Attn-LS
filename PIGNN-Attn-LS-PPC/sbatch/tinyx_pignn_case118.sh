#!/bin/bash -l
#SBATCH --job-name=pignn_opf_case118
#SBATCH --output=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pignn_opf_case118.out
#SBATCH --error=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pignn_opf_case118.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

BASE="/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC"
PY="/home/hpc/iwso/iwso230h/conda-envs/pignn/bin/python"
OPFDATA="${BASE}/opfdata"

export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p "${BASE}/sbatch/Job_out"

cd "${BASE}"

# PINN run on case118 (118 buses, larger model)
srun "${PY}" -u train_valid_test_pignn_opf.py \
  --case_name pglib_opf_case118_ieee \
  --opfdata_root "${OPFDATA}" \
  --num_groups 1 \
  --K 30 --d 10 --d_hi 32 --n_heads 4 \
  --BATCH 4 --EPOCHS 60 --LR 1e-3 \
  --mse_weight 0.0 --pinn_weight 1.0 --physics_weight 1e-2 \
  --opf_limit_weight 1.0 --opf_band_weight 0.0 \
  --run_name pignn_pinn_case118 \
  --log_to_file --log_dir "${BASE}/results/logs/tinyx" \
  --ckpt_dir "${BASE}/results/ckpt/tinyx"

# Supervised run on case118
srun "${PY}" -u train_valid_test_pignn_opf.py \
  --case_name pglib_opf_case118_ieee \
  --opfdata_root "${OPFDATA}" \
  --num_groups 1 \
  --K 30 --d 4 --d_hi 16 --n_heads 4 \
  --BATCH 4 --EPOCHS 60 --LR 1e-3 \
  --mse_weight 1.0 --pinn_weight 0.0 --physics_weight 0.0 \
  --run_name pignn_supervised_case118 \
  --log_to_file --log_dir "${BASE}/results/logs/tinyx" \
  --ckpt_dir "${BASE}/results/ckpt/tinyx"

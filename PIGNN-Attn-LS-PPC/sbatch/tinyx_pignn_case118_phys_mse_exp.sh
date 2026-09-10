#!/bin/bash -l
#SBATCH --job-name=pignn_phys_mse
#SBATCH --output=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pignn_phys_mse_case118.out
#SBATCH --error=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pignn_phys_mse_case118.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

BASE="/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC"
PY="/home/hpc/iwso/iwso230h/conda-envs/pignn/bin/python"
OPFDATA="${BASE}/opfdata"
CKPT="${BASE}/results/ckpt/tinyx"

export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p "${BASE}/sbatch/Job_out" "${CKPT}"

cd "${BASE}"

# Physics-MSE balance experiment:
#   pinn_weight=1.0, mse_weight=5.0, 300 epochs cold start.
#
# Supervisor question: "up to 300 MSE works to visualize that optimization
# manifold — the decrease starts from which point?"
#
# Hypothesis: with pinn_weight≫0 the model first needs to learn a valid power
# flow (physics residual decrease) before the supervised MSE signal can take
# over.  The log will show BOTH loss curves per epoch so we can plot them and
# find the inflection point where the manifold "breaks in".
echo "========================================================"
echo " PIGNN physics-MSE balance experiment — case118"
echo " pinn_weight=1.0  mse_weight=5.0  EPOCHS=300 (cold start)"
echo " $(date)"
echo "========================================================"

srun "${PY}" -u train_valid_test_pignn_opf.py \
  --case_name pglib_opf_case118_ieee \
  --opfdata_root "${OPFDATA}" \
  --num_groups 1 \
  --K 30 --d 4 --d_hi 16 --n_heads 4 \
  --BATCH 4 --EPOCHS 300 --LR 1e-4 \
  --mse_weight 5.0 --pinn_weight 1.0 --physics_weight 0.0 \
  --run_name pignn_phys1_mse5_case118 \
  --log_to_file --log_dir "${BASE}/results/logs/tinyx" \
  --ckpt_dir "${CKPT}"

echo ""
echo "========================================================"
echo " Done. $(date)"
echo "========================================================"

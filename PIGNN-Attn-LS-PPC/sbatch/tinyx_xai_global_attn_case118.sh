#!/bin/bash -l
#SBATCH --job-name=xai_global_118
#SBATCH --output=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/xai_global_case118.out
#SBATCH --error=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/xai_global_case118.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

BASE="/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC"
PY="/home/hpc/iwso/iwso230h/conda-envs/pignn/bin/python"
OPFDATA="${BASE}/opfdata"
CKPT="${BASE}/results/ckpt/tinyx"
XAI="${BASE}/results/xai"

CKPT_118G="${CKPT}/pignn_global_case118_best.pt"
CASE="pglib_opf_case118_ieee"
K=30; D=4; D_HI=16; HEADS=4; N=30
LABEL="GlobalAttn case118 (K=30)"

ATTN="${XAI}/attention_case118_global"
DEEP="${XAI}/deep_case118_global"
HPHY="${XAI}/head_physics_case118_global"
GATTN="${XAI}/global_attn_case118"

export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p "${BASE}/sbatch/Job_out" "${ATTN}" "${DEEP}" "${HPHY}" "${GATTN}"

cd "${BASE}"

echo "========================================================"
echo " XAI pipeline — case118 global attention variant"
echo " checkpoint : ${CKPT_118G}"
echo " $(date)"
echo "========================================================"

# ── Step 1: local edge attention (existing pipeline) ─────────────────────────
echo ""
echo "--- Step 1/4: local edge-attention analysis ---"
srun "${PY}" -u analyze_attention_opfdata.py \
  --ckpt "${CKPT_118G}" \
  --opfdata_root "${OPFDATA}" \
  --case_name "${CASE}" \
  --K ${K} --d ${D} --d_hi ${D_HI} --n_heads ${HEADS} \
  --use_global_attn --global_attn_position after \
  --n_scenarios ${N} \
  --out_dir "${ATTN}"

echo ""
echo "--- Step 2/4: deep per-head local analysis ---"
srun "${PY}" -u project_deep_analysis.py \
  --ckpt "${CKPT_118G}" \
  --opfdata_root "${OPFDATA}" \
  --case_name "${CASE}" \
  --attn_dir "${ATTN}" \
  --out_dir "${DEEP}" \
  --K ${K} --d ${D} --d_hi ${D_HI} --n_heads ${HEADS} \
  --use_global_attn --global_attn_position after \
  --n_scenarios ${N} \
  --label "${LABEL}"

echo ""
echo "--- Step 3/4: head × physics heatmap ---"
srun "${PY}" -u project_head_physics.py \
  --ckpt "${CKPT_118G}" \
  --opfdata_root "${OPFDATA}" \
  --case_name "${CASE}" \
  --deep_dir "${DEEP}" \
  --out_dir "${HPHY}" \
  --K ${K} --d ${D} --d_hi ${D_HI} --n_heads ${HEADS} \
  --use_global_attn --global_attn_position after \
  --label "${LABEL}"

echo ""
echo "--- Step 4/4: global attention analysis (what does the model attend to?) ---"
srun "${PY}" -u project_global_attention.py \
  --ckpt "${CKPT_118G}" \
  --opfdata_root "${OPFDATA}" \
  --case_name "${CASE}" \
  --K ${K} --d ${D} --d_hi ${D_HI} --n_heads ${HEADS} \
  --global_attn_position after \
  --n_scenarios ${N} \
  --plot_steps 0 4 9 19 29 \
  --out_dir "${GATTN}"

echo ""
echo "========================================================"
echo " Done. Key outputs:"
echo "   ${GATTN}/global_attn_vs_dist.png  -- attn weight vs hop distance"
echo "   ${GATTN}/global_attn_entropy.png  -- head entropy per Newton step"
echo "   ${GATTN}/global_attn_heatmap_k*.png -- N×N heatmaps"
echo "   ${HPHY}/head_physics_heatmap.png  -- local head × physics"
echo " $(date)"
echo "========================================================"

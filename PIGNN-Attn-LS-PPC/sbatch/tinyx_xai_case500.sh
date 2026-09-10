#!/bin/bash -l
#SBATCH --job-name=xai_case500
#SBATCH --output=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/xai_case500.out
#SBATCH --error=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/xai_case500.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

BASE="/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC"
PY="/home/hpc/iwso/iwso230h/conda-envs/pignn/bin/python"
VAULT="/home/vault/iwso/iwso230h"
CHUNK_DIR="${VAULT}/opfdata/case500_chunks"
CKPT="${BASE}/results/ckpt/tinyx"
XAI="${BASE}/results/xai"

CKPT_500="${CKPT}/pignn_supervised_case500_best.pt"
CASE="pglib_opf_case500_goc"
K=30; D=4; D_HI=16; HEADS=4; N=50
LABEL="Supervised case500 (K=30)"

ATTN="${XAI}/attention_case500_supervised"
DEEP="${XAI}/deep_case500_supervised"
HPHY="${XAI}/head_physics_case500_supervised"

export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${BASE}/sbatch/Job_out" "${ATTN}" "${DEEP}" "${HPHY}"

cd "${BASE}"

echo "========================================================"
echo " XAI pipeline — case500 supervised"
echo " checkpoint : ${CKPT_500}"
echo " chunk_dir  : ${CHUNK_DIR}"
echo " n_scenarios: ${N}  K=${K}  d=${D}  d_hi=${D_HI}  heads=${HEADS}"
echo " $(date)"
echo "========================================================"

# ── Step 1: receptive field + attention weights ───────────────────────────────
echo ""
echo "--- Step 1/3: analyze_attention_opfdata (receptive field + attn_weights) ---"
srun "${PY}" -u analyze_attention_opfdata.py \
  --ckpt         "${CKPT_500}" \
  --chunk_dir    "${CHUNK_DIR}" \
  --case_name    "${CASE}" \
  --K ${K} --d ${D} --d_hi ${D_HI} --n_heads ${HEADS} \
  --n_scenarios  ${N} \
  --out_dir      "${ATTN}"

echo ""
echo "--- Step 2/3: project_deep_analysis (r(k) curves + per-head attn) ---"
srun "${PY}" -u project_deep_analysis.py \
  --ckpt         "${CKPT_500}" \
  --chunk_dir    "${CHUNK_DIR}" \
  --case_name    "${CASE}" \
  --attn_dir     "${ATTN}" \
  --out_dir      "${DEEP}" \
  --K ${K} --d ${D} --d_hi ${D_HI} --n_heads ${HEADS} \
  --n_scenarios  ${N} \
  --label        "${LABEL}"

echo ""
echo "--- Step 3/3: project_head_physics (head x AC-OPF quantity heatmap) ---"
srun "${PY}" -u project_head_physics.py \
  --ckpt         "${CKPT_500}" \
  --chunk_dir    "${CHUNK_DIR}" \
  --case_name    "${CASE}" \
  --deep_dir     "${DEEP}" \
  --out_dir      "${HPHY}" \
  --K ${K} --d ${D} --d_hi ${D_HI} --n_heads ${HEADS} \
  --label        "${LABEL}"

echo ""
echo "========================================================"
echo " Done. Outputs:"
echo "   ${ATTN}/receptive_field.json  -- graph diameter vs K"
echo "   ${ATTN}/attn_weights.pt       -- (N,K,E) mean-over-heads attention"
echo "   ${DEEP}/step_correlation.png  -- r(k) curves per head"
echo "   ${DEEP}/attn_perhead.pt       -- (N,K,E,H) per-head attention"
echo "   ${HPHY}/head_physics_heatmap.png  -- P/Q/Dth/DV x 4 heads"
echo "   ${HPHY}/head_physics_corr.json    -- Spearman r matrix"
echo " $(date)"
echo "========================================================"

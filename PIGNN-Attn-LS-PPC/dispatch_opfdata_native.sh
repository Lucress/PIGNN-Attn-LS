#!/usr/bin/env bash
set -euo pipefail

# Native-loss variant: each model trained with the loss from its own repo,
# evaluated with the same opf_task metrics as the shared-loss runs.
#
#   GridSFM -> gridsfm.loss.compute_loss           (faithful)
#   LUMINA  -> lumina.model.opf.losses             (faithful, needs lumina-sdk)
#   GridFM  -> gridfm_graphkit MaskedReconstructionMSE (partial: PBELoss unusable)
#
# See native_loss.py for exactly what is and is not faithful. Loss values are
# NOT comparable across models; only the evaluation metrics are.
#
# Cluster-agnostic: the defaults target alex, and helma is reached by
# overriding HOST/BASE/CK/PY/OPFDATA_ROOT/GRES/PARTITION.
#
#   # alex (A40)
#   CASE=both bash dispatch_opfdata_native.sh
#
#   # helma (H100) -- note GRES=gpu:1 and that h200 is account-gated
#   HOST=helma \
#     BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC \
#     CK=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS \
#     PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python \
#     OPFDATA_ROOT=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata \
#     GRES=gpu:1 CASE=case14 PARTITION=h100 bash dispatch_opfdata_native.sh

HOST="${HOST:-alex}"
BASE="${BASE:-/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC}"
CK="${CK:-/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS}"
PY="${PY:-/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python}"
OPFDATA_ROOT="${OPFDATA_ROOT:-/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/opfdata}"

CASE="${CASE:-both}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-40}"
PARTITION="${PARTITION:-a40}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
# Must come after PARTITION. helma uses plain "gpu:1" and partitions
# h100/preempt (h200 is account-gated); alex uses "gpu:a40:1".
GRES="${GRES:-gpu:${PARTITION}:1}"

retry() { local n=0; until "$@"; do n=$((n+1)); [ "$n" -ge 20 ] && return 1; sleep 30; done; }
STAGE="$(mktemp -d)"; trap 'rm -rf "${STAGE}"' EXIT

case "${CASE}" in
  case14)  CASES="pglib_opf_case14_ieee" ;;
  case118) CASES="pglib_opf_case118_ieee" ;;
  both)    CASES="pglib_opf_case14_ieee pglib_opf_case118_ieee" ;;
  *) echo "CASE must be case14|case118|both" >&2; exit 1 ;;
esac

ALL_LABELS=()
for case_name in ${CASES}; do
  short="${case_name#pglib_opf_}"; short="${short%_ieee}"
  GROUP="opfdata_native_${short}_${PARTITION}_${STAMP}"
  LOG_DIR="${BASE}/results/logs/${GROUP}"
  CKPT_DIR="${BASE}/results/ckpt/${GROUP}"
  SB_DIR="${BASE}/sbatch/${GROUP}"
  JOB_OUT="${BASE}/sbatch/Job_out"
  retry ssh "${HOST}" "mkdir -p '${SB_DIR}' '${LOG_DIR}' '${CKPT_DIR}' '${JOB_OUT}'"

  # model | init | batch | extra args
  SPECS=(
    "lumina|pretrained|4|--model_config ${CK}/LUMINA/checkpoints/lumina_config.json --pretrained_checkpoint ${CK}/LUMINA/checkpoints/lumina_model.safetensors"
    "lumina|scratch|4|--model_config ${CK}/LUMINA/checkpoints/lumina_config.json"
    "gridsfm|pretrained|4|--pretrained_checkpoint ${CK}/GridSFM/checkpoints/gridsfm_open_v1.1.pt"
    "gridsfm|scratch|4|"
    "gridfm|scratch|8|--hidden_size 128 --num_layers 6 --n_heads 8 --zero_init_head --feature_transform signed_log --native_physics_weight 1e-2"
  )

  LABELS=()
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r model init batch extra <<< "${spec}"
    lr=1e-4; [ "${model}" = "gridfm" ] && lr=5e-4
    label="opfdn_${short}_${model}_${init}_b${batch}"
    cat > "${STAGE}/${label}.sh" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${label}
#SBATCH --output=${JOB_OUT}/${label}.out
#SBATCH --error=${JOB_OUT}/${label}.err
#SBATCH --gres=${GRES}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail
export PYTHONPATH=${BASE}:\${PYTHONPATH:-}
export HF_HUB_DISABLE_XET=1
cd ${BASE}

srun ${PY} -u train_valid_test_opfdata.py \\
  --model ${model} --init_mode ${init} --loss native \\
  --case_name ${case_name} \\
  --opfdata_root ${OPFDATA_ROOT} --num_groups 1 \\
  --run_name ${label} \\
  --log_to_file --log_dir ${LOG_DIR} --ckpt_dir ${CKPT_DIR} \\
  --BATCH ${batch} --EPOCHS ${EPOCHS} --LR ${lr} \\
  --opf_limit_weight 1.0 --opf_band_weight 0.0 \\
  ${extra}
EOF
    LABELS+=("${label}")
  done

  retry scp -q "${STAGE}"/*.sh "${HOST}:${SB_DIR}/"
  list=""; for l in "${LABELS[@]}"; do list="${list} '${l}.sh'"; done
  echo "GROUP=${GROUP}"
  retry ssh "${HOST}" "cd '${SB_DIR}' && for s in${list}; do jid=\$(sbatch --parsable \"\$s\"); printf '  %s\t%s\n' \"\$jid\" \"\${s%.sh}\"; done"
  rm -f "${STAGE}"/*.sh
  ALL_LABELS+=("${LABELS[@]}")
done

echo "Submitted ${#ALL_LABELS[@]} native-loss jobs."

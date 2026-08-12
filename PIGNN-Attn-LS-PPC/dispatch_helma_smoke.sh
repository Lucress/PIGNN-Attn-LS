#!/usr/bin/env bash
set -euo pipefail

# Short verification runs on helma (account b313dc11, H100/H200).
#
# helma is a separate account from alex (iwi5295h), so although /home/hpc,
# /home/vault and /home/woody are mounted on both, alex's directories are not
# readable here. Everything -- environment, code, data -- is installed under
# b313dc11's own paths. See HELMA_HANDOFF.md.
#
# Covers both pipelines and all three models on each of h100 and h200, with
# tiny sample caps, so a failure shows up in minutes rather than at the end of
# a 24 h job.

HELMA_HOST="${HELMA_HOST:-helma}"
HOME_BASE="${HOME_BASE:-/home/hpc/b313dc/b313dc11}"
VAULT_BASE="${VAULT_BASE:-/home/vault/b313dc/b313dc11}"
BASE="${BASE:-${HOME_BASE}/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC}"
PY="${PY:-${HOME_BASE}/conda-envs/gridfm-py312/bin/python}"
# helma compute nodes mount ONLY /home/hpc -- /home/vault and /home/woody do
# not exist there. Data and checkpoints must therefore live under $HOME even
# though vault is far larger. Vault is still usable from the login node as a
# staging/archive area.
OPFDATA_ROOT="${OPFDATA_ROOT:-${HOME_BASE}/PIGNN-Attn-LS/opfdata}"
PARQUET="${PARQUET:-${HOME_BASE}/PIGNN-Attn-LS/data/case14_opf_task_ready.parquet}"
LUMINA_CKPT_DIR="${LUMINA_CKPT_DIR:-${HOME_BASE}/PIGNN-Attn-LS/LUMINA/checkpoints}"
GRIDSFM_CKPT="${GRIDSFM_CKPT:-${HOME_BASE}/PIGNN-Attn-LS/GridSFM/checkpoints/gridsfm_open_v1.1.pt}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GROUP="${GROUP:-helma_smoke_${STAMP}}"
JOB_OUT="${BASE}/sbatch/Job_out"
LOG_DIR="${BASE}/results/logs/${GROUP}"
CKPT_DIR="${BASE}/results/ckpt/${GROUP}"
SBATCH_DIR="${BASE}/sbatch/${GROUP}"

# h200 is restricted via AllowAccounts to a fixed list that does NOT include
# b313dc, so sbatch rejects it with "Invalid account or account/partition
# combination". Usable GPU partitions here are h100 (24 h) and preempt (48 h,
# preemptible). Override PARTITIONS if the entitlement ever changes.
PARTITIONS="${PARTITIONS:-h100 preempt}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
CPUS="${CPUS:-16}"
EPOCHS="${EPOCHS:-1}"
CAP_TRAIN="${CAP_TRAIN:-128}"
CAP_EVAL="${CAP_EVAL:-64}"

# The NHR login gateway (csnhr.nhr.fau.de) intermittently refuses connections,
# so every remote call goes through a retry wrapper and the job scripts are
# built locally and shipped in a single transfer.
retry() {
  local n=0 max=20
  until "$@"; do
    n=$((n + 1))
    if [ "${n}" -ge "${max}" ]; then
      echo "[dispatch] giving up after ${max} attempts: $*" >&2
      return 1
    fi
    echo "[dispatch] retry ${n}/${max}: $*" >&2
    sleep 30
  done
}

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

# emit_job <label> <partition> <body-file-content-via-stdin>
emit() {
  local label="$1" part="$2"
  cat <<EOF
#!/bin/bash -l
#SBATCH --job-name=${label}
#SBATCH --output=${JOB_OUT}/${label}.out
#SBATCH --error=${JOB_OUT}/${label}.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${part}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}

set -euo pipefail
export PYTHONPATH=${BASE}:\${PYTHONPATH:-}
export HF_HUB_DISABLE_XET=1
cd ${BASE}

echo "[gpu] \$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1)"
${PY} -c "import torch; print('[torch]', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0), 'cc', torch.cuda.get_device_capability(0))"
EOF
}

mk_opfdata_job() {
  local label="$1" part="$2" model="$3" init="$4" extra="$5"
  local f="${STAGE}/${label}.sh"
  { emit "${label}" "${part}"
    cat <<EOF

srun ${PY} -u train_valid_test_opfdata.py \\
  --model ${model} --init_mode ${init} \\
  --case_name pglib_opf_case14_ieee \\
  --opfdata_root ${OPFDATA_ROOT} --num_groups 1 \\
  --run_name ${label} \\
  --log_to_file --log_dir ${LOG_DIR} --ckpt_dir ${CKPT_DIR} \\
  --BATCH 4 --EPOCHS ${EPOCHS} --LR 1e-4 \\
  --max_train_samples ${CAP_TRAIN} --max_valid_samples ${CAP_EVAL} --max_test_samples ${CAP_EVAL} \\
  --hidden_size 24 --num_layers 3 --n_heads 4 --zero_init_head \\
  ${extra}
EOF
  } > "${f}"
}

mk_parquet_job() {
  local label="$1" part="$2" script="$3" extra="$4"
  local f="${STAGE}/${label}.sh"
  { emit "${label}" "${part}"
    cat <<EOF

srun ${PY} -u ${script} \\
  --PARQUET ${PARQUET} --task opf \\
  --run_name ${label} \\
  --log_to_file --log_dir ${LOG_DIR} --ckpt_dir ${CKPT_DIR} \\
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 2 \\
  --dataset_complex_dtype complex128 \\
  --BATCH 4 --EPOCHS ${EPOCHS} --LR 1e-4 \\
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \\
  --max_train_samples ${CAP_TRAIN} --max_valid_samples ${CAP_EVAL} --max_test_samples ${CAP_EVAL} \\
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \\
  ${extra}
EOF
  } > "${f}"
}

LABELS=()
for part in ${PARTITIONS}; do
  # OPFData pipeline: LUMINA and GridSFM native, GridFM adapted
  mk_opfdata_job "smoke_${part}_opfd_lumina"  "${part}" lumina  pretrained \
    "--model_config ${LUMINA_CKPT_DIR}/lumina_config.json --pretrained_checkpoint ${LUMINA_CKPT_DIR}/lumina_model.safetensors"
  mk_opfdata_job "smoke_${part}_opfd_gridsfm" "${part}" gridsfm pretrained \
    "--pretrained_checkpoint ${GRIDSFM_CKPT}"
  mk_opfdata_job "smoke_${part}_opfd_gridfm"  "${part}" gridfm  scratch ""
  # pandapower parquet pipeline
  mk_parquet_job "smoke_${part}_pq_lumina" "${part}" train_valid_test_lumina.py \
    "--model_config ${LUMINA_CKPT_DIR}/lumina_config.json --init_mode scratch --treat_voltage_mismatch_as_transformer"
  mk_parquet_job "smoke_${part}_pq_gridfm" "${part}" train_valid_test_gridfm.py \
    "--hidden_size 24 --num_layers 3 --n_heads 4 --zero_init_head --vn_feature_mode log --feature_transform signed_log"
  LABELS+=("smoke_${part}_opfd_lumina" "smoke_${part}_opfd_gridsfm" "smoke_${part}_opfd_gridfm" \
           "smoke_${part}_pq_lumina" "smoke_${part}_pq_gridfm")
done

retry ssh "${HELMA_HOST}" "mkdir -p '${SBATCH_DIR}' '${LOG_DIR}' '${JOB_OUT}' '${CKPT_DIR}'"
retry scp -q "${STAGE}"/*.sh "${HELMA_HOST}:${SBATCH_DIR}/"

list=""
for l in "${LABELS[@]}"; do list="${list} '${l}.sh'"; done
echo "GROUP=${GROUP}"
echo "Submitted:"
retry ssh "${HELMA_HOST}" "cd '${SBATCH_DIR}' && for s in${list}; do jid=\$(sbatch --parsable \"\$s\"); printf '%s\t%s\n' \"\$jid\" \"\${s%.sh}\"; done"
echo "Logs: ${LOG_DIR}"
echo "Job stdout/err: ${JOB_OUT}"

#!/usr/bin/env bash
set -euo pipefail

# Re-score saved PF checkpoints with MAE and the regression diagnostics.
#
# Inference only, no retraining: reads the same parquet and the same test split
# as the training run, so the RMSE it reports must reproduce the run's final
# RMSE. That agreement is the check that split and checkpoint line up -- it was
# verified on case118 (0.011436366 against a logged 1.1436e-02) before this was
# run across the corpus.
#
# Staging follows the training dispatcher: helma compute nodes see only
# /home/hpc, the corpus does not fit under the soft quota, so each job copies
# its parquet to node-local disk and then deletes the $HOME copy.
#
#   GROUP=pf_gridfm_h100_20260810_221252 IMPL=mirror bash dispatch_rescore_pf.sh
#   GRIDS="case118 case300" bash dispatch_rescore_pf.sh

HOST="${HOST:-helma}"
HOME_BASE="${HOME_BASE:-/home/hpc/b313dc/b313dc11}"
BASE="${BASE:-${HOME_BASE}/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC}"
PY="${PY:-${HOME_BASE}/conda-envs/gridfm-py312/bin/python}"
DATA="${DATA:-${HOME_BASE}/PIGNN-Attn-LS/data/pf}"
VAULT="${VAULT:-/home/vault/b313dc/b313dc11/data/pf}"

# Which training group's checkpoints to score, and which implementation built
# them -- the model has to be reconstructed with the same architecture.
GROUP="${GROUP:-pf_gridfm_h100_20260810_221252}"
IMPL="${IMPL:-mirror}"
CKPT_DIR="${CKPT_DIR:-${BASE}/results/ckpt/${GROUP}}"
OUT_DIR="${OUT_DIR:-${BASE}/results/rescore/${GROUP}}"

PARTITION="${PARTITION:-h100}"
GRES="${GRES:-gpu:1}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
CPUS="${CPUS:-16}"
HIDDEN="${HIDDEN:-48}"; LAYERS="${LAYERS:-12}"; HEADS="${HEADS:-8}"
# This is a ceiling on TOTAL staged bytes, including whatever a concurrent
# training campaign is holding -- so it has to sit above that, or nothing ever
# starts. Rescore jobs delete their staged copy in their first seconds, so the
# marginal footprint is one file at a time and transient.
BUDGET_GB="${BUDGET_GB:-72}"

retry() { local n=0; until "$@"; do n=$((n+1)); [ "$n" -ge 20 ] && return 1; sleep 30; done; }
STAGE="$(mktemp -d)"; trap 'rm -rf "${STAGE}"' EXIT

# Reuse the training dispatcher's manifest verbatim: same grids, same files,
# same measured bus counts. Duplicating it here would let the two drift.
DISPATCH_SRC="${DISPATCH_SRC:-$(dirname "$0")/dispatch_pf_gridfm.sh}"
MANIFEST=()
while IFS= read -r _line; do
  [ -n "${_line}" ] && MANIFEST+=("${_line}")
done < <(sed -n '/^MANIFEST=(/,/^)/p' "${DISPATCH_SRC}" \
         | grep '^  "' | sed 's/^  "//; s/".*$//')
[ "${#MANIFEST[@]}" -gt 0 ] || { echo "could not read manifest from ${DISPATCH_SRC}" >&2; exit 1; }

lookup() { local w="$1"; for e in "${MANIFEST[@]}"; do
  [ "${e%%|*}" = "${w}" ] && { echo "${e#*|}"; return; }; done; }

GRIDS="${GRIDS:-all}"
if [ "${GRIDS}" = "all" ]; then
  SELECTED=(); for e in "${MANIFEST[@]}"; do SELECTED+=("${e%%|*}"); done
else
  read -r -a SELECTED <<< "${GRIDS}"
fi

retry ssh "${HOST}" "mkdir -p '${OUT_DIR}' '${BASE}/sbatch/Job_out' '${DATA}'"

staged_bytes() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 "${HOST}" \
    "du -sb '${DATA}' 2>/dev/null | cut -f1" 2>/dev/null || true
}

n=0
for g in "${SELECTED[@]}"; do
  info="$(lookup "${g}")" || true
  [ -z "${info}" ] && { echo "unknown grid ${g}, skipped" >&2; continue; }
  buses="${info%%|*}"; rest="${info#*|}"; file="${rest%|*}"; nbytes="${info##*|}"

  # The checkpoint name carries the batch the run used; find it rather than
  # recomputing, since the batch rule has changed over the campaign.
  ck="$(retry ssh "${HOST}" "ls '${CKPT_DIR}' 2>/dev/null | grep -E '_${g}_b[0-9]+_best\\.pt$' | head -1")"
  if [ -z "${ck}" ]; then echo "no checkpoint for ${g}, skipped" >&2; continue; fi

  budget=$(( BUDGET_GB * 1073741824 ))
  while true; do
    used="$(staged_bytes)"
    [ -z "${used}" ] && { sleep 30; continue; }
    [ $(( used + nbytes )) -le "${budget}" ] && break
    echo "   waiting for room: $(( used / 1073741824 )) GB staged, ${g} needs $(( nbytes / 1073741824 )) GB"
    sleep 120
  done

  echo "   staging ${g} ..."
  retry ssh "${HOST}" "
    set -e
    if [ ! -s '${DATA}/${file}' ] || \
       [ \"\$(stat -c %s '${DATA}/${file}')\" != \"\$(stat -c %s '${VAULT}/${file}')\" ]; then
      cp '${VAULT}/${file}' '${DATA}/${file}.part' && mv '${DATA}/${file}.part' '${DATA}/${file}'
    fi"

  label="rescore_${IMPL}_${g}"
  cat > "${STAGE}/${label}.sh" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${label}
#SBATCH --output=${BASE}/sbatch/Job_out/${label}.out
#SBATCH --error=${BASE}/sbatch/Job_out/${label}.err
#SBATCH --gres=${GRES}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}

set -euo pipefail
export PYTHONPATH=${BASE}:\${PYTHONPATH:-}
cd ${BASE}
# Deliberately does NOT delete the $HOME copy: a training job for the same grid
# may still be PENDING and needs that exact file, and removing it would fail
# that job at startup. The training jobs clean up after themselves.
cp "${DATA}/${file}" "\${TMPDIR}/p.parquet"

srun ${PY} -u rescore_pf.py \\
  --PARQUET "\${TMPDIR}/p.parquet" \\
  --ckpt "${CKPT_DIR}/${ck}" \\
  --grid ${g} --impl ${IMPL} --BATCH 32 \\
  --hidden_size ${HIDDEN} --num_layers ${LAYERS} --n_heads ${HEADS} \\
  --json_out "${OUT_DIR}/${g}.json"
EOF
  retry scp -q "${STAGE}/${label}.sh" "${HOST}:${BASE}/sbatch/"
  jid="$(retry ssh "${HOST}" "cd '${BASE}/sbatch' && sbatch --parsable '${label}.sh'")"
  echo "   ${jid}  ${label}"
  n=$((n+1))
done
echo "Submitted ${n} rescore jobs -> ${OUT_DIR}"

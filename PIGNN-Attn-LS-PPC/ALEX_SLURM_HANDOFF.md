# Alex Slurm Handoff

This note documents how to access the Alex cluster, submit Slurm jobs from this repo, and inspect results. It is intended for another coding agent working from:

```bash
/Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
```

## 1. Local And Remote Paths

Local repo:

```bash
/Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
```

Alex repo:

```bash
/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
```

Alex generated Slurm scripts:

```bash
/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/<run_group>_<stamp>
```

Alex Slurm stdout/stderr:

```bash
/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out
```

Alex training logs:

```bash
/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/<run_group>_<stamp>
```

Large parquet datasets are usually under:

```bash
/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out
```

## 2. SSH Access

Normal access is:

```bash
ssh alex
```

If local DNS/proxy resolution fails, use a temporary SSH config that pins the CSNHR proxy host by IP.

Create `/private/tmp/alex_via_csnhr_ip_ssh_config`:

```sshconfig
Host csnhr-ip
  HostName 131.188.202.2
  User iwi5295h
  IdentityFile ~/.ssh/id_ed25519_nhr_fau
  IdentitiesOnly yes
  HostKeyAlias csnhr.nhr.fau.de
  StrictHostKeyChecking yes

Host alex-ip
  HostName alex.nhr.fau.de
  User iwi5295h
  IdentityFile ~/.ssh/id_ed25519_nhr_fau
  IdentitiesOnly yes
  ProxyJump csnhr-ip
  StrictHostKeyChecking yes
```

Then run commands through:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "hostname"
```

When using dispatch scripts, pass:

```bash
ALEX_HOST=alex-ip SSH_CONFIG_FILE=/private/tmp/alex_via_csnhr_ip_ssh_config
```

## 3. Sync Code To Alex

If local code changed, upload the changed files before dispatching jobs:

```bash
scp train_valid_test.py alex:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/train_valid_test.py
scp GNSMsg_SelfAttention_armijo.py alex:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/GNSMsg_SelfAttention_armijo.py
scp GNSMsg_SelfAttention_armijo_khop.py alex:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/GNSMsg_SelfAttention_armijo_khop.py
scp Dataset_optimized_complex_columns.py alex:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/Dataset_optimized_complex_columns.py
scp collate_blockdiag_optimized_complex_columns.py alex:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/collate_blockdiag_optimized_complex_columns.py
```

With the temporary SSH config:

```bash
scp -F /private/tmp/alex_via_csnhr_ip_ssh_config train_valid_test.py alex-ip:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/train_valid_test.py
```

## 4. Existing Dispatch Scripts

Important local dispatch scripts:

```bash
dispatch_case1354_1888_capacity_armijo_alex.sh
dispatch_lvn_case300_informed_alex.sh
dispatch_lvn_manual_flat_loss_ablation_alex.sh
dispatch_lvn_manual_flat_khop_ablation_alex.sh
dispatch_case300_top3_armijo_compare_alex.sh
dispatch_case118_145_loss_ablation_alex.sh
```

These scripts usually:

1. Build a timestamped remote sbatch directory.
2. Build a timestamped remote log directory.
3. SSH to Alex.
4. Write one sbatch file per experiment on Alex.
5. Submit each sbatch file using `sbatch`.
6. Save a local manifest TSV under `./results/logs/<run_group>_dispatch_<stamp>/manifest.tsv`.

## 4.1 Alex OS And Constraint Policy

Alex has fully transitioned to AlmaLinux 9.  Do not submit jobs with an
AlmaLinux 8 constraint:

```bash
# Do not use anymore:
#SBATCH --constraint=a100_40&el8
#SBATCH --constraint=el8
```

Jobs submitted with `-C el8` can remain pending indefinitely or be cancelled
by the cluster administrators.  Prefer no OS constraint, or constrain only
on GPU memory/features when needed:

```bash
SBATCH_CONSTRAINT=
SBATCH_CONSTRAINT=a100_40
SBATCH_CONSTRAINT=a100_80
```

If jobs are stuck pending, inspect current features and node states:

```bash
ssh alex "sinfo -p a100 -o '%P|%a|%l|%D|%t|%G|%f|%E'"
```

## 4.2 Filesystem And `$TMPDIR` Policy

To avoid heavy NFS/shared-filesystem use, generated Slurm jobs should stage
large parquet inputs to node-local storage and use that local copy during
training:

```bash
LOCAL_PARQUET="${TMPDIR}/input_${SLURM_JOB_ID}.parquet"
cp "${PARQUET}" "${LOCAL_PARQUET}"
```

Then pass:

```bash
--PARQUET "${LOCAL_PARQUET}"
```

Checkpoint outputs should be written to `$TMPDIR` and copied back at job
exit.  Use a trap so checkpoints are preserved on failure or timeout:

```bash
FINAL_CKPT_DIR=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/<run_group>
LOCAL_CKPT_DIR="${TMPDIR}/ckpt_${SLURM_JOB_ID}"
mkdir -p "${LOCAL_CKPT_DIR}"

copy_back_outputs() {
  status=$?
  mkdir -p "${FINAL_CKPT_DIR}"
  cp -a "${LOCAL_CKPT_DIR}/." "${FINAL_CKPT_DIR}/" 2>/dev/null || true
  exit ${status}
}
trap copy_back_outputs EXIT
```

Training logs are intentionally written directly to the final log directory
instead of `$TMPDIR`, because they are small and need to be live-tail-able
during training:

```bash
--log_to_file --log_dir "${LOG_DIR}"
```

This is the current compromise recommended for profiling/access requests:
stage large reads and checkpoints through `$TMPDIR`, but keep logs directly
visible.

## 5. Example: Dispatch Case1354/Case1888 Capacity Runs

Dry run first:

```bash
DRY_RUN=1 \
RUN_GROUP=case1354_1888_capacity_armijo_test \
LAZY_PARQUET=1 \
ROW_GROUP_CACHE_SIZE=1 \
BATCH=1 \
EPOCHS=1 \
ALEX_HOST=alex-ip \
SSH_CONFIG_FILE=/private/tmp/alex_via_csnhr_ip_ssh_config \
bash ./dispatch_case1354_1888_capacity_armijo_alex.sh
```

Real submission:

```bash
RUN_GROUP=case1354_1888_capacity_armijo_lazy \
LAZY_PARQUET=1 \
ROW_GROUP_CACHE_SIZE=1 \
BATCH=1 \
EPOCHS=100 \
ALEX_HOST=alex-ip \
SSH_CONFIG_FILE=/private/tmp/alex_via_csnhr_ip_ssh_config \
bash ./dispatch_case1354_1888_capacity_armijo_alex.sh
```

Useful environment overrides:

```bash
START_INDEX=5
STAMP=20260521_120000
BATCH=4
EPOCHS=100
TIME_LIMIT=24:00:00
PARTITION=a100
GPU_TYPE=a100
SBATCH_CONSTRAINT=a100_80
CPUS_PER_TASK=16
LAZY_PARQUET=1
ROW_GROUP_CACHE_SIZE=1
CASE1354=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/case1354pegase_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
CASE1888=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/case1888rte_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
```

`START_INDEX` resumes from a later experiment in the dispatch list.

## 6. Manual Sbatch Template

When a one-off focused run is needed, create a remote sbatch script with this shape:

```bash
#!/bin/bash -l
#SBATCH --job-name=case1354_debug
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/case1354_debug.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/case1354_debug.err
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100

set -euo pipefail

BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
LOG_DIR=${BASE}/results/logs/case1354_debug_$(date +%Y%m%d_%H%M%S)
CKPT_DIR=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/case1354_debug_$(date +%Y%m%d_%H%M%S)
PARQUET=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/case1354pegase_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
LOCAL_PARQUET=${TMPDIR}/input_${SLURM_JOB_ID}.parquet
LOCAL_CKPT_DIR=${TMPDIR}/ckpt_${SLURM_JOB_ID}

export PYTHONPATH=${BASE}:${PYTHONPATH:-}

module load python || true

cd "${BASE}"
mkdir -p "${LOG_DIR}" "${LOCAL_CKPT_DIR}"

copy_back_outputs() {
  status=$?
  mkdir -p "${CKPT_DIR}"
  cp -a "${LOCAL_CKPT_DIR}/." "${CKPT_DIR}/" 2>/dev/null || true
  exit ${status}
}
trap copy_back_outputs EXIT

echo "[stage] copying parquet to ${LOCAL_PARQUET}"
cp "${PARQUET}" "${LOCAL_PARQUET}"

srun python train_valid_test.py \
  --d=4 --d_hi=24 --n_heads=8 --num_attn_layers=4 --K=40 \
  --EPOCHS=100 --BATCH=1 --seed_value=42 \
  --train_ratio=0.3333 --valid_ratio=0.3333 \
  --lr_scheduler=CosineAnnealingLR \
  --PARQUET "${LOCAL_PARQUET}" \
  --vlimit --model GNSMsg_EdgeSelfAttn \
  --lazy_parquet --row_group_cache_size=1 \
  --no_cache_dense_ybus \
  --log_to_file --log_dir "${LOG_DIR}" \
  --ckpt_dir "${LOCAL_CKPT_DIR}"
```

Submit manually:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "cd /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/<dir> && sbatch <script>.sh"
```

## 7. Check Job State

Current user jobs:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "squeue -u \$USER -o '%i %.32j %.10T %.12M %.20R'"
```

Specific job:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "squeue -j <job_id> -o '%i %j %T %M %R'"
```

Accounting after or during run:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "sacct -j <job_id> --format=JobID,JobName%32,State,Elapsed,MaxRSS,ReqMem,NodeList%16 -P"
```

Live resource usage:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "sstat -j <job_id>.batch --format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize -P"
```

Live GPU usage on the allocated node:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "srun --jobid=<job_id> --overlap nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits"
```

Live Python process:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "srun --jobid=<job_id> --overlap bash -lc 'ps -o pid,etime,pcpu,pmem,rss,vsz,cmd -C python | head'"
```

## 8. Check Logs And Metrics

Slurm stdout/stderr:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "tail -n 80 /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/<job_name>.out"
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "tail -n 80 /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/<job_name>.err"
```

Training logs:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "ls -lh /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/<run_group>_<stamp>"
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "tail -n 120 /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/<run_group>_<stamp>/*training_log.txt"
```

Extract epoch/test lines:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "grep -E 'Epoch|Test physics-loss|Test residual|Initial metrics|Dataset sizes|Total number of parameters' /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/<run_group>_<stamp>/*training_log.txt"
```

Look for failures:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "grep -RniE 'error|exception|traceback|out_of_memory|oom|cuda|nan|physics loss detached' /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/<run_group>_<stamp> /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out"
```

## 9. Download Results

Download a full result log directory:

```bash
scp -r alex:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/<run_group>_<stamp> ./results/logs/
```

With temporary SSH config:

```bash
scp -F /private/tmp/alex_via_csnhr_ip_ssh_config -r alex-ip:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/<run_group>_<stamp> ./results/logs/
```

Download Slurm outputs:

```bash
scp -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/<job_name>.out ./results/logs/
scp -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/<job_name>.err ./results/logs/
```

## 10. Cancel Jobs

Cancel one job:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "scancel <job_id>"
```

Cancel jobs by name pattern carefully:

```bash
ssh -F /private/tmp/alex_via_csnhr_ip_ssh_config alex-ip "squeue -u \$USER -h -o '%i %j' | awk '/case1354|case1888/ {print \$1}' | xargs -r scancel"
```

Only cancel jobs when the user explicitly asks or when continuing would waste allocation time and the user has approved.

## 11. Memory Notes From Recent Runs

For large cases such as `case1354`, `case1888`, `case2848`, and `case2869`, the main failure mode has been CPU/system RAM OOM, not GPU OOM.

Observed patterns:

```text
lazy_parquet + dense cached Ybus:
  Can still use very high CPU RAM.
  Example: case1354 subset with BATCH=8, 3000 train, 100 valid, 100 test used about 94.8 GB RSS before first epoch.

lazy_parquet + --no_cache_dense_ybus:
  Much lower RAM.
  Often much slower because Ybus is reconstructed during forward/evaluation.

row_group_cache_size:
  Number of parquet row groups kept in the lazy loader cache.
  Smaller values reduce CPU RAM.
  For BATCH=1-4, use 1 or 2 first.
```

Recommended diagnostic strategy for very large grids:

```text
1. Use --lazy_parquet.
2. Start with --no_cache_dense_ybus.
3. Use BATCH=1.
4. Use small caps first, for example --max_train_samples=100 --max_valid_samples=20 --max_test_samples=20.
5. Use --skip_initial_eval to avoid expensive full train/valid evaluation before epoch 1.
6. Confirm one epoch finishes before scaling samples or batch size.
```

Useful `train_valid_test.py` options added for diagnostics:

```bash
--lazy_parquet
--row_group_cache_size=1
--no_cache_dense_ybus
--max_train_samples=3000
--max_valid_samples=100
--max_test_samples=100
--skip_initial_eval
```

## 12. Interpreting Results

Training logs report:

```text
Epoch ... train loss ... valid loss ...
Test physics-loss ...
|V| RMSE
theta RMSE
max |Delta P| over PV + PQ
max |Delta Q| over PQ
mean/median/p95/p99/RMSE residual metrics if enabled in current code
```

Good signs:

```text
Epoch lines appear regularly.
Validation loss decreases.
theta RMSE decreases instead of drifting upward.
Raw residual metrics decrease.
GPU has nonzero utilization during training.
```

Bad signs:

```text
No Epoch 0 line after hours.
Slurm OUT_OF_MEMORY state.
High CPU RAM with tiny GPU memory.
GPU utilization near 0% for a long time.
Repeated "physics loss detached" with Armijo.
NaN/Inf residuals or losses.
```

For large grids, do not assume poor accuracy is model-capacity only. First verify:

```text
The parquet data is consistent.
The reconstructed Ybus matches the reference formulation.
The Newton solution has small residual under the same model-side residual checker.
The loader reaches batches without CPU RAM blowup.
```

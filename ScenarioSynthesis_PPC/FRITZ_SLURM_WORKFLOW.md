# Fritz Slurm Workflow For ScenarioSynthesis_PPC

This note is a practical handoff for agents working from the local macOS project:

```text
/Users/changhunkim/PycharmProjects/PIGNN-Attn-LS
```

The target cluster host alias is:

```bash
ssh fritz
```

The remote project root is:

```text
/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS
```

The ScenarioSynthesis working directory on fritz is:

```text
/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC
```

Large generated parquet outputs should be written to vault, not `/home/hpc`:

```text
/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out
```

## 1. Sync Code To Fritz

Use `scp` from the local repo root. For LVN/manual-flat generation, these files are usually enough:

```bash
scp \
  ScenarioSynthesis_PPC/case_generator_all_test_cases_pandapower_consider_ppc_branch_row.py \
  ScenarioSynthesis_PPC/case_generator_lvn_snapshot_envelope.py \
  ScenarioSynthesis_PPC/main_datagen_lvn_snapshot_envelope.py \
  ScenarioSynthesis_PPC/merge_chunked_parquets.py \
  fritz:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/
```

Sync Slurm scripts:

```bash
scp ScenarioSynthesis_PPC/sbatch/*.sh \
  fritz:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/sbatch/
```

Sync PIGNN-side dataloader/model/checker changes if training or residual checking is needed:

```bash
scp \
  PIGNN-Attn-LS-PPC/Dataset_optimized_complex_columns.py \
  PIGNN-Attn-LS-PPC/check_newton_residual.py \
  PIGNN-Attn-LS-PPC/train_valid_test.py \
  PIGNN-Attn-LS-PPC/GNSMsg_armijo.py \
  PIGNN-Attn-LS-PPC/GNSMsg_SelfAttention_armijo.py \
  fritz:/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/
```

## 2. Compile Check On Fritz

After syncing Python files, run:

```bash
ssh fritz 'cd /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS && \
/home/woody/iwi5/iwi5295h/conda-envs/Python3.12_PandaPower3.4.0/bin/python -m py_compile \
ScenarioSynthesis_PPC/case_generator_all_test_cases_pandapower_consider_ppc_branch_row.py \
ScenarioSynthesis_PPC/case_generator_lvn_snapshot_envelope.py \
ScenarioSynthesis_PPC/main_datagen_lvn_snapshot_envelope.py \
PIGNN-Attn-LS-PPC/Dataset_optimized_complex_columns.py \
PIGNN-Attn-LS-PPC/check_newton_residual.py \
PIGNN-Attn-LS-PPC/train_valid_test.py \
PIGNN-Attn-LS-PPC/GNSMsg_armijo.py \
PIGNN-Attn-LS-PPC/GNSMsg_SelfAttention_armijo.py'
```

No output means the compile check passed.

## 3. Submit Slurm Jobs

Slurm scripts live here on fritz:

```text
/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/sbatch
```

Submit from that directory:

```bash
ssh fritz 'cd /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/sbatch && \
mkdir -p Job_out && \
sbatch LVN_snapshot_envelope_manual_flat_array18.sh'
```

For array generation plus dependent merge:

```bash
ssh fritz 'cd /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/sbatch && \
mkdir -p Job_out && \
jid=$(sbatch --parsable LVN_snapshot_envelope_manual_flat_array18.sh) && \
echo ARRAY_JOB_ID=$jid && \
mid=$(sbatch --parsable --dependency=afterok:$jid LVN_snapshot_envelope_manual_flat_merge.sh) && \
echo MERGE_JOB_ID=$mid && \
squeue -j $jid,$mid'
```

Example output:

```text
ARRAY_JOB_ID=3554006
MERGE_JOB_ID=3554007
```

## 4. Check Slurm Status

Use `squeue` for pending/running jobs:

```bash
ssh fritz 'squeue -j <JOBID>'
```

Use `sacct` for completed jobs:

```bash
ssh fritz 'sacct -j <JOBID> --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS,NodeList -P'
```

For array plus merge:

```bash
ssh fritz 'sacct -j 3554006,3554007 --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS -P'
```

Successful jobs should show:

```text
State=COMPLETED
ExitCode=0:0
```

## 5. Check Slurm Logs

Logs are under:

```text
/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/sbatch/Job_out
```

List logs:

```bash
ssh fritz 'cd /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/sbatch && \
ls -lh Job_out/LVN_manFlat_*'
```

Tail an array task:

```bash
ssh fritz 'cd /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/sbatch && \
tail -n 80 Job_out/LVN_manFlat_<ARRAYJOBID>_1.out'
```

Look for lines like:

```text
[INFO] 2000/2000 rows written
[DONE] Wrote 2000 rows
722- Bus...|converged successfully (misinf)|
```

For failed jobs, inspect both `.out` and `.err`:

```bash
ssh fritz 'cd /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/sbatch && \
tail -n 120 Job_out/<LOG>.out && \
tail -n 120 Job_out/<LOG>.err'
```

## 6. Verify Output Files On Vault

Vault output root:

```text
/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out
```

Count LVN manual-flat chunks:

```bash
ssh fritz 'printf "chunks="; \
ls -d /home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/LVN_snapshot_envelope_manual_flat_chunk_* 2>/dev/null | wc -l'
```

Expected:

```text
chunks=18
```

Check merged parquet:

```bash
ssh fritz 'ls -lh /home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/LVN_snapshot_envelope_manual_flat_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet'
```

Check total size:

```bash
ssh fritz 'du -ch \
/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/LVN_snapshot_envelope_manual_flat_chunk_* \
/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/LVN_snapshot_envelope_manual_flat_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet \
2>/dev/null | tail -n 1'
```

## 7. Verify Parquet Row Counts

Use cluster Python with `pyarrow`:

```bash
ssh fritz '/home/woody/iwi5/iwi5295h/conda-envs/Python3.12_PandaPower3.4.0/bin/python - << "PY"
import pyarrow.parquet as pq

p = "/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/LVN_snapshot_envelope_manual_flat_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet"
pf = pq.ParquetFile(p)
print("rows", pf.metadata.num_rows)
print("row_groups", pf.num_row_groups)
print("columns", len(pf.schema.names))
print("has_Y_matrix", "Y_matrix" in pf.schema.names)
PY'
```

Expected for full LVN manual-flat:

```text
rows 36000
has_Y_matrix False
```

## 8. Download Outputs Locally

Download a merged parquet:

```bash
scp fritz:/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/LVN_snapshot_envelope_manual_flat_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet \
  /Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/
```

Download chunks plus merged parquet:

```bash
scp -r \
  fritz:/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/LVN_snapshot_envelope_manual_flat_chunk_\* \
  fritz:/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/LVN_snapshot_envelope_manual_flat_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet \
  /Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/
```

## 9. Upload Local Parquets To Alex

Create remote out dir:

```bash
ssh alex 'mkdir -p ~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out'
```

Upload one parquet:

```bash
scp /Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/case1354pegase_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet \
  alex:~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/
```

Verify:

```bash
ssh alex 'ls -lh ~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/case1354pegase_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet'
```

## 10. Notes On LVN Generation Modes

The current LVN manual-flat full-generation scripts are:

```text
ScenarioSynthesis_PPC/sbatch/LVN_snapshot_envelope_manual_flat_array18.sh
ScenarioSynthesis_PPC/sbatch/LVN_snapshot_envelope_manual_flat_merge.sh
```

The script uses:

```bash
--start_mode manual_flat
--runs 2000
--workers 12
--rows_per_task 2
--save_steps 20
--no_save_y_matrix
--K 40
--convergence_mode misinf
```

With `#SBATCH --array=1-18`, this gives:

```text
18 chunks * 2000 rows = 36000 scenarios
```

Manual-flat start means:

```text
PQ buses:      |V| = 1.0 pu, angle = 0
PV buses:      |V| = generator voltage setpoint, angle = 0
Slack buses:   |V| = slack/ext_grid voltage setpoint, slack angle if available
```

The older LVN `dc_compile` scripts use:

```bash
--start_mode dc_compile
```

which builds the initial voltage from DC power-flow angles plus voltage magnitudes.

## 11. Notes On Asymmetric Branch Handling

LVN CGMES contains nonzero asymmetric branch impedance via pandapower PPC columns:

```text
BR_R_ASYM
BR_X_ASYM
```

The corrected generator exports:

```text
Branch_y_series_from = 1 / (R + jX)
Branch_y_series_to   = 1 / (R + R_ASYM + j(X + X_ASYM))
```

The dataloader/model reconstruction now uses from-side admittance for `Yft` and to-side admittance for `Ytf`.

Old LVN parquets generated before this fix should not be used for physics-loss checks/training. Standard PPC benchmark old parquets are safe when `BR_R_ASYM = BR_X_ASYM = 0`.

## 12. Common Problems

If Slurm merge is pending forever with dependency:

```bash
squeue -j <MERGE_JOBID>
sacct -j <ARRAY_JOBID> --format=JobID,State,ExitCode
```

If any array task failed, the `afterok` merge will not run.

If `/home/hpc` quota is full, make sure `--save_path` points into:

```text
/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out
```

If SSH from the local machine fails due DNS or proxy issues, retry with:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 fritz 'hostname && date'
```

If upload to `chkim@131.188.35.62` fails with:

```text
Permission denied (publickey,password,keyboard-interactive)
```

then passwordless SSH is not configured for this local environment. Do not keep retrying large `scp` transfers until authentication is fixed.

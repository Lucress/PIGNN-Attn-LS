# CIP SSH Handoff

This note documents how to run experiments on the CIP machines from this repo without Slurm. It is intended for another coding agent working from:

```bash
/Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
```

Unlike Alex, CIP is not treated as a Slurm cluster in these scripts. Each experiment is launched by SSHing into one CIP host and starting a background `nohup` Python process.

## 1. Local And Remote Paths

Local repo:

```bash
/Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
```

CIP repo:

```bash
~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
```

CIP Python environment:

```bash
/proj/aimi-adl/envs/adl23_2
```

The working Python executable is called directly:

```bash
/proj/aimi-adl/envs/adl23_2/bin/python
```

Do not rely on interactive shell behavior such as `conda activate`. In non-interactive SSH, the conda shell hook may not be initialized even if `base` appears active in a normal terminal login.

## 2. Manual SSH Setup

Basic connection:

```bash
ssh cip7b0
```

Manual environment setup after SSH:

```bash
export PATH=/proj/aimi-adl/envs/adl23_2/bin:$PATH
export PYTHONPATH=~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd ~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
```

Equivalent direct Python check:

```bash
ssh cip7b0 "/proj/aimi-adl/envs/adl23_2/bin/python --version"
```

## 3. CIP Host List And Indexing

The dispatch scripts use a 1-based host list. `--start-index 3` starts from the third host, `cip7b2`.

Current host order:

```text
 1  cip7b0
 2  cip7b1
 3  cip7b2
 4  cip7c0
 5  cip7c1
 6  cip7c2
 7  cip7d0
 8  cip7d1
 9  cip7d2
10  cip7e0
11  cip7e1
12  cip7e2
13  cip7f0
14  cip7f1
15  cip7f2
16  cip7g0
17  cip7g1
18  cip7g2
19  cip3a0
20  cip3b0
21  cip3b1
22  cip3b2
23  cip3b3
24  cip3c0
25  cip3c1
26  cip3c2
27  cip3c3
28  cip3d0
29  cip3d1
30  cip3d2
31  cip3d3
32  cip3e0
33  cip3e1
34  cip3e2
35  cip3e3
36  cip3f0
37  cip3f1
38  cip3f2
39  cip3f3
40  cip4a0
41  cip4b0
42  cip4b1
43  cip4b2
44  cip4b3
45  cip4c0
46  cip4c1
47  cip4c2
48  cip4c3
49  cip4d0
50  cip4d1
51  cip4d2
52  cip4d3
53  cip4e1
54  cip4e2
55  cip4e3
```

Use a later `--start-index` if earlier machines are already occupied.

## 4. Existing CIP Dispatch Scripts

Important local CIP dispatch scripts:

```bash
dispatch_case145_incremental_sweep_ssh.sh
dispatch_case145_increasing_nhead_layers_sweep_ssh.sh
dispatch_case145_capacity_sweep_ssh.sh
dispatch_case145_top3_angle_armijo_ssh.sh
dispatch_case145_best_nh8_dhi24_attn8_ssh.sh
dispatch_case118_145_loss_ablation_ssh.sh
dispatch_case300_top3_armijo_compare_ssh.sh
```

Helper used by some workflows:

```bash
remote_cip_launch_train.sh
```

The dispatch scripts usually:

1. Define a host list.
2. Choose hosts starting from `--start-index`.
3. SSH to each host.
4. Resolve `~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC`.
5. Use `/proj/aimi-adl/envs/adl23_2/bin/python`.
6. Export `PATH` and `PYTHONPATH`.
7. Create the remote log directory.
8. Start `nohup python train_valid_test.py ... > <launcher_log> 2>&1 < /dev/null &`.
9. Print `host=<host> pid=<pid> log=<launcher_log>`.
10. Write a local manifest TSV under `./results/logs/.../manifest.tsv`.

## 5. Example: Dry Run

Dry run the case145 incremental sweep from host index 3:

```bash
DRY_RUN=1 bash ./dispatch_case145_incremental_sweep_ssh.sh --start-index 3
```

Expected output shape:

```text
[1/9] host_index=3 cip7b2 <- 01_baseline_d4_dhi16_attn1_k40
[dry-run] cip7b2 <- 01_baseline_d4_dhi16_attn1_k40
```

Dry run only verifies host assignment and command generation. It does not start Python.

## 6. Example: Real Dispatch

Run the case145 incremental sweep from the first host:

```bash
bash ./dispatch_case145_incremental_sweep_ssh.sh --start-index 1
```

Run from the third host:

```bash
bash ./dispatch_case145_incremental_sweep_ssh.sh --start-index 3
```

Override the parquet:

```bash
bash ./dispatch_case145_incremental_sweep_ssh.sh \
  --start-index 3 \
  --parquet ../ScenarioSynthesis_PPC/out/case145_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
```

Useful environment overrides:

```bash
STAMP=20260522_120000
EPOCHS=100
BATCH=16
SEED=42
REMOTE_BASE=~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
REMOTE_ENV=/proj/aimi-adl/envs/adl23_2
REMOTE_LOG_DIR=./results/logs/case145_custom_20260522_120000
LOCAL_MANIFEST_DIR=./results/logs/case145_dispatch_20260522_120000
```

Example with overrides:

```bash
REMOTE_LOG_DIR=./results/logs/case145_capacity_dhi_heads_layers_20260320_213438 \
bash ./dispatch_case145_capacity_sweep_ssh.sh --start-index 21 --start-label-index 20
```

## 7. Example: Single Best Case145 Config

This script runs:

```text
d=4, d_hi=24, n_heads=8, num_attn_layers=8, K=40
```

Launch on one selected host:

```bash
bash ./dispatch_case145_best_nh8_dhi24_attn8_ssh.sh --host cip7b0
```

Override host and log dir:

```bash
REMOTE_LOG_DIR=./results/logs/case145_best_nh8_dhi24_attn8_rerun_20260522 \
bash ./dispatch_case145_best_nh8_dhi24_attn8_ssh.sh --host cip7c1
```

## 8. What The Remote Launch Actually Does

The essential remote command pattern is:

```bash
remote_base="${REMOTE_BASE/#\~/$HOME}"
remote_env="${REMOTE_ENV/#\~/$HOME}"
python_bin="${remote_env%/}/bin/python"

export PATH="${remote_env%/}/bin:${PATH}"
export PYTHONPATH="${remote_base}:${PYTHONPATH:-}"

cd "${remote_base}"
mkdir -p "${remote_log_dir}"

nohup "${python_bin}" train_valid_test.py \
  --EPOCHS="${EPOCHS}" \
  --BATCH="${BATCH}" \
  --seed_value="${SEED}" \
  --train_ratio=0.3333 \
  --valid_ratio=0.3333 \
  --lr_scheduler=CosineAnnealingLR \
  --PARQUET "${PARQUET}" \
  --vlimit \
  --model GNSMsg_EdgeSelfAttn \
  --log_to_file \
  --log_dir "${remote_log_dir}" \
  ${EXTRA_ARGS} \
  > "${launcher_log}" 2>&1 < /dev/null &
```

The `nohup ... &` is why the SSH command returns immediately while training continues on the CIP machine.

## 9. Check Running Processes

Check one host:

```bash
ssh cip7b0 "pgrep -af 'train_valid_test.py|python' | head -n 20"
```

Check a specific PID returned by dispatch:

```bash
ssh cip7b0 "ps -p <pid> -o pid,etime,pcpu,pmem,rss,vsz,cmd"
```

Check GPU:

```bash
ssh cip7b0 "nvidia-smi"
```

Compact GPU view:

```bash
ssh cip7b0 "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv"
```

## 10. Check Launcher Logs

The launcher log captures stdout/stderr from the remote `nohup` command.

Example from a manifest:

```bash
ssh cip7b0 "cd ~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC && tail -n 40 ./results/logs/case145_incremental_20260317_180141/01_baseline_d4_dhi16_attn1_k40_launcher.log"
```

If the launcher log says the directory does not exist, check whether the dispatch script converted relative paths to absolute paths correctly. The fixed scripts create the remote log dir before redirecting the launcher output.

## 11. Check Training Logs

Training logs are written by `--log_to_file --log_dir`.

List remote logs:

```bash
ssh cip7b0 "cd ~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC && ls -lh ./results/logs/<run_dir>"
```

Tail all training logs in a run directory:

```bash
ssh cip7b0 "cd ~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC && tail -n 120 ./results/logs/<run_dir>/*training_log.txt"
```

Extract metrics:

```bash
ssh cip7b0 "cd ~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC && grep -E 'Epoch|Test physics-loss|Test residual|Dataset sizes|Total number of parameters' ./results/logs/<run_dir>/*training_log.txt"
```

Look for failures:

```bash
ssh cip7b0 "cd ~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC && grep -RniE 'error|exception|traceback|out_of_memory|oom|cuda|nan|physics loss detached' ./results/logs/<run_dir>"
```

## 12. Download Results

Download a remote log directory from one CIP host:

```bash
scp -r cip7b0:~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/<run_dir> ./results/logs/
```

Download one launcher log:

```bash
scp cip7b0:~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/<run_dir>/<label>_launcher.log ./results/logs/
```

If runs were launched across multiple hosts, logs are stored on each host's filesystem path. In this CIP setup the home directory has generally behaved as shared enough for `~/Desktop/.../results/logs/...` to be visible across hosts, but if a file is missing, query the specific host shown in the manifest.

## 13. Stop A Run

Stop one PID:

```bash
ssh cip7b0 "kill <pid>"
```

Force stop only if normal kill fails:

```bash
ssh cip7b0 "kill -9 <pid>"
```

Stop matching training processes carefully:

```bash
ssh cip7b0 "pkill -f 'train_valid_test.py.*case145'"
```

Only kill jobs when the user explicitly asks or the target process is clearly the intended run.

## 14. Common Problems

`Unable to initialize conda`:

```text
Cause: non-interactive SSH shell does not load conda hooks.
Fix: do not use conda activate. Call /proj/aimi-adl/envs/adl23_2/bin/python directly.
```

`Unable to find executable python at /proj/aimi-adl/envs/adl23_2/bin/python`:

```text
Cause: wrong REMOTE_ENV or the environment is unavailable on that host.
Fix: ssh to the host and run ls -lh /proj/aimi-adl/envs/adl23_2/bin/python.
```

`launcher.log: No such file or directory`:

```text
Cause: redirect target directory did not exist, or relative path was interpreted outside the repo.
Fix: ensure the script converts REMOTE_LOG_DIR to an absolute path under REMOTE_BASE and runs mkdir -p before nohup redirection.
```

SSH returns immediately:

```text
This is expected. The command starts nohup python in the background and prints the remote PID.
```

No epoch lines:

```text
Check launcher log first for import/data errors.
Then check pgrep/ps and nvidia-smi to see whether Python is still running.
```

## 15. Interpreting CIP Output

Successful dispatch output looks like:

```text
[1/9] host_index=1 cip7b0 <- 01_baseline_d4_dhi16_attn1_k40
host=cip7b0 pid=2374728 log=/home/cip/ai2023/no12neni/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/case145_incremental_20260317_180141/01_baseline_d4_dhi16_attn1_k40_launcher.log
```

This means:

```text
host_index=1: selected the first host in the list.
cip7b0: SSH target.
01_baseline...: experiment label.
pid=2374728: background Python process ID on cip7b0.
log=...launcher.log: stdout/stderr file for the nohup launch.
```

After dispatch, always check:

```bash
column -t -s $'\t' ./results/logs/<dispatch_dir>/manifest.tsv
ssh <host> "cd ~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC && tail -n 40 <remote_log_dir>/<label>_launcher.log"
```

Good signs:

```text
launcher log is quiet or contains normal training output.
training_log.txt exists.
Epoch lines appear.
Test physics-loss and RMSE appear at the end.
GPU memory/utilization are nonzero while running.
```

Bad signs:

```text
Python process disappeared quickly.
launcher log contains import errors or missing parquet errors.
CUDA OOM appears.
No training_log.txt appears.
Repeated "physics loss detached" appears for Armijo runs.
```

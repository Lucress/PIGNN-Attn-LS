# Helma Handoff

Running the GridFM / GridSFM / LUMINA benchmarks on **helma** (NHR@FAU H100/H200
cluster). Companion to `ALEX_SLURM_HANDOFF.md`; helma differs from alex in three
ways that matter, all covered below.

Status as of 2026-08-09: environment built, both data pipelines staged
(OPFData case14 **and** case118), and two job sets completed with zero
failures — 10 verification jobs across `h100`/`preempt`, then the 10
native-loss OPFData runs (§4).

---

## 1. The three things that differ from alex

| | alex | helma |
|---|---|---|
| Account | `iwi5295h` | `b313dc11` (group `b313dc`) |
| Compute-node filesystems | `/home/hpc`, `/home/vault`, `/home/woody` | **`/home/hpc` only** |
| GPU partitions | `a40`, `a100` | `h100`, `preempt` (**not** `h200`) |

### 1.1 Different account ⇒ no direct alex→helma copy

helma is a **separate account** from alex. `/home/hpc` and `/home/vault` are
mounted on both clusters, but `b313dc11` cannot read `iwi5295h`'s directories
(permission denied), and vice versa. There is no server-side shortcut.

**Every transfer must be routed through a local machine:**

```bash
scp alex:/path/to/file  /tmp/file      # pull
scp /tmp/file  helma:/path/to/file     # push
```

### 1.2 `/home/vault` does NOT exist on helma compute nodes

This one silently costs you a job. The login node sees vault (753 T, generous);
compute nodes see **only `/home/hpc`**. A job whose data or `--ckpt_dir` lives
in vault dies immediately with:

```
PermissionError: [Errno 13] Permission denied: '/home/vault'
```

Verified directly on a compute node:

```
$ srun --partition=h100 --gres=gpu:1 ls /home/vault/b313dc/b313dc11
ls: cannot access '/home/vault/b313dc/b313dc11': No such file or directory
   (same for /home/woody)
```

**Consequence:** datasets, checkpoints and logs must live under `$HOME`
(`/home/hpc/b313dc/b313dc11`), whose quota is **100 G soft / 200 G hard**
(currently 52 G used). Vault remains usable from the login node as a staging
and archive area, but stage into `$HOME` before submitting. Compute nodes also
have a large node-local `/tmp` (14 T) if you need scratch within a job.

### 1.3 `h200` is not available to this account

`h200` is gated by `AllowAccounts` to a fixed list (`unrz`, `v111dc`, `y1xx*`,
`y3xx*`, `y5xx*`, `y700dd`, …) that **does not include `b313dc`**. Submitting
there fails at `sbatch` time:

```
sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified
```

This is an entitlement, not a misconfiguration — it needs a project-side change
to fix. Usable GPU partitions:

| Partition | Time limit | Notes |
|---|---|---|
| `h100` | 1-00:00:00 | 96 nodes, H100 94 GB (cc 9.0) |
| `preempt` | 2-00:00:00 | default partition, preemptible, longer wall clock |
| `cpu` / `preempt_cpu` | 1 / 2 days | CPU-only, for dataset generation |

`preempt` happened to schedule H100 nodes in the verification runs and produced
numerically near-identical results to `h100` (see §5), so it is a fine fallback
when `h100` is saturated — just make jobs checkpoint-resumable, since they can
be preempted.

---

## 2. Layout on helma

```
/home/hpc/b313dc/b313dc11/
├── conda-envs/gridfm-py312/          # the environment (python 3.12.13)
├── helma_setup_env.sh                # the script that built it (re-runnable)
├── helma_setup.log                   # its log; ends with "[setup] DONE"
└── PIGNN-Attn-LS/
    ├── LUMINA/                       # cloned repo, pip-installed -e
    │   └── checkpoints/{lumina_config.json,lumina_model.safetensors}
    ├── GridSFM/                      # cloned repo, pip-installed -e
    │   └── checkpoints/gridsfm_open_v1.1.pt
    ├── data/case14_opf_task_ready.parquet        # 139 MB, pandapower pipeline
    ├── opfdata/                                  # 8.5 GB, DeepMind OPFData
    │   └── dataset_release_1/pglib_opf_case{14,118}_ieee/
    ├── lumina-sdk/  gridfm-graphkit/             # native-loss sources (§4)
    └── PIGNN-Attn-LS-PPC/            # the code
        ├── train_valid_test_{gridfm,gridsfm,lumina,opfdata}.py
        ├── opfdata_pipeline.py, opf_task.py
        ├── Dataset_optimized_complex_columns.py
        ├── collate_blockdiag_optimized_complex_columns.py
        ├── read_npy_columns_optimized.py
        ├── summarize_{opf,model}_logs.py
        ├── sbatch/<GROUP>/*.sh        # generated job scripts
        ├── sbatch/Job_out/            # SLURM stdout/stderr
        ├── results/logs/<GROUP>/      # per-run training logs
        └── results/ckpt/<GROUP>/      # best checkpoints
```

`/home/vault/b313dc/b313dc11/` holds staging copies of `data/` and `opfdata/`.
Login-node only — see §1.2.

---

## 3. The environment

Built by `~/helma_setup_env.sh` (idempotent; re-run to rebuild from scratch).
There is **no `python` module** on helma and the system python is 3.9, so the
env is a self-installed miniforge3. Internet works from the login node (pypi and
storage.googleapis.com both reachable), so no offline wheel dance is needed.

```
python 3.12.13
torch 2.7.1+cu126        torch-geometric 2.8.0
numpy 2.5.1              pandas 3.0.5        pyarrow 25.0.0
scipy 1.18.0             safetensors 0.8.0   huggingface_hub 1.27.0
tqdm 4.70.0              joblib 1.5.3
lumina-inference 0.1.0   (editable -> ~/PIGNN-Attn-LS/LUMINA)
gridsfm 1.1.0            (editable -> ~/PIGNN-Attn-LS/GridSFM/model)
```

Interpreter: `/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python`.

Call it by absolute path in job scripts — no `conda activate` needed, which
avoids shell-init issues under `srun`. Two env vars are set in every job:

```bash
export PYTHONPATH=<BASE>:$PYTHONPATH   # so opf_task / opfdata_pipeline import
export HF_HUB_DISABLE_XET=1            # xet backend is flaky here
```

`module load gpu-env/2025` exposes `cuda/12.6.2` and `cuda/12.8.0`, but the
torch cu126 wheel ships its own runtime, so loading a CUDA module is **not**
required.

---

## 4. Submitting

`dispatch_helma_smoke.sh` (in the repo, run from your local machine) generates
job scripts, ships them in one transfer, and submits. It defaults to
`PARTITIONS="h100 preempt"`, 1 epoch, and small sample caps.

```bash
bash dispatch_helma_smoke.sh                          # both partitions
PARTITIONS=h100 EPOCHS=40 CAP_TRAIN=100000 \
  GROUP=my_run bash dispatch_helma_smoke.sh           # a real run
```

For the native-loss OPFData runs use `dispatch_opfdata_native.sh`, which is
cluster-agnostic — its defaults target alex, and helma is reached by overriding
the paths. `GRES=gpu:1` is required here (alex uses `gpu:a40:1`), and `h100` or
`preempt` must be named explicitly since `h200` is account-gated (§1.3):

```bash
HOST=helma \
  BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC \
  CK=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS \
  PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python \
  OPFDATA_ROOT=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata \
  GRES=gpu:1 CASE=case14 PARTITION=h100 bash dispatch_opfdata_native.sh
```

That variant needs extra packages beyond the base environment: `lumina-sdk` and
`gridfm-graphkit` installed `--no-deps` from `~/PIGNN-Attn-LS/`, plus
`opt_einsum`, `lightning`, `matplotlib` and `torch_scatter`. All are already
installed; see `native_loss.py` for which parts of each repo's loss are used and
which are not.

Both dispatchers wrap every `ssh`/`scp` in a retry loop, because the NHR login gateway
(`csnhr.nhr.fau.de`) intermittently refuses connections — a bare `ssh helma`
failing with `Connection refused` is normal and clears within a minute or two.

A minimal job script, if you'd rather hand-write one:

```bash
#!/bin/bash -l
#SBATCH --job-name=my_run
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/my_run.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/my_run.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
export PYTHONPATH=$BASE:${PYTHONPATH:-}
export HF_HUB_DISABLE_XET=1
cd $BASE

srun $PY -u train_valid_test_opfdata.py \
  --model lumina --init_mode pretrained \
  --case_name pglib_opf_case14_ieee \
  --opfdata_root /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata --num_groups 1 \
  --model_config  /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/LUMINA/checkpoints/lumina_config.json \
  --pretrained_checkpoint /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/LUMINA/checkpoints/lumina_model.safetensors \
  --run_name my_run --log_to_file --log_dir $BASE/results/logs/my_run \
  --ckpt_dir $BASE/results/ckpt/my_run \
  --BATCH 32 --EPOCHS 40 --LR 1e-4
```

For the pandapower-parquet pipeline, swap in `train_valid_test_lumina.py`
(or `_gridfm` / `_gridsfm`) with `--PARQUET .../data/case14_opf_task_ready.parquet
--task opf --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet`.
Same flags as on alex — nothing is helma-specific except the paths.

---

## 5. Verification results (2026-08-07)

Group `helma_smoke_20260807_234648`, 128 train / 64 valid / 64 test, 1 epoch,
small model config. All 10 jobs `COMPLETED` in ~1 minute each.

Every job reported `NVIDIA H100, 95830 MiB, cc 9.0` and
`torch 2.7.1+cu126, cuda 12.6`.

| Pipeline | Model | Params | Test RMSE | \|V\| | θ |
|---|---|---:|---|---|---|
| OPFData | GridSFM | 15,148,227 | 1.419e-02 | 1.07e-02 | 0.54° |
| OPFData | LUMINA | 2,375,548 | 7.589e-02 | 2.12e-02 | 4.18° |
| OPFData | GridFM (scratch) | 267,962 | 2.039e-01 | 1.85e-02 | 11.64° |
| Parquet | GridFM (scratch) | 268,154 | 5.491e-02 | 1.93e-02 | 2.95° |
| Parquet | LUMINA (scratch) | 2,375,548 | 9.806e-02 | 5.77e-02 | 4.54° |

Three things this confirms beyond "it ran":

1. **Parameter counts match alex exactly** — LUMINA 2,375,548 and GridSFM
   15,148,227, i.e. the checkpoints and model construction are identical across
   clusters. (GridFM's count reflects the small smoke config, not the 20 M
   production model.)
2. **`h100` and `preempt` agree numerically.** LUMINA/OPFData gave 7.5885e-02
   on `h100` vs 7.5892e-02 on `preempt`; GridFM/parquet was bit-identical
   (5.4909e-02 both). Partition choice does not perturb results.
3. **Both pipelines' OPF metric plumbing is live** — balance, generator limits,
   voltage band and cost gap all populate, and the voltage-band violation is
   0.000 for LUMINA, confirming the per-bus `Bus_vmin`/`Bus_vmax` wiring
   survived the transfer.

The absolute error numbers are meaningless as science — 128 training samples
and one epoch. They exist only to prove the stack executes end to end.

---

## 6. Staging new data

OPFData downloads directly on helma (googleapis is reachable), which is far
faster than routing through a laptop:

```bash
# on the helma login node, in a detached process
~/conda-envs/gridfm-py312/bin/python - <<'PY'
from torch_geometric.datasets import OPFDataset
root = "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata"
for split in ("train", "val", "test"):
    ds = OPFDataset(root, case_name="pglib_opf_case14_ieee",
                    split=split, num_groups=1)
    print(split, len(ds))
PY
```

case14 with `num_groups=1` gives 13500 / 750 / 750 graphs and occupies 1.1 GB;
case118 the same split at 7.3 GB (6.6 GB raw + 0.7 GB processed). Budget quota
before pulling more groups or larger cases — `$HOME` has ~40 G free, and
case118/case300 at 20 groups will not fit.

> **Do not delete `raw/` to save quota.** An earlier revision of this document
> recommended exactly that, and it is wrong: PyG's `OPFDataset` calls
> `_download()` whenever the raw files are absent, *even when `processed_*`
> already exists*. Deleting `raw/` makes every training job re-download, and
> concurrent jobs then race inside the same extract directory and die with
> `FileExistsError: .../raw/gridopt-dataset-tmp/...`. This cost five jobs before
> it was diagnosed. If quota is tight, drop a whole case rather than its `raw/`.

pandapower parquets have to come from alex, so they take the local-machine
detour of §1.1.

---

## 7. Known caveats

- **Login gateway refuses connections intermittently.** Retry; it clears. All
  automation should wrap remote calls in a retry loop.
- **`preempt` jobs can be killed mid-run.** Use it for long jobs only if they
  checkpoint and resume.
- **`$HOME` quota is the real constraint** (100 G soft), not vault. Watch it
  when adding OPFData cases; `quota -s` reports usage.
- **Don't point `--ckpt_dir` or `--log_dir` at vault.** See §1.2.
- **GridSFM's zero-gradient freeze is not confined to the parquet pipeline.**
  It is documented in `opf_surrogate_report.tex` for parquet, and the completed
  alex OPFData runs reproduce it in **3 of 4** training runs — every one except
  `case14`/`pre_ft`:

  | Run | Final train loss | Froze at | Test RMSE |
  |---|---|---|---|
  | case14 pre-ft | 9.9e-05 | — (converged) | 7.92e-03 |
  | case14 scratch | 9.7975e-03 | ~epoch 38 | 3.11e-02 |
  | case118 scratch | 2.2444e-02 | ~epoch 31 | 1.20e-01 |
  | case118 pre-ft | 2.6714e-01 | epoch 12 (after *diverging* from 1.22e-01) | 3.51e-01 |

  The strongest evidence that this is a **collapsed magnitude head**, not slow
  convergence: both case118 runs report |V| RMSE of **exactly 3.3899e-02**
  despite different initialisations (pretrained vs scratch). Two independent
  runs cannot agree to five digits on a learned quantity; the magnitude output
  has almost certainly gone constant. The case118 pre-ft angle error pinned at
  exactly 2.0000e+01 deg points the same way (a clip bound, not a fit).
  Frozen runs still emit valid finals from their best checkpoint, so the
  numbers are real — they just reflect far fewer effective epochs than the 40
  requested. Treat any flat GridSFM loss as suspect and check whether
  consecutive epochs are **bit-identical** rather than merely converged.
  This is a pre-existing model issue, unrelated to helma, and is why the
  parquet smoke set here covers LUMINA and GridFM only.

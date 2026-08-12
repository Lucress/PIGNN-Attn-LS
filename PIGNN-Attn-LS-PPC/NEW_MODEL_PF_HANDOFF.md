# Training a new architecture on the PF corpus

For a thread bringing a new model to the same AC power-flow benchmark that
PIGNN-Attn-LS, the GridFM mirror and the released `gridfm_graphkit` model have
been run on. Everything below is measured on this corpus, not assumed.

The goal is a comparable number: same 31 grids, same splits, same metrics.

---

## 1. The data

**Phase-0 backbone corpus**: 31 grids, 4 to 9241 buses, ~86 GB.

Select by the **solver tag in the filename**, never by size and never by the
`_A_` tag:

```
<grid>_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_..._NR_branchrows_directSI[_rg20].parquet
```

* `_ppNR_` = pandapower Newton, 30 grids.
* `_cNR_` = custom Newton — **LVN_heo1 only**, because pandapower cannot solve
  that grid.
* `_A_` is a **different, older scenario set**. It exists for most grids and is
  a tempting default because the files are large and readable. Picking it was a
  real error in this project and cost a full campaign: it trains fine and
  produces plausible numbers that are not comparable to anything.
* A `u0clean` tag appears in the SimBench / ENTSO-E / LVN_heo1 names but not in
  the earlier ones, so match on the `_ppNR_` substring rather than a full glob.

Canonical location (alex and fritz):

```
/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/
```

The authoritative grid list, with measured bus counts, byte sizes and the exact
filenames, is the `MANIFEST` array in `dispatch_pf_pignn.sh`. Copy it rather
than re-deriving it.

### Row groups — do this before training

Most `*_NR_branchrows_directSI.parquet` files were written with 3000–6000 rows
per group. Training shuffles, so a batch of a few samples decodes a whole group:
measured **3191 ms/batch against 176 ms/batch**, an 18x penalty that dominates
everything else across 31 grids. 28 files have been re-laid-out to 20 rows per
group and carry an `_rg20` suffix; SimBench, ENTSO-E and LVN_heo1 were already
at 20 and have no suffix.

If a new file appears, run `rewrite_rowgroups.py` (it also has a `--scan DIR`
mode that reports the layout of everything under a directory without writing).

### Known data caveats

* **case145 has 7885 scenarios**, 22% of the 36,000 the others got. It is the
  worst grid for both models tested so far (θ ≈ 11.6–11.7° on each), which is
  almost certainly the data, not the model. A ±20% regeneration is outstanding.
  Exclude it or footnote it; do not compare it across grids.
* Other short grids: case9241pegase 18,280, case6515rte 20,135, case6495rte
  23,332, case6470rte 25,551, case300 27,994, case1354pegase 32,627,
  GBnetwork 34,424, iceland 35,605.
* Seven parquets in that directory are **corrupt** (`ArrowInvalid`), including
  `case6470rte_ppcY_A_...` and `case9241pegase_ppcY_A_...`. The manifest already
  routes around them.

---

## 2. The pipeline your model plugs into

```python
from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag, ybus_matvec

ds = ChanghunDataset(PARQUET, per_unit=True, target_S_base=1e8,
                     share_grid=True, share_ybus=True,
                     lazy_row_groups=True, row_group_cache_size=4,
                     complex_dtype="complex128")
batch = collate_blockdiag([ds[i] for i in range(B)])   # sparse Y-bus by default
```

`collate_blockdiag` produces a **single block-diagonal mega-graph** with a
leading batch axis of 1. A batch of 4 case14 samples is one graph of 56 buses,
not 4 graphs of 14. `sizes` / `offsets` / `branch_sizes` carry the per-sample
split if the model needs it.

Batch contract (4 x case14 shown, `N` = 4·14 = 56, `E` = 4·20 = 80):

| key | shape | dtype |
|---|---|---|
| `bus_type` | (1, N) | int64 — 1 slack, 2 PV, else PQ |
| `V_start` | (1, N, 2) | float64 — (magnitude, angle), the DC init |
| `V_newton` | (1, N, 2) | float64 — the label |
| `S_start` | (1, N) | complex128 — the given injection |
| `S_newton` | (1, N) | complex128 |
| `Ybus` | (N, N) | complex128, **SPARSE COO** |
| `Y_shunt_bus` | (1, N) | complex128 |
| `Branch_f_bus` / `_t_bus` | (1, E) | int64, already offset to global indices |
| `Branch_status`, `Branch_tau`, `Branch_shift_deg` | (1, E) | float64 |
| `Branch_y_series_from` / `_to` / `_ft` | (1, E) | complex128 |
| `Branch_y_shunt_from` / `_to` | (1, E) | complex128 |
| `Is_trafo` | (1, E) | int8 |
| `vn_kv`, `V_base_bus` | (1, N) | float64 |
| `sizes`, `offsets`, `branch_sizes` | (B,) | int64 |

Everything is **per-unit on a 100 MVA base** under `--PER_UNIT --target_S_base 1e8`.
That matters: the branch admittances have *already* been converted with the
local bus bases, so re-applying `Vbase²/S_base` is wrong by a factor of ~750.
That mistake was made once here and would not have raised — it would have
trained and produced plausible numbers.

### The Y-bus is sparse — do not densify it

The dense block-diagonal Y-bus costs `(B·N)² × 16` bytes, quadratic in
batch × buses. case2848rte at batch 32 asked for **124 GB** and every large grid
was OOM-killed. It is now assembled as sparse COO:

| case2848rte, batch 8 | dense | sparse |
|---|---|---|
| footprint | 7921 MB | **1.8 MB** |
| nonzeros | — | 77,856 (0.015%) |

Verified identical, not merely close: `to_dense(sparse)` equals the dense matrix
exactly and `S = V ⊙ conj(YV)` agrees to 1.1e-11 absolute (5.2e-13 relative).

**Use `ybus_matvec(Y, Vc)` for every `Y @ V`.** It accepts dense or sparse `Y`
and `[N]` or `[1, N]` vectors. Calling `.squeeze()`, `.unsqueeze()`, `.max()` or
`torch.diag()` on a sparse CUDA tensor raises `NotImplementedError`
(`aten::as_strided`, `aten::max`) — this bit both the model adapter and a
validator here. If you need a dense Y for a one-off diagnostic, `.to_dense()`
explicitly on a small batch.

---

## 3. Configuration

### Do not copy widths across architectures

GridFM runs at `hidden 48, 12 layers, 8 heads`. PIGNN-Attn-LS runs at
`d_hi 24, 8 layers, 8 heads`. That is not an oversight: PIGNN's own capacity
sweep (`latex/results.pdf` §4.4) found **`d_hi = 64` diverging outright**
(physics loss ~1e5, θ 40–88°), and 48 sits next to that. It also found deeper is
not better — L=8 beat L=10 and L=12 on angle — while `d_hi ≤ 16` with `L ≤ 2`
plateaus at θ ≈ 5–6°.

The two models are not comparable in width because they are not the same kind of
model: GridFM is a plain message-passing encoder, PIGNN is an unrolled solver
with K correction steps whose gradients accumulate through all of them. Start
from your own architecture's scaling behaviour, and if you have no sweep, run
one on a mid-size grid (case300 is a good probe — small enough to iterate, hard
enough to separate configurations).

### Reference configurations actually used

| | GridFM mirror / graphkit | PIGNN-Attn-LS |
|---|---|---|
| capacity | hidden 48, 12 layers, 8 heads | d 4, d_hi 24, 8 layers, 8 heads, K 40 |
| LR | 5e-4 | 1e-4 |
| epochs | 40 | 40 |
| loss | MSE(V) + 1e-2 · physics (log-cosh) | same |
| splits | 1/3 train, 1/3 valid, 1/3 test, seed 42 | same |
| params | 20.0 M / 20.1 M | ~65 k |

Keep the loss, splits and epoch count identical if you want the numbers to be
comparable; they are what the existing tables were produced under.

### Batch size — measure it, do not extrapolate

Peak GPU memory is linear in `B × N` for both models tested, but the constant
differs by 2.6x:

| model | per node | source |
|---|---|---|
| GridFM mirror | **479 KB** | 4.3 GB @ B·N=9241 → 67.0 GB @ 147,856, OOM at 221,784 |
| PIGNN-Attn-LS | **1.24 MB** | 11.4 GB @ B·N=9241 → 44.8 GB @ 36,964, OOM at 73,928 |

PIGNN costs more because K=40 unrolled steps each keep activations for the
backward pass. Both dispatchers take a `BN_BUDGET` and derive
`batch = clamp(BN_BUDGET / buses, 1, cap)`:

* GridFM on helma H100 (94 GB): `BN_BUDGET=120000`, cap 64 → ~54 GB peak.
* PIGNN on alex A40 (48 GB): `BN_BUDGET=28000`, cap 32 → ~34 GB peak.

**Measure your own constant before submitting 31 jobs.** Two failure modes cost
real time here:

1. Extrapolating from the point where a job *OOM'd*. The epilogue's
   `max_memory_usage` at an OOM is the allocator's ceiling, not the model's
   demand — it reads ~94 GB whether the true need was 100 GB or 300 GB. That
   gave a 2.9x-wrong estimate and a second round of OOMs.
2. Reproducing the driver's preprocessing in a standalone probe. The dtypes
   came out wrong and the probe measured nothing. Probe *through the driver*:
   submit short jobs at several batch sizes with `--EPOCHS 1
   --max_train_samples 64` and read which succeed.

---

## 4. Clusters

| | alex | helma |
|---|---|---|
| account | `iwi5295h` | `b313dc11` |
| GPU | A40 48 GB (`--gres=gpu:a40:1`, partition `a40`) | H100 94 GB (`--gres=gpu:1`, partition `h100`) |
| `/home/vault` on compute nodes | **visible** | **not visible** |
| `--mem` on GPU jobs | allowed | **rejected** — "Memory corresponding to number of GPUs will be allocated automatically" |
| time limit | 24 h | 24 h (`preempt`: 48 h, preemptible) |

alex is much simpler for this corpus: the job copies its parquet straight from
vault to node-local disk. On helma, everything must be staged into `$HOME`
first, and `$HOME` has ~42 GB of soft-quota headroom against an 86 GB corpus —
see `HELMA_HANDOFF.md` for the rolling-stage arrangement.

Cross-cluster copies go at filesystem speed thanks to `chmod o+x` on
`/home/vault/iwi5/iwi5295h`; the subtree below it was already world-readable.
Do not route data through a laptop.

The NHR login gateway (`csnhr.nhr.fau.de`) refuses connections often enough that
any multi-hour dispatcher must wrap every `ssh`/`scp` in a retry loop. A bare
`ssh` under `set -e` will kill the run — that happened after six of 23 jobs.

---

## 5. Metrics

The drivers log, per epoch, on both train and valid:

```
loss | rmse (mag, ang°) | ΔP∞, ΔQ∞ | mean|ΔP|, mean|ΔQ| | p95|ΔP|, p95|ΔQ| | tol% | conv% | time
```

and a `Final test-set` block at the end. All power quantities are pu on a
100 MVA base (1e-2 pu = 1 MW/MVAr).

**Residual masking.** `ΔS = S_implied − S_set` from `S = V ⊙ conj(YV)`, with
active power not scored at the slack and reactive power not scored at slack or
PV buses — there the injection is a free variable, not a known input. Scoring
them would charge correct behaviour as error.

**Report residuals, not just voltage RMSE.** On this corpus the two rankings
disagree: Spearman 0.78, with individual grids moving 10–11 places. Voltage RMSE
alone rated iceland mid-pack when it had the worst mean active-power residual of
all 31 grids. The distribution is also extremely heavy-tailed — on two grids the
*mean* exceeds the *95th percentile*, so a handful of buses carry everything.
Both a mean and a max are needed; neither alone is honest.

Note that `ΔP∞` must be **pooled over the whole test set**, not averaged across
batches. Averaging per-batch maxima understates the true worst bus by 3–6x, and
a percentile cannot be recovered from per-batch summaries at all. One driver
here had this wrong.

### Worth adding

Two diagnostics from the literature that this project does not yet compute:

* **Per-channel regression slope / R²** of predicted vs true (GridSFM §6.2). MAE
  is small whenever a quantity has a narrow range — a flat predictor of
  |V| = 1.0 already scores well — so neither MAE nor RMSE distinguishes "tracks
  the solver" from "collapsed to a constant". A slope near 0 catches the latter
  immediately. This project hit exactly that failure mode and detected it only
  by noticing two independent runs agreeing to five significant figures.
* **|ΔS| = √(ΔP² + ΔQ²)**, mean and max (PFΔ §4.4). A bus mismatch is one
  complex quantity, not two. PFΔ also argues the residual is the *unsupervised*
  metric — power flow can have multiple solutions, so distance to the specific
  IPOPT/pandapower solution carries solver bias that the residual does not.

PFΔ trains every model **three times from different initialisations** and
reports the standard deviation. That is not decoration here: the same GridFM
configuration gave 1.040e-3 on one cluster and 4.300e-3 on another for case14,
and GridSFM's scratch runs swapped rank between clusters. `dispatch_pf_pignn.sh`
and `dispatch_pf_gridfm.sh` both take `SEEDS="42 43 44"`.

---

## 6. Baselines to compare against

GridFM mirror, PF, 40 epochs, final test set. Full table in
`latex/pf_gridfm_report.pdf`.

| grid | buses | RMSE | \|V\| RMSE | θ [deg] |
|---|---|---|---|---|
| SimBench | 94 | 1.203e-3 | 1.05e-3 | 0.034 |
| case14 | 14 | 5.829e-3 | 2.50e-3 | 0.302 |
| case118 | 118 | 1.144e-2 | 2.94e-3 | 0.633 |
| case300 | 300 | 4.835e-2 | 1.15e-2 | 2.691 |
| LVN_heo1 | 722 | 2.756e-2 | 2.74e-2 | 0.187 |
| case1354pegase | 1354 | 5.199e-2 | 2.41e-2 | 2.638 |
| case2848rte | 2848 | 3.720e-2 | 2.61e-2 | 1.519 |
| case9241pegase | 9241 | 1.227e-1 | 2.76e-2 | 6.849 |
| case145 | 145 | 2.060e-1 | 2.76e-2 | 11.695 |

Two structural findings that any new model should be checked against:

* **Error lives in the angle.** |V| spans a factor of 31 across the corpus,
  θ a factor of 344. The ranking is set almost entirely by the angle.
* **Bus count does not predict difficulty; grid family does.**
  GBreducednetwork (29 buses) is worse than case118; case6495rte (6495) beats
  case300. The IEEE-standard cases sit at the top and the pegase / rte / sp / GB
  families at the bottom, independent of size.

The released `gridfm_graphkit` model and PIGNN-Attn-LS runs are in progress; ask
for the current tables rather than assuming these mirror numbers are the best
available.

---

## 7. Files

| file | role |
|---|---|
| `Dataset_optimized_complex_columns.py` | the parquet loader, per-unit conversion, `share_grid` caching |
| `collate_blockdiag_optimized_complex_columns.py` | block-diagonal collate, sparse Y-bus, `ybus_matvec` |
| `train_valid_test.py` | PIGNN-Attn-LS driver — closest template for a new unrolled model |
| `train_valid_test_gridfm.py` | GridFM driver, `--gridfm_impl {mirror,graphkit}` |
| `dispatch_pf_pignn.sh` | 31-grid dispatcher for alex, with the manifest |
| `dispatch_pf_gridfm.sh` | same for helma, with the rolling-stage arrangement |
| `rewrite_rowgroups.py` | row-group re-layout and `--scan` |
| `latex/results.pdf` | PIGNN capacity sweep, Armijo ablation, earlier per-case results |
| `latex/pf_gridfm_report.pdf` | the GridFM PF baseline, including residual tables |

---

## 8. One result that will matter if your model is also an unrolled solver

PIGNN-Attn-LS **does not train on this corpus without its Armijo line search**,
at case300 scale and above. Measured, case300, 6 epochs:

| | loss | θ |
|---|---|---|
| Armijo on | **1.14** | 20.2° |
| Armijo off | 505 | 38.8° |
| Armijo off, 30 epochs | 582 | **47.4°** (worse) |

LR 1e-5 through 1e-4 and the narrower architecture from the earlier successful
case300 run were all tried; every one plateaued at θ ≈ 37°, which is roughly the
DC initialisation the model starts from. It is not slow convergence — it is no
convergence, and it gets worse with more epochs. case14 converges fine either
way, so a small-grid smoke test will not reveal this.

If your architecture also refines an initial guess over unrolled steps, budget
for a step-size safeguard and verify on a mid-size grid before committing a
campaign.

# Running the released `gridfm_graphkit` model on the PPC power-flow parquets

For the thread training GridFM as an AC power-flow surrogate across the PPC test
cases. This covers what changes when you swap our local re-implementation for
IBM's released model, and the one conversion that has to be right.

---

## 1. What you are swapping, and why it matters

Until now "GridFM" in this project has been
`train_valid_test_gridfm.GridFMHeteroSurrogate` — a local model written in
GridFM's architectural style. Nothing imported `gridfm_graphkit`'s model, and no
GridFM checkpoint was ever loaded. LUMINA and GridSFM, by contrast, are always
run from their own repositories, so GridFM was the odd one out.

The released model is `gridfm_graphkit.models.gnn_heterogeneous_gns.GNS_heterogeneous`.
It is **not** just a renamed copy of our mirror. It additionally has:

| | local mirror | released `GNS_heterogeneous` |
|---|---|---|
| physics feedback | none | every layer: branch flows → node injections → `physics_mlp` |
| \|V\| bounding | fixed `clamp(vmin, vmax)` | `bound_with_sigmoid` against the per-bus band |
| generator head | none | `mlp_gen → Pg` |
| known quantities | all predicted | `mask_dict` pins them to their inputs |
| task-specific decoder | none | `PhysicsDecoderPF` recovers Pg at slack and Qg at PV+slack from the nodal balance |

The `PhysicsDecoderPF` is the part that matters most for power flow: the model
does not have to learn the quantities that the balance equations determine.

**One asymmetry this does not fix.** `gridfm_graphkit` ships no weights — no
checkpoint file and no HuggingFace reference anywhere in the repo. GridFM is
scratch-only regardless of which implementation you use, while LUMINA and
GridSFM have public checkpoints. Do not describe a GridFM run as "pretrained".

---

## 2. What already lines up (nothing to do)

Bus and generator feature layouts are **byte-identical** between this project
and `gridfm_graphkit.datasets.globals`:

```
bus:  PD_H=0 QD_H=1 QG_H=2 VM_H=3 VA_H=4 PQ_H=5 PV_H=6 REF_H=7
      MIN_VM_H=8 MAX_VM_H=9 MIN_QG_H=10 MAX_QG_H=11 GS=12 BS=13 VN_KV=14   (15 cols)
gen:  PG_H=0 MIN_PG=1 MAX_PG=2 C0_H=3 C1_H=4 C2_H=5 [G_ON=6 not used]      (6 cols)
```

So whatever builds bus/gen features today transfers unchanged. The released
model reads **6** generator columns, not 7 — drop `G_ON`.

Capacity also already matches: `gridfm-graphkit/examples/config/HGNS_PF_datakit_*.yaml`
uses `hidden_size 48, num_layers 12, attention_head 8`, which is exactly what our
runs used.

---

## 3. The one real conversion: the edge attribute

This is the only place you can silently get wrong answers, so it gets its own
section.

Our parquet loader hands the model a **2-column** edge attribute — the Y-bus
off-diagonal `Y[i,j]`, i.e. the *mutual* admittance only. The released model
needs a **10-column** attribute that also carries the per-branch *self* term,
because it computes branch flows itself:

```
I_ft = Yff · V_f + Yft · V_t
```

with `edge_attr[:, 2:4] = Yff` (`YFF_TT_R/I`) and `edge_attr[:, 4:6] = Yft`
(`YFT_TF_R/I`). Columns: `P_E=0, Q_E=1, YFF_TT_R=2, YFF_TT_I=3, YFT_TF_R=4,
YFT_TF_I=5, TAP=6, ANG_MIN=7, ANG_MAX=8, RATE_A=9`.

The self term cannot be recovered from the dense Y-bus: its diagonal is the sum
over all incident branches plus the bus shunt. It has to be rebuilt per branch.

### The trap

With `--PER_UNIT` the loader has **already** converted the branch admittances to
pu using the local bus bases:

```python
Branch_y_series_from *= Vf**2 / S_base
Branch_y_series_to   *= Vt**2 / S_base
Branch_y_series_ft   *= Vf*Vt / S_base
Branch_y_shunt_from  *= Vf**2 / S_base      # and _to with Vt**2
Y_shunt_bus          -> pu
```

Applying the bus bases again — which is the natural thing to do, since
`reconstruct_Y_pandapower_branchrows_direct_SI_np` works in SI — is wrong by
`Vbase²/S_base`. The first version of this adapter did exactly that and missed
the Y-bus by a **relative 7.5×10²**. It would not have raised; it would have
trained and produced plausible-looking numbers.

A second consequence: in pu the SI path's Vbase re-referencing
(`y_ft = y_from·Vf/Vt`) collapses to an identity, `y_ft_pu == y_from_pu` and
`y_tf_pu == y_to_pu`. So use the from/to series values directly.

### The correct construction (already implemented)

`gridfm_graphkit_adapter.graphkit_branch_edges_parquet(batch, device)`, with
`a = tau·exp(j·shift_rad)`, all values already pu, `Branch_status == 0` dropped:

```
i→j edge:  YFF_TT = (y_from + ysh_f/2) / |a|²      YFT_TF = -y_from / conj(a)
j→i edge:  YFF_TT = (y_to   + ysh_t/2)             YFT_TF = -y_to   / a
```

### Verify before you train

`validate_edge_attr_parquet(batch)` re-assembles a Y-bus from the split and
compares it against the batch's own. Run it once per grid family:

```python
from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from gridfm_graphkit_adapter import validate_edge_attr_parquet

ds = ChanghunDataset(PARQUET, per_unit=True, target_S_base=1e8, share_grid=True,
                     share_ybus=True, lazy_row_groups=True,
                     row_group_cache_size=2, complex_dtype="complex128")
b = collate_blockdiag([ds[i] for i in range(4)])
print(validate_edge_attr_parquet(b))   # expect rel_err ~1e-16, ok=True
```

Confirmed passing at machine precision on case118, case300, GBnetwork and
SimBench (`rel_err` 1.4e-16 … 2.6e-16). If a new grid family fails, the split is
wrong for that topology — fix it before training rather than after.

---

## 4. What still has to be written

The adapter currently wires the **OPFData** pipeline end to end. For the parquet
PF path these pieces exist and these do not:

| piece | status |
|---|---|
| `build_graphkit_model(task_name="PowerFlow", ...)` | ready |
| `graphkit_branch_edges_parquet` + validator | ready, verified |
| bus/gen feature builders | ready — reuse `to_gridfm_inputs`'s parquet counterpart in `train_valid_test_gridfm.py` |
| `mask_dict` for PF | **needs writing** (see below) |
| batch assembly + `forward` for parquet | **needs writing** — mirror `to_graphkit_batch` / `forward_graphkit`, swapping the edge builder |
| `--gridfm_impl graphkit` in `train_valid_test_gridfm.py` | **needs writing** — copy the two hooks already added to `train_valid_test_opfdata.py` |

### `mask_dict` for power flow

Two different things share the name, and both are required:

**(a) `mask_dict["bus"]` / `["gen"]`** — boolean, in the *feature* layout, marking
what the model must predict rather than copy from its input. `forward` reads
`mask_dict["bus"][:, VM_H:VA_H+1]` and `mask_dict["gen"][:, :PG_H+1]`. For power
flow the unknowns follow bus type:

```python
m = torch.zeros((n, 15), dtype=torch.bool, device=dev)
m[:, VM_H] = is_pq                  # |V| known at PV and slack
m[:, VA_H] = ~is_ref                # angle known only at slack
```

`_bus_mask(x_bus, "PowerFlow")` in the adapter already does this — it is written
against the bus-type one-hots (`PQ_H`, `PV_H`, `REF_H`), so it works for either
pipeline.

**(b) `mask_dict["PV"]`, `["REF"]`** — plain per-bus booleans read by the physics
decoder to recover Qg where it is not free. Omitting them fails with
`KeyError: 'PV'` at the first forward, which is how it was found.

---

## 5. Running it

Once the parquet forward is wired, the command is the existing PF command plus
one flag. `--task pf` is already the default of `train_valid_test_gridfm.py`:

```bash
srun python -u train_valid_test_gridfm.py \
  --PARQUET "${LOCAL_PARQUET}" --task pf --gridfm_impl graphkit \
  --run_name pf_gk_case118 \
  --log_to_file --log_dir results/logs/${GROUP} --ckpt_dir "${LOCAL_CKPT_DIR}" \
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 2 \
  --dataset_complex_dtype complex128 \
  --BATCH 8 --EPOCHS 40 --LR 5e-4 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh --VAL_EVERY 1 \
  --hidden_size 48 --num_layers 12 --n_heads 8
```

Keep `--hidden_size 48 --num_layers 12 --n_heads 8` if you want the released
config; the model ignores `--zero_init_head`, `--vn_feature_mode` and
`--feature_transform` choices that only the mirror's head understands.

---

## 6. Two things that will bite you

**Row groups.** Most `*_NR_branchrows_directSI.parquet` files were never
re-laid-out: case14/118/300 use **6000** rows per group and GBnetwork **3000**,
against 20 for the OPF files. Training shuffles, so each batch decodes a whole
group — measured at 3191 ms/batch versus 176 ms/batch, an **18× penalty**. Run
`rewrite_rowgroups.py` over the target files first; across ~36 grids this
dominates everything else. (`SimBench_ppcY_backbone_*` is already at 20.)

**The released model is slower per step.** It runs a branch-flow, node-injection
and physics-MLP pass at every one of the 12 layers, which the mirror does not.
Budget for it, and note that these workloads are already CPU-bound — GPU
utilisation was 23–34% on H100 for the existing runs, so the extra cost lands on
wall clock rather than on the GPU.

---

## 7. Files

| file | role |
|---|---|
| `gridfm_graphkit_adapter.py` | model construction, both edge builders, both validators, OPFData batch/forward |
| `train_valid_test_opfdata.py` | reference wiring — see the `gridfm_impl == "graphkit"` branches in `build_model` and `forward` |
| `train_valid_test_gridfm.py` | the parquet driver you are extending |
| `Dataset_optimized_complex_columns.py` | `reconstruct_Y_pandapower_branchrows_direct_SI_np` and `ybus_si_to_pu` — the ground truth for the admittance convention |

`~/PIGNN-Attn-LS/gridfm-graphkit` is installed `--no-deps` on both alex and
helma, together with `opt_einsum`, `lightning`, `matplotlib` and `torch_scatter`
which its import chain needs.

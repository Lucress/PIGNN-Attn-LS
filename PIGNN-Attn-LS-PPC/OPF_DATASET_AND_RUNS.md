# AC-OPF datasets and surrogate runs (SimBench, August 2026)

This is the OPF counterpart of `GRIDFM_GRIDSFM_WINDOWS_GPU_RUN.md`. Datasets are
generated on **fritz** (CPU, pandapower), the three surrogates are trained on
**alex** (GPU), and the data pipeline is otherwise unchanged: the same PPC
branch-row direct-SI parquet, the same loader, the same 100 MVA rebase, the same
complex128 arithmetic.

## 1. What the task is

The power-flow datasets give a model the full injection vector and ask for the
voltages that balance it. The OPF task gives it only the **loads** plus the
**decision space** (generation limits, available renewable power, cost slope,
voltage band, branch ratings) and asks for the **OPF operating point**.

A voltage-only head is still enough. Once the voltage profile is predicted, the
dispatch follows from the network equation

```
S_implied = V * conj(Y V)
```

so all three surrogates keep their existing per-bus voltage heads and none of
them needs a new output layer.

What does change is what a metric can mean per bus:

| Bus kind | Injection | Meaningful quantity |
|---|---|---|
| No controllable generation | known (the load) | power-balance residual — a real error |
| Controllable | free within limits | how far the implied dispatch leaves its box |

Reporting a plain power-balance residual over all buses on OPF data would count
**correct redispatch as error**. `opf_task.py` masks by `Gen_controllable`
accordingly and adds voltage-band violation and a cost gap.

## 2. How the OPF problem is defined

The CGMES nets carry **no cost data, no voltage band, and no thermal limits**, so
the optimisation problem has to be constructed. `main_datagen_opf_snapshot_envelope.py`
builds a *curtailment OPF*:

| Element | Treatment |
|---|---|
| Bus voltage | band `--vmin` / `--vmax`, default 0.9 / 1.1 |
| Lines, transformers | `--max_loading_percent`, default 100 |
| Static generators (wind, storage) | dispatchable in `[0, P_available]` — may be curtailed, never increased |
| sgen reactive range | `± max(--sgen_q_frac * P_avail, 1 MVAr)` |
| Synchronous generators | dispatchable over their existing limits |
| External grids | free to import and export |
| Objective | linear: `--sgen_cost`, `--gen_cost`, `--ext_grid_cost` |

**Why not hold the static generators at their snapshot values?** Because that
problem is infeasible on this data. The SimBench snapshots carry ~3.7 GW of
generation against ~0.1 GW of load, individual units sit above their own
`max_p_mw` (356.7 vs 149.1 MW), and the fixed dispatch cannot be pushed through
a network dimensioned for that load without breaching thermal limits. Tested
directly: fixed dispatch fails whether or not the external grid may absorb;
allowing curtailment converges in ~0.6 s.

## 3. The renewable envelope is mandatory

`build_snapshot_load_bounds` scans the snapshots for **load** bounds only, so
every dataset built before this one varies the loads and pins generation to
whatever the base snapshot held. For a power flow that is harmless. For an OPF
it is fatal, because the SimBench base export is a **maximum-wind** snapshot.

Measured on a 24-scenario probe with availability held fixed:

| Quantity | Variation across scenarios |
|---|---|
| \|V\| target std | 1.59e-4 pu — **0.27x** the error GridFM already achieves on PF data |
| theta target std | 8.9e-4 rad |
| OPF cost | std 12 EUR on -108,800 EUR (0.01%) |
| Curtailment | 1342-1394 MW |

A model that ignored its input and predicted the mean profile would have scored
better than GridFM does on power flow. Sampling availability from its own
snapshot envelope fixes this:

| Quantity | fixed availability | sampled availability |
|---|---:|---:|
| \|V\| target std | 1.59e-4 pu (0.27x) | **4.37e-2 pu (72.8x)** |
| theta target std | 8.9e-4 rad (2.5x) | **0.115 rad (330x)** |
| Cost std | 12 EUR | **37,175 EUR** |
| Curtailment range | 1342-1394 MW | **0-1723 MW** |
| \|V\| range | 1.069-1.100 | **0.900-1.100** (full band) |

Always pass `--sgen_bounds_path`. Without it the generator prints a warning and
produces a near-constant dataset.

```bash
# one-off, ~11 min on 72 cores for 39,268 snapshots
sbatch sbatch/SimBench_sgen_bounds.sh
# -> SimBench_snapshot_sgen_bounds_cache.npz (121 units, 1.9-4291.4 MW)
```

## 4. Generating the dataset (fritz)

```bash
sbatch sbatch/SimBench_opf_36000.sh
```

~36 rows/s on 72 cores, so 36,000 scenarios take about 17 minutes.

### Keep the row groups small

Training shuffles, so parquet row-group size decides the random-access cost.
The power-flow parquets use **20 rows per group**. Writing one row group per
flush instead (500 rows) makes every shuffled batch of 4 decode a whole group,
and the extra OPF columns make each decode heavier. Measured on identical data,
only the layout differing:

| Row groups | ms/batch (shuffled) | min/epoch |
|---|---:|---:|
| 500 rows | 3,191 | 159.6 |
| 20 rows | **176** | **8.8** |

An 18x penalty — the first dispatch ran at 2.7 h/epoch and its zero-shot jobs
hit the 4 h wall without finishing one evaluation pass. `--row_group_size`
defaults to 20; do not raise it. Profiling confirmed the cost was purely
layout: `opf_loss` 0.62 ms, `opf_metrics` 1.51 ms, `opf_decision_space` 0.23 ms,
graph construction 5.4 ms, full LUMINA forward 32 ms per batch.

An existing parquet can be re-laid-out without recomputing any OPF solve:

```bash
python rewrite_rowgroups.py <src.parquet> <dst.parquet> 20
```

### Stored supervision and the noise floor

`u_newton` holds the **raw `runopp` voltage solution**, so the target is exactly
the pandapower OPF answer. That solution satisfies the power-flow equations only
to the OPF solver's tolerance. Measured over the full 36,000-row dataset on a
100 MVA base, against 1e-13 pu for the Newton targets of the power-flow
datasets:

| | median | p99 | max |
|---|---:|---:|---:|
| `opf_resid_dp_pu` | 7.1e-7 | 5.3e-5 | 2.0e-4 |
| `opf_resid_dq_pu` | 1.2e-6 | 1.1e-4 | 3.7e-4 |

Each row stores its own residual so this floor is measurable rather than
implicit. It sits well below the residuals the surrogates reach (GridFM's best
power-flow result was 1.4e-2 pu), but a model should not be credited with
beating it.

Polishing the target with a power flow at the fixed OPF dispatch was tested and
rejected: it reaches 9e-10 pu but moves the operating point by 9e-3 pu, which is
larger than a good model's error, so the target would no longer be the `runopp`
answer.

### Schema additions

The parquet keeps every power-flow column, so `ChanghunDataset` reads it
unchanged, and adds:

| Column | Meaning |
|---|---|
| `S_start` | **loads only**, zero at generator buses — the model input |
| `S_newton` | full OPF injection (loads + dispatch) |
| `S_load`, `S_gen_opf` | the two parts separately |
| `Gen_p_min/max`, `Gen_q_min/max` | per-bus generation limits (SI) |
| `Gen_p_avail` | available renewable power per bus (SI) |
| `Gen_cost_c1` | cost slope per bus |
| `Gen_controllable` | 1 where the dispatch is free |
| `Bus_vmin`, `Bus_vmax` | voltage band |
| `Branch_rate_a` | thermal rating (SI VA), mapped row-for-row via pandapower's branch lookup |
| `opf_cost`, `opf_curtailed_mw`, `opf_resid_dp_pu`, `opf_resid_dq_pu` | per-row diagnostics |

These are read through the opt-in `extra_binary_columns` argument, so power-flow
parquets are unaffected — verified bit-identical with and without it.

## 5. Training the three surrogates (alex)

```bash
bash dispatch_opf_simbench_alex.sh
```

Seven jobs: GridFM scratch, and zero-shot / fine-tuned / scratch for each of
GridSFM and LUMINA. Every script takes `--task opf`; the decision space enters
each model through slots its schema already had, several of which carried
nothing until now:

| Model | Decision space goes into |
|---|---|
| LUMINA | `generator.x` pmin/pmax/qmin/qmax + cost c1 (its native ACOPF schema) |
| GridSFM | same hetero schema, same mapping |
| GridFM | bus `MIN_VM/MAX_VM/MIN_QG/MAX_QG`, gen `MIN_PG/MAX_PG/C1` |

Two details of the adapters worth knowing:

* **The voltage band must reach the model.** LUMINA bounds its magnitude head
  with a sigmoid mapped onto `[vmin, vmax]` *read from its bus features*. Feeding
  the CLI default (0.5-1.5) instead of the dataset band (0.9-1.1) makes it
  predict into an interval five times too wide, and it then violates the band at
  ~99% of buses while its voltage RMSE looks normal. All three adapters now take
  `Bus_vmin`/`Bus_vmax` from the dataset in OPF mode. If a comparison ever shows
  one model with a wildly worse band/cost metric but a *similar* voltage RMSE,
  suspect this wiring before concluding anything about the model.
* **Branch ratings are not fed to the models.** `Branch_rate_a` is stored in the
  dataset but the adapters still pass the scalar `--rate_a` into the edge
  attributes. This costs nothing in fairness terms -- thermal violation is not
  among the scored metrics and all three models lack the feature equally -- but
  it does mean none of them is told the thermal limits.

Reading the log line:

```
balance(dPinf ..., dQinf ... pu over N free buses)   equality error where the injection is known
limits(P ..., Q ... pu over M ctrl buses)            how far the implied dispatch leaves its box
vband(max ..., frac ...)                             voltage-band violation
cost(gap ...%)                                       dispatch cost vs the OPF optimum
```

`summarize_model_logs.py` parses the shared columns of these logs as before.

## 6. GridSFM does not train on the larger grids

GridSFM's numbers are only valid on case14. On SimBench, case118 and case300 it
stops learning, so its rows there measure nothing and must not be quoted as its
capability.

The evidence is unambiguous. On case118 the *training* loss is bit-identical
across epochs (3.9576e+00 at epochs 1 through 6 and beyond, with mse 3.9440e+00
and phys 1.3572e+00 likewise), which is impossible under shuffled batches unless
the weights are not moving. A single-batch gradient probe confirms it:

```
params total=15,148,227   requires_grad=15,148,227
Vpred requires_grad=True  grad_fn=UnsqueezeBackward0
loss=6.297e-01 (finite)
gradients: none=16  exactly_zero=1205  nonzero=0
```

Not one parameter receives a nonzero gradient, while the loss is finite and the
autograd graph is connected. Exact zeros everywhere is the signature of a
saturated hard clamp rather than vanishing gradients.

The cause is **not** the voltage band. That was the obvious suspect, since the
OPF adapters feed the dataset's narrow band (0.9-1.1) where the power-flow runs
used the wide CLI default (0.5-1.5). Tested directly on case118:

| bus band supplied | predicted \|V\| range | nonzero grads |
|---|---|---|
| dataset (0.9, 1.1) | [0.9000, 0.9000] | 0 |
| wide (0.5, 1.5) | [0.5000, 0.5000] | 0 |

The magnitude head collapses onto **vmin whatever the band is**, and the clamp's
zero local gradient then locks the whole network. It is a dead-unit pathology:
on the larger grids the model initialises into the saturated region and cannot
leave it. On case14 the initialisation lands inside the band, which is why it
trains there normally.

Both `--init_mode scratch` and `--init_mode pretrained` freeze, so warm-starting
does not avoid it. Fixing it means changing how GridSFM's magnitude output is
produced (a soft bound rather than a clamp, or an output offset that starts
inside the band), which is a change to the surrogate rather than to the adapter.

## 7. LVN Heo1 status: blocked

`pp.runopp` does not converge on the LVN net, and this is **not** a modelling
choice that can be tuned away. Ruled out by direct test:

| Hypothesis | Test | Result |
|---|---|---|
| Constraints infeasible | vm in [0, 2], no thermal limits at all | still fails |
| Zero-impedance branches | 2 found, both already out of service; min-z / switch / drop repairs | all fail |
| Bus count / switch conditioning | fused 3,782 -> 1,182 buses | still fails |
| Bad initialization | `init="flat"`, `init="pf"` | both fail |

Related symptom: pandapower's plain `runpp` only converges on this net at
`tolerance_mva >= 1e-7`, failing at the 1e-8 default, which is why the project
uses its own damped Newton-Raphson. The net is marginally convergent and the
interior-point OPF is far less forgiving.

Two routes remain, both larger than a configuration change:

1. a more robust backend — `pp.runpm` (PowerModels.jl/IPOPT), which needs Julia
   installed on fritz; there is none in `PATH` and `to_pm` is unavailable;
2. genuine model repair of the CGMES import.

Note also that the LVN pandapower net has 3,782 buses; the 722 in the
power-flow datasets is the *ppc-internal* count after switch fusion.


## 11. Second pipeline: DeepMind's OPFData (native)

`opfdata_pipeline.py` + `train_valid_test_opfdata.py` + `dispatch_opfdata_alex.sh`
run the same three models on [OPFData](https://arxiv.org/abs/2406.07234). The
pandapower pipeline is untouched; this is an independent second track.

**Why it exists.** Every pandapower result passes through schema adapters written
here, and two adapter bugs were found during that work (the voltage band, the
angle datum). LUMINA and GridSFM consume OPFData's `HeteroData` *natively*, so on
this track only GridFM is translated. case14 and case118 exist in both pipelines,
which makes the generation methodology the controlled variable.

### How the two pipelines differ

| | pandapower | OPFData |
|---|---|---|
| Scenarios | sampled here from CGMES snapshot envelopes | pre-generated from PGLib base cases |
| Perturbation | coupled load scaling, measured **r = +0.835**, total load varies 23.6% | independent per-load ±20%, measured **r = −0.001**, total varies 5.2%; optional N-1 topology |
| Solver | pandapower PIPS | PowerModels / IPOPT |
| Target exactness | 5.6e-7 (SimBench) to 1e-14 (backbone) | 1.3e-6 |
| Adapters | three (all written here) | one (GridFM only) |
| Decision space | reconstructed; limits from an empirical envelope | native and exact (case14 band is [0.94, 1.06]) |
| Supervision available | bus voltages | voltages + dispatch + branch flows + objective |
| Grids | case14/118/300, SimBench, LVN | 10 PGLib scenarios, 14 to 13,659 buses (no case300) |

The perturbation difference is the substantive one: the two explore **orthogonal
directions**. Independent perturbation varies *which* buses are loaded at nearly
constant total; coupled scaling varies the *total* at nearly constant pattern.
They stress different constraints, so results on the same grid can legitimately
differ.

### Validation

The Y-bus is rebuilt from `edge_attr` (needed for the residual metrics) and the
whole chain was checked against OPFData's own stored quantities:

| Check | Result |
|---|---|
| Y-bus vs stored solution | max\|ΔP\| 1.3e-6 pu (the data's own solver tolerance) |
| Recovered dispatch vs stored `generator.y` | 5.5e-7 pu |
| Limit violation at ground truth | 5.6e-7 (feasible by construction) |
| Cost model vs stored IPOPT objective | ratio **1.000000**, std 2.4e-7 |

The last is the strongest check in the project: Y-bus → implied dispatch →
quadratic cost reproduces DeepMind's objective to seven figures.

### Running

```bash
# one-off: fetch and process (case14 ~1.1 GB raw, case118 ~7.3 GB)
sbatch prep_opfdata.sh
# 7 jobs per case: GridFM scratch + zero-shot/fine-tune/scratch for GridSFM and LUMINA
CASE=both bash dispatch_opfdata_alex.sh
```

`--dispatch_weight` and `--gen_head_weight` default to 0 so the comparison is
like-for-like with the parquet runs; raise them for the dispatch-supervised
variant.

### GridSFM trains here

The finding that matters most so far: **GridSFM trains normally on OPFData**
(one-epoch smoke test, pretrained: RMSE 1.60e-2, θ 0.50°, cost gap 2.96%), while
on the pandapower pipeline it was frozen with zero gradients for 29 epochs
(Section 6). Its native path works. That points the earlier failure at the
parquet adapter rather than at the model, and means the clamp-saturation
hypothesis in Section 6 is more likely a symptom of adapter-induced input scaling
than an intrinsic property of GridSFM.

## 12. `S_demand` semantics and the generation box (important)

`S_demand` in the pandapower ACOPF-backbone parquets is **not the load**. It is the
pre-OPF injection, i.e. base generation minus the scenario load. Verified on
case118: at pure-generator buses it equals the base setpoint exactly (bus 9:
450.00 MW, bus 24: 220.00 MW, bus 25: 314.00 MW).

Consequences:

* `S_opf - S_demand` is a **redispatch delta**, not a generation. On case118 the
  slack absorbs a system-wide correction of about -2271 MW, so an envelope built
  on that delta produces a generation box centred at -28 pu. GridSFM builds its DC
  angle prior from the box midpoint, so this froze it completely (gradient norm
  exactly 0 at every step).
* The load cannot be recovered on case118: its stored base generation does not
  match stock pandapower, so it cannot be subtracted. case14 does recover cleanly.

`convert_opf_backbone_parquet.py --gen_box injection` (the default) therefore
builds the envelope on `S_opf` itself, which is observable and physically bounded,
while **controllability stays decided by the delta** -- a bus is controllable iff
the OPF actually moves its injection. Testing `|S_opf|` instead would flag every
load bus and turn its power-balance equality into a free variable.

The flag rides to the models as the per-bus column `Gen_box_injection`, read by
`opf_task.opf_decision_space` into `space["gen_box_on_injection"]`; when set,
`opf_residual_and_violation` bounds `S_implied` and `target_dispatch` references
`S_newton`. Absent (all power-flow parquets, OPFData) the old delta convention
applies unchanged, so OPFData runs are unaffected.

Converted files: `{case14,case118,case300}_opf_task_ready_injbox.parquet`.

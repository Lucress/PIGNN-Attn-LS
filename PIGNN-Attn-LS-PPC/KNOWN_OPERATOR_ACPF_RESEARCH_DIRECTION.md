# Known-Operator Learning for Certified AC Power Flow

## Recommended research claim

The strongest defensible direction is:

> A graph surrogate predicts a high-quality AC power-flow warm start. A shared,
> non-learned, differentiable feasibility operator then applies the exact AC
> equations and returns both a corrected voltage and an a-posteriori residual
> certificate. The learned component is evaluated by how much solver work it
> removes; physical feasibility is claimed only for samples that meet the
> declared tolerance.

This is more precise than claiming that a physics loss or a finite correction
block guarantees feasibility. It is also a genuinely hybrid solver: the final
inference path evaluates and optimizes the real AC equations rather than only
using them to regularize learned parameters.

Suggested paper title:

**Known-Operator Learning for Certified AC Power Flow: Graph Neural Warm Starts
with a Differentiable Feasibility Layer**

## What the source papers support

### Known Operator Learning (KOL)

The KOL review makes the distinction that matters here. Physics in a loss is a
training-time regularizer and has no test-time plausibility mechanism. Known
operator learning instead places an analytical operation in the computational
graph. It motivates lower approximation burden, fewer trainable parameters,
and better generalization, but it does not by itself prove that every output of
an arbitrary hybrid network is feasible.

### PIGNN-Attn-LS

The ICASSP architecture already contains important known structure:

- branch admittance and transformer parameters enter direction-aware attention;
- power mismatch is evaluated inside message-passing iterations;
- the Armijo mechanism rejects or shrinks learned updates that do not reduce a
  mismatch merit function;
- slack/PV/PQ update masks preserve specified power-flow variables.

This is stronger than a conventional PINN, but the update direction is still
learned. Armijo gives a local decrease test, not convergence to a root, and a
finite number of learned steps can leave a nonzero residual.

### Differentiable Power-Flow (DPF)

DPF minimizes the masked AC mismatch by first-order optimization. For squared
loss, its update is proportional to `-J^T F`, whereas Newton-Raphson uses a
linear solve with `-J^-1 F`. DPF avoids explicit Jacobian factorization and can
use sparse GPU batching.

The paper's experimental result must be represented accurately: its base DPF
needs roughly 1,000 iterations, remains slower than optimized NR on available
grids up to 9,241 buses, and reaches solution quality between NR and the DC
approximation. Its scaling advantage is projected for much larger systems and
batched/time-series use. Therefore, DPF alone is not evidence of a faster or
more accurate operational solver. The research opportunity is to test whether
the learned graph warm start removes enough of those iterations to change that
conclusion.

### Existing GridFM mirror results

The local GridFM-style report gives a strong motivation for the hybrid layer.
Voltage RMSE and power-balance residual rank grids differently, and several
large or ill-conditioned families have extreme heavy-tailed residuals. This is
exactly the failure mode that an inference-time feasibility layer should target.
The report also warns that its numbers are for the local mirror, not the
released `gridfm_graphkit` implementation, so both must remain separate rows in
the experiment table.

## Proposed architecture

For any backbone `G_phi`, let

```text
V_raw = G_phi(grid, injections, initial_state)
V_K   = DPF_K(Ybus, Sset, bus_types, V_raw)
```

The known operator uses the actual equations

```text
S_calc(V) = V * conj(Ybus V)
F(V) = concat(
    P_calc - P_set on PV and PQ buses,
    Q_calc - Q_set on PQ buses,
)
L_pf(V) = 0.5 * mean(F(V)^2).
```

Slack voltage and PV voltage magnitudes are copied from the specified boundary
state and never optimized. PQ magnitudes and all non-slack angles are free.

The implementation uses the exact sparse adjoint

```text
g_V = w * (Ybus V) + Ybus^H (conj(w) * V),
```

where `w` contains the masked and normalized P/Q residual weights. Projection
onto `dV/d|V|` and `dV/dtheta` gives `J^T F` without constructing `J`. Adam or
gradient descent then updates the free polar variables.

This gives two deliberately different inference modes:

1. **Fixed K-step KOL:** differentiable and fast, intended for training and a
   fixed-latency ablation. It should reduce residuals, but is not certified.
2. **Adaptive certified KOL:** runs up to `K_max`, stops each graph when
   `max(|Delta P|, |Delta Q|) <= tolerance`, and exposes a convergence flag.
   Only successful samples are described as physically feasible to that
   numerical tolerance.

## Why an auxiliary raw loss is necessary

If the inner PF solver converges exactly to a locally unique solution, that
solution no longer depends on its initialization. Consequently, the derivative
of the converged voltage with respect to the warm start approaches zero. A loss
applied only after a fully converged inner solver cannot reliably teach the GNN
to produce a better initializer.

Use a joint objective such as

```text
L = lambda_raw * L_supervised(V_raw, V_star)
  + lambda_raw_phys * L_pf(V_raw)
  + lambda_K * L_supervised(V_K, V_star)
  + lambda_K_phys * L_pf(V_K).
```

The current scripts expose `--kol_raw_loss_weight` for this reason. The fixed
training unroll supplies an additional algorithm-aware gradient: it rewards a
raw prediction from which a small number of physical steps works well.

## Core hypotheses

1. A graph warm start lowers DPF iterations-to-tolerance relative to flat and
   DC starts.
2. A small fixed DPF block lowers median and tail P/Q residuals for every
   backbone, even when voltage RMSE changes little.
3. Adaptive refinement produces a high certification rate on in-distribution
   cases and exposes failures rather than hiding them behind mean RMSE.
4. The gain is largest on the hard grid families identified by the GridFM
   report, but those grids may also require per-family inner learning rates.
5. The same non-learned operator narrows differences between backbone output
   residuals, while iteration count and runtime still reveal which backbone is
   the best initializer.

## Required experiment matrix

### Backbones

- PIGNN-Attn-LS (`GNSMsg_SelfAttention_armijo.py`)
- GridFM local mirror
- released `gridfm_graphkit`
- released GridSFM
- released LUMINA

Keep mirror/graphkit results separate. Also distinguish scratch, pretrained,
and fine-tuned GridSFM/LUMINA runs.

### Solver variants

For each backbone:

| Variant | Training path | Inference path | Valid claim |
|---|---|---|---|
| Raw | surrogate | surrogate | learned approximation |
| PINN/Armijo | existing physics loss/operator | finite learned steps | residual-aware surrogate |
| KOL-1 | 1 differentiable DPF step | 1 DPF step | fixed physical correction |
| KOL-3 | 3 differentiable DPF steps | 3 DPF steps | fixed physical correction |
| KOL-adaptive | KOL-1/3 training | DPF until tolerance or `K_max` | certified only on converged cases |
| Warm-NR | surrogate | NR until tolerance | operational hybrid baseline |

Include standalone DPF from flat and DC starts, plus standard NR from flat/DC,
so a reviewer can tell whether the neural initializer provides value beyond the
known solver itself.

### Data sequence

1. Debug and tune on case14 and case118.
2. Confirm scaling on case300, case1354pegase, and case9241pegase.
3. Stress the hard residual families from the GridFM report: case145,
   GBnetwork, GBreducednetwork, iceland, RTE, PEGASE, and LVN_heo1.
4. Add load-range OOD tests and, where the backbone permits it, topology or
   contingency OOD tests.
5. Use identical train/validation/test indices across all backbones and
   correction variants.

### Metrics that must be reported

- raw and corrected voltage-magnitude RMSE;
- raw and corrected wrapped-angle RMSE;
- per-case `max |Delta P|`, `max |Delta Q|`, mean, median, p95, p99, and RMSE;
- certification rate at `1e-4`, `1e-6`, and `1e-8` pu;
- inner iterations to tolerance, including failures at `K_max`;
- wall-clock latency and throughput including the correction layer;
- peak GPU memory during training and inference;
- fixed-variable errors for slack voltage/angle and PV magnitude;
- comparison with warm-start NR iterations and runtime;
- results split by grid family and operating-stress quantile.

Always report voltage error and physical residual separately. A low voltage
RMSE is not a feasibility certificate.

## Claim boundaries

Do not claim unconditional physical-law satisfaction. The safe wording is:

> The hybrid returns an a-posteriori feasibility certificate when its masked
> AC power mismatch is below the declared tolerance; otherwise it returns an
> explicit non-convergence flag and can fall back to NR.

The certificate does not establish global uniqueness, voltage stability,
thermal security, generator-limit feasibility, or N-1 security. It only
certifies the specified PF equations and fixed bus-type constraints at the
reported numerical tolerance.

Do not apply the PF operator directly to the repository's OPF task. In OPF,
generator injections are decision variables; treating the current `S_start` as
fully specified would erase the dispatch problem. A future differentiable OPF
layer needs KKT/constraint handling and is a separate paper contribution.

## Implementation map

- `known_operator_pf.py`: shared differentiable PF feasibility operator,
  exact sparse adjoint, CLI options, and certificate diagnostics.
- `train_valid_test.py`: PIGNN-Attn-LS integration.
- `train_valid_test_gridfm.py`: GridFM mirror and graphkit integration.
- `train_valid_test_gridsfm.py`: GridSFM integration.
- `train_valid_test_lumina.py`: LUMINA integration.
- `run_known_operator_pf_experiments.py`: reproducible raw/KOL experiment
  command matrix.
- `tests/test_known_operator_pf.py`: equation-mask, sparse-gradient,
  constraint-preservation, convergence, and gradient-flow tests.

Example dry run:

```bash
python run_known_operator_pf_experiments.py \
  --parquet /path/to/case118_ppNR_branchrows_directSI.parquet \
  --backbones pignn gridfm-mirror gridfm-graphkit gridsfm lumina
```

Execute only after reviewing the generated commands and providing released
checkpoint/config paths where required:

```bash
python run_known_operator_pf_experiments.py \
  --parquet /path/to/case118_ppNR_branchrows_directSI.parquet \
  --gridsfm-checkpoint /path/to/gridsfm_open_v1.1.pt \
  --lumina-checkpoint /path/to/lumina_model.safetensors \
  --lumina-config /path/to/lumina_config.json \
  --execute
```

## Go/no-go criteria after the pilot

Proceed to a full paper-scale sweep only if the case14/case118 pilot shows all
of the following:

- the analytic `J^T F` test agrees with automatic differentiation;
- fixed slack/PV quantities remain exact;
- KOL-1 or KOL-3 improves residual tails without destabilizing training;
- the adaptive mode reports failures correctly and achieves a materially
  higher certification rate than raw surrogates;
- a learned warm start reduces DPF or NR work compared with flat/DC starts;
- total hybrid latency is measured, not inferred from iteration count alone.

If DPF remains too slow or fails to certify hard cases, retain it as the
differentiable training operator and use warm-start NR as the operational
certification/fallback path. That result would still be scientifically useful:
it would identify where differentiable first-order physics helps learning and
where a second-order solver remains necessary.

# Audit against the differentiable-power-flow author repository

Audited source: `../differentiable-power-flow/src/dpf/solvers/solver_torch.py`,
the dense equivalent, batched/time-series variants, and the experiment
hyperparameters under `src/dpf/scripts/`.

## Exact correspondences

Both implementations now use

```text
S_calc = V * conj(Ybus @ V)
```

and pack the DPF objective exactly as the author does:

```text
P mismatch: PV and PQ buses
Q mismatch: PQ buses only
loss: torch.nn.MSELoss(packed_calculated, packed_specified)
```

The optimized variables also match:

```text
voltage angle: PV and PQ buses
voltage magnitude: PQ buses only
slack magnitude/angle: fixed
PV magnitude: fixed
```

The reference Adam preset is

```text
lr = 0.003377
betas = (0.979681, 0.963442)
ReduceLROnPlateau(
    factor=0.547191,
    patience=41,
    threshold_mode="rel",
    threshold=0.067321,
    cooldown=97,
)
```

The author uses `lr=1e-4` in several very-large-grid experiments. Both the
standalone layer and integrated model expose the learning rate explicitly.

## Defects found and corrected

1. The local loss was half the author's MSE, making its gradient smaller by a
   factor of two. It now matches `torch.nn.MSELoss` exactly.
2. The local adaptive evaluator omitted the author's plateau scheduler. With a
   constant Adam learning rate it could stall above a tight residual tolerance.
   The author's scheduler semantics are now implemented for evaluation.
3. A converged graph in a block-diagonal Adam batch could still move because
   old momentum was not frozen. Both the gradient and final state update are
   now masked for certified graphs.
4. The integrated PIGNN normalized gradients over the entire block-diagonal
   mega-graph. One difficult scenario could therefore scale every other
   scenario. Gradient features are now normalized independently per graph.
5. The integrated line search selected one step for the whole mega-graph. Step
   acceptance/rejection is now independent per graph.

## Intentional extensions—not claims of verbatim reproduction

The standalone `DifferentiablePowerFlow` is a sparse, packed-batch adaptation
of the author's optimizer. It uses an infinity-norm AC residual for the final
certificate rather than treating the author's MSE stopping criterion as a
physical certificate.

The integrated model is a new method, not code supplied by the DPF authors. At
iteration `k`, it computes the author's exact DPF MSE gradient and Adam
direction, then PIGNN predicts positive coordinate-wise multipliers:

```text
d_adam[k] = AdamDirection(gradient(MSE_AC(x[k])))
d[k]      = positive_PIGNN_scale[k] * d_adam[k]
x[k+1]    = projected_residual_safeguarded_step(x[k], -d[k])
```

Consequently, the correct description is **PIGNN-modulated DPF**, not an exact
copy of the author's solver. The AC equations, variable partition, objective,
and base optimizer are author-compatible; the graph-conditioned
preconditioning and per-grid safeguard are the research contribution.

## Verification

The numerical suite checks:

- exact equality to the author's packed MSE objective;
- analytic sparse-adjoint gradients against PyTorch autograd;
- one optimizer step against `torch.optim.Adam` with the author's preset;
- hard slack/PV constraints;
- differentiability back to the surrogate output;
- independent stopping of packed block-diagonal grids;
- residual certification and sparse execution.

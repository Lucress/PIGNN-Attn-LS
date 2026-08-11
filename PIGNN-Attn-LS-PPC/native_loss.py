"""Train each surrogate with the loss function from its own repository.

The shared objective used elsewhere in this project supervises bus voltages
only. That keeps the three models comparable but is not architecture neutral:
all three natively supervise generation as well, and GridSFM additionally
supervises cost, branch flows and thermal limits. This module lets each model
be trained with its own loss instead, so the comparison becomes "each model at
its best, evaluated on a common metric" rather than "all models on one
objective".

Evaluation is unchanged in either mode -- `opf_task.opf_metrics` still scores
every run -- so the runs remain comparable even though the training objectives
differ. Loss *values* are not comparable across models here and should never be
tabulated side by side.

Provenance and fidelity of each adapter:

* **GridSFM** -- fully faithful. Calls `gridsfm.loss.compute_loss` directly:
  tanh-capped squared error on theta/V/Pg/Qg, BCE on the feasibility logit,
  log-MSE on cost, log1p-compressed KCL P/Q, log1p branch flow P/Q, thermal
  loading, a thermal-limit barrier and a stress-feasibility regulariser.
  OPFData ships solved instances only, so there is no feasibility label to
  learn: `lambda_feas` and `lambda_stress_feas` default to 0 here and
  `batch.feasible` is set to all-ones. Every other term is live, because
  OPFData supplies `edge_label` (Pij, Qij, Pji, Qji) and `rate_a`.

* **LUMINA** -- fully faithful. Uses `lumina.model.opf.losses.ACOPFLossFunction`
  from `lumina-sdk`, which weights a per-node-type regression loss over `bus`
  and `generator` (default weights 1.0/1.0). `PhysicsInformedLoss` is available
  and takes our violation computation as its `constraint_computer`.

* **GridFM** -- partially faithful, and the reason is worth stating. The
  "GridFM" in this project is a local re-implementation
  (`train_valid_test_gridfm.GridFMHeteroSurrogate`), not the released model, so
  there is no native training recipe attached to a released checkpoint. We use
  `gridfm_graphkit.training.loss.MaskedReconstructionMSE`, the substantive part
  of their objective: a masked MSE over [VM, VA, PG, QG, PD, QD] rather than
  over voltages alone. Their `PBELoss` is **not** used: it reads a six-column
  edge attribute carrying GridFM's YFF/YFT split, whereas the OPFData adapter
  carries only the two-column Y-bus off-diagonal, and fabricating the split
  would not reproduce their loss anyway. The caller supplies its own
  power-balance term in its place; that substitution is reported in the run log.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# GridSFM
# --------------------------------------------------------------------------- #

# OPFData contains only solved instances, so the feasibility classifier and the
# stress-feasibility regulariser have no signal to learn from. Zeroing them is
# a statement about the dataset, not a judgement about the loss.
GRIDSFM_OPFDATA_LAMBDAS = dict(lambda_feas=0.0, lambda_stress_feas=0.0)


def gridsfm_native(gb, lambdas: Optional[Dict[str, float]] = None
                   ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """GridSFM's own `compute_loss` on a prepared PyG batch carrying preds."""
    from gridsfm.loss import compute_loss

    if not hasattr(gb, "feasible") or gb.feasible is None:
        n_g = int(getattr(gb, "num_graphs", 1) or 1)
        gb.feasible = torch.ones(n_g, device=gb["bus"].x.device)

    kw = dict(GRIDSFM_OPFDATA_LAMBDAS)
    if lambdas:
        kw.update(lambdas)
    loss, parts = compute_loss(gb, **kw)
    return loss, {k: float(v) for k, v in parts.items()}


# --------------------------------------------------------------------------- #
# LUMINA
# --------------------------------------------------------------------------- #

class LuminaNative:
    """`ACOPFLossFunction` over LUMINA's own bus and generator heads."""

    def __init__(self, loss_type: str = "mse",
                 bus_weight: float = 1.0, gen_weight: float = 1.0):
        from lumina.model.opf.losses import ACOPFLossFunction
        self.fn = ACOPFLossFunction(
            loss_type=loss_type,
            node_weights={"bus": bus_weight, "generator": gen_weight},
        )

    def __call__(self, out: Dict[str, torch.Tensor], hb
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
        preds, tgts = {}, {}
        for nt in ("bus", "generator"):
            if nt in out and nt in hb.node_types and hasattr(hb[nt], "y"):
                preds[nt] = out[nt]
                tgts[nt] = hb[nt].y.to(out[nt].dtype)
        if not preds:
            raise RuntimeError("LUMINA native loss: no bus/generator targets in batch")
        res = self.fn(preds, tgts)
        parts = {k: float(v) for k, v in res.items()
                 if k != "total_loss" and torch.is_tensor(v)}
        return res["total_loss"], parts


# --------------------------------------------------------------------------- #
# GridFM
# --------------------------------------------------------------------------- #

class GridFMNative:
    """`MaskedReconstructionMSE` over [VM, VA, PG, QG, PD, QD].

    PD and QD are inputs to the OPF task, not unknowns, so they are left out of
    the mask; VM, VA, PG and QG are what the surrogate must reconstruct.
    """

    def __init__(self):
        from gridfm_graphkit.training.loss import MaskedReconstructionMSE
        from gridfm_graphkit.datasets.globals import (
            VM_H, VA_H, QG_H, PD_H, QD_H, PG_H,
        )
        self.fn = MaskedReconstructionMSE(None, None)
        self.VM_H, self.VA_H, self.QG_H = VM_H, VA_H, QG_H
        self.PD_H, self.QD_H, self.PG_H = PD_H, QD_H, PG_H

    def _targets(self, batch, gen_bus_cpu, device):
        """Ground truth in GridFM's 15-column bus / 7-column gen layout."""
        n = int(batch["V_newton"].squeeze(0).shape[0])
        tb = torch.zeros((n, 15), dtype=torch.float32, device=device)
        Vt = batch["V_newton"].squeeze(0).float().to(device)
        tb[:, self.VM_H] = Vt[:, 0]
        tb[:, self.VA_H] = Vt[:, 1]

        # Generation at the bus: OPFData ships it directly as S_gen_opf.
        if "S_gen_opf" in batch:
            g = batch["S_gen_opf"].squeeze(0).to(device)
        else:
            g = (batch["S_newton"].squeeze(0).to(device)
                 - batch["S_start"].squeeze(0).to(device))
        tb[:, self.QG_H] = g.imag.float()
        S = batch["S_start"].squeeze(0).to(device)
        tb[:, self.PD_H] = -S.real.float()
        tb[:, self.QD_H] = -S.imag.float()

        tg = torch.zeros((gen_bus_cpu.numel(), 7), dtype=torch.float32, device=device)
        tg[:, self.PG_H] = g.real.float()[gen_bus_cpu.to(device)]
        return {"bus": tb, "gen": tg}

    def _masks(self, n_bus, n_gen, device):
        mb = torch.zeros((n_bus, 15), dtype=torch.bool, device=device)
        mb[:, self.VM_H] = True   # unknown: to be reconstructed
        mb[:, self.VA_H] = True
        mb[:, self.QG_H] = True
        mb[:, self.PD_H] = False  # known: OPF input
        mb[:, self.QD_H] = False
        mg = torch.zeros((n_gen, 7), dtype=torch.bool, device=device)
        mg[:, self.PG_H] = True
        return {"bus": mb, "gen": mg}

    def __call__(self, pred6, batch, edge_index_dict, gen_bus_cpu, device
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
        tgt = self._targets(batch, gen_bus_cpu, device)
        msk = self._masks(pred6.shape[0], gen_bus_cpu.numel(), device)
        res = self.fn({"bus": pred6}, tgt, edge_index_dict, {}, msk)
        return res["loss"], {"masked_recon_mse": float(res["loss"].detach())}

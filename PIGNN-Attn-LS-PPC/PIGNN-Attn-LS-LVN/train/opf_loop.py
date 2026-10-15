"""Training and evaluation loop for OPF (Optimal Power Flow) models.

Parallel to loop.py but handles the dict return of GNSMsg_EdgeSelfAttn_OPF:
    (V_out, Pg_out, loss_dict)

where loss_dict contains:
    "total"             differentiable scalar for .backward()
    "gen_cost"          γ-weighted generation cost (exact, not heuristic)
    "kcl_loss"          γ-weighted KCL mismatch penalty
    "lim_loss"          γ-weighted generator limit penalty
    "branch_loss"       γ-weighted branch thermal penalty
    "v_loss"            γ-weighted soft voltage penalty
    "final_kcl_inf"     max KCL violation at last iteration (monitoring)
    "final_branch_max"  max branch thermal violation at last iteration
    "final_lim_max"     max generator limit violation at last iteration

For compatibility with entrypoint.py's EpochMetrics-based logging, OPFEpochMetrics
exposes .loss, .rmse, .rmse_mag, .rmse_ang_deg, .phys properties.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .logger import console, log
from rich.progress import track


@dataclass
class OPFEpochMetrics:
    """Metrics for one epoch of OPF training/evaluation.

    All five loss components are tracked exactly (not via heuristic split).
    Constraint diagnostics at the final iteration are tracked separately.
    """

    total_loss: float
    gen_cost: float
    kcl_loss: float
    lim_loss: float
    branch_loss: float = 0.0
    v_loss: float = 0.0
    # Infinity-norm constraint violations at the final GNN iteration
    kcl_inf_norm: float = 0.0
    branch_max: float = 0.0
    lim_max: float = 0.0
    v_inf_norm: float = 0.0        # max pre-clamp voltage violation
    # Fraction of batches where ALL max violations < FEASIBILITY_EPS (proxy)
    feasibility_rate: float = 0.0
    # Average inference time in milliseconds per sample
    runtime_ms_per_sample: float = 0.0

    # ── EpochMetrics-compatible properties used by entrypoint.py ──────────

    @property
    def loss(self) -> float:
        return self.total_loss

    @property
    def mse(self) -> float:
        return self.gen_cost

    @property
    def mse_mag(self) -> float:
        return self.gen_cost

    @property
    def mse_ang(self) -> float:
        return self.kcl_loss

    @property
    def phys(self) -> float:
        # All constraint penalties aggregated (for history.csv "phys" column)
        return self.lim_loss + self.branch_loss + self.v_loss

    @property
    def rmse(self) -> float:
        return math.sqrt(max(self.gen_cost, 0.0))

    @property
    def rmse_mag(self) -> float:
        return math.sqrt(max(self.gen_cost, 0.0))

    @property
    def rmse_ang_deg(self) -> float:
        # KCL mismatch scaled for readability (pu mismatch × 100)
        return math.sqrt(max(self.kcl_loss, 0.0)) * 100.0


def run_opf_epoch(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    train: bool,
    block_diag: bool,
    optim: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    clip_grad_norm: float = 1.0,
    desc: str | None = None,
    show_progress: bool = True,
) -> OPFEpochMetrics:
    """Run one epoch of OPF training or evaluation.

    Args:
        model:          GNSMsg_EdgeSelfAttn_OPF instance.
        loader:         DataLoader yielding standard PF batches.
        device:         torch device.
        train:          If True, compute gradients and step the optimiser.
        block_diag:     Whether the loader uses block-diagonal batching.
        optim:          Optimiser (required when train=True).
        scheduler:      LR scheduler (stepped once per batch).
        clip_grad_norm: Gradient clipping max norm.
        desc:           Progress bar label.
        show_progress:  Whether to render a rich progress bar.

    Returns:
        OPFEpochMetrics aggregated over the full epoch.
    """
    model.train() if train else model.eval()

    # Feasibility threshold: a batch is feasible if all max violations < this (pu)
    FEASIBILITY_EPS = 0.05

    sum_loss       = 0.0
    sum_gen_cost   = 0.0
    sum_kcl        = 0.0
    sum_lim        = 0.0
    sum_branch     = 0.0
    sum_v          = 0.0
    sum_kcl_inf    = 0.0
    sum_branch_max = 0.0
    sum_lim_max    = 0.0
    sum_v_max      = 0.0
    n_feasible     = 0
    n_batches      = 0
    total_time_s   = 0.0
    n_samples      = 0

    if desc is None:
        desc = "opf-train" if train else "opf-eval"

    try:
        total = len(loader)
    except Exception:
        total = None

    iterator = loader
    if show_progress:
        iterator = track(loader, total=total, description=desc, console=console)

    with torch.set_grad_enabled(train):
        for batch in iterator:
            B = batch["bus_type"].size(0)
            n_samples += B
            n_batches += 1

            n_nodes_per_graph = batch["sizes"].to(device) if block_diag else None

            bus_type = batch["bus_type"].to(device)
            Line     = batch["Lines_connected"].to(device)
            Y_raw    = batch.get("Ybus", None)
            Y        = Y_raw.to(device, non_blocking=True) if isinstance(Y_raw, torch.Tensor) else None
            Ys       = batch["Y_Lines"].to(device)
            Yc       = batch["Y_C_Lines"].to(device)
            Sstart   = batch["S_start"].to(device)
            Vstart   = batch["V_start"].to(device)

            # OPF metadata: synthesised inside model when not in batch
            Pg_min = batch["Pg_min"].to(device)     if "Pg_min"    in batch else None
            Pg_max = batch["Pg_max"].to(device)     if "Pg_max"    in batch else None
            c1     = batch["cost_c1"].to(device)    if "cost_c1"   in batch else None
            c2     = batch["cost_c2"].to(device)    if "cost_c2"   in batch else None

            t_start = time.time()
            _V_pred, _Pg_pred, loss_dict = model(
                bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph,
                Pg_min=Pg_min, Pg_max=Pg_max, c1=c1, c2=c2,
            )
            total_time_s += time.time() - t_start

            opf_loss = loss_dict["total"]

            if train:
                assert optim is not None
                optim.zero_grad()
                opf_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                optim.step()
                if scheduler is not None:
                    scheduler.step()

            # Accumulate exact per-component values (all are floats in loss_dict)
            kcl_inf_b    = float(loss_dict["final_kcl_inf"])
            branch_max_b = float(loss_dict["final_branch_max"])
            lim_max_b    = float(loss_dict["final_lim_max"])
            v_max_b      = float(loss_dict.get("final_v_max", 0.0))

            sum_loss       += float(opf_loss.item()) * B
            sum_gen_cost   += float(loss_dict["gen_cost"])   * B
            sum_kcl        += float(loss_dict["kcl_loss"])   * B
            sum_lim        += float(loss_dict["lim_loss"])   * B
            sum_branch     += float(loss_dict["branch_loss"]) * B
            sum_v          += float(loss_dict["v_loss"])     * B
            sum_kcl_inf    += kcl_inf_b    * B
            sum_branch_max += branch_max_b * B
            sum_lim_max    += lim_max_b    * B
            sum_v_max      += v_max_b      * B

            # Batch is feasible if ALL max violations fall below threshold
            if (kcl_inf_b < FEASIBILITY_EPS
                    and branch_max_b < FEASIBILITY_EPS
                    and lim_max_b < FEASIBILITY_EPS):
                n_feasible += 1

    n = max(n_samples, 1)
    nb = max(n_batches, 1)
    return OPFEpochMetrics(
        total_loss            = sum_loss       / n,
        gen_cost              = sum_gen_cost   / n,
        kcl_loss              = sum_kcl        / n,
        lim_loss              = sum_lim        / n,
        branch_loss           = sum_branch     / n,
        v_loss                = sum_v          / n,
        kcl_inf_norm          = sum_kcl_inf    / n,
        branch_max            = sum_branch_max / n,
        lim_max               = sum_lim_max    / n,
        v_inf_norm            = sum_v_max      / n,
        feasibility_rate      = n_feasible     / nb,
        runtime_ms_per_sample = 1000.0 * total_time_s / n,
    )


@dataclass
class OPFTrainHistory:
    train_loss: list[float]
    train_gen_cost: list[float]
    train_kcl: list[float]
    train_branch: list[float]
    val_loss: list[float]
    val_gen_cost: list[float]
    val_kcl: list[float]
    val_branch: list[float]
    best_epoch: int
    best_score: float
    best_val_gen_cost: float
    best_val_kcl: float
    best_val_branch: float = 0.0


def train_validate_opf(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    pinn: bool,
    block_diag: bool,
    optim: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    val_every: int,
    best_ckpt_path: str | None = None,
    on_epoch_metrics: Callable[[int, str, object], None] | None = None,
    show_progress: bool = True,
) -> OPFTrainHistory:
    """Full OPF train + validation loop.

    Signature mirrors train_validate() so entrypoint.py can call either
    function interchangeably based on the model type.

    Best checkpoint selected by min(val_gen_cost + val_kcl_loss + val_branch_loss):
      - val_gen_cost    economic optimality
      - val_kcl_loss    power balance feasibility
      - val_branch_loss thermal feasibility
    """
    train_loss_hist:   list[float] = []
    train_gen_hist:    list[float] = []
    train_kcl_hist:    list[float] = []
    train_branch_hist: list[float] = []
    val_loss_hist:     list[float] = []
    val_gen_hist:      list[float] = []
    val_kcl_hist:      list[float] = []
    val_branch_hist:   list[float] = []

    best_epoch        = 0
    best_score        = float("inf")
    best_val_gen_cost = float("inf")
    best_val_kcl      = float("inf")
    best_val_branch   = float("inf")

    log.info("OPF — initial metrics before training:")
    m_train0 = run_opf_epoch(
        model=model, loader=train_loader, device=device,
        train=False, block_diag=block_diag, desc="init/train",
        show_progress=show_progress,
    )
    m_val0 = run_opf_epoch(
        model=model, loader=val_loader, device=device,
        train=False, block_diag=block_diag, desc="init/valid",
        show_progress=show_progress,
    )

    log.info(
        "Epoch %3d | train loss %.4e  gen %.4e  kcl %.4e  branch %.4e | "
        "valid loss %.4e  gen %.4e  kcl %.4e  branch %.4e",
        0,
        m_train0.total_loss, m_train0.gen_cost, m_train0.kcl_loss, m_train0.branch_loss,
        m_val0.total_loss,   m_val0.gen_cost,   m_val0.kcl_loss,   m_val0.branch_loss,
    )

    if on_epoch_metrics is not None:
        on_epoch_metrics(0, "train", m_train0)
        on_epoch_metrics(0, "val",   m_val0)

    score0 = m_val0.gen_cost + m_val0.kcl_loss + m_val0.branch_loss
    best_score        = float(score0)
    best_val_gen_cost = float(m_val0.gen_cost)
    best_val_kcl      = float(m_val0.kcl_loss)
    best_val_branch   = float(m_val0.branch_loss)
    if best_ckpt_path is not None:
        torch.save(model.state_dict(), best_ckpt_path)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)

        m_train = run_opf_epoch(
            model=model, loader=train_loader, device=device,
            train=True, block_diag=block_diag,
            optim=optim, scheduler=scheduler,
            desc=f"epoch {epoch}/{epochs} opf-train",
            show_progress=show_progress,
        )

        train_loss_hist.append(m_train.total_loss)
        train_gen_hist.append(m_train.gen_cost)
        train_kcl_hist.append(m_train.kcl_loss)
        train_branch_hist.append(m_train.branch_loss)

        if on_epoch_metrics is not None:
            on_epoch_metrics(epoch, "train", m_train)

        if epoch % val_every == 0 or epoch == epochs:
            m_val = run_opf_epoch(
                model=model, loader=val_loader, device=device,
                train=False, block_diag=block_diag,
                desc=f"epoch {epoch}/{epochs} opf-valid",
                show_progress=show_progress,
            )

            val_loss_hist.append(m_val.total_loss)
            val_gen_hist.append(m_val.gen_cost)
            val_kcl_hist.append(m_val.kcl_loss)
            val_branch_hist.append(m_val.branch_loss)

            if on_epoch_metrics is not None:
                on_epoch_metrics(epoch, "val", m_val)

            log.info(
                "Epoch %3d | "
                "train loss %.4e  gen %.4e  kcl %.4e  branch %.4e | "
                "valid loss %.4e  gen %.4e  kcl %.4e  lim %.4e  branch %.4e  v %.4e | "
                "kcl_inf %.4e  branch_max %.4e  v_max %.4e  lim_max %.4e | "
                "feasible %.1f%%  rt %.2fms/s | time %.2fs",
                epoch,
                m_train.total_loss, m_train.gen_cost, m_train.kcl_loss, m_train.branch_loss,
                m_val.total_loss,   m_val.gen_cost,   m_val.kcl_loss,   m_val.lim_loss,
                m_val.branch_loss,  m_val.v_loss,
                m_val.kcl_inf_norm, m_val.branch_max, m_val.v_inf_norm,  m_val.lim_max,
                m_val.feasibility_rate * 100.0, m_val.runtime_ms_per_sample,
                time.time() - t0,
            )

            score = m_val.gen_cost + m_val.kcl_loss + m_val.branch_loss
            if score < best_score:
                best_epoch        = epoch
                best_score        = float(score)
                best_val_gen_cost = float(m_val.gen_cost)
                best_val_kcl      = float(m_val.kcl_loss)
                best_val_branch   = float(m_val.branch_loss)
                if best_ckpt_path is not None:
                    torch.save(model.state_dict(), best_ckpt_path)
                    log.info(
                        "OPF best checkpoint (epoch %d, score=%.4e) -> %s",
                        best_epoch, best_score, best_ckpt_path,
                    )
        else:
            log.info(
                "Epoch %3d | train loss %.4e  gen %.4e  kcl %.4e  branch %.4e | time %.2fs",
                epoch,
                m_train.total_loss, m_train.gen_cost, m_train.kcl_loss, m_train.branch_loss,
                time.time() - t0,
            )

    return OPFTrainHistory(
        train_loss     = train_loss_hist,
        train_gen_cost = train_gen_hist,
        train_kcl      = train_kcl_hist,
        train_branch   = train_branch_hist,
        val_loss       = val_loss_hist,
        val_gen_cost   = val_gen_hist,
        val_kcl        = val_kcl_hist,
        val_branch     = val_branch_hist,
        best_epoch     = int(best_epoch),
        best_score     = float(best_score),
        best_val_gen_cost = float(best_val_gen_cost),
        best_val_kcl   = float(best_val_kcl),
        best_val_branch = float(best_val_branch),
    )


def evaluate_test_opf(
    *,
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    pinn: bool,
    block_diag: bool,
    show_progress: bool = True,
) -> OPFEpochMetrics:
    """Evaluate the OPF model on the test set."""
    return run_opf_epoch(
        model=model, loader=test_loader, device=device,
        train=False, block_diag=block_diag,
        desc="opf-test", show_progress=show_progress,
    )

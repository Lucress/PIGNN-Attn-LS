"""Supervised (label-based) training loop for OPF with PIGNN-Attn-LS.

Model 1 in the research objectives: same GNSMsg_EdgeSelfAttn_OPF architecture
trained with MSE loss against OPF labels (Pg_opt, V_opt) instead of the
physics-informed loss used by Model 2.

Loss = MSE(V_pred, V_opt) + lambda_pg * MSE(Pg_pred[pv], Pg_opt[pv])
     = v_mse + lambda_pg * pg_mse

Objective gap = (cost_pred_pv - cost_opt_pv) / |cost_opt_pv|
where cost_pv = C1_PV * Pg_pv + C2_PV * Pg_pv^2   (PV generator only).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .logger import console, log
from rich.progress import track

# Must match generate_synthetic.py / opf_model.py defaults
C1_PV = 2.0
C2_PV = 0.5

# PG MSE weight in the combined supervised loss
LAMBDA_PG = 0.10


@dataclass
class OPFSupervisedMetrics:
    """Metrics for one epoch of supervised OPF training/evaluation."""

    total_loss: float
    v_mse: float            # voltage (mag + ang) MSE vs optimal
    pg_mse: float           # PV dispatch MSE vs optimal
    obj_gap: float          # (cost_pred_pv - cost_opt_pv) / |cost_opt_pv|
    cost_pred: float        # mean predicted PV generation cost per graph
    cost_opt: float         # mean optimal PV generation cost per graph
    runtime_ms_per_sample: float = 0.0

    # EpochMetrics-compatible properties (used by entrypoint.py callback)
    @property
    def loss(self) -> float:
        return self.total_loss

    @property
    def rmse(self) -> float:
        return math.sqrt(max(self.v_mse, 0.0))

    @property
    def rmse_mag(self) -> float:
        return math.sqrt(max(self.v_mse, 0.0))

    @property
    def rmse_ang_deg(self) -> float:
        return math.sqrt(max(self.pg_mse, 0.0)) * 10.0

    @property
    def phys(self) -> float:
        return self.obj_gap


def _compute_pv_cost_per_graph(
    Pg: torch.Tensor,        # (B, N) or (1, M)  — all buses
    pv_mask: torch.Tensor,   # same shape as Pg, float
    block_diag: bool,
    sizes: Optional[torch.Tensor],  # (n_graphs,) if block_diag
) -> torch.Tensor:
    """Compute per-graph PV generation cost (C1*Pg + C2*Pg^2), returns (n_graphs,)."""
    Pg_pv = Pg * pv_mask   # zero out non-PV buses

    if block_diag and sizes is not None:
        # Flatten to (M,) and scatter-sum per graph
        Pg_flat = Pg_pv.squeeze(0)   # (M,)
        n_graphs = sizes.numel()
        idx = torch.repeat_interleave(
            torch.arange(n_graphs, device=Pg.device),
            sizes.to(Pg.device),
        )
        Pg_per_graph = torch.zeros(n_graphs, device=Pg.device, dtype=Pg.dtype)
        Pg_per_graph.scatter_add_(0, idx, Pg_flat)
    else:
        # Standard batching: (B, N) → sum over N → (B,)
        Pg_per_graph = Pg_pv.sum(dim=-1)

    return C1_PV * Pg_per_graph + C2_PV * Pg_per_graph**2


def run_opf_supervised_epoch(
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
) -> OPFSupervisedMetrics:
    """One epoch of supervised OPF training/evaluation."""
    model.train() if train else model.eval()

    sum_loss = 0.0
    sum_v_mse = 0.0
    sum_pg_mse = 0.0
    sum_cost_pred = 0.0
    sum_cost_opt = 0.0
    n_graphs_total = 0
    total_time_s = 0.0

    if desc is None:
        desc = "sup-train" if train else "sup-eval"

    try:
        total = len(loader)
    except Exception:
        total = None

    iterator = loader
    if show_progress:
        iterator = track(loader, total=total, description=desc, console=console)

    with torch.set_grad_enabled(train):
        for batch in iterator:
            n_nodes_per_graph = batch["sizes"].to(device) if block_diag else None
            n_graphs_batch = int(batch["sizes"].numel()) if block_diag else int(batch["bus_type"].size(0))

            bus_type = batch["bus_type"].to(device)
            Line     = batch["Lines_connected"].to(device)
            Y        = batch.get("Ybus")
            if isinstance(Y, torch.Tensor):
                Y = Y.to(device, non_blocking=True)
            Ys       = batch["Y_Lines"].to(device)
            Yc       = batch["Y_C_Lines"].to(device)
            Sstart   = batch["S_start"].to(device)
            Vstart   = batch["V_start"].to(device)

            V_opt_mag = batch["V_opt_mag"].to(device)   # (1, M) or (B, N)
            V_opt_ang = batch["V_opt_ang"].to(device)   # (1, M) or (B, N)
            Pg_opt    = batch["Pg_opt"].to(device)       # (1, M) or (B, N)

            t0 = time.time()
            V_pred, Pg_pred, _ = model(
                bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph,
            )
            total_time_s += time.time() - t0

            # V_pred: (B_or_1, N_or_M, 2)  — [v_magnitude, theta]
            v_mse_mag = F.mse_loss(V_pred[..., 0], V_opt_mag)
            v_mse_ang = F.mse_loss(V_pred[..., 1], V_opt_ang)
            v_mse = v_mse_mag + v_mse_ang

            pv_mask = (bus_type == 2).float()
            pg_mse = F.mse_loss(Pg_pred * pv_mask, Pg_opt * pv_mask)

            total_loss = v_mse + LAMBDA_PG * pg_mse

            if train:
                assert optim is not None
                optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                optim.step()
                if scheduler is not None:
                    scheduler.step()

            # Objective gap (PV cost component only — model predicts PV bus)
            with torch.no_grad():
                sizes = batch.get("sizes")
                cost_pred_per = _compute_pv_cost_per_graph(
                    Pg_pred.detach(), pv_mask, block_diag, sizes
                )
                cost_opt_per = _compute_pv_cost_per_graph(
                    Pg_opt, pv_mask, block_diag, sizes
                )
                cp = float(cost_pred_per.mean().item())
                co = float(cost_opt_per.mean().item())

            sum_loss       += float(total_loss.item()) * n_graphs_batch
            sum_v_mse      += float(v_mse.item())      * n_graphs_batch
            sum_pg_mse     += float(pg_mse.item())     * n_graphs_batch
            sum_cost_pred  += cp * n_graphs_batch
            sum_cost_opt   += co * n_graphs_batch
            n_graphs_total += n_graphs_batch

    n = max(n_graphs_total, 1)
    cost_pred_mean = sum_cost_pred / n
    cost_opt_mean  = sum_cost_opt  / n
    obj_gap = (cost_pred_mean - cost_opt_mean) / (abs(cost_opt_mean) + 1e-8)

    return OPFSupervisedMetrics(
        total_loss            = sum_loss   / n,
        v_mse                 = sum_v_mse  / n,
        pg_mse                = sum_pg_mse / n,
        obj_gap               = obj_gap,
        cost_pred             = cost_pred_mean,
        cost_opt              = cost_opt_mean,
        runtime_ms_per_sample = 1000.0 * total_time_s / max(n_graphs_total, 1),
    )


@dataclass
class OPFSupervisedTrainHistory:
    train_loss:    list[float]
    train_v_mse:   list[float]
    train_pg_mse:  list[float]
    train_obj_gap: list[float]
    val_loss:      list[float]
    val_v_mse:     list[float]
    val_pg_mse:    list[float]
    val_obj_gap:   list[float]
    best_epoch:    int
    best_score:    float
    best_val_v_mse:float
    best_obj_gap:  float


def train_validate_opf_supervised(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    pinn: bool,           # unused; kept for signature parity with train_validate_opf
    block_diag: bool,
    optim: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    val_every: int,
    best_ckpt_path: str | None = None,
    on_epoch_metrics: Callable[[int, str, object], None] | None = None,
    show_progress: bool = True,
) -> OPFSupervisedTrainHistory:
    """Full supervised train + validation loop.

    Best checkpoint selected by min(val_v_mse): lowest voltage error vs labels.
    """
    train_loss_h:   list[float] = []
    train_vmse_h:   list[float] = []
    train_pgmse_h:  list[float] = []
    train_gap_h:    list[float] = []
    val_loss_h:     list[float] = []
    val_vmse_h:     list[float] = []
    val_pgmse_h:    list[float] = []
    val_gap_h:      list[float] = []

    best_epoch   = 0
    best_score   = float("inf")
    best_v_mse   = float("inf")
    best_obj_gap = float("inf")

    log.info("Supervised OPF — initial metrics before training:")
    m0t = run_opf_supervised_epoch(
        model=model, loader=train_loader, device=device,
        train=False, block_diag=block_diag, desc="init/train",
        show_progress=show_progress,
    )
    m0v = run_opf_supervised_epoch(
        model=model, loader=val_loader, device=device,
        train=False, block_diag=block_diag, desc="init/valid",
        show_progress=show_progress,
    )
    log.info(
        "Epoch %3d | train loss %.4e  v_mse %.4e  pg_mse %.4e  gap %.3f%% | "
        "valid loss %.4e  v_mse %.4e  gap %.3f%%",
        0, m0t.total_loss, m0t.v_mse, m0t.pg_mse, m0t.obj_gap * 100.0,
        m0v.total_loss, m0v.v_mse, m0v.obj_gap * 100.0,
    )
    if on_epoch_metrics is not None:
        on_epoch_metrics(0, "train", m0t)
        on_epoch_metrics(0, "val",   m0v)

    best_score   = float(m0v.v_mse)
    best_v_mse   = float(m0v.v_mse)
    best_obj_gap = float(m0v.obj_gap)
    if best_ckpt_path is not None:
        torch.save(model.state_dict(), best_ckpt_path)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        m_train = run_opf_supervised_epoch(
            model=model, loader=train_loader, device=device,
            train=True, block_diag=block_diag,
            optim=optim, scheduler=scheduler,
            desc=f"epoch {epoch}/{epochs} sup-train",
            show_progress=show_progress,
        )

        train_loss_h.append(m_train.total_loss)
        train_vmse_h.append(m_train.v_mse)
        train_pgmse_h.append(m_train.pg_mse)
        train_gap_h.append(m_train.obj_gap)

        if on_epoch_metrics is not None:
            on_epoch_metrics(epoch, "train", m_train)

        if epoch % val_every == 0 or epoch == epochs:
            m_val = run_opf_supervised_epoch(
                model=model, loader=val_loader, device=device,
                train=False, block_diag=block_diag,
                desc=f"epoch {epoch}/{epochs} sup-valid",
                show_progress=show_progress,
            )

            val_loss_h.append(m_val.total_loss)
            val_vmse_h.append(m_val.v_mse)
            val_pgmse_h.append(m_val.pg_mse)
            val_gap_h.append(m_val.obj_gap)

            if on_epoch_metrics is not None:
                on_epoch_metrics(epoch, "val", m_val)

            log.info(
                "Epoch %3d | train loss %.4e  v_mse %.4e  pg %.4e  gap %.3f%% | "
                "valid loss %.4e  v_mse %.4e  pg %.4e  gap %.3f%% | time %.2fs",
                epoch,
                m_train.total_loss, m_train.v_mse, m_train.pg_mse, m_train.obj_gap * 100.0,
                m_val.total_loss,   m_val.v_mse,   m_val.pg_mse,   m_val.obj_gap * 100.0,
                time.time() - t0,
            )

            if m_val.v_mse < best_score:
                best_epoch   = epoch
                best_score   = float(m_val.v_mse)
                best_v_mse   = float(m_val.v_mse)
                best_obj_gap = float(m_val.obj_gap)
                if best_ckpt_path is not None:
                    torch.save(model.state_dict(), best_ckpt_path)
                    log.info(
                        "Supervised best ckpt (epoch %d, v_mse=%.4e, gap=%.3f%%) -> %s",
                        best_epoch, best_v_mse, best_obj_gap * 100.0, best_ckpt_path,
                    )
        else:
            log.info(
                "Epoch %3d | train loss %.4e  v_mse %.4e  pg %.4e  gap %.3f%% | time %.2fs",
                epoch,
                m_train.total_loss, m_train.v_mse, m_train.pg_mse, m_train.obj_gap * 100.0,
                time.time() - t0,
            )

    return OPFSupervisedTrainHistory(
        train_loss    = train_loss_h,
        train_v_mse   = train_vmse_h,
        train_pg_mse  = train_pgmse_h,
        train_obj_gap = train_gap_h,
        val_loss      = val_loss_h,
        val_v_mse     = val_vmse_h,
        val_pg_mse    = val_pgmse_h,
        val_obj_gap   = val_gap_h,
        best_epoch    = int(best_epoch),
        best_score    = float(best_score),
        best_val_v_mse= float(best_v_mse),
        best_obj_gap  = float(best_obj_gap),
    )


def evaluate_test_opf_supervised(
    *,
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    pinn: bool,
    block_diag: bool,
    show_progress: bool = True,
) -> OPFSupervisedMetrics:
    """Evaluate the supervised OPF model on the test set."""
    return run_opf_supervised_epoch(
        model=model, loader=test_loader, device=device,
        train=False, block_diag=block_diag,
        desc="sup-test", show_progress=show_progress,
    )

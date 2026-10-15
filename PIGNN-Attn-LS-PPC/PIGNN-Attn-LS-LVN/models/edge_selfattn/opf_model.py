"""GNSMsg_EdgeSelfAttn_OPF — physics-informed OPF (Model 2 from project PDF).

Extends GNSMsg_EdgeSelfAttn with a joint active-power dispatch head (Pg) and a
fully physics-informed loss that penalises five distinct constraint categories:

  1. Generation cost         Σ_g (c1·Pg + c2·Pg²)                [economic objective]
  2. KCL mismatch (L2)      mean(ΔP² + ΔQ²)                      [power balance — average]
  2b. KCL mismatch (L∞)     mean_graphs(max_nodes √(ΔP²+ΔQ²))    [worst-node feasibility]
  3. Generator limits       mean(relu(Pg-Pg_max)² + relu(Pg_min-Pg)²)
  4. Branch thermal limits  mean(relu(|S_line| - S_max)²)         [pi-model current flow]
  5. Soft voltage limits    mean(relu(v-v_max)² + relu(v_min-v)²) [pre-clamp so grad flows]

The branch thermal penalty uses the pi-model:
    I_ij = Y_s·(V_i - V_j) + Y_c/2·V_i
    S_ij = V_i · conj(I_ij)
    penalty = mean(relu(|S_ij| - S_max)²)

All five penalties are weighted and γ-discounted across K iterations.
All loss components are measured AFTER each iteration's state update so the
loss directly reflects the quality of the model's own outputs.

Returns (V_out, Pg_out, loss_dict) where loss_dict["total"] is the differentiable
scalar for .backward() and each component is tracked exactly (no heuristic split).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .admittance import build_dense_Y, build_edges_blockdiag, build_edges_plain
from .model import GNSMsg_EdgeSelfAttn


class GNSMsg_EdgeSelfAttn_OPF(GNSMsg_EdgeSelfAttn):
    """Model 2: physics-informed PIGNN-OPF with full five-component constraint suite.

    New bus features appended to [v, θ, ΔP, ΔQ]:
        Pg          current generator dispatch (pu; 0 for PQ/slack buses)
        margin_lo   Pg - Pg_min   (headroom to decrease)
        margin_hi   Pg_max - Pg   (headroom to increase)
        c1          linear cost coefficient (0 for non-generators)
        c2          quadratic cost coefficient (0 for non-generators)

    New prediction head per iteration:
        Pg_head[k]: Linear(d_model, 1) → ΔPg
            clamped to ±pg_step_frac·(Pg_max-Pg_min), masked to PV buses only

    Combined loss at iteration k (γ-discounted):
        L_k = λ_cost     · gen_cost_k        Σ_g(c1·Pg + c2·Pg²)
            + λ_kcl      · kcl_l2_k          mean(ΔP² + ΔQ²)       [average violation]
            + λ_kcl_inf  · kcl_linf_k        mean_G max_n √(ΔP²+ΔQ²) [worst-node violation]
            + λ_lim      · lim_k             mean(relu(Pg-Pg_max)² + relu(Pg_min-Pg)²)
            + λ_branch   · branch_k          mean(relu(|S_line| - S_max)²)
            + λ_v        · v_k               mean(relu(v-v_max)² + relu(v_min-v)²)

    Setting λ_kcl_inf > 0 directly penalises the worst-node KCL residual,
    which is the quantity that determines feasibility_rate. λ_kcl_inf=0 (default)
    recovers the original L2-only loss for backward compatibility.

    Returns:
        V_out     (B, N, 2)  [v_magnitude, theta]
        Pg_out    (B, N)     final generator dispatch
        loss_dict {
            "total"             differentiable scalar for .backward()
            "gen_cost"          γ-weighted generation cost   (float)
            "kcl_loss"          γ-weighted KCL penalty       (float)
            "lim_loss"          γ-weighted gen-limit penalty  (float)
            "branch_loss"       γ-weighted branch penalty     (float)
            "v_loss"            γ-weighted voltage penalty    (float)
            "final_kcl_inf"     max|ΔP,ΔQ| at final iter     (float, no grad)
            "final_branch_max"  max relu(|S|-S_max) at final  (float, no grad)
            "final_lim_max"     max gen-limit viol at final   (float, no grad)
        }
    """

    OPF_EXTRA_DIM = 5  # [Pg, margin_lo, margin_hi, c1, c2]

    def __init__(
        self,
        d: int = 10,
        d_hi: int = 32,
        K: int = 30,
        pinn: bool = True,
        gamma: float = 0.9,
        v_limit: bool = True,
        use_armijo: bool = True,
        d_model: int | None = None,
        n_heads: int = 4,
        num_attn_layers: int = 1,
        attn_dropout: float = 0.0,
        dtheta_max: float = 0.30,
        dvm_frac: float = 0.10,
        v_min: float = 0.75,
        v_max: float = 1.20,
        tied_heads: bool = False,
        # Economic objective
        lambda_cost: float = 1.0,
        # Physics constraint weights
        lambda_kcl: float = 1.0,      # KCL L2: mean squared mismatch
        lambda_kcl_inf: float = 0.0,  # KCL L∞: worst-node residual (0 = disabled)
        lambda_lim: float = 10.0,
        lambda_branch: float = 1.0,
        lambda_v: float = 1.0,
        # Dispatch control
        pg_step_frac: float = 0.05,
        c1_default: float = 2.0,
        c2_default: float = 0.5,
        pg_lim_frac: float = 0.30,
        # Thermal limits
        s_max_pu: float = 1.0,
    ):
        super().__init__(
            d=d, d_hi=d_hi, K=K, pinn=pinn, gamma=gamma,
            v_limit=v_limit, use_armijo=use_armijo, d_model=d_model,
            n_heads=n_heads, num_attn_layers=num_attn_layers,
            attn_dropout=attn_dropout, dtheta_max=dtheta_max,
            dvm_frac=dvm_frac, v_min=v_min, v_max=v_max,
            tied_heads=tied_heads,
            bus_feat_extra_dim=self.OPF_EXTRA_DIM,
        )

        self.lambda_cost    = float(lambda_cost)
        self.lambda_kcl     = float(lambda_kcl)
        self.lambda_kcl_inf = float(lambda_kcl_inf)
        self.lambda_lim     = float(lambda_lim)
        self.lambda_branch  = float(lambda_branch)
        self.lambda_v       = float(lambda_v)

        self.pg_step_frac = float(pg_step_frac)
        self.c1_default   = float(c1_default)
        self.c2_default   = float(c2_default)
        self.pg_lim_frac  = float(pg_lim_frac)
        self.s_max_pu     = float(s_max_pu)

        n_head_sets = 1 if self.tied_heads else self.K
        self.Pg_head = nn.ModuleList([
            nn.Linear(self.d_model, 1) for _ in range(n_head_sets)
        ])
        for j in range(n_head_sets):
            nn.init.zeros_(self.Pg_head[j].weight)
            nn.init.zeros_(self.Pg_head[j].bias)

    # ------------------------------------------------------------------
    # Branch thermal limit helper (pi-model)
    # ------------------------------------------------------------------

    @staticmethod
    def _branch_penalty_1d(
        Vc_1d: torch.Tensor,
        edge_index_dir: torch.Tensor,
        edge_feat_dir: torch.Tensor,
        s_max: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Branch thermal penalty for one flat (single-graph) voltage vector.

        Pi-model: I_ij = Y_s·(V_i - V_j) + Y_c/2·V_i
                  S_ij = V_i · conj(I_ij)
                  penalty = mean(relu(|S_ij| - s_max)²)

        Args:
            Vc_1d:          (N,) complex node voltages
            edge_index_dir: (2E, 2) directed edges; forward edges are rows 0..E-1
            edge_feat_dir:  (2E, 4) [Ysr, Ysi, Yc_real, Yc_imag]
            s_max:          scalar thermal limit (pu apparent power per line)

        Returns:
            s_mag:   (E,) apparent power magnitude per forward edge (real tensor)
            penalty: scalar mean(relu(s_mag - s_max)²), part of computation graph
        """
        if edge_index_dir.numel() == 0 or Vc_1d.numel() == 0:
            if torch.is_complex(Vc_1d):
                z = Vc_1d.real.new_zeros(1).squeeze()
            else:
                z = Vc_1d.new_zeros(1).squeeze()
            return z.new_zeros(0), z

        E = edge_index_dir.shape[0] // 2
        ei = edge_index_dir[:E]   # forward edges only
        ef = edge_feat_dir[:E]

        Ys_ij   = torch.complex(ef[:, 0], ef[:, 1])
        Yc_half = torch.complex(ef[:, 2], ef[:, 3]) / 2.0

        if Vc_1d.dtype != Ys_ij.dtype:
            Ys_ij   = Ys_ij.to(Vc_1d.dtype)
            Yc_half = Yc_half.to(Vc_1d.dtype)

        V_i = Vc_1d[ei[:, 0]]
        V_j = Vc_1d[ei[:, 1]]

        I_ij = Ys_ij * (V_i - V_j) + Yc_half * V_i
        S_ij = V_i * I_ij.conj()
        s_mag = S_ij.abs()  # real, differentiable

        penalty = torch.relu(s_mag - s_max).pow(2).mean()
        return s_mag, penalty

    def _compute_branch_penalty(
        self,
        Vc: torch.Tensor,
        n_nodes_per_graph,
        edge_index_dir,
        edge_feat_dir,
        edge_index_dir_list,
        edge_feat_dir_list,
        B: int,
    ) -> tuple[torch.Tensor, float]:
        """Branch penalty dispatcher for block-diagonal and per-batch modes.

        Returns:
            pen_tensor: differentiable real scalar (part of computation graph)
            pen_float:  same value as a Python float (for monitoring)
        """
        if n_nodes_per_graph is not None:
            _, pen = self._branch_penalty_1d(
                Vc[0], edge_index_dir, edge_feat_dir, self.s_max_pu
            )
        else:
            pen = Vc.real.new_zeros(1).squeeze() if torch.is_complex(Vc) else Vc.new_zeros(1).squeeze()
            for b in range(B):
                e_b = edge_index_dir_list[b]
                if e_b.numel() > 0:
                    _, bp = self._branch_penalty_1d(
                        Vc[b], e_b, edge_feat_dir_list[b], self.s_max_pu
                    )
                    pen = pen + bp
            if B > 0:
                pen = pen / B

        return pen, float(pen.detach().item())

    def _branch_max_violation(
        self,
        Vc: torch.Tensor,
        n_nodes_per_graph,
        edge_index_dir,
        edge_feat_dir,
        edge_index_dir_list,
        edge_feat_dir_list,
        B: int,
    ) -> float:
        """Max relu(|S_line| - S_max) across all lines (monitoring only, no grad)."""
        with torch.no_grad():
            if n_nodes_per_graph is not None:
                s_mag, _ = self._branch_penalty_1d(
                    Vc[0], edge_index_dir, edge_feat_dir, self.s_max_pu
                )
                if s_mag.numel() == 0:
                    return 0.0
                return float(torch.relu(s_mag - self.s_max_pu).amax().item())
            else:
                worst = 0.0
                for b in range(B):
                    e_b = edge_index_dir_list[b]
                    if e_b.numel() > 0:
                        s_b, _ = self._branch_penalty_1d(
                            Vc[b], e_b, edge_feat_dir_list[b], self.s_max_pu
                        )
                        if s_b.numel() > 0:
                            worst = max(worst, float(torch.relu(s_b - self.s_max_pu).amax().item()))
                return worst

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        bus_type,
        Line,
        Y,
        Ys,
        Yc,
        S,
        V0,
        n_nodes_per_graph,
        *,
        Pg_min=None,
        Pg_max=None,
        c1=None,
        c2=None,
        s_max=None,
        **_unused,
    ):
        """OPF physics-informed forward pass (five-component loss, Model 2).

        Args:
            bus_type, Line, Y, Ys, Yc, S, V0, n_nodes_per_graph:
                Same as GNSMsg_EdgeSelfAttn.forward.
            Pg_min: (B, N) lower generator limits. Synthesised if None.
            Pg_max: (B, N) upper generator limits. Synthesised if None.
            c1:     (B, N) linear cost coefficients. Synthesised if None.
            c2:     (B, N) quadratic cost coefficients. Synthesised if None.
            s_max:  reserved for per-line limits (not yet used; scalar s_max_pu applied).

        Returns:
            (V_out, Pg_out, loss_dict) — see class docstring.
        """
        device = bus_type.device
        B, N = bus_type.shape

        P_set_orig = S.real
        Q_set      = S.imag

        v  = V0[..., 0].clone()
        th = V0[..., 1].clone()
        m  = v.new_zeros(B, N, self.d)

        slack_mask = bus_type == 1
        pv_mask    = bus_type == 2

        # ── Synthesise OPF metadata when not supplied ─────────────────────
        Pg = (P_set_orig * pv_mask.float()).clone()

        if Pg_min is None or Pg_max is None:
            pg_range = torch.clamp(Pg.abs(), min=0.05)
            Pg_min = Pg - self.pg_lim_frac * pg_range
            Pg_max = Pg + self.pg_lim_frac * pg_range
        Pg_min = Pg_min.masked_fill(~pv_mask, 0.0)
        Pg_max = Pg_max.masked_fill(~pv_mask, 0.0)

        if c1 is None:
            c1 = Pg.new_full(Pg.shape, self.c1_default).masked_fill(~pv_mask, 0.0)
        if c2 is None:
            c2 = Pg.new_full(Pg.shape, self.c2_default).masked_fill(~pv_mask, 0.0)

        # ── Build edges + admittance matrix ──────────────────────────────
        edge_index_dir      = None
        edge_feat_dir       = None
        edge_index_dir_list = None
        edge_feat_dir_list  = None

        if n_nodes_per_graph is not None:
            Line_1d = Line.squeeze(0) if Line.dim() == 2 else Line
            Ys_1d   = Ys.squeeze(0)
            Yc_1d   = Yc.squeeze(0)
            undirected, _, edge_index_dir, edge_feat_dir, ys_edge, yc_edge = (
                build_edges_blockdiag(
                    line_mask_1d=Line_1d, Ys_1d=Ys_1d, Yc_1d=Yc_1d,
                    n_nodes_per_graph=n_nodes_per_graph,
                    edge_feat_dim=self.edge_feat_dim,
                    pairs_for_n=self._pairs_for_n,
                    device=device,
                )
            )
            if Y is None:
                Y = build_dense_Y(N, undirected, ys_edge, yc_edge, device=device)
        else:
            pairs = self._pairs_for_n(N, device)
            edge_index_dir_list, edge_feat_dir_list, undirected_list, mask_list = (
                build_edges_plain(
                    Line=Line, Ys=Ys, Yc=Yc, N=N,
                    edge_feat_dim=self.edge_feat_dim,
                    pairs=pairs, device=device,
                )
            )
            if Y is None:
                Y_list = []
                for b in range(B):
                    mask      = mask_list[b]
                    undirected = undirected_list[b]
                    ys_b      = Ys[b][mask]
                    yc_b      = Yc[b][mask]
                    Y_list.append(build_dense_Y(N, undirected, ys_b, yc_b, device=device))
                Y = torch.stack(Y_list, dim=0)

        # Number of independent graphs in this batch.
        # In block-diag mode B=1 but n_graphs can be >> 1.
        # Used to normalise per-graph costs so they are comparable to
        # per-node losses (kcl, v_penalty, branch) that use .mean().
        n_graphs = (len(n_nodes_per_graph) if n_nodes_per_graph is not None else B)

        # ── Accumulators ─────────────────────────────────────────────────
        opf_loss     = v.new_zeros(1)
        acc_gen_cost = 0.0
        acc_kcl      = 0.0
        acc_lim      = 0.0
        acc_branch   = 0.0
        acc_v        = 0.0

        # Final-iteration diagnostics (detached; returned for monitoring)
        final_kcl_inf    = 0.0
        final_branch_max = 0.0
        final_lim_max    = 0.0
        final_v_max      = 0.0  # max pre-clamp voltage constraint violation

        # ── K-step iterative OPF loop ─────────────────────────────────────
        for k in range(self.K):
            # Current effective power injection (variable Pg for PV buses)
            P_eff = torch.where(pv_mask, Pg, P_set_orig)

            # KCL mismatch at current state (used as input features)
            Vc = v * torch.exp(1j * th)
            Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
            Sc = Vc * Ic.conj()
            DP = (P_eff - Sc.real).masked_fill(slack_mask, 0.0)
            DQ = (Q_set  - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

            # Bus feature vector: [v, θ, ΔP, ΔQ, Pg, margin_lo, margin_hi, c1, c2]
            margin_lo = (Pg - Pg_min).masked_fill(~pv_mask, 0.0)
            margin_hi = (Pg_max - Pg).masked_fill(~pv_mask, 0.0)
            extra     = torch.stack([Pg, margin_lo, margin_hi, c1, c2], dim=-1)
            bus_feat  = torch.cat([torch.stack([v, th, DP, DQ], dim=-1), extra], dim=-1)

            # Attention message passing
            x = self.in_proj(torch.cat([bus_feat, m], dim=-1))
            if n_nodes_per_graph is not None:
                x = self._apply_blocks(x, edge_index_dir, edge_feat_dir)
            else:
                x_out = x.clone()
                for b in range(B):
                    e_b = edge_index_dir_list[b]
                    if e_b.numel() == 0:
                        continue
                    xb = self._apply_blocks(x[b:b+1], e_b, edge_feat_dir_list[b])
                    x_out[b:b+1] = xb
                x = x_out

            head_idx = 0 if self.tied_heads else k

            # Predict updates
            dth = self.theta_head[head_idx](x).squeeze(-1)
            dv  = self.v_head[head_idx](x).squeeze(-1)
            dm  = torch.tanh(self.m_head[head_idx](x))
            dm  = F.layer_norm(dm, dm.shape[-1:])

            dPg = self.Pg_head[head_idx](x).squeeze(-1)
            dPg = dPg.masked_fill(~pv_mask, 0.0)
            pg_range_local = torch.clamp(Pg_max - Pg_min, min=1e-6)
            dPg = torch.clamp(dPg,
                              -self.pg_step_frac * pg_range_local,
                               self.pg_step_frac * pg_range_local)

            dth, dv = self._apply_constraints(
                v=v, dth=dth, dv=dv, slack_mask=slack_mask, pv_mask=pv_mask,
            )

            # Armijo line search on KCL residual (voltage only; Pg uses unit step)
            if self.use_armijo:
                a = self._armijo_alpha(
                    Y=Y, v=v, th=th, dv=dv, dth=dth,
                    P_set=P_eff, Q_set=Q_set,
                    slack_mask=slack_mask, pv_mask=pv_mask,
                )
            else:
                a = v.new_tensor(1.0)

            # ── Soft voltage penalty (pre-clamp so gradient flows back) ───
            v_new     = v + a * dv
            v_penalty = (
                torch.relu(v_new - self.v_max).pow(2)
                + torch.relu(self.v_min - v_new).pow(2)
            ).mean()

            # Apply state updates
            th = self._wrap_theta(th + a * dth)
            v  = torch.clamp(v_new, self.v_min, self.v_max)
            m  = m + a * dm
            Pg = torch.clamp(Pg + dPg, Pg_min, Pg_max)

            # ── Post-update metrics (reflect quality of this iteration's output) ──

            # Recompute power flow with updated state
            P_eff_new = torch.where(pv_mask, Pg, P_set_orig)
            Vc_new    = v * torch.exp(1j * th)
            Ic_new    = torch.matmul(Y, Vc_new.unsqueeze(-1)).squeeze(-1)
            Sc_new    = Vc_new * Ic_new.conj()
            DP_new    = (P_eff_new - Sc_new.real).masked_fill(slack_mask, 0.0)
            DQ_new    = (Q_set     - Sc_new.imag).masked_fill(slack_mask | pv_mask, 0.0)

            # Generation cost — normalised per graph so it is on the same
            # scale as per-node penalties (kcl, v_penalty, branch).
            # Block-diag: .sum() over the mega-graph / n_graphs.
            # Per-batch:  .sum(dim=-1).mean() (already per-sample).
            if n_nodes_per_graph is not None:
                gen_cost_k = (
                    (c1 * Pg + c2 * Pg.pow(2)) * pv_mask.float()
                ).sum() / max(n_graphs, 1)
            else:
                gen_cost_k = (
                    (c1 * Pg + c2 * Pg.pow(2)) * pv_mask.float()
                ).sum(dim=-1).mean()

            # KCL mismatch — L2 (average) + optional L∞ (worst node per graph)
            node_sq  = DP_new.pow(2) + DQ_new.pow(2)   # per-node squared residual
            kcl_l2_k = node_sq.mean()                   # averaged: logged as kcl_loss
            if self.lambda_kcl_inf > 0.0:
                # Per-graph worst-node Euclidean residual, then mean over graphs.
                # Block-diagonal: node_sq is flat (N_total,); split by graph sizes.
                # Standard batch: node_sq is (B, N); amax over node dim.
                if n_nodes_per_graph is not None:
                    _segs = torch.split(node_sq.sqrt(), n_nodes_per_graph.tolist())
                    kcl_linf_k = torch.stack([s.amax() for s in _segs]).mean()
                else:
                    kcl_linf_k = node_sq.sqrt().amax(dim=-1).mean()
            else:
                kcl_linf_k = kcl_l2_k.new_tensor(0.0)
            kcl_k = kcl_l2_k  # alias for logging (L2 component)

            # Generator limit violation — same per-graph normalisation as gen_cost
            if n_nodes_per_graph is not None:
                lim_k = (
                    (torch.relu(Pg - Pg_max).pow(2) + torch.relu(Pg_min - Pg).pow(2))
                    * pv_mask.float()
                ).sum() / max(n_graphs, 1)
            else:
                lim_k = (
                    (torch.relu(Pg - Pg_max).pow(2) + torch.relu(Pg_min - Pg).pow(2))
                    * pv_mask.float()
                ).sum(dim=-1).mean()

            # Branch thermal limit violation (updated Vc)
            branch_pen, branch_pen_f = self._compute_branch_penalty(
                Vc_new, n_nodes_per_graph,
                edge_index_dir, edge_feat_dir,
                edge_index_dir_list, edge_feat_dir_list, B,
            )

            # ── γ-discounted combined loss ────────────────────────────────
            gamma_k   = self.gamma ** (self.K - 1 - k)
            step_loss = gamma_k * (
                self.lambda_cost    * gen_cost_k
                + self.lambda_kcl   * kcl_l2_k
                + self.lambda_kcl_inf * kcl_linf_k
                + self.lambda_lim   * lim_k
                + self.lambda_branch * branch_pen
                + self.lambda_v     * v_penalty
            )
            opf_loss = opf_loss + step_loss

            # Exact per-component accumulation (detached for monitoring)
            acc_gen_cost += float(gen_cost_k.detach()) * gamma_k
            acc_kcl      += float(kcl_k.detach())      * gamma_k
            acc_lim      += float(lim_k.detach())       * gamma_k
            acc_branch   += branch_pen_f                * gamma_k
            acc_v        += float(v_penalty.detach())   * gamma_k

            # Final-iteration constraint diagnostics (no grad)
            if k == self.K - 1:
                with torch.no_grad():
                    final_kcl_inf = float(
                        torch.maximum(DP_new.abs(), DQ_new.abs()).amax().item()
                    )
                    final_lim_max = float(
                        (torch.relu(Pg - Pg_max) + torch.relu(Pg_min - Pg))
                        .masked_fill(~pv_mask, 0.0).amax().item()
                    )
                    final_branch_max = self._branch_max_violation(
                        Vc_new, n_nodes_per_graph,
                        edge_index_dir, edge_feat_dir,
                        edge_index_dir_list, edge_feat_dir_list, B,
                    )
                    # Pre-clamp voltage violation: how much does the model want to
                    # exceed voltage limits before the hard projection clips it.
                    final_v_max = float(
                        torch.maximum(
                            torch.relu(v_new - self.v_max),
                            torch.relu(self.v_min - v_new),
                        ).amax().item()
                    )

        V_out = torch.stack([v, th], dim=-1)
        loss_dict = {
            "total":             opf_loss,      # differentiable tensor
            "gen_cost":          acc_gen_cost,  # float
            "kcl_loss":          acc_kcl,       # float
            "lim_loss":          acc_lim,       # float
            "branch_loss":       acc_branch,    # float
            "v_loss":            acc_v,         # float
            "final_kcl_inf":     final_kcl_inf,
            "final_branch_max":  final_branch_max,
            "final_lim_max":     final_lim_max,
            "final_v_max":       final_v_max,   # pre-clamp voltage violation
        }
        return V_out, Pg, loss_dict

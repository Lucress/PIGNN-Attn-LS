"""OPF adaptation of the PIGNN solver in `GNSMsg_SelfAttention_armijo.py`.

The power-flow solver is unrolled Newton with an Armijo line search on the
power-balance mismatch. Reusing it for AC-OPF is not a matter of swapping the
loss, because the power-flow version conflates two masks that the OPF problem
separates:

* **which equations are enforced.** In PF the injection is specified everywhere
  except the slack (P) and PV buses (Q), so the mismatch is driven to zero
  outside those. In OPF the dispatch is a *decision*: at any controllable bus
  the injection is free, and a residual there is not an error but the optimiser
  redispatching. Enforcing it would fight the solution.
* **which state variables may move.** In PF the slack angle and the PV/slack
  magnitudes are given, so the update is frozen there
  (`GNSMsg_SelfAttention_armijo.py:564-565`). In OPF the voltage profile is
  exactly what the optimiser chooses, so every bus is free except the angle
  reference, which stays pinned only to fix the gauge.

In PF these two coincide, which is why one `bus_type` drives both. In OPF they
do not, so this module carries them separately: `enforce` (from the OPF
decision space) and `free_th` / `free_v` (all buses, minus the angle datum).
Marking controllable buses as slack to reuse the parent masking would freeze
their voltages, which is precisely wrong.

Everything else --- the attention blocks, the per-iteration heads, the Y-bus
assembly, the residual-loss shaping --- is inherited unchanged from the parent
module, so the two solvers stay in sync.

The Armijo merit function is the **masked** mismatch infinity norm. Using the
unmasked one would make the line search reject steps that correctly redispatch.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from GNSMsg_SelfAttention_armijo import (
    GNSMsg_EdgeSelfAttn,
    _build_dense_Y_from_branchrows_single,
    _build_directed_edges_single,
)


def _masked_mismatch_inf_norm(Y, v, th, P_set, Q_set, enforce_p, enforce_q):
    """Worst enforceable power-balance violation.

    `enforce_*` is True where the injection is *known*, i.e. where a residual is
    a genuine error. At controllable buses it is False and the term drops out.
    """
    Vc = v * torch.exp(1j * th)
    if Y.dim() == 2:
        Y = Y.unsqueeze(0)
    Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
    Sc = Vc * Ic.conj()
    DP = (P_set - Sc.real) * enforce_p.to(P_set.dtype)
    DQ = (Q_set - Sc.imag) * enforce_q.to(Q_set.dtype)
    return torch.maximum(DP.abs().amax(dim=-1), DQ.abs().amax(dim=-1)).amax()


class GNSMsg_EdgeSelfAttn_OPF(GNSMsg_EdgeSelfAttn):
    """PIGNN solver driving an AC-OPF operating point.

    Constructor arguments are the parent's. `forward` differs in its masking and
    in taking the OPF decision space instead of a bus-type vector alone.
    """

    def forward(
        self,
        bus_type,
        Branch_f_bus,
        Branch_t_bus,
        Branch_status,
        Branch_tau,
        Branch_shift_deg,
        Branch_y_series_from,
        Branch_y_series_to,
        Branch_y_series_ft,
        Branch_y_shunt_from,
        Branch_y_shunt_to,
        Is_trafo,
        Y,
        S,                      # load-only injection (S_load), NOT a full setpoint
        V0,
        ctrl_mask,              # True where the OPF may redispatch
        ref_mask=None,          # angle datum; REQUIRED -- conventions differ
        n_nodes_per_graph=None,
        Y_shunt_bus=None,
        vn_log=None,
        v_min=None,             # per-bus band; falls back to the parent's scalars
        v_max=None,
        return_attn=False,      # if True, return collected attention weights
        return_attn_heads=False,# if True, keep per-head (E,H); else mean over heads (E,)
    ):
        device = bus_type.device
        B, N = bus_type.shape

        if Y is None:
            if Y_shunt_bus is None:
                raise ValueError("Y is None and Y_shunt_bus is also None; cannot reconstruct Y.")
            if B != 1:
                raise ValueError("Y reconstruction here supports blockdiag batching (B=1) only.")
            Y = _build_dense_Y_from_branchrows_single(
                N,
                Branch_f_bus.squeeze(0), Branch_t_bus.squeeze(0),
                Branch_status.squeeze(0), Branch_tau.squeeze(0),
                Branch_shift_deg.squeeze(0),
                Branch_y_series_from.squeeze(0), Branch_y_series_to.squeeze(0),
                Branch_y_series_ft.squeeze(0),
                Branch_y_shunt_from.squeeze(0), Branch_y_shunt_to.squeeze(0),
                Y_shunt_bus.squeeze(0) if Y_shunt_bus is not None else None,
            ).unsqueeze(0)

        if B == 1:
            edge_index_dir, edge_feat_dir = _build_directed_edges_single(
                Branch_f_bus.squeeze(0), Branch_t_bus.squeeze(0),
                Branch_status.squeeze(0), Branch_tau.squeeze(0),
                Branch_shift_deg.squeeze(0),
                Branch_y_series_from.squeeze(0), Branch_y_series_to.squeeze(0),
                Branch_y_series_ft.squeeze(0),
                Branch_y_shunt_from.squeeze(0), Branch_y_shunt_to.squeeze(0),
                Is_trafo.squeeze(0),
            )
        else:
            raise ValueError("OPF solver expects blockdiag batching (B=1).")

        # The edge features inherit the branch rows' float64; the attention
        # blocks are float32. Cast once here rather than at every block.
        edge_feat_dir = edge_feat_dir.to(V0.dtype)

        # The pipeline carries Y and S in complex128 while the network state is
        # float32, and torch will not mix the two in matmul. Align them on the
        # state's precision here; the metrics keep the caller's complex128 Y,
        # so scoring accuracy is unaffected by this cast.
        v = V0[..., 0].clone()
        th = V0[..., 1].clone()
        cdtype = torch.complex128 if v.dtype == torch.float64 else torch.complex64
        Y = Y.to(cdtype)
        S = S.to(cdtype)
        P_set, Q_set = S.real, S.imag
        m = torch.zeros(B, N, self.d, device=device, dtype=V0.dtype)

        ctrl = ctrl_mask.to(torch.bool)
        if ctrl.dim() == 1:
            ctrl = ctrl.unsqueeze(0)

        # Enforce balance only where the injection is known.
        enforce_p = ~ctrl
        enforce_q = ~ctrl

        # The optimiser owns the voltage profile, so nothing is frozen except
        # the angle datum -- kept fixed purely to remove the global-rotation
        # gauge freedom, exactly as the PF slack does.
        #
        # `ref_mask` must be supplied by the caller because the bus_type coding
        # is pipeline specific: the PPC branch-row schema uses 1=slack, while
        # OPFData follows MATPOWER with 1=PQ, 2=PV, 3=ref. Defaulting to the PF
        # convention silently froze the angle at every PQ bus on OPFData and
        # flatlined training, so guessing here is not acceptable.
        if ref_mask is None:
            raise ValueError(
                "ref_mask is required: bus_type coding differs per pipeline "
                "(PPC 1=slack, OPFData/MATPOWER 3=ref)."
            )
        ref_mask = ref_mask.to(torch.bool)
        if ref_mask.dim() == 1:
            ref_mask = ref_mask.unsqueeze(0)
        free_th = ~ref_mask
        free_v = torch.ones_like(ref_mask)

        if v_min is None:
            v_min = torch.full_like(v, 0.75)
        if v_max is None:
            v_max = torch.full_like(v, 1.20)

        phys_terms = []
        collected_attn = []  # list of (E, H) tensors, one per step

        for k in range(self.K):
            Vc = v * torch.exp(1j * th)
            Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
            Sc = Vc * Ic.conj()

            DP = (P_set - Sc.real) * enforce_p.to(P_set.dtype)
            DQ = (Q_set - Sc.imag) * enforce_q.to(Q_set.dtype)

            bus_feat = torch.stack([v, th, DP, DQ], dim=-1)
            if self.bus_feat_extra_dim > 0:
                if vn_log is None:
                    extra = bus_feat.new_zeros(bus_feat.shape[:-1] + (self.bus_feat_extra_dim,))
                else:
                    extra = vn_log.unsqueeze(-1)
                bus_feat = torch.cat([bus_feat, extra], dim=-1)
            x = self.in_proj(torch.cat([bus_feat, m], dim=-1))

            step_alpha = None
            for blk in self.blocks:
                if return_attn:
                    x, step_alpha = blk(x, edge_index_dir, edge_feat_dir, return_attn=True)
                else:
                    x = blk(x, edge_index_dir, edge_feat_dir)
            if return_attn and step_alpha is not None:
                raw = step_alpha[0].detach().cpu()  # (E, H)
                collected_attn.append(raw if return_attn_heads else raw.mean(dim=-1))

            dth = self.theta_head[k](x).squeeze(-1)
            dv = self.v_head[k](x).squeeze(-1)
            dm = torch.tanh(self.m_head[k](x))
            dm = F.layer_norm(dm, dm.shape[-1:])

            dth = dth * free_th.to(dth.dtype)
            dv = dv * free_v.to(dv.dtype)

            if self.v_limit:
                v_abs = v.abs()
                dth = torch.clamp(dth, -self.dtheta_max, self.dtheta_max)
                dv = torch.clamp(dv, -self.dvm_frac * v_abs, self.dvm_frac * v_abs)

            if self.use_armijo:
                with torch.no_grad():
                    F0 = _masked_mismatch_inf_norm(Y, v, th, P_set, Q_set,
                                                   enforce_p, enforce_q)

                max_backtracks = max(1, int(self.armijo_max_backtracks))
                rho = min(max(float(self.armijo_rho), 1e-12), 1.0 - 1e-12)
                c1 = float(self.armijo_c1)
                min_alpha = max(0.0, float(self.armijo_min_alpha))

                if self.armijo_mode == "fixed":
                    alphas = v.new_tensor([rho ** i for i in range(max_backtracks)])
                    if min_alpha > 0.0:
                        alphas = alphas[alphas >= min_alpha]
                    if alphas.numel() == 0:
                        alphas = v.new_tensor([min_alpha])
                else:
                    tmp, a_tmp = [], 1.0
                    for _ in range(max_backtracks):
                        tmp.append(a_tmp)
                        a_tmp *= rho
                        if min_alpha > 0.0 and a_tmp < min_alpha:
                            break
                    alphas = v.new_tensor(tmp)

                def candidate(a):
                    v_try = torch.clamp(v + a * dv, v_min, v_max)
                    th_try = (th + a * dth + math.pi) % (2 * math.pi) - math.pi
                    with torch.no_grad():
                        F_try = _masked_mismatch_inf_norm(Y, v_try, th_try, P_set,
                                                          Q_set, enforce_p, enforce_q)
                        ok = bool(F_try <= (1.0 - c1 * a) * F0)
                    return v_try, th_try, ok

                accepted, a_sel = False, float(alphas[-1])
                for a_tensor in alphas:
                    a = float(a_tensor)
                    _, _, ok = candidate(a)
                    if ok:
                        accepted, a_sel = True, a
                        break
                if self.armijo_mode == "geometric_safe" and not accepted:
                    a_sel = min_alpha
                if not (self.armijo_mode == "reject" and not accepted):
                    v = torch.clamp(v + a_sel * dv, v_min, v_max)
                    th = (th + a_sel * dth + math.pi) % (2 * math.pi) - math.pi
                    m = m + a_sel * dm
            else:
                th = (th + dth + math.pi) % (2 * math.pi) - math.pi
                v = torch.clamp(v + dv, v_min, v_max)
                m = m + dm

            if self.pinn:
                term = (self.gamma ** (self.K - 1 - k)) * self._physics_residual_loss(
                    DP, DQ, P_set, Q_set, enforce_p, enforce_q, n_nodes_per_graph
                )
                phys_terms.append(term)

        out = torch.stack([v, th], dim=-1)

        if self.pinn:
            if self.physics_final_weight != 0.0:
                Vc = v * torch.exp(1j * th)
                Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
                Sc = Vc * Ic.conj()
                DP = (P_set - Sc.real) * enforce_p.to(P_set.dtype)
                DQ = (Q_set - Sc.imag) * enforce_q.to(Q_set.dtype)
                phys_terms.append(self.physics_final_weight * self._physics_residual_loss(
                    DP, DQ, P_set, Q_set, enforce_p, enforce_q, n_nodes_per_graph))
            if return_attn:
                return out, torch.sum(torch.stack(phys_terms)), collected_attn, edge_index_dir.cpu()
            return out, torch.sum(torch.stack(phys_terms))
        if return_attn:
            return out, None, collected_attn, edge_index_dir.cpu()
        return out

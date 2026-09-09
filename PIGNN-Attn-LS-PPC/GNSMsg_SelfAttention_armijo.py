import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max
from collate_blockdiag_optimized_complex_columns import ybus_matvec
from known_operator_pf import adjoint_ybus_matvec


# --------------------------- utils ---------------------------

def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    return dtype

def _segmented_softmax(logits_b_e_h: torch.Tensor, dst_e: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Softmax over incoming edges per (batch, head, destination-node).
    logits_b_e_h: (B, E, H)
    dst_e:        (E,)
    """
    B, E, H = logits_b_e_h.shape
    device = logits_b_e_h.device
    N = num_nodes

    b_ids = torch.arange(B, device=device).view(B, 1, 1)
    h_ids = torch.arange(H, device=device).view(1, 1, H)
    dst = dst_e.view(1, E, 1)

    seg = (b_ids * (N * H)) + (h_ids * N) + dst
    seg = seg.reshape(-1)
    src = logits_b_e_h.reshape(-1)

    max_per_seg, _ = scatter_max(src, seg, dim=0, dim_size=B * N * H)
    max_g = max_per_seg.index_select(0, seg)
    x = torch.exp(src - max_g)

    denom = scatter_add(x, seg, dim=0, dim_size=B * N * H)
    denom_g = denom.index_select(0, seg)

    alpha = (x / (denom_g + 1e-12)).reshape(B, E, H)
    return alpha


def _batched_mismatch_inf_norm(Y, v, th, P_set, Q_set, slack_mask, pv_mask):
    Vc = v * torch.exp(1j * th)
    Ic = ybus_matvec(Y, Vc)
    Sc = Vc * Ic.conj()

    DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
    DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

    return torch.maximum(DP.abs().amax(dim=-1), DQ.abs().amax(dim=-1)).amax()


def _per_graph_mismatch_inf_norm(
    Y, v, th, P_set, Q_set, slack_mask, pv_mask, n_nodes_per_graph=None
):
    """Return one masked AC mismatch infinity norm per independent grid."""
    Vc = v * torch.exp(1j * th)
    Sc = Vc * ybus_matvec(Y, Vc).conj()
    DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
    DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

    if n_nodes_per_graph is None:
        return torch.maximum(DP.abs().amax(dim=-1), DQ.abs().amax(dim=-1))

    maxima = []
    offset = 0
    for size in n_nodes_per_graph.detach().cpu().tolist():
        size = int(size)
        sl = slice(offset, offset + size)
        maxima.append(torch.maximum(DP[0, sl].abs().amax(), DQ[0, sl].abs().amax()))
        offset += size
    return torch.stack(maxima)


def _expand_graph_values(values, reference, n_nodes_per_graph=None):
    """Expand one scalar per grid to the packed node layout."""
    if n_nodes_per_graph is None:
        return values.view(-1, 1).expand_as(reference)
    return torch.repeat_interleave(
        values, n_nodes_per_graph.to(device=values.device)
    ).view(1, -1)


def _reference_mse_gradient_scale(p_mask, q_mask, n_nodes_per_graph=None):
    """Derivative scale of the author's masked ``torch.nn.MSELoss``."""
    if n_nodes_per_graph is None:
        n_eq = (p_mask.sum(dim=-1) + q_mask.sum(dim=-1)).clamp_min(1)
        return (2.0 / n_eq).to(dtype=torch.float64).view(-1, 1)

    values = []
    offset = 0
    for size in n_nodes_per_graph.detach().cpu().tolist():
        size = int(size)
        sl = slice(offset, offset + size)
        n_eq = (p_mask[0, sl].sum() + q_mask[0, sl].sum()).clamp_min(1)
        values.append((2.0 / n_eq).expand(size))
        offset += size
    return torch.cat(values).view(1, -1).to(dtype=torch.float64)


def _normalize_per_graph(grad_vm, grad_va, n_nodes_per_graph=None):
    """Normalize gradient features independently for each packed grid."""
    if n_nodes_per_graph is None:
        scale = torch.maximum(
            grad_vm.abs().amax(dim=-1, keepdim=True),
            grad_va.abs().amax(dim=-1, keepdim=True),
        ).clamp_min(1e-12)
        return grad_vm / scale, grad_va / scale

    vm_parts = []
    va_parts = []
    offset = 0
    for size in n_nodes_per_graph.detach().cpu().tolist():
        size = int(size)
        sl = slice(offset, offset + size)
        scale = torch.maximum(
            grad_vm[0, sl].abs().amax(), grad_va[0, sl].abs().amax()
        ).clamp_min(1e-12)
        vm_parts.append(grad_vm[:, sl] / scale)
        va_parts.append(grad_va[:, sl] / scale)
        offset += size
    return torch.cat(vm_parts, dim=1), torch.cat(va_parts, dim=1)


def _build_dense_Y_from_branchrows_single(
    N: int,
    Branch_f_bus: torch.Tensor,
    Branch_t_bus: torch.Tensor,
    Branch_status: torch.Tensor,
    Branch_tau: torch.Tensor,
    Branch_shift_deg: torch.Tensor,
    Branch_y_series_from: torch.Tensor,
    Branch_y_series_to: torch.Tensor,
    Branch_y_series_ft: torch.Tensor,
    Branch_y_shunt_from: torch.Tensor,
    Branch_y_shunt_to: torch.Tensor,
    Y_shunt_bus: torch.Tensor,
) -> torch.Tensor:
    """
    Dense Ybus reconstruction from one metadata row per PPC branch row.
    """
    device = Branch_f_bus.device
    dtype = Branch_y_series_from.dtype

    Y = torch.zeros(N, N, dtype=dtype, device=device)
    Y.diagonal().add_(Y_shunt_bus.to(dtype))

    mask = (Branch_status != 0)
    if mask.sum() == 0:
        return Y

    f = Branch_f_bus[mask].long()
    t = Branch_t_bus[mask].long()

    real_dtype = _real_dtype(dtype)
    tau = Branch_tau[mask].to(real_dtype)
    theta = torch.deg2rad(Branch_shift_deg[mask].to(real_dtype))
    a = tau.to(dtype) * torch.exp(1j * theta.to(dtype))

    y_from = Branch_y_series_from[mask].to(dtype)
    y_to   = Branch_y_series_to[mask].to(dtype)
    ysh_f  = Branch_y_shunt_from[mask].to(dtype)
    ysh_t  = Branch_y_shunt_to[mask].to(dtype)

    Yff = (y_from + ysh_f / 2.0) / (a * torch.conj(a))
    Ytt = (y_to   + ysh_t / 2.0)
    Yft = -y_from / torch.conj(a)
    Ytf = -y_to / a

    Y.index_put_((f, f), Yff, accumulate=True)
    Y.index_put_((t, t), Ytt, accumulate=True)
    Y.index_put_((f, t), Yft, accumulate=True)
    Y.index_put_((t, f), Ytf, accumulate=True)

    return Y


def _build_directed_edges_single(
    Branch_f_bus: torch.Tensor,
    Branch_t_bus: torch.Tensor,
    Branch_status: torch.Tensor,
    Branch_tau: torch.Tensor,
    Branch_shift_deg: torch.Tensor,
    Branch_y_series_from: torch.Tensor,
    Branch_y_series_to: torch.Tensor,
    Branch_y_series_ft: torch.Tensor,
    Branch_y_shunt_from: torch.Tensor,
    Branch_y_shunt_to: torch.Tensor,
    Is_trafo: torch.Tensor,
):
    """
    Build directed sparse edge list and direction-aware edge features from branch rows.

    Edge feature for f->t:
      [Re(Yft_dir), Im(Yft_dir),
       Re(ysh_f),   Im(ysh_f),
       Re(ysh_t),   Im(ysh_t),
       tau, theta_rad, is_trafo]

    Edge feature for t->f swaps the shunts and flips theta sign.
    """
    device = Branch_f_bus.device
    ctype = Branch_y_series_from.dtype
    rtype = Branch_tau.dtype

    mask = (Branch_status != 0)
    if mask.sum() == 0:
        return (
            torch.empty(0, 2, dtype=torch.long, device=device),
            torch.empty(0, 9, dtype=rtype, device=device)
        )

    f = Branch_f_bus[mask].long()
    t = Branch_t_bus[mask].long()

    tau = Branch_tau[mask].to(rtype)
    theta = torch.deg2rad(Branch_shift_deg[mask].to(rtype))
    is_tr = Is_trafo[mask].to(rtype)

    a = tau.to(ctype) * torch.exp(1j * theta.to(ctype))

    y_from = Branch_y_series_from[mask].to(ctype)
    y_to = Branch_y_series_to[mask].to(ctype)
    ysh_f = Branch_y_shunt_from[mask].to(ctype)
    ysh_t = Branch_y_shunt_to[mask].to(ctype)

    ydir_ft = -y_from / torch.conj(a)  # f -> t
    ydir_tf = -y_to / a                # t -> f

    feat_ft = torch.stack([
        ydir_ft.real.to(rtype), ydir_ft.imag.to(rtype),
        ysh_f.real.to(rtype),   ysh_f.imag.to(rtype),
        ysh_t.real.to(rtype),   ysh_t.imag.to(rtype),
        tau, theta, is_tr
    ], dim=-1)

    feat_tf = torch.stack([
        ydir_tf.real.to(rtype), ydir_tf.imag.to(rtype),
        ysh_t.real.to(rtype),   ysh_t.imag.to(rtype),
        ysh_f.real.to(rtype),   ysh_f.imag.to(rtype),
        tau, -theta, is_tr
    ], dim=-1)

    edge_index = torch.cat([
        torch.stack([f, t], dim=1),
        torch.stack([t, f], dim=1),
    ], dim=0)

    edge_feat = torch.cat([feat_ft, feat_tf], dim=0)
    return edge_index, edge_feat


# --------------------- attention block -----------------------

class EdgeSelfAttnBlock(nn.Module):
    """
    Sparse graph self-attention with edge bias.
    """
    def __init__(self, d_model: int, n_heads: int, edge_feat_dim: int, ffn_hidden: int = None, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.h = n_heads
        self.dh = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.edge_bias = nn.Sequential(
            nn.Linear(edge_feat_dim, max(16, 2 * edge_feat_dim)),
            nn.LeakyReLU(0.1),
            nn.Linear(max(16, 2 * edge_feat_dim), self.h)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        hid = ffn_hidden or (4 * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index_dir: torch.Tensor, edge_feat_dir: torch.Tensor,
                return_attn: bool = False):
        """
        x:             (B, N, D)
        edge_index_dir:(E, 2)
        edge_feat_dir: (E, F)
        return_attn:   if True, also return alpha (B, E, H) — head-averaged edge weights
        """
        B, N, D = x.shape
        device = x.device
        src = edge_index_dir[:, 0]
        dst = edge_index_dir[:, 1]

        y = self.ln1(x)
        Q = self.q(y).view(B, N, self.h, self.dh)
        K = self.k(y).view(B, N, self.h, self.dh)
        V = self.v(y).view(B, N, self.h, self.dh)

        Qi = Q[:, dst, :, :]
        Kj = K[:, src, :, :]
        Vj = V[:, src, :, :]

        logits = (Qi * Kj).sum(dim=-1) / math.sqrt(self.dh)
        bias = self.edge_bias(edge_feat_dir).unsqueeze(0)
        logits = logits + bias

        alpha = _segmented_softmax(logits, dst, N)
        attn_msg = alpha.unsqueeze(-1) * Vj

        out = torch.zeros(B, N, self.h, self.dh, device=device, dtype=x.dtype)
        out.index_add_(1, dst, attn_msg)
        out = self.drop(self.out(out.reshape(B, N, D)))
        x = x + out

        z = self.ln2(x)
        z = self.drop(self.ffn(z))
        result = x + z
        if return_attn:
            return result, alpha  # alpha: (B, E, H)
        return result


# ------------------ main model ------------------

class GNSMsg_EdgeSelfAttn(nn.Module):
    def __init__(
        self,
        d: int = 10,
        d_hi: int = 32,
        K: int = 30,
        pinn: bool = True,
        gamma: float = 0.9,
        v_limit: bool = True,
        use_armijo: bool = True,
        d_model: int = None,
        n_heads: int = 4,
        num_attn_layers: int = 1,
        attn_dropout: float = 0.0,
        armijo_mode: str = "fixed",
        armijo_rho: float = 0.5,
        armijo_c1: float = 1e-4,
        armijo_max_backtracks: int = 5,
        armijo_min_alpha: float = 0.0625,
        bus_feat_extra_dim: int = 0,
        dtheta_max: float = 0.30,
        dvm_frac: float = 0.10,
        physics_loss_form: str = "mse",
        physics_residual_norm: str = "none",
        physics_norm_eps: float = 1e-6,
        physics_huber_delta: float = 1.0,
        physics_final_weight: float = 0.0,
        solver_update_mode: str = "direct",
        preconditioner_vm_step: float = 0.003377,
        preconditioner_va_step: float = 0.003377,
        preconditioner_log_clip: float = 5.0,
        preconditioner_optimizer: str = "author_adam",
        preconditioner_beta1: float = 0.979681,
        preconditioner_beta2: float = 0.963442,
        preconditioner_eps: float = 1e-8,
    ):
        super().__init__()
        self.K = K
        self.d = d
        self.d_hi = d_hi
        self.pinn = pinn
        self.gamma = gamma
        self.v_limit = v_limit
        self.use_armijo = use_armijo
        self.armijo_mode = armijo_mode
        self.armijo_rho = armijo_rho
        self.armijo_c1 = armijo_c1
        self.armijo_max_backtracks = armijo_max_backtracks
        self.armijo_min_alpha = armijo_min_alpha
        self.dtheta_max = dtheta_max
        self.dvm_frac = dvm_frac
        self.physics_loss_form = physics_loss_form
        self.physics_residual_norm = physics_residual_norm
        self.physics_norm_eps = physics_norm_eps
        self.physics_huber_delta = physics_huber_delta
        self.physics_final_weight = physics_final_weight
        if solver_update_mode not in ("direct", "physics_preconditioner"):
            raise ValueError(
                f"Unknown solver_update_mode={solver_update_mode!r}; expected "
                "'direct' or 'physics_preconditioner'."
            )
        if preconditioner_vm_step <= 0.0 or preconditioner_va_step <= 0.0:
            raise ValueError("Preconditioner initial step scales must be positive.")
        if preconditioner_log_clip <= 0.0:
            raise ValueError("preconditioner_log_clip must be positive.")
        if preconditioner_optimizer not in ("author_adam", "gd"):
            raise ValueError("preconditioner_optimizer must be 'author_adam' or 'gd'.")
        if not 0.0 <= preconditioner_beta1 < 1.0:
            raise ValueError("preconditioner_beta1 must lie in [0, 1).")
        if not 0.0 <= preconditioner_beta2 < 1.0:
            raise ValueError("preconditioner_beta2 must lie in [0, 1).")
        if preconditioner_eps <= 0.0:
            raise ValueError("preconditioner_eps must be positive.")
        self.solver_update_mode = solver_update_mode
        self.preconditioner_vm_step = preconditioner_vm_step
        self.preconditioner_va_step = preconditioner_va_step
        self.preconditioner_log_clip = preconditioner_log_clip
        self.preconditioner_optimizer = preconditioner_optimizer
        self.preconditioner_beta1 = preconditioner_beta1
        self.preconditioner_beta2 = preconditioner_beta2
        self.preconditioner_eps = preconditioner_eps

        self.d_model = d_model if d_model is not None else d_hi
        self.n_heads = n_heads
        assert self.d_model % self.n_heads == 0
        self.num_attn_layers = num_attn_layers

        self.bus_feat_extra_dim = int(bus_feat_extra_dim)
        # The integrated solver exposes the exact polar AC-PF gradient to the
        # graph network.  The direct predictor retains the historical feature
        # contract for checkpoint compatibility.
        physics_gradient_dim = 2 if solver_update_mode == "physics_preconditioner" else 0
        self.bus_feat_dim = 4 + physics_gradient_dim + self.bus_feat_extra_dim + d
        self.edge_feat_dim = 9   # <-- now includes tau + theta + is_trafo

        self.in_proj = nn.Linear(self.bus_feat_dim, self.d_model)
        self.blocks = nn.ModuleList([
            EdgeSelfAttnBlock(
                self.d_model,
                self.n_heads,
                self.edge_feat_dim,
                ffn_hidden=4 * self.d_model,
                dropout=attn_dropout
            )
            for _ in range(self.num_attn_layers)
        ])

        self.theta_head = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(K)])
        self.v_head     = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(K)])
        self.m_head     = nn.ModuleList([nn.Linear(self.d_model, d) for _ in range(K)])

        # Output heads: zero-init so the model starts with identity corrections
        # (Vpred = V_start at epoch 0), matching original behaviour.
        for k in range(K):
            nn.init.zeros_(self.theta_head[k].weight); nn.init.zeros_(self.theta_head[k].bias)
            nn.init.zeros_(self.v_head[k].weight);     nn.init.zeros_(self.v_head[k].bias)
            nn.init.zeros_(self.m_head[k].weight);     nn.init.zeros_(self.m_head[k].bias)

        # Feature extraction layers (in_proj + attention blocks): sd=0.02 normal
        # init, matching the student's weight_init="sd0.02" recipe. This gives
        # the attention layers a meaningful starting point so gradient flow is
        # better conditioned from the first update.
        for module in [self.in_proj] + list(self.blocks.modules()):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _graph_scale_from_s(self, S_abs, n_nodes_per_graph):
        if n_nodes_per_graph is None:
            return S_abs.amax(dim=-1, keepdim=True).clamp_min(self.physics_norm_eps)

        scales = torch.empty_like(S_abs)
        offset = 0
        for size in n_nodes_per_graph.tolist():
            size = int(size)
            sl = slice(offset, offset + size)
            scale = S_abs[:, sl].amax(dim=-1, keepdim=True).clamp_min(self.physics_norm_eps)
            scales[:, sl] = scale
            offset += size
        return scales

    def _physics_residual_loss(self, DP, DQ, P_set, Q_set, p_mask, q_mask, n_nodes_per_graph):
        if self.physics_residual_norm == "none":
            if self.physics_loss_form == "mse":
                return (DP ** 2 + DQ ** 2).mean()
            residual = torch.cat([DP.reshape(-1), DQ.reshape(-1)])
        else:
            S_abs = torch.sqrt(P_set ** 2 + Q_set ** 2)
            if self.physics_residual_norm == "setpoint":
                scale = S_abs.clamp_min(self.physics_norm_eps)
            elif self.physics_residual_norm == "graph":
                scale = self._graph_scale_from_s(S_abs, n_nodes_per_graph)
            else:
                raise ValueError(
                    f"Unknown physics_residual_norm={self.physics_residual_norm!r}; "
                    "expected 'none', 'setpoint', or 'graph'."
                )

            residual = torch.cat([
                (DP / scale)[p_mask],
                (DQ / scale)[q_mask],
            ])
            if residual.numel() == 0:
                return DP.sum() * 0.0

        if self.physics_loss_form == "mse":
            return (residual ** 2).mean()
        if self.physics_loss_form == "huber":
            return F.huber_loss(
                residual,
                torch.zeros_like(residual),
                delta=self.physics_huber_delta,
                reduction="mean",
            )
        if self.physics_loss_form == "logcosh":
            return (residual + F.softplus(-2.0 * residual) - math.log(2.0)).mean()

        raise ValueError(
            f"Unknown physics_loss_form={self.physics_loss_form!r}; "
            "expected 'mse', 'huber', or 'logcosh'."
        )

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
        S,
        V0,
        n_nodes_per_graph=None,
        Y_shunt_bus=None,
        vn_log=None,
    ):
        """
        Works for:
          - blockdiag batching (recommended): B=1, global node indices in branch rows
          - plain batching fallback: B>1, each batch item has local branch rows
        """
        device = bus_type.device
        B, N = bus_type.shape

        # reconstruct Y if needed
        if Y is None:
            if Y_shunt_bus is None:
                raise ValueError("Y is None and Y_shunt_bus is also None; cannot reconstruct Y.")
            if B == 1:
                Y = _build_dense_Y_from_branchrows_single(
                    N,
                    Branch_f_bus.squeeze(0),
                    Branch_t_bus.squeeze(0),
                    Branch_status.squeeze(0),
                    Branch_tau.squeeze(0),
                    Branch_shift_deg.squeeze(0),
                    Branch_y_series_from.squeeze(0),
                    Branch_y_series_to.squeeze(0),
                    Branch_y_series_ft.squeeze(0),
                    Branch_y_shunt_from.squeeze(0),
                    Branch_y_shunt_to.squeeze(0),
                    Y_shunt_bus.squeeze(0),
                ).unsqueeze(0)
            else:
                Ys = []
                for b in range(B):
                    Ys.append(_build_dense_Y_from_branchrows_single(
                        N,
                        Branch_f_bus[b], Branch_t_bus[b], Branch_status[b],
                        Branch_tau[b], Branch_shift_deg[b],
                        Branch_y_series_from[b], Branch_y_series_to[b], Branch_y_series_ft[b],
                        Branch_y_shunt_from[b], Branch_y_shunt_to[b],
                        Y_shunt_bus[b],
                    ))
                Y = torch.stack(Ys, dim=0)
        else:
            if Y.dim() == 2 and not Y.is_sparse:
                Y = Y.unsqueeze(0)

        # -------- build sparse attention graph --------
        if B == 1:
            edge_index_dir, edge_feat_dir = _build_directed_edges_single(
                Branch_f_bus.squeeze(0),
                Branch_t_bus.squeeze(0),
                Branch_status.squeeze(0),
                Branch_tau.squeeze(0),
                Branch_shift_deg.squeeze(0),
                Branch_y_series_from.squeeze(0),
                Branch_y_series_to.squeeze(0),
                Branch_y_series_ft.squeeze(0),
                Branch_y_shunt_from.squeeze(0),
                Branch_y_shunt_to.squeeze(0),
                Is_trafo.squeeze(0),
            )
            edge_index_dir_list = None
            edge_feat_dir_list = None
        else:
            edge_index_dir = None
            edge_feat_dir = None
            edge_index_dir_list = []
            edge_feat_dir_list = []
            for b in range(B):
                ei, ef = _build_directed_edges_single(
                    Branch_f_bus[b],
                    Branch_t_bus[b],
                    Branch_status[b],
                    Branch_tau[b],
                    Branch_shift_deg[b],
                    Branch_y_series_from[b],
                    Branch_y_series_to[b],
                    Branch_y_series_ft[b],
                    Branch_y_shunt_from[b],
                    Branch_y_shunt_to[b],
                    Is_trafo[b],
                )
                edge_index_dir_list.append(ei)
                edge_feat_dir_list.append(ef)

        # -------- initialize states --------
        P_set, Q_set = S.real, S.imag
        v = V0[..., 0].clone()
        th = V0[..., 1].clone()
        m = torch.zeros(B, N, self.d, device=device, dtype=V0.dtype)

        slack_mask = (bus_type == 1)
        pv_mask = (bus_type == 2)
        p_mask = ~slack_mask
        q_mask = ~(slack_mask | pv_mask)

        phys_terms = []
        dpf_m_vm = torch.zeros_like(v)
        dpf_m_va = torch.zeros_like(th)
        dpf_v_vm = torch.zeros_like(v)
        dpf_v_va = torch.zeros_like(th)

        # ------------------------- K iterations -------------------------
        for k in range(self.K):
            Vc = v * torch.exp(1j * th)
            Ic = ybus_matvec(Y, Vc)
            Sc = Vc * Ic.conj()

            DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
            DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

            grad_vm = None
            grad_va = None
            dpf_direction_vm = None
            dpf_direction_va = None
            if self.solver_update_mode == "physics_preconditioner":
                # Exact gradient of the author's masked torch.nn.MSELoss:
                # P at PV/PQ buses and Q at PQ buses. Each block-diagonal grid
                # is scaled independently so batching cannot couple its solve.
                rP = (-DP).to(Vc.dtype)
                rQ = (-DQ).to(Vc.dtype)
                mse_scale = _reference_mse_gradient_scale(
                    p_mask, q_mask, n_nodes_per_graph
                ).to(device=v.device, dtype=v.dtype)
                w = rP * mse_scale + 1j * (rQ * mse_scale)
                g_complex = w * Ic + adjoint_ybus_matvec(Y, w.conj() * Vc)
                phase = torch.exp(1j * th.to(Vc.dtype))
                grad_vm = (g_complex.conj() * phase).real.to(v.dtype)
                grad_va = (g_complex.conj() * (1j * Vc)).real.to(th.dtype)
                grad_vm = grad_vm.masked_fill(slack_mask | pv_mask, 0.0)
                grad_va = grad_va.masked_fill(slack_mask, 0.0)

                if self.preconditioner_optimizer == "author_adam":
                    beta1 = self.preconditioner_beta1
                    beta2 = self.preconditioner_beta2
                    dpf_m_vm = beta1 * dpf_m_vm + (1.0 - beta1) * grad_vm
                    dpf_m_va = beta1 * dpf_m_va + (1.0 - beta1) * grad_va
                    dpf_v_vm = beta2 * dpf_v_vm + (1.0 - beta2) * grad_vm.square()
                    dpf_v_va = beta2 * dpf_v_va + (1.0 - beta2) * grad_va.square()
                    bias1 = 1.0 - beta1 ** (k + 1)
                    bias2 = 1.0 - beta2 ** (k + 1)
                    dpf_direction_vm = (dpf_m_vm / bias1) / (
                        torch.sqrt(dpf_v_vm / bias2) + self.preconditioner_eps
                    )
                    dpf_direction_va = (dpf_m_va / bias1) / (
                        torch.sqrt(dpf_v_va / bias2) + self.preconditioner_eps
                    )
                else:
                    dpf_direction_vm = grad_vm
                    dpf_direction_va = grad_va

                # Only the neural features are normalized; the actual update
                # below remains the reference optimizer direction.
                grad_vm, grad_va = _normalize_per_graph(
                    grad_vm, grad_va, n_nodes_per_graph
                )

            bus_parts = [v, th, DP, DQ]
            if grad_vm is not None:
                bus_parts.extend([grad_vm, grad_va])
            bus_feat = torch.stack(bus_parts, dim=-1)
            if self.bus_feat_extra_dim > 0:
                if vn_log is None:
                    extra = bus_feat.new_zeros(bus_feat.shape[:-1] + (self.bus_feat_extra_dim,))
                else:
                    extra = vn_log.unsqueeze(-1)
                bus_feat = torch.cat([bus_feat, extra], dim=-1)
            x = self.in_proj(torch.cat([bus_feat, m], dim=-1))

            # sparse attention message passing
            if B == 1:
                for blk in self.blocks:
                    x = blk(x, edge_index_dir, edge_feat_dir)
            else:
                x_new = x.clone()
                for b in range(B):
                    xb = x[b:b + 1]
                    ei = edge_index_dir_list[b]
                    ef = edge_feat_dir_list[b]
                    if ei.numel() == 0:
                        x_new[b:b + 1] = xb
                        continue
                    for blk in self.blocks:
                        xb = blk(xb, ei, ef)
                    x_new[b:b + 1] = xb
                x = x_new

            theta_raw = self.theta_head[k](x).squeeze(-1)
            v_raw = self.v_head[k](x).squeeze(-1)
            if self.solver_update_mode == "physics_preconditioner":
                clip = self.preconditioner_log_clip
                theta_scale = self.preconditioner_va_step * torch.exp(
                    torch.clamp(theta_raw, -clip, clip)
                )
                v_scale = self.preconditioner_vm_step * torch.exp(
                    torch.clamp(v_raw, -clip, clip)
                )
                # PIGNN learns positive coordinate-wise multipliers around the
                # author's DPF optimizer direction. The residual safeguard
                # below accepts or rejects this proposal independently per grid.
                dth = -theta_scale * dpf_direction_va
                dv = -v_scale * dpf_direction_vm
            else:
                dth = theta_raw
                dv = v_raw
            dm  = torch.tanh(self.m_head[k](x))
            dm = F.layer_norm(dm, dm.shape[-1:])

            dth = dth.masked_fill(slack_mask, 0.0)
            dv  = dv.masked_fill(slack_mask | pv_mask, 0.0)

            if self.v_limit:
                v_abs = v.abs()
                dth = torch.clamp(dth, -self.dtheta_max, self.dtheta_max)
                dv = torch.clamp(dv, -self.dvm_frac * v_abs, self.dvm_frac * v_abs)

            if self.use_armijo:
                v_min, v_max = 0.75, 1.20

                if self.solver_update_mode == "physics_preconditioner":
                    with torch.no_grad():
                        F0_graph = _per_graph_mismatch_inf_norm(
                            Y, v, th, P_set, Q_set, slack_mask, pv_mask,
                            n_nodes_per_graph,
                        )

                    max_backtracks = max(1, int(self.armijo_max_backtracks))
                    rho = min(max(float(self.armijo_rho), 1e-12), 1.0 - 1e-12)
                    c1 = float(self.armijo_c1)
                    min_alpha = max(0.0, float(self.armijo_min_alpha))
                    alpha_values = []
                    a_tmp = 1.0
                    for _ in range(max_backtracks):
                        alpha_values.append(a_tmp)
                        a_tmp *= rho
                        if min_alpha > 0.0 and a_tmp < min_alpha:
                            break

                    selected = F0_graph.new_zeros(F0_graph.shape)
                    accepted = torch.zeros_like(F0_graph, dtype=torch.bool)
                    for a in alpha_values:
                        v_try = torch.clamp(v + a * dv, v_min, v_max)
                        th_try = (th + a * dth + math.pi) % (2 * math.pi) - math.pi
                        with torch.no_grad():
                            F_try_graph = _per_graph_mismatch_inf_norm(
                                Y, v_try, th_try, P_set, Q_set,
                                slack_mask, pv_mask, n_nodes_per_graph,
                            )
                            ok = F_try_graph <= (1.0 - c1 * a) * F0_graph
                            take = (~accepted) & ok
                            selected = torch.where(
                                take, selected.new_full(selected.shape, a), selected
                            )
                            accepted = accepted | take

                    if self.armijo_mode == "geometric_safe":
                        selected = torch.where(
                            accepted, selected, selected.new_full(selected.shape, min_alpha)
                        )
                    # For the integrated solver, fixed/geometric/reject all
                    # reject a grid if no candidate decreases its physical
                    # residual. This is the inference-time safety property.
                    alpha_node = _expand_graph_values(
                        selected, v, n_nodes_per_graph
                    )
                    v = torch.clamp(v + alpha_node * dv, v_min, v_max)
                    th = (th + alpha_node * dth + math.pi) % (2 * math.pi) - math.pi
                    m = m + alpha_node.unsqueeze(-1) * dm
                    if self.pinn:
                        term = (self.gamma ** (self.K - 1 - k)) * self._physics_residual_loss(
                            DP, DQ, P_set, Q_set, p_mask, q_mask, n_nodes_per_graph
                        )
                        phys_terms.append(term)
                    continue

                with torch.no_grad():
                    F0 = _batched_mismatch_inf_norm(Y, v, th, P_set, Q_set, slack_mask, pv_mask)

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
                elif self.armijo_mode in ("geometric", "geometric_safe", "reject"):
                    alphas = []
                    a_tmp = 1.0
                    for _ in range(max_backtracks):
                        alphas.append(a_tmp)
                        a_tmp *= rho
                        if min_alpha > 0.0 and a_tmp < min_alpha:
                            break
                    alphas = v.new_tensor(alphas)
                else:
                    raise ValueError(f"Unknown armijo_mode={self.armijo_mode!r}; expected 'fixed', 'geometric', 'geometric_safe', or 'reject'.")

                def armijo_candidate(a):
                    v_try = torch.clamp(v + a * dv, v_min, v_max)
                    th_try = (th + a * dth + math.pi) % (2 * math.pi) - math.pi
                    with torch.no_grad():
                        F_try = _batched_mismatch_inf_norm(Y, v_try, th_try, P_set, Q_set, slack_mask, pv_mask)
                        ok = bool(F_try <= (1.0 - c1 * a) * F0)
                    return v_try, th_try, ok

                if self.armijo_mode == "fixed":
                    accepted = False
                    for a_tensor in alphas:
                        a = float(a_tensor)
                        v_try, th_try, ok = armijo_candidate(a)
                        if ok:
                            v = v_try
                            th = th_try
                            m = m + a * dm
                            accepted = True
                            break

                    if not accepted:
                        a = float(alphas[-1])
                        v2 = torch.clamp(v + a * dv, v_min, v_max)
                        th2 = (th + a * dth + math.pi) % (2 * math.pi) - math.pi
                        with torch.no_grad():
                            ok = _batched_mismatch_inf_norm(Y, v2, th2, P_set, Q_set, slack_mask, pv_mask) < F0
                        if ok:
                            v, th, m = v2, th2, m + a * dm
                else:
                    accepted = False
                    a_sel = float(alphas[-1])
                    for a_tensor in alphas:
                        a = float(a_tensor)
                        _, _, ok = armijo_candidate(a)
                        if ok:
                            accepted = True
                            a_sel = a
                            break

                    if self.armijo_mode == "geometric_safe" and not accepted:
                        a_sel = min_alpha

                    if self.armijo_mode == "reject" and not accepted:
                        pass
                    else:
                        # For geometric, match the LVN/student Armijo semantics:
                        # the line search only selects a scalar step size. In
                        # geometric_safe, rejected searches use a tiny configured
                        # fallback to keep gradients alive while staying near old
                        # PPC. In reject mode, rejected searches discard the
                        # update entirely.
                        v = torch.clamp(v + a_sel * dv, v_min, v_max)
                        th = (th + a_sel * dth + math.pi) % (2 * math.pi) - math.pi
                        m = m + a_sel * dm
            else:
                th = (th + dth + math.pi) % (2 * math.pi) - math.pi
                v = torch.clamp(v + dv, 0.75, 1.20)
                m = m + dm

            if self.pinn:
                term = (self.gamma ** (self.K - 1 - k)) * self._physics_residual_loss(
                    DP, DQ, P_set, Q_set, p_mask, q_mask, n_nodes_per_graph
                )
                phys_terms.append(term)

        out = torch.stack([v, th], dim=-1)

        if self.pinn:
            if self.physics_final_weight != 0.0:
                Vc = v * torch.exp(1j * th)
                Ic = ybus_matvec(Y, Vc)
                Sc = Vc * Ic.conj()
                DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
                DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)
                final_term = self._physics_residual_loss(
                    DP, DQ, P_set, Q_set, p_mask, q_mask, n_nodes_per_graph
                )
                phys_terms.append(self.physics_final_weight * final_term)
            phys_loss = torch.sum(torch.stack(phys_terms))
            return out, phys_loss
        else:
            return out

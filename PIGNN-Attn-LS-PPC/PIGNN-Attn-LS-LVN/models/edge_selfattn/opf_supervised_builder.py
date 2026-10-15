"""Builder for GNSMsg_EdgeSelfAttn_OPF_Supervised (Model 1).

Same architecture as GNSMsg_EdgeSelfAttn_OPF (Model 2) but registered under
a distinct name so entrypoint.py can dispatch to the supervised training loop.

Self-registers in MODEL_REGISTRY when this module is imported.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.registry import register_model
from .opf_model import GNSMsg_EdgeSelfAttn_OPF


@register_model("GNSMsg_EdgeSelfAttn_OPF_Supervised")
def _build_opf_supervised(
    *,
    d: int,
    d_hi: int,
    K: int,
    pinn: bool,
    gamma: float,
    v_limit: bool,
    use_armijo: bool,
    dtheta_max: float,
    dvm_frac: float,
    num_attn_layers: int,
    device: torch.device,
    lambda_cost: float = 1.0,
    lambda_kcl: float = 1.0,
    lambda_lim: float = 10.0,
    lambda_branch: float = 1.0,
    lambda_v: float = 1.0,
    pg_step_frac: float = 0.05,
    c1_default: float = 2.0,
    c2_default: float = 0.5,
    pg_lim_frac: float = 0.30,
    s_max_pu: float = 1.0,
    **_unused,
) -> nn.Module:
    """Build GNSMsg_EdgeSelfAttn_OPF_Supervised (identical architecture to Model 2)."""
    return GNSMsg_EdgeSelfAttn_OPF(
        d=d, d_hi=d_hi, K=K, pinn=pinn, gamma=gamma,
        v_limit=v_limit, use_armijo=use_armijo,
        dtheta_max=dtheta_max, dvm_frac=dvm_frac,
        num_attn_layers=num_attn_layers,
        lambda_cost=lambda_cost,
        lambda_kcl=lambda_kcl,
        lambda_lim=lambda_lim,
        lambda_branch=lambda_branch,
        lambda_v=lambda_v,
        pg_step_frac=pg_step_frac,
        c1_default=c1_default,
        c2_default=c2_default,
        pg_lim_frac=pg_lim_frac,
        s_max_pu=s_max_pu,
    ).to(device)

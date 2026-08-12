"""Metrics that say whether a surrogate tracks the solver or collapses to a constant.

Motivated by GridSFM's own critique of MAE/MAPE (arXiv white paper, section 6.2,
"Beyond MAE"):

  * MAE is small whenever the quantity has a narrow natural range. Bus voltage
    magnitude in pu only spans ~0.95-1.10, so a flat predictor of V = 1.0 already
    posts a tiny MAE while carrying no information about the voltage profile.
  * MAPE is dominated by the small-denominator tail.
  * Neither says whether the model follows the solver's variation.

Their fix, reproduced here: a per-channel pooled linear regression of predicted
against reference values. Slope 1 means the model tracks the reference spread
one-for-one; slope well below 1 means it is collapsing toward the mean. R^2 says
how much of the reference variance is explained.

This is not academic for this project: a GridSFM run on OPFData case118 was
found to have collapsed its magnitude head only because two independently
initialised runs reported a |V| RMSE identical to five significant figures.
Slope ~ 0 with R^2 ~ 0 would have shown it directly, on a single run.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1).to(torch.float64)


def regression_diag(pred: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """Least-squares fit of pred = a * ref + b, pooled over everything given.

    Returns slope, intercept, Pearson R, R^2, and the two standard deviations
    whose ratio is the blunt version of the same question.
    """
    p, r = _flat(pred), _flat(ref)
    n = p.numel()
    if n < 2:
        return {"slope": float("nan"), "intercept": float("nan"),
                "R": float("nan"), "R2": float("nan"),
                "std_pred": float("nan"), "std_ref": float("nan"), "n": int(n)}
    pm, rm = p.mean(), r.mean()
    dp, dr = p - pm, r - rm
    var_r = float((dr * dr).mean())
    var_p = float((dp * dp).mean())
    cov = float((dp * dr).mean())
    slope = cov / var_r if var_r > 0 else float("nan")
    denom = (var_p * var_r) ** 0.5
    if denom > 0:
        R = cov / denom
    elif var_p == 0.0:
        # A constant prediction explains none of the reference variance. R is
        # formally undefined here, but reporting nan invites reading it as a
        # failed computation when it is in fact the collapse this diagnostic
        # exists to detect; 0 is the meaningful value.
        R = 0.0
    else:
        R = float("nan")
    return {
        "slope": float(slope),
        "intercept": float(rm.new_tensor(0.0) + (pm - slope * rm)) if var_r > 0 else float("nan"),
        "R": float(R),
        "R2": float(R * R) if R == R else float("nan"),
        "std_pred": float(var_p ** 0.5),
        "std_ref": float(var_r ** 0.5),
        "n": int(n),
    }


def angle_wrap(d: torch.Tensor) -> torch.Tensor:
    """Wrap an angle difference into (-pi, pi]."""
    return torch.atan2(torch.sin(d), torch.cos(d))


def voltage_mae(V_pred: torch.Tensor, V_ref: torch.Tensor) -> Dict[str, float]:
    """MAE alongside RMSE for magnitude and angle.

    Reported together on purpose: MAE describes the typical bus, RMSE is pulled
    by the tail, and their ratio is a cheap read on how heavy that tail is.
    Angle error is wrapped before averaging, so a prediction near -pi against a
    reference near +pi counts as small rather than as 2*pi.
    """
    vp, vr = V_pred[..., 0].to(torch.float64), V_ref[..., 0].to(torch.float64)
    ap, ar = V_pred[..., 1].to(torch.float64), V_ref[..., 1].to(torch.float64)
    dmag = (vp - vr).abs()
    dang = angle_wrap(ap - ar).abs()
    deg = 180.0 / torch.pi
    return {
        "mae_vmag_pu": float(dmag.mean()),
        "rmse_vmag_pu": float((dmag * dmag).mean() ** 0.5),
        "mae_theta_deg": float(dang.mean() * deg),
        "rmse_theta_deg": float(((dang * dang).mean() ** 0.5) * deg),
        "medae_vmag_pu": float(dmag.median()),
        "medae_theta_deg": float(dang.median() * deg),
    }


def voltage_regression(V_pred: torch.Tensor, V_ref: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Per-channel regression diagnostics for the voltage state."""
    return {
        "vmag": regression_diag(V_pred[..., 0], V_ref[..., 0]),
        # Angles are regressed on their sine and cosine rather than the raw
        # value: a wrapped angle is circular, and a raw fit would be corrupted
        # by any pair straddling the branch cut.
        "theta_sin": regression_diag(torch.sin(V_pred[..., 1]), torch.sin(V_ref[..., 1])),
        "theta_cos": regression_diag(torch.cos(V_pred[..., 1]), torch.cos(V_ref[..., 1])),
    }


def format_diagnostics(mae: Dict[str, float],
                       reg: Dict[str, Dict[str, float]]) -> str:
    v, ts, tc = reg["vmag"], reg["theta_sin"], reg["theta_cos"]
    return (
        f"MAE |V| {mae['mae_vmag_pu']:.4e} pu, theta {mae['mae_theta_deg']:.4e} deg | "
        f"median |V| {mae['medae_vmag_pu']:.4e}, theta {mae['medae_theta_deg']:.4e} deg\n"
        f"Regression  |V| : slope {v['slope']:.4f} R {v['R']:.4f} R2 {v['R2']:.4f} "
        f"(std pred {v['std_pred']:.4e} vs ref {v['std_ref']:.4e})\n"
        f"Regression sin(theta): slope {ts['slope']:.4f} R2 {ts['R2']:.4f} | "
        f"cos(theta): slope {tc['slope']:.4f} R2 {tc['R2']:.4f}"
    )

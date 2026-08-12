import argparse
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag


def _real_dtype(dtype):
    return torch.float64 if dtype == torch.complex128 else torch.float32


def build_y_old_torch(N, b):
    y_from = b["Branch_y_series_from"]
    y_to = b["Branch_y_series_to"]
    dtype = y_from.dtype
    device = y_from.device
    Y = torch.zeros(N, N, dtype=dtype, device=device)
    Y.diagonal().add_(b["Y_shunt_bus"].to(dtype))
    mask = b["Branch_status"] != 0
    f = b["Branch_f_bus"][mask].long()
    t = b["Branch_t_bus"][mask].long()
    tau = b["Branch_tau"][mask].to(_real_dtype(dtype))
    theta = torch.deg2rad(b["Branch_shift_deg"][mask].to(_real_dtype(dtype)))
    a = tau.to(dtype) * torch.exp(1j * theta.to(dtype))
    y_from = y_from[mask].to(dtype)
    y_to = y_to[mask].to(dtype)
    ysh_f = b["Branch_y_shunt_from"][mask].to(dtype)
    ysh_t = b["Branch_y_shunt_to"][mask].to(dtype)
    Y.index_put_((f, f), (y_from + ysh_f / 2) / (a * torch.conj(a)), accumulate=True)
    Y.index_put_((t, t), y_to + ysh_t / 2, accumulate=True)
    Y.index_put_((f, t), -y_from / torch.conj(a), accumulate=True)
    Y.index_put_((t, f), -y_to / a, accumulate=True)
    return Y


def build_y_corrected_torch(N, b):
    y_from = b["Branch_y_series_from"]
    dtype = y_from.dtype
    device = y_from.device
    Y = torch.zeros(N, N, dtype=dtype, device=device)
    Y.diagonal().add_(b["Y_shunt_bus"].to(dtype))
    mask = b["Branch_status"] != 0
    f = b["Branch_f_bus"][mask].long()
    t = b["Branch_t_bus"][mask].long()
    tau = b["Branch_tau"][mask].to(_real_dtype(dtype))
    theta = torch.deg2rad(b["Branch_shift_deg"][mask].to(_real_dtype(dtype)))
    a = tau.to(dtype) * torch.exp(1j * theta.to(dtype))
    y_from = b["Branch_y_series_from"][mask].to(dtype)
    y_to = b["Branch_y_series_to"][mask].to(dtype)
    y_ft = b["Branch_y_series_ft"][mask].to(dtype)
    # In per-unit, Branch_y_series_ft is scaled by Vf*Vt/Sbase. The reciprocal
    # direction is represented by the conjugate row's own from/to scaling in
    # pandapower branch-row metadata; for this LVN parquet y_ft is symmetric.
    y_tf = y_ft
    ysh_f = b["Branch_y_shunt_from"][mask].to(dtype)
    ysh_t = b["Branch_y_shunt_to"][mask].to(dtype)
    Y.index_put_((f, f), (y_from + ysh_f / 2) / (a * torch.conj(a)), accumulate=True)
    Y.index_put_((t, t), y_to + ysh_t / 2, accumulate=True)
    Y.index_put_((f, t), -y_ft / torch.conj(a), accumulate=True)
    Y.index_put_((t, f), -y_tf / a, accumulate=True)
    return Y


def residual(Y, V, S, bus_type):
    if Y.dim() == 2:
        Y = Y.unsqueeze(0)
    real_dtype = _real_dtype(Y.dtype)
    v = V[..., 0].to(real_dtype)
    th = V[..., 1].to(real_dtype)
    Vc = v * torch.exp(1j * th)
    Sc = Vc * torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1).conj()
    p_mask = bus_type != 1
    q_mask = (bus_type != 1) & (bus_type != 2)
    dp = (S.real - Sc.real).abs().masked_fill(~p_mask, 0)
    dq = (S.imag - Sc.imag).abs().masked_fill(~q_mask, 0)
    return dp.amax(dim=-1), dq.amax(dim=-1)


def summarize(vals):
    a = torch.cat([v.detach().cpu().float().reshape(-1) for v in vals]).numpy()
    return f"mean={a.mean():.6e} p50={np.percentile(a,50):.6e} p95={np.percentile(a,95):.6e} max={a.max():.6e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--PARQUET", nargs="+", required=True)
    ap.add_argument("--BATCH", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=1)
    ap.add_argument("--target_S_base", type=float, default=1e8)
    ap.add_argument("--complex_dtype", default="complex64", choices=["complex64", "complex128"])
    ap.add_argument("--share_grid", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    ds = ChanghunDataset(
        args.PARQUET,
        per_unit=True,
        device=None,
        target_S_base=args.target_S_base,
        share_grid=args.share_grid,
        complex_dtype=args.complex_dtype,
    )
    loader = DataLoader(ds, batch_size=args.BATCH, shuffle=False, collate_fn=collate_blockdiag)
    stats = {}
    mag_sse = ang_sse = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if args.max_batches and i >= args.max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        bus_type = batch["bus_type"]
        Vstart = batch["V_start"]
        Vnewton = batch["V_newton"]
        Sstart = batch["S_start"]
        Snewton = batch["S_newton"]
        dmag = Vstart[..., 0] - Vnewton[..., 0]
        dang = torch.atan2(torch.sin(Vstart[..., 1] - Vnewton[..., 1]), torch.cos(Vstart[..., 1] - Vnewton[..., 1]))
        mag_sse += float((dmag ** 2).sum().item())
        ang_sse += float((dang ** 2).sum().item())
        n += int(dmag.numel())
        Y_cached = batch["Ybus"]
        b0 = {k: v[0] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == args.BATCH else v for k, v in batch.items()}
        # Collated branch arrays are stacked for normal batching.
        b0 = {k: (v[0] if torch.is_tensor(v) and v.dim() >= 1 and k not in ("Ybus", "sizes") else v) for k, v in batch.items()}
        N = int(bus_type.shape[-1])
        Y_old = build_y_old_torch(N, b0).unsqueeze(0).expand_as(Y_cached)
        Y_corr = build_y_corrected_torch(N, b0).unsqueeze(0).expand_as(Y_cached)
        for name, Y in [("cached", Y_cached), ("old_torch_reconstruct", Y_old), ("corrected_torch_reconstruct", Y_corr)]:
            diff = (Y - Y_cached).abs().max().detach().cpu().item()
            print(f"batch={i} Ydiff_vs_cached {name}: {diff:.6e}")
            for vname, V, sname, S in [
                ("V_start", Vstart, "S_start", Sstart),
                ("V_newton", Vnewton, "S_newton", Snewton),
                ("V_newton", Vnewton, "S_start", Sstart),
            ]:
                dp, dq = residual(Y, V, S, bus_type)
                stats.setdefault((name, vname, sname, "dp"), []).append(dp)
                stats.setdefault((name, vname, sname, "dq"), []).append(dq)
    print(f"rows={len(ds)} checked_graphs={n // 722 if n else 0} nodes={n}")
    print(f"V_start_vs_newton_mag_rmse={math.sqrt(mag_sse/max(n,1)):.6e}")
    print(f"V_start_vs_newton_ang_rmse_deg={math.sqrt(ang_sse/max(n,1))*180/math.pi:.6e}")
    for key in sorted(stats):
        print("|".join(key), summarize(stats[key]))


if __name__ == "__main__":
    main()

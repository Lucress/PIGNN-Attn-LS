import argparse
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    return dtype


def build_dense_y_from_branchrows_single(
    N,
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
    Y_shunt_bus,
):
    device = Branch_f_bus.device
    dtype = Branch_y_series_ft.dtype

    Y = torch.zeros(N, N, dtype=dtype, device=device)
    Y.diagonal().add_(Y_shunt_bus.to(dtype))

    mask = Branch_status != 0
    if mask.sum() == 0:
        return Y

    f = Branch_f_bus[mask].long()
    t = Branch_t_bus[mask].long()

    real_dtype = _real_dtype(dtype)
    tau = Branch_tau[mask].to(real_dtype)
    theta = torch.deg2rad(Branch_shift_deg[mask].to(real_dtype))
    a = tau.to(dtype) * torch.exp(1j * theta.to(dtype))

    y_from = Branch_y_series_from[mask].to(dtype)
    y_to = Branch_y_series_to[mask].to(dtype)
    ysh_f = Branch_y_shunt_from[mask].to(dtype)
    ysh_t = Branch_y_shunt_to[mask].to(dtype)

    Yff = (y_from + ysh_f / 2.0) / (a * torch.conj(a))
    Ytt = y_to + ysh_t / 2.0
    Yft = -y_from / torch.conj(a)
    Ytf = -y_to / a

    Y.index_put_((f, f), Yff, accumulate=True)
    Y.index_put_((t, t), Ytt, accumulate=True)
    Y.index_put_((f, t), Yft, accumulate=True)
    Y.index_put_((t, f), Ytf, accumulate=True)

    return Y


def ensure_dense_y(batch, device):
    Y = batch.get("Ybus", None)
    if Y is not None:
        Y = Y.to(device)
        return Y.unsqueeze(0) if Y.dim() == 2 else Y

    bus_type = batch["bus_type"].to(device)
    Branch_f_bus = batch["Branch_f_bus"].to(device)
    Branch_t_bus = batch["Branch_t_bus"].to(device)
    Branch_status = batch["Branch_status"].to(device)
    Branch_tau = batch["Branch_tau"].to(device)
    Branch_shift_deg = batch["Branch_shift_deg"].to(device)
    Branch_y_series_from = batch["Branch_y_series_from"].to(device)
    Branch_y_series_to = batch["Branch_y_series_to"].to(device)
    Branch_y_series_ft = batch["Branch_y_series_ft"].to(device)
    Branch_y_shunt_from = batch["Branch_y_shunt_from"].to(device)
    Branch_y_shunt_to = batch["Branch_y_shunt_to"].to(device)
    Y_shunt_bus = batch["Y_shunt_bus"].to(device)

    B, N = bus_type.shape
    if B == 1:
        return build_dense_y_from_branchrows_single(
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

    Ys = []
    for b in range(B):
        Ys.append(build_dense_y_from_branchrows_single(
            N,
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
            Y_shunt_bus[b],
        ))
    return torch.stack(Ys, dim=0)


def residual_by_graph(Y, V, Sset, bus_type, sizes=None):
    v = V[..., 0]
    th = V[..., 1]
    if Y.dim() == 2:
        Y = Y.unsqueeze(0)

    Vc = v * torch.exp(1j * th)
    Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
    Sc = Vc * Ic.conj()

    slack_mask = bus_type == 1
    pv_mask = bus_type == 2
    p_mask = ~slack_mask
    q_mask = ~(slack_mask | pv_mask)

    dp_abs = (Sset.real - Sc.real).abs()
    dq_abs = (Sset.imag - Sc.imag).abs()

    if sizes is None:
        max_dp = dp_abs.masked_fill(~p_mask, 0.0).amax(dim=-1)
        max_dq = dq_abs.masked_fill(~q_mask, 0.0).amax(dim=-1)
        max_dq = torch.where(q_mask.any(dim=-1), max_dq, torch.zeros_like(max_dq))
        return max_dp, max_dq

    max_dp = []
    max_dq = []
    offset = 0
    for size in sizes.tolist():
        size = int(size)
        sl = slice(offset, offset + size)
        p_mask_g = p_mask[0, sl]
        q_mask_g = q_mask[0, sl]
        dp_g = dp_abs[0, sl]
        dq_g = dq_abs[0, sl]
        max_dp.append(dp_g[p_mask_g].max() if p_mask_g.any() else dp_g.new_zeros(()))
        max_dq.append(dq_g[q_mask_g].max() if q_mask_g.any() else dq_g.new_zeros(()))
        offset += size

    return torch.stack(max_dp), torch.stack(max_dq)


def update_stats(stats, values):
    values = values.detach().cpu().float().numpy()
    stats["sum"] += float(values.sum())
    stats["max"] = max(stats["max"], float(values.max(initial=0.0)))
    stats["values"].extend(float(v) for v in values.tolist())


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "mean=nan median=nan p90=nan max=nan"
    return (
        f"mean={arr.mean():.6e} "
        f"median={np.median(arr):.6e} "
        f"p90={np.percentile(arr, 90):.6e} "
        f"max={arr.max():.6e}"
    )


def main():
    parser = argparse.ArgumentParser(description="Check residuals at V_start and V_newton.")
    parser.add_argument("--PARQUET", nargs="+", required=True)
    parser.add_argument("--BATCH", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--lazy_parquet", action="store_true")
    parser.add_argument("--row_group_cache_size", type=int, default=2)
    parser.add_argument("--no_cache_dense_ybus", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ds = ChanghunDataset(
        args.PARQUET,
        per_unit=True,
        device=None,
        no_cache_dense_ybus=args.no_cache_dense_ybus,
        lazy_row_groups=args.lazy_parquet,
        row_group_cache_size=args.row_group_cache_size,
    )
    loader = DataLoader(ds, batch_size=args.BATCH, shuffle=False, collate_fn=collate_blockdiag)

    stats = {
        key: {"sum": 0.0, "max": 0.0, "values": []}
        for key in [
            "newton_sstart_dp",
            "newton_sstart_dq",
            "newton_snewton_dp",
            "newton_snewton_dq",
            "start_sstart_dp",
            "start_sstart_dq",
        ]
    }
    mag_sse = 0.0
    ang_sse = 0.0
    n_nodes = 0
    n_graphs = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break

            Y = ensure_dense_y(batch, device)
            bus_type = batch["bus_type"].to(device)
            sizes = batch.get("sizes", None)
            if sizes is not None:
                sizes = sizes.to(device)

            V_start = batch["V_start"].to(device)
            V_newton = batch["V_newton"].to(device)
            S_start = batch["S_start"].to(device)
            S_newton = batch["S_newton"].to(device)

            for prefix, V, S in [
                ("newton_sstart", V_newton, S_start),
                ("newton_snewton", V_newton, S_newton),
                ("start_sstart", V_start, S_start),
            ]:
                dp, dq = residual_by_graph(Y, V, S, bus_type, sizes=sizes)
                update_stats(stats[f"{prefix}_dp"], dp)
                update_stats(stats[f"{prefix}_dq"], dq)

            dmag = V_start[..., 0] - V_newton[..., 0]
            dang = torch.atan2(
                torch.sin(V_start[..., 1] - V_newton[..., 1]),
                torch.cos(V_start[..., 1] - V_newton[..., 1]),
            )
            mag_sse += float((dmag ** 2).sum().item())
            ang_sse += float((dang ** 2).sum().item())
            n_nodes += int(dmag.numel())
            n_graphs += int(sizes.numel()) if sizes is not None else int(bus_type.size(0))

    print(f"dataset_rows={len(ds)}")
    print(f"graphs_checked={n_graphs}")
    print(f"device={device}")
    print(f"v_start_mag_rmse_pu={math.sqrt(mag_sse / max(n_nodes, 1)):.6e}")
    print(f"v_start_ang_rmse_deg={math.sqrt(ang_sse / max(n_nodes, 1)) * 180.0 / math.pi:.6f}")
    print()
    print("Residual at V_newton using S_start:")
    print(f"  max |dP| pu: {summarize(stats['newton_sstart_dp']['values'])}")
    print(f"  max |dQ| pu: {summarize(stats['newton_sstart_dq']['values'])}")
    print("Residual at V_newton using S_newton:")
    print(f"  max |dP| pu: {summarize(stats['newton_snewton_dp']['values'])}")
    print(f"  max |dQ| pu: {summarize(stats['newton_snewton_dq']['values'])}")
    print("Residual at V_start using S_start:")
    print(f"  max |dP| pu: {summarize(stats['start_sstart_dp']['values'])}")
    print(f"  max |dQ| pu: {summarize(stats['start_sstart_dq']['values'])}")


if __name__ == "__main__":
    main()

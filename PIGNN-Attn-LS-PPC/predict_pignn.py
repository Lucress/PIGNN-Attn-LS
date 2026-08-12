#!/usr/bin/env python
"""Run a trained PIGNN-Attn-LS power-flow surrogate on a PPC parquet.

Given a checkpoint and a scenario parquet, predicts the bus voltage state
(magnitude and angle) and reports how far that state is from both the reference
Newton solution and the power-balance equations.

    python predict_pignn.py --ckpt SimBench_best.ckpt --parquet SimBench.parquet
    python predict_pignn.py --ckpt ... --parquet ... --limit 500 --save-csv out.csv

The checkpoint is a plain state_dict, so the architecture is NOT stored in it.
The defaults here match the configuration the released checkpoints were trained
with; if you trained your own with different flags, pass the same values.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag, ybus_matvec
from GNSMsg_SelfAttention_armijo import GNSMsg_EdgeSelfAttn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="trained state_dict (.ckpt)")
    p.add_argument("--parquet", required=True, help="scenario parquet")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=8,
                   help="scenarios per forward. Memory scales with batch*buses; "
                        "~1.24 MB per bus-sample was measured on an A40.")
    p.add_argument("--limit", type=int, default=0,
                   help="only score the first N scenarios (0 = all)")
    p.add_argument("--save-csv", default="", help="write per-scenario metrics here")

    # Architecture. Must match the checkpoint -- a state_dict carries weights,
    # not shapes, so a mismatch surfaces as a load error rather than bad output.
    g = p.add_argument_group("architecture (must match how the ckpt was trained)")
    g.add_argument("--d", type=int, default=4)
    g.add_argument("--d_hi", type=int, default=24)
    g.add_argument("--n_heads", type=int, default=8)
    g.add_argument("--num_attn_layers", type=int, default=8)
    g.add_argument("--K", type=int, default=40)
    g.add_argument("--use_armijo", action="store_true", default=True,
                   help="on by default: the released checkpoints were trained with it")
    g.add_argument("--no_armijo", dest="use_armijo", action="store_false")

    # Data conventions. The released parquets are per-unit on a 100 MVA base.
    g2 = p.add_argument_group("data conventions")
    g2.add_argument("--per_unit", action="store_true", default=True)
    g2.add_argument("--target_S_base", type=float, default=1e8)
    g2.add_argument("--complex_dtype", default="complex128",
                    choices=("complex64", "complex128"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dev = torch.device(args.device)

    ds = ChanghunDataset(
        args.parquet, per_unit=args.per_unit, target_S_base=args.target_S_base,
        share_grid=True, share_ybus=True, lazy_row_groups=True,
        row_group_cache_size=4, complex_dtype=args.complex_dtype,
    )
    n_total = len(ds)
    n = min(args.limit, n_total) if args.limit else n_total
    n_bus = int(ds[0]["N"])
    print(f"[data] {os.path.basename(args.parquet)}: {n_total} scenarios, "
          f"{n_bus} buses; scoring {n}")

    model = GNSMsg_EdgeSelfAttn(
        d=args.d, d_hi=args.d_hi, K=args.K, pinn=True,
        n_heads=args.n_heads, num_attn_layers=args.num_attn_layers,
        use_armijo=args.use_armijo,
    ).to(dev)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict mismatch -- missing {len(missing)}, "
              f"unexpected {len(unexpected)}. The architecture flags probably do "
              f"not match how this checkpoint was trained.", file=sys.stderr)
        for k in list(missing)[:5]:
            print(f"        missing: {k}", file=sys.stderr)
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_par:,} parameters, K={args.K}, d_hi={args.d_hi}, "
          f"heads={args.n_heads}, layers={args.num_attn_layers}, "
          f"armijo={args.use_armijo}, device={dev}")

    dmag2 = dang2 = 0.0
    dp_all, dq_all = [], []
    rows = []
    t0 = time.time()
    done = 0

    with torch.no_grad():
        for start in range(0, n, args.batch):
            idx = range(start, min(start + args.batch, n))
            b = collate_blockdiag([ds[i] for i in idx])
            bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}

            # The network runs in float32/complex64 while the dataset decodes in
            # complex128 (kept for the residual diagnostics below, which suffer
            # complex64 cancellation). Cast the model's inputs, not the metric
            # ones -- this mirrors what the training driver does.
            net = dict(bd)
            for k in ("Branch_tau", "Branch_shift_deg"):
                if torch.is_tensor(net.get(k)):
                    net[k] = net[k].float()
            for k in ("Branch_y_series_from", "Branch_y_series_to", "Branch_y_series_ft",
                      "Branch_y_shunt_from", "Branch_y_shunt_to", "Y_shunt_bus",
                      "S_start"):
                if torch.is_tensor(net.get(k)):
                    net[k] = net[k].to(torch.complex64)
            net["V_start"] = net["V_start"].float()
            net["Ybus"] = (bd["Ybus"].to(torch.complex64) if not bd["Ybus"].is_sparse
                           else torch.sparse_coo_tensor(
                               bd["Ybus"].coalesce().indices(),
                               bd["Ybus"].coalesce().values().to(torch.complex64),
                               bd["Ybus"].shape).coalesce())

            out = model(
                bus_type=net["bus_type"],
                Branch_f_bus=net["Branch_f_bus"], Branch_t_bus=net["Branch_t_bus"],
                Branch_status=net["Branch_status"], Branch_tau=net["Branch_tau"],
                Branch_shift_deg=net["Branch_shift_deg"],
                Branch_y_series_from=net["Branch_y_series_from"],
                Branch_y_series_to=net["Branch_y_series_to"],
                Branch_y_series_ft=net["Branch_y_series_ft"],
                Branch_y_shunt_from=net["Branch_y_shunt_from"],
                Branch_y_shunt_to=net["Branch_y_shunt_to"],
                Is_trafo=net.get("Is_trafo"),
                Y=net["Ybus"], S=net["S_start"], V0=net["V_start"],
                Y_shunt_bus=net.get("Y_shunt_bus"),
            )
            V = (out[0] if isinstance(out, (tuple, list)) else out)[..., :2]

            tgt = bd["V_newton"].to(V.dtype)
            dmag = V[..., 0] - tgt[..., 0]
            dang = torch.atan2(torch.sin(V[..., 1] - tgt[..., 1]),
                               torch.cos(V[..., 1] - tgt[..., 1]))
            dmag2 += float((dmag ** 2).sum())
            dang2 += float((dang ** 2).sum())

            # Power-balance residual of the predicted state. P is undetermined at
            # the slack and Q at slack and PV buses -- there the injection is a
            # free variable, so scoring it would charge correct behaviour as error.
            Vc = V[..., 0].to(torch.float64) * torch.exp(1j * V[..., 1].to(torch.float64))
            S_impl = Vc * torch.conj(ybus_matvec(bd["Ybus"], Vc.squeeze(0)).unsqueeze(0))
            resid = S_impl - bd["S_start"].to(torch.complex128)
            bt = bd["bus_type"]
            p_mask, q_mask = bt != 1, (bt != 1) & (bt != 2)
            dp = resid.real.abs()[p_mask]
            dq = resid.imag.abs()[q_mask]
            dp_all.append(dp.cpu()); dq_all.append(dq.cpu())

            if args.save_csv:
                sizes = b["sizes"].tolist(); off = 0
                for j, sz in enumerate(sizes):
                    sl = slice(off, off + sz); off += sz
                    # Mask per scenario exactly as the aggregate does. Without
                    # this the slack bus dominates every row and the column
                    # reads ~1000x the pooled number it is supposed to detail.
                    pm = p_mask[0, sl]; qm = q_mask[0, sl]
                    rp = resid.real[0, sl][pm]
                    rq = resid.imag[0, sl][qm]
                    rows.append((
                        start + j,
                        float((dmag[0, sl] ** 2).mean() ** 0.5),
                        float((dang[0, sl] ** 2).mean() ** 0.5) * 180 / math.pi,
                        float(rp.abs().max()) if rp.numel() else 0.0,
                        float(rq.abs().max()) if rq.numel() else 0.0,
                    ))
            done += len(list(idx))

    dt = time.time() - t0
    tot = done * n_bus
    rmse_mag = math.sqrt(dmag2 / tot)
    rmse_ang = math.sqrt(dang2 / tot)
    dp = torch.cat(dp_all); dq = torch.cat(dq_all)

    print(f"\n[accuracy vs the Newton reference]  n={done} scenarios")
    print(f"  |V| RMSE            {rmse_mag:.4e} pu")
    print(f"  theta RMSE          {rmse_ang * 180 / math.pi:.4e} deg")
    print(f"  combined RMSE       {math.sqrt(dmag2 / tot + dang2 / tot):.4e}")
    print(f"\n[power-balance residual of the predicted state]  pu on 100 MVA")
    print(f"  dP_inf {float(dp.max()):.4e}   dQ_inf {float(dq.max()):.4e}")
    print(f"  mean|dP| {float(dp.mean()):.4e}   mean|dQ| {float(dq.mean()):.4e}")
    print(f"  p95|dP|  {float(dp.quantile(0.95)):.4e}   p95|dQ|  {float(dq.quantile(0.95)):.4e}")
    print(f"\n[speed] {dt:.1f}s total, {1000 * dt / max(done, 1):.1f} ms/scenario")

    if args.save_csv:
        with open(args.save_csv, "w") as f:
            f.write("scenario,vmag_rmse_pu,theta_rmse_deg,dP_inf_pu,dQ_inf_pu\n")
            for r in rows:
                f.write("%d,%.6e,%.6e,%.6e,%.6e\n" % r)
        print(f"[csv] {len(rows)} rows -> {args.save_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

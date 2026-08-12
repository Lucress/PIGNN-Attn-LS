#!/usr/bin/env python
"""Load a trained PF surrogate and predict the voltage state for a parquet.

Covers all three variants trained in this project:

  mirror     our GridFMHeteroSurrogate, predicts a correction at every bus
  mirrorpin  the same model with --pin_known: |V| at PV/slack and the angle at
             the slack are held to their given inputs, which is the correct
             power-flow formulation
  graphkit   IBM's released gridfm_graphkit GNS_heterogeneous

The variant is not stored in the checkpoint, so it has to be given. Passing the
wrong one either fails to load (graphkit vs mirror have different parameter
names) or silently changes what is predicted (mirror vs mirrorpin load the same
weights but differ in whether known quantities are pinned).

    python pf_predict.py --PARQUET grid.parquet --ckpt model.pt --variant mirror
    python pf_predict.py --PARQUET grid.parquet --ckpt model.pt --variant mirrorpin \
        --save_npz out.npz --n 100
"""

from __future__ import annotations

import argparse
import math

import torch

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from train_valid_test_gridfm import (GridFMHeteroSurrogate, make_gridfm_inputs,
                                     split_dataset, cap_subset)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--PARQUET", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--variant", required=True, choices=("mirror", "mirrorpin", "graphkit"))
    p.add_argument("--BATCH", type=int, default=32)
    p.add_argument("--n", type=int, default=0, help="cap on scenarios (0 = whole test split)")
    p.add_argument("--split", default="test", choices=("test", "all"),
                   help="'test' reproduces the held-out third the models were scored on")
    p.add_argument("--hidden_size", type=int, default=48)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--save_npz", default="")
    p.add_argument("--device", default="")
    return p.parse_args()


def main():
    a = parse()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # These loader settings are not cosmetic: the models were trained on data in
    # per unit on a 100 MVA base with complex128 admittances, and feeding them
    # anything else silently changes the inputs.
    ds = ChanghunDataset(a.PARQUET, per_unit=True, target_S_base=1e8,
                         share_grid=True, share_ybus=True, lazy_row_groups=True,
                         row_group_cache_size=4, complex_dtype="complex128")
    if a.split == "test":
        _, _, use = split_dataset(ds, 0.3333, 0.3333, 42)
    else:
        use = ds
    if a.n:
        use = cap_subset(use, a.n)
    loader = torch.utils.data.DataLoader(use, batch_size=a.BATCH, shuffle=False,
                                         collate_fn=collate_blockdiag, num_workers=0)

    sample = collate_blockdiag([ds[0]])
    sd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in sample.items()}
    if a.variant == "graphkit":
        from gridfm_graphkit_adapter import build_graphkit_model, forward_graphkit_parquet
        model = build_graphkit_model(task_name="PowerFlow", hidden_size=a.hidden_size,
                                     num_layers=a.num_layers,
                                     attention_head=a.n_heads).to(dev)
    else:
        x, _, ea, _, _, _ = make_gridfm_inputs(sd, dev, "log", "signed_log", opf_space=None)
        model = GridFMHeteroSurrogate(
            input_bus_dim=x["bus"].shape[-1], input_gen_dim=x["gen"].shape[-1],
            edge_dim=ea[("bus", "connects", "bus")].shape[-1],
            hidden_size=a.hidden_size, heads=a.n_heads, num_layers=a.num_layers,
            pin_known=(a.variant == "mirrorpin"),
        ).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] {a.variant}  params={n_par:,}  device={dev}")

    P, R, BT = [], [], []
    with torch.no_grad():
        for batch in loader:
            bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if a.variant == "graphkit":
                V = forward_graphkit_parquet(model, bd, dev, task_name="PowerFlow",
                                             feature_transform="signed_log",
                                             vn_feature_mode="log")
            else:
                x, ei, ea, _, _, _ = make_gridfm_inputs(bd, dev, "log", "signed_log",
                                                        opf_space=None)
                V = model(x, ei, ea)
            P.append(V.squeeze(0).cpu())
            R.append(bd["V_newton"].squeeze(0).float().cpu())
            BT.append(bd["bus_type"].squeeze(0).cpu())
    Vp, Vr, bt = torch.cat(P), torch.cat(R), torch.cat(BT)

    dmag = (Vp[:, 0] - Vr[:, 0]).to(torch.float64)
    d = (Vp[:, 1] - Vr[:, 1]).to(torch.float64)
    dang = torch.atan2(torch.sin(d), torch.cos(d))
    deg = 180.0 / math.pi
    # bus_type here follows the parquet convention: 1 = slack, 2 = PV, else PQ.
    pq = (bt != 1) & (bt != 2)
    print(f"[data ] {Vp.shape[0]} bus-samples, {float(pq.float().mean()):.1%} PQ")
    print(f"[error] |V|  MAE {dmag.abs().mean():.4e}  RMSE {(dmag**2).mean()**0.5:.4e} pu")
    print(f"[error] theta MAE {dang.abs().mean()*deg:.4e}  RMSE {((dang**2).mean()**0.5)*deg:.4e} deg")
    if pq.any():
        dm = dmag[pq]
        print(f"[error] |V| at PQ only: MAE {dm.abs().mean():.4e}  RMSE {(dm**2).mean()**0.5:.4e} pu")

    if a.save_npz:
        import numpy as np
        np.savez_compressed(a.save_npz, vmag=Vp[:, 0].numpy(), theta=Vp[:, 1].numpy(),
                            vmag_ref=Vr[:, 0].numpy(), theta_ref=Vr[:, 1].numpy(),
                            bus_type=bt.numpy())
        print(f"[saved] {a.save_npz}")


if __name__ == "__main__":
    main()

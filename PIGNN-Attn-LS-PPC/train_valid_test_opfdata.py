#!/usr/bin/env python3
"""
Train/evaluate GridFM, GridSFM and LUMINA on DeepMind's OPFData, natively.

Counterpart to the three PPC-parquet scripts. The difference that matters is
that LUMINA and GridSFM consume OPFData's `HeteroData` directly -- the format
they were built around -- so their results here do not depend on the schema
adapters written for the pandapower pipeline. Only GridFM needs translating.

Metrics are the shared OPF ones (`opf_task.py`), computed from a Y-bus rebuilt
from `edge_attr`; the log format matches the parquet runs so
`summarize_opf_logs.py` parses both.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from opfdata_pipeline import (
    BUS,
    collate_opfdata,
    make_opfdata_loaders,
    to_gridfm_inputs,
)
from opf_task import (
    dispatch_loss,
    opf_residual_and_violation,
    format_opf_metrics,
    gen_head_loss,
    opf_decision_space,
    opf_loss,
    opf_metrics,
    target_dispatch,
)
from train_valid_test_gridfm import Tee, angle_diff


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("gridfm", "gridsfm", "lumina"), required=True)
    p.add_argument("--case_name", type=str, default="pglib_opf_case14_ieee")
    p.add_argument("--opfdata_root", type=str, required=True)
    p.add_argument("--num_groups", type=int, default=1)
    p.add_argument("--topological_perturbations", action="store_true")

    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--log_to_file", action="store_true")
    p.add_argument("--log_dir", type=str, default="./results/logs/opfdata")
    p.add_argument("--ckpt_dir", type=str, default="./results/ckpt/opfdata")

    p.add_argument("--init_mode", choices=("pretrained", "scratch"), default="scratch")
    p.add_argument("--pretrained_checkpoint", type=str, default="")
    p.add_argument("--model_config", type=str, default="")

    p.add_argument("--BATCH", type=int, default=4)
    p.add_argument("--EPOCHS", type=int, default=40)
    p.add_argument("--LR", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed_value", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_valid_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--VAL_EVERY", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--mse_weight", type=float, default=1.0)
    p.add_argument("--physics_weight", type=float, default=1e-2)
    p.add_argument("--physics_loss_form", choices=("mse", "huber", "logcosh"), default="logcosh")
    # "shared": one voltage-only objective for all three models (the controlled
    # comparison). "native": each model trained with the loss from its own
    # repository (see native_loss.py). Evaluation metrics are identical either
    # way, so runs stay comparable; loss values do not and must not be compared.
    p.add_argument("--loss", choices=("shared", "native"), default="shared")
    # "mirror": our local GridFMHeteroSurrogate. "graphkit": the released
    # gridfm_graphkit model, so GridFM is evaluated on the same footing as
    # LUMINA and GridSFM (which always use their own repo models).
    p.add_argument("--gridfm_impl", choices=("mirror", "graphkit"), default="mirror")
    # Load a *training-format* state_dict (what this script saves as
    # <run_name>_best.pt) into the freshly built model. --init_mode pretrained
    # goes through each repo's release loader and cannot read these; and
    # --EPOCHS 0 returns before the post-training checkpoint load, so without
    # this flag there is no way to score a saved checkpoint.
    p.add_argument("--resume_state_dict", type=str, default="")
    p.add_argument("--lumina_loss_type", default="mse",
                   choices=("mse", "mae", "smooth_l1", "rmse", "mape"))
    p.add_argument("--native_physics_weight", type=float, default=0.0,
                   help="stands in for GridFM's unusable PBELoss; 0 disables")
    p.add_argument("--opf_limit_weight", type=float, default=1.0)
    p.add_argument("--opf_band_weight", type=float, default=0.0)
    p.add_argument("--dispatch_weight", type=float, default=0.0)
    p.add_argument("--gen_head_weight", type=float, default=0.0)

    # GridFM architecture (it has no OPFData-native path)
    p.add_argument("--hidden_size", type=int, default=48)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--zero_init_head", action="store_true")
    p.add_argument("--feature_transform", type=str, default="signed_log")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Per-model input preparation and forward
# --------------------------------------------------------------------------- #

def lumina_bus_features(bus_x: torch.Tensor) -> torch.Tensor:
    """OPFData bus.x [base_kv, bus_type, vmin, vmax] -> LUMINA's 7 columns.

    This is the same transform LUMINA's own OPFDataset ingestion performs:
    drop bus_type and append its one-hot encoding.
    """
    bt = bus_x[:, BUS["bus_type"]].long().clamp(1, 4) - 1
    keep = torch.stack([bus_x[:, BUS["base_kv"]], bus_x[:, BUS["vmin"]],
                        bus_x[:, BUS["vmax"]]], dim=-1)
    onehot = torch.zeros((bus_x.shape[0], 4), dtype=bus_x.dtype, device=bus_x.device)
    onehot[torch.arange(bus_x.shape[0], device=bus_x.device), bt] = 1.0
    return torch.cat([keep, onehot], dim=-1)


def build_model(args, device, sample_batch):
    if args.model == "lumina":
        from lumina_inference.model.hetero_model import HGT
        cfg = json.load(open(args.model_config))
        h = cfg["config"]["models"]["HGT"]
        model = HGT(metadata=cfg["metadata"], input_channels=cfg["input_channels"],
                    hidden_channels=h["hidden_channels"], num_layers=h["num_layers"],
                    num_heads=h.get("num_heads", 1), dropout=h.get("dropout", 0.0)).to(device)
        if args.init_mode == "pretrained":
            from lumina_inference import Modeler
            from safetensors.torch import load_file
            sd = load_file(args.pretrained_checkpoint) if args.pretrained_checkpoint.endswith(
                ".safetensors") else torch.load(args.pretrained_checkpoint, map_location="cpu")
            model, _ = Modeler(device, verbose=False).load_model(cfg, sd)
        model.train()
        return model
    if args.model == "gridsfm":
        if args.init_mode == "pretrained":
            from gridsfm import load_model
            m = load_model(args.pretrained_checkpoint, device=device)
        else:
            from gridsfm import GridTransformerBackbone
            m = GridTransformerBackbone().to(device)
        m.train()
        return m
    if getattr(args, "gridfm_impl", "mirror") == "graphkit":
        # The released model. Its config dims are fixed by
        # gridfm_graphkit.datasets.globals, so only capacity is ours to choose.
        from gridfm_graphkit_adapter import build_graphkit_model
        m = build_graphkit_model(task_name="OptimalPowerFlow",
                                 hidden_size=args.hidden_size,
                                 num_layers=args.num_layers,
                                 attention_head=args.n_heads).to(device)
        m.train()
        return m

    from train_valid_test_gridfm import GridFMHeteroSurrogate
    x_dict, edge_index_dict, edge_attr_dict = to_gridfm_inputs(
        sample_batch, device, args.feature_transform)
    vmin = float(sample_batch["Bus_vmin"].min())
    vmax = float(sample_batch["Bus_vmax"].max())
    model = GridFMHeteroSurrogate(
        input_bus_dim=x_dict["bus"].shape[1],
        input_gen_dim=x_dict["gen"].shape[1],
        edge_dim=edge_attr_dict[("bus", "connects", "bus")].shape[1],
        hidden_size=args.hidden_size, heads=args.n_heads,
        num_layers=args.num_layers, zero_init_head=args.zero_init_head,
        out_bus_dim=(6 if args.loss == "native" else 2),
        vmin=vmin, vmax=vmax).to(device)
    model.train()
    return model


def forward(model, args, batch, device, want_gen=False, ctx=None):
    """Returns (V [1,N,2] as (mag, angle), gen_pred or None, gen_bus or None).

    When `ctx` is a dict it is populated with the model-specific raw outputs
    that the native losses need; the shared loss ignores it.
    """
    if args.model == "gridfm" and getattr(args, "gridfm_impl", "mirror") == "graphkit":
        from gridfm_graphkit_adapter import forward_graphkit
        V, gp, gb, pred6 = forward_graphkit(
            model, batch, device, task_name="OptimalPowerFlow",
            feature_transform=args.feature_transform, want_pred6=True)
        if ctx is not None:
            _, ei, _ = to_gridfm_inputs(batch, device, args.feature_transform)
            ctx["pred6"] = pred6
            ctx["edge_index_dict"] = ei
            ctx["gen_bus_cpu"] = torch.where(
                batch["Gen_controllable"].squeeze(0) > 0)[0]
        return V, (gp if want_gen else None), (gb if want_gen else None)

    if args.model == "gridfm":
        x_dict, ei, ea = to_gridfm_inputs(batch, device, args.feature_transform)
        V = model(x_dict, ei, ea)
        if ctx is not None:
            ctx["pred6"] = getattr(model, "_last_pred6", None)
            ctx["edge_index_dict"] = ei
            ctx["gen_bus_cpu"] = torch.where(
                batch["Gen_controllable"].squeeze(0) > 0)[0]
        return V, None, None

    hb = batch["hetero"].to(device)
    if args.model == "lumina":
        x_dict = {
            "bus": lumina_bus_features(hb["bus"].x.float()),
            "generator": hb["generator"].x.float(),
            "load": hb["load"].x.float(),
            "shunt": hb["shunt"].x.float() if "shunt" in hb.node_types
                     else torch.zeros((0, 2), device=device),
        }
        ei = {et: hb[et].edge_index for et in hb.edge_types}
        ea = {et: hb[et].edge_attr.float() for et in hb.edge_types
              if getattr(hb[et], "edge_attr", None) is not None}
        out = model(x_dict, ei, ea or None, minmax_scaling=True)
        pred = out["bus"]
        V = torch.stack([pred[:, 1], torch.atan2(torch.sin(pred[:, 0]),
                                                 torch.cos(pred[:, 0]))], dim=-1).unsqueeze(0)
        gen_pred = out.get("generator") if want_gen else None
        gen_bus = hb["generator", "generator_link", "bus"].edge_index[1] if want_gen else None
        if ctx is not None:
            ctx["out"] = out; ctx["hb"] = hb
        return V, gen_pred, gen_bus

    # GridSFM: native OPFData consumption via its own preparation step
    from gridsfm import prepare_for_inference
    from torch_geometric.data import Batch as PyGBatch
    graphs = hb.to_data_list()
    prepared = [prepare_for_inference(g) for g in graphs]
    gb = PyGBatch.from_data_list(prepared).to(device)
    out = model(gb)
    pred = out["bus"].pred
    V = torch.stack([pred[:, 1], torch.atan2(torch.sin(pred[:, 0]),
                                             torch.cos(pred[:, 0]))], dim=-1).unsqueeze(0)
    gen_pred = out["generator"].pred if (want_gen and "generator" in out.node_types) else None
    gen_bus = (out["generator", "generator_link", "bus"].edge_index[1]
               if (want_gen and ("generator", "generator_link", "bus") in out.edge_types) else None)
    if ctx is not None:
        ctx["gb"] = out
    return V, gen_pred, gen_bus


def run():
    args = parse_args()
    set_seed(args.seed_value)
    if not args.run_name:
        args.run_name = f"opfdata_{args.model}_{args.init_mode}_{args.case_name}"
    os.makedirs(args.log_dir, exist_ok=True); os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.log_to_file:
        f = open(os.path.join(args.log_dir, f"{args.run_name}_training_log.txt"), "a", buffering=1)
        sys.stdout = Tee(sys.__stdout__, f); sys.stderr = Tee(sys.__stderr__, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__}")
    print(f"[run] {args.run_name}")
    print(f"[task] opf (pipeline=opfdata, case={args.case_name}, groups={args.num_groups}, "
          f"topo_perturb={args.topological_perturbations})")
    print(f"[model] {args.model} init_mode={args.init_mode} "
          f"dispatch_weight={args.dispatch_weight} gen_head_weight={args.gen_head_weight}")

    train_loader, train_eval_loader, val_loader, test_loader = make_opfdata_loaders(
        args.opfdata_root, args.case_name, args.BATCH, num_groups=args.num_groups,
        topological_perturbations=args.topological_perturbations,
        max_train=args.max_train_samples, max_valid=args.max_valid_samples,
        max_test=args.max_test_samples, num_workers=args.num_workers)
    print(f"[split] train={len(train_loader.dataset)} valid={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}")

    sample = next(iter(val_loader))
    model = build_model(args, device, sample)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable_params={n_params:,}")

    native_fn = None
    if getattr(args, "resume_state_dict", ""):
        model.load_state_dict(torch.load(args.resume_state_dict, map_location=device))
        print(f"[init] loaded state_dict: {args.resume_state_dict}")

    if args.loss == "native":
        import native_loss as NL
        if args.model == "lumina":
            native_fn = NL.LuminaNative(loss_type=args.lumina_loss_type)
            print(f"[loss] native: lumina ACOPFLossFunction({args.lumina_loss_type}) "
                  f"over bus+generator")
        elif args.model == "gridsfm":
            native_fn = NL.gridsfm_native
            print("[loss] native: gridsfm compute_loss (theta/V/Pg/Qg + cost + KCL + "
                  "flows + thermal); lambda_feas=lambda_stress_feas=0 because "
                  "OPFData ships solved instances only")
        else:
            native_fn = NL.GridFMNative()
            print("[loss] native: gridfm MaskedReconstructionMSE over "
                  "[VM,VA,PG,QG,PD,QD]; PBELoss NOT used (edge attrs lack the "
                  f"YFF/YFT split) -- substituting physics_weight="
                  f"{args.native_physics_weight}")
    else:
        print("[loss] shared: voltage MSE + physics (all three models)")

    optim = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_path = os.path.join(args.ckpt_dir, f"{args.run_name}_best.pt")
    want_gen = args.gen_head_weight > 0.0

    def run_epoch(loader, train, collect_residuals=False):
        model.train(train)
        sums = {"loss": 0.0, "mse": 0.0, "mse_mag": 0.0, "mse_ang": 0.0, "phys": 0.0}
        opf_sums: Dict[str, float] = {}
        # Per-batch summaries cannot be averaged into a max or a percentile:
        # averaging per-batch maxima understates the true worst bus, and there is
        # no way to recover a global p95 from per-batch p95s. So for the test
        # pass we pool the masked per-bus residuals and summarise once.
        dp_chunks: list = []
        dq_chunks: list = []
        n_graphs = 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                B = int(batch["sizes"].numel()); n_graphs += B
                bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                      for k, v in batch.items() if k != "hetero"}
                bd["hetero"] = batch["hetero"]
                space = opf_decision_space(bd, batch["sizes"], device)
                Y = bd["Ybus"]; S_load = bd["S_start"]
                target = bd["V_newton"].float()

                nctx = {} if native_fn is not None else None
                V, gen_pred, gen_bus = forward(model, args, batch, device,
                                               want_gen=want_gen, ctx=nctx)
                dmag = V[..., 0] - target[..., 0]
                dang = angle_diff(V[..., 1], target[..., 1])
                mse_mag = torch.mean(dmag * dmag); mse_ang = torch.mean(dang * dang)
                mse = mse_mag + mse_ang
                phys = opf_loss(Y, V, S_load, space, limit_weight=args.opf_limit_weight,
                                band_weight=args.opf_band_weight, form=args.physics_loss_form)
                if native_fn is None:
                    loss = args.mse_weight * mse + args.physics_weight * phys
                elif args.model == "lumina":
                    loss, _ = native_fn(nctx["out"], nctx["hb"])
                elif args.model == "gridsfm":
                    loss, _ = native_fn(nctx["gb"])
                else:
                    loss, _ = native_fn(nctx["pred6"], bd, nctx["edge_index_dict"],
                                        nctx["gen_bus_cpu"], device)
                    if args.native_physics_weight > 0.0:
                        loss = loss + args.native_physics_weight * phys
                if args.dispatch_weight > 0.0 or args.gen_head_weight > 0.0:
                    tgt = target_dispatch(bd, space, device)
                    if tgt is not None:
                        if args.dispatch_weight > 0.0:
                            loss = loss + args.dispatch_weight * dispatch_loss(Y, V, S_load, space, tgt)
                        if args.gen_head_weight > 0.0 and gen_pred is not None:
                            loss = loss + args.gen_head_weight * gen_head_loss(gen_pred, tgt, gen_bus)

                if train:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optim.step()

                om = opf_metrics(Y, V.detach(), target, S_load, space, batch["sizes"])
                for k, v in om.items():
                    opf_sums[k] = opf_sums.get(k, 0.0) + float(v) * B
                if collect_residuals:
                    r = opf_residual_and_violation(Y, V.detach(), S_load, space)
                    free = ~r["ctrl"]
                    if int(free.sum().item()) > 0:
                        dp_chunks.append(r["dp"][free].detach().float().cpu())
                        dq_chunks.append(r["dq"][free].detach().float().cpu())
                for k, v in (("loss", loss), ("mse", mse), ("mse_mag", mse_mag),
                             ("mse_ang", mse_ang), ("phys", phys)):
                    sums[k] += float(v.item()) * B
        d = max(n_graphs, 1)
        out = {k: v / d for k, v in sums.items()}
        out["opf"] = {k: v / d for k, v in opf_sums.items()}
        if collect_residuals and dp_chunks:
            def summarize(chunks):
                x = torch.cat(chunks, dim=0)
                return {
                    "inf": float(x.max().item()),
                    "mean": float(x.mean().item()),
                    "p95": float(torch.quantile(x, 0.95).item()),
                    "p99": float(torch.quantile(x, 0.99).item()),
                    "n": int(x.numel()),
                }
            out["resid"] = {"dp": summarize(dp_chunks), "dq": summarize(dq_chunks)}
        return out

    def fmt(prefix, m):
        rmse = math.sqrt(max(m["mse"], 0.0))
        rmag = math.sqrt(max(m["mse_mag"], 0.0))
        rang = math.sqrt(max(m["mse_ang"], 0.0)) * 180.0 / math.pi
        return (f"{prefix} loss {m['loss']:.4e} mse {m['mse']:.4e} phys {m['phys']:.4e} "
                f"rmse {rmse:.4e} (mag {rmag:.4e}, ang {rang:.4e}deg) "
                f"{format_opf_metrics(m['opf'])}")

    def print_residuals(m):
        if "resid" not in m:
            return
        dp, dq = m["resid"]["dp"], m["resid"]["dq"]
        # Pooled over every free bus in the test set, not averaged per batch.
        print(f"Final test-set residuals (free buses, n={dp['n']}) : "
              f"dPinf {dp['inf']:.4e} | dQinf {dq['inf']:.4e} | "
              f"mean|dP| {dp['mean']:.4e} | mean|dQ| {dq['mean']:.4e} | "
              f"p95|dP| {dp['p95']:.4e} | p95|dQ| {dq['p95']:.4e} | "
              f"p99|dP| {dp['p99']:.4e} | p99|dQ| {dq['p99']:.4e} pu")

    if args.EPOCHS <= 0:
        # Eval-only pass (typically with --resume_state_dict). The residual
        # statistics must be emitted here too: this branch returns before the
        # post-training test block below.
        print("Initial metrics before training:")
        for tag, ld in (("train", train_eval_loader), ("valid", val_loader)):
            print("Epoch   0 | " + fmt(tag, run_epoch(ld, train=False)))
        te0 = run_epoch(test_loader, train=False, collect_residuals=True)
        print("Epoch   0 | " + fmt("test", te0))
        print("Final test-set OPF : " + format_opf_metrics(te0["opf"]))
        print_residuals(te0)
        return

    print("Epoch   0 | " + fmt("valid", run_epoch(val_loader, train=False)))
    for epoch in range(1, args.EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, train=True)
        if epoch % args.VAL_EVERY == 0 or epoch == args.EPOCHS:
            va = run_epoch(val_loader, train=False)
            print(f"Epoch {epoch:3d} | {fmt('train', tr)} | {fmt('valid', va)} "
                  f"| time {time.time()-t0:.2f}s")
            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(model.state_dict(), best_path)
                print(f"  checkpoint saved to {best_path}")
        else:
            print(f"Epoch {epoch:3d} | {fmt('train', tr)} | time {time.time()-t0:.2f}s")

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[test] loaded best checkpoint: {best_path}")
    te = run_epoch(test_loader, train=False, collect_residuals=True)
    rmse = math.sqrt(max(te["mse"], 0.0))
    rmag = math.sqrt(max(te["mse_mag"], 0.0))
    rang = math.sqrt(max(te["mse_ang"], 0.0)) * 180.0 / math.pi
    print(f"\nFinal test-set RMSE : {rmse:.4e} (|V|: {rmag:.4e}, theta: {rang:.4e}deg)"
          f" | dPinf : {te['opf']['dp_inf_pu']:.4e} pu"
          f" | dQinf : {te['opf']['dq_inf_pu']:.4e} pu")
    print("Final test-set OPF : " + format_opf_metrics(te["opf"]))
    print_residuals(te)


if __name__ == "__main__":
    run()

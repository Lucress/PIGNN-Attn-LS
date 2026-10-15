"""Train the PIGNN OPF solver on DeepMind OPFData.

Uses the same loaders (`opfdata_pipeline`) and the same scoring
(`opf_task.opf_metrics`) as the GridFM / GridSFM / LUMINA drivers, so the
numbers drop straight into the comparison. The log format matches too.

OPFData ships a hetero graph, not the branch-row schema the solver reads, so
`branch_rows_from_opfdata` reconstructs the rows from the per-branch
r/x/b_fr/b_to/tap/shift. `--validate_branch_rows` re-assembles a Y-bus from
those rows and compares it against the dataset's own; the solver's own
`_build_dense_Y_from_branchrows_single` does the assembly, so this checks the
conversion against the code that will consume it.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch

from opfdata_pipeline import make_opfdata_loaders, build_ybus
from opf_task import (
    opf_decision_space, opf_loss, opf_metrics, format_opf_metrics,
    opf_residual_and_violation,
)
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def angle_diff(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))
from GNSMsg_SelfAttention_armijo_opf import GNSMsg_EdgeSelfAttn_OPF
from GNSMsg_SelfAttention_armijo import _build_dense_Y_from_branchrows_single


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case_name", default="pglib_opf_case14_ieee")
    p.add_argument("--opfdata_root", default="",
                   help="root dir for PyG OPFData; required unless --chunk_dir is set")
    p.add_argument("--num_groups", type=int, default=1)
    p.add_argument("--topological_perturbations", action="store_true")
    p.add_argument("--run_name", default="")
    p.add_argument("--log_to_file", action="store_true")
    p.add_argument("--log_dir", default="./results/logs")
    p.add_argument("--ckpt_dir", default="./results/ckpt")
    p.add_argument("--BATCH", type=int, default=4)
    p.add_argument("--EPOCHS", type=int, default=40)
    p.add_argument("--LR", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed_value", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_valid_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--chunk_dir", default="",
                   help="case500 chunk dir (group_N.pt files); if set, uses ChunkedOPFDataset")
    p.add_argument("--n_train_groups", type=int, default=4,
                   help="number of training groups when using --chunk_dir")
    p.add_argument("--VAL_EVERY", type=int, default=1)
    p.add_argument("--resume_state_dict", default="")
    p.add_argument("--validate_branch_rows", action="store_true")
    # solver hyperparameters (names mirror the PF solver)
    p.add_argument("--K", type=int, default=30)
    p.add_argument("--d", type=int, default=10)
    p.add_argument("--d_hi", type=int, default=32)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--num_attn_layers", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--use_armijo", action="store_true", default=True)
    p.add_argument("--no_armijo", dest="use_armijo", action="store_false")
    p.add_argument("--armijo_mode", default="geometric_safe",
                   choices=("fixed", "geometric", "geometric_safe", "reject"))
    p.add_argument("--dtheta_max", type=float, default=0.30)
    p.add_argument("--dvm_frac", type=float, default=0.10)
    # global attention
    p.add_argument("--use_global_attn", action="store_true", default=False,
                   help="add a dense all-to-all attention block at each Newton step")
    p.add_argument("--global_attn_position", default="after",
                   choices=("before", "after", "both"),
                   help="where to insert the global attention relative to local edge-attn")
    p.add_argument("--attn_k_hop", type=int, default=1,
                   help="expand local edge-attention graph to k-hop neighborhood (1=1-hop only)")
    p.add_argument("--attn_rank", type=int, default=0,
                   help="low-rank Q/K projection rank (0=full-rank, 16=exploit OPF low-dim structure)")
    # partial-cost OPF curriculum
    p.add_argument("--partial_cost_frac", type=float, default=0.0,
                   help="fraction of controllable generators whose cost is randomly zeroed "
                        "per batch (0=disabled); creates a free-dispatch curriculum")
    # loss weights
    p.add_argument("--mse_weight", type=float, default=1.0)
    p.add_argument("--pinn_weight", type=float, default=1e-2,
                   help="weight on the solver's own unrolled physics residual")
    p.add_argument("--physics_weight", type=float, default=1e-2,
                   help="weight on opf_task.opf_loss (balance + limit violation)")
    p.add_argument("--cost_weight", type=float, default=0.0,
                   help="weight on generation dispatch cost (differentiable via implied injection)")
    p.add_argument("--physics_loss_form", default="logcosh",
                   choices=("mse", "huber", "logcosh"))
    p.add_argument("--opf_balance_weight", type=float, default=1.0,
                   help="weight on P/Q balance term inside opf_loss")
    p.add_argument("--opf_limit_weight", type=float, default=1.0,
                   help="weight on generator P/Q limit violation inside opf_loss")
    p.add_argument("--opf_band_weight", type=float, default=0.0)
    # Primal-Dual Learning (PDL): adaptive Lagrange multipliers
    p.add_argument("--use_pdl", action="store_true",
                   help="replace fixed opf_balance/limit weights with adaptive dual variables")
    p.add_argument("--pdl_lr",     type=float, default=0.01,
                   help="dual ascent step size (used for both balance and limit if specific rates not set)")
    p.add_argument("--pdl_lr_bal", type=float, default=0.0,
                   help="dual ascent step for power-balance multiplier (overrides --pdl_lr if >0)")
    p.add_argument("--pdl_lr_lim", type=float, default=0.0,
                   help="dual ascent step for P/Q-limit multiplier (overrides --pdl_lr if >0)")
    p.add_argument("--pdl_clip", type=float, default=100.0,
                   help="maximum value for PDL multipliers (prevents runaway)")
    # XAI-guided regularizer: L2 penalty on known always-zero generators
    p.add_argument("--xai_pen_buses", type=int, nargs="*", default=[],
                   help="1-indexed bus numbers whose predicted gen_p is penalized "
                        "(XAI finding: these generators are always OFF in optimal dispatch)")
    p.add_argument("--xai_pen_weight", type=float, default=0.0,
                   help="weight on XAI-guided L2 gen_p penalty (0=disabled)")
    p.add_argument("--xai_pen_top_frac", type=float, default=0.0,
                   help="auto-detect always-zero generators via merit order: "
                        "top-frac of ctrl generators ranked by c1+c2*Pmax "
                        "(marginal cost at full capacity); merges with --xai_pen_buses; "
                        "0=disabled  example: 0.05 picks top-5 percent most expensive")
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------------------- #
# OPFData -> branch rows
# --------------------------------------------------------------------------- #

def branch_rows_from_opfdata(hetero, device) -> Dict[str, torch.Tensor]:
    """Branch-row tensors in the schema the PIGNN solver reads.

    Conventions follow `opfdata_pipeline.build_ybus`: ys = 1/(r+jx) and the tap
    ratio carries the phase shift as tau*exp(j*shift). Two conversions are not
    one-to-one and both are verified by `validate_branch_rows`:

    * `Branch_shift_deg` is in degrees, because the solver applies `deg2rad`,
      while OPFData stores radians;
    * the shunts are doubled, because the solver halves them on the assumption
      that the row carries total line charging, whereas OPFData already stores
      per-end values.
    """
    from opfdata_pipeline import AC_LINE, TRAFO

    f_l, t_l, ys_l, shf_l, sht_l, tau_l, sdeg_l, tr_l = [], [], [], [], [], [], [], []
    for key, cols, is_trafo in ((("bus", "ac_line", "bus"), AC_LINE, False),
                                (("bus", "transformer", "bus"), TRAFO, True)):
        if key not in hetero.edge_types:
            continue
        st = hetero[key]
        ei, ea = st.edge_index, getattr(st, "edge_attr", None)
        if ea is None or ei.numel() == 0:
            continue
        a = ea.detach().to(torch.float64)
        r, x = a[:, cols["r"]], a[:, cols["x"]]
        z = torch.complex(r, x)
        ys = torch.where(z.abs() > 0, 1.0 / torch.where(z.abs() > 0, z,
                                                        torch.ones_like(z)),
                         torch.zeros_like(z))
        b_fr, b_to = a[:, cols["b_fr"]], a[:, cols["b_to"]]
        if is_trafo:
            tau = a[:, cols["tap"]].clone()
            tau[tau == 0] = 1.0
            shift = a[:, cols["shift"]]
        else:
            tau = torch.ones_like(r)
            shift = torch.zeros_like(r)
        f_l.append(ei[0].detach()); t_l.append(ei[1].detach())
        ys_l.append(ys)
        # The solver's assembler halves the shunt (`ysh_f / 2.0`) because the PPC
        # branch-row schema stores the *total* line charging and splits it over
        # the two ends. OPFData's b_fr / b_to are already per-end, so they are
        # doubled here to survive that halving. Caught by
        # --validate_branch_rows, which failed at rel 1.0e-3 without it.
        shf_l.append(torch.complex(torch.zeros_like(b_fr), 2.0 * b_fr))
        sht_l.append(torch.complex(torch.zeros_like(b_to), 2.0 * b_to))
        tau_l.append(tau)
        sdeg_l.append(torch.rad2deg(shift))
        tr_l.append(torch.full_like(r, 1.0 if is_trafo else 0.0))

    if not f_l:
        raise ValueError("no branches found in OPFData sample")

    cat = lambda xs: torch.cat(xs, dim=0).to(device)
    ys = cat(ys_l)
    return {
        "Branch_f_bus": cat(f_l).long(),
        "Branch_t_bus": cat(t_l).long(),
        "Branch_status": torch.ones(ys.shape[0], dtype=torch.float64, device=device),
        "Branch_tau": cat(tau_l),
        "Branch_shift_deg": cat(sdeg_l),
        "Branch_y_series_from": ys,
        "Branch_y_series_to": ys,
        "Branch_y_series_ft": ys,
        "Branch_y_shunt_from": cat(shf_l),
        "Branch_y_shunt_to": cat(sht_l),
        "Is_trafo": cat(tr_l),
    }


def bus_shunt_from_opfdata(hetero, n, device) -> torch.Tensor:
    y = torch.zeros(n, dtype=torch.complex128, device=device)
    if "shunt" in hetero.node_types and hetero["shunt"].x.numel():
        sx = hetero["shunt"].x.detach().to(torch.float64)
        sb = hetero["shunt", "shunt_link", "bus"].edge_index[1].detach().to(device)
        y.index_add_(0, sb.long(), torch.complex(sx[:, 1], sx[:, 0]).to(device))
    return y


def validate_branch_rows(hetero, device) -> Dict[str, float]:
    """Rebuild Y from the synthesized rows with the solver's own assembler."""
    n = int(hetero["bus"].x.shape[0])
    br = branch_rows_from_opfdata(hetero, device)
    ysh = bus_shunt_from_opfdata(hetero, n, device)
    Y = _build_dense_Y_from_branchrows_single(
        n, br["Branch_f_bus"], br["Branch_t_bus"], br["Branch_status"],
        br["Branch_tau"], br["Branch_shift_deg"],
        br["Branch_y_series_from"], br["Branch_y_series_to"],
        br["Branch_y_series_ft"], br["Branch_y_shunt_from"],
        br["Branch_y_shunt_to"], ysh,
    )
    ref = build_ybus(hetero).to(device).to(torch.complex128)
    err = (Y.to(torch.complex128) - ref).abs()
    rel = float(err.max().item() / max(ref.abs().max().item(), 1e-12))
    return {"max_abs_err": float(err.max().item()), "rel_err": rel,
            "ok": bool(rel <= 1e-10)}


# --------------------------------------------------------------------------- #

def compute_merit_order_buses(first_batch, top_frac):
    """Rank ctrl generators by marginal cost (c1 + c2*Pmax) — merit order.

    Returns 1-indexed bus numbers of the top-frac most expensive generators.
    These are structurally always-zero in optimal dispatch when cheaper
    generators have enough capacity to cover the entire load range.
    """
    # Batch tensors have shape (1, B*n_nodes); take only first graph's n_nodes
    n_nodes = int(first_batch["sizes"][0].item())
    c1   = first_batch["Gen_cost_c1"].squeeze(0).float()[:n_nodes]
    c2_t = first_batch.get("Gen_cost_c2")
    c2   = c2_t.squeeze(0).float()[:n_nodes] if c2_t is not None else torch.zeros_like(c1)
    pm_t = first_batch.get("Gen_p_max")
    pmax = pm_t.squeeze(0).float()[:n_nodes] if pm_t is not None else torch.ones_like(c1)
    ctrl = (first_batch["Gen_controllable"].squeeze(0)[:n_nodes] > 0)

    # Marginal cost at full capacity: MC(Pmax) = c1 + 2*c2*Pmax
    # (quadratic cost C(p)=c2*p²+c1*p, so MC=2*c2*p+c1)
    # For ranking, the factor of 2 cancels so we use c1 + c2*Pmax
    mc = c1 + c2 * pmax

    ctrl_idx = ctrl.nonzero(as_tuple=False).squeeze(-1)  # 0-indexed positions
    mc_ctrl  = mc[ctrl_idx]
    n_top = max(1, int(len(ctrl_idx) * top_frac))
    _, order = mc_ctrl.sort(descending=True)
    top_0 = ctrl_idx[order[:n_top]].tolist()
    return sorted(int(i) + 1 for i in top_0)  # 1-indexed bus numbers


def run():
    args = parse_args()
    set_seed(args.seed_value)
    if not args.run_name:
        args.run_name = f"pignn_opf_{args.case_name}"
    os.makedirs(args.log_dir, exist_ok=True); os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.log_to_file:
        f = open(os.path.join(args.log_dir, f"{args.run_name}_training_log.txt"),
                 "a", buffering=1)
        sys.stdout = Tee(sys.stdout, f)

    if not args.chunk_dir and not args.opfdata_root:
        raise SystemExit("error: --opfdata_root is required when --chunk_dir is not set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] {args.run_name}")
    print(f"[task] opf (pipeline=opfdata, case={args.case_name}, solver=PIGNN)")
    print(f"[solver] K={args.K} d_hi={args.d_hi} heads={args.n_heads} "
          f"layers={args.num_attn_layers} armijo={args.use_armijo}/{args.armijo_mode}")

    if args.chunk_dir:
        from opfdata_pipeline import make_case500_loaders
        train_loader, train_eval_loader, val_loader, test_loader = make_case500_loaders(
            args.chunk_dir, args.BATCH,
            n_train_groups=args.n_train_groups,
            max_train=args.max_train_samples, max_valid=args.max_valid_samples,
            max_test=args.max_test_samples, num_workers=args.num_workers)
    else:
        train_loader, train_eval_loader, val_loader, test_loader = make_opfdata_loaders(
            args.opfdata_root, args.case_name, args.BATCH, num_groups=args.num_groups,
            topological_perturbations=args.topological_perturbations,
            max_train=args.max_train_samples, max_valid=args.max_valid_samples,
            max_test=args.max_test_samples, num_workers=args.num_workers)

    if args.validate_branch_rows:
        b0 = next(iter(test_loader))
        g0 = b0["hetero"].to_data_list()[0]
        v = validate_branch_rows(g0, torch.device("cpu"))
        print(f"[check] branch-row -> Ybus rel_err={v['rel_err']:.3e} ok={v['ok']}")
        if not v["ok"]:
            raise SystemExit("branch-row conversion does not reproduce the Y-bus")

    model = GNSMsg_EdgeSelfAttn_OPF(
        d=args.d, d_hi=args.d_hi, K=args.K, pinn=True, gamma=args.gamma,
        v_limit=True, use_armijo=args.use_armijo, armijo_mode=args.armijo_mode,
        n_heads=args.n_heads, num_attn_layers=args.num_attn_layers,
        dtheta_max=args.dtheta_max, dvm_frac=args.dvm_frac,
        physics_loss_form=args.physics_loss_form,
        use_global_attn=args.use_global_attn,
        global_attn_position=args.global_attn_position,
        attn_k_hop=args.attn_k_hop,
        attn_rank=args.attn_rank,
    ).to(device)
    if args.resume_state_dict:
        model.load_state_dict(torch.load(args.resume_state_dict, map_location=device))
        print(f"[init] loaded state_dict: {args.resume_state_dict}")
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable_params={n_par:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.LR,
                              weight_decay=args.weight_decay)
    best_val = float("inf")
    best_path = os.path.join(args.ckpt_dir, f"{args.run_name}_best.pt")
    best_feas = float("inf")   # min(max(P-limit, dp_inf)) across val epochs
    best_feas_path = os.path.join(args.ckpt_dir, f"{args.run_name}_best_feas.pt")

    # PDL: mutable lists so the closure inside run_epoch can read the live value
    lam_bal = [args.opf_balance_weight]
    lam_lim = [args.opf_limit_weight]
    _pdl_lr_bal = args.pdl_lr_bal if args.pdl_lr_bal > 0 else args.pdl_lr
    _pdl_lr_lim = args.pdl_lr_lim if args.pdl_lr_lim > 0 else args.pdl_lr
    if args.use_pdl:
        print(f"[PDL] enabled — lr_bal={_pdl_lr_bal}  lr_lim={_pdl_lr_lim}  clip={args.pdl_clip}  "
              f"init lam_bal={lam_bal[0]}  lam_lim={lam_lim[0]}")

    # Merit-order auto-detection of always-zero generators
    if args.xai_pen_top_frac > 0:
        _first_batch = next(iter(train_loader))
        mo_buses = compute_merit_order_buses(_first_batch, args.xai_pen_top_frac)
        args.xai_pen_buses = sorted(set(args.xai_pen_buses) | set(mo_buses))
        print(f"[XAI-pen] merit-order top {args.xai_pen_top_frac:.1%} of ctrl generators "
              f"-> auto-detected buses {mo_buses}")
    if args.xai_pen_buses:
        print(f"[XAI-pen] penalizing gen_p at 1-indexed buses {args.xai_pen_buses}  "
              f"weight={args.xai_pen_weight}")

    def run_epoch(loader, train, collect_residuals=False):
        model.train(train)
        sums = {"loss": 0.0, "mse": 0.0, "mse_mag": 0.0, "mse_ang": 0.0, "phys": 0.0, "cost": 0.0, "xai_pen": 0.0}
        opf_sums: Dict[str, float] = {}
        dp_chunks, dq_chunks = [], []
        n_graphs = 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                B = int(batch["sizes"].numel()); n_graphs += B
                bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                      for k, v in batch.items() if k != "hetero"}
                hb = batch["hetero"].to(device)
                space = opf_decision_space(bd, batch["sizes"], device)
                # Partial-cost curriculum: randomly zero cost coefficients for a
                # fraction of controllable generators so the model learns OPF
                # with "free dispatch" generators mixed in.
                if train and args.partial_cost_frac > 0.0:
                    ctrl_idx = space["Gen_controllable"].nonzero(as_tuple=False).squeeze(-1)
                    n_drop = int(len(ctrl_idx) * args.partial_cost_frac)
                    if n_drop > 0:
                        perm = torch.randperm(len(ctrl_idx), device=device)[:n_drop]
                        drop = ctrl_idx[perm]
                        for key in ("Gen_cost_c1", "Gen_cost_c2"):
                            if key in space:
                                space[key] = space[key].clone()
                                space[key][drop] = 0.0
                Y = bd["Ybus"]; S_load = bd["S_start"]
                target = bd["V_newton"].float()
                n = target.shape[1]

                br = branch_rows_from_opfdata(hb, device)
                ctrl = space["Gen_controllable"].to(torch.bool)
                bus_type = bd["bus_type"].reshape(1, -1)
                V0 = bd["V_start"].float()
                vmin = space.get("Bus_vmin"); vmax = space.get("Bus_vmax")

                out, pinn_loss = model(
                    bus_type=bus_type,
                    Branch_f_bus=br["Branch_f_bus"].unsqueeze(0),
                    Branch_t_bus=br["Branch_t_bus"].unsqueeze(0),
                    Branch_status=br["Branch_status"].unsqueeze(0),
                    Branch_tau=br["Branch_tau"].unsqueeze(0),
                    Branch_shift_deg=br["Branch_shift_deg"].unsqueeze(0),
                    Branch_y_series_from=br["Branch_y_series_from"].unsqueeze(0),
                    Branch_y_series_to=br["Branch_y_series_to"].unsqueeze(0),
                    Branch_y_series_ft=br["Branch_y_series_ft"].unsqueeze(0),
                    Branch_y_shunt_from=br["Branch_y_shunt_from"].unsqueeze(0),
                    Branch_y_shunt_to=br["Branch_y_shunt_to"].unsqueeze(0),
                    Is_trafo=br["Is_trafo"].unsqueeze(0),
                    Y=Y, S=S_load, V0=V0,
                    ctrl_mask=ctrl.reshape(1, -1),
                    # OPFData follows MATPOWER: 1=PQ, 2=PV, 3=ref
                    # (see opfdata_pipeline.to_gridfm_inputs).
                    ref_mask=(bus_type == 3),
                    n_nodes_per_graph=batch["sizes"].to(device),
                    v_min=(vmin.reshape(1, -1).float() if vmin is not None else None),
                    v_max=(vmax.reshape(1, -1).float() if vmax is not None else None),
                )
                V = out.unsqueeze(0) if out.dim() == 2 else out

                dmag = V[..., 0] - target[..., 0]
                dang = angle_diff(V[..., 1], target[..., 1])
                mse_mag = torch.mean(dmag * dmag); mse_ang = torch.mean(dang * dang)
                mse = mse_mag + mse_ang
                phys = opf_loss(Y, V, S_load, space,
                                balance_weight=lam_bal[0],
                                limit_weight=lam_lim[0],
                                band_weight=args.opf_band_weight,
                                form=args.physics_loss_form)
                from opf_task import opf_residual_and_violation, dispatch_cost
                _need_parts = args.cost_weight > 0 or (args.xai_pen_weight > 0 and args.xai_pen_buses)
                if _need_parts:
                    _parts = opf_residual_and_violation(Y, V, S_load, space)
                else:
                    _parts = None
                cost_loss = (dispatch_cost(_parts["gen_p"].to(torch.float32), space)
                             if args.cost_weight > 0 else torch.zeros((), device=device))
                if args.xai_pen_weight > 0 and args.xai_pen_buses:
                    # gen_p shape: (B * n_nodes,); n = B * n_nodes from target.shape[1]
                    _gen_p_2d = _parts["gen_p"].float().view(B, n // B)
                    _xai_idx = [b - 1 for b in args.xai_pen_buses]
                    xai_pen_loss = _gen_p_2d[:, _xai_idx].pow(2).mean()
                else:
                    xai_pen_loss = torch.zeros((), device=device)
                loss = (args.mse_weight * mse + args.physics_weight * phys
                        + args.pinn_weight * pinn_loss
                        + args.cost_weight * cost_loss
                        + args.xai_pen_weight * xai_pen_loss)

                if train:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optim.step()

                om = opf_metrics(Y, V.detach(), target, S_load, space, batch["sizes"])
                for k, v_ in om.items():
                    opf_sums[k] = opf_sums.get(k, 0.0) + float(v_) * B
                if collect_residuals:
                    r = opf_residual_and_violation(Y, V.detach(), S_load, space)
                    free = ~r["ctrl"]
                    if int(free.sum().item()) > 0:
                        dp_chunks.append(r["dp"][free].detach().float().cpu())
                        dq_chunks.append(r["dq"][free].detach().float().cpu())
                for k, v_ in (("loss", loss), ("mse", mse), ("mse_mag", mse_mag),
                              ("mse_ang", mse_ang), ("phys", phys), ("cost", cost_loss),
                              ("xai_pen", xai_pen_loss)):
                    sums[k] += float(v_.item()) * B
        d = max(n_graphs, 1)
        out_d = {k: v / d for k, v in sums.items()}
        out_d["opf"] = {k: v / d for k, v in opf_sums.items()}
        if collect_residuals and dp_chunks:
            def summarize(chunks):
                x = torch.cat(chunks, dim=0)
                return {"inf": float(x.max()), "mean": float(x.mean()),
                        "p95": float(torch.quantile(x, 0.95)),
                        "p99": float(torch.quantile(x, 0.99)), "n": int(x.numel())}
            out_d["resid"] = {"dp": summarize(dp_chunks), "dq": summarize(dq_chunks)}
        return out_d

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
        print(f"Final test-set residuals (free buses, n={dp['n']}) : "
              f"dPinf {dp['inf']:.4e} | dQinf {dq['inf']:.4e} | "
              f"mean|dP| {dp['mean']:.4e} | mean|dQ| {dq['mean']:.4e} | "
              f"p95|dP| {dp['p95']:.4e} | p95|dQ| {dq['p95']:.4e} | "
              f"p99|dP| {dp['p99']:.4e} | p99|dQ| {dq['p99']:.4e} pu")

    if args.EPOCHS <= 0:
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
        if args.use_pdl:
            lam_bal[0] = min(args.pdl_clip, max(0.0,
                lam_bal[0] + _pdl_lr_bal * tr["opf"]["dp_inf_pu"]))
            lam_lim[0] = min(args.pdl_clip, max(0.0,
                lam_lim[0] + _pdl_lr_lim * tr["opf"]["p_limit_max_pu"]))
        if epoch % args.VAL_EVERY == 0 or epoch == args.EPOCHS:
            va = run_epoch(val_loader, train=False)
            pdl_str = f" | λbal={lam_bal[0]:.2f} λlim={lam_lim[0]:.2f}" if args.use_pdl else ""
            print(f"Epoch {epoch:3d} | {fmt('train', tr)} | {fmt('valid', va)}"
                  f"{pdl_str} | time {time.time()-t0:.2f}s")
            if va["loss"] < best_val:
                best_val = va["loss"]
                tmp_path = best_path + ".tmp"
                torch.save(model.state_dict(), tmp_path)
                os.replace(tmp_path, best_path)
                print(f"  checkpoint saved to {best_path}")
            feas_proxy = max(va["opf"].get("p_limit_max_pu", 0.0),
                             va["opf"].get("dp_inf_pu", 0.0))
            if feas_proxy < best_feas:
                best_feas = feas_proxy
                tmp_path = best_feas_path + ".tmp"
                torch.save(model.state_dict(), tmp_path)
                os.replace(tmp_path, best_feas_path)
                print(f"  checkpoint (feas proxy={feas_proxy:.4e}) saved to {best_feas_path}")
        else:
            pdl_str = f" | λbal={lam_bal[0]:.2f} λlim={lam_lim[0]:.2f}" if args.use_pdl else ""
            print(f"Epoch {epoch:3d} | {fmt('train', tr)}{pdl_str} | time {time.time()-t0:.2f}s")

    # For PDL runs prefer the feasibility checkpoint; fall back to loss checkpoint.
    load_path = (best_feas_path
                 if args.use_pdl and os.path.exists(best_feas_path)
                 else best_path)
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"[test] loaded checkpoint: {load_path}")
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

"""Generate a synthetic AC power-flow dataset for an N-bus ring grid.

Supports variable N for multi-size generalization experiments (Objective 5).
Structure: bus 0 = slack, bus 1 = PV generator, buses 2..N-1 = PQ loads.
Includes economic dispatch OPF labels (Pg_opt, V_opt_mag, V_opt_ang, cost_opt).

Usage:
    python datasets/generate_ring.py [N] [n_samples] [output_path]
    python datasets/generate_ring.py 8 1000
    python datasets/generate_ring.py 10 800 datasets/synthetic_opf_n10_ring.parquet
"""
from __future__ import annotations

import io
import sys
import os

import numpy as np
import pandas as pd


# ── encoding helpers ─────────────────────────────────────────────────────────

def _npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


# ── Economic dispatch constants ───────────────────────────────────────────────
C1_PV    = 2.0
C2_PV    = 0.5
C1_SLACK = 3.0
C2_SLACK = 0.8
PG_LIM_FRAC = 0.30


def _economic_dispatch(P_demand: float, Pg_nom_pv: float) -> tuple[float, float, float]:
    Pg_min = max(0.0, Pg_nom_pv * (1.0 - PG_LIM_FRAC))
    Pg_max = Pg_nom_pv * (1.0 + PG_LIM_FRAC)
    Pg_pv  = float(np.clip(
        (C1_SLACK - C1_PV + 2.0 * C2_SLACK * P_demand) / (2.0 * (C2_PV + C2_SLACK)),
        Pg_min, Pg_max,
    ))
    Pg_sl  = P_demand - Pg_pv
    cost   = C1_PV * Pg_pv + C2_PV * Pg_pv**2 + C1_SLACK * Pg_sl + C2_SLACK * Pg_sl**2
    return Pg_pv, Pg_sl, float(cost)


# ── Ring grid builder ─────────────────────────────────────────────────────────

def build_ring_constants(N: int):
    """Build static arrays for an N-bus ring grid.

    Returns a dict with BUS_TYP, PAIRS_I, PAIRS_J, LINES, Y_LINES_BASE, Y_C_BASE,
    P_NOM, Q_NOM.
    """
    assert N >= 3, "Ring requires at least 3 buses"

    BUS_TYP = np.zeros(N, dtype=np.int64)
    BUS_TYP[0] = 1   # slack
    BUS_TYP[1] = 2   # PV generator
    BUS_TYP[2:] = 3  # PQ loads

    PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)
    E = len(PAIRS_I)
    _pair_map = {(int(i), int(j)): idx for idx, (i, j) in enumerate(zip(PAIRS_I, PAIRS_J))}

    # Ring topology
    RING_EDGES = [(i, i + 1) for i in range(N - 1)] + [(0, N - 1)]
    LINES = np.zeros(E, dtype=bool)
    for a, b in RING_EDGES:
        if a > b:
            a, b = b, a
        LINES[_pair_map[(a, b)]] = True

    # Base admittances
    _base_ys = 4.0 - 1.2j
    _base_yc = 0.005j
    Y_LINES_BASE = np.zeros(E, dtype=np.complex64)
    Y_C_BASE     = np.zeros(E, dtype=np.complex64)
    # Vary slightly per edge for diversity
    rng_base = np.random.default_rng(0)
    _scale = 1.0 + 0.1 * (rng_base.random(N) - 0.5)
    for k, (a, b) in enumerate(RING_EDGES):
        if a > b:
            a, b = b, a
        idx = _pair_map[(a, b)]
        Y_LINES_BASE[idx] = _base_ys * float(_scale[k % N])
        Y_C_BASE[idx]     = _base_yc

    # Nominal powers: 1 PV generator + (N-2) PQ loads
    n_loads = N - 2
    P_load_each = 0.35  # pu per load (total demand ≈ 0.5 pu for N=5)
    Q_load_each = 0.15
    P_NOM = np.zeros(N, dtype=np.float32)
    Q_NOM = np.zeros(N, dtype=np.float32)
    P_NOM[1]  =  0.5   # PV nominal injection
    P_NOM[2:] = -P_load_each
    Q_NOM[2:] = -Q_load_each

    return dict(
        N=N, BUS_TYP=BUS_TYP,
        PAIRS_I=PAIRS_I, PAIRS_J=PAIRS_J, E=E,
        LINES=LINES, _pair_map=_pair_map,
        Y_LINES_BASE=Y_LINES_BASE, Y_C_BASE=Y_C_BASE,
        P_NOM=P_NOM, Q_NOM=Q_NOM,
    )


def _build_ybus(N, PAIRS_I, PAIRS_J, LINES, y_lines, y_c):
    Ybus = np.zeros((N, N), dtype=np.complex64)
    for idx, (i, j) in enumerate(zip(PAIRS_I, PAIRS_J)):
        if not LINES[idx]:
            continue
        ys = y_lines[idx]
        yc = y_c[idx]
        Ybus[i, j] -= ys
        Ybus[j, i] -= ys
        Ybus[i, i] += ys + yc
        Ybus[j, j] += ys + yc
    return Ybus


def _nr_solve(Ybus, P_set, Q_set, bus_typ, v0, th0, iters=50):
    v, th = v0.copy(), th0.copy()
    slack = (bus_typ == 1)
    pv    = (bus_typ == 2)
    for _ in range(iters):
        Vc = v * np.exp(1j * th)
        Sc = Vc * np.conj(Ybus @ Vc)
        dP = np.where(slack, 0.0, P_set - Sc.real)
        dQ = np.where(slack | pv, 0.0, Q_set - Sc.imag)
        if np.max(np.abs(np.concatenate([dP, dQ]))) < 1e-6:
            break
        th += 0.05 * dP
        v  += 0.05 * dQ
        v   = np.clip(v, 0.75, 1.25)
    return v.astype(np.float32), th.astype(np.float32)


def generate_sample(rng: np.random.Generator, grid: dict) -> dict:
    """Generate one sample for the given N-bus ring grid."""
    N         = grid["N"]
    BUS_TYP   = grid["BUS_TYP"]
    PAIRS_I   = grid["PAIRS_I"]
    PAIRS_J   = grid["PAIRS_J"]
    E         = grid["E"]
    LINES     = grid["LINES"]
    P_NOM     = grid["P_NOM"]
    Q_NOM     = grid["Q_NOM"]
    Y_LINES_BASE = grid["Y_LINES_BASE"]
    Y_C_BASE     = grid["Y_C_BASE"]

    pq_mask     = (BUS_TYP == 3)
    load_factor = 1.0 + 0.4 * (rng.random(N).astype(np.float32) - 0.5)
    P_set = np.where(pq_mask, P_NOM * load_factor, P_NOM)
    Q_set = np.where(pq_mask, Q_NOM * load_factor, Q_NOM)

    y_scale = 1.0 + 0.05 * (rng.random(E).astype(np.float32) - 0.5)
    y_lines = (Y_LINES_BASE * y_scale).astype(np.complex64)
    y_c     = Y_C_BASE.copy()

    Ybus = _build_ybus(N, PAIRS_I, PAIRS_J, LINES, y_lines, y_c)

    v0  = np.ones(N, dtype=np.float32)
    th0 = np.zeros(N, dtype=np.float32)
    v0[BUS_TYP == 2] = 1.02

    v_sol, th_sol = _nr_solve(Ybus, P_set, Q_set, BUS_TYP, v0.copy(), th0.copy())

    u_start  = (v0   * np.exp(1j * th0)).astype(np.complex64)
    u_newton = (v_sol * np.exp(1j * th_sol)).astype(np.complex64)
    S_start  = (P_set + 1j * Q_set).astype(np.complex64)

    # OPF labels
    P_demand = float(-np.sum(P_set[pq_mask]))
    Pg_nom_pv = float(P_NOM[1])
    Pg_pv_opt, Pg_sl_opt, cost_opt = _economic_dispatch(P_demand, Pg_nom_pv)

    P_set_opt = P_set.copy()
    P_set_opt[1] = Pg_pv_opt
    v_opt, th_opt = _nr_solve(Ybus, P_set_opt, Q_set, BUS_TYP, v0.copy(), th0.copy())

    Pg_opt_arr = np.zeros(N, dtype=np.float32)
    Pg_opt_arr[0] = float(Pg_sl_opt)
    Pg_opt_arr[1] = float(Pg_pv_opt)

    return {
        "bus_typ":         _npy(BUS_TYP),
        "Lines_connected": _npy(LINES),
        "Y_Lines":         _npy(y_lines),
        "Y_C_Lines":       _npy(y_c.imag.astype(np.float32)),
        "Y_matrix":        _npy(Ybus.reshape(-1)),
        "u_start":         _npy(u_start),
        "u_newton":        _npy(u_newton),
        "S_start":         _npy(S_start),
        "S_newton":        _npy(S_start.copy()),
        "U_base":          1.0,
        "S_base":          1.0,
        "Pg_opt":          _npy(Pg_opt_arr),
        "V_opt_mag":       _npy(v_opt),
        "V_opt_ang":       _npy(th_opt),
        "cost_opt":        float(cost_opt),
    }


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_bus     = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    default_out = os.path.join(
        os.path.dirname(__file__), f"synthetic_opf_n{N_bus}_ring.parquet"
    )
    out_path = sys.argv[3] if len(sys.argv) > 3 else default_out

    print(f"Generating {n_samples} samples for {N_bus}-bus ring -> {out_path}")
    grid = build_ring_constants(N_bus)
    rng  = np.random.default_rng(42)
    rows = [generate_sample(rng, grid) for _ in range(n_samples)]
    df   = pd.DataFrame(rows)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"Done. Shape: {df.shape}  Columns: {list(df.columns)}")

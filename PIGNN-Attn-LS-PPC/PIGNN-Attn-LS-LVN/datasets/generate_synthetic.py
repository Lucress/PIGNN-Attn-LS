"""Generate a small synthetic AC power-flow dataset for smoke-testing the OPF model.

Produces N samples of a 5-bus ring grid with random load variations.
Also generates OPF labels via closed-form economic dispatch (DC-OPF) followed by
an AC power-flow solve at the optimal dispatch point.

The file is intentionally NOT named LVN_converted_* so the dataset loader
uses the lighter eager (non-lazy) path.

Usage:
    python datasets/generate_synthetic.py [n_samples]
    python datasets/generate_synthetic.py 2000
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


# ── Economic dispatch constants (must match opf_model.py defaults) ────────────
C1_PV    = 2.0    # linear  cost coeff for PV generator (bus 1)
C2_PV    = 0.5    # quadratic cost coeff for PV generator
C1_SLACK = 3.0    # linear  cost coeff for slack bus  (bus 0, more expensive)
C2_SLACK = 0.8    # quadratic cost coeff for slack bus
PG_LIM_FRAC = 0.30   # ±30 % limits around nominal PV dispatch


def _economic_dispatch(P_demand: float, Pg_nom_pv: float) -> tuple[float, float, float]:
    """Closed-form DC economic dispatch for the 2-generator (PV + slack) system.

    Minimises  c1_pv*Pg_pv + c2_pv*Pg_pv^2 + c1_sl*(Pd-Pg_pv) + c2_sl*(Pd-Pg_pv)^2
    subject to Pg_min ≤ Pg_pv ≤ Pg_max.

    Returns (Pg_pv_opt, Pg_slack_opt, cost_opt).
    """
    Pg_min = max(0.0, Pg_nom_pv * (1.0 - PG_LIM_FRAC))
    Pg_max = Pg_nom_pv * (1.0 + PG_LIM_FRAC)
    # Unconstrained optimum from KKT stationarity
    Pg_pv_unc = (C1_SLACK - C1_PV + 2.0 * C2_SLACK * P_demand) / (2.0 * (C2_PV + C2_SLACK))
    Pg_pv = float(np.clip(Pg_pv_unc, Pg_min, Pg_max))
    Pg_sl = P_demand - Pg_pv
    cost  = C1_PV * Pg_pv + C2_PV * Pg_pv**2 + C1_SLACK * Pg_sl + C2_SLACK * Pg_sl**2
    return Pg_pv, Pg_sl, float(cost)


# ── 5-bus ring grid constants ─────────────────────────────────────────────────

N = 5
# bus types:  0=slack  1=PV  2=PQ  3=PQ  4=PQ  (1-indexed in the dataset)
BUS_TYP = np.array([1, 2, 3, 3, 3], dtype=np.int64)   # 1=slack, 2=PV, 3=PQ

# Pair indices for upper triangle (row i < col j), N=5 → E=10
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)   # i is row (smaller index)
E = len(PAIRS_I)

# Build pair lookup: (i,j) → flat index
_pair_map = {}
for idx, (i, j) in enumerate(zip(PAIRS_I, PAIRS_J)):
    _pair_map[(int(i), int(j))] = idx

# Ring topology: 0-1, 1-2, 2-3, 3-4, 0-4
RING_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
LINES = np.zeros(E, dtype=bool)
for (a, b) in RING_EDGES:
    if a > b:
        a, b = b, a
    LINES[_pair_map[(a, b)]] = True

# Series admittances for the 5 active lines (pu)
_Ys_vals = np.array(
    [4.0 - 1.2j, 3.5 - 1.0j, 5.0 - 1.5j, 4.5 - 1.3j, 3.0 - 0.8j],
    dtype=np.complex64,
)
Y_LINES_BASE = np.zeros(E, dtype=np.complex64)
Y_LINES_BASE[LINES] = _Ys_vals

# Shunt admittances (purely capacitive, small)
Y_C_BASE = np.zeros(E, dtype=np.complex64)
Y_C_BASE[LINES] = 0.005j


def _build_ybus(y_lines: np.ndarray, y_c: np.ndarray) -> np.ndarray:
    """Build N×N dense Y_bus from edge arrays (pu)."""
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


# ── AC power-flow solver (Newton-Raphson, 50 iterations) ─────────────────────

def _nr_solve(Ybus: np.ndarray, P_set: np.ndarray, Q_set: np.ndarray,
              bus_typ: np.ndarray, v0: np.ndarray, th0: np.ndarray,
              iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Simple full-Newton power-flow for the test grid."""
    v, th = v0.copy(), th0.copy()
    slack = (bus_typ == 1)
    pv    = (bus_typ == 2)
    pq    = (bus_typ == 3)

    for _ in range(iters):
        Vc = v * np.exp(1j * th)
        Ic = Ybus @ Vc
        Sc = Vc * np.conj(Ic)
        dP = np.where(slack, 0.0, P_set - Sc.real)
        dQ = np.where(slack | pv, 0.0, Q_set - Sc.imag)
        if np.max(np.abs(np.concatenate([dP, dQ]))) < 1e-6:
            break
        # simple gradient step (not true Newton, but good enough for synthetic)
        th += 0.05 * dP
        v  += 0.05 * dQ
        v   = np.clip(v, 0.75, 1.25)

    return v.astype(np.float32), th.astype(np.float32)


# ── sample generator ─────────────────────────────────────────────────────────

def generate_sample(rng: np.random.Generator) -> dict:
    """Generate one sample with randomised load levels."""
    # Randomise PQ bus loads (±40% of nominal)
    P_nom = np.array([0.0,  0.5, -0.4, -0.35, -0.3], dtype=np.float32)
    Q_nom = np.array([0.0,  0.0, -0.2, -0.15, -0.1], dtype=np.float32)
    load_factor = 1.0 + 0.4 * (rng.random(N).astype(np.float32) - 0.5)
    pq_mask = (BUS_TYP == 3)
    P_set = np.where(pq_mask, P_nom * load_factor, P_nom)
    Q_set = np.where(pq_mask, Q_nom * load_factor, Q_nom)

    # Slightly perturb admittances (±5%) for diversity
    y_scale = 1.0 + 0.05 * (rng.random(E).astype(np.float32) - 0.5)
    y_lines = (Y_LINES_BASE * y_scale).astype(np.complex64)
    y_c     = Y_C_BASE.copy()

    Ybus = _build_ybus(y_lines, y_c)

    # Flat start
    v0  = np.ones(N, dtype=np.float32)
    th0 = np.zeros(N, dtype=np.float32)
    # Slight PV voltage set-point
    v0[BUS_TYP == 2] = 1.02

    # Compute a decent starting solution
    v_sol, th_sol = _nr_solve(Ybus, P_set, Q_set, BUS_TYP, v0.copy(), th0.copy())

    # Complex voltage
    u_start  = (v0  * np.exp(1j * th0)).astype(np.complex64)
    u_newton = (v_sol * np.exp(1j * th_sol)).astype(np.complex64)

    S_start  = (P_set + 1j * Q_set).astype(np.complex64)
    S_newton = S_start.copy()

    # ── OPF labels: economic dispatch + AC-PF at the optimal setpoint ────────
    P_demand = float(-np.sum(P_set[pq_mask]))   # total load (positive scalar)
    Pg_nom_pv = float(P_nom[1])                  # nominal PV output = 0.5 pu
    Pg_pv_opt, Pg_sl_opt, cost_opt = _economic_dispatch(P_demand, Pg_nom_pv)

    # AC power flow at optimal dispatch (P_set[1] ← Pg_pv_opt)
    P_set_opt = P_set.copy()
    P_set_opt[1] = Pg_pv_opt
    v_opt, th_opt = _nr_solve(Ybus, P_set_opt, Q_set, BUS_TYP, v0.copy(), th0.copy())

    # Per-bus optimal dispatch (non-zero only at generator buses)
    Pg_opt_arr = np.zeros(N, dtype=np.float32)
    Pg_opt_arr[0] = float(Pg_sl_opt)    # slack
    Pg_opt_arr[1] = float(Pg_pv_opt)   # PV

    return {
        "bus_typ":         _npy(BUS_TYP),
        "Lines_connected": _npy(LINES),
        "Y_Lines":         _npy(y_lines),
        "Y_C_Lines":       _npy(y_c.imag.astype(np.float32)),   # store as real susceptance
        "Y_matrix":        _npy(Ybus.reshape(-1)),
        "u_start":         _npy(u_start),
        "u_newton":        _npy(u_newton),
        "S_start":         _npy(S_start),
        "S_newton":        _npy(S_newton),
        "U_base":          1.0,
        "S_base":          1.0,
        # OPF labels
        "Pg_opt":          _npy(Pg_opt_arr),
        "V_opt_mag":       _npy(v_opt),
        "V_opt_ang":       _npy(th_opt),
        "cost_opt":        float(cost_opt),
    }


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    out_path  = os.path.join(os.path.dirname(__file__), "synthetic_opf_n5_ring.parquet")

    print(f"Generating {n_samples} samples -> {out_path}")
    rng = np.random.default_rng(42)

    rows = [generate_sample(rng) for _ in range(n_samples)]
    df   = pd.DataFrame(rows)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"Done. Shape: {df.shape}  Columns: {list(df.columns)}")

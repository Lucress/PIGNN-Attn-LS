"""
perturbation_options.py
=======================

Selectable, option-gated perturbation primitives for the NR / PIGNN-Attn-LS
data-generation pipeline.

The whole point of this module is that *every* perturbation is opt-in and
configured by an explicit option, so you can build a spectrum of datasets
ranging from "load-only, fixed Y-bus" (cheap, share_grid-compatible) up to
"load + topology + admittance" (expensive, Y-bus varies per row).

Two families of perturbation, distinguished by whether they change Y_bus:

  FAMILY 1 — Y_bus FIXED  (share_grid=True is SAFE and fast)
      * global load scale          (load_perturbation)
      * per-bus P / Q load jitter
      * per-generator P jitter
      * PV voltage-setpoint jitter
      * start-point noise (angle / magnitude)

  FAMILY 2 — Y_bus VARIES  (share_grid MUST be False — correctness, not speed)
      * topology contingency       (sample_contingency)
      * admittance R/X jitter       (apply_admittance_jitter)

`PerturbationConfig.ybus_varies` tells the rest of the pipeline which family a
given dataset belongs to, so the training-time loader can pick share_grid
automatically and never silently reuse row-0's Y-bus for a row whose topology
or admittance is different.

Contingency sampling implements three research-backed schemes (see the journal
notes / perturbation_design.tex):

  * "bernoulli" — legacy: each line out independently w.p. p  (no k control)
  * "ratio"     — PF-Delta style: draw the contingency ORDER k from a fixed
                  categorical, e.g. P(N-0,N-1,N-2) = (0.55, 0.27, 0.18)
  * "poisson"   — probabilistically realistic: k ~ Poisson(L * q), capped,
                  so the expected number of simultaneous outages scales with
                  grid size (small grids ~N-0/N-1, large grids up to N-k)

All schemes enforce connectivity (slack can reach every bus) and never drop
the reference/slack generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Connectivity helper
# ----------------------------------------------------------------------------
def _is_connected(net) -> bool:
    """True if no bus is left unsupplied (every bus reachable from a source)."""
    try:
        from pandapower.topology import unsupplied_buses
        return len(unsupplied_buses(net)) == 0
    except Exception:
        # If the topology routine itself fails, be conservative.
        return False


def _active_line_index(net) -> np.ndarray:
    if not hasattr(net, "line") or len(net.line) == 0:
        return np.array([], dtype=int)
    return np.asarray(net.line.index[net.line["in_service"].to_numpy(bool)], dtype=int)


def _droppable_gen_index(net) -> np.ndarray:
    """
    Generators that may be taken offline.  The slack / reference machine is
    never droppable (its outage would leave the system without an angle
    reference).  pandapower's slack is typically an ext_grid; a `gen` row
    flagged slack=True is also excluded.
    """
    if not hasattr(net, "gen") or len(net.gen) == 0:
        return np.array([], dtype=int)
    g = net.gen
    mask = g["in_service"].to_numpy(bool)
    if "slack" in g.columns:
        mask &= ~g["slack"].to_numpy(bool)
    return np.asarray(g.index[mask], dtype=int)


# ----------------------------------------------------------------------------
# Contingency order sampler  (how many components to drop this sample)
# ----------------------------------------------------------------------------
def _draw_order(
    mode: str,
    n_branches: int,
    rng,
    *,
    bernoulli_p: float,
    ratio_weights: Sequence[float],
    poisson_q: float,
    k_cap: int,
) -> int:
    """Return the contingency order k (number of components to drop)."""
    if mode == "none":
        return 0

    if mode == "bernoulli":
        # Legacy behaviour kept for reproducibility: number of lines out is
        # Binomial(L, p).  No explicit k control; we still cap it.
        if n_branches == 0:
            return 0
        k = int(np.sum(rng.random(n_branches) < bernoulli_p))
        return min(k, k_cap)

    if mode == "ratio":
        # PF-Delta style: P(N-0), P(N-1), P(N-2), ...
        w = np.asarray(ratio_weights, dtype=float)
        w = w / w.sum()
        return int(rng.choice(len(w), p=w))

    if mode == "poisson":
        # Size-scaled realism: expected outages = L * q.
        lam = max(n_branches, 0) * float(poisson_q)
        k = int(rng.poisson(lam))
        return min(k, k_cap)

    raise ValueError(f"Unknown contingency mode {mode!r}")


@dataclass
class ContingencyResult:
    """Description of what was dropped (for logging / dataset columns)."""
    order: int = 0                      # realised k (components actually out)
    dropped_lines: List[int] = field(default_factory=list)
    dropped_gens: List[int] = field(default_factory=list)
    changed_ybus: bool = False          # True iff any line was dropped


def sample_contingency(
    net,
    rng,
    *,
    mode: str = "none",                 # "none" | "bernoulli" | "ratio" | "poisson"
    elements: Sequence[str] = ("line",),  # subset of {"line", "gen"}
    # --- bernoulli mode ---
    bernoulli_p: float = 0.0,
    # --- ratio mode (PF-Delta) ---
    ratio_weights: Sequence[float] = (0.55, 0.27, 0.18),  # P(N-0,N-1,N-2)
    # --- poisson mode (size-scaled) ---
    poisson_q: float = 5e-4,            # per-branch steady-state unavailability
    # --- shared ---
    k_cap: int = 2,                     # hard cap on simultaneous outages
    max_tries: int = 25,                # connectivity retries before giving up
) -> ContingencyResult:
    """
    Draw and APPLY a topology contingency in place on `net`.

    Selectable by `mode`:
        "none"      -> never drops anything (N-0 only).
        "bernoulli" -> legacy per-line Bernoulli(p), capped at k_cap.
        "ratio"     -> draw order k from a fixed categorical `ratio_weights`
                       (e.g. PF-Delta's ~55/27/18 for N-0/N-1/N-2).
        "poisson"   -> draw k ~ Poisson(L * poisson_q), capped — k grows with
                       grid size, so small grids stay near N-0/N-1 and large
                       grids occasionally reach higher orders.

    `elements` chooses what may be dropped: lines only, or lines and
    generators (generator outages create a real supply deficit the slack must
    absorb — useful hard cases, never drops the slack itself).

    Connectivity is enforced: if a draw islands the network, it is reverted and
    re-drawn up to `max_tries`; if no feasible contingency of the drawn order is
    found, the sample falls back to N-0 (nothing dropped).

    NOTE: dropping a *line* changes Y_bus (sets `changed_ybus=True`); dropping a
    *generator* does NOT change Y_bus (it changes the injection / bus type).
    Datasets using line outages therefore require share_grid=False.
    """
    res = ContingencyResult()
    if mode == "none":
        return res

    line_pool = _active_line_index(net) if "line" in elements else np.array([], dtype=int)
    gen_pool = _droppable_gen_index(net) if "gen" in elements else np.array([], dtype=int)
    pool = [("line", i) for i in line_pool] + [("gen", i) for i in gen_pool]
    if len(pool) == 0:
        return res

    k = _draw_order(
        mode, n_branches=len(line_pool), rng=rng,
        bernoulli_p=bernoulli_p, ratio_weights=ratio_weights,
        poisson_q=poisson_q, k_cap=k_cap,
    )
    k = min(k, len(pool))
    if k <= 0:
        return res

    for _ in range(max_tries):
        sel = rng.choice(len(pool), size=k, replace=False)
        picked = [pool[s] for s in sel]
        lines = [i for kind, i in picked if kind == "line"]
        gens = [i for kind, i in picked if kind == "gen"]

        # apply
        if lines:
            net.line.loc[lines, "in_service"] = False
        if gens:
            net.gen.loc[gens, "in_service"] = False

        if _is_connected(net):
            res.order = k
            res.dropped_lines = list(map(int, lines))
            res.dropped_gens = list(map(int, gens))
            res.changed_ybus = len(lines) > 0
            return res

        # revert and retry
        if lines:
            net.line.loc[lines, "in_service"] = True
        if gens:
            net.gen.loc[gens, "in_service"] = True

    # No feasible contingency of this order found -> N-0 fallback.
    return res


# ----------------------------------------------------------------------------
# Admittance perturbation  (R / X jitter -> Y_bus varies per sample)
# ----------------------------------------------------------------------------
def apply_admittance_jitter(
    net,
    rng,
    *,
    sigma: float = 0.0,                 # 0.0 disables; e.g. 0.1 = +/-10%
    jitter_lines: bool = True,
    jitter_trafos: bool = True,
) -> bool:
    """
    Multiply branch resistance and reactance by independent uniform factors
    drawn from U(1-sigma, 1+sigma), per branch, per sample.  Recomputing Y_bus
    from the perturbed branch data is handled downstream by the normal PPC
    compilation (this only edits the pandapower element tables).

    Models temperature/measurement uncertainty in line parameters; also acts as
    data augmentation so the surrogate learns sensitivity to Y_bus, not just to
    the injections.  Returns True iff anything was perturbed (Y_bus varies).

    Selectable: set sigma=0.0 to disable entirely, or toggle lines/trafos.
    """
    if sigma <= 0.0:
        return False

    lo, hi = max(0.0, 1.0 - sigma), 1.0 + sigma
    changed = False

    if jitter_lines and hasattr(net, "line") and len(net.line):
        n = len(net.line)
        net.line["r_ohm_per_km"] = net.line["r_ohm_per_km"].to_numpy(float) * rng.uniform(lo, hi, n)
        net.line["x_ohm_per_km"] = net.line["x_ohm_per_km"].to_numpy(float) * rng.uniform(lo, hi, n)
        changed = True

    if jitter_trafos and hasattr(net, "trafo") and len(net.trafo):
        n = len(net.trafo)
        # vk_percent ~ |z|, vkr_percent ~ r.  Scale both; keep vkr <= vk.
        vk = net.trafo["vk_percent"].to_numpy(float) * rng.uniform(lo, hi, n)
        vkr = net.trafo["vkr_percent"].to_numpy(float) * rng.uniform(lo, hi, n)
        net.trafo["vk_percent"] = vk
        net.trafo["vkr_percent"] = np.minimum(vkr, 0.999 * vk)
        changed = True

    return changed


# ----------------------------------------------------------------------------
# Top-level config: records which knobs change Y_bus
# ----------------------------------------------------------------------------
@dataclass
class PerturbationConfig:
    """
    One object describing an entire perturbation regime.  Pass the relevant
    fields into case_generation_pandapower(); read `ybus_varies` to decide
    share_grid at training time.
    """
    # ---- Family 1: Y_bus FIXED ----
    load_scale_range: Optional[Tuple[float, float]] = None
    scale_gen_with_load: bool = True
    jitter_load: float = 0.0
    jitter_load_q: float = 0.0
    jitter_gen: float = 0.0
    pv_vset_range: Optional[Tuple[float, float]] = None
    rand_u_start: bool = False
    angle_jitter_deg: float = 0.0
    mag_jitter_pq: float = 0.0

    # ---- Family 2: Y_bus VARIES ----
    contingency_mode: str = "none"      # none|bernoulli|ratio|poisson
    contingency_elements: Tuple[str, ...] = ("line",)
    contingency_ratio_weights: Tuple[float, ...] = (0.55, 0.27, 0.18)
    contingency_poisson_q: float = 5e-4
    contingency_k_cap: int = 2
    admittance_sigma: float = 0.0

    @property
    def ybus_varies(self) -> bool:
        """
        True if this regime can change Y_bus across rows, in which case the
        training loader MUST use share_grid=False (else every row silently
        gets row-0's Y_bus).  Generator-only contingencies do NOT change Y_bus.
        """
        topo_changes_ybus = (
            self.contingency_mode != "none" and "line" in self.contingency_elements
        )
        return bool(topo_changes_ybus or self.admittance_sigma > 0.0)

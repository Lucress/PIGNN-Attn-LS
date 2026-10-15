# Experiments Log — PIGNN-Attn-LS

All results on IEEE case118, test set, feasibility = all of {dP, dQ, VP, VQ} < 0.01 pu simultaneously.
Metric defined in `PIGNN-Attn-LS-PPC/opf_task.py:339` (`_FEAS_THRESH = 0.01`).

---

## Main Results (for Table 1 in paper)

| Model | Seed | Feas Rate | Cost Gap | dP_inf | P-lim max | Notes |
|-------|------|-----------|----------|--------|-----------|-------|
| PIGNN-Attn-LS **dhi32** | s1 | **97.6%** | +0.38% | 2.776e-3 | 7.002e-3 | **Phase-2 best** |
| PIGNN-Attn-LS dhi32 | s2 | 96.7% | +1.45% | — | — | |
| PIGNN-Attn-LS dhi32 | s3 | 92.1% | +0.97% | — | — | |
| **PIGNN-Attn-LS dhi32 mean** | — | **95.5% ± 2.3%** | **+0.93% ± 0.44%** | — | — | 3-seed summary |
| PIGNN-Attn-LS v15 (baseline, dhi=16) | s1 | ~96% | ~0.5% | — | — | Previous best |
| GridFM (mirror impl, 20M params) | — | **0.0%** | −4.75% | 5.580e-01 | 4.676e-01 | Job 1839459 done; dQinf=2.416 pu |

---

## Ablation Study

| Ablation | Feas Rate | Cost Gap | Key Finding |
|----------|-----------|----------|-------------|
| No attention (num_attn_layers=0) | **0.0%** | N/A | Attention is structurally required |
| Cost weight (cost_wt=0.1, λ reset) | **0.0%** | N/A | λ_lim reset→1.0 on resume; cost loss dominates |
| d=8, d_hi=32 (wider input embed) | 83.5% | +0.50% | d=4 is correct; wider input embedding hurts |

---

## Architecture Search

| Config | d | d_hi | Feas (s1) | Cost Gap | Notes |
|--------|---|------|-----------|----------|-------|
| v15 (baseline) | 4 | 16 | ~96% | ~0.5% | |
| **dhi32 (best)** | 4 | **32** | **97.6%** | +0.38% | Doubling hidden dim is the key |
| d8dhi32 | 8 | 32 | 83.5% | +0.50% | Wider embedding hurts convergence |

---

## Training Protocol — Two-Stage PDL

**Phase 1**: `pdl_lr_lim=0.08`, 200 epochs, LR=5e-5, random init  
**Phase 2**: `pdl_lr_lim=0.16`, 200 epochs, LR=2e-5, warm-start from Phase-1 best_feas  
Note: `resume_state_dict` resets λ_lim → 1.0 (loads primal weights only).

---

## XAI Findings (dhi32 Phase-2)

| Bus | Role | attn_feas | attn_fail | ratio | Interpretation |
|-----|------|-----------|-----------|-------|----------------|
| 46 | Free bus | 1.0000 | 1.0000 | 1.00 | Uniform attention |
| 49 | Free bus | 1.0000 | 1.0000 | 1.00 | Uniform attention |
| 31 | Free bus | 1.0000 | 1.0000 | 1.00 | Uniform attention |
| 59 | Ctrl bus | 1.0000 | 1.0000 | 1.00 | Structural bottleneck (load +0.06 pu) |

**Finding**: Edge self-attention saturates (softmax → 1/N uniform) in both feasible and infeasible scenarios.  
Attention is architecturally necessary but does NOT encode interpretable dispatch decisions.  
Bus 59 is a structural bottleneck regardless of attention.

---

## Checkpoint Locations (on tinyx cluster)

```
BASE=/home/hpc/iwso/iwso230h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/tinyx/

pignn_dhi32_p2_case118_best_feas.pt    ← MAIN MODEL (97.6% feas)
pignn_dhi32_p1_case118_best_feas.pt    ← Phase-1 checkpoint
pignn_dhi32_s2_p2_case118_best_feas.pt ← Seed 2
pignn_dhi32_s3_p2_case118_best_feas.pt ← Seed 3
pignn_pdl15_case118_best_feas.pt       ← v15 baseline
gridfm_case118_best_feas.pt            ← GridFM (pending)
```

---

## Feasibility Metric — Attribution

- Defined in `opf_task.py:339`, authored by changhunkim (supervisor), commit `14e7a37` (Aug 11 2026)
- NOT from any paper — supervisor's own threshold for cross-model comparison (GridFM / LUMINA / PIGNN)
- Physical grounding: 0.01 pu = 1 MW on 100 MVA base ≈ Newton-Raphson convergence tolerance
- SOTA papers (DeepOPF, DC3, OPFData) report mean/max violations, not binary feasibility rate

---

## Pending

- [x] GridFM baseline results (job 1839459, done Oct 15) — **0% feas, dPinf=0.558 pu, dQinf=2.416 pu**
- [ ] PDL-v15 exact feas/cost numbers from eval log
- [ ] 3-seed results for v15 (for fair comparison table)

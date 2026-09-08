"""Generate combined figures for all three deep analyses."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figures/deep_analysis"
os.makedirs(OUT, exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────
with open("results/deep_supervised/step_correlation.json")  as f: sup_sc = json.load(f)
with open("results/deep_pinn/step_correlation.json")         as f: pnn_sc = json.load(f)
with open("results/deep_supervised/head_specialization.json") as f: sup_hs = json.load(f)
with open("results/deep_pinn/head_specialization.json")       as f: pnn_hs = json.load(f)
with open("results/source_attention/source_attention_results.json") as f: src = json.load(f)

TYPE_LABEL = {"slack": "Slack\n(ref bus)", "gen": "Generator\n(PV)", "load": "Load\n(PQ)"}

# ═══════════════════════════════════════════════════════════════════════════
# Figure A: r(k) per correction step — both models
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

r_sup  = sup_sc["r_by_step"]
r_pinn = pnn_sc["r_by_step"]
K_sup  = len(r_sup)
K_pinn = len(r_pinn)

# per-head curves (lighter)
H = len(sup_sc["r_by_step_per_head"])
colors_h_sup  = ["#00CFA8"] * H
colors_h_pinn = ["#FF6B35"] * H
for h in range(H):
    ax.plot(range(1, K_sup +1), sup_sc["r_by_step_per_head"][h],
            lw=0.9, alpha=0.30, color="#00CFA8")
    ax.plot(range(1, K_pinn+1), pnn_sc["r_by_step_per_head"][h],
            lw=0.9, alpha=0.30, color="#FF6B35")

# mean curves
ax.plot(range(1, K_sup +1), r_sup,  lw=2.5, color="#00CFA8",
        label=f"Supervised (K={K_sup}) — mean heads")
ax.plot(range(1, K_pinn+1), r_pinn, lw=2.5, color="#FF6B35",
        label=f"PINN (K={K_pinn}) — mean heads")

# annotate key behaviours
ax.annotate("Supervised: starts strong\nthen partially relaxes",
            xy=(1, r_sup[0]), xytext=(3.5, r_sup[0]-0.02),
            fontsize=8, color="#00CFA8",
            arrowprops=dict(arrowstyle="->", color="#00CFA8", lw=1.2))
ax.annotate("PINN: stable throughout\n(physics structure is baked in)",
            xy=(K_pinn//2, r_pinn[K_pinn//2]), xytext=(K_pinn//2-8, r_pinn[K_pinn//2]+0.06),
            fontsize=8, color="#FF6B35",
            arrowprops=dict(arrowstyle="->", color="#FF6B35", lw=1.2))

ax.axhline(0, color="#555", lw=0.8, ls="--")
ax.set_xlabel("Correction step k", fontsize=11)
ax.set_ylabel("Spearman r   (edge attention  vs  |S_ij| line loading)", fontsize=10)
ax.set_title(
    "AC-OPF: Does attention–loading correlation evolve over correction steps?\n"
    "Thin lines = individual heads · Thick = mean across 4 heads · 30 test scenarios",
    fontsize=9.5,
)
ax.legend(fontsize=9)
ax.set_xlim(1, max(K_sup, K_pinn))
ax.set_ylim(-0.62, 0.12)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "A_step_correlation_combined.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[A] step correlation saved")

# ═══════════════════════════════════════════════════════════════════════════
# Figure B: Head specialisation — both models side by side
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "AC-OPF Attention Head Specialisation\n"
    "Does any head specifically track line loading (|S_ij|)?",
    fontsize=10, fontweight="bold"
)

for ax, hs, label, col in [
    (axes[0], sup_hs, "Supervised (K=15)", "#00CFA8"),
    (axes[1], pnn_hs, "PINN (K=30)",       "#FF6B35"),
]:
    H = len(hs["r_head_all"])
    x = np.arange(H)
    w = 0.26
    r_all   = [r for r, p in hs["r_head_all"]]
    r_early = hs["r_head_early"]
    r_late  = hs["r_head_late"]

    b1 = ax.bar(x-w,  r_early, w, label=f"Early steps (1–{hs['K_early']})",  color="#4DA6FF", alpha=0.85)
    b2 = ax.bar(x,    r_all,   w, label="All steps (mean)",                   color=col,       alpha=0.85)
    b3 = ax.bar(x+w,  r_late,  w, label=f"Late steps",                        color="#9B59B6", alpha=0.85)

    # significance stars
    for h, (r, p) in enumerate(hs["r_head_all"]):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else "ns")
        ypos = r_all[h] - 0.025
        ax.text(h, ypos, sig, ha="center", va="top", fontsize=10, color="white", fontweight="bold")

    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Head {h+1}" for h in range(H)], fontsize=10)
    ax.set_ylabel("Spearman r  (attention vs |S_ij|)", fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.72, 0.18)
    ax.grid(axis="y", alpha=0.2)

    # spread annotation
    vals = r_all
    spread = max(vals) - min(vals)
    ax.annotate(f"Spread = {spread:.3f}\n({'strong' if spread>0.15 else 'weak'} specialisation)",
                xy=(0.02, 0.04), xycoords="axes fraction",
                fontsize=8, color=col, style="italic")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "B_head_specialization_combined.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[B] head specialisation saved")

# ═══════════════════════════════════════════════════════════════════════════
# Figure C: Source vs Destination bus type breakdown
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(
    "AC-OPF: Which bus type SENDS vs RECEIVES the most attention?\n"
    "Source = information hub | Destination = information consumer",
    fontsize=10, fontweight="bold"
)

types = ["slack", "gen", "load"]
sup_src_means  = src["supervised"]["mean_source_by_type"]
pinn_src_means = src["pinn"]["mean_source_by_type"]

# Left: source attention by type
ax = axes[0]
x = np.arange(len(types)); w = 0.35
b1 = ax.bar(x-w/2, [sup_src_means[t]  for t in types], w,
            label="Supervised", color="#00CFA8", alpha=0.85, edgecolor="white")
b2 = ax.bar(x+w/2, [pinn_src_means[t] for t in types], w,
            label="PINN",       color="#FF6B35", alpha=0.85, edgecolor="white")
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.0004, f"{h:.4f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([TYPE_LABEL[t] for t in types], fontsize=9)
ax.set_ylabel("Normalised source attention", fontsize=10)
ax.set_title("Source attention by bus type", fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.2)

# annotate the reversal
ax.annotate(
    "Supervised: generators > loads\n(dispatch variables drive info flow)",
    xy=(1, sup_src_means["gen"]+0.0004), xytext=(1.3, sup_src_means["gen"]+0.004),
    fontsize=7.5, color="#00CFA8",
    arrowprops=dict(arrowstyle="->", color="#00CFA8", lw=1.0)
)
ax.annotate(
    "PINN: loads > generators\n(physics balance hardest at load buses)",
    xy=(2, pinn_src_means["load"]+0.0004), xytext=(1.5, pinn_src_means["load"]+0.005),
    fontsize=7.5, color="#FF6B35",
    arrowprops=dict(arrowstyle="->", color="#FF6B35", lw=1.0)
)

# Right: top-3 source vs destination table (text)
ax2 = axes[1]
ax2.axis("off")
table_data = [
    ["", "Top-3 SOURCE buses", "Top-3 DEST buses"],
    ["Supervised", "8 (gen), 11 (load), 13 (load)", "8 (gen), 10 (load), 1 (slack)"],
    ["PINN",       "13 (load), 11 (load), 8 (gen)", "8 (gen), 12 (load), 3 (gen)"],
]
tbl = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                loc="center", cellLoc="left")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.05, 2.2)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#243B56"); cell.set_text_props(color="white", fontweight="bold")
    elif c == 0:
        cell.set_facecolor("#1E3348" if r % 2 == 1 else "#162638")
        cell.set_text_props(color="#00CFA8" if r==1 else "#FF6B35", fontweight="bold")
    else:
        cell.set_facecolor("#1A2E45")
        cell.set_text_props(color="#E4EDF8")
    cell.set_edgecolor("#2D4A6A")

ax2.set_title(
    "Bus 8 (generator) appears as top source in BOTH models\n"
    "— the OPF reference generator drives information flow",
    fontsize=8.5, pad=16
)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "C_source_dest_breakdown.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[C] source/dest breakdown saved")
print(f"\nAll combined figures -> {OUT}/")

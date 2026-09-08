"""Generate combined head-physics heatmap figure for both models."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SUP_DIR  = "results/head_physics_supervised"
PINN_DIR = "results/head_physics_pinn"
OUT      = "results/figures/deep_analysis"
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(SUP_DIR,  "head_physics_corr.json")) as f: sup  = json.load(f)
with open(os.path.join(PINN_DIR, "head_physics_corr.json")) as f: pinn = json.load(f)

QTYS   = ["|P_ij|", "|Q_ij|", "|Dθ_ij|", "|ΔV_ij|"]
H      = 4
LABELS = ["Active Power\n|P_ij| (MW)", "Reactive Power\n|Q_ij| (MVAr)",
          "Angle Diff\n|Δθ_ij| (rad)", "Volt. Mag. Diff\n|ΔV_ij| (pu)"]

def build_mat(corr):
    r_mat = np.full((len(QTYS), H), np.nan)
    p_mat = np.full((len(QTYS), H), 1.0)
    for i, qty in enumerate(QTYS):
        for h in range(H):
            if qty in corr:
                r_mat[i, h] = corr[qty][h][0]
                p_mat[i, h] = corr[qty][h][1]
    return r_mat, p_mat

sup_r,  sup_p  = build_mat(sup)
pinn_r, pinn_p = build_mat(pinn)

vmax = max(
    np.nanmax(np.abs(sup_r)),
    np.nanmax(np.abs(pinn_r)),
    0.05
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "AC-OPF Attention Head Specialisation: What Does Each Head Track?\n"
    "Spearman r (edge attention weight ↔ physical quantity) · All K steps × 30 scenarios · *** p<0.001 · ns = not significant",
    fontsize=10, fontweight="bold", y=1.02,
)

cmap = "RdYlGn_r"

for ax, r_mat, p_mat, label, col in [
    (axes[0], sup_r,  sup_p,  "Supervised (K=15, d=4, H=4)",  "#00CFA8"),
    (axes[1], pinn_r, pinn_p, "PINN (K=30, d=10, H=4)",       "#FF6B35"),
]:
    im = ax.imshow(r_mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    # cell annotations
    for i in range(len(QTYS)):
        for j in range(H):
            r  = r_mat[i, j]
            pv = p_mat[i, j]
            if np.isnan(r):
                txt = "nan"; c = "grey"
            else:
                sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
                txt = f"{r:+.3f}\n{sig}"
                c = "white" if abs(r) > vmax * 0.45 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=c,
                    fontweight="bold" if pv < 0.001 else "normal")

    # outline the max-|r| cell per row
    for i in range(len(QTYS)):
        row = r_mat[i]
        if not np.all(np.isnan(row)):
            j_max = int(np.nanargmax(np.abs(row)))
            ax.add_patch(plt.Rectangle(
                (j_max - 0.48, i - 0.48), 0.96, 0.96,
                fill=False, edgecolor="gold", lw=2.5, zorder=5
            ))

    ax.set_xticks(range(H))
    ax.set_xticklabels([f"Head {h+1}" for h in range(H)], fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(QTYS)))
    ax.set_yticklabels(LABELS, fontsize=9)
    ax.set_title(label, fontsize=11, color=col, fontweight="bold", pad=8)
    plt.colorbar(im, ax=ax, label="Spearman r", shrink=0.85, aspect=14)

# dominant-quantity annotation for PINN
dominant = {
    "PINN": ["|Dθ_ij|", "|ΔV_ij|", "|P_ij|", "|ΔV_ij|"],
    "Supervised": ["|Q_ij|", "|Q_ij|", "|P_ij|", "|Q_ij|"],
}
for ax, label, dq in [
    (axes[0], "Supervised", dominant["Supervised"]),
    (axes[1], "PINN",       dominant["PINN"]),
]:
    note = "Dominant:  " + " | ".join(f"H{h+1}→{dq[h]}" for h in range(H))
    ax.set_xlabel(note, fontsize=8, style="italic", color="#555", labelpad=8)

plt.tight_layout()
path = os.path.join(OUT, "D_head_physics_heatmap.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[D] head-physics heatmap saved -> {path}")


# ── companion bar chart ───────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle(
    "Per-Head Physical Specialisation — AC-OPF GNN with Edge Self-Attention\n"
    "Each bar group = one head; bars = physical quantities",
    fontsize=10, fontweight="bold"
)

qty_colors = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"]
qty_short  = ["|P|", "|Q|", "|Δθ|", "|ΔV|"]
x = np.arange(H)
w = 0.2

for ax, r_mat, p_mat, label in [
    (axes2[0], sup_r,  sup_p,  "Supervised"),
    (axes2[1], pinn_r, pinn_p, "PINN"),
]:
    for qi, (qty, col, ql) in enumerate(zip(QTYS, qty_colors, qty_short)):
        rs = r_mat[qi]
        bars = ax.bar(x + (qi - 1.5)*w, rs, w, label=ql, color=col,
                      alpha=0.85, edgecolor="white", linewidth=0.4)
        for bar, r, p in zip(bars, rs, p_mat[qi]):
            if np.isnan(r): continue
            sig = "***" if p < 0.001 else ("*" if p < 0.05 else "")
            if sig:
                yy = r + (0.012 if r >= 0 else -0.03)
                ax.text(bar.get_x()+bar.get_width()/2, yy, sig,
                        ha="center", fontsize=8, color=col, fontweight="bold")

    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Head {h+1}" for h in range(H)], fontsize=11)
    ax.set_ylabel("Spearman r  (attention ↔ physical quantity)", fontsize=10)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", title="Quantity", title_fontsize=8)
    ax.set_ylim(-0.72, 0.42)
    ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
path2 = os.path.join(OUT, "D_head_physics_bar.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"[D] head-physics bar saved -> {path2}")
print("\nKey findings:")
print("PINN specialisation: H1→angle, H2→volt-diff(strong!), H3→active-power, H4→volt-diff(+positive)")
print("Supervised: H1,H2,H4→reactive-power | H3→active-power  (less differentiated)")

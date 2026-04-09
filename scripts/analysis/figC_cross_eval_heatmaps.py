"""
4×4 cross-evaluation heatmaps (train strategy × eval strategy) for 9 metrics.
Color normalization is shared within each metric across all cells.

Layout: 3×3 grid of 4×4 heatmaps
Metrics: NLL/tok, Topo.Unc.(CV), ECE overall,
         ECE node index, ECE node label, ECE edge label,
         ECE special, ECE new node, ECE revisit
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path("experiments/invariance_eval/qm9")
OUT_DIR  = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["random_order", "min_degree_first", "max_degree_first", "anchor_expansion"]
LABELS     = ["Random", "Min-deg", "Max-deg", "Anchor"]
SEEDS      = [0, 1, 2]

METRICS = [
    # Row 1: overall metrics
    ("nll_per_tok",      "NLL/tok",                        "YlOrRd"),
    ("cv",               "Topo. Unc. (CV)",                "YlOrRd"),
    ("ECE_overall",      "ECE Overall",                    "YlOrRd"),
    # Row 2: Node Index and its two sub-types (same line = same row)
    ("ECE_node_index",   "ECE Node Index\n(New Node + Revisit)", "YlOrRd"),
    ("ECE_new_node",     "ECE New Node\n(sub-type of Node Index)", "YlOrRd"),
    ("ECE_revisit",      "ECE Revisit\n(sub-type of Node Index)", "YlOrRd"),
    # Row 3: remaining token types
    ("ECE_node_label",   "ECE Node Label",                 "YlOrRd"),
    ("ECE_edge_label",   "ECE Edge Label",                 "YlOrRd"),
    ("ECE_special",      "ECE Special",                    "YlOrRd"),
]

ECE_COLS = ["ECE_overall", "ECE_node_index", "ECE_new_node",
            "ECE_revisit", "ECE_node_label", "ECE_edge_label", "ECE_special"]

# ── Aggregate data ────────────────────────────────────────────────────────────
# means[metric_key][train_idx, eval_idx] = mean across seeds
# stds [metric_key][train_idx, eval_idx] = std across seeds

seed_vals = {m[0]: np.zeros((len(STRATEGIES), len(STRATEGIES), len(SEEDS)))
             for m in METRICS}

for si, seed in enumerate(SEEDS):
    for ti, train in enumerate(STRATEGIES):
        for ei, evl in enumerate(STRATEGIES):
            path = BASE_DIR / train / f"seed_{seed}" / "k32" / f"population_{evl}.csv"
            if not path.exists():
                print(f"MISSING: {path}")
                seed_vals["nll_per_tok"][ti, ei, si] = np.nan
                continue
            df = pd.read_csv(path)
            df["nll_per_tok"] = df["Mean_NLL"] / df["Mean_Seq_Len"]
            df = df.rename(columns={"CV": "cv"})
            for col, _, _ in METRICS:
                seed_vals[col][ti, ei, si] = df[col].mean()

means = {col: seed_vals[col].mean(axis=2) for col, _, _ in METRICS}
stds  = {col: seed_vals[col].std(axis=2)  for col, _, _ in METRICS}

# Shared vmax across all ECE panels (same colorbar normalization)
ece_vmax = max(np.nanmax(means[col]) for col in ECE_COLS)

# ── Print summary ─────────────────────────────────────────────────────────────
print("=== NLL/tok matrix (mean across seeds) ===")
df_nll = pd.DataFrame(means["nll_per_tok"], index=LABELS, columns=LABELS)
df_nll.index.name = "Train \\ Eval"
print(df_nll.to_string(float_format="{:.3f}".format))

print("\n=== ECE overall matrix ===")
df_ece = pd.DataFrame(means["ECE_overall"], index=LABELS, columns=LABELS)
print(df_ece.to_string(float_format="{:.4f}".format))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(13, 12))

for ax, (col, title, cmap) in zip(axes.flat, METRICS, strict=True):
    mat = means[col]
    std = stds[col]

    vmin = 0.0
    vmax = ece_vmax if col in ECE_COLS else np.nanmax(mat)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)

    # Annotate cells: mean ± std
    for ti in range(4):
        for ei in range(4):
            v = mat[ti, ei]
            s = std[ti, ei]
            if np.isnan(v):
                txt = "n/a"
            elif vmax > 0.5:
                txt = f"{v:.3f}\n±{s:.3f}"
            else:
                txt = f"{v:.4f}\n±{s:.4f}"
            brightness = v / (vmax + 1e-12)
            color = "white" if brightness > 0.6 else "black"
            ax.text(ei, ti, txt, ha="center", va="center",
                    fontsize=6.5, color=color, linespacing=1.3)

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS, fontsize=7.5, rotation=30, ha="right")
    ax.set_yticklabels(LABELS, fontsize=7.5)
    if ax in axes[:, 0]:
        ax.set_ylabel("Train", fontsize=8)
    if ax in axes[2, :]:
        ax.set_xlabel("Eval", fontsize=8)

fig.suptitle("Cross-Evaluation: Train Strategy × Eval Strategy  (QM9, K=32, 3-seed mean ± std)",
             fontsize=11, fontweight="bold")
plt.tight_layout()

out_pdf = OUT_DIR / "figC_cross_eval_heatmaps.pdf"
out_png = OUT_DIR / "figC_cross_eval_heatmaps.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_png}")
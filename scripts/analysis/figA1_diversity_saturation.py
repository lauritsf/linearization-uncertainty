"""Main-text Figure 2: compact 1×2 diversity saturation panel (Planar only).

Panel (a): Unique Ratio — deterministic strategies saturate at N=128.
Panel (b): 3-gram Coverage — stochastic linearization maintains higher coverage.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Style & palette (shared with plot_diversity_master.py)
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.constrained_layout.use": False,
    }
)

STRATEGY_PALETTE = {
    "random_order": "#2ca02c",
    "max_degree_first": "#d62728",
    "min_degree_first": "#ff7f0e",
    "anchor_expansion": "#1f77b4",
}
STRATEGY_LABELS = {
    "random_order": "Random",
    "max_degree_first": "Max Degree",
    "min_degree_first": "Min Degree",
    "anchor_expansion": "Anchor Exp.",
}
STRATEGY_ORDER = ["random_order", "max_degree_first", "anchor_expansion", "min_degree_first"]


def human_format(x, pos):
    if x >= 1_000_000:
        return f"{int(x / 1_000_000)}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k"
    return f"{int(x)}"


formatter = FuncFormatter(human_format)


def plot_saturation(ax, df, metric="Unique_Ratio"):
    for strat in STRATEGY_ORDER:
        sub = df[df["Strategy"] == strat]
        if sub.empty:
            continue
        agg = sub.groupby("Total_Samples")[metric].agg(["mean", "std", "count"])
        agg["se"] = agg["std"] / np.sqrt(agg["count"])
        x = agg.index.values
        y = agg["mean"].values
        ci = 1.96 * agg["se"].values
        ax.plot(x, y, label=STRATEGY_LABELS[strat], color=STRATEGY_PALETTE[strat], lw=1.8)
        ax.fill_between(x, y - ci, y + ci, color=STRATEGY_PALETTE[strat], alpha=0.15)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which="both", ls="--", alpha=0.5)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    planar = pd.read_csv("experiments/diversity/diversity_saturation_planar.csv")
except Exception as e:
    print(f"Error loading CSV: {e}")
    print("Ensure diversity_saturation_planar.csv exists in experiments/diversity/")
    exit(1)

# ---------------------------------------------------------------------------
# Figure: 1×2 panels
# ---------------------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.5, 2.2))

# Panel (a): Unique Ratio
plot_saturation(ax_a, planar, metric="Unique_Ratio")
ax_a.axhline(1.0, ls="-", color="0.3", lw=0.8, alpha=0.5)
ax_a.set_ylabel("Unique Ratio")
ax_a.set_xlabel("Total training samples")
ax_a.set_title("(a) Unique Ratio", fontsize=9, fontweight="bold")

# Panel (b): 3-gram Coverage
plot_saturation(ax_b, planar, metric="Ngram_Coverage")
ax_b.set_ylabel("3-gram Coverage")
ax_b.set_xlabel("Total training samples")
ax_b.set_title("(b) 3-gram Coverage", fontsize=9, fontweight="bold")

# Legend in panel (a)
ax_a.legend(loc="lower left", frameon=True, framealpha=1.0, edgecolor="0.5")

plt.subplots_adjust(bottom=0.2, left=0.1, top=0.85, right=0.97, wspace=0.3)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_dir = "results/figures"
os.makedirs(output_dir, exist_ok=True)

fig.savefig(f"{output_dir}/figA1_diversity_saturation.pdf", bbox_inches="tight")
fig.savefig(f"{output_dir}/figA1_diversity_saturation.png", dpi=300, bbox_inches="tight")
print(f"Saved {output_dir}/figA1_diversity_saturation.{{pdf,png}}")
